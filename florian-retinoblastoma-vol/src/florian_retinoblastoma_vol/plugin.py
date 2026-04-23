import numpy as np
from scipy.interpolate import interp1d

from .scale import ScaleCalibration
from .enface import process_enface_roi
from .edges import extract_edges
from .surfaces import smooth_curve, find_healthy_columns
from .mesh import generate_tumor_mesh_from_2d_masks

def interpolate_baselines_3d(baselines: dict, n_slices: int, width: int) -> np.ndarray:
    """
    Interpolate completely missing baselines along the Z-axis (across slices).
    """
    full_baselines = np.full((n_slices, width), np.nan)
    for i, b in baselines.items():
        full_baselines[i] = b
        
    for x in range(width):
        z_vals = full_baselines[:, x]
        valid_z = ~np.isnan(z_vals)
        if not np.any(valid_z):
            continue
            
        z_indices = np.arange(n_slices)
        valid_indices = z_indices[valid_z]
        valid_data = z_vals[valid_z]
        
        if len(valid_indices) == 1:
            full_baselines[:, x] = valid_data[0]
            continue
            
        # Linearly interpolate missing Z, flat extrapolate
        interpolator = interp1d(
            valid_indices, valid_data,
            kind='linear',
            bounds_error=False,
            fill_value=(valid_data[0], valid_data[-1])
        )
        full_baselines[:, x] = interpolator(z_indices)
        
    return full_baselines

def execute_retinoblastoma_pipeline(params: dict):
    print("Starting Robust 3D Interpolation Pipeline...")
    
    scale_calib = ScaleCalibration(
        axial_resolution=params['axial_resolution'],
        lateral_resolution=params['lateral_resolution'],
        inter_slice_spacing=params['inter_slice_spacing']
    )
    
    bscan_data = params['bscan_data']
    n_slices, h, w = bscan_data.shape
    
    enface_data = params.get('enface_data')
    if enface_data is not None:
        if enface_data.ndim == 2:
            enface_mask = (enface_data == params['enface_tumor_label'])
        else:
            raise ValueError("Error: En-face image must be 2D")
    else:
        enface_mask = np.ones((n_slices, w), dtype=bool)
        
    enface_res = process_enface_roi(
        enface_mask=enface_mask,
        bscan_shape=bscan_data.shape,
        prior_sigma=params.get('prior_sigma', 10.0),
        mapping_mode=params.get('mapping_mode', 'linear')
    )
    
    seg_data = params['seg_data']
    
    retina_surfaces = {}
    choroid_surfaces = {}
    retinal_baselines = {}
    choroid_baselines = {}
    
    # 1. Extract 2D Surfaces and Fit Preliminary Baselines
    for i in range(n_slices):
        retina_edge, choroid_edge = extract_edges(
            seg_data[i],
            retina_label=params['retina_label'],
            choroid_label=params['choroid_label']
        )
        
        retina_surfaces[i] = retina_edge
        choroid_surfaces[i] = choroid_edge
        
        # Determine ROI mask for this slice
        roi_mask = None
        if enface_data is not None and i in enface_res.roi_prior:
            roi_mask = enface_res.roi_prior[i] > 0.5
            
        healthy_cols = find_healthy_columns(
            retina_edge, choroid_edge, roi_mask,
            sigma=2.0, n_iters=2
        )
        
        if healthy_cols.sum() >= 4:
            # Fit retinal baseline (Label 4 baseline for detection)
            temp_ret = np.full(w, np.nan)
            temp_ret[healthy_cols] = retina_edge[healthy_cols]
            retinal_baselines[i] = smooth_curve(temp_ret, sigma=20.0)
            
            # Fit choroid baseline (Label 5 boundary for volume fill)
            temp_chor = np.full(w, np.nan)
            temp_chor[healthy_cols] = choroid_edge[healthy_cols]
            choroid_baselines[i] = smooth_curve(temp_chor, sigma=20.0)
            
    # 2. 3D Cross-Slice Interpolation for missing surfaces/baselines
    full_retina_surfaces = interpolate_baselines_3d(retina_surfaces, n_slices, w)
    full_retinal_baselines = interpolate_baselines_3d(retinal_baselines, n_slices, w)
    full_choroid_baselines = interpolate_baselines_3d(choroid_baselines, n_slices, w)
    
    # 3. Detection and Mask Building
    output_mask_3d = np.zeros((n_slices, h, w), dtype=np.uint8)
    elevation_threshold = params.get('elevation_threshold', 5.0) # px
    
    retina_line_label = 4
    baseline_line_label = 5
    tumor_label = params['output_tumor_label']
    
    total_voxels = 0
    
    show_lines = params.get('show_diagnostic_lines', False)
    edge_margin_cols = params.get('edge_margin_cols', 15)
    
    for i in range(n_slices):
        r_surf = full_retina_surfaces[i]
        r_base = full_retinal_baselines[i]
        c_base = full_choroid_baselines[i]
        
        # Calculate valid columns for edge blanking from the RAW (non-Z-interpolated) surface!
        raw_r_surf = retina_surfaces[i]
        valid_cols = np.where(~np.isnan(raw_r_surf))[0]
        blank_left = 0
        blank_right = w
        if len(valid_cols) > 0 and edge_margin_cols > 0:
            first_valid = int(valid_cols[0])
            last_valid = int(valid_cols[-1])
            blank_left = min(first_valid + edge_margin_cols, w)
            blank_right = max(last_valid - edge_margin_cols + 1, 0)
            
        # Get strict en-face bounds for this slice if ROI was provided
        roi_allowed = np.ones(w, dtype=bool)
        if enface_data is not None and i in enface_res.roi_x_range:
            x_min, x_max = enface_res.roi_x_range[i]
            # Disallow tumor outside the en-face bounds
            roi_allowed[:x_min] = False
            roi_allowed[x_max+1:] = False
        elif enface_data is not None and i not in enface_res.roi_x_range:
            # ROI is provided, but this slice has no ROI drawn
            roi_allowed[:] = False
        
        # Fill Tumor Volume (Label 6)
        for x in range(w):
            # Apply edge margin blanking and strict ROI masking
            if edge_margin_cols > 0 and (x < blank_left or x >= blank_right):
                continue
            if not roi_allowed[x]:
                continue
                
            if np.isnan(r_surf[x]) or np.isnan(r_base[x]) or np.isnan(c_base[x]):
                continue
                
            elevation = r_base[x] - r_surf[x]
            if elevation >= elevation_threshold:
                top = int(round(r_surf[x]))
                bot = int(round(c_base[x]))
                if bot > top:
                    output_mask_3d[i, max(0, top):min(h, bot+1), x] = tumor_label
                    total_voxels += (min(h, bot+1) - max(0, top))
                    
        # Draw Diagnostic Curves (Label 4 & 5) over the volume
        if show_lines:
            for x in range(w):
                # Baseline curve / Choroid surface (Label 5)
                if not np.isnan(c_base[x]):
                    y_base = int(round(c_base[x]))
                    if 0 <= y_base < h:
                        output_mask_3d[i, max(0, y_base-1):min(h, y_base+2), x] = baseline_line_label
                        
                # Retina curve (Label 4)
                if not np.isnan(r_surf[x]):
                    y_ret = int(round(r_surf[x]))
                    if 0 <= y_ret < h:
                        output_mask_3d[i, max(0, y_ret-1):min(h, y_ret+2), x] = retina_line_label

    # 4. Volume Calculation (Simple voxel count * physical scale)
    # Convert scale to mm
    ax_mm = scale_calib.axial_resolution / 1000.0
    lat_mm = scale_calib.lateral_resolution / 1000.0
    z_mm = scale_calib.inter_slice_spacing / 1000.0
    
    total_volume_mm3 = total_voxels * ax_mm * lat_mm * z_mm
    
    print("Pipeline Complete!")
    print(f"Total Volume: {total_volume_mm3:.4f} mm³")
    
    # 5. Mesh Generation
    mesh_data = None
    if params.get('generate_3d_render', True):
        bool_vol = (output_mask_3d == tumor_label)
        verts, faces = generate_tumor_mesh_from_2d_masks(
            bool_vol, scale_calib, params.get('mesh_smoothing_iters', 10)
        )
        if len(verts) > 0:
            vals = np.ones(len(verts))
            mesh_data = (verts, faces, vals)
        else:
            print("Could not generate 3D mesh (insufficient volume).")
            
    return output_mask_3d, total_volume_mm3, 0.0, mesh_data, tumor_label

def recalculate_volume(mask_3d: np.ndarray, tumor_label: int, params: dict) -> float:
    """
    Recalculate volume from an existing edited mask.
    """
    scale_calib = ScaleCalibration(
        axial_resolution=params['axial_resolution'],
        lateral_resolution=params['lateral_resolution'],
        inter_slice_spacing=params['inter_slice_spacing']
    )
    
    total_voxels = np.count_nonzero(mask_3d == tumor_label)
    
    ax_mm = scale_calib.axial_resolution / 1000.0
    lat_mm = scale_calib.lateral_resolution / 1000.0
    z_mm = scale_calib.inter_slice_spacing / 1000.0
    
    return float(total_voxels * ax_mm * lat_mm * z_mm)
