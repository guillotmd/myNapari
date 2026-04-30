import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import binary_opening, binary_closing, label as ndlabel

from .scale import ScaleCalibration
from .enface import process_enface_roi
from .edges import extract_edges
from .surfaces import smooth_curve, find_healthy_columns, filter_choroid_outliers
from .mesh import generate_tumor_mesh_from_2d_masks


# ──────────────────────────────────────────────────────────────────────────────
# Progress bar helper — fixed-width output to prevent \r overwrite artifacts
# ──────────────────────────────────────────────────────────────────────────────

_BAR_WIDTH = 30
_LINE_PAD = 80
_progress_callback = None  # set by pipeline from params


def _progress(phase: str, i: int, total: int, done: bool = False):
    """Print a fixed-width progress bar that overwrites in-place."""
    filled = _BAR_WIDTH if done else ((i + 1) * _BAR_WIDTH // total)
    bar = '█' * filled + ' ' * (_BAR_WIDTH - filled)
    check = ' ✓' if done else ''
    msg = f"\r  {phase} [{bar}] {i+1 if not done else total}/{total}{check}"
    print(msg.ljust(_LINE_PAD), end="" if not done else "\n", flush=True)
    # Notify UI callback if set
    if _progress_callback is not None:
        pct = 100 if done else int((i + 1) * 100 / total)
        short_phase = phase.split('—')[-1].strip().rstrip(':')
        _progress_callback(f"⏳ {short_phase}… {pct}%")


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

        interpolator = interp1d(
            valid_indices, valid_data,
            kind='linear',
            bounds_error=False,
            fill_value=(valid_data[0], valid_data[-1])
        )
        full_baselines[:, x] = interpolator(z_indices)

    return full_baselines


def _roi_aware_cleanup(
    output_mask_3d: np.ndarray,
    tumor_label: int,
    enface_data: np.ndarray | None,
    enface_tumor_label: int,
    min_component_size: int = 50,
):
    """
    ROI-aware morphological cleanup.

    Strategy:
      1. Binary opening  → remove isolated noise voxels
      2. Binary closing  → fill small holes
      3. Label connected components in 3-D
      4. For each component decide KEEP vs REMOVE:
         a. If an en-face ROI is provided: keep any component whose
            Z-X footprint overlaps a highlighted en-face tumor region.
            Remove only components with ZERO overlap that are also
            smaller than `min_component_size`.
         b. If no ROI: just remove tiny components < min_component_size.

    This ensures that each user-highlighted tumor on the en-face is
    treated as a distinct tumor region, and nearby B-scan layers that
    belong to the same en-face blob are kept together.
    """
    mask_bool = (output_mask_3d == tumor_label)
    mask_bool = binary_opening(mask_bool, iterations=1)
    mask_bool = binary_closing(mask_bool, iterations=1)

    labeled_arr, n_components = ndlabel(mask_bool)
    if n_components <= 1:
        # 0 or 1 component — nothing to filter
        output_mask_3d[output_mask_3d == tumor_label] = 0
        output_mask_3d[mask_bool] = tumor_label
        return output_mask_3d

    # Build en-face overlap mask if ROI is available
    has_roi = (enface_data is not None)
    if has_roi:
        enface_mask_2d = (enface_data == enface_tumor_label)  # (Z, X)
    else:
        enface_mask_2d = None

    for c in range(1, n_components + 1):
        comp_mask = (labeled_arr == c)
        comp_size = int(np.sum(comp_mask))

        if has_roi and enface_mask_2d is not None:
            # Project the 3D component down to Z-X (collapse the depth axis)
            # comp_mask is (Z, H, W) — project to (Z, W) footprint
            footprint_zx = comp_mask.any(axis=1)  # (Z, W)
            overlap = np.sum(footprint_zx & enface_mask_2d)
            if overlap > 0:
                # Component overlaps with a user-highlighted tumor → KEEP
                continue
            # No overlap with any en-face region — remove if tiny
            if comp_size < min_component_size:
                mask_bool[comp_mask] = False
        else:
            # No ROI — just use size threshold
            if comp_size < min_component_size:
                mask_bool[comp_mask] = False

    output_mask_3d[output_mask_3d == tumor_label] = 0
    output_mask_3d[mask_bool] = tumor_label
    return output_mask_3d


def execute_retinoblastoma_pipeline(params: dict):
    global _progress_callback
    _progress_callback = params.get('progress_callback', None)

    print("\n" + "="*60)
    print("  Generate Tumor Mask (Linear) Pipeline")
    print("="*60)

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
    N = n_slices

    # Read optional improvement toggles (all default OFF)
    use_choroid_filter = params.get('use_choroid_filter', False)
    use_morphological_cleanup = params.get('use_morphological_cleanup', False)
    interpolation_mode = params.get('interpolation_mode', 'linear')
    smoothing_sigma = params.get('smoothing_sigma', 20.0)
    use_weighted_fitting = params.get('use_weighted_fitting', False)

    retina_surfaces = {}
    choroid_surfaces_raw = {}
    choroid_surfaces = {}
    retinal_baselines = {}
    choroid_baselines = {}

    # ── Phase 1: Extract edges ────────────────────────────────────────────
    for i in range(N):
        _progress("Phase 1/4 — Extracting edges:", i, N)
        retina_edge, choroid_edge = extract_edges(
            seg_data[i],
            retina_label=params['retina_label'],
            choroid_label=params['choroid_label'],
            min_thickness=params.get('min_layer_thickness', 5),
            ignore_top_px=params.get('ignore_top_px', 0)
        )
        retina_surfaces[i] = retina_edge
        choroid_surfaces_raw[i] = choroid_edge
    _progress("Phase 1/4 — Extracting edges:", N, N, done=True)

    # ── Phase 2: Filter & fit baselines ───────────────────────────────────
    for i in range(N):
        _progress("Phase 2/4 — Fitting baselines:", i, N)

        retina_edge = retina_surfaces[i]
        choroid_edge = choroid_surfaces_raw[i]

        roi_mask = None
        if enface_data is not None and i in enface_res.roi_prior:
            roi_mask = enface_res.roi_prior[i] > 0.5

        healthy_cols = find_healthy_columns(
            retina_edge, choroid_edge, roi_mask,
            sigma=2.0, n_iters=2
        )

        if use_choroid_filter:
            choroid_edge = filter_choroid_outliers(
                choroid_edge, retina_edge, healthy_cols,
                max_above_baseline_px=smoothing_sigma,
                sigma=smoothing_sigma,
            )
            healthy_cols = find_healthy_columns(
                retina_edge, choroid_edge, roi_mask,
                sigma=2.0, n_iters=2
            )

        choroid_surfaces[i] = choroid_edge

        if healthy_cols.sum() >= 4:
            if use_weighted_fitting:
                temp_ret = np.full(w, np.nan)
                temp_chor = np.full(w, np.nan)
                cho_valid = ~np.isnan(choroid_edge)
                if cho_valid.any():
                    from scipy.ndimage import distance_transform_edt
                    dist = distance_transform_edt(cho_valid.astype(float))
                    max_dist = dist.max() if dist.max() > 0 else 1.0
                    weights = 0.3 + 0.7 * (dist / max_dist)
                    for col in np.where(healthy_cols)[0]:
                        temp_ret[col] = retina_edge[col]
                        temp_chor[col] = choroid_edge[col]
                else:
                    temp_ret[healthy_cols] = retina_edge[healthy_cols]
                    temp_chor[healthy_cols] = choroid_edge[healthy_cols]
            else:
                temp_ret = np.full(w, np.nan)
                temp_ret[healthy_cols] = retina_edge[healthy_cols]
                temp_chor = np.full(w, np.nan)
                temp_chor[healthy_cols] = choroid_edge[healthy_cols]

            retinal_baselines[i] = smooth_curve(
                temp_ret, sigma=smoothing_sigma,
                interpolation_mode=interpolation_mode,
            )
            choroid_baselines[i] = smooth_curve(
                temp_chor, sigma=smoothing_sigma,
                interpolation_mode=interpolation_mode,
            )
    _progress("Phase 2/4 — Fitting baselines:", N, N, done=True)

    # ── Phase 3: 3D Cross-Slice Interpolation ─────────────────────────────
    print("  Phase 3/4 — Interpolating across slices...")
    full_retina_surfaces = interpolate_baselines_3d(retina_surfaces, n_slices, w)
    full_retinal_baselines = interpolate_baselines_3d(retinal_baselines, n_slices, w)
    full_choroid_baselines = interpolate_baselines_3d(choroid_baselines, n_slices, w)

    # ── Phase 4: Detection and Mask Building ──────────────────────────────
    output_mask_3d = np.zeros((n_slices, h, w), dtype=np.uint8)
    elevation_threshold = params.get('elevation_threshold', 5.0)

    retina_line_label = 4
    baseline_line_label = 5
    tumor_label = params['output_tumor_label']

    total_voxels = 0

    show_lines = params.get('show_diagnostic_lines', False)
    edge_margin_cols = params.get('edge_margin_cols', 15)

    for i in range(N):
        _progress("Phase 4/4 — Building tumor masks:", i, N)

        r_surf = full_retina_surfaces[i]
        r_base = full_retinal_baselines[i]
        c_base = full_choroid_baselines[i]

        raw_r_surf = retina_surfaces[i]
        valid_cols = np.where(~np.isnan(raw_r_surf))[0]
        blank_left = 0
        blank_right = w
        if len(valid_cols) > 0 and edge_margin_cols > 0:
            first_valid = int(valid_cols[0])
            last_valid = int(valid_cols[-1])
            blank_left = min(first_valid + edge_margin_cols, w)
            blank_right = max(last_valid - edge_margin_cols + 1, 0)

        roi_allowed = np.ones(w, dtype=bool)
        if enface_data is not None and i in enface_res.roi_x_range:
            x_min, x_max = enface_res.roi_x_range[i]
            roi_allowed[:x_min] = False
            roi_allowed[x_max+1:] = False
        elif enface_data is not None and i not in enface_res.roi_x_range:
            roi_allowed[:] = False

        for x in range(w):
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

        if show_lines:
            for x in range(w):
                if not np.isnan(c_base[x]):
                    y_base = int(round(c_base[x]))
                    if 0 <= y_base < h:
                        output_mask_3d[i, max(0, y_base-1):min(h, y_base+2), x] = baseline_line_label
                if not np.isnan(r_surf[x]):
                    y_ret = int(round(r_surf[x]))
                    if 0 <= y_ret < h:
                        output_mask_3d[i, max(0, y_ret-1):min(h, y_ret+2), x] = retina_line_label

    _progress("Phase 4/4 — Building tumor masks:", N, N, done=True)

    # ── Optional morphological cleanup (ROI-aware) ────────────────────────
    if use_morphological_cleanup:
        print("  Cleaning up (ROI-aware)...")
        output_mask_3d = _roi_aware_cleanup(
            output_mask_3d,
            tumor_label,
            enface_data=enface_data,
            enface_tumor_label=params.get('enface_tumor_label', 1),
            min_component_size=50,
        )
        total_voxels = int(np.count_nonzero(output_mask_3d == tumor_label))

    # ── Volume calculation ────────────────────────────────────────────────
    ax_mm = scale_calib.axial_resolution / 1000.0
    lat_mm = scale_calib.lateral_resolution / 1000.0
    z_mm = scale_calib.inter_slice_spacing / 1000.0

    total_volume_mm3 = total_voxels * ax_mm * lat_mm * z_mm

    # Uncertainty estimation
    uncertainty_mm3 = 0.0
    if params.get('compute_uncertainty', False):
        print("  Estimating uncertainty...")
        voxel_vol = ax_mm * lat_mm * z_mm
        counts = []
        for delta in [-1.0, 0.0, 1.0]:
            perturbed_threshold = max(1.0, elevation_threshold + delta)
            count = 0
            for i in range(n_slices):
                r_surf = full_retina_surfaces[i]
                r_base = full_retinal_baselines[i]
                c_base = full_choroid_baselines[i]
                for x in range(w):
                    if np.isnan(r_surf[x]) or np.isnan(r_base[x]) or np.isnan(c_base[x]):
                        continue
                    elev = r_base[x] - r_surf[x]
                    if elev >= perturbed_threshold:
                        top = int(round(r_surf[x]))
                        bot = int(round(c_base[x]))
                        if bot > top:
                            count += min(h, bot+1) - max(0, top)
            counts.append(count)
        vol_range = [c * voxel_vol for c in counts]
        uncertainty_mm3 = (max(vol_range) - min(vol_range)) / 2.0

    print(f"\n  ✅ Pipeline Complete!")
    print(f"  Total Volume: {total_volume_mm3:.4f} mm³ ± {uncertainty_mm3:.4f}")
    print("="*60 + "\n")

    # ── Mesh Generation ───────────────────────────────────────────────────
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

    _progress_callback = None
    bscan_name = params.get('bscan_name', 'Retinoblastoma')
    return output_mask_3d, total_volume_mm3, uncertainty_mm3, mesh_data, tumor_label, bscan_name

def recalculate_volume(mask_3d: np.ndarray, tumor_label: int, params: dict) -> tuple[float, float]:
    """
    Recalculate volume from an existing edited mask.
    Returns (volume_mm3, uncertainty_mm3).
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

    volume = float(total_voxels * ax_mm * lat_mm * z_mm)
    return volume, 0.0
