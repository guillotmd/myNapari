import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from .scale import ScaleCalibration

@dataclass
class VolumeMetrics:
    total_volume_mm3: float
    volume_uncertainty_mm3: float
    volume_confirmed_mm3: float
    volume_interpolated_inter_mm3: float
    volume_interpolated_intra_mm3: float
    peak_elevation_um: float
    base_diameter_um: float
    n_slices_confirmed: int
    n_slices_interpolated: int
    n_slices_empty: int
    
def calculate_volume_and_uncertainty(
    areas_um2: Dict[int, float],
    slice_classifications: Dict[int, str],
    deviation_maps: Dict[int, np.ndarray],
    tumor_masks: Dict[int, np.ndarray],
    scale_calib: ScaleCalibration,
    n_slices_total: int,
    # Simplified uncertainties: just percentage or fixed estimates for now
    # as proper bootstrapping per slice requires storing full bootstrapped area variants.
    # In full impl, this would sum up variances.
) -> VolumeMetrics:
    spacing_mm = scale_calib.inter_slice_spacing / 1000.0
    
    vol_total = 0.0
    vol_conf = 0.0
    vol_inter_inter = 0.0
    vol_inter_intra = 0.0
    
    n_conf = 0
    n_inter = 0
    
    peak_elevation = 0.0
    max_width_px = 0
    
    # Very simple uncertainty estimate: 
    # Assume 5% error on area for each confirmed slice, 15% for interpolated
    var_total_mm6 = 0.0
    
    for i, area in areas_um2.items():
        area_mm2 = area / 1e6
        slice_vol = area_mm2 * spacing_mm
        
        vol_total += slice_vol
        
        c_class = slice_classifications.get(i, "No tumor")
        if c_class == "Confirmed":
            vol_conf += slice_vol
            n_conf += 1
            var_total_mm6 += (0.05 * slice_vol)**2
        elif c_class == "Interpolated_inter":
            vol_inter_inter += slice_vol
            n_inter += 1
            var_total_mm6 += (0.15 * slice_vol)**2
        elif c_class == "Interpolated_intra":
            vol_inter_intra += slice_vol
            n_conf += 1 # Or count as its own category
            var_total_mm6 += (0.10 * slice_vol)**2
            
        # Peak elevation
        d_map = deviation_maps.get(i)
        mask = tumor_masks.get(i)
        if d_map is not None and mask is not None and np.any(mask):
            peak = np.max(d_map[mask])
            if peak > peak_elevation:
                peak_elevation = peak
                
            # Base width
            indices = np.where(mask)[0]
            w_px = indices[-1] - indices[0] + 1
            if w_px > max_width_px:
                max_width_px = w_px
                
    base_diameter_um = max_width_px * scale_calib.lateral_resolution
    n_empty = n_slices_total - n_conf - n_inter
    
    uncertainty_mm3 = np.sqrt(var_total_mm6)
    
    return VolumeMetrics(
        total_volume_mm3=vol_total,
        volume_uncertainty_mm3=uncertainty_mm3,
        volume_confirmed_mm3=vol_conf,
        volume_interpolated_inter_mm3=vol_inter_inter,
        volume_interpolated_intra_mm3=vol_inter_intra,
        peak_elevation_um=peak_elevation,
        base_diameter_um=base_diameter_um,
        n_slices_confirmed=n_conf,
        n_slices_interpolated=n_inter,
        n_slices_empty=n_empty
    )
