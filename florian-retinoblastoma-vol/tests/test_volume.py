import numpy as np
import pytest
from florian_retinoblastoma_vol.volume import calculate_volume_and_uncertainty
from florian_retinoblastoma_vol.scale import ScaleCalibration

def test_flat_slab_volume():
    # 10 slices, each 100 um2 area, 10um spacing
    areas = {i: 100.0 for i in range(10)}
    classifications = {i: "Confirmed" for i in range(10)}
    
    deviations = {i: np.zeros(10) for i in range(10)}
    masks = {i: np.ones(10, dtype=bool) for i in range(10)}
    
    calib = ScaleCalibration(axial_resolution=1.0, lateral_resolution=1.0, inter_slice_spacing=10.0)
    
    metrics = calculate_volume_and_uncertainty(
        areas, classifications, deviations, masks, calib, 10
    )
    
    # Volume = area_mm2 * spacing_mm * n_slices
    # area_mm2 = 100 / 1e6 = 1e-4 mm2
    # spacing_mm = 10 / 1000 = 0.01 mm
    # Vol = 1e-4 * 0.01 * 10 = 1e-5 mm3
    
    assert np.isclose(metrics.total_volume_mm3, 1e-5)
    assert metrics.n_slices_confirmed == 10
    assert metrics.n_slices_interpolated == 0
