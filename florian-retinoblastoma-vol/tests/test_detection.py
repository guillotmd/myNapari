import numpy as np
import pytest
from florian_retinoblastoma_vol.detection import detect_tumor
from florian_retinoblastoma_vol.baseline import BaselineResult
from florian_retinoblastoma_vol.scale import ScaleCalibration

def test_flat_retina_tumor_area():
    width = 100
    x_coords = np.arange(width)
    
    baseline_y = np.full(width, 50.0)
    baseline_x = x_coords.astype(float)
    choroid_y = np.full(width, 55.0)
    
    # Flat retina with a 20px wide, 10px high bump
    retina_y = np.full(width, 50.0)
    retina_y[40:60] = 40.0
    
    res = BaselineResult(
        baseline_y=baseline_y,
        baseline_x=baseline_x,
        choroid_spline_y=choroid_y,
        t_perp=np.full(width, 5.0),
        shadow_zone=np.zeros(width, dtype=bool),
        prior_tumor_zone=np.zeros(width, dtype=bool),
        tumor_candidate_zone=np.zeros(width, dtype=bool),
        validation_warning=False,
        validation_divergence=0.0
    )
    
    calib = ScaleCalibration(axial_resolution=1.0, lateral_resolution=1.0, inter_slice_spacing=1.0)
    
    prior = np.zeros(width)
    prior[40:60] = 1.0
    
    raw_img = np.zeros((100, 100))
    
    # Detect
    tumor_res = detect_tumor(
        raw_img, retina_y, res,
        roi_x_range=(40, 59),
        roi_prior=prior,
        scale_calib=calib,
        elevation_threshold=5.0,
        fine_sigma=0.1,  # Keep sharp
        coarse_sigma=0.1,
        min_tumor_width=5
    )
    
    # Area = width * height = 20 * 10 = 200
    assert np.isclose(tumor_res.cross_sectional_area_um2, 200.0, rtol=0.05)

def test_multi_scale_smoothing():
    width = 100
    retina_y = np.full(width, 50.0)
    retina_y[50] = 30.0 # Spike
    
    from florian_retinoblastoma_vol.detection import smooth_edge_ignoring_nans
    
    fine = smooth_edge_ignoring_nans(retina_y, 0.5)
    coarse = smooth_edge_ignoring_nans(retina_y, 15.0)
    
    # Spike should be somewhat preserved in fine, flattened in coarse
    assert fine[50] < 40.0
    assert coarse[50] > 48.0
