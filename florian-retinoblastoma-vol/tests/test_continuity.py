import numpy as np
import pytest
from florian_retinoblastoma_vol.continuity import fill_intra_slice_gaps, propagate_inter_slice

def test_intra_slice_fill():
    width = 100
    tumor_mask = np.zeros(width, dtype=bool)
    tumor_mask[10:20] = True
    tumor_mask[30:40] = True # Gap of 10
    
    conf = np.zeros(width)
    conf[10:20] = 0.8
    conf[30:40] = 0.8
    
    raw = np.zeros((100, 100))
    raw[:, 20:30] = 50.0 # Bright gap
    
    retina = np.full(width, 50.0)
    
    new_mask, new_conf, interp = fill_intra_slice_gaps(
        tumor_mask, conf, raw, retina,
        max_intra_gap=15,
        gap_intensity_threshold=40.0,
        min_tumor_width=5
    )
    
    assert np.all(new_mask[10:40])
    assert np.all(interp[20:30])

def test_inter_slice_propagate():
    masks = {
        0: np.ones(100, dtype=bool),
        1: np.zeros(100, dtype=bool),
        2: np.ones(100, dtype=bool)
    }
    areas = {0: 100.0, 1: 0.0, 2: 100.0}
    deviations = {0: np.ones(100), 1: np.ones(100), 2: np.ones(100)}
    dl_maps = {0: np.ones(100), 1: np.ones(100), 2: np.ones(100)}
    priors = {0: np.ones(100), 1: np.ones(100), 2: np.ones(100)}
    
    new_masks, new_areas, interps = propagate_inter_slice(
        masks, areas, deviations, priors, dl_maps, 3, interp_neighbor_window=1, prior_threshold=0.5
    )
    
    assert 1 in interps
    assert np.all(new_masks[1])
