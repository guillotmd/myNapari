import numpy as np
import pytest
from florian_retinoblastoma_vol.baseline import fit_baseline
from florian_retinoblastoma_vol.normals import fit_spline, compute_normal_vectors

def test_perpendicular_thickness():
    # Synthetic U-curve (parabola y = x^2)
    # Actually, let's use a simpler curve where we know exact normals
    # y = a * x^2, dy/dx = 2ax
    # Normal vector (-2ax, 1) or (-2ax, -1)
    
    x = np.linspace(0, 100, 100)
    y_choroid = np.full(100, 50.0)
    
    # Let's say thickness is exactly 5 units perpendicular
    # To construct retina, we offset by T=5 along inward normal
    from florian_retinoblastoma_vol.normals import compute_perpendicular_thickness
    
    spline = fit_spline(x, y_choroid, smoothing_factor=0) # exact fit
    nx, ny = compute_normal_vectors(spline, x)
    
    T_exact = 5.0
    y_retina = y_choroid - T_exact * ny
    x_retina = x + T_exact * nx
    
    # Compute thickness
    # interpolate retina to integer grid
    x_int = np.arange(len(x))
    # We'll just test the core normal computation
    t_perp = compute_perpendicular_thickness(x, spline, x_retina, y_retina, max_search_dist=10.0)
    
    valid_t_perp = t_perp[~np.isnan(t_perp)]
    assert len(valid_t_perp) > 0
    assert valid_t_perp.min() > 4.5
    assert valid_t_perp.max() < 5.5

def test_baseline_ignores_bump():
    x = np.arange(100)
    y_choroid = np.full(100, 50.0)
    
    # Gap in choroid
    y_choroid[40:60] = np.nan
    
    y_retina = np.full(100, 45.0)
    y_retina[40:60] -= 10.0 # Bump
    
    prior = np.zeros(100)
    prior[40:60] = 1.0
    
    res = fit_baseline(y_retina, y_choroid, prior, method="Choroid+perpendicular offset (recommended)")
    
    # Baseline should interpolate straight across the gap
    assert np.allclose(res.baseline_y[40:60], 45.0, atol=1e-1)
    
def test_gap_spanning():
    x = np.arange(100)
    y_choroid = np.linspace(50, 60, 100)
    y_choroid[30:70] = np.nan
    
    prior = np.zeros(100)
    prior[30:70] = 1.0
    
    y_retina = y_choroid - 5.0
    y_retina[30:70] = np.linspace(45, 55, 100)[30:70] - 10.0 # Tumor
    
    res = fit_baseline(y_retina, y_choroid, prior, method="Choroid+perpendicular offset (recommended)")
    
    # The choroid spline should interpolate straight across
    assert np.allclose(res.choroid_spline_y, np.linspace(50, 60, 100), atol=1e-1)
