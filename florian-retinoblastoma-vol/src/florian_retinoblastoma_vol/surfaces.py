import numpy as np
from scipy.interpolate import interp1d
from skimage.filters import gaussian

def smooth_curve(surface: np.ndarray, sigma: float = 20.0) -> np.ndarray:
    """
    Linearly interpolate NaNs, flat-extrapolate to edges, and apply Gaussian smoothing.
    This guarantees a continuous curve that never 'dives down' at the edges.
    """
    valid = ~np.isnan(surface)
    if not np.any(valid):
        return np.full_like(surface, np.nan)
        
    x = np.arange(len(surface))
    valid_x = x[valid]
    valid_y = surface[valid]
    
    if len(valid_x) == 1:
        return np.full_like(surface, valid_y[0])
        
    interpolator = interp1d(
        valid_x, valid_y, 
        kind='linear', 
        bounds_error=False, 
        fill_value=(valid_y[0], valid_y[-1])
    )
    
    filled = interpolator(x)
    if sigma > 0:
        return gaussian(filled, sigma=sigma)
    return filled

def find_healthy_columns(
    retina_surface: np.ndarray,
    choroid_surface: np.ndarray,
    roi_mask: np.ndarray | None = None,
    sigma: float = 2.0,
    n_iters: int = 2
) -> np.ndarray:
    """
    Identify columns where the retina is NOT elevated (tumor-free).
    If an enface ROI mask is provided, uses that.
    Otherwise, uses iterative robust rejection to find elevated columns.
    """
    W = len(retina_surface)
    if roi_mask is not None:
        healthy = ~np.isnan(choroid_surface) & ~roi_mask
        if healthy.sum() >= 4:
            return healthy
            
    # Fallback to robust iterative logic
    healthy = ~np.isnan(choroid_surface) & ~np.isnan(retina_surface)
    if healthy.sum() < 4:
        return healthy
        
    for _ in range(n_iters):
        temp_surface = np.full(W, np.nan)
        temp_surface[healthy] = retina_surface[healthy]
        
        baseline = smooth_curve(temp_surface, sigma=20.0)
        if np.isnan(baseline[0]):
            break
            
        elevation = baseline - retina_surface # Positive = retina is above (smaller y) baseline
        valid = ~np.isnan(retina_surface)
        elev_valid = elevation[valid & healthy]
        
        if len(elev_valid) < 2:
            break
            
        threshold = elev_valid.mean() + sigma * elev_valid.std()
        outliers = valid & (elevation > threshold)
        healthy = healthy & ~outliers
        
        if healthy.sum() < 4:
            break
            
    return healthy
