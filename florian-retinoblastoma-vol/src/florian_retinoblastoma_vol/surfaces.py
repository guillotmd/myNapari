import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
from skimage.filters import gaussian


def smooth_curve(
    surface: np.ndarray,
    sigma: float = 20.0,
    interpolation_mode: str = 'linear',
) -> np.ndarray:
    """
    Interpolate NaNs, extrapolate to edges, and apply Gaussian smoothing.

    Parameters
    ----------
    surface : (W,) float — raw surface with NaN gaps
    sigma : Gaussian smoothing sigma (0 = no smoothing)
    interpolation_mode : 'linear' or 'pchip'
        'linear' — straight-line interpolation (original behavior)
        'pchip'  — monotone cubic Hermite (gentle curve, no overshoot)
    """
    valid = ~np.isnan(surface)
    if not np.any(valid):
        return np.full_like(surface, np.nan)

    x = np.arange(len(surface))
    valid_x = x[valid]
    valid_y = surface[valid]

    if len(valid_x) == 1:
        return np.full_like(surface, valid_y[0])

    if interpolation_mode == 'pchip' and len(valid_x) >= 2:
        # PchipInterpolator: monotone-preserving cubic Hermite spline.
        # Produces gentle curves without the oscillation of full cubic
        # splines, while avoiding the straight-line rigidity of linear.
        interpolator = PchipInterpolator(valid_x, valid_y, extrapolate=True)
    else:
        interpolator = interp1d(
            valid_x, valid_y,
            kind='linear',
            bounds_error=False,
            fill_value=(valid_y[0], valid_y[-1]),
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


def filter_choroid_outliers(
    choroid_surface: np.ndarray,
    retina_surface: np.ndarray,
    healthy_cols: np.ndarray,
    max_above_baseline_px: float = 20.0,
    sigma: float = 20.0,
) -> np.ndarray:
    """
    Reject choroid detections that are anatomically impossible.

    The ONNX segmentation sometimes mislabels tumor tissue as 'choroid'.
    These appear at unreasonably shallow depths. Two rules:

    Rule 1: Choroid must be BELOW the retina (larger row index).
    Rule 2: Choroid must not be significantly above a preliminary
            baseline fitted from healthy columns.

    Parameters
    ----------
    choroid_surface : (W,) raw choroid edge, NaN where absent
    retina_surface : (W,) raw retina edge, NaN where absent
    healthy_cols : (W,) bool — columns considered healthy
    max_above_baseline_px : max px above baseline before rejection
    sigma : smoothing sigma for the preliminary baseline fit

    Returns
    -------
    cleaned : (W,) choroid surface with outliers set to NaN
    """
    cleaned = choroid_surface.copy()

    # Rule 1: Choroid above retina = impossible anatomy
    both_valid = ~np.isnan(cleaned) & ~np.isnan(retina_surface)
    above_retina = both_valid & (cleaned < retina_surface)
    cleaned[above_retina] = np.nan

    # Rule 2: Fit preliminary baseline, reject outliers above it
    prelim_healthy = healthy_cols & ~np.isnan(cleaned)
    if prelim_healthy.sum() >= 4:
        temp = np.full(len(cleaned), np.nan)
        temp[prelim_healthy] = cleaned[prelim_healthy]
        prelim_baseline = smooth_curve(temp, sigma=sigma)

        if not np.all(np.isnan(prelim_baseline)):
            valid_cho = ~np.isnan(cleaned)
            deviation = prelim_baseline - cleaned  # positive = choroid above baseline
            too_shallow = valid_cho & (deviation > max_above_baseline_px)
            cleaned[too_shallow] = np.nan

    return cleaned
