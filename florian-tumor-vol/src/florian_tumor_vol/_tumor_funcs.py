"""
_tumor_funcs.py
================
Training-free, signal-processing-based functions for detecting and measuring
intraocular tumors (e.g. choroidal melanoma) in OCT B-scan volumes.

Geometry / clinical context
---------------------------
In a standard OCT B-scan (row 0 = anterior / vitreous):

   Row 0   ──── vitreous ──────────────────────────────────
           ╭─────────────╮  ← retinal dome (ELEVATED by tumor)
   retina  │  label 1    │  ← normal retina sits flat here in healthy areas
           ╰──── label 2 ╯  ← choroid (eye wall)
   deeper  ─────────────────────────────────────────────────

The tumor pushes the retina UPWARD.  In healthy regions, the retinal surface
sits at a roughly predictable depth; over the tumor it is elevated by tens-
to-hundreds of pixels.  The choroid (label 2) is absent under the tumor.

Algorithm (per B-scan)
-----------------------
Two splines are fit from the SAME set of *healthy* columns (columns where the
en-face ROI says "no tumor" OR where the choroid is present):

  ① Retinal expected surface spline  — from label-1 anterior surface
     → "where the retina would be without the tumor"
     → used for DETECTION: column is tumor if
         retinal_baseline_row - actual_retina_row  >=  min_elevation_px

  ② Choroid eye-wall baseline spline — from label-2 anterior surface
     → "where the choroid / bottom of the retina would be"
     → used for EXTENT: the tumor mask fills from the elevated retinal
       surface (top) DOWN to the choroid baseline (bottom)

This gives:
  • Correct DETECTION  — healthy retina is NOT labeled (it matches spline ①)
  • Correct DEPTH      — tumor fills all the way to the eye wall (spline ②)
  • Correct CONSTRAINT — the choroid depth clamp prevents wild extrapolation

Label convention (confirmed from ONNX model source):
    0 = vitreous / background
    1 = retina   (dome / anterior surface for detection)
    2 = choroid  (eye-wall baseline for depth extent)
    3 = tumor mask output (default)
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import UnivariateSpline


# ──────────────────────────────────────────────────────────────────────────────
# Surface extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_anterior_surface(
    bscan_labels: np.ndarray,
    label_val: int,
) -> np.ndarray:
    """
    Per-column TOPMOST (smallest row) pixel belonging to ``label_val``.

    Returns
    -------
    surface_rows : shape (W,), float   —   NaN where label absent.
    """
    mask = bscan_labels == label_val
    W = mask.shape[1]
    has_label = mask.any(axis=0)
    surface_rows = np.full(W, np.nan, dtype=float)
    surface_rows[has_label] = mask.argmax(axis=0)[has_label].astype(float)
    return surface_rows


def extract_posterior_surface(
    bscan_labels: np.ndarray,
    label_val: int,
) -> np.ndarray:
    """
    Per-column DEEPEST (largest row) pixel belonging to ``label_val``.

    Returns
    -------
    surface_rows : shape (W,), float   —   NaN where label absent.
    """
    mask = bscan_labels == label_val
    H, W = mask.shape
    has_label = mask.any(axis=0)
    surface_rows = np.full(W, np.nan, dtype=float)
    surface_rows[has_label] = (H - 1 - mask[::-1].argmax(axis=0))[has_label].astype(float)
    return surface_rows


# ──────────────────────────────────────────────────────────────────────────────
# Spline fitting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _fit_spline(
    anchor_rows: np.ndarray,
    col_include: np.ndarray,      # bool (W,)
    smoothing: float | None,
    min_valid_frac: float = 0.05,
) -> np.ndarray | None:
    """Fit a UnivariateSpline through selected columns; evaluate on all columns."""
    W = len(anchor_rows)
    valid = col_include & ~np.isnan(anchor_rows)
    n_valid = valid.sum()

    if n_valid < 4 or n_valid < W * min_valid_frac:
        return None

    cols = np.where(valid)[0].astype(float)
    rows = anchor_rows[valid]
    k = min(3, n_valid - 1)

    spline = UnivariateSpline(cols, rows, s=smoothing, k=k)
    result = spline(np.arange(W, dtype=float))
    return np.clip(result, 0, None)


def clamp_baseline(
    baseline: np.ndarray,
    anchor_rows: np.ndarray,      # the surface used as anchor (may have NaNs)
    margin_above_px: float = 20.0,
    margin_below_px: float = 60.0,
    image_height: int | None = None,
) -> np.ndarray:
    """
    Clamp a spline baseline to the observed range of anchor_rows ± margins.
    Prevents wild extrapolation through or beyond the eye wall.
    """
    valid = ~np.isnan(anchor_rows)
    if not valid.any():
        return baseline

    min_obs = float(anchor_rows[valid].min())
    max_obs = float(anchor_rows[valid].max())

    lo = max(0.0, min_obs - margin_above_px)
    hi = max_obs + margin_below_px
    if image_height is not None:
        hi = min(hi, float(image_height - 1))

    return np.clip(baseline, lo, hi)


def fit_spline_with_clamp(
    anchor_rows: np.ndarray,
    healthy_col_mask: np.ndarray,   # bool (W,) — columns used for fitting
    smoothing: float | None = None,
    margin_above_px: float = 20.0,
    margin_below_px: float = 60.0,
    image_height: int | None = None,
    min_valid_frac: float = 0.05,
) -> np.ndarray | None:
    """
    Fit a bounded spline through the anchor surface using only healthy columns,
    then clamp the extrapolated values to stay within the observed range.
    """
    baseline = _fit_spline(anchor_rows, healthy_col_mask, smoothing, min_valid_frac)
    if baseline is None:
        return None
    return clamp_baseline(baseline, anchor_rows,
                          margin_above_px=margin_above_px,
                          margin_below_px=margin_below_px,
                          image_height=image_height)


# ──────────────────────────────────────────────────────────────────────────────
# Robust healthy-column identification (no-ROI fallback)
# ──────────────────────────────────────────────────────────────────────────────

def _find_healthy_columns_robust(
    retina_surface: np.ndarray,     # shape (W,)
    choroid_surface: np.ndarray,    # shape (W,)
    sigma: float = 2.0,
    n_iters: int = 2,
) -> np.ndarray:
    """
    Without a user ROI, identify "healthy" columns automatically by:
    1. Starting with columns where the choroid IS present.
    2. Fitting an initial retinal spline through those columns.
    3. Iteratively rejecting columns where the retina is significantly
       elevated above the current spline estimate.

    Returns a boolean mask of shape (W,).
    """
    # Start: columns where choroid is present (absent under the tumor)
    healthy = ~np.isnan(choroid_surface) & ~np.isnan(retina_surface)

    if healthy.sum() < 4:
        return healthy   # not enough — caller will skip

    for _ in range(n_iters):
        baseline = _fit_spline(retina_surface, healthy, smoothing=None)
        if baseline is None:
            break

        elevation = baseline - retina_surface    # positive = retina above baseline
        valid = ~np.isnan(retina_surface)
        elev_valid = elevation[valid & healthy]

        if len(elev_valid) < 2:
            break

        threshold = elev_valid.mean() + sigma * elev_valid.std()
        # Reject columns where elevation is a large positive outlier
        outliers = valid & (elevation > threshold)
        healthy = healthy & ~outliers

        if healthy.sum() < 4:
            break

    return healthy


# ──────────────────────────────────────────────────────────────────────────────
# Tumor mask building
# ──────────────────────────────────────────────────────────────────────────────

def build_tumor_mask_bscan(
    bscan_shape: tuple[int, int],
    retina_surface: np.ndarray,      # actual (elevated) retinal dome top
    choroid_baseline: np.ndarray,    # extrapolated eye-wall bottom
    retinal_baseline: np.ndarray,    # expected retinal surface (for detection)
    min_elevation_px: float = 5.0,
    label_val: int = 3,
) -> np.ndarray:
    """
    Build a 2-D tumor label mask for a single B-scan.

    A column is labeled as tumor if the retinal surface is elevated by at
    least ``min_elevation_px`` above the expected retinal baseline. The mask
    then fills all rows from the elevated retinal surface down to the choroid
    eye-wall baseline.

    Layout ::

       row 0        ── vitreous ──────────────────────────────────
       retina_surface[col]   ← actual dome top  (top of mask)
       ████████████████████████  ← tumor label fills this band
       choroid_baseline[col] ← eye-wall floor (bottom of mask)
       deeper       ── choroid / sclera ─────────────────────────

    Detection condition:  retinal_baseline[col] - retina_surface[col] >= min_elevation_px
    """
    H, W = bscan_shape
    mask = np.zeros((H, W), dtype=np.uint8)

    for col in range(W):
        if (np.isnan(retina_surface[col])
                or np.isnan(choroid_baseline[col])
                or np.isnan(retinal_baseline[col])):
            continue

        elevation = retinal_baseline[col] - retina_surface[col]
        if elevation < min_elevation_px:
            continue

        top = int(round(retina_surface[col]))
        bot = int(round(choroid_baseline[col]))

        if bot <= top:
            continue

        mask[max(0, top): min(H, bot + 1), col] = label_val

    return mask


# ──────────────────────────────────────────────────────────────────────────────
# Volume calculation
# ──────────────────────────────────────────────────────────────────────────────

def compute_tumor_volume_mm3(
    tumor_label_3d: np.ndarray,
    voxel_size_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    label_val: int = 3,
) -> float:
    """Count labeled voxels × physical voxel volume."""
    n_voxels = int(np.count_nonzero(tumor_label_3d == label_val))
    return n_voxels * voxel_size_mm[0] * voxel_size_mm[1] * voxel_size_mm[2]


# ──────────────────────────────────────────────────────────────────────────────
# Full-volume pipeline
# ──────────────────────────────────────────────────────────────────────────────

def compute_tumor_mask_volume(
    vol_img: np.ndarray,
    vol_labels: np.ndarray,
    anterior_label_val: int = 1,        # retina (dome surface)
    baseline_label_val: int = 2,        # choroid (eye-wall baseline)
    spline_smoothing: float | None = None,
    min_elevation_px: float = 5.0,
    margin_below_px: float = 60.0,
    margin_above_px: float = 20.0,
    edge_margin_cols: int = 15,
    robust_sigma: float = 2.0,
    robust_iters: int = 2,
    tumor_label_val: int = 3,
    enface_roi_mask: np.ndarray | None = None,
    voxel_size_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, float]:
    """
    Run the hybrid tumor detection pipeline over a 3-D volume.

    Per B-scan
    ----------
    1. Extract the anterior RETINAL surface (label 1) — the tumor dome.
    2. Extract the anterior CHOROID surface (label 2) — the eye wall.
    3. Identify HEALTHY columns:
       - With en-face ROI  → columns NOT inside the ROI  AND  where choroid present
       - Without ROI       → robust automatic (iteratively reject elevated columns)
    4. Fit TWO clamped splines from healthy columns:
       - Retinal baseline spline  (from the retinal surface)
       - Choroid baseline spline  (from the choroid surface)
    5. Per column: label as tumor if
           retinal_baseline[col] - retinal_surface[col]  >=  min_elevation_px
       (= retina is elevated above what it should be)
    6. Tumor mask fills from  retinal_surface[col]  (dome top)
                           to  choroid_baseline[col]  (eye-wall floor)

    Parameters
    ----------
    vol_img : np.ndarray, shape (D, H, W)  [reserved for future intensity refinement]
    vol_labels : np.ndarray, shape (D, H, W)
    anterior_label_val : int    Retina label. Default 1.
    baseline_label_val : int    Choroid label. Default 2.
    spline_smoothing : float or None
    min_elevation_px : float
        Minimum retinal elevation above the retinal baseline to count as tumor.
        Should be larger than the typical retina-to-choroid band thickness.
    margin_below_px : float
        Anti-dive: max px the choroid spline can extrapolate below the deepest
        observed choroid point.  **Reduce this (e.g. to 30) if the baseline
        spline dips too far down in some slices.**
    margin_above_px : float
        Anti-undershoot: max px above the shallowest choroid observation.
    edge_margin_cols : int
        Number of columns to blank out at the left and right edges of each
        B-scan.  The spline extrapolation at the extreme edges of the retinal
        scan is unreliable and often produces false vertical-line artifacts.
        Default 15.  Increase if artifacts persist; set to 0 to disable.
    robust_sigma : float    (no-ROI mode only)
    robust_iters : int      (no-ROI mode only)
    tumor_label_val : int
    enface_roi_mask : np.ndarray, shape (D, W), bool or None
        True = (B-scan, column) contains tumor tissue.  When provided these
        columns are EXCLUDED from the healthy-region fitting.
        The user draws this on the en-face projection; non-zero values are
        treated as True (supports multiple tumors with different label values).
    voxel_size_mm : (dz, dy, dx)

    Returns
    -------
    tumor_mask_3d : np.ndarray, shape (D, H, W), dtype uint8
    volume_mm3    : float
    """
    D, H, W = vol_labels.shape
    tumor_mask_3d = np.zeros((D, H, W), dtype=np.uint8)

    for bscan_idx in range(D):
        bscan_labels = vol_labels[bscan_idx]

        # 1 & 2. Actual surfaces
        retina_surface  = extract_anterior_surface(bscan_labels, label_val=anterior_label_val)
        choroid_surface = extract_anterior_surface(bscan_labels, label_val=baseline_label_val)

        # 3. Determine healthy columns for spline fitting
        if enface_roi_mask is not None:
            roi_cols = enface_roi_mask[bscan_idx]   # bool (W,)
            # Healthy = choroid present AND not inside the ROI
            healthy_cols = ~np.isnan(choroid_surface) & ~roi_cols
            # Edge case: entire B-scan inside the ROI or no choroid anywhere
            if healthy_cols.sum() < 4:
                # Fall back: ignore ROI constraint, still require choroid
                healthy_cols = ~np.isnan(choroid_surface)
            if healthy_cols.sum() < 4:
                continue
        else:
            # Robust automatic: reject elevated columns iteratively
            healthy_cols = _find_healthy_columns_robust(
                retina_surface, choroid_surface,
                sigma=robust_sigma, n_iters=robust_iters,
            )
            if healthy_cols.sum() < 4:
                continue

        # 4. Fit TWO clamped splines from the same healthy columns
        retinal_baseline = fit_spline_with_clamp(
            retina_surface, healthy_cols,
            smoothing=spline_smoothing,
            margin_above_px=margin_above_px,
            margin_below_px=margin_above_px,   # retina shouldn't vary much
            image_height=H,
        )

        choroid_baseline = fit_spline_with_clamp(
            choroid_surface, healthy_cols,
            smoothing=spline_smoothing,
            margin_above_px=margin_above_px,
            margin_below_px=margin_below_px,   # choroid needs more slack below
            image_height=H,
        )

        if retinal_baseline is None or choroid_baseline is None:
            continue

        # 5 & 6. Build the per-B-scan tumor mask
        bscan_mask = build_tumor_mask_bscan(
            (H, W),
            retina_surface=retina_surface,
            choroid_baseline=choroid_baseline,
            retinal_baseline=retinal_baseline,
            min_elevation_px=min_elevation_px,
            label_val=tumor_label_val,
        )

        # Zero out columns near the ACTUAL DATA BOUNDARY (adaptive edge margin).
        # The retinal scan is oval-shaped: narrow B-scans at the start/end of
        # the volume have fewer valid columns than wide ones in the middle.
        # Blanking a fixed number of columns from the image edge would be too
        # aggressive on narrow B-scans and too lenient on wide ones.
        # Instead, find where the retinal label data actually starts/ends per
        # B-scan and blank edge_margin_cols columns inward from those positions.
        if edge_margin_cols > 0:
            valid_cols = np.where(~np.isnan(retina_surface))[0]
            if len(valid_cols) > 0:
                first_valid = int(valid_cols[0])
                last_valid  = int(valid_cols[-1])
                # Blank from image left up to and including (first_valid + margin - 1)
                blank_left  = min(first_valid + edge_margin_cols, W)
                # Blank from (last_valid - margin + 1) to image right
                blank_right = max(last_valid - edge_margin_cols + 1, 0)
                bscan_mask[:, :blank_left]   = 0
                bscan_mask[:, blank_right:]  = 0

        tumor_mask_3d[bscan_idx] = bscan_mask

    volume_mm3 = compute_tumor_volume_mm3(
        tumor_mask_3d,
        voxel_size_mm=voxel_size_mm,
        label_val=tumor_label_val,
    )
    return tumor_mask_3d, volume_mm3
