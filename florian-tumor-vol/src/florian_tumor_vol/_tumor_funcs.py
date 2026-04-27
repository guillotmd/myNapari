"""
_tumor_funcs.py
================
Training-free, signal-processing-based functions for detecting and measuring
intraocular tumors (e.g. retinoblastoma) in OCT B-scan volumes.

Geometry / clinical context
---------------------------
In a standard OCT B-scan (row 0 = anterior / vitreous):

   Row 0   ──── vitreous ──────────────────────────────────────
           ╭─────────────╮  ← retinal dome (ELEVATED by tumor)
   retina  │  label 1    │  ← normal retina sits flat here in healthy areas
           ╰──── label 2 ╯  ← choroid (eye wall)
   deeper  ─────────────────────────────────────────────────────

The tumor pushes the retina UPWARD.  In healthy regions, the retinal surface
sits at a roughly predictable depth; over the tumor it is elevated by tens-
to-hundreds of pixels.  The choroid (label 2) is absent under the tumor.

Why quadratic (k=2) splines?
-----------------------------
Both the retinal surface and the RPE/choroid follow the curvature of the
eye globe.  In a B-scan cross-section, this globe curvature appears as a
parabola (concave-up).  A degree-2 polynomial:
  • Always produces a parabolic arc — matches the globe curvature inherently
  • Cannot oscillate (no inflection points, unlike cubic k=3)
  • Extrapolates smoothly across the tumor gap with a natural arc
  • The retina and choroid are concentric arcs of the same globe

Algorithm (per B-scan)
-----------------------
1. Extract anterior retinal and choroid surfaces using robust group-based
   edge detection (filters thin fragments and vitreous seeds).
2. Identify healthy columns (where choroid is present and outside the ROI).
3. Fit TWO quadratic (k=2) splines from healthy columns:
   ① Retinal baseline  → "where the retina would be without the tumor"
   ② Choroid baseline   → "where the eye wall would be"
4. A column is tumor if:  retinal_baseline - actual_retina > min_elevation_px
5. Tumor mask fills from retinal dome (top) to choroid baseline (bottom).

Label convention (confirmed from ONNX model source):
    0 = vitreous / background
    1 = retina   (dome / anterior surface for detection)
    2 = choroid  (eye-wall baseline for depth extent)
    3 = tumor mask output (default)
    4 = diagnostic: retinal baseline curve (optional)
    5 = diagnostic: choroid baseline curve (optional)
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d


# ──────────────────────────────────────────────────────────────────────────────
# Robust edge extraction (ported from florian-retinoblastoma-vol)
# ──────────────────────────────────────────────────────────────────────────────

def extract_edges(
    segmentation_slice: np.ndarray,
    retina_label: int = 1,
    choroid_label: int = 2,
    min_thickness: int = 5,
    ignore_top_px: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the topmost retinal edge and the topmost choroidal edge from
    a 2D segmentation slice, with robustness against thin floating artifacts
    (e.g. vitreous seeds) and noise at the top of the image.

    Unlike simple argmax, this groups contiguous label runs per column and
    filters out groups thinner than ``min_thickness``.  This prevents isolated
    vitreous seeds or segmentation noise from being mistaken for the true
    retinal or choroidal surface.

    Parameters
    ----------
    segmentation_slice : (H, W) uint8
    retina_label, choroid_label : label IDs in the segmentation
    min_thickness : minimum contiguous pixel run to accept as a real layer
    ignore_top_px : zero out labels in the topmost N rows (artifact filter)

    Returns
    -------
    inner_retina_edge : (W,) float, NaN where absent
    inner_choroid_edge : (W,) float, NaN where absent
    """
    # Handle both indexed labels and RGB segmentation images
    if segmentation_slice.ndim == 3 and segmentation_slice.shape[2] >= 3:
        is_retina = segmentation_slice[..., 0] > 128
        is_choroid = segmentation_slice[..., 2] > 128
    else:
        is_retina = segmentation_slice == retina_label
        is_choroid = segmentation_slice == choroid_label

    height, width = is_retina.shape

    # Zero out the top region to filter floating artifacts
    if ignore_top_px > 0:
        is_retina = is_retina.copy()
        is_choroid = is_choroid.copy()
        is_retina[:ignore_top_px, :] = False
        is_choroid[:ignore_top_px, :] = False

    inner_retina_edge = np.full(width, np.nan)
    inner_choroid_edge = np.full(width, np.nan)

    for x in range(width):
        # --- Retina: find contiguous groups, filter thin ones, take topmost ---
        ret_indices = np.where(is_retina[:, x])[0]
        if len(ret_indices) > 0:
            breaks = np.where(np.diff(ret_indices) > 1)[0]
            splits = np.split(ret_indices, breaks + 1)
            valid_splits = [s for s in splits if len(s) >= min_thickness]
            if valid_splits:
                inner_retina_edge[x] = valid_splits[0][0]  # topmost valid group
            else:
                inner_retina_edge[x] = max(splits, key=len)[0]  # fallback: largest

        # --- Choroid: find contiguous groups, filter thin ones, take deepest ---
        cho_indices = np.where(is_choroid[:, x])[0]
        if len(cho_indices) > 0:
            breaks = np.where(np.diff(cho_indices) > 1)[0]
            splits = np.split(cho_indices, breaks + 1)
            valid_splits = [s for s in splits if len(s) >= min_thickness]
            if valid_splits:
                inner_choroid_edge[x] = valid_splits[-1][0]  # deepest valid group
            else:
                inner_choroid_edge[x] = max(splits, key=len)[0]

    return inner_retina_edge, inner_choroid_edge


# ──────────────────────────────────────────────────────────────────────────────
# Simple surface extraction (kept as utility / test helpers)
# ──────────────────────────────────────────────────────────────────────────────

def extract_anterior_surface(bscan_labels: np.ndarray, label_val: int) -> np.ndarray:
    """Per-column TOPMOST pixel belonging to ``label_val``. NaN where absent."""
    mask = bscan_labels == label_val
    W = mask.shape[1]
    has_label = mask.any(axis=0)
    surface = np.full(W, np.nan, dtype=float)
    surface[has_label] = mask.argmax(axis=0)[has_label].astype(float)
    return surface


def extract_posterior_surface(bscan_labels: np.ndarray, label_val: int) -> np.ndarray:
    """Per-column DEEPEST pixel belonging to ``label_val``. NaN where absent."""
    mask = bscan_labels == label_val
    H, W = mask.shape
    has_label = mask.any(axis=0)
    surface = np.full(W, np.nan, dtype=float)
    surface[has_label] = (H - 1 - mask[::-1].argmax(axis=0))[has_label].astype(float)
    return surface


# ──────────────────────────────────────────────────────────────────────────────
# Quadratic spline fitting
# ──────────────────────────────────────────────────────────────────────────────

def _fit_quadratic_spline(
    anchor_rows: np.ndarray,
    col_include: np.ndarray,
    smoothing: float | None,
    min_valid_frac: float = 0.05,
) -> np.ndarray | None:
    """
    Fit a degree-2 (quadratic) UnivariateSpline through selected columns,
    then evaluate on ALL columns to produce a continuous baseline.

    Clinical rationale
    ------------------
    Both the retinal surface and the RPE/choroid follow the curvature of
    the eye globe.  In a B-scan cross-section this globe curvature appears
    as a parabola (concave-up in image coordinates where row 0 is at top).
    A quadratic polynomial (degree 2) is the natural mathematical model:
      • Always parabolic — matches the globe curvature inherently
      • Cannot oscillate (no inflection points like cubic k=3 can)
      • Extrapolates smoothly across the tumor gap with a natural arc

    Parameters
    ----------
    anchor_rows : (W,) float — surface row positions, NaN where absent
    col_include : (W,) bool — which columns to use as fitting anchors
    smoothing : smoothing factor for UnivariateSpline (None = interpolating)
    min_valid_frac : minimum fraction of columns needed to attempt fitting

    Returns
    -------
    baseline : (W,) float — evaluated spline, or None if too few anchors
    """
    W = len(anchor_rows)
    valid = col_include & ~np.isnan(anchor_rows)
    n_valid = valid.sum()

    # Need at least 3 points for a quadratic, and a minimum fraction of W
    if n_valid < 3 or n_valid < W * min_valid_frac:
        return None

    cols = np.where(valid)[0].astype(float)
    rows = anchor_rows[valid]

    # k=2: ALWAYS quadratic — the globe is a parabola in cross-section
    k = 2
    spline = UnivariateSpline(cols, rows, s=smoothing, k=k)
    result = spline(np.arange(W, dtype=float))
    return np.clip(result, 0, None)


def clamp_baseline(
    baseline: np.ndarray,
    anchor_rows: np.ndarray,
    margin_above_px: float = 20.0,
    margin_below_px: float = 60.0,
    image_height: int | None = None,
) -> np.ndarray:
    """
    Clamp a spline baseline to the observed range of anchor_rows ± margins.

    Even with quadratic fitting, the extrapolated parabola may extend slightly
    beyond the physical eye wall.  This safety clamp prevents the baseline
    from reaching impossibly deep or shallow positions.
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
    healthy_col_mask: np.ndarray,
    smoothing: float | None = None,
    margin_above_px: float = 20.0,
    margin_below_px: float = 60.0,
    image_height: int | None = None,
    min_valid_frac: float = 0.05,
) -> np.ndarray | None:
    """
    Fit a bounded quadratic spline through the anchor surface using only
    healthy columns, then clamp the extrapolated values.
    """
    baseline = _fit_quadratic_spline(
        anchor_rows, healthy_col_mask, smoothing, min_valid_frac
    )
    if baseline is None:
        return None
    return clamp_baseline(
        baseline, anchor_rows,
        margin_above_px=margin_above_px,
        margin_below_px=margin_below_px,
        image_height=image_height,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Robust healthy-column identification
# ──────────────────────────────────────────────────────────────────────────────

def _find_healthy_columns_robust(
    retina_surface: np.ndarray,
    choroid_surface: np.ndarray,
    sigma: float = 2.0,
    n_iters: int = 2,
) -> np.ndarray:
    """
    Without a user ROI, identify "healthy" columns automatically by:
    1. Starting with columns where the choroid IS present (absent under tumor).
    2. Fitting an initial quadratic baseline through those columns.
    3. Iteratively rejecting columns where the retina is significantly
       elevated above the current baseline estimate.

    Returns a boolean mask of shape (W,).
    """
    healthy = ~np.isnan(choroid_surface) & ~np.isnan(retina_surface)

    if healthy.sum() < 3:
        return healthy

    for _ in range(n_iters):
        baseline = _fit_quadratic_spline(retina_surface, healthy, smoothing=None)
        if baseline is None:
            break

        # Positive elevation = retina is above (shallower than) baseline
        elevation = baseline - retina_surface
        valid = ~np.isnan(retina_surface)
        elev_valid = elevation[valid & healthy]

        if len(elev_valid) < 2:
            break

        threshold = elev_valid.mean() + sigma * elev_valid.std()
        outliers = valid & (elevation > threshold)
        healthy = healthy & ~outliers

        if healthy.sum() < 3:
            break

    return healthy


def _filter_choroid_outliers(
    choroid_surface: np.ndarray,
    retina_surface: np.ndarray,
    healthy_cols: np.ndarray,
    max_above_retina_px: float = 10.0,
) -> np.ndarray:
    """
    Reject choroid detections that are ABOVE (shallower than) the retinal
    surface.  This catches a common ONNX segmentation error where tissue
    inside the tumor is mislabeled as choroid.

    Clinical rationale
    ------------------
    The choroid always sits BELOW the retina (larger row index).  If the
    segmentation places "choroid" at a row that is above or only slightly
    below the retinal surface, it's a mislabel — real choroid can't be
    shallower than the retina.

    Additionally, if a preliminary choroid baseline can be fitted from
    healthy columns, any choroid detection more than ``max_above_retina_px``
    pixels ABOVE that baseline is also rejected as an outlier (the real
    choroid follows a smooth parabolic arc and shouldn't jump upward).

    Parameters
    ----------
    choroid_surface : (W,) float — raw choroid edge, NaN where absent
    retina_surface : (W,) float — raw retina edge, NaN where absent
    healthy_cols : (W,) bool — columns considered healthy for fitting
    max_above_retina_px : max px a choroid detection can be above the
        preliminary baseline before being rejected

    Returns
    -------
    cleaned_choroid : (W,) float — choroid surface with outliers set to NaN
    """
    cleaned = choroid_surface.copy()

    # Rule 1: Choroid must be BELOW retina (larger row index).
    # If choroid is at a shallower row than retina, it's a segmentation error.
    both_valid = ~np.isnan(cleaned) & ~np.isnan(retina_surface)
    above_retina = both_valid & (cleaned < retina_surface)
    cleaned[above_retina] = np.nan

    # Rule 2: Fit a preliminary baseline from healthy columns and reject
    # choroid detections that are significantly above (shallower than) it.
    prelim_healthy = healthy_cols & ~np.isnan(cleaned)
    if prelim_healthy.sum() >= 3:
        prelim_baseline = _fit_quadratic_spline(
            cleaned, prelim_healthy, smoothing=None
        )
        if prelim_baseline is not None:
            valid_cho = ~np.isnan(cleaned)
            # Positive = choroid is above (shallower than) the baseline
            deviation = prelim_baseline - cleaned
            too_shallow = valid_cho & (deviation > max_above_retina_px)
            cleaned[too_shallow] = np.nan

    return cleaned


# ──────────────────────────────────────────────────────────────────────────────
# 3D cross-slice baseline interpolation (from retinoblastoma-vol)
# ──────────────────────────────────────────────────────────────────────────────

def interpolate_baselines_3d(
    baselines: dict[int, np.ndarray],
    n_slices: int,
    width: int,
) -> np.ndarray:
    """
    Interpolate completely missing baselines along the Z-axis (across slices).

    Some B-scans may fail to produce a valid spline (e.g. the entire slice is
    inside the tumor, or the segmentation is too noisy).  This function fills
    those gaps by linearly interpolating each column's baseline value from
    neighbouring slices that DID succeed, with flat extrapolation at the edges.

    This ensures every slice has a continuous baseline for mask building.
    """
    full = np.full((n_slices, width), np.nan)
    for i, b in baselines.items():
        full[i] = b

    z_indices = np.arange(n_slices)
    for x in range(width):
        col = full[:, x]
        valid_z = ~np.isnan(col)
        if not np.any(valid_z):
            continue

        valid_idx = z_indices[valid_z]
        valid_data = col[valid_z]

        if len(valid_idx) == 1:
            full[:, x] = valid_data[0]
            continue

        interp = interp1d(
            valid_idx, valid_data,
            kind='linear', bounds_error=False,
            fill_value=(valid_data[0], valid_data[-1]),
        )
        full[:, x] = interp(z_indices)

    return full


# ──────────────────────────────────────────────────────────────────────────────
# Tumor mask building
# ──────────────────────────────────────────────────────────────────────────────

def build_tumor_mask_bscan(
    bscan_shape: tuple[int, int],
    retina_surface: np.ndarray,
    choroid_baseline: np.ndarray,
    retinal_baseline: np.ndarray,
    min_elevation_px: float = 5.0,
    label_val: int = 3,
) -> np.ndarray:
    """
    Build a 2-D tumor label mask for a single B-scan.

    Detection logic:
      A column is labeled as tumor if the retinal surface is elevated by at
      least ``min_elevation_px`` above the expected retinal baseline.

    Mask fill:
      The mask fills all rows from the elevated retinal dome (top) down to
      the choroid eye-wall baseline (bottom).

    Layout::

       row 0        ── vitreous ──────────────────────────────────
       retina_surface[col]   ← actual dome top  (top of mask)
       ████████████████████████  ← tumor label fills this band
       choroid_baseline[col] ← eye-wall floor (bottom of mask)
       deeper       ── choroid / sclera ─────────────────────────
    """
    H, W = bscan_shape
    mask = np.zeros((H, W), dtype=np.uint8)

    for col in range(W):
        if (np.isnan(retina_surface[col])
                or np.isnan(choroid_baseline[col])
                or np.isnan(retinal_baseline[col])):
            continue

        # Detection: retina elevated above where it "should be"
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
    anterior_label_val: int = 1,
    baseline_label_val: int = 2,
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
    min_layer_thickness: int = 5,
    ignore_top_px: int = 0,
    show_diagnostic_lines: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Run the hybrid tumor detection pipeline over a 3-D volume.

    This combines the best approaches from both florian-tumor-vol and
    florian-retinoblastoma-vol:

    1. Robust edge extraction — group-based filtering to reject vitreous
       seeds and thin artifacts (from retinoblastoma-vol)
    2. Quadratic (k=2) spline fitting — both retinal and choroid baselines
       use degree-2 parabolic splines matching the natural globe curvature
    3. 3D cross-slice interpolation — fills missing baselines across the
       Z-axis for slices where fitting fails (from retinoblastoma-vol)
    4. Adaptive edge blanking — data-boundary-relative column blanking
    5. Optional diagnostic lines — retinal curve (label 4) and choroid
       baseline (label 5) overlaid on the output mask

    Per B-scan
    ----------
    1. Extract surfaces using robust group-based edge detection.
    2. Identify HEALTHY columns (outside ROI, choroid present).
    3. Fit TWO quadratic (k=2) clamped splines from healthy columns.
    4. Label as tumor if retina elevated above retinal baseline.
    5. Fill mask from dome top to choroid baseline.
    6. (Optional) Draw diagnostic lines on the output.

    Parameters
    ----------
    vol_img : (D, H, W) — reserved for future intensity refinement
    vol_labels : (D, H, W) — segmentation labels
    min_layer_thickness : min contiguous pixels to accept as real layer
    ignore_top_px : zero out labels in the topmost N rows (artifact filter)
    show_diagnostic_lines : if True, draw retinal (4) and choroid (5) curves
    [other params same as before]

    Returns
    -------
    tumor_mask_3d : (D, H, W) uint8
    volume_mm3 : float
    """
    D, H, W = vol_labels.shape
    tumor_mask_3d = np.zeros((D, H, W), dtype=np.uint8)

    print(f"\n{'='*60}")
    print(f"  Tumor Detection Pipeline — {D} B-scans, {H}×{W} px")
    print(f"{'='*60}")

    # Phase 1: Extract surfaces using robust edge detection
    retina_surfaces = {}
    choroid_surfaces_raw = {}

    for bscan_idx in range(D):
        print(f"\r  Phase 1/4 — Extracting edges: "
              f"[{'█' * ((bscan_idx+1)*30//D)}{' ' * (30-(bscan_idx+1)*30//D)}] "
              f"{bscan_idx+1}/{D}", end="", flush=True)

        retina_surface, choroid_surface = extract_edges(
            vol_labels[bscan_idx],
            retina_label=anterior_label_val,
            choroid_label=baseline_label_val,
            min_thickness=min_layer_thickness,
            ignore_top_px=ignore_top_px,
        )
        retina_surfaces[bscan_idx] = retina_surface
        choroid_surfaces_raw[bscan_idx] = choroid_surface

    print(f"\r  Phase 1/4 — Extracting edges: "
          f"[{'█'*30}] {D}/{D} ✓                  ")

    # Phase 1.5: Choroid sanity filter
    # The ONNX segmentation sometimes mislabels tumor tissue as "choroid".
    # These false choroid detections appear at unreasonably shallow depths
    # (above the retina or far above the expected choroid baseline).
    # We reject them before fitting the final baselines.
    choroid_surfaces = {}
    for bscan_idx in range(D):
        print(f"\r  Phase 2/4 — Filtering choroid outliers: "
              f"[{'█' * ((bscan_idx+1)*30//D)}{' ' * (30-(bscan_idx+1)*30//D)}] "
              f"{bscan_idx+1}/{D}", end="", flush=True)

        retina_s = retina_surfaces[bscan_idx]
        choroid_s = choroid_surfaces_raw[bscan_idx]

        # Identify preliminary healthy columns for the sanity check
        if enface_roi_mask is not None:
            roi_cols = enface_roi_mask[bscan_idx]
            prelim_healthy = ~np.isnan(choroid_s) & ~roi_cols
            if prelim_healthy.sum() < 3:
                prelim_healthy = ~np.isnan(choroid_s)
        else:
            prelim_healthy = ~np.isnan(choroid_s) & ~np.isnan(retina_s)

        choroid_surfaces[bscan_idx] = _filter_choroid_outliers(
            choroid_s, retina_s, prelim_healthy,
            max_above_retina_px=margin_above_px,
        )

    print(f"\r  Phase 2/4 — Filtering choroid outliers: "
          f"[{'█'*30}] {D}/{D} ✓                  ")

    # Phase 2: Fit per-slice baselines from healthy columns
    retinal_baselines_dict = {}
    choroid_baselines_dict = {}

    for bscan_idx in range(D):
        print(f"\r  Phase 3/4 — Fitting quadratic baselines: "
              f"[{'█' * ((bscan_idx+1)*30//D)}{' ' * (30-(bscan_idx+1)*30//D)}] "
              f"{bscan_idx+1}/{D}", end="", flush=True)

        retina_surface = retina_surfaces[bscan_idx]
        choroid_surface = choroid_surfaces[bscan_idx]

        # Determine healthy columns for spline fitting
        if enface_roi_mask is not None:
            roi_cols = enface_roi_mask[bscan_idx]
            healthy_cols = ~np.isnan(choroid_surface) & ~roi_cols
            if healthy_cols.sum() < 3:
                healthy_cols = ~np.isnan(choroid_surface)
            if healthy_cols.sum() < 3:
                continue
        else:
            healthy_cols = _find_healthy_columns_robust(
                retina_surface, choroid_surface,
                sigma=robust_sigma, n_iters=robust_iters,
            )
            if healthy_cols.sum() < 3:
                continue

        # Fit TWO quadratic (k=2) clamped splines from healthy columns
        # Both use k=2 because both surfaces follow the globe curvature
        retinal_baseline = fit_spline_with_clamp(
            retina_surface, healthy_cols,
            smoothing=spline_smoothing,
            margin_above_px=margin_above_px,
            margin_below_px=margin_above_px,  # retina doesn't vary much
            image_height=H,
        )

        choroid_baseline = fit_spline_with_clamp(
            choroid_surface, healthy_cols,
            smoothing=spline_smoothing,
            margin_above_px=margin_above_px,
            margin_below_px=margin_below_px,  # choroid needs more slack
            image_height=H,
        )

        if retinal_baseline is not None:
            retinal_baselines_dict[bscan_idx] = retinal_baseline
        if choroid_baseline is not None:
            choroid_baselines_dict[bscan_idx] = choroid_baseline

    print(f"\r  Phase 3/4 — Fitting quadratic baselines: "
          f"[{'█'*30}] {D}/{D} ✓                  ")

    # 3D cross-slice interpolation — fill gaps where per-slice fitting
    # failed by interpolating from neighbouring successful slices
    print("  Interpolating baselines across slices...")
    full_retina_surfaces = interpolate_baselines_3d(retina_surfaces, D, W)
    full_retinal_baselines = interpolate_baselines_3d(retinal_baselines_dict, D, W)
    full_choroid_baselines = interpolate_baselines_3d(choroid_baselines_dict, D, W)

    # Phase 4: Build masks and (optionally) draw diagnostic lines
    for bscan_idx in range(D):
        print(f"\r  Phase 4/4 — Building tumor masks: "
              f"[{'█' * ((bscan_idx+1)*30//D)}{' ' * (30-(bscan_idx+1)*30//D)}] "
              f"{bscan_idx+1}/{D}", end="", flush=True)

        r_surf = full_retina_surfaces[bscan_idx]
        r_base = full_retinal_baselines[bscan_idx]
        c_base = full_choroid_baselines[bscan_idx]

        # Skip if any baseline is entirely NaN (no data anywhere)
        if np.all(np.isnan(r_surf)) or np.all(np.isnan(r_base)) or np.all(np.isnan(c_base)):
            continue

        # Build per-B-scan tumor mask
        bscan_mask = build_tumor_mask_bscan(
            (H, W),
            retina_surface=r_surf,
            choroid_baseline=c_base,
            retinal_baseline=r_base,
            min_elevation_px=min_elevation_px,
            label_val=tumor_label_val,
        )

        # Adaptive edge blanking — blank columns near the ACTUAL DATA
        # BOUNDARY (not the image edge).
        if edge_margin_cols > 0:
            raw_surf = retina_surfaces.get(bscan_idx, r_surf)
            valid_cols = np.where(~np.isnan(raw_surf))[0]
            if len(valid_cols) > 0:
                first_valid = int(valid_cols[0])
                last_valid = int(valid_cols[-1])
                blank_left = min(first_valid + edge_margin_cols, W)
                blank_right = max(last_valid - edge_margin_cols + 1, 0)
                bscan_mask[:, :blank_left] = 0
                bscan_mask[:, blank_right:] = 0

        tumor_mask_3d[bscan_idx] = bscan_mask

        # Optional diagnostic lines
        if show_diagnostic_lines:
            for x in range(W):
                if not np.isnan(c_base[x]):
                    y = int(round(c_base[x]))
                    if 0 <= y < H:
                        tumor_mask_3d[bscan_idx, max(0, y-1):min(H, y+2), x] = 5
                if not np.isnan(r_base[x]):
                    y = int(round(r_base[x]))
                    if 0 <= y < H:
                        tumor_mask_3d[bscan_idx, max(0, y-1):min(H, y+2), x] = 4

    print(f"\r  Phase 4/4 — Building tumor masks: "
          f"[{'█'*30}] {D}/{D} ✓                  ")

    volume_mm3 = compute_tumor_volume_mm3(
        tumor_mask_3d,
        voxel_size_mm=voxel_size_mm,
        label_val=tumor_label_val,
    )

    n_tumor = int(np.count_nonzero(tumor_mask_3d == tumor_label_val))
    print(f"\n  ✅ Done — {n_tumor:,} tumor voxels, volume = {volume_mm3:.4f} mm³")
    print(f"{'='*60}\n")

    return tumor_mask_3d, volume_mm3
