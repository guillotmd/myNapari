"""
Smoke tests for florian_quadratic_tumor_vol._tumor_funcs — quadratic dual-spline algorithm.

Detection:  retinal surface is elevated vs. retinal baseline spline (k=2)
Extent:     mask fills from elevated dome to choroid baseline spline (k=2)
Healthy cols: columns outside en-face ROI where choroid is present
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
from florian_quadratic_tumor_vol._tumor_funcs import (
    extract_edges,
    extract_anterior_surface,
    extract_posterior_surface,
    fit_spline_with_clamp,
    clamp_baseline,
    build_tumor_mask_bscan,
    compute_tumor_volume_mm3,
    compute_tumor_mask_volume,
    _find_healthy_columns_robust,
    interpolate_baselines_3d,
)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def check(name, cond, extra=""):
    print(f"  [{PASS if cond else FAIL}] {name}" + (f"  ({extra})" if extra else ""))
    if not cond:
        raise AssertionError(f"Test failed: {name}")

H, W = 120, 80

# ── T1: robust edge extraction ───────────────────────────────────────────────
print("\n=== T1: robust edge extraction (extract_edges) ===")
labels = np.zeros((H, W), dtype=np.uint8)
labels[20:30, :] = 1    # retina band rows 20-29
labels[30:45, :] = 2    # choroid band rows 30-44
ret_edge, cho_edge = extract_edges(labels, retina_label=1, choroid_label=2)
check("Retina edge = 20", np.nanmean(ret_edge) == 20.0)
check("Choroid edge = 30", np.nanmean(cho_edge) == 30.0)

# Test min_thickness filtering — add a thin floating seed
labels_seed = labels.copy()
labels_seed[5:7, 10] = 1  # 2px seed — should be filtered with min_thickness=5
ret_seed, _ = extract_edges(labels_seed, min_thickness=5)
check("Thin seed filtered (still picks row 20)", ret_seed[10] == 20.0)

# Test ignore_top_px
labels_top = labels.copy()
labels_top[2:10, 15] = 1  # artifact near top
ret_top, _ = extract_edges(labels_top, ignore_top_px=15)
check("ignore_top_px filters top artifact", ret_top[15] == 20.0)

# ── T1b: simple surface extraction (backward compat) ─────────────────────────
print("\n=== T1b: simple surface extraction ===")
check("Anterior retina = 20", np.all(extract_anterior_surface(labels, 1) == 20.0))
check("Anterior choroid = 30", np.all(extract_anterior_surface(labels, 2) == 30.0))
check("Posterior retina = 29", np.all(extract_posterior_surface(labels, 1) == 29.0))

# ── T2: clamp_baseline ───────────────────────────────────────────────────────
print("\n=== T2: clamp_baseline ===")
anchor = np.full(W, 40.0); anchor[30:50] = np.nan
raw = np.full(W, 40.0);   raw[30:50] = 95.0
clamped = clamp_baseline(raw, anchor, margin_above_px=10, margin_below_px=15, image_height=H)
check("Deep region clamped to ≤ 55", np.all(clamped[30:50] <= 55.0))
check("Healthy region unchanged", np.allclose(clamped[:30], 40.0))

# ── T3: fit_spline_with_clamp (quadratic k=2) ────────────────────────────────
print("\n=== T3: fit_spline_with_clamp (quadratic) ===")
retina_s  = np.full(W, 20.0); retina_s [25:55] = np.nan
choroid_s = np.full(W, 30.0); choroid_s[25:55] = np.nan
healthy = ~np.isnan(retina_s)
ret_spl = fit_spline_with_clamp(retina_s,  healthy, image_height=H, margin_below_px=10)
cho_spl = fit_spline_with_clamp(choroid_s, healthy, image_height=H, margin_below_px=15)
check("Retina spline not None", ret_spl is not None)
check("Choroid spline not None", cho_spl is not None)
check("Retina healthy region ≈ 20", np.allclose(ret_spl[:25], 20.0, atol=1.0))
check("Choroid healthy region ≈ 30", np.allclose(cho_spl[:25], 30.0, atol=1.0))
check("No wild extrapolation", np.all(cho_spl[25:55] <= 45.0))

# ── T4: build_tumor_mask_bscan ───────────────────────────────────────────────
print("\n=== T4: build_tumor_mask_bscan ===")
rsurf  = np.full(W, 10.0)
cbline = np.full(W, 40.0)
rbline = np.full(W, 22.0)

mask = build_tumor_mask_bscan((H, W), rsurf, cbline, rbline, min_elevation_px=5.0)
check("Rows 10-40 labeled", np.all(mask[10:41, :] == 3))
check("Vitreous (rows 0-9) = 0", np.all(mask[:10, :] == 0))
check("Deep (rows 41+) = 0", np.all(mask[41:, :] == 0))

mask_flat = build_tumor_mask_bscan((H, W), np.full(W, 20.0), cbline, rbline, min_elevation_px=5.0)
check("Small elevation → no label", np.all(mask_flat == 0))

# ── T5: robust healthy-column detection ──────────────────────────────────────
print("\n=== T5: robust healthy-column detection ===")
ret_full = np.full(W, 20.0);  ret_full[30:50] = 5.0
cho_full = np.full(W, 30.0);  cho_full[30:50] = np.nan
hc = _find_healthy_columns_robust(ret_full, cho_full, sigma=1.5, n_iters=2)
check("Healthy cols exist", hc.sum() > 0)
check("Most elevated cols excluded", hc[30:50].mean() < 0.3)
check("Non-elevated cols mostly kept", hc[:30].mean() > 0.7)

# ── T6: 3D cross-slice interpolation ─────────────────────────────────────────
print("\n=== T6: interpolate_baselines_3d ===")
baselines = {0: np.full(W, 30.0), 4: np.full(W, 34.0)}
full = interpolate_baselines_3d(baselines, 5, W)
check("Shape correct", full.shape == (5, W))
check("Slice 0 = 30", np.allclose(full[0], 30.0))
check("Slice 4 = 34", np.allclose(full[4], 34.0))
check("Slice 2 interpolated ≈ 32", np.allclose(full[2], 32.0, atol=0.1))

# ── T7: volume calculation ───────────────────────────────────────────────────
print("\n=== T7: compute_tumor_volume_mm3 ===")
vol3d = np.zeros((5, H, W), dtype=np.uint8)
vol3d[1:4, 10:41, :] = 3
vox = (0.01, 0.005, 0.005)
expected = 3 * 31 * 80 * vox[0] * vox[1] * vox[2]
got = compute_tumor_volume_mm3(vol3d, vox)
check("Volume correct", abs(got - expected) < 1e-9)

# ── T8: full pipeline (ROI-guided) ───────────────────────────────────────────
print("\n=== T8: full pipeline (hybrid, ROI-guided) ===")
D = 5
vol_img = np.zeros((D, H, W), dtype=np.float32)
vol_lbl = np.zeros((D, H, W), dtype=np.uint8)

vol_lbl[:, 20:30, :] = 1
vol_lbl[:, 30:45, :] = 2

for b in [1, 2, 3]:
    vol_lbl[b, 20:30, 25:55] = 0
    vol_lbl[b, 30:45, 25:55] = 0
    vol_lbl[b, 5:15,  25:55] = 1

roi = np.zeros((D, W), dtype=np.uint8)
roi[1:4, 25:55] = 3

mask3d, vol_mm3 = compute_tumor_mask_volume(
    vol_img, vol_lbl,
    anterior_label_val=1, baseline_label_val=2,
    min_elevation_px=5.0, margin_below_px=25.0,
    tumor_label_val=3,
    enface_roi_mask=(roi > 0),
    voxel_size_mm=(1.0, 1.0, 1.0),
)
check("Shape correct", mask3d.shape == (D, H, W))
check("Tumor voxels exist", np.count_nonzero(mask3d == 3) > 0)
check("Volume > 0", vol_mm3 > 0)
check("Healthy B-scans not labeled", np.all(mask3d[[0, 4], :, :] == 0))
check("Vitreous clear", np.all(mask3d[:, :5, :] == 0))
print(f"  → Total tumor voxels: {np.count_nonzero(mask3d == 3)}, volume: {vol_mm3:.2f} mm³")

# ── T9: full pipeline (no ROI, robust auto) ──────────────────────────────────
print("\n=== T9: full pipeline (robust auto, no ROI) ===")
mask3d_auto, vol_auto = compute_tumor_mask_volume(
    vol_img, vol_lbl,
    anterior_label_val=1, baseline_label_val=2,
    min_elevation_px=5.0, margin_below_px=25.0,
    robust_sigma=1.5, robust_iters=2,
    tumor_label_val=3,
    voxel_size_mm=(1.0, 1.0, 1.0),
)
check("Shape correct", mask3d_auto.shape == (D, H, W))
check("Healthy B-scans not labeled", np.all(mask3d_auto[[0, 4], :, :] == 0))
print(f"  → Total tumor voxels: {np.count_nonzero(mask3d_auto == 3)}, volume: {vol_auto:.2f} mm³")

# ── T10: diagnostic lines ────────────────────────────────────────────────────
print("\n=== T10: diagnostic lines ===")
mask3d_diag, _ = compute_tumor_mask_volume(
    vol_img, vol_lbl,
    anterior_label_val=1, baseline_label_val=2,
    min_elevation_px=5.0, margin_below_px=25.0,
    tumor_label_val=3,
    enface_roi_mask=(roi > 0),
    voxel_size_mm=(1.0, 1.0, 1.0),
    show_diagnostic_lines=True,
)
has_label4 = np.any(mask3d_diag == 4)
has_label5 = np.any(mask3d_diag == 5)
check("Retinal baseline lines drawn (label 4)", has_label4)
check("Choroid baseline lines drawn (label 5)", has_label5)

print("\n✅  All tests passed.\n")
