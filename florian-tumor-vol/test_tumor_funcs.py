"""
Smoke test for florian_tumor_vol._tumor_funcs — hybrid dual-spline algorithm.

Detection:  retinal surface is elevated vs. retinal baseline spline
Extent:     mask fills from elevated dome to choroid baseline spline
Healthy cols: columns outside en-face ROI where choroid is present
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
from florian_tumor_vol._tumor_funcs import (
    extract_anterior_surface,
    extract_posterior_surface,
    fit_spline_with_clamp,
    clamp_baseline,
    build_tumor_mask_bscan,
    compute_tumor_volume_mm3,
    compute_tumor_mask_volume,
    _find_healthy_columns_robust,
)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def check(name, cond, extra=""):
    print(f"  [{PASS if cond else FAIL}] {name}" + (f"  ({extra})" if extra else ""))
    if not cond:
        raise AssertionError(f"Test failed: {name}")

H, W = 120, 80

# ── T1: surface extraction ────────────────────────────────────────────────────
print("\n=== T1: surface extraction ===")
labels = np.zeros((H, W), dtype=np.uint8)
labels[20:30, :] = 1    # retina band rows 20-29
labels[30:45, :] = 2    # choroid band rows 30-44
check("Retina anterior = 20", np.all(extract_anterior_surface(labels, 1) == 20.0))
check("Choroid anterior = 30", np.all(extract_anterior_surface(labels, 2) == 30.0))
check("Retina posterior = 29", np.all(extract_posterior_surface(labels, 1) == 29.0))
check("Choroid posterior = 44", np.all(extract_posterior_surface(labels, 2) == 44.0))
labels_gap = labels.copy(); labels_gap[:, 5] = 0
check("Missing col NaN", np.isnan(extract_anterior_surface(labels_gap, 2)[5]))

# ── T2: clamp_baseline ───────────────────────────────────────────────────────
print("\n=== T2: clamp_baseline ===")
anchor = np.full(W, 40.0); anchor[30:50] = np.nan
raw = np.full(W, 40.0);   raw[30:50] = 95.0   # diving too deep
clamped = clamp_baseline(raw, anchor, margin_above_px=10, margin_below_px=15, image_height=H)
check("Deep region clamped to ≤ 55", np.all(clamped[30:50] <= 55.0))
check("Healthy region unchanged", np.allclose(clamped[:30], 40.0))

# ── T3: fit_spline_with_clamp ────────────────────────────────────────────────
print("\n=== T3: fit_spline_with_clamp ===")
retina_s  = np.full(W, 20.0); retina_s [25:55] = np.nan
choroid_s = np.full(W, 30.0); choroid_s[25:55] = np.nan
healthy = ~np.isnan(retina_s)
ret_spl = fit_spline_with_clamp(retina_s,  healthy, image_height=H, margin_below_px=10)
cho_spl = fit_spline_with_clamp(choroid_s, healthy, image_height=H, margin_below_px=15)
check("Retina spline not None", ret_spl is not None)
check("Choroid spline not None", cho_spl is not None)
check("Retina healthy region ≈ 20", np.allclose(ret_spl[:25], 20.0, atol=1.0))
check("Choroid healthy region ≈ 30", np.allclose(cho_spl[:25], 30.0, atol=1.0))
check("No wild extrapolation", np.all(cho_spl[25:55] <= 45.0))  # deepest+ margin_below

# ── T4: build_tumor_mask_bscan (new 3-argument version) ──────────────────────
print("\n=== T4: build_tumor_mask_bscan ===")
rsurf   = np.full(W, 10.0)    # actual elevated retina dome (top)
cbline  = np.full(W, 40.0)    # choroid baseline (bottom)
rbline  = np.full(W, 22.0)    # expected retina level (detection)

# elevation = 22-10 = 12 px  ≥  min_elevation_px=5 → should label rows 10-40
mask = build_tumor_mask_bscan((H, W), rsurf, cbline, rbline, min_elevation_px=5.0)
check("Rows 10-40 labeled", np.all(mask[10:41, :] == 3))
check("Vitreous (rows 0-9) = 0", np.all(mask[:10, :] == 0))
check("Deep (rows 41+) = 0",     np.all(mask[41:, :] == 0))

# Below threshold: elevation = 22-20 = 2 px < 5 → no label
mask_flat = build_tumor_mask_bscan((H, W), np.full(W, 20.0), cbline, rbline, min_elevation_px=5.0)
check("Small elevation → no label", np.all(mask_flat == 0))

# Healthy retina (no elevation): surface == baseline → no label
mask_healthy = build_tumor_mask_bscan((H, W), rbline.copy(), cbline, rbline, min_elevation_px=5.0)
check("Healthy retina (surface=baseline) → no label", np.all(mask_healthy == 0))

# NaN → skip column
rsurf_nan = rsurf.copy(); rsurf_nan[3] = np.nan
mask_nan = build_tumor_mask_bscan((H, W), rsurf_nan, cbline, rbline, min_elevation_px=5.0)
check("NaN column empty", np.all(mask_nan[:, 3] == 0))
check("Other cols labeled", np.all(mask_nan[10:41, [0,1,2,4]] == 3))

# ── T5: _find_healthy_columns_robust ─────────────────────────────────────────
print("\n=== T5: robust healthy-column detection ===")
ret_full  = np.full(W, 20.0);   ret_full[30:50]  = 5.0    # elevated by 15 px
cho_full  = np.full(W, 30.0);   cho_full[30:50]  = np.nan  # absent under tumor
hc = _find_healthy_columns_robust(ret_full, cho_full, sigma=1.5, n_iters=2)
check("Healthy cols exist", hc.sum() > 0)
check("Most elevated cols excluded", hc[30:50].mean() < 0.3,
      f"fraction included={hc[30:50].mean():.2f}")
check("Non-elevated cols mostly kept", hc[:30].mean() > 0.7,
      f"fraction kept={hc[:30].mean():.2f}")

# ── T6: volume calculation ────────────────────────────────────────────────────
print("\n=== T6: compute_tumor_volume_mm3 ===")
vol3d = np.zeros((5, H, W), dtype=np.uint8)
vol3d[1:4, 10:41, :] = 3   # 3 × 31 × 80 = 7440 voxels
vox = (0.01, 0.005, 0.005)
expected = 3 * 31 * 80 * vox[0] * vox[1] * vox[2]
got = compute_tumor_volume_mm3(vol3d, vox)
check("Volume correct", abs(got - expected) < 1e-9, f"{got:.8f} vs {expected:.8f}")

# ── T7: full pipeline — hybrid detection ─────────────────────────────────────
print("\n=== T7: full pipeline (hybrid, ROI-guided) ===")
D = 5
vol_img = np.zeros((D, H, W), dtype=np.float32)
vol_lbl = np.zeros((D, H, W), dtype=np.uint8)

# Healthy anatomy everywhere: retina 20-29, choroid 30-44
vol_lbl[:, 20:30, :] = 1
vol_lbl[:, 30:45, :] = 2

# Tumor in B-scans 1-3, cols 25-54:
#   retina elevated to rows 5-14 (dome), choroid absent
for b in [1, 2, 3]:
    vol_lbl[b, 20:30, 25:55] = 0   # remove normal retina
    vol_lbl[b, 30:45, 25:55] = 0   # remove choroid (absent under tumor)
    vol_lbl[b, 5:15,  25:55] = 1   # elevated dome

# en-face ROI: mark tumor columns in those B-scans with label 3
roi = np.zeros((D, W), dtype=np.uint8)
roi[1:4, 25:55] = 3   # non-zero → tumor

mask3d, vol_mm3 = compute_tumor_mask_volume(
    vol_img, vol_lbl,
    anterior_label_val=1, baseline_label_val=2,
    min_elevation_px=5.0,  # tumor elevation = 15 px >> 5 px threshold
    margin_below_px=25.0,
    tumor_label_val=3,
    enface_roi_mask=(roi > 0),  # any nonzero = tumor
    voxel_size_mm=(1.0, 1.0, 1.0),
)
check("Shape correct", mask3d.shape == (D, H, W))
check("Tumor voxels exist", np.count_nonzero(mask3d == 3) > 0)
check("Volume > 0", vol_mm3 > 0)
# KEY: healthy B-scans (0, 4) must NOT be labeled
check("Healthy B-scans not labeled", np.all(mask3d[[0, 4], :, :] == 0))
# KEY: healthy columns in tumor B-scans must NOT be labeled
check("Healthy cols in tumor B-scans not labeled",
      np.all(mask3d[[1,2,3], :, :25] == 0))
# Vitreous always clear
check("Vitreous clear", np.all(mask3d[:, :5, :] == 0))
# Tumor region labeled in the right B-scans
tumor_vox = np.count_nonzero(mask3d[[1,2,3], 5:45, 25:55] == 3)
check("Tumor region labeled", tumor_vox > 0, f"{tumor_vox} voxels")
print(f"  \u2192 Total tumor voxels: {np.count_nonzero(mask3d == 3)}, volume: {vol_mm3:.2f} mm\u00b3")

# ── T8: full pipeline — no ROI (robust auto) ─────────────────────────────────
print("\n=== T8: full pipeline (robust auto, no ROI) ===")
mask3d_auto, vol_auto = compute_tumor_mask_volume(
    vol_img, vol_lbl,
    anterior_label_val=1, baseline_label_val=2,
    min_elevation_px=5.0,
    margin_below_px=25.0,
    robust_sigma=1.5, robust_iters=2,
    tumor_label_val=3,
    voxel_size_mm=(1.0, 1.0, 1.0),
)
check("Shape correct", mask3d_auto.shape == (D, H, W))
check("Output is ndarray", isinstance(mask3d_auto, np.ndarray))
# In auto mode just verify it doesn't crash and healthy B-scans are clear
check("Healthy B-scans not labeled", np.all(mask3d_auto[[0, 4], :, :] == 0))
print(f"  \u2192 Total tumor voxels: {np.count_nonzero(mask3d_auto == 3)}, volume: {vol_auto:.2f} mm\u00b3")

print("\n\u2705  All tests passed.\n")
