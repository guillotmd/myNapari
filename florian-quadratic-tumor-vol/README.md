# florian-quadratic-tumor-vol

> **Napari plugin for OCT retinal tumor volumetry using quadratic (k=2) spline baseline fitting.**

---

## Purpose

Detects and measures the volume of intraocular tumors in 3D panretinal OCT. Unlike the linear plugin, it fits a **quadratic (parabolic) spline** to both the retinal and choroidal surfaces, naturally modelling the spherical geometry of the eye globe.

### When to Use This Plugin (vs Linear)

| Situation | Plugin |
|---|---|
| Dome-shaped tumors that follow globe curvature | ✅ **Quadratic** |
| Adult eyes, wide field of view (>90°) | ✅ **Quadratic** |
| Flat plateau-like elevations | Linear |
| Pediatric eyes with short axial length | Linear |

---

## How It Works

1. **Layer boundary extraction** — reads per-column retinal and choroidal depths from the segmentation.
2. **Quadratic spline fitting** — fits `z(x) = ax² + bx + c` to columns outside the tumor region. The parabola naturally matches the bowl curvature of the posterior retina.
3. **Tumor mask generation** — voxels more than `min_elevation_px` above the fitted baseline, and below the actual retinal surface, are labelled as tumor.
4. **Volume calculation** — `N_voxels × vox_z × vox_y × vox_x` in mm³.

---

## Input Requirements

| Layer | Type | Description |
|---|---|---|
| **OCT Volume** | Image | `(n_slices, depth_px, width_px)` |
| **Segmentation** | Labels | Same shape; `1`=retina, `2`=choroid |
| **En-face ROI** *(optional)* | Labels | 2D mask `(n_slices, width_px)` marking the tumor column |

---

## UI Reference

### Input Layers

| Control | What it does |
|---|---|
| **OCT Volume** | The 3D OCT intensity volume (Image layer). |
| **Segmentation** | Tissue segmentation with retina and choroid labels. |
| **En-face ROI** | Optional. Restricts baseline fitting to columns outside the drawn tumor area. Highly recommended for irregular tumors. |

### Detection Parameters

| Control | Default | Effect of changing |
|---|---|---|
| **Retina label** | `1` | Change if segmentation uses a different label scheme. |
| **Choroid label** | `2` | Defines the depth floor of the tumor mask. |
| **Output tumor label** | `3` | Label ID written into the output mask. Use this same value in the 3D render plugin. |
| **Spline smoothing** | `0` (Auto) | `0` = interpolating spline. Increase to `10–500` for noisy data to prevent overfitting to segmentation noise. |
| **Min. elevation (px)** | `5 px` | Increase → fewer false positives but may miss shallow tumors. Decrease → more sensitive. |
| **Edge margin (cols)** | `15` | Wider margin → safer against edge artefacts but clips peripheral columns. |
| **Max depth margin (px)** | `60` | Reduce to `30–40` if baseline extrapolates too deep past the scan boundary. |
| **Min layer thickness (px)** | `5` | Rejects segmentation fragments thinner than this (vitreous seeds, noise). |
| **Ignore top margin (px)** | `0` | Ignore labels in the top N rows — use if floating vitreous artefacts appear. |

### Output Options

| Control | Default | What it does |
|---|---|---|
| **Diagnostic lines** | ☐ | Draws the fitted baselines (label 4 = retina, label 5 = choroid) on the output mask. **Turn on to debug baseline fit quality.** |
| **Diagnostic line thickness** | `5 px` | Pixel width of drawn diagnostic lines. |

### Physical Voxel Size

| Control | Default | What it does |
|---|---|---|
| **Read from layer .scale** | ✅ | Reads `(z, y, x)` voxel spacing in mm from the Image layer `.scale` attribute. |
| **Voxel Z / depth** | `1.0 mm` | Axial pixel size. Only editable when auto-read is off. |
| **Voxel Y / lateral** | `1.0 mm` | Lateral pixel size within a B-scan. |
| **Voxel X / width** | `1.0 mm` | Inter-slice spacing. |

> Values must be in **mm** (e.g. 3.87 µm = `0.00387`).

### Advanced Options *(collapsed — expand only if needed)*

| Control | Default | When to change |
|---|---|---|
| **Robust σ** | `2.0` | Sigma-clipping threshold for no-ROI mode. Reduce for aggressive outlier rejection; increase for very noisy data. |
| **Robust iters** | `2` | Rejection passes. `0` disables iterative rejection entirely. |
| **Morphological cleanup** | ☐ | Removes isolated noise voxels and fills small holes (opening → closing → largest component). |
| **Weighted baseline fitting** | ☐ | Upweights columns far from the tumor edge to reduce baseline edge-bleed. |

### Action Buttons

| Button | Action |
|---|---|
| **▶ Run Tumor Detection** | Runs the full pipeline in a background thread. Progress bar pulses until complete. |
| **🔢 Recalculate Volume from Mask** | Instantly re-counts voxels in the existing mask — use after manual edits in paint mode. |

---

## Output

`{vol_name}_Tumor_Mask_Quadratic` — Labels layer, tumor voxels = output tumor label.

Result card shows:
```
Volume: 2.3456 mm³
Voxel: 0.0039 × 0.0039 × 0.0115 mm
```
