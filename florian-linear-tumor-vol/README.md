# florian-linear-tumor-vol

> **Napari plugin for OCT retinoblastoma tumor volumetry using linear baseline extrapolation.**

---

## Purpose

This plugin detects and measures the volume of intraocular tumors (typically retinoblastoma or choroidal tumors) that cause the retina to bulge inward. It works on 3D panretinal OCT volumes, using a **linear interpolation strategy** to reconstruct what the healthy retinal surface *would* look like without the tumor, then measuring how much tissue lies above that reconstructed baseline.

### When to Use This Plugin (vs Quadratic)

| Situation | Recommended Plugin |
|---|---|
| Tumors that cause a relatively flat, plateau-like elevation | ✅ **Linear** |
| Data with a wide field of view where the healthy retina is approximately flat | ✅ **Linear** |
| Pediatric eyes (shorter axial length, less pronounced curvature) | ✅ **Linear** |
| Large dome-shaped tumors that follow the natural globe curvature | Quadratic |
| Adult eyes scanned over a very wide field (>90°) | Quadratic |

---

## How It Works

### 1. Baseline Detection

For each B-scan slice, the plugin reads the segmentation labels to find:
- The **retinal surface** (inner limiting membrane, label = retina)
- The **choroid / RPE surface** (label = choroid)

In columns that are **outside the tumor region** (either marked explicitly in the en-face ROI, or estimated via iterative outlier rejection), the retinal depth values are collected as "healthy" reference points.

### 2. Baseline Reconstruction (Linear)

The healthy reference points are connected with **linear interpolation** (straight line segments, or optionally pchip monotone spline) across the tumor gap. This reconstructed curve represents the predicted position of the retinal surface if no tumor were present.

```
    ╭─ actual retina (elevated by tumor)
    │
────┘  reconstructed linear baseline  ──────
```

The **choroid baseline** is reconstructed analogously — the choroid curve defines the *floor* of the tumor mask (so the mask captures the full tumor thickness, not just the retinal bulge).

### 3. Tumor Mask Generation

Voxels that lie **above the choroid baseline** *and* **below the actual retina** *and* whose retinal position is more than `elevation_threshold` pixels above the linear baseline are labelled as tumor.

### 4. Volume Calculation

```
Volume (mm³) = N_tumor_voxels × axial_res(mm) × lateral_res(mm) × inter_slice_spacing(mm)
```

---

## Input Requirements

| Layer | Type | Description |
|---|---|---|
| **B-Scan** | Image | 3D stack: `(n_slices, depth_px, width_px)` |
| **Segmentation** | Labels | Same shape as B-Scan; values: `1`=retina, `2`=choroid |
| **En-face ROI** *(optional)* | Labels | 2D en-face mask `(n_slices, width_px)`; non-zero pixels mark the tumor column |

---

## UI Reference

### Input Layers

| Control | What it does |
|---|---|
| **B-Scan (Image)** | Selects the 3D OCT intensity volume. Must be an Image layer. |
| **Segmentation (Labels)** | Selects the layer containing retina and choroid labels. |
| **En-face ROI (opt.)** | Optionally constrains baseline fitting to columns outside the tumor. Without this, the plugin uses iterative outlier rejection (slower, less precise). |

---

### Label ID Configuration

| Control | Default | What it does |
|---|---|---|
| **Retina ID** | `1` | The label value in the segmentation that marks the retinal surface. Change if your segmentation uses a different scheme. |
| **Choroid ID** | `2` | The label value that marks the choroid / RPE. Defines the depth floor of the tumor mask. |
| **En-face Tumor** | `3` | The label value used in the en-face ROI layer to mark the tumor column. Used to separate tumor from non-tumor columns when fitting the baseline. |
| **Output Mask ID** | `4` | The label value written into the output tumor mask layer. Use the 3D render plugin with this same value. |

---

### Physical Resolution

These values convert voxel counts into physical millimetres for volume calculation.

| Control | Default | What it does |
|---|---|---|
| **Axial (µm/px)** | `3.87 µm` | Physical size of one depth pixel. For a 400 kHz system with 12 mm range over 2048 pixels ≈ 3.87 µm/px in tissue (after dividing by n=1.336). |
| **Lateral (µm/px)** | `11.5 µm` | Size of one horizontal pixel within a single B-scan. |
| **Slice gap (µm)** | `120 µm` | Distance between consecutive B-scan slices. Set this to your actual inter-slice spacing; it has the largest impact on volume accuracy. |

---

### Detection Parameters

| Control | Default | Effect of increasing | Effect of decreasing |
|---|---|---|---|
| **Elevation threshold (px)** | `5 px` | Fewer false positives; may miss thin tumors | More sensitive; more false positives at flat areas |
| **Edge margin (cols)** | `15 cols` | Wider safety band at scan edge; may clip small peripheral tumors | Narrower safety band; risk of edge artefacts |
| **Prior sigma (px)** | `10 px` | Softer ROI prior; detection spreads further from the drawn region | Tighter prior; detection constrained close to the drawn region |
| **Mapping mode** | `linear` | — | — |
| **Min layer thickness (px)** | `5 px` | Fewer thin artefact fragments; may miss genuinely thin tumors | More sensitive to thin fragments; more noise voxels |
| **Ignore top margin (px)** | `0 px` | More rows ignored at scan top; use if vitreous shows floating artefacts | — |

**Mapping mode** selects how the 2D en-face ROI is mapped onto the 3D B-scan stack:
- `linear` — assumes uniform inter-slice spacing (default, suitable for most acquisitions).
- `custom_affine` — applies a custom affine transform (for resampled or non-uniform datasets).

---

### Output Options

| Control | Default | What it does |
|---|---|---|
| **Show tumor mask layer** | ✅ Checked | Adds the output Labels layer to the viewer after detection. Uncheck to suppress the layer and only report volume. |
| **Diagnostic lines** | ☐ Unchecked | Draws the reconstructed retinal baseline (label 4) and choroid baseline (label 5) directly on the output mask. **Turn on when debugging baseline fitting accuracy.** |
| **Diagnostic line thickness** | `5 px` | How many pixels wide the diagnostic lines are drawn. Increase for higher-resolution data. |

---

### Advanced Options *(collapsed by default)*

| Control | Default | When to change |
|---|---|---|
| **Filter choroid outliers** | ☐ | Enable if your ONNX segmentation mislabels bright choroid reflections above the retina. Rejects anatomically impossible choroid positions. |
| **Morphological cleanup** | ☐ | Enable if the output mask contains isolated voxel clusters disconnected from the main tumor. Applies opening → closing → keep largest component. |
| **Interpolation** | `linear` | Switch to `pchip` for a smoother baseline curve that avoids abrupt slope changes at the tumor boundary. |
| **Weighted baseline fitting** | ☐ | Upweights columns far from the tumor edge. Reduces edge-bleed — baseline pulled too strongly toward the tumor boundary. |
| **Smoothing σ (px)** | `20 px` | Gaussian smoothing applied to the baseline curve. Increase if the fitted baseline is erratic (noisy segmentation); decrease if smoothing blurs a sharp tumor boundary. |
| **Estimate volume uncertainty** | ☐ | Runs detection three times (nominal, +1 px, −1 px threshold) to report a ± volume range. Doubles run time. Useful for publication-quality measurements. |

---

### Action Buttons

#### ▶ Calculate Tumor Volume *(Primary)*
Runs the full detection pipeline in a background thread. The **5 px gradient progress bar** below the button pulses while running. On completion, the tumor mask Labels layer is added to the viewer and the result card shows the volume.

#### 🔢 Recalculate Volume from Mask *(Secondary)*
Instantly recomputes the volume from the **existing** tumor mask layer without re-running detection. Use this after manually editing the mask in the napari paint mode. The result card updates to show the new value.

---

## Output

| Layer added | Description |
|---|---|
| `{scan_name}_Tumor_Mask_Linear` | Labels layer — label `output_tumor_label` = tumor voxels |

The result card below the buttons shows:
```
Volume: 1.2345 mm³
```
or with uncertainty:
```
Volume: 1.2345 ± 0.0432 mm³
```
