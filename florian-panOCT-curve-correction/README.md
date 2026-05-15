# florian-panOCT-curve-correction

> **Napari plugin that corrects fan-beam geometric distortion in PanOCT data.**  
> Based on: *Bayhaqi et al., "Quantitative measurement using panretinal optical coherence tomography with three-dimensional curve distortion correction," Biomedical Optics Express, Vol. 16, No. 9, Sep 2025. Casey Eye Institute, OHSU.*

---

## Purpose

PanOCT scanners acquire A-scans in a **fan-beam pattern** from a virtual pivot point, but store them in a rectangular image array. This creates a geometric distortion: pixels at the scan periphery are physically further apart than pixels at the scan centre, and the entire image is slightly "curved" compared to the true anatomy.

This plugin corrects that distortion by transforming each B-scan from **polar coordinates (R, θ)** to true **Cartesian coordinates (x, z)** using calibrated scan angles. The result is a geometrically accurate volume where physical distances are uniform in all directions.

### When to Use This Plugin

- **Always run before tumor detection** if you want geometrically accurate volume measurements.
- You can skip it for qualitative inspection, but volumes computed on uncorrected data will be systematically over- or under-estimated depending on tumor location.
- Run it **once per acquisition session** — the parameters are system-level constants, not per-patient.

---

## How It Works

### The Math (Bayhaqi et al., Eq. 1–3)

The plugin maps coordinates using the true 3D spherical geometry of the scanner:

1. **2D Polar (Eq. 1):** `x = R·sin(θ)`, `z = R·cos(θ)`
2. **3D Spherical (Eq. 2):** `φ = √(θ_fast² + θ_slow²)`, `θ_azimuth = arctan(θ_fast/θ_slow)`
3. **Cartesian (Eq. 3):** `x = R·sin(φ)·cos(θ_azimuth)`, `y = R·sin(φ)·sin(θ_azimuth)`, `z = R·cos(φ)`

Where `φ` is the total deflection angle from the optical axis. Because `φ` couples both the fast and slow axis scan angles, pixels at the far corners of the volume are deeper and further laterally displaced than pixels along the centre cross. 

> **Note on Scan Angles (Eq. 12):** The paper specifies that scan angle conversion best follows a 3rd-order polynomial to account for optical misalignment. Because the exact numerical coefficients for that 3rd-order polynomial were not published for the PanOCT system, this plugin implements a standard linear interpolation between the calibrated max bounds. This is the most mathematically accurate approach without having the exact empirical lab coefficients.

### Correction Modes

The plugin offers two modes to handle this 3D mapping:

1. **3D-Coupled Slicewise (Default)**
   - Computes the true `φ` for each B-scan slice based on its slow-axis position, and solves for the 3D-coupled depth (`z`) and lateral (`x`) coordinates.
   - Maps each B-scan into a shared 3D output grid independently.
   - **Advantage:** Low memory usage (~1 GB), fast.
   - **Limitation:** No inter-slice interpolation along the slow axis (y-axis).

2. **Full 3D Interpolation (Checkbox)**
   - Performs a true 3D inverse mapping from the Cartesian output grid back to the spherical source volume.
   - Interpolates across adjacent B-scans to create a perfectly smooth 3D output.
   - **Advantage:** Maximum geometric fidelity; smooths slow-axis artifacts.
   - **Limitation:** High memory usage (~13–15 GB for an 800×2048×800 volume); requires processing in Z-chunks.

### Memory Impact

| Mode | Input | Output | Peak RAM | Notes |
|---|---|---|---|---|
| 3D-Coupled (Default) | `(800, 2048, 800)` | `(800, ~437, ~856)` | **~1.5 GB** | Fast, per-slice. |
| Full 3D Interpolation | `(800, 2048, 800)` | `(800, ~437, ~856)` | **~15.0 GB** | Smooth inter-slice interpolation. |

*(Note: Output Z-dimension is smaller because the uniform pixels are ~44 µm, larger than the 4.38 µm raw axial spacing).*

---

## Default Calibration Values (400 kHz PanOCT, Casey Eye Institute)

| Parameter | Default | Source |
|---|---|---|
| Fast axis max angle | `49.85°` | Physical calibration |
| Fast axis offset | `−0.61°` | Optical misalignment |
| Slow axis max angle | `49.96°` | Physical calibration |
| Slow axis offset | `+0.27°` | Optical misalignment |
| Pivot distance | `16.0 mm` | Adult eye geometry |
| Refractive index | `1.336` | Ocular media average |
| Axial range | `12.0 mm` | System specification |

> If your system has different calibration constants, update the fields accordingly and re-run the preview to verify visually.

---

## Input Requirements

Select any combination of:
- **Image layers** — OCT intensity volumes `(D, H, W)`
- **Labels layers** — segmentation masks `(D, H, W)` in the same geometry as the Image

All selected layers **must have the same shape**. Corrected copies are added with `_CurveCorr` suffix and identical dtype.

---

## UI Reference

### Layers to Correct

| Control | What it does |
|---|---|
| **Layer list** | Multi-select list of all Image and Labels layers in the viewer. Hold **Ctrl / ⌘** to select multiple layers. Correct your B-scan volume and its segmentation together in one run. |

---

### Quick Preview (single slice)

Use this section to **tune calibration parameters without processing the full volume**. A single slice corrects in under a second.

| Control | What it does |
|---|---|
| **Image layer** | Selects which Image layer to sample a preview slice from. |
| **Slice index** | Which B-scan (0-indexed along axis 0) to preview. |
| **👁 Preview Slice** | Corrects the single selected slice and adds it as `{layer_name}_preview_s{N}_CurveCorr`. Re-run after changing parameters to compare. **Does not affect the full run.** |

**Workflow:** Adjust parameters → Preview → compare raw vs corrected in the viewer → repeat until the retina appears straight and uniformly spaced → run the full volume.

---

### PanOCT Calibration Parameters

| Control | Default | What it does |
|---|---|---|
| **Pivot distance (mm)** | `16.0 mm` | Distance from the virtual scanner pivot to the top of the image. Approximately the lens-to-retina distance. Adult ≈ 16 mm; infant/child ≈ 10–13 mm. **Increase** for larger eyes; **decrease** for smaller eyes. |
| **Fast axis Max°** | `49.85°` | Half-angle of the fast (horizontal) scan axis. Set from your system calibration report. |
| **Fast axis Offset°** | `−0.61°` | Angular offset of the fast axis optical centre from mechanical zero. |
| **Slow axis Max°** | `49.96°` | Half-angle of the slow (depth / slice) scan axis. |
| **Slow axis Offset°** | `+0.27°` | Angular offset of the slow axis. |
| **Refractive index** | `1.336` | Average refractive index of ocular media. Converts the axial range from air to optical path length in tissue. Standard value: 1.336. |
| **Axial range (mm)** | `12.0 mm` | Total imaging depth in air. Set from your system specification sheet. |

> **Do not guess calibration values.** They come from a calibration phantom measurement on your specific system. Using wrong values produces a distorted output that looks different from the input but is not geometrically accurate.

---

### Action Buttons

| Button | Action |
|---|---|
| **▶ Apply Curve Correction** | Corrects all selected layers in a background thread. The progress bar shows per-slice progress as a percentage. Corrected layers are added to the viewer with `_CurveCorr` suffix. |
| **👁 Preview Slice** | Corrects a single B-scan slice instantly (synchronous, on UI thread). Result is a 2D Image layer. |

### Status / Result Card

After a successful full run:
```
✓ 2 layers corrected — '_CurveCorr' suffix added.
```
After a preview:
```
✓ Preview: slice 400 corrected → 'MyScan_preview_s400_CurveCorr'
```

---

## Output

For each selected layer, a new layer is added:

| Input layer | Output layer |
|---|---|
| `MyScan` (Image) | `MyScan_CurveCorr` (Image) |
| `MyScan_seg` (Labels) | `MyScan_seg_CurveCorr` (Labels) |
| Preview | `MyScan_preview_s{N}_CurveCorr` (Image) |

Pass the `_CurveCorr` layers into the detection plugins for geometrically accurate volumes.
