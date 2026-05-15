# florian-tumor-3d-render

> **Napari plugin for 3D surface rendering of tumor masks produced by the florian detection plugins.**

---

## Purpose

Takes any tumor mask Labels layer (produced by `florian-linear-tumor-vol` or `florian-quadratic-tumor-vol`) and generates a **3D triangulated surface mesh** using marching cubes, optionally smoothed with Laplacian filtering. The result is added as a napari Surface layer, the viewer switches to 3D mode, and the tumor volume is reported in mm³.

This plugin is deliberately **post-processing only** — it does not alter or re-run detection. It reads the mask as-is, giving you a clean 3D visualisation and a fast way to recompute volume after manual edits.

---

## How It Works

### 1. Volume Calculation (fast, pre-mesh)
Before running any mesh algorithm, the volume is computed instantly by counting voxels:

```
N_voxels  = np.sum(mask == tumor_label_id)
Volume    = N_voxels × axial_res(mm) × lateral_res(mm) × inter_slice_spacing(mm)
```

### 2. Marching Cubes
The `scikit-image` marching cubes algorithm extracts an isosurface from the binary tumor mask at `level=0.5`. The output is a triangulated mesh: vertices `(V, 3)` and faces `(F, 3)`.

Marching cubes is run in **voxel coordinates** (uniform `spacing=(1,1,1)`). Physical scaling is applied via the napari Surface layer's `.scale` attribute, so the mesh is rendered at the correct aspect ratio without inflating memory.

### 3. Laplacian Mesh Smoothing (optional)
If `smooth_iters > 0`, the mesh is passed to `trimesh.smoothing.filter_laplacian()`. Laplacian smoothing moves each vertex toward the average of its neighbours, rounding off the staircase artefacts from the voxel grid. It preserves overall shape well at 10–20 iterations.

### 4. Scale Normalisation (prevents Z-axis squish)
To prevent the 3D render from appearing squished when axial, lateral, and inter-slice spacings differ significantly, a normalised scale tuple is computed:

```python
max_res = max(axial_um, lateral_um, spacing_um)
scale = (spacing_um/max_res, axial_um/max_res, lateral_um/max_res)
```

For typical PanOCT data (axial ≈ 3.87 µm, lateral ≈ 11.5 µm, gap ≈ 120 µm), the Z axis (inter-slice) is the largest, so the mesh is scaled to `(1.0, 0.032, 0.096)` — the axial and lateral axes are compressed relative to the inter-slice axis, giving a proportional render.

---

## Input Requirements

| Layer | Type | Description |
|---|---|---|
| **Tumor mask** | Labels | Output from a detection plugin. Must have a distinct non-zero label for the tumor. |
| **OCT Volume** *(optional)* | Image | Shown as a low-opacity context volume alongside the surface mesh. |

---

## UI Reference

### Input

| Control | What it does |
|---|---|
| **Tumor mask (Labels)** | Selects the detection output. The **tumor label ID** is auto-detected when you change this combo — the largest non-zero unique value in the mask is used, since detection plugins write retina=1, choroid=2, tumor=3 or 4. |
| **Tumor label ID** | Label value that identifies tumor voxels. Shows an `auto-detected: N` badge when detected automatically. Override manually if your mask uses a non-standard scheme. |
| **OCT volume (Image, opt.)** | Optional reference volume. Added as `{vol_name}_3D` with `visible=False` — toggle it on in the layer list to see the volume alongside the mesh. |

---

### Physical Resolution (µm / px)

These values are used for **two purposes**: calculating the volume in mm³ and setting the mesh scale for correct 3D proportions.

| Control | Default | What it does |
|---|---|---|
| **Axial (µm/px)** | `3.87 µm` | Physical size of one depth pixel. For 400 kHz with 12 mm range in 2048 pixels: `12 mm / 2048 / 1.336 ≈ 3.87 µm`. |
| **Lat (µm/px)** | `11.5 µm` | Physical size of one horizontal pixel within a B-scan. |
| **Gap (µm)** | `120 µm` | Physical distance between consecutive B-scan planes. This has the **largest impact on rendered proportions** — verify with your acquisition protocol. |

> These values must match what you used in the detection plugin for volume consistency. If you ran curve correction first, the lateral resolution changes (output pixel size ≈ arc-length at centre depth) — use the corrected dataset's spacing here.

---

### Mesh Options

| Control | Default | Effect of changing |
|---|---|---|
| **Laplacian smoothing** | `10` iterations | **0** = raw marching cubes surface (blocky, voxelated). **10–20** = smooth, natural-looking surface (recommended). **>30** = over-smoothed, may shrink small features. |

---

### Action Buttons

#### ▶ Generate 3D Render *(Primary)*
Runs marching cubes and optional smoothing in a background thread. The progress bar pulses during computation. On completion:
- The 2D tumor mask is hidden (set `visible=False`) so it doesn't appear as a flat plane in 3D.
- The optional reference volume is added with `visible=False`.
- The Surface layer `{mask_name}_3D_Surface` is added with the `turbo` colormap.
- The viewer switches to 3D display mode (`ndisplay=3`) and resets the camera.
- The result card shows the tumor volume.

#### 🔢 Recalculate Volume from Mask *(Secondary)*
Counts voxels in the current mask instantly (no mesh generation). Use after:
- Manual edits in napari paint mode.
- Changing the physical resolution values.
- Selecting a different mask layer.

Does **not** regenerate or update the 3D surface — only the volume number changes.

---

## Output

| Layer added | Description |
|---|---|
| `{mask_name}_3D_Surface` | Surface layer, `turbo` colormap, physically scaled |
| `{vol_name}_3D` *(if volume selected)* | Image layer, hidden by default, same scale as surface |

The result card shows:
```
Volume: 1.2345 mm³
```
After recalculate:
```
Volume: 1.2345 mm³  (recalculated)
```

---

## Tips

**Mesh looks squished/stretched** — Check that Axial / Lat / Gap match your acquisition. The inter-slice gap is the most common source of proportion errors.

**"No surface found — wrong label ID?"** — The tumor label ID doesn't match any voxels in the selected mask. Use the auto-detect or inspect `np.unique(mask.data)` in the console.

**Mesh is very rough/spiky** — Increase Laplacian smoothing iterations from 10 to 20–30.

**Mesh has holes** — Enable morphological cleanup in the detection plugin before re-detecting, then re-render.

**Want to export the mesh** — Right-click the Surface layer in napari and use "Save selected layer(s)". napari will export as `.obj` or `.stl` depending on the installed writer plugins.
