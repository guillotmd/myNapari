"""
_tumor_widget.py — Napari widget for training-free OCT tumor volumetric measurement.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
from napari.layers import Image, Labels
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
    QLabel, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)
from florian_tumor_vol._tumor_funcs import compute_tumor_mask_volume, compute_tumor_volume_mm3


class TumorVolumeWidget(QWidget):
    """Napari widget for training-free OCT tumor volumetric measurement."""

    def __init__(self, napari_viewer, parent=None):
        super().__init__(parent)
        self.viewer = napari_viewer
        self._tumor_layer: Optional[Labels] = None
        self._build_ui()
        self._connect_signals()

    @staticmethod
    def _add_row(form, label_text, widget, tooltip):
        lbl = QLabel(label_text)
        lbl.setToolTip(tooltip)
        widget.setToolTip(tooltip)
        form.addRow(lbl, widget)

    def _build_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setAlignment(Qt.AlignTop)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setAlignment(Qt.AlignTop)
        scroll.setWidget(inner)
        root_layout.addWidget(scroll)

        # ── Layer selection ───────────────────────────────────────────────
        layer_group = QGroupBox("Input Layers")
        layer_form = QFormLayout(layer_group)

        self._vol_combo = QComboBox()
        self._add_row(layer_form, "OCT Volume (Image):", self._vol_combo,
            "The 3-D OCT intensity volume (Image layer).")

        self._seg_combo = QComboBox()
        self._add_row(layer_form, "Segmentation (Labels):", self._seg_combo,
            "B-scan segmentation Labels layer.\n"
            "Expected labels:  0 = vitreous,  1 = retina,  2 = choroid.")

        self._roi_combo = QComboBox()
        self._add_row(layer_form, "En-face ROI (Labels, opt.):", self._roi_combo,
            "Optional: Labels layer drawn on the en-face projection.\n"
            "Non-zero values mark tumor area, excluded from baseline fitting.")

        layout.addWidget(layer_group)

        # ── Algorithm parameters ──────────────────────────────────────────
        algo_group = QGroupBox("Detection Parameters")
        algo_form = QFormLayout(algo_group)

        self._anterior_label = QSpinBox()
        self._anterior_label.setRange(0, 255); self._anterior_label.setValue(1)
        self._add_row(algo_form, "Retina label value:", self._anterior_label,
            "Label value for the retina layer. Default: 1.")

        self._baseline_label = QSpinBox()
        self._baseline_label.setRange(0, 255); self._baseline_label.setValue(2)
        self._add_row(algo_form, "Choroid baseline label:", self._baseline_label,
            "Label value for the choroid layer. Default: 2.")

        self._tumor_label = QSpinBox()
        self._tumor_label.setRange(1, 255); self._tumor_label.setValue(3)
        self._add_row(algo_form, "Tumor output label:", self._tumor_label,
            "Label value written into the output mask. Default: 3.")

        self._spline_smoothing = QDoubleSpinBox()
        self._spline_smoothing.setRange(0.0, 1e6); self._spline_smoothing.setDecimals(1)
        self._spline_smoothing.setValue(0.0)
        self._spline_smoothing.setSpecialValueText("Auto (scipy default)")
        self._add_row(algo_form, "Spline smoothing:", self._spline_smoothing,
            "Smoothing factor for the quadratic (k=2) spline.\n"
            "0 / 'Auto' = interpolating. Increase if surfaces are noisy.")

        self._min_elevation = QDoubleSpinBox()
        self._min_elevation.setRange(0.0, 500.0); self._min_elevation.setDecimals(1)
        self._min_elevation.setValue(5.0)
        self._add_row(algo_form, "Min. elevation (px):", self._min_elevation,
            "Minimum retinal elevation above baseline to count as tumor.")

        self._edge_margin = QSpinBox()
        self._edge_margin.setRange(0, 200); self._edge_margin.setValue(15)
        self._add_row(algo_form, "Edge margin (cols):", self._edge_margin,
            "Columns to blank inward from the actual data boundary.\n"
            "Prevents false positives at B-scan edges.")

        self._margin_below = QDoubleSpinBox()
        self._margin_below.setRange(0.0, 500.0); self._margin_below.setDecimals(0)
        self._margin_below.setValue(60.0)
        self._add_row(algo_form, "Max depth margin (px):", self._margin_below,
            "Max pixels below deepest observed choroid for extrapolation.\n"
            "Reduce (e.g. 30-40) if baseline dips too far down.")

        self._min_layer_thickness = QSpinBox()
        self._min_layer_thickness.setRange(1, 100); self._min_layer_thickness.setValue(5)
        self._add_row(algo_form, "Min layer thickness (px):", self._min_layer_thickness,
            "Ignore isolated tissue fragments (e.g. vitreous seeds)\n"
            "thinner than this value during edge extraction.")

        self._ignore_top_px = QSpinBox()
        self._ignore_top_px.setRange(0, 1000); self._ignore_top_px.setValue(0)
        self._add_row(algo_form, "Ignore top margin (px):", self._ignore_top_px,
            "Ignore segmentation labels in the top N pixels.\n"
            "Useful for filtering floating artifacts near the vitreous.")

        self._robust_sigma = QDoubleSpinBox()
        self._robust_sigma.setRange(0.1, 10.0); self._robust_sigma.setDecimals(1)
        self._robust_sigma.setValue(2.0)
        self._add_row(algo_form, "Robust sigma (no ROI):", self._robust_sigma,
            "Outlier rejection threshold (no-ROI mode only).")

        self._robust_iters = QSpinBox()
        self._robust_iters.setRange(0, 5); self._robust_iters.setValue(2)
        self._add_row(algo_form, "Robust iters (no ROI):", self._robust_iters,
            "Iterative rejection passes (no-ROI mode only).")

        layout.addWidget(algo_group)

        # ── Output options ────────────────────────────────────────────────
        opt_group = QGroupBox("Output Options")
        opt_form = QFormLayout(opt_group)

        self._show_diagnostic_lines = QCheckBox("Show diagnostic lines (labels 4 & 5)")
        self._show_diagnostic_lines.setChecked(False)
        self._show_diagnostic_lines.setToolTip(
            "Draw the fitted retinal baseline (label 4) and choroid\n"
            "baseline (label 5) on the output mask for visual verification.")
        opt_form.addRow(self._show_diagnostic_lines)

        self._generate_3d = QCheckBox("Generate 3D surface mesh")
        self._generate_3d.setChecked(False)
        self._generate_3d.setToolTip(
            "Generate a 3D Surface mesh of the tumor using marching cubes.")
        opt_form.addRow(self._generate_3d)

        self._mesh_smoothing = QSpinBox()
        self._mesh_smoothing.setRange(0, 50); self._mesh_smoothing.setValue(10)
        self._add_row(opt_form, "Mesh smoothing iters:", self._mesh_smoothing,
            "Laplacian smoothing iterations for 3D mesh (if generated).")

        layout.addWidget(opt_group)

        # ── Voxel / physical dimensions ───────────────────────────────────
        vox_group = QGroupBox("Physical Voxel Size (mm)")
        vox_form = QFormLayout(vox_group)

        self._use_layer_scale = QCheckBox("Use layer scale attribute")
        self._use_layer_scale.setChecked(True)
        self._use_layer_scale.setToolTip(
            "Read voxel size from the Image layer's .scale property.\n"
            "Uncheck to enter dimensions manually.")
        vox_form.addRow(self._use_layer_scale)

        self._vox_z = QDoubleSpinBox()
        self._vox_z.setRange(0.0001, 100.0); self._vox_z.setDecimals(4); self._vox_z.setValue(1.0)
        self._vox_z_lbl = QLabel("Voxel Z / depth (mm):")
        vox_form.addRow(self._vox_z_lbl, self._vox_z)

        self._vox_y = QDoubleSpinBox()
        self._vox_y.setRange(0.0001, 100.0); self._vox_y.setDecimals(4); self._vox_y.setValue(1.0)
        self._vox_y_lbl = QLabel("Voxel Y / height (mm):")
        vox_form.addRow(self._vox_y_lbl, self._vox_y)

        self._vox_x = QDoubleSpinBox()
        self._vox_x.setRange(0.0001, 100.0); self._vox_x.setDecimals(4); self._vox_x.setValue(1.0)
        self._vox_x_lbl = QLabel("Voxel X / width (mm):")
        vox_form.addRow(self._vox_x_lbl, self._vox_x)

        layout.addWidget(vox_group)

        # ── Action buttons ────────────────────────────────────────────────
        self._run_btn = QPushButton("▶  Run Tumor Detection")
        self._run_btn.setFixedHeight(36)
        self._run_btn.setToolTip(
            "Run the full tumor detection pipeline.\n\n"
            "Uses quadratic (k=2) spline fitting for both the retinal\n"
            "and choroid baselines — matching the natural parabolic\n"
            "curvature of the eye globe.")
        layout.addWidget(self._run_btn)

        self._recalc_btn = QPushButton("🔢  Recalculate Volume (from current mask)")
        self._recalc_btn.setToolTip(
            "Recalculate volume from the existing mask without re-running detection.")
        layout.addWidget(self._recalc_btn)

        self._result_label = QLabel("Volume: —")
        self._result_label.setAlignment(Qt.AlignCenter)
        self._result_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self._result_label)

        self._refresh_layer_combos()
        self._on_use_layer_scale_toggled(self._use_layer_scale.isChecked())

    # ──────────────────────────────────────────────────────────────────────
    # Signals
    # ──────────────────────────────────────────────────────────────────────

    def _connect_signals(self):
        self._run_btn.clicked.connect(self._on_run)
        self._recalc_btn.clicked.connect(self._on_recalculate)
        self._use_layer_scale.toggled.connect(self._on_use_layer_scale_toggled)
        self.viewer.layers.events.inserted.connect(self._refresh_layer_combos)
        self.viewer.layers.events.removed.connect(self._refresh_layer_combos)

    def _refresh_layer_combos(self, *_):
        image_names = [l.name for l in self.viewer.layers if isinstance(l, Image)]
        label_names = [l.name for l in self.viewer.layers if isinstance(l, Labels)]
        for combo, names in [
            (self._vol_combo, image_names),
            (self._seg_combo, label_names),
            (self._roi_combo, ["<None>"] + label_names),
        ]:
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            idx = combo.findText(current)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    def _get_layer(self, name):
        for layer in self.viewer.layers:
            if layer.name == name:
                return layer
        return None

    def _on_use_layer_scale_toggled(self, checked):
        for w in (self._vox_z, self._vox_y, self._vox_x,
                  self._vox_z_lbl, self._vox_y_lbl, self._vox_x_lbl):
            w.setEnabled(not checked)

    def _get_voxel_size(self):
        if self._use_layer_scale.isChecked():
            vol_layer = self._get_layer(self._vol_combo.currentText())
            if vol_layer is not None and hasattr(vol_layer, "scale"):
                s = vol_layer.scale
                if len(s) >= 3:
                    return float(s[-3]), float(s[-2]), float(s[-1])
            show_info("Could not read scale — falling back to (1, 1, 1) mm.")
        return (self._vox_z.value(), self._vox_y.value(), self._vox_x.value())

    # ──────────────────────────────────────────────────────────────────────
    # Run detection
    # ──────────────────────────────────────────────────────────────────────

    def _on_run(self):
        vol_layer = self._get_layer(self._vol_combo.currentText())
        seg_layer = self._get_layer(self._seg_combo.currentText())
        if vol_layer is None or seg_layer is None:
            show_info("Please select valid Vol and Seg layers.")
            return

        roi_name = self._roi_combo.currentText()
        roi_layer = None if roi_name == "<None>" else self._get_layer(roi_name)

        params = dict(
            vol_img=vol_layer.data,
            vol_labels=seg_layer.data,
            anterior_label_val=self._anterior_label.value(),
            baseline_label_val=self._baseline_label.value(),
            tumor_label_val=self._tumor_label.value(),
            spline_smoothing=(
                None if self._spline_smoothing.value() == 0.0
                else self._spline_smoothing.value()
            ),
            min_elevation_px=self._min_elevation.value(),
            edge_margin_cols=self._edge_margin.value(),
            margin_below_px=self._margin_below.value(),
            robust_sigma=self._robust_sigma.value(),
            robust_iters=self._robust_iters.value(),
            enface_roi_mask=self._build_roi_mask(roi_layer, vol_layer.data.shape),
            voxel_size_mm=self._get_voxel_size(),
            min_layer_thickness=self._min_layer_thickness.value(),
            ignore_top_px=self._ignore_top_px.value(),
            show_diagnostic_lines=self._show_diagnostic_lines.isChecked(),
            vol_name=vol_layer.name,
            generate_3d_render=self._generate_3d.isChecked(),
            mesh_smoothing_iters=self._mesh_smoothing.value(),
        )

        self._run_btn.setEnabled(False)
        self._run_btn.setText("⏳  Running…")
        show_info("Tumor detection started…")

        worker = _run_detection_thread(**params)
        worker.returned.connect(self._on_detection_finished)
        worker.errored.connect(self._on_error)
        worker.start()
        self._worker = worker

    def _build_roi_mask(self, roi_layer, vol_shape):
        if roi_layer is None:
            return None
        roi_data = roi_layer.data
        D, H, W = vol_shape
        if roi_data.ndim == 2 and roi_data.shape == (D, W):
            return roi_data > 0
        if roi_data.ndim == 3 and roi_data.shape[0] == 1:
            roi_data = roi_data[0]
            if roi_data.shape == (D, W):
                return roi_data > 0
        show_info(
            f"En-face ROI shape {roi_layer.data.shape} does not match expected "
            f"({D}, {W}). ROI ignored."
        )
        return None

    def _on_detection_finished(self, result):
        tumor_mask, volume_mm3, vol_name, tumor_label_val, voxel_size, mesh_data = result

        layer_name = f"{vol_name}_Tumor_Mask_Quadratic"
        existing = self._get_layer(layer_name)
        if existing is not None:
            existing.data = tumor_mask
            self._tumor_layer = existing
        else:
            self._tumor_layer = self.viewer.add_labels(tumor_mask, name=layer_name)

        # Add 3D mesh if generated
        if mesh_data is not None:
            verts, faces, vals = mesh_data
            surf_name = f"{vol_name}_Tumor_3D_Quadratic"
            existing_surf = self._get_layer(surf_name)
            if existing_surf is not None:
                existing_surf.data = (verts, faces, vals)
            else:
                self.viewer.add_surface((verts, faces, vals), name=surf_name, colormap="turbo")

        self._display_volume(volume_mm3, voxel_size)
        show_info(f"Tumor detection complete. Volume: {volume_mm3:.3f} mm³")
        self._run_btn.setEnabled(True)
        self._run_btn.setText("▶  Run Tumor Detection")

    def _on_error(self, exc):
        show_info(f"Tumor detection failed: {exc}")
        self._run_btn.setEnabled(True)
        self._run_btn.setText("▶  Run Tumor Detection")
        raise exc

    # ──────────────────────────────────────────────────────────────────────
    # Recalculate
    # ──────────────────────────────────────────────────────────────────────

    def _on_recalculate(self):
        mask_layer = self._tumor_layer
        if mask_layer is None:
            for layer in self.viewer.layers:
                if isinstance(layer, Labels) and layer.name.endswith("_Tumor_Mask_Quadratic"):
                    mask_layer = layer
                    break
        if mask_layer is None:
            show_info("No tumor mask layer found. Run detection first.")
            return

        voxel_size = self._get_voxel_size()
        label_val = self._tumor_label.value()
        volume_mm3 = compute_tumor_volume_mm3(
            mask_layer.data, voxel_size_mm=voxel_size, label_val=label_val
        )
        self._display_volume(volume_mm3, voxel_size)
        show_info(f"Volume recalculated: {volume_mm3:.3f} mm³")

    def _display_volume(self, volume_mm3, voxel_size):
        vz, vy, vx = voxel_size
        self._result_label.setText(
            f"Volume: {volume_mm3:.4f} mm³\n"
            f"(voxel: {vz:.4f} × {vy:.4f} × {vx:.4f} mm)"
        )


# ──────────────────────────────────────────────────────────────────────────
# Background thread
# ──────────────────────────────────────────────────────────────────────────

@thread_worker
def _run_detection_thread(
    vol_img, vol_labels,
    anterior_label_val, baseline_label_val, tumor_label_val,
    spline_smoothing, min_elevation_px, edge_margin_cols, margin_below_px,
    robust_sigma, robust_iters, enface_roi_mask, voxel_size_mm,
    min_layer_thickness, ignore_top_px, show_diagnostic_lines,
    vol_name, generate_3d_render, mesh_smoothing_iters,
):
    tumor_mask, volume_mm3 = compute_tumor_mask_volume(
        vol_img=vol_img,
        vol_labels=vol_labels,
        anterior_label_val=anterior_label_val,
        baseline_label_val=baseline_label_val,
        spline_smoothing=spline_smoothing,
        min_elevation_px=min_elevation_px,
        edge_margin_cols=edge_margin_cols,
        margin_below_px=margin_below_px,
        robust_sigma=robust_sigma,
        robust_iters=robust_iters,
        tumor_label_val=tumor_label_val,
        enface_roi_mask=enface_roi_mask,
        voxel_size_mm=voxel_size_mm,
        min_layer_thickness=min_layer_thickness,
        ignore_top_px=ignore_top_px,
        show_diagnostic_lines=show_diagnostic_lines,
    )

    # Optional 3D mesh generation
    mesh_data = None
    if generate_3d_render:
        try:
            from skimage.measure import marching_cubes
            import trimesh
            bool_vol = (tumor_mask == tumor_label_val)
            spacing = (voxel_size_mm[0], voxel_size_mm[1], voxel_size_mm[2])
            verts, faces, _, _ = marching_cubes(bool_vol, level=0.5, spacing=spacing)
            if mesh_smoothing_iters > 0 and len(verts) > 0:
                mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                trimesh.smoothing.filter_laplacian(mesh, iterations=mesh_smoothing_iters)
                verts = mesh.vertices
                faces = mesh.faces
            if len(verts) > 0:
                mesh_data = (verts, faces, np.ones(len(verts)))
        except Exception as e:
            print(f"3D mesh generation failed: {e}")

    return tumor_mask, volume_mm3, vol_name, tumor_label_val, voxel_size_mm, mesh_data
