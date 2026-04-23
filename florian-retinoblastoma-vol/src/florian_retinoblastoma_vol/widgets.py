from typing import Optional, Tuple
import numpy as np
import napari
from napari.layers import Image, Labels
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info, show_warning
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox, QPushButton, QLabel, QScrollArea
)

from florian_retinoblastoma_vol.plugin import execute_retinoblastoma_pipeline

class RetinoblastomaWidget(QWidget):
    """Modern Napari QWidget for Retinoblastoma Volumetric Measurement."""

    def __init__(self, napari_viewer: "napari.viewer.Viewer", parent=None):
        super().__init__(parent)
        self.viewer = napari_viewer
        self._tumor_layer = None
        self._mesh_layer = None

        self._build_ui()
        self._connect_signals()

    # ──────────────────────────────────────────────────────────────────────────
    # UI Construction
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _add_row(form: "QFormLayout", label_text: str, widget, tooltip: str) -> None:
        """Add a labelled row with tooltip to both label and widget."""
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

        # ── 1. Layer selection ────────────────────────────────────────────────
        layer_group = QGroupBox("Input Layers")
        layer_form = QFormLayout(layer_group)

        self._bscan_combo = QComboBox()
        self._add_row(layer_form, "B-Scan Image:", self._bscan_combo,
            "The 3D OCT intensity volume (Image layer).")

        self._seg_combo = QComboBox()
        self._add_row(layer_form, "Segmentation:", self._seg_combo,
            "The corresponding Labels layer for B-scans.")

        self._roi_combo = QComboBox()
        self._add_row(layer_form, "En-face ROI (opt.):", self._roi_combo,
            "Optional: En-face ROI Labels layer highlighting the tumor.\n"
            "If left empty (<None>), the entire volume is scanned.")

        layout.addWidget(layer_group)

        # ── 2. Label Setup (Side-by-side) ─────────────────────────────────────
        label_group = QGroupBox("Label ID Configuration")
        label_vbox = QVBoxLayout(label_group)
        
        row1 = QHBoxLayout()
        self._retina_label = QSpinBox(); self._retina_label.setRange(0, 255); self._retina_label.setValue(1)
        self._choroid_label = QSpinBox(); self._choroid_label.setRange(0, 255); self._choroid_label.setValue(2)
        row1.addWidget(QLabel("Retina ID:"))
        row1.addWidget(self._retina_label)
        row1.addWidget(QLabel("Choroid ID:"))
        row1.addWidget(self._choroid_label)
        self._retina_label.setToolTip("Label ID for retina in the segmentation layer.")
        self._choroid_label.setToolTip("Label ID for choroid in the segmentation layer.")
        label_vbox.addLayout(row1)

        row2 = QHBoxLayout()
        self._enface_tumor_label = QSpinBox(); self._enface_tumor_label.setRange(1, 255); self._enface_tumor_label.setValue(3)
        self._output_tumor_label = QSpinBox(); self._output_tumor_label.setRange(1, 255); self._output_tumor_label.setValue(4)
        row2.addWidget(QLabel("En-face Tumor ID:"))
        row2.addWidget(self._enface_tumor_label)
        row2.addWidget(QLabel("Output Mask ID:"))
        row2.addWidget(self._output_tumor_label)
        self._enface_tumor_label.setToolTip("Label ID drawn on the en-face ROI representing the tumor.")
        self._output_tumor_label.setToolTip("Label ID to use when creating the final 3D tumor output mask.")
        label_vbox.addLayout(row2)

        layout.addWidget(label_group)

        # ── 3. Physical Scale (Side-by-side) ──────────────────────────────────
        scale_group = QGroupBox("Physical Resolution (µm/px)")
        scale_layout = QHBoxLayout(scale_group)

        self._axial_res = QDoubleSpinBox()
        self._axial_res.setRange(0.1, 100.0); self._axial_res.setDecimals(2); self._axial_res.setValue(3.87)
        self._axial_res.setToolTip("Depth resolution per pixel in the B-scan.")
        
        self._lateral_res = QDoubleSpinBox()
        self._lateral_res.setRange(0.1, 100.0); self._lateral_res.setDecimals(2); self._lateral_res.setValue(11.5)
        self._lateral_res.setToolTip("Horizontal resolution per pixel in the B-scan.")
        
        self._spacing = QDoubleSpinBox()
        self._spacing.setRange(0.1, 500.0); self._spacing.setDecimals(2); self._spacing.setValue(120.0)
        self._spacing.setToolTip("Physical distance between consecutive B-scans.")

        scale_layout.addWidget(QLabel("Axial:")); scale_layout.addWidget(self._axial_res)
        scale_layout.addWidget(QLabel("Lat:")); scale_layout.addWidget(self._lateral_res)
        scale_layout.addWidget(QLabel("Space:")); scale_layout.addWidget(self._spacing)

        layout.addWidget(scale_group)

        # ── 4. Detection Parameters ───────────────────────────────────────────
        algo_group = QGroupBox("Detection Parameters")
        algo_form = QFormLayout(algo_group)

        self._prior_sigma = QDoubleSpinBox()
        self._prior_sigma.setRange(0.1, 500.0); self._prior_sigma.setDecimals(1); self._prior_sigma.setValue(10.0)
        self._add_row(algo_form, "Prior Sigma (px):", self._prior_sigma,
            "Gaussian falloff for the en-face ROI prior probability.")

        self._mapping_mode = QComboBox()
        self._mapping_mode.addItems(["linear", "custom_affine"])
        self._add_row(algo_form, "Mapping Mode:", self._mapping_mode,
            "How to map en-face image to B-scan stack.")
            
        self._elevation_threshold = QDoubleSpinBox()
        self._elevation_threshold.setRange(0.0, 500.0); self._elevation_threshold.setDecimals(1); self._elevation_threshold.setValue(5.0)
        self._add_row(algo_form, "Elevation Threshold (px):", self._elevation_threshold,
            "Minimum elevation of the retina above the expected baseline to classify as tumor.")
            
        self._edge_margin_cols = QSpinBox()
        self._edge_margin_cols.setRange(0, 500); self._edge_margin_cols.setValue(15)
        self._add_row(algo_form, "Edge Margin (cols):", self._edge_margin_cols,
            "Number of columns to ignore at the left and right edges of the valid retina scan to prevent false positives.")
            
        self._min_layer_thickness = QSpinBox()
        self._min_layer_thickness.setRange(1, 100); self._min_layer_thickness.setValue(5)
        self._add_row(algo_form, "Min Layer Thickness (px):", self._min_layer_thickness,
            "Ignore isolated tissue fragments (like vitreous seeds) thinner than this value.")
            
        self._ignore_top_px = QSpinBox()
        self._ignore_top_px.setRange(0, 1000); self._ignore_top_px.setValue(0)
        self._add_row(algo_form, "Ignore Top Margin (px):", self._ignore_top_px,
            "Ignore any segmentation labels in the top N pixels of the image (useful for filtering out floating artifacts).")

        layout.addWidget(algo_group)

        # ── 6. Options ────────────────────────────────────────────────────────
        opt_group = QGroupBox("Output Options")
        opt_layout = QVBoxLayout(opt_group)

        self._generate_3d = QCheckBox("Generate 3D Render")
        self._generate_3d.setChecked(False)
        self._generate_3d.setToolTip("If checked, generates a 3D Surface mesh layer of the tumor.")
        opt_layout.addWidget(self._generate_3d)

        self._show_diagnostic_lines = QCheckBox("Show Diagnostic Lines (Labels 4 & 5)")
        self._show_diagnostic_lines.setChecked(False)
        self._show_diagnostic_lines.setToolTip("If checked, adds the computed retina curve and choroid baseline to the output mask.")
        opt_layout.addWidget(self._show_diagnostic_lines)

        self._mesh_smoothing = QSpinBox()
        self._mesh_smoothing.setRange(0, 50); self._mesh_smoothing.setValue(10)
        self._mesh_smoothing.setToolTip("Laplacian smoothing iterations for 3D mesh (if generated).")
        form_mesh = QFormLayout()
        form_mesh.addRow("Mesh Smoothing Iters:", self._mesh_smoothing)
        opt_layout.addLayout(form_mesh)

        layout.addWidget(opt_group)

        # ── 7. Action Buttons ─────────────────────────────────────────────────
        self._run_btn = QPushButton("▶ Calculate Tumor Volume")
        self._run_btn.setFixedHeight(40)
        self._run_btn.setStyleSheet("font-weight: bold;")
        self._run_btn.setToolTip(
            "This plugin calculates tumor volume by identifying the anterior retinal surface\n"
            "and the posterior choroidal boundary. It interpolates a healthy baseline across\n"
            "these surfaces to handle gaps or shadowing. Columns where the retina is\n"
            "significantly elevated above the baseline are flagged as tumor. The volume is\n"
            "calculated by counting the voxels between the elevated retinal dome and the\n"
            "underlying choroidal boundary, bounded by your en-face ROI, and multiplying\n"
            "by the physical voxel dimensions."
        )
        layout.addWidget(self._run_btn)
        
        self._recalc_btn = QPushButton("🔢 Recalculate Volume")
        self._recalc_btn.setFixedHeight(30)
        self._recalc_btn.setToolTip("Recalculate volume from the currently selected mask (useful after manual editing).")
        layout.addWidget(self._recalc_btn)

        self._result_label = QLabel("Volume: —")
        self._result_label.setAlignment(Qt.AlignCenter)
        self._result_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 10px;")
        layout.addWidget(self._result_label)

        self._refresh_layer_combos()

    # ──────────────────────────────────────────────────────────────────────────
    # Signal wiring
    # ──────────────────────────────────────────────────────────────────────────

    def _connect_signals(self):
        self._run_btn.clicked.connect(self._on_run)
        self._recalc_btn.clicked.connect(self._on_recalc_clicked)

        # Refresh combos when layers are added / removed
        self.viewer.layers.events.inserted.connect(self._refresh_layer_combos)
        self.viewer.layers.events.removed.connect(self._refresh_layer_combos)

    def _refresh_layer_combos(self, *_):
        image_names = [l.name for l in self.viewer.layers if isinstance(l, Image)]
        label_names = [l.name for l in self.viewer.layers if isinstance(l, Labels)]

        for combo, names in [
            (self._bscan_combo, image_names),
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

    def _get_layer(self, name: str):
        for layer in self.viewer.layers:
            if layer.name == name:
                return layer
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Execution
    # ──────────────────────────────────────────────────────────────────────────

    def _on_run(self):
        bscan_layer = self._get_layer(self._bscan_combo.currentText())
        seg_layer = self._get_layer(self._seg_combo.currentText())

        if bscan_layer is None or seg_layer is None:
            show_info("Please select valid B-scan and Segmentation layers.")
            return

        roi_name = self._roi_combo.currentText()
        roi_layer = None if roi_name == "<None>" else self._get_layer(roi_name)
        
        # Build params dictionary
        params = dict(
            bscan_data=bscan_layer.data,
            seg_data=seg_layer.data,
            enface_data=roi_layer.data if roi_layer is not None else None,
            retina_label=self._retina_label.value(),
            choroid_label=self._choroid_label.value(),
            enface_tumor_label=self._enface_tumor_label.value(),
            output_tumor_label=self._output_tumor_label.value(),
            axial_resolution=self._axial_res.value(),
            lateral_resolution=self._lateral_res.value(),
            inter_slice_spacing=self._spacing.value(),
            prior_sigma=self._prior_sigma.value(),
            mapping_mode=self._mapping_mode.currentText(),
            elevation_threshold=self._elevation_threshold.value(),
            edge_margin_cols=self._edge_margin_cols.value(),
            min_layer_thickness=self._min_layer_thickness.value(),
            ignore_top_px=self._ignore_top_px.value(),
            mesh_smoothing_iters=self._mesh_smoothing.value(),
            generate_3d_render=self._generate_3d.isChecked(),
            show_diagnostic_lines=self._show_diagnostic_lines.isChecked()
        )

        self._run_btn.setEnabled(False)
        self._run_btn.setText("⏳ Calculating Tumor Volume...")
        show_info("Retinoblastoma Volume Pipeline started...")

        # We pass the data rather than the layer objects to the thread worker
        worker = _run_pipeline_thread(params)
        worker.returned.connect(self._on_pipeline_finished)
        worker.errored.connect(self._on_error)
        worker.start()

    def _on_pipeline_finished(self, result):
        if result is None:
            # Aborted early
            self._run_btn.setEnabled(True)
            self._run_btn.setText("▶ Calculate Tumor Volume")
            return

        tumor_mask, volume, uncertainty, mesh_data, output_tumor_label = result

        # 1. Update Labels layer
        mask_name = "Retinoblastoma Output Mask"
        existing_mask = self._get_layer(mask_name)
        if existing_mask is not None:
            existing_mask.data = tumor_mask
        else:
            self.viewer.add_labels(tumor_mask, name=mask_name)

        # 2. Update Surface layer if requested
        if mesh_data is not None:
            verts, faces, vals = mesh_data
            surface_name = "Retinoblastoma 3D Surface"
            existing_surf = self._get_layer(surface_name)
            if existing_surf is not None:
                existing_surf.data = (verts, faces, vals)
            else:
                self.viewer.add_surface((verts, faces, vals), name=surface_name, colormap="turbo")

        self._result_label.setText(f"Volume: {volume:.4f} mm³ ± {uncertainty:.4f}")
        show_info(f"Pipeline Complete! Volume: {volume:.4f} mm³")
        
        self._run_btn.setEnabled(True)
        self._run_btn.setText("▶ Calculate Tumor Volume")

    def _on_error(self, exc):
        show_warning(f"Pipeline failed: {exc}")
        self._run_btn.setEnabled(True)
        self._run_btn.setText("▶ Calculate Tumor Volume")
        raise exc

    def _on_recalc_clicked(self):
        from napari.layers import Labels
        
        output_layer = None
        for layer in self.viewer.layers:
            if layer.name == "Retinoblastoma Output Mask":
                output_layer = layer
                break
                
        if output_layer is None or not isinstance(output_layer, Labels):
            selected = self.viewer.layers.selection.active
            if isinstance(selected, Labels):
                output_layer = selected
            else:
                show_info("Could not find Retinoblastoma Output Mask layer. Select it and try again.")
                return

        params = dict(
            axial_resolution=self._axial_res.value(),
            lateral_resolution=self._lateral_res.value(),
            inter_slice_spacing=self._spacing.value()
        )
        
        from florian_retinoblastoma_vol.plugin import recalculate_volume
        vol_mm3 = recalculate_volume(output_layer.data, self._output_tumor_label.value(), params)
        self._result_label.setText(f"Volume: {vol_mm3:.4f} mm³")
        show_info(f"Recalculated Volume: {vol_mm3:.4f} mm³")



@thread_worker
def _run_pipeline_thread(params: dict):
    """
    Run the logic defined in plugin.py on a background thread.
    Returns:
        (tumor_mask, volume, uncertainty, mesh_data, output_tumor_label)
    """
    return execute_retinoblastoma_pipeline(params)
