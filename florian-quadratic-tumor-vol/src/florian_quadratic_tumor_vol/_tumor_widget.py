"""
_tumor_widget.py — Quadratic-spline tumor volumetry widget (modern UI).
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
    QHBoxLayout, QLabel, QProgressBar, QPushButton,
    QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)
from florian_quadratic_tumor_vol._tumor_funcs import compute_tumor_mask_volume, compute_tumor_volume_mm3
from florian_quadratic_tumor_vol._style import (
    STYLESHEET, CollapsibleSection,
    make_plugin_header, make_separator,
    style_primary_btn, style_secondary_btn,
    set_result_success, set_result_error,
)


class TumorVolumeWidget(QWidget):
    """Quadratic-spline OCT tumor volumetric measurement widget."""

    def __init__(self, napari_viewer, parent=None):
        super().__init__(parent)
        self.viewer = napari_viewer
        self._tumor_layer: Optional[Labels] = None
        self.setStyleSheet(STYLESHEET)
        self._build_ui()
        self._connect_signals()

    @staticmethod
    def _form_row(form, label, widget, tip):
        lbl = QLabel(label); lbl.setToolTip(tip); widget.setToolTip(tip)
        form.addRow(lbl, widget)

    def _spin(self, lo, hi, val, decimals=0):
        if decimals:
            w = QDoubleSpinBox(); w.setDecimals(decimals); w.setValue(float(val))
        else:
            w = QSpinBox(); w.setValue(int(val))
        w.setRange(lo, hi)
        return w

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(10, 6, 10, 16); lay.setSpacing(8)
        scroll.setWidget(inner); root.addWidget(scroll)

        lay.addWidget(make_plugin_header(
            "", "Quadratic Tumor Volume",
            "Detects retinal tumors using quadratic (k=2) spline fitting — models the natural globe curvature."
        ))
        lay.addWidget(make_separator())

        # ── Input Layers ──────────────────────────────────────────────
        grp_in = QGroupBox("Input Layers"); f_in = QFormLayout(grp_in); f_in.setSpacing(6)
        self._vol_combo = QComboBox()
        self._form_row(f_in, "OCT Volume (Image):", self._vol_combo,
            "3D OCT intensity volume (Image layer).")
        self._seg_combo = QComboBox()
        self._form_row(f_in, "Segmentation (Labels):", self._seg_combo,
            "B-scan segmentation Labels layer.\nExpected: 0=vitreous, 1=retina, 2=choroid.")
        self._roi_combo = QComboBox()
        self._form_row(f_in, "En-face ROI (opt.):", self._roi_combo,
            "Optional Labels layer drawn on the en-face projection.\n"
            "Non-zero values mark the tumor region.")
        lay.addWidget(grp_in)

        # ── Detection Parameters ──────────────────────────────────────
        grp_det = QGroupBox("Detection Parameters"); f_det = QFormLayout(grp_det); f_det.setSpacing(6)

        self._anterior_label = self._spin(0, 255, 1)
        self._form_row(f_det, "Retina label:", self._anterior_label,
            "Label value for the retina layer. Default: 1")

        self._baseline_label = self._spin(0, 255, 2)
        self._form_row(f_det, "Choroid label:", self._baseline_label,
            "Label value for the choroid layer. Default: 2")

        self._tumor_label = self._spin(1, 255, 3)
        self._form_row(f_det, "Output tumor label:", self._tumor_label,
            "Label value written into the output mask. Default: 3")

        self._spline_smoothing = self._spin(0, 1_000_000, 0.0, 1)
        self._spline_smoothing.setSpecialValueText("Auto (interpolating)")
        self._form_row(f_det, "Spline smoothing:", self._spline_smoothing,
            "Smoothing factor for the quadratic (k=2) spline.\n"
            "0 = interpolating (default). Increase if surfaces are noisy.")

        self._min_elevation = self._spin(0, 500, 5.0, 1)
        self._form_row(f_det, "Min. elevation (px):", self._min_elevation,
            "Minimum retinal elevation above baseline to classify as tumor. Default: 5 px")

        self._edge_margin = self._spin(0, 200, 15)
        self._form_row(f_det, "Edge margin (cols):", self._edge_margin,
            "Columns to blank at the scan boundary.\nPrevents false positives at B-scan edges. Default: 15")

        self._margin_below = self._spin(0, 500, 60.0, 0)
        self._form_row(f_det, "Max depth margin (px):", self._margin_below,
            "Max pixels below deepest observed choroid for extrapolation.\n"
            "Reduce (30–40) if baseline dips too far down. Default: 60")

        self._min_layer_thickness = self._spin(1, 100, 5)
        self._form_row(f_det, "Min layer thickness (px):", self._min_layer_thickness,
            "Ignore tissue fragments thinner than this. Filters vitreous seeds. Default: 5 px")

        self._ignore_top_px = self._spin(0, 1000, 0)
        self._form_row(f_det, "Ignore top margin (px):", self._ignore_top_px,
            "Ignore labels in the top N rows. Filters floating artefacts. Default: 0")

        lay.addWidget(grp_det)

        # ── Output Options ────────────────────────────────────────────
        grp_out = QGroupBox("Output Options"); f_out = QFormLayout(grp_out); f_out.setSpacing(6)
        diag_row = QHBoxLayout()
        self._show_diagnostic_lines = QCheckBox("Diagnostic lines")
        self._show_diagnostic_lines.setChecked(False)
        self._show_diagnostic_lines.setToolTip(
            "Draw the fitted retinal baseline (label 4) and choroid baseline (label 5)\n"
            "on the output mask — useful for verifying the baseline fit.")
        self._diagnostic_line_thickness = self._spin(1, 20, 5)
        self._diagnostic_line_thickness.setToolTip("Thickness of diagnostic lines in pixels. Default: 5")
        diag_row.addWidget(self._show_diagnostic_lines)
        diag_row.addSpacing(8); diag_row.addWidget(QLabel("Thickness:")); diag_row.addWidget(self._diagnostic_line_thickness); diag_row.addStretch()
        f_out.addRow(diag_row)
        lay.addWidget(grp_out)

        # ── Physical Voxel Size ───────────────────────────────────────
        grp_vox = QGroupBox("Physical Voxel Size (mm)"); f_vox = QFormLayout(grp_vox); f_vox.setSpacing(6)
        self._use_layer_scale = QCheckBox("Read from layer .scale attribute")
        self._use_layer_scale.setChecked(True)
        self._use_layer_scale.setToolTip(
            "Auto-read voxel size from the Image layer's .scale property.\n"
            "Uncheck to enter dimensions manually.")
        f_vox.addRow(self._use_layer_scale)

        self._vox_z = self._spin(0.0001, 100, 1.0, 4); self._vox_z_lbl = QLabel("Voxel Z / depth (mm):")
        self._vox_y = self._spin(0.0001, 100, 1.0, 4); self._vox_y_lbl = QLabel("Voxel Y / lateral (mm):")
        self._vox_x = self._spin(0.0001, 100, 1.0, 4); self._vox_x_lbl = QLabel("Voxel X / width (mm):")
        f_vox.addRow(self._vox_z_lbl, self._vox_z)
        f_vox.addRow(self._vox_y_lbl, self._vox_y)
        f_vox.addRow(self._vox_x_lbl, self._vox_x)
        lay.addWidget(grp_vox)

        # ── Advanced (collapsible) ────────────────────────────────────
        adv = CollapsibleSection("Advanced Options", collapsed=True)
        f_adv = QFormLayout(); f_adv.setSpacing(6)

        self._robust_sigma = self._spin(0.1, 10, 2.0, 1)
        self._form_row(f_adv, "Robust σ (no ROI):", self._robust_sigma,
            "Outlier rejection threshold when no en-face ROI is provided. Default: 2.0")

        self._robust_iters = self._spin(0, 5, 2)
        self._form_row(f_adv, "Robust iters (no ROI):", self._robust_iters,
            "Number of iterative outlier-rejection passes. Default: 2")

        self._use_morphological_cleanup = QCheckBox("Morphological cleanup")
        self._use_morphological_cleanup.setToolTip(
            "Binary opening → closing → keep largest component.\nRemoves noise voxels and fills holes.")
        f_adv.addRow(self._use_morphological_cleanup)

        self._use_weighted_fitting = QCheckBox("Weighted baseline fitting")
        self._use_weighted_fitting.setToolTip(
            "Upweight columns far from the tumor boundary when fitting baselines.")
        f_adv.addRow(self._use_weighted_fitting)

        adv.addLayout(f_adv)
        lay.addWidget(adv)

        # ── Buttons ───────────────────────────────────────────────────
        lay.addSpacing(4)
        self._run_btn = QPushButton("Run Tumor Detection")
        self._run_btn.setFixedHeight(44)
        style_primary_btn(self._run_btn)
        lay.addWidget(self._run_btn)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0); self._progress.setFixedHeight(5); self._progress.setVisible(False)
        lay.addWidget(self._progress)

        self._recalc_btn = QPushButton("Recalculate Volume from Mask")
        self._recalc_btn.setFixedHeight(32)
        style_secondary_btn(self._recalc_btn)
        self._recalc_btn.setToolTip("Recompute volume from the existing mask after manual editing.")
        lay.addWidget(self._recalc_btn)

        self._result_label = QLabel("")
        self._result_label.setAlignment(Qt.AlignCenter); self._result_label.setWordWrap(True)
        self._result_label.setVisible(False); self._result_label.setMinimumHeight(42)
        lay.addWidget(self._result_label)
        lay.addStretch()

    # ──────────────────────────────────────────────────────────────────
    # Signals
    # ──────────────────────────────────────────────────────────────────

    def _connect_signals(self):
        self._run_btn.clicked.connect(self._on_run)
        self._recalc_btn.clicked.connect(self._on_recalculate)
        self._use_layer_scale.toggled.connect(self._on_scale_toggled)
        self.viewer.layers.events.inserted.connect(self._refresh_combos)
        self.viewer.layers.events.removed.connect(self._refresh_combos)
        self._on_scale_toggled(self._use_layer_scale.isChecked())
        self._refresh_combos()

    def _refresh_combos(self, event=None):
        imgs   = [l.name for l in self.viewer.layers if isinstance(l, Image)]
        labels = [l.name for l in self.viewer.layers if isinstance(l, Labels)]
        for combo, names in [(self._vol_combo, imgs), (self._seg_combo, labels)]:
            cur = combo.currentText(); combo.blockSignals(True)
            combo.clear(); combo.addItems(names)
            idx = combo.findText(cur)
            if idx >= 0: combo.setCurrentIndex(idx)
            combo.blockSignals(False)
        roi = ["<None>"] + labels; cur_r = self._roi_combo.currentText()
        self._roi_combo.blockSignals(True); self._roi_combo.clear(); self._roi_combo.addItems(roi)
        idx = self._roi_combo.findText(cur_r)
        if idx >= 0: self._roi_combo.setCurrentIndex(idx)
        self._roi_combo.blockSignals(False)

    def _get_layer(self, name):
        for l in self.viewer.layers:
            if l.name == name: return l
        return None

    def _on_scale_toggled(self, checked):
        for w in (self._vox_z, self._vox_y, self._vox_x,
                  self._vox_z_lbl, self._vox_y_lbl, self._vox_x_lbl):
            w.setEnabled(not checked)

    def _get_voxel_size(self):
        if self._use_layer_scale.isChecked():
            lyr = self._get_layer(self._vol_combo.currentText())
            if lyr and hasattr(lyr, "scale") and len(lyr.scale) >= 3:
                s = lyr.scale
                return float(s[-3]), float(s[-2]), float(s[-1])
            show_info("Could not read scale — falling back to (1,1,1) mm.")
        return (self._vox_z.value(), self._vox_y.value(), self._vox_x.value())

    def _build_roi_mask(self, roi_layer, vol_shape):
        if roi_layer is None: return None
        roi = roi_layer.data; D, H, W = vol_shape
        if roi.ndim == 2 and roi.shape == (D, W): return roi > 0
        if roi.ndim == 3 and roi.shape[0] == 1:
            roi = roi[0]
            if roi.shape == (D, W): return roi > 0
        show_info(f"En-face ROI shape {roi_layer.data.shape} doesn't match ({D},{W}). ROI ignored.")
        return None

    # ──────────────────────────────────────────────────────────────────
    # Run
    # ──────────────────────────────────────────────────────────────────

    def _set_running(self, running: bool):
        self._run_btn.setEnabled(not running)
        self._run_btn.setText("Running..." if running else "Run Tumor Detection")
        self._progress.setVisible(running)
        self._recalc_btn.setEnabled(not running)

    def _on_run(self):
        vol_layer = self._get_layer(self._vol_combo.currentText())
        seg_layer = self._get_layer(self._seg_combo.currentText())
        if not (vol_layer and seg_layer):
            show_info("Please select valid Volume and Segmentation layers."); return

        roi_name  = self._roi_combo.currentText()
        roi_layer = None if roi_name == "<None>" else self._get_layer(roi_name)

        params = dict(
            vol_img=vol_layer.data, vol_labels=seg_layer.data,
            anterior_label_val=self._anterior_label.value(),
            baseline_label_val=self._baseline_label.value(),
            tumor_label_val=self._tumor_label.value(),
            spline_smoothing=(None if self._spline_smoothing.value() == 0
                              else self._spline_smoothing.value()),
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
            diagnostic_line_thickness=self._diagnostic_line_thickness.value(),
            vol_name=vol_layer.name,
            use_morphological_cleanup=self._use_morphological_cleanup.isChecked(),
            use_weighted_fitting=self._use_weighted_fitting.isChecked(),
        )

        self._set_running(True); self._result_label.setVisible(False)
        worker = _run_detection_thread(**params)
        worker.yielded.connect(self._on_progress)
        worker.returned.connect(self._on_detection_finished)
        worker.errored.connect(self._on_error)
        worker.start(); self._worker = worker

    def _on_progress(self, msg: str):
        import re
        m = re.search(r'(\d+)\s*%', msg)
        if m:
            self._progress.setRange(0, 100); self._progress.setValue(int(m.group(1)))
        else:
            self._progress.setRange(0, 0)

    def _on_detection_finished(self, result):
        tumor_mask, volume_mm3, vol_name, tumor_label_val, voxel_size = result
        self._set_running(False)

        layer_name = f"{vol_name}_Tumor_Mask_Quadratic"
        existing = self._get_layer(layer_name)
        if existing:
            existing.data = tumor_mask; self._tumor_layer = existing
        else:
            self._tumor_layer = self.viewer.add_labels(tumor_mask, name=layer_name)

        vz, vy, vx = voxel_size
        set_result_success(self._result_label,
            f"Volume: {volume_mm3:.4f} mm³\n"
            f"Voxel: {vz:.4f} × {vy:.4f} × {vx:.4f} mm")
        show_info(f"Detection complete — Volume: {volume_mm3:.3f} mm³\n"
                  "Use 'Generate 3D Tumor Render' to view in 3D.")

    def _on_error(self, exc):
        self._set_running(False)
        set_result_error(self._result_label, f"Error: {exc}")
        show_info(f"Tumor detection failed: {exc}"); raise exc

    def _on_recalculate(self):
        mask_layer = self._tumor_layer
        if mask_layer is None:
            for l in self.viewer.layers:
                if isinstance(l, Labels) and "_Tumor_Mask_Quadratic" in l.name:
                    mask_layer = l; break
        if mask_layer is None:
            show_info("No tumor mask layer found. Run detection first."); return

        voxel_size = self._get_voxel_size()
        volume_mm3 = compute_tumor_volume_mm3(
            mask_layer.data, voxel_size_mm=voxel_size,
            label_val=self._tumor_label.value())
        vz, vy, vx = voxel_size
        set_result_success(self._result_label,
            f"Volume: {volume_mm3:.4f} mm³\nVoxel: {vz:.4f} × {vy:.4f} × {vx:.4f} mm")
        show_info(f"Volume recalculated: {volume_mm3:.3f} mm³")


@thread_worker
def _run_detection_thread(
    vol_img, vol_labels, anterior_label_val, baseline_label_val, tumor_label_val,
    spline_smoothing, min_elevation_px, edge_margin_cols, margin_below_px,
    robust_sigma, robust_iters, enface_roi_mask, voxel_size_mm,
    min_layer_thickness, ignore_top_px, show_diagnostic_lines, diagnostic_line_thickness,
    vol_name, use_morphological_cleanup, use_weighted_fitting,
):
    import queue, threading
    pq = queue.Queue(); result_holder = [None, None]

    def _run():
        try:
            tumor_mask, volume_mm3 = compute_tumor_mask_volume(
                vol_img=vol_img, vol_labels=vol_labels,
                anterior_label_val=anterior_label_val, baseline_label_val=baseline_label_val,
                spline_smoothing=spline_smoothing, min_elevation_px=min_elevation_px,
                edge_margin_cols=edge_margin_cols, margin_below_px=margin_below_px,
                robust_sigma=robust_sigma, robust_iters=robust_iters,
                tumor_label_val=tumor_label_val, enface_roi_mask=enface_roi_mask,
                voxel_size_mm=voxel_size_mm, min_layer_thickness=min_layer_thickness,
                ignore_top_px=ignore_top_px, show_diagnostic_lines=show_diagnostic_lines,
                diagnostic_line_thickness=diagnostic_line_thickness,
                use_morphological_cleanup=use_morphological_cleanup,
                use_weighted_fitting=use_weighted_fitting,
                progress_callback=lambda m: pq.put(m),
            )
            result_holder[0] = (tumor_mask, volume_mm3, vol_name, tumor_label_val, voxel_size_mm)
        except Exception as e:
            result_holder[1] = e

    t = threading.Thread(target=_run, daemon=True); t.start()
    while t.is_alive():
        try: yield pq.get(timeout=0.1)
        except queue.Empty: pass
    while not pq.empty(): yield pq.get_nowait()
    if result_holder[1]: raise result_holder[1]
    return result_holder[0]
