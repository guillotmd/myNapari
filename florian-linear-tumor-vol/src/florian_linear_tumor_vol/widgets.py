"""
widgets.py — Linear-baseline tumor volumetry widget (modern UI).
"""
from __future__ import annotations
import numpy as np
import napari
from napari.layers import Image, Labels
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info, show_warning
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QProgressBar, QPushButton,
    QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from florian_linear_tumor_vol.plugin import execute_retinoblastoma_pipeline
from florian_linear_tumor_vol._style import (
    STYLESHEET, CollapsibleSection,
    make_plugin_header, make_separator,
    style_primary_btn, style_secondary_btn,
    set_result_success, set_result_error, set_result_info,
    TEXT_MUTED, TEXT_SECOND, BG_CARD, BORDER,
)


class RetinoblastomaWidget(QWidget):
    """Linear-baseline retinoblastoma tumor volume measurement."""

    def __init__(self, napari_viewer: "napari.viewer.Viewer", parent=None):
        super().__init__(parent)
        self.viewer = napari_viewer
        self.setStyleSheet(STYLESHEET)
        self._build_ui()
        self._connect_signals()

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _form_row(form: QFormLayout, label: str, widget, tip: str):
        lbl = QLabel(label)
        lbl.setToolTip(tip)
        widget.setToolTip(tip)
        form.addRow(lbl, widget)

    def _combo(self) -> QComboBox:
        return QComboBox()

    def _spin(self, lo, hi, val, decimals=0, suffix="") -> QSpinBox | QDoubleSpinBox:
        if decimals:
            w = QDoubleSpinBox()
            w.setDecimals(decimals)
        else:
            w = QSpinBox()
        w.setRange(lo, hi)
        if decimals:
            w.setValue(float(val))
        else:
            w.setValue(int(val))
        if suffix:
            w.setSuffix(suffix)
        return w

    # ──────────────────────────────────────────────────────────────────────
    # UI
    # ──────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(10, 6, 10, 16)
        lay.setSpacing(8)
        scroll.setWidget(inner)
        root.addWidget(scroll)

        # Header
        lay.addWidget(make_plugin_header(
            "", "Linear Tumor Volume",
            "Detects retinal tumors using linear baseline extrapolation."
        ))
        lay.addWidget(make_separator())

        # ── Input Layers ───────────────────────────────────────────────
        grp_in = QGroupBox("Input Layers")
        f_in = QFormLayout(grp_in)
        f_in.setSpacing(6)

        self._bscan_combo = self._combo()
        self._form_row(f_in, "B-Scan (Image):", self._bscan_combo,
            "3D OCT intensity volume — the raw B-scan stack.")

        self._seg_combo = self._combo()
        self._form_row(f_in, "Segmentation (Labels):", self._seg_combo,
            "Labels layer: 1 = retina, 2 = choroid.")

        self._roi_combo = self._combo()
        self._form_row(f_in, "En-face ROI (opt.):", self._roi_combo,
            "Optional Labels layer with tumor region drawn on the en-face view.\n"
            "Leave as <None> to scan the entire volume.")

        lay.addWidget(grp_in)

        # ── Label IDs ──────────────────────────────────────────────────
        grp_ids = QGroupBox("Label ID Configuration")
        ids_v = QVBoxLayout(grp_ids)
        ids_v.setSpacing(6)

        row1 = QHBoxLayout()
        self._retina_label = self._spin(0, 255, 1)
        self._choroid_label = self._spin(0, 255, 2)
        self._retina_label.setToolTip("Label value for retina in segmentation. Default: 1")
        self._choroid_label.setToolTip("Label value for choroid in segmentation. Default: 2")
        row1.addWidget(QLabel("Retina ID:")); row1.addWidget(self._retina_label)
        row1.addSpacing(10)
        row1.addWidget(QLabel("Choroid ID:")); row1.addWidget(self._choroid_label)
        ids_v.addLayout(row1)

        row2 = QHBoxLayout()
        self._enface_tumor_label = self._spin(1, 255, 3)
        self._output_tumor_label = self._spin(1, 255, 4)
        self._enface_tumor_label.setToolTip(
            "Label ID drawn on the en-face ROI that marks the tumor region. Default: 3")
        self._output_tumor_label.setToolTip(
            "Label ID written into the output tumor mask. Default: 4")
        row2.addWidget(QLabel("En-face Tumor:")); row2.addWidget(self._enface_tumor_label)
        row2.addSpacing(10)
        row2.addWidget(QLabel("Output Mask ID:")); row2.addWidget(self._output_tumor_label)
        ids_v.addLayout(row2)

        lay.addWidget(grp_ids)

        # ── Physical Resolution ────────────────────────────────────────
        grp_res = QGroupBox("Physical Resolution (µm / px)")
        res_h = QHBoxLayout(grp_res)
        res_h.setSpacing(6)

        self._axial_res   = self._spin(0.1, 100, 3.87, 2)
        self._lateral_res = self._spin(0.1, 100, 11.5, 2)
        self._spacing     = self._spin(0.1, 500, 120.0, 1)
        self._axial_res.setToolTip("Axial (depth) pixel size in µm. Default: 3.87 µm")
        self._lateral_res.setToolTip("Lateral (horizontal) pixel size in µm. Default: 11.5 µm")
        self._spacing.setToolTip(
            "Inter-slice spacing in µm — distance between consecutive B-scans. Default: 120 µm")

        res_h.addWidget(QLabel("Axial:")); res_h.addWidget(self._axial_res)
        res_h.addWidget(QLabel("Lat:")); res_h.addWidget(self._lateral_res)
        res_h.addWidget(QLabel("Slice gap:")); res_h.addWidget(self._spacing)

        lay.addWidget(grp_res)

        # ── Detection Parameters ───────────────────────────────────────
        grp_det = QGroupBox("Detection Parameters")
        f_det = QFormLayout(grp_det)
        f_det.setSpacing(6)

        self._elevation_threshold = self._spin(0, 500, 5.0, 1)
        self._form_row(f_det, "Elevation threshold (px):", self._elevation_threshold,
            "Minimum retinal elevation above the baseline to classify as tumor.\n"
            "Increase to reduce false positives. Default: 5 px")

        self._edge_margin_cols = self._spin(0, 500, 15)
        self._form_row(f_det, "Edge margin (cols):", self._edge_margin_cols,
            "Columns to ignore at the left/right scan boundary.\n"
            "Prevents false positives from edge artefacts. Default: 15 cols")

        self._prior_sigma = self._spin(0.1, 500, 10.0, 1)
        self._form_row(f_det, "Prior sigma (px):", self._prior_sigma,
            "Gaussian falloff width for the en-face ROI prior probability.\n"
            "Smaller = tighter prior. Default: 10 px")

        self._mapping_mode = QComboBox()
        self._mapping_mode.addItems(["linear", "custom_affine"])
        self._form_row(f_det, "Mapping mode:", self._mapping_mode,
            "How to map the en-face ROI onto the B-scan stack.\n"
            "• linear — assumes uniform slice spacing (default)\n"
            "• custom_affine — applies a custom affine transform")

        self._min_layer_thickness = self._spin(1, 100, 5)
        self._form_row(f_det, "Min layer thickness (px):", self._min_layer_thickness,
            "Ignore segmented fragments thinner than this.\n"
            "Filters vitreous seeds and small artefacts. Default: 5 px")

        self._ignore_top_px = self._spin(0, 1000, 0)
        self._form_row(f_det, "Ignore top margin (px):", self._ignore_top_px,
            "Ignore labels in the top N rows of each B-scan.\n"
            "Use if floating artifacts appear near the vitreous. Default: 0")

        lay.addWidget(grp_det)

        # ── Output Options ─────────────────────────────────────────────
        grp_out = QGroupBox("Output Options")
        f_out = QFormLayout(grp_out)
        f_out.setSpacing(6)

        from qtpy.QtWidgets import QCheckBox
        self._show_tumor_mask = QCheckBox("Show tumor mask layer")
        self._show_tumor_mask.setChecked(True)
        self._show_tumor_mask.setToolTip(
            "Add the detected tumor mask as a Labels layer in the viewer.")
        f_out.addRow(self._show_tumor_mask)

        diag_row = QHBoxLayout()
        self._show_diagnostic_lines = QCheckBox("Diagnostic lines")
        self._show_diagnostic_lines.setChecked(False)
        self._show_diagnostic_lines.setToolTip(
            "Draw the fitted retina curve (label 4) and choroid baseline (label 5)\n"
            "on the output mask — useful for debugging baseline fitting.")
        self._diagnostic_line_thickness = self._spin(1, 20, 5)
        self._diagnostic_line_thickness.setToolTip(
            "Thickness of diagnostic lines in pixels. Default: 5")
        diag_row.addWidget(self._show_diagnostic_lines)
        diag_row.addSpacing(8)
        diag_row.addWidget(QLabel("Thickness:"))
        diag_row.addWidget(self._diagnostic_line_thickness)
        diag_row.addStretch()
        f_out.addRow(diag_row)

        lay.addWidget(grp_out)

        # ── Advanced Options (collapsible) ─────────────────────────────
        adv = CollapsibleSection("Advanced Options", collapsed=True)

        f_adv = QFormLayout()
        f_adv.setSpacing(6)

        self._use_choroid_filter = QCheckBox("Filter choroid outliers")
        self._use_choroid_filter.setChecked(False)
        self._use_choroid_filter.setToolTip(
            "Reject choroid detections that appear above the retina or far above\n"
            "the expected choroid baseline. Fixes ONNX mislabeling artefacts.")
        f_adv.addRow(self._use_choroid_filter)

        self._use_morphological_cleanup = QCheckBox("Morphological cleanup")
        self._use_morphological_cleanup.setChecked(False)
        self._use_morphological_cleanup.setToolTip(
            "Apply binary opening → closing → keep largest component.\n"
            "Removes isolated noise voxels and fills small holes in the mask.")
        f_adv.addRow(self._use_morphological_cleanup)

        self._interpolation_mode = QComboBox()
        self._interpolation_mode.addItems(["linear", "pchip"])
        self._interpolation_mode.setToolTip(
            "Baseline interpolation across the tumor gap:\n"
            "• linear — straight line (original behaviour, default)\n"
            "• pchip — monotone cubic Hermite spline (smoother, no overshoot)")
        f_adv.addRow(QLabel("Interpolation:"), self._interpolation_mode)

        self._use_weighted_fitting = QCheckBox("Weighted baseline fitting")
        self._use_weighted_fitting.setChecked(False)
        self._use_weighted_fitting.setToolTip(
            "Upweight columns far from the tumor boundary when fitting baselines.\n"
            "Reduces edge-bleed artefacts near the tumor margins.")
        f_adv.addRow(self._use_weighted_fitting)

        self._smoothing_sigma = self._spin(0, 100, 20.0, 1)
        self._smoothing_sigma.setToolTip(
            "Gaussian smoothing sigma applied to baseline curves.\n"
            "Higher = smoother baseline, less sensitive to local noise. Default: 20")
        f_adv.addRow(QLabel("Smoothing σ (px):"), self._smoothing_sigma)

        self._compute_uncertainty = QCheckBox("Estimate volume uncertainty")
        self._compute_uncertainty.setChecked(False)
        self._compute_uncertainty.setToolTip(
            "Perturb the elevation threshold ±1 px and report the resulting\n"
            "volume range as an uncertainty estimate. Adds ~2× run time.")
        f_adv.addRow(self._compute_uncertainty)

        adv.addLayout(f_adv)
        lay.addWidget(adv)

        # ── Buttons ────────────────────────────────────────────────────
        lay.addSpacing(4)
        self._run_btn = QPushButton("Calculate Tumor Volume")
        self._run_btn.setFixedHeight(44)
        style_primary_btn(self._run_btn)
        lay.addWidget(self._run_btn)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)   # indeterminate
        self._progress.setFixedHeight(5)
        self._progress.setVisible(False)
        lay.addWidget(self._progress)

        self._recalc_btn = QPushButton("Recalculate Volume from Mask")
        self._recalc_btn.setFixedHeight(32)
        style_secondary_btn(self._recalc_btn)
        self._recalc_btn.setToolTip(
            "Recompute volume from the existing tumor mask after manual editing.\n"
            "Does not re-run the detection pipeline.")
        lay.addWidget(self._recalc_btn)

        # Result card
        self._result_label = QLabel("")
        self._result_label.setAlignment(Qt.AlignCenter)
        self._result_label.setWordWrap(True)
        self._result_label.setVisible(False)
        self._result_label.setMinimumHeight(42)
        lay.addWidget(self._result_label)

        lay.addStretch()

    # ──────────────────────────────────────────────────────────────────────
    # Signals
    # ──────────────────────────────────────────────────────────────────────

    def _connect_signals(self):
        self._run_btn.clicked.connect(self._on_run)
        self._recalc_btn.clicked.connect(self._on_recalc_clicked)
        self.viewer.layers.events.inserted.connect(self._refresh_combos)
        self.viewer.layers.events.removed.connect(self._refresh_combos)
        self._refresh_combos()

    def _refresh_combos(self, event=None):
        image_names = [l.name for l in self.viewer.layers if isinstance(l, Image)]
        label_names = [l.name for l in self.viewer.layers if isinstance(l, Labels)]

        for combo, names in [
            (self._bscan_combo, image_names),
            (self._seg_combo, label_names),
        ]:
            cur = combo.currentText()
            combo.blockSignals(True)
            combo.clear(); combo.addItems(names)
            idx = combo.findText(cur)
            if idx >= 0: combo.setCurrentIndex(idx)
            combo.blockSignals(False)

        roi_names = ["<None>"] + label_names
        cur_roi = self._roi_combo.currentText()
        self._roi_combo.blockSignals(True)
        self._roi_combo.clear(); self._roi_combo.addItems(roi_names)
        idx = self._roi_combo.findText(cur_roi)
        if idx >= 0: self._roi_combo.setCurrentIndex(idx)
        self._roi_combo.blockSignals(False)

    def _get_layer(self, name: str):
        for l in self.viewer.layers:
            if l.name == name: return l
        return None

    # ──────────────────────────────────────────────────────────────────────
    # Run
    # ──────────────────────────────────────────────────────────────────────

    def _on_run(self):
        bscan_layer = self._get_layer(self._bscan_combo.currentText())
        seg_layer   = self._get_layer(self._seg_combo.currentText())

        if bscan_layer is None or seg_layer is None:
            show_info("Please select valid B-scan and Segmentation layers.")
            return

        roi_name  = self._roi_combo.currentText()
        roi_layer = None if roi_name == "<None>" else self._get_layer(roi_name)

        params = dict(
            bscan_data=bscan_layer.data,
            seg_data=seg_layer.data,
            enface_data=roi_layer.data if roi_layer else None,
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
            mesh_smoothing_iters=0,
            show_tumor_mask=self._show_tumor_mask.isChecked(),
            generate_3d_render=False,
            show_diagnostic_lines=self._show_diagnostic_lines.isChecked(),
            diagnostic_line_thickness=self._diagnostic_line_thickness.value(),
            bscan_name=bscan_layer.name,
            use_choroid_filter=self._use_choroid_filter.isChecked(),
            use_morphological_cleanup=self._use_morphological_cleanup.isChecked(),
            interpolation_mode=self._interpolation_mode.currentText(),
            use_weighted_fitting=self._use_weighted_fitting.isChecked(),
            smoothing_sigma=self._smoothing_sigma.value(),
            compute_uncertainty=self._compute_uncertainty.isChecked(),
        )

        self._set_running(True)
        self._result_label.setVisible(False)

        worker = _run_pipeline_thread(params)
        worker.yielded.connect(self._on_progress)
        worker.returned.connect(self._on_pipeline_finished)
        worker.errored.connect(self._on_error)
        worker.start()

    def _set_running(self, running: bool):
        self._run_btn.setEnabled(not running)
        self._run_btn.setText("Running..." if running else "Calculate Tumor Volume")
        self._progress.setVisible(running)
        self._recalc_btn.setEnabled(not running)

    def _on_progress(self, msg: str):
        # Parse percentage from messages like "Curve correcting… 45%"
        import re
        m = re.search(r'(\d+)\s*%', msg)
        if m:
            pct = int(m.group(1))
            self._progress.setRange(0, 100)
            self._progress.setValue(pct)
        else:
            self._progress.setRange(0, 0)  # indeterminate pulse

    def _on_pipeline_finished(self, result):
        self._set_running(False)
        if result is None:
            return

        tumor_mask, volume, uncertainty, mesh_data, output_tumor_label, bscan_name, params = result

        mask_name = f"{bscan_name}_Tumor_Mask_Linear"
        existing  = self._get_layer(mask_name)
        if existing:
            existing.data = tumor_mask
        else:
            self.viewer.add_labels(tumor_mask, name=mask_name)

        if uncertainty > 0:
            set_result_success(self._result_label,
                f"Volume: {volume:.4f} ± {uncertainty:.4f} mm³")
        else:
            set_result_success(self._result_label, f"Volume: {volume:.4f} mm³")

        show_info(f"Linear pipeline complete — Volume: {volume:.4f} mm³\n"
                  "Use 'Generate 3D Tumor Render' to view in 3D.")

    def _on_error(self, exc):
        self._set_running(False)
        set_result_error(self._result_label, f"Error: {exc}")
        show_warning(f"Pipeline failed: {exc}")
        raise exc

    def _on_recalc_clicked(self):
        output_layer = None
        for l in self.viewer.layers:
            if isinstance(l, Labels) and "_Tumor_Mask_Linear" in l.name:
                output_layer = l; break

        if output_layer is None:
            sel = self.viewer.layers.selection.active
            if isinstance(sel, Labels):
                output_layer = sel
            else:
                show_info("No _Tumor_Mask_Linear layer found. Select a mask layer.")
                return

        params = dict(
            axial_resolution=self._axial_res.value(),
            lateral_resolution=self._lateral_res.value(),
            inter_slice_spacing=self._spacing.value(),
            compute_uncertainty=self._compute_uncertainty.isChecked(),
        )
        from florian_linear_tumor_vol.plugin import recalculate_volume
        volume, uncertainty = recalculate_volume(
            output_layer.data, self._output_tumor_label.value(), params)
        if uncertainty > 0:
            set_result_success(self._result_label,
                f"Volume: {volume:.4f} ± {uncertainty:.4f} mm³")
        else:
            set_result_success(self._result_label, f"Volume: {volume:.4f} mm³")
        show_info(f"Volume recalculated: {volume:.4f} mm³")


@thread_worker
def _run_pipeline_thread(params: dict):
    import queue, threading
    pq = queue.Queue()
    result_holder = [None, None]

    params['progress_callback'] = lambda m: pq.put(m)

    def _run():
        try:
            result_holder[0] = execute_retinoblastoma_pipeline(params)
        except Exception as e:
            result_holder[1] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    while t.is_alive():
        try: yield pq.get(timeout=0.1)
        except queue.Empty: pass

    while not pq.empty():
        yield pq.get_nowait()

    if result_holder[1]:
        raise result_holder[1]
    return result_holder[0]
