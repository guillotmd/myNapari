"""
_widget.py — PanOCT curve correction widget (modern UI).
"""
from __future__ import annotations
import numpy as np
from napari.layers import Image, Labels
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info, show_warning
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QProgressBar, QPushButton, QScrollArea, QSpinBox,
    QVBoxLayout, QWidget, QCheckBox,
)
from florian_panOCT_curve_correction.curve_correction import (
    CurveCorrectionParams, correct_volume_slicewise, correct_bscan_2d,
    compute_output_scale,
)
from florian_panOCT_curve_correction._style import (
    STYLESHEET, CollapsibleSection,
    make_plugin_header, make_separator,
    style_primary_btn, style_secondary_btn,
    set_result_success, set_result_error, set_result_info,
    SUCCESS, WARNING, TEXT_MUTED,
)


class CurveCorrectionWidget(QWidget):
    """Napari widget for PanOCT fan-beam curve distortion correction."""

    def __init__(self, napari_viewer, parent=None):
        super().__init__(parent)
        self.viewer = napari_viewer
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
        root = QVBoxLayout(self); root.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget()
        lay = QVBoxLayout(inner); lay.setContentsMargins(10, 6, 10, 16); lay.setSpacing(8)
        scroll.setWidget(inner); root.addWidget(scroll)

        lay.addWidget(make_plugin_header(
            "", "PanOCT Curve Correction",
            "Corrects fan-beam geometric distortion — transforms B-scans from polar to Cartesian coordinates (Bayhaqi et al. 2025)."
        ))
        lay.addWidget(make_separator())

        # ── Layer Selection ───────────────────────────────────────────
        grp_in = QGroupBox("Layers to Correct")
        in_v = QVBoxLayout(grp_in); in_v.setSpacing(6)
        in_v.addWidget(QLabel(
            "Select Image and/or Labels layers to correct.\n"
            "Corrected copies are added with a '_CurveCorr' suffix."
        ))
        self._layer_list = QListWidget()
        self._layer_list.setSelectionMode(QListWidget.MultiSelection)
        self._layer_list.setToolTip(
            "Hold Ctrl / ⌘ to select multiple layers.\n"
            "Images use linear interpolation; Labels use nearest-neighbour.")
        self._layer_list.setMinimumHeight(90); self._layer_list.setMaximumHeight(160)
        in_v.addWidget(self._layer_list)
        lay.addWidget(grp_in)

        # ── Quick Preview ─────────────────────────────────────────────
        grp_prev = QGroupBox("Quick Preview (single slice)")
        prev_f = QFormLayout(grp_prev); prev_f.setSpacing(6)

        prev_info = QLabel(
            "Preview one B-scan slice to tune parameters before running the full volume.")
        prev_info.setWordWrap(True)
        prev_info.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        prev_f.addRow(prev_info)

        self._preview_layer_combo = QComboBox()
        self._form_row(prev_f, "Image layer:", self._preview_layer_combo,
            "Image layer to preview the correction on (single slice only).")

        slice_row = QHBoxLayout()
        self._preview_slice = self._spin(0, 9999, 0)
        self._preview_slice.setToolTip("B-scan slice index to preview (0-indexed).")
        self._preview_btn = QPushButton("Preview Slice")
        self._preview_btn.setFixedHeight(30)
        style_secondary_btn(self._preview_btn)
        self._preview_btn.setToolTip(
            "Correct one slice and add it to the viewer immediately.")
        slice_row.addWidget(QLabel("Slice:")); slice_row.addWidget(self._preview_slice)
        slice_row.addSpacing(6); slice_row.addWidget(self._preview_btn)
        prev_f.addRow(slice_row)
        lay.addWidget(grp_prev)

        # ── Calibration Parameters ────────────────────────────────────
        grp_cal = QGroupBox("PanOCT Calibration Parameters")
        f_cal = QFormLayout(grp_cal); f_cal.setSpacing(6)

        self._pivot_dist = self._spin(5.0, 40.0, 16.0, 1)
        self._form_row(f_cal, "Pivot distance (mm):", self._pivot_dist,
            "Distance from virtual pivot to the top of the image.\n"
            "≈ lens-to-retina distance.  Adult ≈ 16 mm, infant ≈ 10–13 mm.")

        fast_row = QHBoxLayout()
        self._fast_max    = self._spin(10, 90, 49.85, 2)
        self._fast_offset = self._spin(-10, 10, -0.61, 2)
        self._fast_max.setToolTip("Max scan angle, fast axis (°). Default: 49.85°")
        self._fast_offset.setToolTip("Angle offset, fast axis (°). Default: −0.61°")
        fast_row.addWidget(QLabel("Max°:")); fast_row.addWidget(self._fast_max)
        fast_row.addSpacing(6); fast_row.addWidget(QLabel("Offset°:")); fast_row.addWidget(self._fast_offset)
        f_cal.addRow("Fast axis:", fast_row)

        slow_row = QHBoxLayout()
        self._slow_max    = self._spin(10, 90, 49.96, 2)
        self._slow_offset = self._spin(-10, 10, 0.27, 2)
        self._slow_max.setToolTip("Max scan angle, slow axis (°). Default: 49.96°")
        self._slow_offset.setToolTip("Angle offset, slow axis (°). Default: 0.27°")
        slow_row.addWidget(QLabel("Max°:")); slow_row.addWidget(self._slow_max)
        slow_row.addSpacing(6); slow_row.addWidget(QLabel("Offset°:")); slow_row.addWidget(self._slow_offset)
        f_cal.addRow("Slow axis:", slow_row)

        self._ref_index = self._spin(1.0, 2.0, 1.336, 3)
        self._form_row(f_cal, "Refractive index:", self._ref_index,
            "Average ocular media refractive index (air→tissue). Default: 1.336")

        self._axial_range = self._spin(1.0, 50.0, 12.0, 1)
        self._form_row(f_cal, "Axial range (mm):", self._axial_range,
            "Total axial imaging range in air. Default: 12 mm")

        lay.addWidget(grp_cal)

        self._enable_3d_interp = QCheckBox("Enable Full 3D Interpolation")
        self._enable_3d_interp.setChecked(True)
        self._enable_3d_interp.setToolTip(
            "Uses true 3D spherical mapping for the whole volume simultaneously.\n"
            "This interpolates smoothly between slices but uses ~15 GB RAM.\n"
            "If unchecked, processes 3D-coupled slices sequentially (lower memory)."
        )
        lay.addWidget(self._enable_3d_interp)

        # ── Buttons ───────────────────────────────────────────────────
        lay.addSpacing(4)
        self._run_btn = QPushButton("Apply Curve Correction")
        self._run_btn.setFixedHeight(44); style_primary_btn(self._run_btn)
        self._run_btn.setToolTip(
            "Correct all selected layers from fan-beam → Cartesian geometry.\n"
            "Run this BEFORE tumor detection for geometrically accurate volumes.")
        lay.addWidget(self._run_btn)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0); self._progress.setFixedHeight(5); self._progress.setVisible(False)
        lay.addWidget(self._progress)

        self._status_label = QLabel("")
        self._status_label.setVisible(False); self._status_label.setMinimumHeight(42)
        lay.addWidget(self._status_label)
        
        note_label = QLabel(
            "Note on Scan Angles: The paper specifies a 3rd-order polynomial "
            "for scan angle conversion. Because exact coefficients were unpublished, "
            "this plugin uses linear interpolation between the calibrated max bounds."
        )
        note_label.setWordWrap(True)
        note_label.setStyleSheet("color: #888888; font-size: 10px; margin-top: 10px;")
        lay.addWidget(note_label)
        
        lay.addStretch()

    # ──────────────────────────────────────────────────────────────────
    # Signals
    # ──────────────────────────────────────────────────────────────────

    def _connect_signals(self):
        self._run_btn.clicked.connect(self._on_run)
        self._preview_btn.clicked.connect(self._on_preview)
        self.viewer.layers.events.inserted.connect(self._refresh_layer_list)
        self.viewer.layers.events.removed.connect(self._refresh_layer_list)
        self._refresh_layer_list()

    def _refresh_layer_list(self, event=None):
        self._layer_list.clear()
        img_names = []
        for layer in self.viewer.layers:
            if isinstance(layer, (Image, Labels)):
                self._layer_list.addItem(QListWidgetItem(layer.name))
            if isinstance(layer, Image):
                img_names.append(layer.name)

        if self._layer_list.count() > 0:
            self._layer_list.item(0).setSelected(True)

        cur = self._preview_layer_combo.currentText()
        self._preview_layer_combo.blockSignals(True)
        self._preview_layer_combo.clear()
        self._preview_layer_combo.addItems(img_names)
        idx = self._preview_layer_combo.findText(cur)
        if idx >= 0: self._preview_layer_combo.setCurrentIndex(idx)
        lyr = self._get_layer(self._preview_layer_combo.currentText())
        if lyr is not None:
            self._preview_slice.setMaximum(max(0, lyr.data.shape[0] - 1))
        self._preview_layer_combo.blockSignals(False)

    def _build_params(self) -> CurveCorrectionParams:
        return CurveCorrectionParams(
            theta_max_fast_deg=self._fast_max.value(),
            theta_max_slow_deg=self._slow_max.value(),
            theta_offset_fast_deg=self._fast_offset.value(),
            theta_offset_slow_deg=self._slow_offset.value(),
            pivot_distance_mm=self._pivot_dist.value(),
            refractive_index=self._ref_index.value(),
            axial_range_mm=self._axial_range.value(),
        )

    def _get_layer(self, name):
        for l in self.viewer.layers:
            if l.name == name: return l
        return None

    def _set_running(self, running: bool):
        self._run_btn.setEnabled(not running)
        self._run_btn.setText("Correcting..." if running else "Apply Curve Correction")
        self._progress.setVisible(running)
        self._preview_btn.setEnabled(not running)

    # ──────────────────────────────────────────────────────────────────
    # Preview
    # ──────────────────────────────────────────────────────────────────

    def _on_preview(self):
        layer = self._get_layer(self._preview_layer_combo.currentText())
        if layer is None or not isinstance(layer, Image):
            show_info("Select an Image layer in the preview dropdown."); return

        vol = layer.data
        idx = min(self._preview_slice.value(), vol.shape[0] - 1)
        params = self._build_params()
        try:
            corrected = correct_bscan_2d(vol[idx], params, order=1)
        except Exception as e:
            show_warning(f"Preview failed: {e}"); return

        pname = f"{layer.name}_preview_s{idx}_CurveCorr"
        existing = self._get_layer(pname)
        if existing:
            existing.data = corrected
        else:
            self.viewer.add_image(corrected, name=pname, colormap=layer.colormap)

        set_result_info(self._status_label,
            f"Preview: slice {idx} corrected → '{pname}'")
        show_info(f"Preview added: '{pname}'")

    # ──────────────────────────────────────────────────────────────────
    # Full correction run
    # ──────────────────────────────────────────────────────────────────

    def _on_run(self):
        selected = self._layer_list.selectedItems()
        if not selected:
            show_info("Select at least one layer to correct."); return

        layers = [self._get_layer(it.text()) for it in selected]
        layers = [l for l in layers if l is not None]
        if not layers:
            show_info("No valid layers found."); return

        self._set_running(True); self._status_label.setVisible(False)
        job = dict(
            layers=layers,
            params=self._build_params(),
            full_3d=self._enable_3d_interp.isChecked()
        )
        worker = _run_correction_thread(job)
        worker.yielded.connect(self._on_progress)
        worker.returned.connect(self._on_finished)
        worker.errored.connect(self._on_error)
        worker.start()

    def _on_progress(self, msg: str):
        import re
        m = re.search(r'(\d+)\s*%', msg)
        if m:
            self._progress.setRange(0, 100); self._progress.setValue(int(m.group(1)))
        else:
            self._progress.setRange(0, 0)

    def _on_finished(self, results: list):
        self._set_running(False)
        for orig_name, is_labels, corrected, scale_yx in results:
            out_name = f"{orig_name}_CurveCorr"
            existing = self._get_layer(out_name)
            if is_labels:
                if existing:
                    existing.data = corrected
                    existing.scale[-2:] = scale_yx
                else:
                    lyr = self.viewer.add_labels(corrected, name=out_name)
                    lyr.scale[-2:] = scale_yx
            else:
                if existing:
                    existing.data = corrected
                    existing.scale[-2:] = scale_yx
                else:
                    lyr = self.viewer.add_image(corrected, name=out_name)
                    lyr.scale[-2:] = scale_yx

        n = len(results)
        set_result_success(self._status_label,
            f"{n} layer{'s' if n != 1 else ''} corrected — '_CurveCorr' suffix added.")
        show_info(f"Curve correction complete — {n} layer(s) added.")


    def _on_error(self, exc):
        self._set_running(False)
        set_result_error(self._status_label, f"Error: {exc}")
        show_warning(f"Curve correction failed: {exc}"); raise exc


# ──────────────────────────────────────────────────────────────────────────────
# Background thread
# ──────────────────────────────────────────────────────────────────────────────

@thread_worker
def _run_correction_thread(job: dict):
    import queue, threading
    from .curve_correction import correct_volume_3d_full
    pq = queue.Queue(); result_holder = [None, None]

    def _run():
        try:
            layers = job['layers']; params = job['params']
            use_full_3d = job.get('full_3d', False)
            results = []
            for i, layer in enumerate(layers):
                is_labels = isinstance(layer, Labels)
                pq.put(f"Correcting '{layer.name}' ({i+1}/{len(layers)})...")
                if use_full_3d:
                    corrected = correct_volume_3d_full(
                        layer.data, params, is_labels=is_labels,
                        progress_callback=lambda m: pq.put(m))
                else:
                    corrected = correct_volume_slicewise(
                        layer.data, params, is_labels=is_labels,
                        progress_callback=lambda m: pq.put(m))
                
                # Compute physical scale — pass D for 3D geometry
                D, H, W = layer.data.shape[-3], layer.data.shape[-2], layer.data.shape[-1]
                scale_yx = compute_output_scale(H, W, D, params)  # (z_res_mm, x_res_mm)
                results.append((layer.name, is_labels, corrected, scale_yx))
            result_holder[0] = results
        except Exception as e:
            result_holder[1] = e

    t = threading.Thread(target=_run, daemon=True); t.start()
    while t.is_alive():
        try: yield pq.get(timeout=0.1)
        except queue.Empty: pass
    while not pq.empty(): yield pq.get_nowait()
    if result_holder[1]: raise result_holder[1]
    return result_holder[0]
