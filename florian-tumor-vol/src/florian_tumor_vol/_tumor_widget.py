"""
_tumor_widget.py
================
Napari QWidget for training-free OCT tumor volumetric measurement.

Workflow
--------
1. Select the OCT volume Image layer and the B-scan segmentation Labels layer.
2. (Optional) select an en-face ROI Labels layer to restrict processing to the
   tumor region drawn by the user on the en-face projection.
3. Adjust algorithm parameters if needed.
4. Click "Run Tumor Detection" — the tool:
   a. Extracts the anterior retinal surface from the segmentation.
   b. Fits a smoothing spline as a "healthy retina" baseline.
   c. Finds the deepest high-intensity pixels below the baseline.
   d. Builds a 3-D tumor mask (label value 3) and adds it to the viewer.
   e. Displays the tumor volume in mm³.
5. Optionally refine the mask manually using napari's built-in paint / erase
   tools on the resulting Labels layer, then click "Recalculate Volume"
   to recompute the volume without re-running the detection.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from napari.layers import Image, Labels
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
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

    # ──────────────────────────────────────────────────────────────────────────
    # UI Construction
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _add_row(form: "QFormLayout", label_text: str, widget, tooltip: str) -> None:
        """
        Add a labelled row to *form* and apply *tooltip* to BOTH the label
        and the widget so the tooltip appears whether the user hovers the
        label text or the control itself.
        """
        lbl = QLabel(label_text)
        lbl.setToolTip(tooltip)
        widget.setToolTip(tooltip)
        form.addRow(lbl, widget)

    def _build_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setAlignment(Qt.AlignTop)

        # Wrap everything in a scroll area so the widget is usable on small
        # screens even with all parameters visible
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setAlignment(Qt.AlignTop)
        scroll.setWidget(inner)
        root_layout.addWidget(scroll)

        # ── Layer selection ───────────────────────────────────────────────────
        layer_group = QGroupBox("Input Layers")
        layer_form = QFormLayout(layer_group)

        self._vol_combo = QComboBox()
        self._add_row(layer_form, "OCT Volume (Image):", self._vol_combo,
            "The 3-D OCT intensity volume (Image layer).\n"
            "All B-scans in this volume will be processed.")

        self._seg_combo = QComboBox()
        self._add_row(layer_form, "Segmentation (Labels):", self._seg_combo,
            "The B-scan segmentation Labels layer produced by the ONNX segmentation tool.\n"
            "Expected labels:  0 = vitreous,  1 = retina,  2 = choroid.")

        self._roi_combo = QComboBox()
        self._add_row(layer_form, "En-face ROI (Labels, opt.):", self._roi_combo,
            "Optional: a Labels layer drawn on the en-face projection marking the\n"
            "tumor region(s). Any non-zero label value counts as 'tumor area'.\n"
            "  • Columns inside the ROI are excluded from the healthy-baseline fit.\n"
            "  • Columns outside are used as the healthy reference for the spline.\n"
            "  • Multiple tumors are supported — paint them all on one layer.\n"
            "Recommended: always provide this for best results.")

        layout.addWidget(layer_group)

        # ── Algorithm parameters ──────────────────────────────────────────────
        algo_group = QGroupBox("Detection Parameters")
        algo_form = QFormLayout(algo_group)

        self._anterior_label = QSpinBox()
        self._anterior_label.setRange(0, 255)
        self._anterior_label.setValue(1)
        self._add_row(algo_form, "Retina label value:", self._anterior_label,
            "Label value that the ONNX segmentation assigns to the retina layer.\n"
            "Default: 1  (matches the standard ONNX B-scan model output).\n"
            "This surface is used to detect the elevated tumor dome.")

        self._baseline_label = QSpinBox()
        self._baseline_label.setRange(0, 255)
        self._baseline_label.setValue(2)
        self._add_row(algo_form, "Choroid baseline label:", self._baseline_label,
            "Label value that the ONNX segmentation assigns to the choroid layer.\n"
            "Default: 2.  The choroid represents the natural eye-wall curvature.\n"
            "It is naturally absent under the tumor, which automatically identifies\n"
            "the tumor region without extra input.")

        self._tumor_label = QSpinBox()
        self._tumor_label.setRange(1, 255)
        self._tumor_label.setValue(3)
        self._add_row(algo_form, "Tumor output label:", self._tumor_label,
            "Label value written into the output tumor mask Labels layer.\n"
            "Default: 3.  Change if 3 conflicts with another label in your project.")

        self._spline_smoothing = QDoubleSpinBox()
        self._spline_smoothing.setRange(0.0, 1e6)
        self._spline_smoothing.setDecimals(1)
        self._spline_smoothing.setValue(0.0)
        self._spline_smoothing.setSpecialValueText("Auto (scipy default)")
        self._add_row(algo_form, "Spline smoothing:", self._spline_smoothing,
            "Smoothing factor for the UnivariateSpline fitted through the healthy\n"
            "retinal and choroid surfaces.\n"
            "  0 / 'Auto' = interpolating (passes exactly through every anchor point).\n"
            "  Increase (e.g. 10–100) if the healthy surfaces are noisy and the\n"
            "  baseline looks jagged.")

        self._min_elevation = QDoubleSpinBox()
        self._min_elevation.setRange(0.0, 500.0)
        self._min_elevation.setDecimals(1)
        self._min_elevation.setValue(5.0)
        self._add_row(algo_form, "Min. elevation (px):", self._min_elevation,
            "Minimum number of pixels by which the actual retinal surface must be\n"
            "elevated above the expected healthy retinal baseline for a column to be\n"
            "counted as containing tumor.\n"
            "  • Too low  → noisy / spurious labels in healthy areas.\n"
            "  • Too high → small tumors or thin edges may be missed.\n"
            "Tip: set slightly above your typical retina-band thickness.")

        self._edge_margin = QSpinBox()
        self._edge_margin.setRange(0, 200)
        self._edge_margin.setValue(15)
        self._add_row(algo_form, "Edge margin (cols):", self._edge_margin,
            "Number of columns to blank inward from the ACTUAL DATA BOUNDARY of\n"
            "each B-scan (i.e. from where the retinal label starts/ends per slice,\n"
            "not from the image edge).\n\n"
            "The OCT volume is oval-shaped: B-scans at the start and end of the\n"
            "volume are narrower than those in the middle. This adaptive approach\n"
            "correctly handles the varying width without over-clipping wide slices\n"
            "or under-clipping narrow ones.\n\n"
            "  • Increase (e.g. 25–40) if edge artifacts still appear.\n"
            "  • Set to 0 to disable entirely.")

        self._margin_below = QDoubleSpinBox()
        self._margin_below.setRange(0.0, 500.0)
        self._margin_below.setDecimals(0)
        self._margin_below.setValue(60.0)
        self._add_row(algo_form, "Max depth margin (px):", self._margin_below,
            "Maximum number of pixels below the deepest observed choroid surface\n"
            "that the extrapolated baseline spline is allowed to reach.\n"
            "This prevents the baseline from 'diving' through or beyond the eye wall\n"
            "in B-scans where the healthy choroid flanks are narrow.\n"
            "  ▶▶ REDUCE THIS (e.g. to 30–40) if the baseline dips too far down. ◀◀")

        self._robust_sigma = QDoubleSpinBox()
        self._robust_sigma.setRange(0.1, 10.0)
        self._robust_sigma.setDecimals(1)
        self._robust_sigma.setValue(2.0)
        self._add_row(algo_form, "Robust sigma (no ROI):", self._robust_sigma,
            "Used only when NO en-face ROI layer is selected.\n"
            "After fitting an initial spline, columns where the choroid surface is\n"
            "more than (sigma × std) above the estimate are rejected as likely-tumor\n"
            "and excluded from the next fitting iteration.\n"
            "  • Lower  → more aggressive rejection (useful for large tumors).\n"
            "  • Higher → more permissive (fewer healthy columns excluded).\n"
            "Has no effect when an en-face ROI is provided.")

        self._robust_iters = QSpinBox()
        self._robust_iters.setRange(0, 5)
        self._robust_iters.setValue(2)
        self._add_row(algo_form, "Robust iters (no ROI):", self._robust_iters,
            "Number of iterative outlier-rejection passes for the automatic\n"
            "healthy-column detection (no-ROI mode only).\n"
            "  0 = single pass, no rejection.\n"
            "  2–3 = recommended when no ROI is available.\n"
            "Has no effect when an en-face ROI is provided.")

        layout.addWidget(algo_group)

        # ── Voxel / physical dimensions ───────────────────────────────────────
        vox_group = QGroupBox("Physical Voxel Size (mm)")
        vox_form = QFormLayout(vox_group)

        self._use_layer_scale = QCheckBox("Use layer scale attribute")
        self._use_layer_scale.setChecked(True)
        self._use_layer_scale.setToolTip(
            "When checked, the physical voxel size is read automatically from the\n"
            "napari Image layer's .scale property (set when the volume was loaded).\n"
            "Uncheck to enter the three voxel dimensions manually below."
        )
        vox_form.addRow(self._use_layer_scale)

        _vox_z_tip = (
            "Physical size of one voxel along the B-scan / slice (Z) axis in mm.\n"
            "This is the inter-B-scan spacing (slow scan axis).\n"
            "Only active when 'Use layer scale attribute' is unchecked."
        )
        self._vox_z = QDoubleSpinBox()
        self._vox_z.setRange(0.0001, 100.0)
        self._vox_z.setDecimals(4)
        self._vox_z.setValue(1.0)
        self._vox_z_lbl = QLabel("Voxel Z / depth (mm):")
        self._vox_z_lbl.setToolTip(_vox_z_tip)
        self._vox_z.setToolTip(_vox_z_tip)
        vox_form.addRow(self._vox_z_lbl, self._vox_z)

        _vox_y_tip = (
            "Physical size of one voxel along the A-scan / height (Y) axis in mm.\n"
            "This is the depth resolution within each B-scan.\n"
            "Only active when 'Use layer scale attribute' is unchecked."
        )
        self._vox_y = QDoubleSpinBox()
        self._vox_y.setRange(0.0001, 100.0)
        self._vox_y.setDecimals(4)
        self._vox_y.setValue(1.0)
        self._vox_y_lbl = QLabel("Voxel Y / height (mm):")
        self._vox_y_lbl.setToolTip(_vox_y_tip)
        self._vox_y.setToolTip(_vox_y_tip)
        vox_form.addRow(self._vox_y_lbl, self._vox_y)

        _vox_x_tip = (
            "Physical size of one voxel along the lateral / width (X) axis in mm.\n"
            "This is the A-scan spacing within each B-scan.\n"
            "Only active when 'Use layer scale attribute' is unchecked."
        )
        self._vox_x = QDoubleSpinBox()
        self._vox_x.setRange(0.0001, 100.0)
        self._vox_x.setDecimals(4)
        self._vox_x.setValue(1.0)
        self._vox_x_lbl = QLabel("Voxel X / width (mm):")
        self._vox_x_lbl.setToolTip(_vox_x_tip)
        self._vox_x.setToolTip(_vox_x_tip)
        vox_form.addRow(self._vox_x_lbl, self._vox_x)

        layout.addWidget(vox_group)

        # ── Action buttons ────────────────────────────────────────────────────
        self._run_btn = QPushButton("▶  Run Tumor Detection")
        self._run_btn.setFixedHeight(36)
        self._run_btn.setToolTip(
            "Run the full tumor detection pipeline on the selected layers.\n"
            "This runs in a background thread — napari stays responsive.\n\n"
            "Steps performed:\n"
            "  1. Extract the retinal and choroid surfaces per A-scan column.\n"
            "  2. Fit healthy-baseline splines from columns outside the ROI (or\n"
            "     using robust auto-detection if no ROI is provided).\n"
            "  3. Label columns where the retinal surface is elevated above the\n"
            "     expected baseline by at least 'Min. elevation (px)'.\n"
            "  4. Fill the tumor mask from the elevation dome down to the choroid\n"
            "     baseline for each affected column.\n"
            "  5. Compute and display the tumor volume in mm³."
        )
        layout.addWidget(self._run_btn)

        self._recalc_btn = QPushButton("🔢  Recalculate Volume (from current mask)")
        self._recalc_btn.setToolTip(
            "Recalculate the mm³ volume from the existing tumor mask Labels layer\n"
            "without re-running the detection algorithm.\n\n"
            "Use this after manually correcting the mask with napari's built-in\n"
            "paint (P) and erase tools to get an updated volume measurement."
        )
        layout.addWidget(self._recalc_btn)

        # ── Result label ──────────────────────────────────────────────────────
        self._result_label = QLabel("Volume: —")
        self._result_label.setAlignment(Qt.AlignCenter)
        self._result_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self._result_label)

        # Initial refresh of layer combos and sensitivity
        self._refresh_layer_combos()
        self._on_use_layer_scale_toggled(self._use_layer_scale.isChecked())

    # ──────────────────────────────────────────────────────────────────────────
    # Signal wiring
    # ──────────────────────────────────────────────────────────────────────────

    def _connect_signals(self):
        self._run_btn.clicked.connect(self._on_run)
        self._recalc_btn.clicked.connect(self._on_recalculate)
        self._use_layer_scale.toggled.connect(self._on_use_layer_scale_toggled)

        # Refresh combos when layers are added / removed
        self.viewer.layers.events.inserted.connect(self._refresh_layer_combos)
        self.viewer.layers.events.removed.connect(self._refresh_layer_combos)

    # ──────────────────────────────────────────────────────────────────────────
    # Layer combo helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _refresh_layer_combos(self, *_):
        """Repopulate all layer combo boxes from the current viewer state."""
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

    def _get_layer(self, name: str):
        for layer in self.viewer.layers:
            if layer.name == name:
                return layer
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Voxel-size helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _on_use_layer_scale_toggled(self, checked: bool):
        for w in (self._vox_z, self._vox_y, self._vox_x,
                  self._vox_z_lbl, self._vox_y_lbl, self._vox_x_lbl):
            w.setEnabled(not checked)

    def _get_voxel_size(self) -> tuple[float, float, float]:
        if self._use_layer_scale.isChecked():
            vol_layer = self._get_layer(self._vol_combo.currentText())
            if vol_layer is not None and hasattr(vol_layer, "scale"):
                s = vol_layer.scale
                if len(s) >= 3:
                    return float(s[-3]), float(s[-2]), float(s[-1])
            show_info("Could not read scale from layer — falling back to (1, 1, 1) mm.")
        return (self._vox_z.value(), self._vox_y.value(), self._vox_x.value())

    # ──────────────────────────────────────────────────────────────────────────
    # Run detection
    # ──────────────────────────────────────────────────────────────────────────

    def _on_run(self):
        vol_layer = self._get_layer(self._vol_combo.currentText())
        seg_layer = self._get_layer(self._seg_combo.currentText())

        if vol_layer is None or seg_layer is None:
            show_info("Please select valid Vol and Seg layers.")
            return

        roi_name = self._roi_combo.currentText()
        roi_layer = None if roi_name == "<None>" else self._get_layer(roi_name)

        vol_img = vol_layer.data
        params = dict(
            vol_img=vol_img,
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
            enface_roi_mask=self._build_roi_mask(roi_layer, vol_img.shape),
            voxel_size_mm=self._get_voxel_size(),
            vol_name=vol_layer.name,
        )

        self._run_btn.setEnabled(False)
        self._run_btn.setText("⏳  Running…")
        show_info("Tumor detection started…")

        worker = _run_detection_thread(**params)
        worker.returned.connect(self._on_detection_finished)
        worker.errored.connect(self._on_error)
        worker.start()
        self._worker = worker  # keep reference

    def _build_roi_mask(self, roi_layer, vol_shape) -> np.ndarray | None:
        """
        Convert a 2-D en-face Labels layer into a (D, W) boolean array
        aligned with the volume dimensions.

        The en-face is expected to be shape (D, W) — i.e. B-scan index × column
        (the projection along the Y / height axis). If the layer has a different
        shape, a warning is shown and None is returned.
        """
        if roi_layer is None:
            return None

        roi_data = roi_layer.data
        D, H, W = vol_shape

        if roi_data.ndim == 2 and roi_data.shape == (D, W):
            return roi_data > 0

        # Try squeezing an extra axis
        if roi_data.ndim == 3 and roi_data.shape[0] == 1:
            roi_data = roi_data[0]
            if roi_data.shape == (D, W):
                return roi_data > 0

        show_info(
            f"En-face ROI shape {roi_layer.data.shape} does not match expected "
            f"({D}, {W}). ROI ignored — processing full volume."
        )
        return None

    def _on_detection_finished(self, result):
        tumor_mask, volume_mm3, vol_name, tumor_label_val, voxel_size = result

        # Add or update the Labels layer
        layer_name = f"{vol_name}_tumor_mask"
        existing = self._get_layer(layer_name)
        if existing is not None:
            existing.data = tumor_mask
            self._tumor_layer = existing
        else:
            self._tumor_layer = self.viewer.add_labels(
                tumor_mask, name=layer_name
            )

        self._display_volume(volume_mm3, voxel_size)
        show_info(f"Tumor detection complete. Volume: {volume_mm3:.3f} mm³")
        self._run_btn.setEnabled(True)
        self._run_btn.setText("▶  Run Tumor Detection")

    def _on_error(self, exc):
        show_info(f"Tumor detection failed: {exc}")
        self._run_btn.setEnabled(True)
        self._run_btn.setText("▶  Run Tumor Detection")
        raise exc

    # ──────────────────────────────────────────────────────────────────────────
    # Recalculate volume from existing mask
    # ──────────────────────────────────────────────────────────────────────────

    def _on_recalculate(self):
        # Try to find a suitable tumor mask layer
        mask_layer = self._tumor_layer
        if mask_layer is None:
            # Fall back: look for any layer whose name ends with _tumor_mask
            for layer in self.viewer.layers:
                if isinstance(layer, Labels) and layer.name.endswith("_tumor_mask"):
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

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _display_volume(self, volume_mm3: float, voxel_size: tuple):
        vz, vy, vx = voxel_size
        self._result_label.setText(
            f"Volume: {volume_mm3:.4f} mm³\n"
            f"(voxel: {vz:.4f} × {vy:.4f} × {vx:.4f} mm)"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Background thread
# ──────────────────────────────────────────────────────────────────────────────

@thread_worker
def _run_detection_thread(
    vol_img,
    vol_labels,
    anterior_label_val,
    baseline_label_val,
    tumor_label_val,
    spline_smoothing,
    min_elevation_px,
    edge_margin_cols,
    margin_below_px,
    robust_sigma,
    robust_iters,
    enface_roi_mask,
    voxel_size_mm,
    vol_name,
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
    )
    return tumor_mask, volume_mm3, vol_name, tumor_label_val, voxel_size_mm
