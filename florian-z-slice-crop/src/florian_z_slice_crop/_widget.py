"""
_widget.py — Widget to crop edge slices from 3D volumes/masks.
"""
from __future__ import annotations

import numpy as np
from napari.layers import Image, Labels
from napari.utils.notifications import show_info, show_warning
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget
)

from florian_z_slice_crop._style import (
    STYLESHEET, make_plugin_header, make_separator,
    style_primary_btn, set_result_success, set_result_error
)

class ZSliceCropWidget(QWidget):
    """Crops slices from the Z-axis (axis 0) of 3D layers."""

    def __init__(self, napari_viewer, parent=None):
        super().__init__(parent)
        self.viewer = napari_viewer
        self.setStyleSheet(STYLESHEET)
        self._build_ui()
        self._connect_signals()

    def _spin(self, lo, hi, val):
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setValue(int(val))
        return w

    def _form_row(self, form, label, widget, tip):
        lbl = QLabel(label)
        lbl.setToolTip(tip)
        widget.setToolTip(tip)
        form.addRow(lbl, widget)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(10, 6, 10, 16)
        lay.setSpacing(8)
        
        scroll.setWidget(inner)
        root.addWidget(scroll)

        lay.addWidget(make_plugin_header(
            "✂️", "Crop Z-Slices",
            "Remove edge slices from 3D OCT volumes or masks (e.g., fundus projections at the end of the scan)."
        ))
        lay.addWidget(make_separator())

        # ── Input ─────────────────────────────────────────────────────
        grp_in = QGroupBox("Input Layer")
        f_in = QFormLayout(grp_in)
        f_in.setSpacing(6)

        self._layer_combo = QComboBox()
        self._form_row(f_in, "Select Layer:", self._layer_combo,
            "Select the Image or Labels layer to crop.")
            
        lay.addWidget(grp_in)

        # ── Cropping Options ──────────────────────────────────────────
        grp_crop = QGroupBox("Crop Parameters")
        f_crop = QFormLayout(grp_crop)
        f_crop.setSpacing(6)
        
        self._crop_start = self._spin(0, 500, 0)
        self._form_row(f_crop, "Remove from Start:", self._crop_start,
            "Number of slices to remove from the very beginning of the volume (Z-axis).")
            
        self._crop_end = self._spin(0, 500, 6)
        self._form_row(f_crop, "Remove from End:", self._crop_end,
            "Number of slices to remove from the end of the volume (Z-axis).\nDefault is 6 for typical fundus projection artifacts.")

        lay.addWidget(grp_crop)

        # ── Buttons ───────────────────────────────────────────────────
        lay.addSpacing(4)
        
        self._run_btn = QPushButton("Crop Layer")
        self._run_btn.setFixedHeight(44)
        style_primary_btn(self._run_btn)
        self._run_btn.setToolTip("Create a new layer with the specified Z-slices removed.")
        lay.addWidget(self._run_btn)

        self._result_label = QLabel("")
        self._result_label.setAlignment(Qt.AlignCenter)
        self._result_label.setWordWrap(True)
        self._result_label.setVisible(False)
        self._result_label.setMinimumHeight(42)
        lay.addWidget(self._result_label)
        
        lay.addStretch()

    # ──────────────────────────────────────────────────────────────────
    # Signals
    # ──────────────────────────────────────────────────────────────────

    def _connect_signals(self):
        self._run_btn.clicked.connect(self._on_run)
        self.viewer.layers.events.inserted.connect(self._refresh_combos)
        self.viewer.layers.events.removed.connect(self._refresh_combos)
        self._refresh_combos()

    def _refresh_combos(self, event=None):
        layers = ["<None>"] + [l.name for l in self.viewer.layers if isinstance(l, (Image, Labels))]

        cur = self._layer_combo.currentText()
        self._layer_combo.blockSignals(True)
        self._layer_combo.clear()
        self._layer_combo.addItems(layers)
        idx = self._layer_combo.findText(cur)
        if idx >= 0:
            self._layer_combo.setCurrentIndex(idx)
        self._layer_combo.blockSignals(False)

    def _get_layer(self, name):
        for l in self.viewer.layers:
            if l.name == name:
                return l
        return None

    # ──────────────────────────────────────────────────────────────────
    # Run
    # ──────────────────────────────────────────────────────────────────

    def _on_run(self):
        layer_name = self._layer_combo.currentText()
        layer = self._get_layer(layer_name)

        if layer is None:
            show_warning("Please select a valid layer to crop.")
            return

        crop_s = self._crop_start.value()
        crop_e = self._crop_end.value()

        if crop_s == 0 and crop_e == 0:
            show_warning("Both start and end crop values are 0. Nothing to do.")
            return

        try:
            data = layer.data
            original_shape = data.shape
            
            if len(original_shape) < 3:
                show_warning(f"Layer '{layer_name}' has only {len(original_shape)} dimensions. Expected at least 3D volume.")
                return

            end_idx = -crop_e if crop_e > 0 else None
            cropped_data = data[crop_s:end_idx]
            
            if cropped_data.shape[0] == 0:
                show_warning("Cropping parameters would remove all slices. Aborting.")
                return

            new_name = f"{layer_name}_Cropped"
            
            if isinstance(layer, Image):
                self.viewer.add_image(
                    cropped_data,
                    name=new_name,
                    colormap=layer.colormap,
                    scale=layer.scale,
                    metadata=layer.metadata.copy()
                )
            elif isinstance(layer, Labels):
                self.viewer.add_labels(
                    cropped_data,
                    name=new_name,
                    scale=layer.scale,
                    metadata=layer.metadata.copy()
                )

            layer.visible = False
            
            set_result_success(self._result_label, f"✓ Cropped {original_shape[0]} → {cropped_data.shape[0]} slices")
            show_info(f"Created '{new_name}' ({cropped_data.shape[0]} slices)")
            
        except Exception as e:
            set_result_error(self._result_label, f"Error: {e}")
            show_warning(f"Crop failed: {e}")

