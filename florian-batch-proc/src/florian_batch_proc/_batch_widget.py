"""
Modern Qt widget for batch processing UNP files in napari.

Provides folder pickers, auto-discovery of .unp files, processing
options, and real-time progress tracking with a styled dark UI.
"""

from pathlib import Path

from qtpy.QtCore import Qt, QThread
from qtpy.QtGui import QFont, QColor
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QLineEdit,
    QCheckBox,
    QComboBox,
    QSpinBox,
    QProgressBar,
    QTextEdit,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QFrame,
    QSizePolicy,
)

# Updated import to florian_batch_proc
from florian_batch_proc._batch_processor import (
    BatchProcessorWorker,
    BatchSettings,
    discover_unp_files,
)

# ──────────────────────────────────────────────────────────────────────
# Stylesheet — dark theme matching napari's aesthetics
# ──────────────────────────────────────────────────────────────────────
STYLESHEET = """
QGroupBox {
    font-weight: bold;
    font-size: 13px;
    border: 1px solid #555;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 14px;
    color: #ddd;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}
QPushButton {
    background-color: #3a3f47;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 6px 14px;
    color: #eee;
    font-size: 12px;
}
QPushButton:hover {
    background-color: #4a5060;
    border-color: #7aa2f7;
}
QPushButton:pressed {
    background-color: #2a2f37;
}
QPushButton:disabled {
    background-color: #2a2d33;
    color: #666;
}
QPushButton#startButton {
    background-color: #2d6a4f;
    font-weight: bold;
    font-size: 13px;
    padding: 8px 20px;
}
QPushButton#startButton:hover {
    background-color: #3d8a6f;
}
QPushButton#startButton:disabled {
    background-color: #1a3a2a;
    color: #555;
}
QPushButton#cancelButton {
    background-color: #6a2d2d;
    font-weight: bold;
}
QPushButton#cancelButton:hover {
    background-color: #8a3d3d;
}
QLineEdit {
    background-color: #2a2d33;
    border: 1px solid #555;
    border-radius: 3px;
    padding: 4px 8px;
    color: #ddd;
}
QComboBox {
    background-color: #2a2d33;
    border: 1px solid #555;
    border-radius: 3px;
    padding: 3px 8px;
    color: #ddd;
}
QSpinBox {
    background-color: #2a2d33;
    border: 1px solid #555;
    border-radius: 3px;
    padding: 3px 6px;
    color: #ddd;
}
QListWidget {
    background-color: #1e2127;
    border: 1px solid #444;
    border-radius: 4px;
    color: #ccc;
    font-size: 11px;
}
QListWidget::item {
    padding: 3px 6px;
}
QListWidget::item:selected {
    background-color: #3a4a5a;
}
QTextEdit {
    background-color: #1a1d22;
    border: 1px solid #444;
    border-radius: 4px;
    color: #b0b8c0;
    font-family: "JetBrains Mono", "Cascadia Code", "Consolas", monospace;
    font-size: 11px;
}
QProgressBar {
    border: 1px solid #555;
    border-radius: 4px;
    text-align: center;
    color: #eee;
    background-color: #2a2d33;
    font-size: 11px;
}
QProgressBar::chunk {
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #2d6a4f, stop:1 #52b788
    );
    border-radius: 3px;
}
QCheckBox {
    color: #ccc;
    spacing: 6px;
}
QLabel {
    color: #bbb;
}
"""


class BatchUNPWidget(QWidget):
    """
    Napari dockable widget for batch processing UNP → NPY/NPZ.
    """

    def __init__(self, napari_viewer=None):
        super().__init__()
        self.viewer = napari_viewer
        self.setStyleSheet(STYLESHEET)
        self._worker = None
        self._thread = None
        self._unp_files: list[Path] = []

        self._build_ui()

    # ──────────────────────────────────────────────────────────────────
    # UI Construction
    # ──────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(6)

        # Title
        title = QLabel("Batch UNP Processor")
        title.setFont(QFont("Segoe UI", 15, QFont.Bold))
        title.setStyleSheet("color: #7aa2f7; margin-bottom: 2px;")
        title.setAlignment(Qt.AlignCenter)
        root_layout.addWidget(title)

        subtitle = QLabel("Process UNP files → NPY / NPZ")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #666; font-size: 11px; margin-bottom: 6px;")
        root_layout.addWidget(subtitle)

        # Splitter for top (config) and bottom (log)
        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)

        # ── Top section ──
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(6)

        # Folder pickers
        top_layout.addWidget(self._build_folder_section())

        # File list
        top_layout.addWidget(self._build_file_list_section())

        # Processing options
        top_layout.addWidget(self._build_options_section())

        # Output options
        top_layout.addWidget(self._build_output_section())

        splitter.addWidget(top_widget)

        # ── Bottom section (progress + log) ──
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(4)

        bottom_layout.addWidget(self._build_progress_section())
        splitter.addWidget(bottom_widget)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        root_layout.addWidget(splitter, 1)

        # Action buttons
        root_layout.addWidget(self._build_action_buttons())

    def _build_folder_section(self) -> QGroupBox:
        group = QGroupBox("Folders")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)

        # Input folder
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Input:"))
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText("Select folder containing .unp files...")
        self.input_path_edit.setReadOnly(True)
        row1.addWidget(self.input_path_edit, 1)
        btn_input = QPushButton("Browse…")
        btn_input.clicked.connect(self._browse_input)
        row1.addWidget(btn_input)
        layout.addLayout(row1)

        # Output folder
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Output:"))
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("Select output folder...")
        self.output_path_edit.setReadOnly(True)
        row2.addWidget(self.output_path_edit, 1)
        btn_output = QPushButton("Browse…")
        btn_output.clicked.connect(self._browse_output)
        row2.addWidget(btn_output)
        layout.addLayout(row2)

        return group

    def _build_file_list_section(self) -> QGroupBox:
        group = QGroupBox("Discovered Files")
        layout = QVBoxLayout(group)

        self.file_count_label = QLabel("No folder selected")
        self.file_count_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.file_count_label)

        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(120)
        self.file_list.setAlternatingRowColors(True)
        layout.addWidget(self.file_list)

        return group

    def _build_options_section(self) -> QGroupBox:
        group = QGroupBox("Processing Options")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)

        # Row 1: DC subtract + Double side + Full range
        row1 = QHBoxLayout()

        self.chk_dc = QCheckBox("DC Subtraction")
        self.chk_dc.setChecked(True)  # ✅ matches manual (enabled by default in dialog)
        self.chk_dc.setToolTip(
            "✅ ON by default — matches your manual workflow.\n\n"
            "Removes the DC background signal from each A-scan by subtracting\n"
            "the mean spectrum. Reduces the bright band at the top of B-scans.\n"
            "The dialog has this checked by default."
        )
        row1.addWidget(self.chk_dc)

        self.chk_double_side = QCheckBox("Double-Sided")
        self.chk_double_side.setChecked(True)  # ✅ matches manual
        self.chk_double_side.setToolTip(
            "✅ ON by default — matches your manual workflow.\n\n"
            "Flips every other B-scan horizontally to correct for the bidirectional\n"
            "scanning pattern (forward + reverse sweeps). Required for all standard\n"
            "bidirectional OCT scans."
        )
        row1.addWidget(self.chk_double_side)

        self.chk_full_range = QCheckBox("Full Range")
        self.chk_full_range.setChecked(False)  # ❌ off by default (matches manual)
        self.chk_full_range.setToolTip(
            "❌ OFF by default — matches your manual workflow.\n\n"
            "When OFF: only the negative-frequency half of the FFT is kept,\n"
            "giving the standard OCT depth image.\n"
            "When ON: the full FFT output is kept (both halves), doubling the\n"
            "depth range. Rarely used for standard scanning."
        )
        row1.addWidget(self.chk_full_range)
        row1.addStretch()
        layout.addLayout(row1)

        # Row 2: Desine + scale factor + log scale
        row2 = QHBoxLayout()

        self.chk_desine = QCheckBox("Desine")
        self.chk_desine.setChecked(True)  # ✅ matches manual (you check this in the dialog)
        self.chk_desine.setToolTip(
            "✅ ON by default — matches your manual workflow.\n\n"
            "Corrects the non-uniform (sine-wave) spacing of A-scans caused by\n"
            "the resonant scanner mirror. Resamples the B-scan to uniform spatial\n"
            "spacing using bilinear interpolation.\n"
            "This is the 'Desine' checkbox in the UNP dialog."
        )
        row2.addWidget(self.chk_desine)

        row2.addWidget(QLabel("Scale:"))
        self.spin_desine_scale = QSpinBox()
        self.spin_desine_scale.setRange(1, 8)
        self.spin_desine_scale.setValue(2)
        self.spin_desine_scale.setFixedWidth(60)
        self.spin_desine_scale.setToolTip(
            "Upsampling factor used internally during the desine resampling.\n"
            "Higher values = slightly better interpolation quality but more memory.\n"
            "Default: 2 (matches the dialog default)."
        )
        row2.addWidget(self.spin_desine_scale)

        self.chk_log_scale = QCheckBox("Log Scale")
        self.chk_log_scale.setChecked(False)  # ❌ off (matches manual)
        self.chk_log_scale.setToolTip(
            "❌ OFF by default — matches your manual workflow.\n\n"
            "When ON: applies 20·log10 to the OCT magnitude before saving,\n"
            "compressing the dynamic range into dB units.\n"
            "When OFF: raw linear magnitude is saved (what you save manually)."
        )
        row2.addWidget(self.chk_log_scale)
        row2.addStretch()
        layout.addLayout(row2)

        # Row 3: Auto-dispersion + range
        row3 = QHBoxLayout()

        self.chk_auto_disp = QCheckBox("Auto Dispersion")
        self.chk_auto_disp.setChecked(True)  # ✅ matches manual ('Auto Compensate' button)
        self.chk_auto_disp.setToolTip(
            "✅ ON by default — matches your manual workflow.\n\n"
            "Automatically finds the best dispersion compensation coefficients\n"
            "(c2 and c3) by searching for the sharpest B-scan using entropy.\n"
            "This replicates clicking 'Auto Compensate' in the UNP dialog."
        )
        row3.addWidget(self.chk_auto_disp)

        row3.addWidget(QLabel("Range:"))
        self.spin_disp_range = QSpinBox()
        self.spin_disp_range.setRange(10, 500)
        self.spin_disp_range.setValue(100)
        self.spin_disp_range.setFixedWidth(70)
        self.spin_disp_range.setToolTip(
            "Search range for dispersion coefficients (±N integer steps).\n"
            "Larger range finds a solution for more extreme dispersion mismatches\n"
            "but takes longer. Default: 100 (matches the dialog default)."
        )
        row3.addWidget(self.spin_disp_range)
        row3.addStretch()
        layout.addLayout(row3)

        # Row 4: Window type + dispersion mode
        row4 = QHBoxLayout()

        row4.addWidget(QLabel("Window:"))
        self.combo_window = QComboBox()
        self.combo_window.addItems(["Hamming", "Tukey", "Gaussian"])
        self.combo_window.setCurrentIndex(0)
        self.combo_window.setToolTip(
            "Spectral window applied to the raw A-scan before FFT.\n"
            "Reduces side-lobes (ringing artefacts) in the depth profile.\n\n"
            "• Hamming — standard choice, good balance (dialog default)\n"
            "• Tukey — flat-top with tapered edges, preserves signal amplitude\n"
            "• Gaussian — smoothest roll-off, most side-lobe suppression\n\n"
            "Default: Hamming (matches the dialog default)."
        )
        row4.addWidget(self.combo_window)

        row4.addWidget(QLabel("Dispersion:"))
        self.combo_disp_mode = QComboBox()
        self.combo_disp_mode.addItems(["Global", "Quadratic", "Sinusoidal", "Quad + Sine"])
        self.combo_disp_mode.setCurrentIndex(0)
        self.combo_disp_mode.setToolTip(
            "How dispersion compensation is applied across A-scans in a B-scan.\n\n"
            "• Global — same correction for every A-scan (fastest, dialog default)\n"
            "• Quadratic — correction varies quadratically along the B-scan\n"
            "• Sinusoidal — correction follows a sine curve along the B-scan\n"
            "• Quad + Sine — combination of quadratic and sinusoidal\n\n"
            "Default: Global (matches the dialog default)."
        )
        row4.addWidget(self.combo_disp_mode)
        row4.addStretch()
        layout.addLayout(row4)

        # Row 5: Auto contrast split (Initial Bscan Preprocessing)
        row5 = QHBoxLayout()
        self.chk_auto_contrast = QCheckBox("Auto Contrast Split  (Initial Bscan Preprocessing)")
        self.chk_auto_contrast.setChecked(True)  # ✅ matches manual
        self.chk_auto_contrast.setToolTip(
            "✅ ON by default — matches your manual workflow.\n\n"
            "After OCT reconstruction, applies 'Auto Contrast Image Split A/B'\n"
            "(the 'Initial Bscan Preprocessing' plugin) to normalize the intensity\n"
            "difference between the two interleaved beam directions (A and B scans).\n"
            "This is the step you run manually before saving the NPY file."
        )
        row5.addWidget(self.chk_auto_contrast)
        row5.addStretch()
        layout.addLayout(row5)

        return group

    def _build_output_section(self) -> QGroupBox:
        group = QGroupBox("Output Formats")
        layout = QHBoxLayout(group)

        self.chk_npy = QCheckBox("NPY (NumPy)")
        self.chk_npy.setChecked(True)
        layout.addWidget(self.chk_npy)

        self.chk_npz = QCheckBox("NPZ (Compressed)")
        self.chk_npz.setChecked(True)
        layout.addWidget(self.chk_npz)

        layout.addStretch()
        return group

    def _build_progress_section(self) -> QGroupBox:
        group = QGroupBox("Progress")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v / %m files")
        layout.addWidget(self.progress_bar)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("Processing log will appear here…")
        layout.addWidget(self.log_output, 1)

        return group

    def _build_action_buttons(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 4, 0, 0)

        self.btn_start = QPushButton("Start Processing")
        self.btn_start.setObjectName("startButton")
        self.btn_start.clicked.connect(self._start_processing)
        self.btn_start.setEnabled(False)
        layout.addWidget(self.btn_start, 1)

        self.btn_cancel = QPushButton("■  Cancel")
        self.btn_cancel.setObjectName("cancelButton")
        self.btn_cancel.clicked.connect(self._cancel_processing)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setFixedWidth(100)
        layout.addWidget(self.btn_cancel)

        return widget

    # ──────────────────────────────────────────────────────────────────
    # Folder browser callbacks
    # ──────────────────────────────────────────────────────────────────

    def _browse_input(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.input_path_edit.setText(folder)
            self._scan_folder(Path(folder))

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_path_edit.setText(folder)
            self._update_start_button()

    def _scan_folder(self, folder: Path):
        """Discover all .unp files recursively and populate the list."""
        self._unp_files = discover_unp_files(folder)
        self.file_list.clear()

        for f in self._unp_files:
            relative = f.relative_to(folder)
            size_mb = f.stat().st_size / (1024 * 1024)

            # Check for companion metadata
            has_ini = f.with_suffix(".ini").exists()
            has_xml = f.with_suffix(".xml").exists()
            meta_tag = ""
            if has_ini:
                meta_tag = " [.ini]"
            elif has_xml:
                meta_tag = " [.xml]"
            else:
                meta_tag = " [no meta!]"

            item = QListWidgetItem(f"  {relative}  —  {size_mb:.1f} MB{meta_tag}")
            if not has_ini and not has_xml:
                item.setForeground(QColor("#e06060"))
            else:
                item.setForeground(QColor("#80c080"))
            self.file_list.addItem(item)

        count = len(self._unp_files)
        self.file_count_label.setText(
            f"Found {count} .unp file{'s' if count != 1 else ''} in {folder.name}/"
        )
        self._update_start_button()

    def _update_start_button(self):
        has_input = len(self._unp_files) > 0
        has_output = bool(self.output_path_edit.text())
        has_format = self.chk_npy.isChecked() or self.chk_npz.isChecked()
        self.btn_start.setEnabled(has_input and has_output and has_format)

    # ──────────────────────────────────────────────────────────────────
    # Processing control
    # ──────────────────────────────────────────────────────────────────

    def _gather_settings(self) -> BatchSettings:
        return BatchSettings(
            input_folder=Path(self.input_path_edit.text()),
            output_folder=Path(self.output_path_edit.text()),
            dc_subtract=self.chk_dc.isChecked(),
            auto_dispersion=self.chk_auto_disp.isChecked(),
            dispersion_range=self.spin_disp_range.value(),
            desine=self.chk_desine.isChecked(),
            log_scale=self.chk_log_scale.isChecked(),
            double_side=self.chk_double_side.isChecked(),
            full_range=self.chk_full_range.isChecked(),
            window_type=self.combo_window.currentIndex(),
            dispersion_mode=self.combo_disp_mode.currentIndex(),
            init_preproc=self.chk_auto_contrast.isChecked(),
            save_npy=self.chk_npy.isChecked(),
            save_npz=self.chk_npz.isChecked(),
        )

    def _start_processing(self):
        settings = self._gather_settings()
        self.log_output.clear()
        self.progress_bar.setRange(0, len(self._unp_files))
        self.progress_bar.setValue(0)

        # Set up worker + thread
        self._thread = QThread()
        self._worker = BatchProcessorWorker(settings, self._unp_files)
        self._worker.moveToThread(self._thread)

        # Connect signals
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_message.connect(self._on_log)
        self._worker.file_completed.connect(self._on_file_completed)
        self._worker.all_completed.connect(self._on_all_completed)
        self._worker.cancelled.connect(self._on_cancelled)

        # Clean up thread when done
        self._worker.all_completed.connect(self._thread.quit)
        self._worker.cancelled.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)

        # UI state
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)

        self._thread.start()

    def _cancel_processing(self):
        if self._worker:
            self._worker.request_cancel()
            self.btn_cancel.setEnabled(False)
            self.log_output.append("\nRequesting cancellation...")

    def _on_progress(self, current: int, total: int):
        self.progress_bar.setValue(current)

    def _on_log(self, message: str):
        self.log_output.append(message)
        # Auto-scroll to bottom
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_file_completed(self, path: str, status: str):
        # Update the list item color
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if Path(path).name in item.text():
                if status == "ok":
                    item.setForeground(QColor("#52b788"))
                else:
                    item.setForeground(QColor("#e06060"))
                break

    def _on_all_completed(self, success: int, fail: int):
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)

    def _on_cancelled(self):
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)

    def _cleanup_thread(self):
        if self._thread:
            self._thread.deleteLater()
            self._thread = None
        if self._worker:
            self._worker.deleteLater()
            self._worker = None
