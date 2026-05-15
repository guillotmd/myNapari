"""
Headless batch processing engine for UNP → NPY/NPZ conversion.

Replicates the full manual workflow exactly:

  Step 1 — UNP dialog (settings applied, then OK):
    • DC Subtraction (dcSubtract=True by default in dialog)
    • Double-Sided (from .ini Bidirectional field, default True in dialog)
    • Desine (user checks this manually → batch default True)
    • Auto Compensate (user clicks this → auto_dispersion=True)
      - Reference frame: math.ceil(len(imageIndexing) / 2)
        identical to how the dialog selects it in set_unp_path()
      - imageIndexing excludes Sine_Pause pause frames
    • Window: Hamming (index 0, dialog default)
    • Dispersion Mode: Global (index 0, dialog default)
    • Log Scale: False (dialog default, unchecked)
    • Full Range: False (dialog default, unchecked)
    • Split Dispersion: False (dialog default, unchecked)
    • Split Spectrum: False (dialog default, unchecked)

  Step 2 — Initial Bscan Preprocessing plugin (auto_contrast_split_thread):
    • double_side=True pre-flip → auto_contrast_split → post-flip
    • lower/upper percentile: 1.0 / 99.0
    • num_averages: 1

  Step 3 — Save layer as NPY and NPZ.
"""

import math
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from qtpy.QtCore import QObject, QThread, Signal
from tqdm import tqdm

from napari_cool_tools_io import device, getWindow, unp_meta
from napari_cool_tools_io._unp_reader import unp_batch_proc_meta
from napari_cool_tools_io.process_unp import (
    dc_subtraction_double_sweep_torch,
    process_unp,
    process_unp_sine_pause,
    reshuffle_vista_indices,
    set_dispersion_coefficients_torch,
    unpack12_torch,
)
from napari_cool_tools_img_proc._equalization_funcs import init_bscan_preproc_pt, DType


# ---------------------------------------------------------------------------
# Settings dataclass
# ---------------------------------------------------------------------------

@dataclass
class BatchSettings:
    """
    Settings applied uniformly to all UNP files in the batch.
    All defaults match what the user manually sets in the UNP dialog.
    """
    input_folder: Path = Path()
    output_folder: Path = Path()

    # ── Step 1: UNP → OCT volume ─────────────────────────────────────────
    dc_subtract: bool = True        # Dialog: dcSubtractCheckBox checked by default
    double_side: bool = True        # Overridden per-file from .ini Bidirectional
    desine: bool = True             # User checks Desine in dialog
    auto_dispersion: bool = True    # User clicks Auto Compensate
    dispersion_range: int = 100     # autoDispRangeSpinBox default = 100
    log_scale: bool = False
    full_range: bool = False
    window_type: int = 0            # Hamming
    dispersion_mode: int = 0        # Global
    split_dispersion: bool = False
    split_spectrum: bool = False

    # ── Step 2: Initial Bscan Preprocessing (init_bscan_preproc_pt) ──────
    # Plugin: "Initial Bscan Preprocessing" → creates "_InitPreproc" layer
    # Background removal (normalize → clamp subtract mean → normalize)
    # then auto-brightness adjust (clip mean + num_std*std → normalize to [0,1])
    init_preproc: bool = True           # Apply init_bscan_preproc_pt after process_unp
    init_preproc_num_std: int = 16      # Plugin default
    init_preproc_min_intensity: float = 0.0   # Plugin default
    init_preproc_max_intensity: float = 1.0   # Plugin default

    # ── Output ───────────────────────────────────────────────────────────
    save_npy: bool = True
    save_npz: bool = True


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def discover_unp_files(root_folder: Path) -> list[Path]:
    """Recursively find all .unp files under the root folder."""
    return sorted(root_folder.rglob("*.unp"))


def _apply_settings_to_meta(meta: unp_meta, settings: BatchSettings) -> unp_meta:
    """
    Apply batch settings to a metadata object.
    Note: meta.double_side is already set from the .ini file (Bidirectional field).
    We only override it if the user has explicitly changed it in the batch widget.
    """
    meta.dcSubtract = settings.dc_subtract
    meta.desine = settings.desine
    meta.log_scale = settings.log_scale
    meta.full_range = settings.full_range
    meta.windowType = settings.window_type
    meta.dispersion_mode = settings.dispersion_mode
    meta.split_dispersion = settings.split_dispersion
    meta.split_spectrum = settings.split_spectrum
    # double_side is read from .ini — we keep the file's value (matches dialog behavior
    # where the checkbox is set from meta.double_side in _unp_reader.py line 138)
    # Only override if user explicitly changed it:
    # meta.double_side = settings.double_side  # <- kept from .ini
    return meta


def _build_imageIndexing(meta: unp_meta) -> np.ndarray:
    """
    Build the imageIndexing array exactly as the dialog does in set_unp_path().
    This excludes Sine_Pause pause frames and applies vista reshuffling.

    Mirrors Unp_Preview_Widget.set_unp_path():
        self.imageIndexing = np.arange(meta.depth)
        if bmscan > 1 and vista > 1: reshuffle
        if Sine_Pause: remove pause indices
    """
    imageIndexing = np.arange(meta.depth)

    if meta.bmscan > 1 and meta.vista > 1:
        imageIndexing = reshuffle_vista_indices(imageIndexing, meta.vista, meta.bmscan)

    if meta.pattern == "Sine_Pause" and meta.sine_frame_indices:
        start_pause = meta.sine_frame_indices[0::2]
        stop_pause = meta.sine_frame_indices[1::2]
        pause_indices = np.concatenate([
            np.arange(s, e) for s, e in zip(start_pause, stop_pause)
        ])
        imageIndexing = imageIndexing[~np.isin(imageIndexing, pause_indices)]

    return imageIndexing


def _compute_dispersion_headless(
    unp_path: Path,
    meta: unp_meta,
    coef_range: int = 100,
) -> tuple[int, int]:
    """
    Compute dispersion coefficients (c2A, c3A) exactly as the dialog does:

    1. Build imageIndexing (same as dialog's set_unp_path)
    2. Reference frame = math.ceil(len(imageIndexing) / 2)  ← CRITICAL: matches dialog
    3. Load that specific raw frame from the .unp file
    4. DC subtract + window
    5. Run set_dispersion_coefficients_torch with the same coefRange

    This is the key fix: process_unp's auto_dispersion uses int(meta.depth/2)
    which is a raw byte offset and can differ significantly from the dialog's
    math.ceil(len(imageIndexing) / 2) reference frame.
    """
    # Build imageIndexing exactly as the dialog does
    imageIndexing = _build_imageIndexing(meta)

    # Reference frame index — exactly as dialog: math.ceil(len(...) / 2)
    ref_frame_idx = math.ceil(len(imageIndexing) / 2)
    ref_frame = imageIndexing[ref_frame_idx]

    if meta.packed:
        data_size_bytes = int(1.5 * meta.width * meta.height)
    else:
        data_size_bytes = 2 * meta.width * meta.height

    with open(unp_path, "rb", buffering=0) as f:
        f.seek(int(data_size_bytes) * int(ref_frame), 0)

        if meta.packed:
            raw_bytes = f.read(data_size_bytes)
            array = np.frombuffer(raw_bytes, dtype="<u1")
            if array.size != data_size_bytes:
                return 0, 0
            array = torch.tensor(array).to(device)
            array = unpack12_torch(array)
            raw = array.reshape((meta.height, meta.width))
        else:
            raw_bytes = f.read(data_size_bytes)
            array = np.frombuffer(raw_bytes, dtype=np.uint16)
            if array.size != meta.width * meta.height:
                return 0, 0
            raw = array.reshape((meta.height, meta.width)).astype(np.float32)
            raw = torch.tensor(raw).to(device)

    # DC subtract (same as dialog's autoDispersionFinder)
    if meta.dcSubtract:
        subtracted = dc_subtraction_double_sweep_torch(raw)
    else:
        subtracted = raw

    # Window (same as dialog)
    hamming = getWindow(meta.width, meta.windowType, subtracted.dtype, subtracted.device)
    hamming_signal = subtracted * hamming

    # Compute dispersion coefficients (same algorithm as dialog's autoDispersionFinder)
    coeffs = set_dispersion_coefficients_torch(
        hamming_signal,
        maxDispOrders=3,
        coefRange=coef_range,
        dispersion_mode=meta.dispersion_mode,
    )
    c2, c3 = coeffs
    return int(c2), int(c3)


def _build_output_path(unp_path: Path, input_root: Path, output_root: Path) -> Path:
    """
    Mirror the input directory structure under the output folder.
    e.g. input_root/subdir/scan.unp -> output_root/subdir/scan_BatchProcessed
    """
    relative = unp_path.parent.relative_to(input_root)
    out_dir = output_root / relative
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{unp_path.stem}_BatchProcessed"


def init_bscan_preproc_headless(
    data: np.ndarray,
    num_std: int = 16,
    min_intensity: float = 0.0,
    max_intensity: float = 1.0,
) -> np.ndarray:
    """
    Headless version of the 'Initial Bscan Preprocessing' plugin
    (init_bscan_preproc_plugin → init_bscan_preproc_pt).

    Exactly replicates the plugin's two-step pipeline:
      1. background_removal_pt:
            normalize to [0,1] → subtract global mean then clamp to [0,1]
            → normalize to [0,1] again
      2. auto_brightness_adjust_pt:
            compute mean and std of non-zero pixels
            → clip anything above (mean + num_std * std)
            → normalize to [min_intensity, max_intensity]

    Plugin defaults (match what the user confirmed in the UI):
        num_std=16, min_intensity=0.0, max_intensity=1.0, dtype=NP_FLOAT32

    Args:
        data:          numpy array (any shape) from process_unp
        num_std:       stddev multiplier for outlier clipping (default 16)
        min_intensity: output minimum (default 0.0)
        max_intensity: output maximum (default 1.0)

    Returns:
        Processed numpy float32 array in [min_intensity, max_intensity].
    """
    return init_bscan_preproc_pt(
        img=data,
        num_std=num_std,
        min_intensity=min_intensity,
        max_intensity=max_intensity,
        dtype=DType.NP_FLOAT32,
        use_accelerator=True,   # uses MPS/CUDA if available, else CPU
        numpy_out=True,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class BatchProcessorWorker(QObject):
    """
    Worker that processes UNP files in a background thread.
    Emits signals for progress tracking.
    """
    progress = Signal(int, int)       # (current_index, total)
    file_started = Signal(str)        # file path string
    file_completed = Signal(str, str) # (file path, "ok" or error message)
    log_message = Signal(str)
    all_completed = Signal(int, int)  # (success_count, fail_count)
    cancelled = Signal()

    def __init__(self, settings: BatchSettings, unp_files: list[Path]):
        super().__init__()
        self.settings = settings
        self.unp_files = unp_files
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    def run(self):
        """Main processing loop — called from a QThread."""
        total = len(self.unp_files)
        success_count = 0
        fail_count = 0

        steps = []
        if self.settings.auto_dispersion:
            steps.append("auto-dispersion (dialog-matched reference frame)")
        if self.settings.desine:
            steps.append("desine")
        if self.settings.init_preproc:
            steps.append("init-bscan-preproc (background removal + brightness adjust)")

        self.log_message.emit(
            f"Starting batch processing of {total} UNP file(s)...\n"
            f"  Device: {device}\n"
            f"  Pipeline: {' → '.join(steps) if steps else 'raw FFT only'}\n"
            f"  Output: {self.settings.output_folder}\n"
        )

        for idx, unp_path in enumerate(self.unp_files):
            if self._cancel_requested:
                self.log_message.emit("\n⚠ Batch processing cancelled by user.")
                self.cancelled.emit()
                return

            self.progress.emit(idx, total)
            self.file_started.emit(str(unp_path))
            self.log_message.emit(f"\n[{idx + 1}/{total}] Processing: {unp_path.name}")

            try:
                result = self._process_single_file(unp_path)
                if result:
                    success_count += 1
                    self.file_completed.emit(str(unp_path), "ok")
                    self.log_message.emit("  Saved successfully")
                else:
                    fail_count += 1
                    self.file_completed.emit(str(unp_path), "skipped: no metadata")
                    self.log_message.emit("  ✗ Skipped — no .ini/.xml metadata found")

            except Exception as e:
                fail_count += 1
                err_msg = f"{type(e).__name__}: {e}"
                self.file_completed.emit(str(unp_path), err_msg)
                self.log_message.emit(f"  ✗ Error: {err_msg}")
                self.log_message.emit(traceback.format_exc())

        self.progress.emit(total, total)
        self.log_message.emit(
            f"\n{'=' * 50}\n"
            f"Batch complete: {success_count} succeeded, {fail_count} failed\n"
        )
        self.all_completed.emit(success_count, fail_count)

    def _process_single_file(self, unp_path: Path) -> bool:
        """
        Process a single UNP file, exactly replicating the manual workflow.

        Returns True on success, False if no metadata found.
        """
        # ── Load metadata (headless, no Qt dialog) ──────────────────────
        meta = unp_batch_proc_meta(str(unp_path))
        if meta is None:
            return False

        # Apply batch settings (keeps meta.double_side from .ini)
        meta = _apply_settings_to_meta(meta, self.settings)

        self.log_message.emit(
            f"  Metadata: {meta.width}×{meta.height}, {meta.depth} frames, "
            f"bmscan={meta.bmscan}, pattern={meta.pattern}, "
            f"double_side={meta.double_side}"
        )

        # ── Auto dispersion: use dialog-matched reference frame ──────────
        if self.settings.auto_dispersion:
            self.log_message.emit("  Computing dispersion (dialog-matched frame)...")
            c2, c3 = _compute_dispersion_headless(
                unp_path, meta,
                coef_range=self.settings.dispersion_range,
            )
            meta.c2A = c2
            meta.c3A = c3
            self.log_message.emit(f"  Dispersion coefficients: c2={c2}, c3={c3}")

        # ── Step 1: UNP → OCT volume ─────────────────────────────────────
        if meta.pattern == "Sine_Pause":
            volume_data, volume_hires = process_unp_sine_pause(
                unp_path, meta, include_hires_in_lowres=True
            )
            volumes = [(volume_data, ""), (volume_hires, "_hires")]
        else:
            # auto_dispersion=False because we already computed and set coefficients above
            volume_data = process_unp(unp_path, meta, auto_dispersion=False)
            volumes = [(volume_data, "")]

        # ── Step 2: Initial Bscan Preprocessing ──────────────────────────
        if self.settings.init_preproc:
            self.log_message.emit("  Applying Initial Bscan Preprocessing (background removal + brightness adjust)...")
            processed = []
            for vol, suffix in volumes:
                vol = init_bscan_preproc_headless(
                    vol,
                    num_std=self.settings.init_preproc_num_std,
                    min_intensity=self.settings.init_preproc_min_intensity,
                    max_intensity=self.settings.init_preproc_max_intensity,
                )
                processed.append((vol, suffix))
            volumes = processed

        # ── Step 3: Save outputs ──────────────────────────────────────────
        for vol, suffix in volumes:
            self._save_volume(unp_path, vol, suffix=suffix)

        return True

    def _save_volume(self, unp_path: Path, data: np.ndarray, suffix: str = ""):
        """Save processed volume as NPY and/or NPZ."""
        out_base = _build_output_path(
            unp_path, self.settings.input_folder, self.settings.output_folder
        )

        if suffix:
            out_base = out_base.parent / f"{out_base.name}{suffix}"

        if self.settings.save_npy:
            npy_path = out_base.with_suffix(".npy")
            np.save(str(npy_path), data)
            self.log_message.emit(f"  → NPY: {npy_path.name} ({_format_size(npy_path)})")

        if self.settings.save_npz:
            npz_path = out_base.with_suffix(".npz")
            np.savez_compressed(str(npz_path), data=data)
            self.log_message.emit(f"  → NPZ: {npz_path.name} ({_format_size(npz_path)})")


def _format_size(path: Path) -> str:
    try:
        size = path.stat().st_size
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    except OSError:
        return "?"
