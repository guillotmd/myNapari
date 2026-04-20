"""
Pipeline Diagnostic Script
==========================
Run this with a real .unp file to compare every step of the manual and batch
pipelines side by side.  It saves intermediate arrays so you can diff them
in napari or numpy.

Usage (from napari-env):
    python diagnose_pipeline.py /path/to/scan.unp /path/to/output/dir

The script will:
  1. Load metadata (headless)
  2. Compute dispersion coefficients using BOTH methods and compare
  3. Run process_unp and log stats at each intermediate step
  4. Run auto_contrast_split and log stats before/after
  5. Save intermediate .npy files for visual inspection in napari
  6. Compare against an existing manual .npy file if you supply one as arg 3
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch

# ── bootstrap napari viewer (headless) ──────────────────────────────────────
import napari
viewer = napari.Viewer(show=False)  # needed to satisfy napari imports

from napari_cool_tools_io import device, getWindow, unp_meta
from napari_cool_tools_io._unp_reader import unp_batch_proc_meta
from napari_cool_tools_io.process_unp import (
    dc_subtraction_double_sweep_torch,
    process_unp,
    reshuffle_vista_indices,
    set_dispersion_coefficients_torch,
    unpack12_torch,
)
from napari_cool_tools_oct_preproc._oct_preproc_func import auto_contrast_split

# ── helpers ──────────────────────────────────────────────────────────────────

def stats(arr, label):
    a = arr.numpy() if isinstance(arr, torch.Tensor) else arr
    print(f"  {label:40s}  min={a.min():.4f}  max={a.max():.4f}  "
          f"mean={a.mean():.4f}  std={a.std():.4f}  shape={a.shape}")


def sep(title=""):
    w = 70
    if title:
        pad = (w - len(title) - 2) // 2
        print("\n" + "─" * pad + f" {title} " + "─" * pad)
    else:
        print("\n" + "─" * w)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print("Usage: python diagnose_pipeline.py <scan.unp> <output_dir> [manual.npy]")
        sys.exit(1)

    unp_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    manual_npy = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  UNP FILE : {unp_path}")
    print(f"  OUTPUT   : {out_dir}")
    print(f"  DEVICE   : {device}")
    print(f"{'='*70}")

    # ── Step 0: Load metadata ─────────────────────────────────────────────
    sep("Step 0: Metadata")

    # Show what companion files actually exist in the folder
    unp_stem = unp_path.stem
    siblings = list(unp_path.parent.iterdir())
    print(f"  Folder: {unp_path.parent}")
    print(f"  Files with matching stem:")
    for f in sorted(siblings):
        marker = " ← (the UNP)" if f.suffix.lower() == ".unp" else ""
        if f.stem == unp_stem or f.name.startswith(unp_stem):
            print(f"    {f.name}{marker}")
    ini_path = unp_path.with_suffix(".ini")
    xml_path = unp_path.with_suffix(".xml")
    if not ini_path.exists() and not xml_path.exists():
        print(f"\n  ERROR: Neither {ini_path.name} nor {xml_path.name} found!")
        print(f"  Copy the .ini or .xml companion file into:")
        print(f"    {unp_path.parent}")
        sys.exit(1)

    meta = unp_batch_proc_meta(str(unp_path))
    if meta is None:
        print("ERROR: metadata file found but failed to parse — check .ini sections")
        sys.exit(1)

    # Apply defaults matching dialog
    meta.dcSubtract = True
    meta.desine = True
    meta.log_scale = False
    meta.full_range = False
    meta.windowType = 0        # Hamming
    meta.dispersion_mode = 0   # Global
    meta.split_dispersion = False
    meta.split_spectrum = False

    print(f"  width={meta.width}  height={meta.height}  depth={meta.depth}")
    print(f"  bmscan={meta.bmscan}  vista={meta.vista}  pattern={meta.pattern}")
    print(f"  double_side={meta.double_side} (from .ini)")
    print(f"  packed={meta.packed}")

    # ── Step 1: Build imageIndexing (dialog's set_unp_path logic) ────────
    sep("Step 1: imageIndexing (dialog logic)")
    imageIndexing = np.arange(meta.depth)
    if meta.bmscan > 1 and meta.vista > 1:
        imageIndexing = reshuffle_vista_indices(imageIndexing, meta.vista, meta.bmscan)
        print(f"  Applied vista reshuffle (vista={meta.vista}, bmscan={meta.bmscan})")
    if meta.pattern == "Sine_Pause" and meta.sine_frame_indices:
        start_p = meta.sine_frame_indices[0::2]
        stop_p  = meta.sine_frame_indices[1::2]
        pause   = np.concatenate([np.arange(s, e) for s, e in zip(start_p, stop_p)])
        imageIndexing = imageIndexing[~np.isin(imageIndexing, pause)]
        print(f"  Sine_Pause: removed {len(pause)} pause frames, {len(imageIndexing)} remain")

    dialog_ref_idx = math.ceil(len(imageIndexing) / 2)
    dialog_ref_frame = imageIndexing[dialog_ref_idx]
    process_unp_ref_frame = int(meta.depth / 2)

    print(f"  imageIndexing length : {len(imageIndexing)}")
    print(f"  DIALOG ref idx       : {dialog_ref_idx} → raw frame {dialog_ref_frame}")
    print(f"  process_unp ref frame: {process_unp_ref_frame} (raw byte offset)")
    match = "✓ SAME" if dialog_ref_frame == process_unp_ref_frame else "✗ DIFFERENT — this causes dispersion mismatch!"
    print(f"  Reference frame      : {match}")

    # ── Step 2: Load reference frame raw data ────────────────────────────
    sep("Step 2: Load reference frame")
    if meta.packed:
        data_size_bytes = int(1.5 * meta.width * meta.height)
    else:
        data_size_bytes = 2 * meta.width * meta.height

    with open(unp_path, "rb", buffering=0) as f:
        f.seek(int(data_size_bytes) * int(dialog_ref_frame), 0)
        if meta.packed:
            raw_bytes = f.read(data_size_bytes)
            arr = np.frombuffer(raw_bytes, dtype="<u1")
            arr = torch.tensor(arr).to(device)
            raw = unpack12_torch(arr).reshape((meta.height, meta.width))
        else:
            raw_bytes = f.read(data_size_bytes)
            arr = np.frombuffer(raw_bytes, dtype=np.uint16)
            raw = torch.tensor(arr.reshape((meta.height, meta.width)).astype(np.float32)).to(device)

    stats(raw, "raw frame data")

    # ── Step 3: DC subtract + window ─────────────────────────────────────
    sep("Step 3: DC subtract + Hamming window")
    if meta.dcSubtract:
        subtracted = dc_subtraction_double_sweep_torch(raw)
        stats(subtracted, "after DC subtract")
    else:
        subtracted = raw
        print("  DC subtract: SKIPPED")

    hamming = getWindow(meta.width, meta.windowType, subtracted.dtype, subtracted.device)
    windowed = subtracted * hamming
    stats(windowed, "after Hamming window")

    # ── Step 4: Dispersion (DIALOG METHOD) ───────────────────────────────
    sep("Step 4: Dispersion coefficients")
    print("  Computing with dialog-matched reference frame...")
    coeffs_dialog = set_dispersion_coefficients_torch(
        windowed, maxDispOrders=3, coefRange=100, dispersion_mode=meta.dispersion_mode
    )
    c2_dialog, c3_dialog = int(coeffs_dialog[0]), int(coeffs_dialog[1])
    print(f"  Dialog method   : c2={c2_dialog}, c3={c3_dialog}")

    # Also compute what process_unp's auto_dispersion would use (different frame)
    with open(unp_path, "rb", buffering=0) as f:
        f.seek(int(data_size_bytes) * process_unp_ref_frame, 0)
        if meta.packed:
            raw_bytes2 = f.read(data_size_bytes)
            arr2 = torch.tensor(np.frombuffer(raw_bytes2, dtype="<u1")).to(device)
            raw2 = unpack12_torch(arr2).reshape((meta.height, meta.width))
        else:
            raw_bytes2 = f.read(data_size_bytes)
            arr2 = np.frombuffer(raw_bytes2, dtype=np.uint16)
            raw2 = torch.tensor(arr2.reshape((meta.height, meta.width)).astype(np.float32)).to(device)

    if meta.dcSubtract:
        sub2 = dc_subtraction_double_sweep_torch(raw2)
    else:
        sub2 = raw2
    win2 = sub2 * hamming

    coeffs_proc = set_dispersion_coefficients_torch(
        win2, maxDispOrders=3, coefRange=100, dispersion_mode=meta.dispersion_mode
    )
    c2_proc, c3_proc = int(coeffs_proc[0]), int(coeffs_proc[1])
    print(f"  process_unp method: c2={c2_proc}, c3={c3_proc}")
    if c2_dialog == c2_proc and c3_dialog == c3_proc:
        print("  ✓ Coefficients match")
    else:
        print(f"  ✗ Mismatch! delta c2={c2_dialog - c2_proc}, delta c3={c3_dialog - c3_proc}")

    # ── Step 5: Run process_unp with dialog-matched coefficients ─────────
    sep("Step 5: process_unp (desine=True, auto_dispersion=False, using dialog c2/c3)")
    meta.c2A = c2_dialog
    meta.c3A = c3_dialog

    volume = process_unp(unp_path, meta, auto_dispersion=False)
    stats(volume, "volume after process_unp")
    np.save(str(out_dir / "step5_after_process_unp.npy"), volume)

    # ── Step 6: auto_contrast_split ───────────────────────────────────────
    sep("Step 6: auto_contrast_split (double_side=True)")
    tensor = torch.tensor(volume, dtype=torch.float32).to(device)
    stats(tensor, "input to auto_contrast_split")

    # pre-flip (double_side)
    tensor_flipped = tensor.clone()
    tensor_flipped[:, :, 1::2] = torch.flip(tensor_flipped[:, :, 1::2], dims=[2])
    stats(tensor_flipped, "after pre-flip")

    result = auto_contrast_split(tensor_flipped, 1.0, 99.0, 1.0, 99.0, 1)
    stats(result, "after auto_contrast_split (before post-flip)")

    # post-flip
    result[:, :, 1::2] = torch.flip(result[:, :, 1::2], dims=[2])
    stats(result, "after post-flip")

    out_arr = result.cpu().numpy()
    print(f"\n  NOTE: auto_contrast_split multiplies by mmax (center frame peak)")
    center = tensor_flipped.shape[0] // 2
    mmax = float(tensor_flipped[center].max())
    print(f"  mmax = {mmax:.4f}  →  output range ≈ [0, {mmax:.4f}]")

    # ── Step 7: Final [0,1] normalize ────────────────────────────────────
    sep("Step 7: Normalize to [0, 1]")
    out_min, out_max = float(out_arr.min()), float(out_arr.max())
    print(f"  Pre-normalize  range: [{out_min:.4f}, {out_max:.4f}]")
    if out_max > out_min:
        out_arr = (out_arr - out_min) / (out_max - out_min)
    stats(out_arr, "final batch output")
    np.save(str(out_dir / "step7_final_batch.npy"), out_arr)

    # ── Step 8: Compare against manual NPY if provided ───────────────────
    if manual_npy and manual_npy.exists():
        sep("Step 8: Compare vs manual NPY")
        manual = np.load(str(manual_npy))
        stats(manual, "manual NPY")
        diff = np.abs(manual - out_arr)
        print(f"\n  Max absolute difference : {diff.max():.6f}")
        print(f"  Mean absolute difference: {diff.mean():.6f}")
        print(f"  Pixel-wise correlation  : {np.corrcoef(manual.flatten(), out_arr.flatten())[0,1]:.6f}")
        np.save(str(out_dir / "step8_diff_vs_manual.npy"), diff)

        if diff.max() < 0.01:
            print("\n  ✓ Files match within 1% — pipeline is correct!")
        elif diff.max() < 0.1:
            print("\n  ⚠ Small differences — likely dispersion coefficient rounding")
        else:
            print("\n  ✗ Large differences — pipeline mismatch, check steps above")
    else:
        print("\n  (No manual NPY provided for comparison)")

    sep("Done")
    print(f"  Intermediate files saved to: {out_dir}")
    print(f"    step5_after_process_unp.npy")
    print(f"    step7_final_batch.npy")
    if manual_npy:
        print(f"    step8_diff_vs_manual.npy")
    print()


if __name__ == "__main__":
    main()
