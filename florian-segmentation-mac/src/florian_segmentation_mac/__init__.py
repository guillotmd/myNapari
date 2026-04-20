"""
Mac-compatible segmentation plugin for napari.

Provides cross-platform device detection (CPU / MPS / CUDA) and
points to the same ONNX models as the original segmentation plugin.
"""

__version__ = "0.0.1"
__all__ = ()

import sys
import platform
from enum import Enum
from pathlib import Path

import napari
import torch

# ──────────────────────────────────────────────────────────────────────
# Cross-platform device detection
# ──────────────────────────────────────────────────────────────────────

def get_best_device() -> torch.device:
    """Detect the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clear_gpu_cache():
    """Safely clear GPU memory on any platform."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass  # MPS cache clearing may not be available on all versions


def get_device_name(dev: torch.device) -> str:
    """Get a human-readable name for the current device."""
    if dev.type == "cuda":
        return torch.cuda.get_device_name(torch.cuda.current_device())
    elif dev.type == "mps":
        return f"Apple {platform.processor()} (MPS)"
    return platform.processor() or "CPU"


viewer = napari.current_viewer()
device = get_best_device()

# ──────────────────────────────────────────────────────────────────────
# ONNX model paths — same models as the original segmentation plugin,
# located relative to this file (in onnx_models/ at repo root)
# ──────────────────────────────────────────────────────────────────────

def _find_onnx_models_root() -> Path:
    """
    Find the onnx_models directory robustly.
    Walks upward from this file's location looking for an 'onnx_models' folder.
    This works whether the package is installed editable or not, as long as
    the onnx_models folder is accessible in the repo.
    Falls back to the original napari-cool-tools-segmentation package location.
    """
    # First: walk up from this file looking for onnx_models/
    candidate = Path(__file__).resolve()
    for _ in range(8):  # walk up max 8 levels
        candidate = candidate.parent
        if (candidate / "onnx_models").exists():
            return candidate

    # Second: look next to the napari-cool-tools-segmentation package
    try:
        import napari_cool_tools_segmentation
        seg_root = Path(napari_cool_tools_segmentation.__file__).parents[3]
        if (seg_root / "onnx_models").exists():
            return seg_root
    except ImportError:
        pass

    # Third: check common repo locations
    for root in [
        Path.home() / "git" / "napari-cool-tools",
        Path("/Users/florianguillot/git/napari-cool-tools"),
    ]:
        if (root / "onnx_models").exists():
            return root

    raise FileNotFoundError(
        "Could not locate 'onnx_models' directory. "
        "Please ensure the napari-cool-tools repo is accessible."
    )


onnx_folder_parent_path = _find_onnx_models_root()
onnx_bscan_path = onnx_folder_parent_path / "onnx_models/bscan/"
onnx_bscan_retina_seg_path = onnx_bscan_path / "retina"
onnx_bscan_deconjugate_path = onnx_bscan_path / "deconjugate"

onnx_enface_path = onnx_folder_parent_path / "onnx_models/enface"
onnx_enface_vessels_path = onnx_enface_path / "vessels"
onnx_enface_optic_nerve_path = onnx_enface_path / "optic_nerve"
onnx_enface_ridge_path = onnx_enface_path / "ridge"

onnx_bscan_retina_seg = list(onnx_bscan_retina_seg_path.rglob("*.onnx"))[0]
onnx_bscan_deconjugate = list(onnx_bscan_deconjugate_path.rglob("*.onnx"))[0]

onnx_enface_vessels = list(onnx_enface_vessels_path.rglob("*.onnx"))[0]
onnx_enface_optic_nerve = list(onnx_enface_optic_nerve_path.rglob("*.onnx"))[0]
onnx_enface_ridge = list(onnx_enface_ridge_path.rglob("*.onnx"))[0]


class BscanSegmentationType(Enum):
    RETINASEG = onnx_bscan_retina_seg
    DECONJUGATE = onnx_bscan_deconjugate


class EnfaceSegmentationType(Enum):
    VESSEL = onnx_enface_vessels
    OPTICNERVEHEAD = onnx_enface_optic_nerve
    RIDGE = onnx_enface_ridge
