"""
Mac-compatible segmentation functions for B-scan and Enface images.

Adapted from napari_cool_tools_segmentation._segmentation_funcs with
cross-platform device support (CPU / MPS / CUDA) and safe GPU memory
management.
"""

import gc
import platform
import sys
from typing import List, Tuple

import numpy as np
from napari.layers import Layer
from napari.types import ImageData
from tqdm import tqdm

import torch

from napari_cool_tools_img_proc import DType
from napari_cool_tools_img_proc._normalization_funcs import convert_dtype_and_rescale

# Updated import to florian_segmentation_mac
from florian_segmentation_mac import (
    BscanSegmentationType,
    EnfaceSegmentationType,
    Path,
    clear_gpu_cache,
    get_device_name,
)


# ──────────────────────────────────────────────────────────────────────
# Provider helpers
# ──────────────────────────────────────────────────────────────────────

def _get_onnx_providers(use_cpu: bool, use_mps: bool, onnx_folder_path=None, gpu_limit: int = 6):
    """
    Build the ONNX Runtime execution provider list based on platform
    and user preferences.

    Returns (providers_list, device_string_for_binding)
    """
    if use_cpu:
        return ["CPUExecutionProvider"], "cpu"

    # MPS / CoreML path (macOS Apple Silicon)
    if use_mps and sys.platform == "darwin":
        providers = []
        try:
            import onnxruntime
            available = onnxruntime.get_available_providers()
            if "CoreMLExecutionProvider" in available:
                providers.append("CoreMLExecutionProvider")
        except Exception:
            pass
        providers.append("CPUExecutionProvider")
        return providers, "cpu"

    # CUDA path (Linux/Windows with NVIDIA GPU)
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": device_id,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": gpu_limit * 1024 * 1024 * 1024,
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                    "cudnn_conv_use_max_workspace": "1",
                },
            ),
            "CPUExecutionProvider",
        ]
        return providers, "cuda"

    # Fallback
    return ["CPUExecutionProvider"], "cpu"


def _safe_device_info(processor_device) -> str:
    """Get device info string without crashing on any platform."""
    if isinstance(processor_device, str):
        dev = torch.device(processor_device)
    else:
        dev = processor_device

    if dev.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(torch.cuda.current_device())
    elif dev.type == "mps":
        return f"Apple {platform.processor()} (MPS)"
    return platform.processor() or "CPU"


# ──────────────────────────────────────────────────────────────────────
# B-scan ONNX Segmentation (Mac-compatible)
# ──────────────────────────────────────────────────────────────────────

def bscan_onnx_seg_func(
    img: ImageData,
    onnx_path=BscanSegmentationType.RETINASEG.value,
    target_shape: list = [864, 864],
    batch_size: int = 32,
    num_workers: int = 0,
    gpu_limit: int = 6,
    use_cpu: bool = True,
    use_mps: bool = False,
    output_preproc: bool = False,
    old_preproc: bool = False,
    verbose: bool = True,
    debug: bool = False,
):
    """
    Run B-scan segmentation using an ONNX model.
    """
    import onnxruntime
    import torch.nn as nn
    from jj_nn_framework.data_setup import LoadNumpyData
    from jj_nn_framework.nn_transforms import (
        BscanPreproc2,
        NormalizeCLAHE2,
        PadToTargetM,
        ResizeToFit,
    )
    from torch.utils.data import DataLoader
    from torchvision.transforms.functional import InterpolationMode
    from torchvision.transforms.v2.functional import resize

    if img.dtype.type not in (np.float16, np.float32):
        if img.dtype.type == np.float16:
            img = convert_dtype_and_rescale(img, datatype=DType.NP_FLOAT32)
        else:
            raise ValueError(f"Unsupported dtype: {img.dtype}")

    init_shape = (img.shape[-2], img.shape[-1])

    if use_cpu:
        processor = "cpu"
    else:
        # For MPS, we still use CPU for LoadNumpyData prep
        processor = "cpu"

    providers, onnx_dev = _get_onnx_providers(
        use_cpu=use_cpu, use_mps=use_mps, gpu_limit=gpu_limit
    )

    num_bscans = len(img)
    rem = num_bscans % batch_size
    if rem != 0:
        missing_bscans = batch_size - rem
        batch_fill = np.empty((missing_bscans, img.shape[1], img.shape[2]), dtype=img.dtype)
        img = np.concatenate([img, batch_fill])

    pttm_params = {
        "h": target_shape[-2],
        "w": target_shape[-1],
        "X_data_format": "NHW",
        "y_data_format": "NHW",
        "mode": "constant",
        "value": None,
        "pad_gt": False,
        "device": processor,
    }

    if old_preproc:
        pred_trans = nn.Sequential(
            PadToTargetM(**pttm_params),
            BscanPreproc2(log_gain=2.5, clahe_clip_limit=1.0),
        )
    else:
        pred_trans = nn.Sequential(
            ResizeToFit(target_shape), PadToTargetM(**pttm_params), NormalizeCLAHE2()
        )

    pred_ds = LoadNumpyData(
        img,
        chunk_size=batch_size,
        transform=pred_trans,
        preprocessing=None,
        device=processor,
    )

    pred_dl = DataLoader(
        pred_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    onnx_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
    input_name = onnx_session.get_inputs()[0].name

    preproc_bscans = []
    label_preds = []

    use_io_binding = (onnx_dev == "cuda" and torch.cuda.is_available())

    for image_batch in tqdm(pred_dl, desc="Segmenting B-scans:"):
        if use_io_binding:
            binding = onnx_session.io_binding()
            it_shape = image_batch.shape
            binding.bind_input(
                name="input",
                device_type="cuda",
                device_id=0,
                element_type=np.float32,
                shape=tuple(it_shape),
                buffer_ptr=image_batch.data_ptr(),
            )
            pred_shape = (it_shape[0], 1, it_shape[2], it_shape[3])
            pred_tensor = torch.empty(pred_shape, dtype=torch.uint8, device="cuda")
            binding.bind_output(
                name="output",
                device_type="cuda",
                device_id=0,
                element_type=np.uint8,
                shape=tuple(pred_tensor.shape),
                buffer_ptr=pred_tensor.data_ptr(),
            )
            onnx_session.run_with_iobinding(binding)
            labels = pred_tensor.detach().squeeze().cpu().numpy()
        else:
            images_np = image_batch.numpy().astype(np.float32)
            onnx_outputs = onnx_session.run(None, {input_name: images_np})
            labels = onnx_outputs[0].squeeze().astype(np.uint8)

        preproc_bscans.append(image_batch.detach().squeeze().cpu().numpy()[:num_bscans])
        label_preds.append(labels)

    del pred_ds, pred_dl
    gc.collect()
    clear_gpu_cache()

    preproc_bscans = np.concatenate(preproc_bscans, axis=0)
    label_preds = np.concatenate(label_preds, axis=0)[:num_bscans]
    reshaped_out = resize(
        torch.tensor(label_preds.copy()),
        (init_shape),
        interpolation=InterpolationMode.NEAREST_EXACT,
    ).numpy()

    output = []
    if output_preproc:
        output.append((preproc_bscans[:num_bscans], "image"))
    output.append((reshaped_out, "labels"))

    return output


# ──────────────────────────────────────────────────────────────────────
# Enface ONNX Segmentation (Mac-compatible)
# ──────────────────────────────────────────────────────────────────────

def enface_onnx_seg_func(
    data: ImageData,
    onnx_path=EnfaceSegmentationType.VESSEL.value,
    segmentation_type="vessel",
    label_val: int = 1,
    use_cpu: bool = True,
    use_mps: bool = False,
    DoG: bool = False,
    blur: bool = False,
    log_adjust: bool = False,
    output_preproc: bool = False,
    debug: bool = False,
) -> np.ndarray:
    """
    Run enface segmentation using an ONNX model.
    """
    from jj_nn_framework.image_funcs import bw_1_to_3ch, normalize_in_range, pad_to_targetM_2d
    from jj_nn_framework.nn_transforms import DiffOfGausPred
    from kornia.enhance import adjust_log, equalize_clahe
    from kornia.filters import gaussian_blur2d
    from onnxruntime import InferenceSession
    from torchvision.transforms import v2

    providers, _ = _get_onnx_providers(use_cpu=use_cpu, use_mps=use_mps)

    data = data.copy().astype("float32")
    if data.max() > 1.0:
        data = normalize_in_range(data, 0.0, 1.0)

    pt_data = torch.tensor(data, device="cpu")
    ch3_data = bw_1_to_3ch(pt_data, data_format="HW")
    
    pad_flag = False
    resize_flag = False
    if ch3_data.shape[-1] < 864 and ch3_data.shape[-2] < 864:
        pad_flag = True
        mod_data = pad_to_targetM_2d(ch3_data, (864, 864), "NCHW")
    elif ch3_data.shape[-1] > 864 or ch3_data.shape[-2] > 864:
        resize_flag = True
        original_shape = (ch3_data.shape[-2], ch3_data.shape[-1])
        mod_data = v2.functional.resize(ch3_data, (864, 864), interpolation=v2.InterpolationMode.BICUBIC)
    else:
        mod_data = ch3_data

    x_eq = equalize_clahe(mod_data, clip_limit=3.0)
    if log_adjust:
        x_eq = adjust_log(x_eq, gain=1)
    if DoG:
        x_eq = DiffOfGausPred(low_sigma=0.5, high_sigma=6.0)(x_eq)
    if blur:
        x_eq = gaussian_blur2d(x_eq, kernel_size=3, sigma=(1.0, 1.0), border_type="reflect")

    x_eq_cpu = x_eq.detach().cpu().numpy()
    onnx_session = InferenceSession(onnx_path, providers=providers)
    onnx_out = onnx_session.run(None, {onnx_session.get_inputs()[0].name: x_eq_cpu})[0].squeeze().astype(np.uint8)

    if pad_flag:
        offset_0 = (onnx_out.shape[0] - data.shape[0]) // 2
        offset_1 = (onnx_out.shape[1] - data.shape[1]) // 2
        final_seg = onnx_out[offset_0:offset_0+data.shape[0], offset_1:offset_1+data.shape[1]]
    elif resize_flag:
        final_seg = v2.functional.resize(torch.tensor(onnx_out).unsqueeze(0), original_shape, v2.InterpolationMode.NEAREST_EXACT).numpy()
    else:
        final_seg = onnx_out

    return (final_seg > 0).astype(np.uint8) * label_val


# ──────────────────────────────────────────────────────────────────────
# B-scan ONNX Deconjugation (Mac-compatible)
# ──────────────────────────────────────────────────────────────────────

def bscan_onnx_deconj_func(
    data: ImageData,
    onnx_path: Path = BscanSegmentationType.DECONJUGATE.value,
    target_bscan_dimension: tuple[int, int] = (512, 1024),
    batch_size: int = 8,
    num_workers: int = 0,
    gpu_limit: int = 6,
    use_cpu: bool = True,
    use_mps: bool = False,
    verbose: bool = True,
    debug: bool = False,
) -> tuple[ImageData, str]:
    """
    Remove complex conjugate artifacts from B-scans using an ONNX model.
    """
    import onnxruntime
    import torch.nn as nn
    from jj_nn_framework.data_setup import LoadNumpyData
    from jj_nn_framework.nn_transforms import PadToTargetM, ResizeToFit
    from torch.utils.data import DataLoader
    from torchvision.transforms.functional import InterpolationMode
    from torchvision.transforms.v2.functional import resize

    data = data.transpose(-3, -1, -2).astype("float32")
    data = (data - data.min()) / (data.max() - data.min() + 1e-8)

    init_shape = (data.shape[-2], data.shape[-1])
    providers, onnx_dev = _get_onnx_providers(use_cpu=use_cpu, use_mps=use_mps, gpu_limit=gpu_limit)

    num_bscans = len(data)
    rem = num_bscans % batch_size
    if rem != 0:
        batch_fill = np.empty((batch_size - rem, data.shape[1], data.shape[2]), dtype=data.dtype)
        data = np.concatenate([data, batch_fill])

    pred_trans = nn.Sequential(ResizeToFit(target_bscan_dimension), PadToTargetM(h=target_bscan_dimension[0], w=target_bscan_dimension[1], device="cpu"))
    pred_ds = LoadNumpyData(data, transform=pred_trans, device="cpu")
    pred_dl = DataLoader(pred_ds, batch_size=batch_size, shuffle=False)

    onnx_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
    input_name = onnx_session.get_inputs()[0].name
    
    preds = []
    for image_batch in tqdm(pred_dl, desc="Deconjugating B-scans:"):
        out_np = onnx_session.run(None, {input_name: image_batch.numpy()})[0].squeeze()
        preds.append(out_np[:num_bscans])

    preds = np.concatenate(preds, axis=0)[:num_bscans]
    reshaped_out = resize(torch.tensor(preds), init_shape, interpolation=InterpolationMode.BICUBIC).numpy()
    
    return reshaped_out.transpose(-3, -1, -2), "deconjugated"
