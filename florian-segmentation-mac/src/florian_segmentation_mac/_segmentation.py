"""
Mac-compatible napari plugin interface for segmentation.

Provides separate CPU-safe and MPS (experimental) variants of each
segmentation tool, registered as individual napari widgets.
"""

import gc
from typing import List

from magicgui import magic_factory
from napari.layers import Image, Layer
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info

import torch

# Updated import to florian_segmentation_mac
from florian_segmentation_mac import (
    BscanSegmentationType,
    EnfaceSegmentationType,
    Path,
    clear_gpu_cache,
    viewer,
)
from florian_segmentation_mac._segmentation_funcs import (
    bscan_onnx_deconj_func,
    bscan_onnx_seg_func,
    enface_onnx_seg_func,
)


@magic_factory()
def bscan_onnx_seg_plugin_cpu(
    img: Image,
    segmentation: BscanSegmentationType = BscanSegmentationType.RETINASEG,
    target_shape: list = [864, 864],
    batch_size: int = 32,
    num_workers: int = 0,
    output_preproc: bool = False,
    old_preproc: bool = False,
    debug: bool = False,
):
    """B-scan segmentation using CPU — fully safe on all platforms."""
    _bscan_onnx_seg_thread(
        img, segmentation=segmentation, target_shape=target_shape,
        batch_size=batch_size, num_workers=num_workers,
        use_cpu=True, use_mps=False, output_preproc=output_preproc,
        old_preproc=old_preproc, debug=debug,
    )

@magic_factory()
def bscan_onnx_seg_plugin_mps(
    img: Image,
    segmentation: BscanSegmentationType = BscanSegmentationType.RETINASEG,
    target_shape: list = [864, 864],
    batch_size: int = 32,
    num_workers: int = 0,
    output_preproc: bool = False,
    old_preproc: bool = False,
    debug: bool = False,
):
    """B-scan segmentation with MPS/CoreML acceleration (experimental)."""
    _bscan_onnx_seg_thread(
        img, segmentation=segmentation, target_shape=target_shape,
        batch_size=batch_size, num_workers=num_workers,
        use_cpu=False, use_mps=True, output_preproc=output_preproc,
        old_preproc=old_preproc, debug=debug,
    )

@thread_worker(connect={"yielded": viewer.add_layer if viewer else None})
def _bscan_onnx_seg_thread(img, **kwargs):
    show_info("B-scan segmentation started...")
    outputs = bscan_onnx_seg_func(img.data, **kwargs)
    for layer_data, layer_type in outputs:
        yield Layer.create(layer_data, {"name": f"{img.name}_{layer_type}"}, layer_type)
    show_info("Completed.")

@magic_factory()
def bscan_onnx_deconj_plugin_cpu(img: Image):
    _bscan_onnx_deconj_thread(img, use_cpu=True, use_mps=False)

@magic_factory()
def bscan_onnx_deconj_plugin_mps(img: Image):
    _bscan_onnx_deconj_thread(img, use_cpu=False, use_mps=True)

@thread_worker(connect={"returned": viewer.add_layer if viewer else None})
def _bscan_onnx_deconj_thread(img, **kwargs):
    deconj, suffix = bscan_onnx_deconj_func(img.data, **kwargs)
    return Layer.create(deconj, {"name": f"{img.name}_{suffix}"}, "image")

@magic_factory()
def enface_onnx_seg_plugin_cpu(img: Image):
    return _enface_seg_common(img, use_cpu=True, use_mps=False)

@magic_factory()
def enface_onnx_seg_plugin_mps(img: Image):
    return _enface_seg_common(img, use_cpu=False, use_mps=True)

def _enface_seg_common(img, **kwargs):
    res = enface_onnx_seg_func(img.data, **kwargs)
    return [Layer.create(res, {"name": f"{img.name}_Seg"}, "labels")]
