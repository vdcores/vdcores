"""Offline MXFP4 FFN image contract for DeepSeek-V4-Flash.

The NVIDIA checkpoint stores routed experts as ModelOpt NVFP4 and the shared
expert as block-scaled FP8.  Neither source representation is consumed by the
resident MXFP4 x MXFP8 FFN.  This module defines the durable, per-layer image
that an offline conversion produces and the lightweight runtime loader for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


FORMAT_VERSION = 1
DIRECTORY_NAME = "vdcores-mxfp4-ffn-v1"
CHECKPOINT_EXPERTS = 256
STREAM_EXPERTS = CHECKPOINT_EXPERTS + 1
LINEAR1_SLICES = 16
DOWN_SLICES = 32
LINEAR1_OPERATIONS = 16
LINEAR1_K128_PER_OPERATION = 4
DOWN_OPERATIONS = 8
DOWN_K128_PER_OPERATION = 2
NATIVE_SCALE_BYTES_PER_K128 = 512


@dataclass(frozen=True)
class DeepSeekV4MxfpFfnLayer:
    """Four homogeneous tensors consumed by one routed resident FFN layer."""

    linear1_weights: torch.Tensor
    linear1_scales: torch.Tensor
    down_weights: torch.Tensor
    down_scales: torch.Tensor

    @property
    def nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.linear1_weights,
                self.linear1_scales,
                self.down_weights,
                self.down_scales,
            )
        )


def default_mxfp_ffn_directory(checkpoint: str | Path) -> Path:
    return Path(checkpoint) / DIRECTORY_NAME


def mxfp_ffn_layer_path(root: str | Path, layer_id: int) -> Path:
    layer_id = int(layer_id)
    if layer_id < 0:
        raise ValueError("MXFP FFN layer id must be non-negative")
    return Path(root) / f"layer-{layer_id:03d}.safetensors"


def pack_mxfp4_data(packed: torch.Tensor, *, tile_k: int) -> torch.Tensor:
    """Reorder linear packed-E2M1 bytes into the resident TMA task layout."""

    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise ValueError("MXFP4 packed data must be rank-2 uint8 [M,K/2]")
    tile_k = int(tile_k)
    if tile_k not in (128, 256, 512):
        raise ValueError("MXFP4 task tile K must be 128, 256, or 512")
    rows, packed_k = packed.shape
    k = packed_k * 2
    if rows % 128 or k % tile_k:
        raise ValueError("MXFP4 matrix must be M128 and task-tile-K aligned")
    return (
        packed.reshape(rows // 128, 128, k // tile_k, tile_k // 128, 64)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )


def pack_mxfp4_scales(
    scales_128x4: torch.Tensor,
    *,
    rows: int,
    k: int,
    tile_k: int,
) -> torch.Tensor:
    """Group FlashInfer/CUTLASS 128x4 SFA bytes into task K records.

    FlashInfer's ``SfLayout.layout_128x4`` is the native CUTLASS order
    ``[M/128,K/128,32,4,4]``.  Each M128/K128 tile therefore contributes the
    exact 512-byte SFA image consumed by the VDCores UTCCP path.
    """

    if scales_128x4.dtype != torch.uint8:
        raise ValueError("MXFP4 scale factors must use uint8 UE8M0 storage")
    rows = int(rows)
    k = int(k)
    tile_k = int(tile_k)
    if rows <= 0 or rows % 128 or k <= 0 or k % tile_k:
        raise ValueError("MXFP4 scales require M128/task-tile-K aligned shapes")
    if tile_k not in (128, 256, 512):
        raise ValueError("MXFP4 task tile K must be 128, 256, or 512")
    expected = rows * (k // 32)
    if scales_128x4.numel() != expected:
        raise ValueError(
            f"MXFP4 scale payload has {scales_128x4.numel()} bytes, "
            f"expected {expected}"
        )
    return scales_128x4.reshape(
        rows // 128,
        k // tile_k,
        (tile_k // 128) * NATIVE_SCALE_BYTES_PER_K128,
    ).contiguous()


def validate_mxfp_ffn_layer(layer: DeepSeekV4MxfpFfnLayer) -> None:
    expected = (
        (
            "linear1_weights",
            layer.linear1_weights,
            (
                STREAM_EXPERTS * LINEAR1_SLICES,
                LINEAR1_OPERATIONS,
                LINEAR1_K128_PER_OPERATION,
                128,
                64,
            ),
        ),
        (
            "linear1_scales",
            layer.linear1_scales,
            (
                STREAM_EXPERTS * LINEAR1_SLICES,
                LINEAR1_OPERATIONS,
                LINEAR1_K128_PER_OPERATION * NATIVE_SCALE_BYTES_PER_K128,
            ),
        ),
        (
            "down_weights",
            layer.down_weights,
            (
                STREAM_EXPERTS * DOWN_SLICES,
                DOWN_OPERATIONS,
                DOWN_K128_PER_OPERATION,
                128,
                64,
            ),
        ),
        (
            "down_scales",
            layer.down_scales,
            (
                STREAM_EXPERTS * DOWN_SLICES,
                DOWN_OPERATIONS,
                DOWN_K128_PER_OPERATION * NATIVE_SCALE_BYTES_PER_K128,
            ),
        ),
    )
    devices = set()
    for name, tensor, shape in expected:
        if (
            tensor.dtype != torch.uint8
            or tuple(tensor.shape) != shape
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"offline MXFP FFN {name} must be contiguous uint8{shape}"
            )
        devices.add(tensor.device)
    if len(devices) != 1:
        raise ValueError("offline MXFP FFN tensors must share one device")


def load_mxfp_ffn_layer(
    root: str | Path,
    layer_id: int,
    *,
    device: torch.device | str,
) -> DeepSeekV4MxfpFfnLayer:
    """Load one already-converted layer without any format transformation."""

    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("loading offline MXFP FFN data requires safetensors") from error

    path = mxfp_ffn_layer_path(root, layer_id)
    if not path.is_file():
        raise FileNotFoundError(
            f"offline MXFP FFN layer is missing: {path}; run the offline "
            "DeepSeek-V4 MXFP conversion first"
        )
    names = (
        "linear1_weights",
        "linear1_scales",
        "down_weights",
        "down_scales",
    )
    with safe_open(str(path), framework="pt", device="cpu") as file:
        metadata = file.metadata() or {}
        if metadata.get("format") != "vdcores-deepseek-v4-mxfp-ffn":
            raise ValueError(f"unrecognized offline MXFP FFN file: {path}")
        if int(metadata.get("version", "-1")) != FORMAT_VERSION:
            raise ValueError(f"unsupported offline MXFP FFN version in {path}")
        if int(metadata.get("layer", "-1")) != int(layer_id):
            raise ValueError(f"offline MXFP FFN layer metadata mismatch in {path}")
        available = set(file.keys())
        missing = set(names) - available
        if missing:
            raise ValueError(f"offline MXFP FFN file {path} misses {sorted(missing)}")
        cpu_tensors = {name: file.get_tensor(name) for name in names}

    target = torch.device(device)
    layer = DeepSeekV4MxfpFfnLayer(
        linear1_weights=cpu_tensors["linear1_weights"].to(target),
        linear1_scales=cpu_tensors["linear1_scales"].to(target),
        down_weights=cpu_tensors["down_weights"].to(target),
        down_scales=cpu_tensors["down_scales"].to(target),
    )
    validate_mxfp_ffn_layer(layer)
    return layer


__all__ = [
    "FORMAT_VERSION",
    "DIRECTORY_NAME",
    "CHECKPOINT_EXPERTS",
    "STREAM_EXPERTS",
    "LINEAR1_SLICES",
    "DOWN_SLICES",
    "DeepSeekV4MxfpFfnLayer",
    "default_mxfp_ffn_directory",
    "mxfp_ffn_layer_path",
    "pack_mxfp4_data",
    "pack_mxfp4_scales",
    "validate_mxfp_ffn_layer",
    "load_mxfp_ffn_layer",
]
