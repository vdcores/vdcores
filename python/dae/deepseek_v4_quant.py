"""Reference helpers for DeepSeek-V4 ModelOpt NVFP4 tensors.

The runtime kernels consume packed E2M1 values, per-16 E4M3 scales, and one
FP32 dequantization scale per tensor.  These helpers intentionally preserve
that checkpoint-level contract and are used for correctness tests and weight
loading, not as the eventual fused runtime quantizer.
"""

from __future__ import annotations

import torch


_FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
               -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
_NVFP4_MAX = 6.0
_E4M3_MAX = 448.0


def _positive_e4m3_values(device: torch.device) -> torch.Tensor:
    bits = torch.arange(256, dtype=torch.uint8, device=device)
    values = bits.view(torch.float8_e4m3fn).float()
    values = values[torch.isfinite(values) & (values > 0)]
    return torch.unique(values, sorted=True)


def _ceil_e4m3(values: torch.Tensor) -> torch.Tensor:
    table = _positive_e4m3_values(values.device)
    flat = values.float().clamp(min=table[0], max=_E4M3_MAX).reshape(-1)
    indices = torch.searchsorted(table, flat, right=False).clamp_max(table.numel() - 1)
    return table[indices].reshape_as(values).to(torch.float8_e4m3fn)


def quantize_nvfp4(
    tensor: torch.Tensor,
    global_dequant_scale: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize the last dimension to ModelOpt's packed NVFP4 representation."""
    if tensor.shape[-1] % 16:
        raise ValueError("NVFP4 requires the last dimension to be divisible by 16")
    source = tensor.float()
    if global_dequant_scale is None:
        scale2 = source.abs().amax().clamp_min(torch.finfo(torch.float32).tiny)
        scale2 = scale2 / (_NVFP4_MAX * _E4M3_MAX)
    else:
        scale2 = torch.as_tensor(
            global_dequant_scale, dtype=torch.float32, device=tensor.device
        ).reshape(())
        if not bool((scale2 > 0).item()):
            raise ValueError("global_dequant_scale must be positive")

    blocks = source.reshape(*source.shape[:-1], -1, 16)
    requested_sf = blocks.abs().amax(dim=-1) / (_NVFP4_MAX * scale2)
    block_scale = _ceil_e4m3(requested_sf)
    normalized = blocks / (block_scale.float().unsqueeze(-1) * scale2)

    codebook = torch.tensor(_FP4_VALUES, dtype=torch.float32, device=tensor.device)
    codes = (normalized.unsqueeze(-1) - codebook).abs().argmin(dim=-1).to(torch.uint8)
    codes = codes.reshape(*source.shape[:-1], source.shape[-1])
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return packed.contiguous(), block_scale.contiguous(), scale2.contiguous()


def dequantize_nvfp4(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    global_dequant_scale: torch.Tensor | float,
) -> torch.Tensor:
    """Dequantize packed E2M1 values using per-16 E4M3 and FP32 scales."""
    if packed.dtype != torch.uint8:
        raise ValueError("packed NVFP4 values must use uint8 storage")
    codebook = torch.tensor(_FP4_VALUES, dtype=torch.float32, device=packed.device)
    codes = torch.empty(
        (*packed.shape[:-1], packed.shape[-1] * 2),
        dtype=torch.long,
        device=packed.device,
    )
    codes[..., 0::2] = (packed & 0x0F).long()
    codes[..., 1::2] = (packed >> 4).long()
    values = codebook[codes]
    scales = block_scale.float().repeat_interleave(16, dim=-1)
    scale2 = torch.as_tensor(
        global_dequant_scale, dtype=torch.float32, device=packed.device
    )
    return values * scales * scale2


def _ceil_ue8m0(values: torch.Tensor) -> torch.Tensor:
    minimum = torch.tensor(2.0**-127, dtype=torch.float32, device=values.device)
    exponents = torch.ceil(torch.log2(values.float().clamp_min(minimum)))
    scales = torch.exp2(exponents.clamp(min=-127, max=127))
    return scales.to(torch.float8_e8m0fnu)


def quantize_fp8_block128(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a vector or matrix to E4M3 with UE8M0 block-128 scales."""
    source = tensor.float()
    if source.ndim == 1:
        if source.numel() % 128:
            raise ValueError("FP8 vector length must be divisible by 128")
        blocks = source.reshape(-1, 128)
        scale = _ceil_ue8m0(blocks.abs().amax(dim=-1) / _E4M3_MAX)
        expanded = scale.float().repeat_interleave(128)
    elif source.ndim == 2:
        m, k = source.shape
        if m % 128 or k % 128:
            raise ValueError("FP8 matrix dimensions must be divisible by 128")
        tiles = source.reshape(m // 128, 128, k // 128, 128)
        scale = _ceil_ue8m0(tiles.abs().amax(dim=(1, 3)) / _E4M3_MAX)
        expanded = scale.float().repeat_interleave(128, 0).repeat_interleave(128, 1)
    else:
        raise ValueError("FP8 block quantization supports only vectors and matrices")
    quantized = (source / expanded).clamp(-_E4M3_MAX, _E4M3_MAX)
    return quantized.to(torch.float8_e4m3fn).contiguous(), scale.contiguous()


def dequantize_fp8_block128(
    quantized: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Dequantize the vector or matrix form emitted by quantize_fp8_block128."""
    if quantized.dtype != torch.float8_e4m3fn:
        raise ValueError("quantized FP8 values must use torch.float8_e4m3fn")
    if scale.dtype != torch.float8_e8m0fnu:
        raise ValueError("FP8 block scales must use torch.float8_e8m0fnu")
    if quantized.ndim == 1:
        expanded = scale.float().repeat_interleave(128)
    elif quantized.ndim == 2:
        expanded = scale.float().repeat_interleave(128, 0).repeat_interleave(128, 1)
    else:
        raise ValueError("FP8 block dequantization supports only vectors and matrices")
    return quantized.float() * expanded


__all__ = [
    "quantize_nvfp4",
    "dequantize_nvfp4",
    "quantize_fp8_block128",
    "dequantize_fp8_block128",
]
