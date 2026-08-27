"""Benchmark-only repair for vLLM 0.27.1 DeepSeek-V4 KV page views.

The FP8 DeepSeek-V4 record is 584 bytes and the FlashMLA SM100 decoder
requires every physical page to begin on a 576-byte stride.  vLLM budgets the
aligned page but can lose the padded outer stride while constructing its final
view.  Keep every logical token row unchanged and restore only that outer
page stride.
"""

from math import prod

import torch

from vllm.v1.worker.gpu import attn_utils
from vllm.v1.worker import gpu_model_runner


_original_reshape_attention_kv_cache = attn_utils._reshape_attention_kv_cache
_reported = False


def _reshape_attention_kv_cache_tma_aligned(
    kv_raw_tensor,
    kv_cache_spec,
    kv_cache_shape,
    kv_cache_stride_order,
    num_blocks,
    packing,
):
    global _reported
    result = _original_reshape_attention_kv_cache(
        kv_raw_tensor,
        kv_cache_spec,
        kv_cache_shape,
        kv_cache_stride_order,
        num_blocks,
        packing,
    )
    if (
        result.dtype != torch.uint8
        or result.ndim != 3
        or result.shape[-1] != 584
        or result.stride(0) % 576 == 0
    ):
        return result

    logical_page = prod(result.shape[1:])
    aligned_page = (logical_page + 575) // 576 * 576
    physical_stride = aligned_page
    if packing is not None:
        _, packed_stride = packing
        # A singleton block has no observable inter-page distance, so expose
        # the closest valid stride instead of the much larger hybrid-pool
        # packing pitch.  Multiple blocks must retain the packing pitch to
        # address their actual backing locations.
        if num_blocks > 1 and packed_stride % 576 == 0:
            physical_stride = packed_stride

    # Re-stride the existing packed-pool view in place.  Using ``result``
    # preserves its absolute storage offset; no cache allocation, clear, or
    # copy is introduced by this compatibility shim.
    result = torch.as_strided(
        result,
        size=result.shape,
        stride=(physical_stride, *result.stride()[1:]),
        storage_offset=result.storage_offset(),
    )
    if not _reported:
        print(
            "VLLM_TMA_KV_STRIDE_PATCH "
            f"logical_page={logical_page} physical_stride={physical_stride} "
            f"alignment=576 blocks={num_blocks}",
            flush=True,
        )
        _reported = True
    return result


attn_utils._reshape_attention_kv_cache = _reshape_attention_kv_cache_tma_aligned
gpu_model_runner._reshape_attention_kv_cache = (
    _reshape_attention_kv_cache_tma_aligned
)
