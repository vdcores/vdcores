from __future__ import annotations

import os

import torch


def _fallback_attention(module, query, key, value, attention_mask, dropout, scaling, **kwargs):
    from transformers.models.opt.modeling_opt import eager_attention_forward

    return eager_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=dropout,
        scaling=scaling,
        **kwargs,
    )


def _supported(module, query, key, value, attention_mask, dropout, **kwargs) -> bool:
    if kwargs.get("output_attentions", False):
        return False
    if dropout != 0.0:
        return False
    if getattr(getattr(module, "config", None), "model_type", None) != "opt":
        return False
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        return False
    if query.dtype not in (torch.float16, torch.bfloat16):
        return False
    if key.dtype != query.dtype or value.dtype != query.dtype:
        return False
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        return False
    if query.shape[2] != 1 or query.shape[-1] != 128:
        return False
    if key.shape != value.shape:
        return False
    if key.shape[0] != query.shape[0] or key.shape[1] != query.shape[1] or key.shape[-1] != query.shape[-1]:
        return False
    if query.stride(-1) != 1 or key.stride(-1) != 1 or value.stride(-1) != 1:
        return False
    if attention_mask is not None:
        if not attention_mask.is_cuda or attention_mask.dim() != 4:
            return False
        if attention_mask.shape[0] != query.shape[0] or attention_mask.shape[-1] != key.shape[2]:
            return False
        if attention_mask.shape[1] != 1 or attention_mask.shape[2] != 1:
            return False
    return True


def opt_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    scale = 1.0 if scaling is None else float(scaling)
    if not _supported(module, query, key, value, attention_mask, float(dropout), **kwargs):
        return _fallback_attention(module, query, key, value, attention_mask, dropout, scale, **kwargs)

    from . import _C

    mask = attention_mask
    if mask is not None and mask.dtype is not torch.float32:
        mask = mask.to(torch.float32)
    split_size = int(os.environ.get("OPT_ATTENTION_SPLIT_SIZE", "256"))
    return _C.decode(query, key, value, mask, scale, split_size), None


def register(name: str = "vdcores_opt") -> None:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS.register(name, opt_attention_forward)
