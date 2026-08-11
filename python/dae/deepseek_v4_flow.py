"""Decode-stage planning for a single-GPU DeepSeek-V4-Flash runtime."""

from __future__ import annotations

from dataclasses import dataclass

from .deepseek_v4 import DeepSeekV4FlashConfig


COMMON_ATTENTION_STAGES = (
    "hc_attn_project",
    "hc_attn_pre",
    "attn_norm",
    "q_a_fp8",
    "q_norm",
    "q_b_fp8",
    "q_head_norm",
    "q_rope",
    "kv_fp8",
    "kv_norm",
    "kv_rope",
)

COMMON_ATTENTION_TAIL = (
    "sparse_attention",
    "attention_inverse_rope",
    "grouped_o_a_fp8",
    "o_b_fp8",
    "hc_attn_post",
)

COMMON_FFN_STAGES = (
    "hc_ffn_project",
    "hc_ffn_pre",
    "ffn_norm",
    "router",
    "routed_expert_nvfp4",
    "shared_expert_fp8",
    "expert_reduce",
    "hc_ffn_post",
)


@dataclass(frozen=True)
class DeepSeekV4LayerDecodePlan:
    """One layer's ordered decode work at a fixed token position."""

    layer_id: int
    start_pos: int
    attention_kind: str
    compress_ratio: int
    hash_routing: bool
    compressed_rows: int
    compressed_selected: int
    requires_index_selection: bool
    attention_candidates: int
    should_compress: bool
    stages: tuple[str, ...]


def build_layer_decode_plan(
    layer_id: int,
    start_pos: int,
    config: DeepSeekV4FlashConfig | None = None,
) -> DeepSeekV4LayerDecodePlan:
    """Build the official single-token stage order for one transformer layer."""
    config = config or DeepSeekV4FlashConfig()
    if start_pos < 0:
        raise ValueError("decode position must be non-negative")

    attention_kind = config.attention_kind(layer_id)
    ratio = config.compress_ratios[layer_id]
    compressed_rows = (start_pos + 1) // ratio if ratio else 0
    compressed_selected = (
        min(config.index_topk, compressed_rows)
        if attention_kind == "csa"
        else compressed_rows
    )
    requires_index_selection = (
        attention_kind == "csa"
        and compressed_selected < compressed_rows
    )
    valid_window = min(config.sliding_window, start_pos + 1)
    attention_candidates = valid_window + compressed_selected
    should_compress = bool(ratio and (start_pos + 1) % ratio == 0)

    compression_stages: tuple[str, ...] = ()
    if ratio:
        compression_stages += ("attention_compressor_project",)
        if should_compress:
            compression_stages += (
                "attention_compressor_pool",
                "attention_compressor_norm",
                "attention_compressor_rope",
            )
    if attention_kind == "csa":
        if requires_index_selection:
            compression_stages += (
                "index_q_b_fp8",
                "index_q_rope",
                "index_q_hadamard",
            )
        compression_stages += ("index_compressor_project",)
        if should_compress:
            compression_stages += (
                "index_compressor_pool",
                "index_compressor_norm",
                "index_compressor_rope",
                "index_kv_hadamard",
            )
        if requires_index_selection:
            compression_stages += ("index_score", "index_topk")

    return DeepSeekV4LayerDecodePlan(
        layer_id=layer_id,
        start_pos=start_pos,
        attention_kind=attention_kind,
        compress_ratio=ratio,
        hash_routing=layer_id < config.num_hash_layers,
        compressed_rows=compressed_rows,
        compressed_selected=compressed_selected,
        requires_index_selection=requires_index_selection,
        attention_candidates=attention_candidates,
        should_compress=should_compress,
        stages=(
            COMMON_ATTENTION_STAGES
            + compression_stages
            + COMMON_ATTENTION_TAIL
            + COMMON_FFN_STAGES
        ),
    )


def build_decode_plan(
    start_pos: int,
    config: DeepSeekV4FlashConfig | None = None,
) -> tuple[DeepSeekV4LayerDecodePlan, ...]:
    """Build all 43 transformer-layer plans for one decode token."""
    config = config or DeepSeekV4FlashConfig()
    return tuple(
        build_layer_decode_plan(layer_id, start_pos, config)
        for layer_id in range(config.num_layers)
    )


__all__ = [
    "DeepSeekV4LayerDecodePlan",
    "build_layer_decode_plan",
    "build_decode_plan",
]
