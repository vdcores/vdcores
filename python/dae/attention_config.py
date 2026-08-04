from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BlackwellAttentionConfig:
    kv_tile: int
    split_kv: int
    num_sms: int

    @property
    def use_split_kv(self) -> bool:
        return self.split_kv > 1


def select_blackwell_attention_config(
    batch_size: int,
    seq_len: int,
    *,
    num_kv_heads: int = 8,
    device_sms: int = 152,
    max_split: int = 16,
) -> BlackwellAttentionConfig:
    """Select the measured SM100 GQA decode tile and split count.

    Short contexts avoid the split-reduction fixed cost.  For longer contexts,
    KV64 is retained only when its smaller tile exposes more parallel CTAs;
    when occupancy ties, KV128 wins by halving QK issues and online-softmax
    rescale stages.
    """
    if batch_size <= 0 or seq_len <= 0 or num_kv_heads <= 0:
        raise ValueError("batch_size, seq_len, and num_kv_heads must be positive")
    base_ctas = batch_size * num_kv_heads
    if base_ctas > device_sms:
        raise ValueError(
            f"one GQA CTA per request/KV head needs {base_ctas} SMs, but only {device_sms} are available"
        )

    if seq_len <= 64:
        return BlackwellAttentionConfig(kv_tile=64, split_kv=1, num_sms=base_ctas)
    if seq_len <= 128:
        return BlackwellAttentionConfig(kv_tile=128, split_kv=1, num_sms=base_ctas)

    split_budget = max(1, min(max_split, device_sms // base_ctas))

    def candidate(kv_tile: int) -> BlackwellAttentionConfig:
        blocks = (seq_len + kv_tile - 1) // kv_tile
        split = min(blocks, split_budget)
        return BlackwellAttentionConfig(
            kv_tile=kv_tile,
            split_kv=split,
            num_sms=base_ctas * split,
        )

    kv64 = candidate(64)
    kv128 = candidate(128)
    if kv64.num_sms > kv128.num_sms:
        return kv64
    return kv128


__all__ = ["BlackwellAttentionConfig", "select_blackwell_attention_config"]
