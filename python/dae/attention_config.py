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
    """Select the measured swapped-A/B SM100 GQA decode configuration.

    KV128 wins across the measured short- and long-context regimes.  One-block
    contexts avoid the split-reduction fixed cost.  Longer contexts use the
    largest equal-sized split that fits the SM budget; requiring a divisor of
    the padded block count prevents a split from dropping a KV block.
    """
    if batch_size <= 0 or seq_len <= 0 or num_kv_heads <= 0 or device_sms <= 0:
        raise ValueError(
            "batch_size, seq_len, num_kv_heads, and device_sms must be positive"
        )
    if not 1 <= max_split <= 16:
        raise ValueError("max_split must be in [1, 16]")
    base_ctas = batch_size * num_kv_heads
    if base_ctas > device_sms:
        raise ValueError(
            f"one GQA CTA per request/KV head needs {base_ctas} SMs, but only {device_sms} are available"
        )

    kv_tile = 128
    blocks = (seq_len + kv_tile - 1) // kv_tile
    if blocks == 1:
        return BlackwellAttentionConfig(kv_tile=128, split_kv=1, num_sms=base_ctas)

    split_budget = max(1, min(max_split, device_sms // base_ctas))
    split = max(
        candidate
        for candidate in range(1, min(blocks, split_budget) + 1)
        if blocks % candidate == 0
    )
    return BlackwellAttentionConfig(
        kv_tile=kv_tile,
        split_kv=split,
        num_sms=base_ctas * split,
    )


__all__ = ["BlackwellAttentionConfig", "select_blackwell_attention_config"]
