CANONICAL_DEBUG_STAGE_ORDER = (
    "embed",
    "q_proj",
    "q_rope",
    "k_proj",
    "k_rope",
    "v_proj",
    "attn",
    "out",
    "post_attn_rms",
    "gate_fused",
    "up_fused",
    "silu_fused",
    "down_low",
    "final_rms",
    "logits",
    "argmax",
    "restore",
)

LEGACY_DEBUG_STAGE_ALIASES = {
    "gate_low": "post_attn_rms",
    "gate_high": "post_attn_rms",
    "up_low": "post_attn_rms",
    "up_high": "post_attn_rms",
    "silu_split": "post_attn_rms",
    "down_high": "down_low",
}

DEBUG_STAGE_ORDER = CANONICAL_DEBUG_STAGE_ORDER + tuple(LEGACY_DEBUG_STAGE_ALIASES) + (
    "full",
)


def normalize_stage_name(stage_name: str) -> str:
    if stage_name == "full":
        return stage_name
    return LEGACY_DEBUG_STAGE_ALIASES.get(stage_name, stage_name)


def stage_enabled(stop_after: str, stage_name: str) -> bool:
    normalized_stop = normalize_stage_name(stop_after)
    normalized_stage = normalize_stage_name(stage_name)
    if normalized_stop == "full":
        return True
    requested_idx = CANONICAL_DEBUG_STAGE_ORDER.index(normalized_stop)
    stage_idx = CANONICAL_DEBUG_STAGE_ORDER.index(normalized_stage)
    return stage_idx <= requested_idx


def bind_unused_late_barriers_to_zero(dae):
    for group in dae.resource_groups.values():
        for name, bar_info in group.bars.items():
            if bar_info["late_bind"] and bar_info["count"] is None:
                group.bindBarrier(name, 0)


def print_barrier_counts(dae):
    print("[debug] barrier counts:")
    for group_name, group in dae.resource_groups.items():
        for name, bar_info in group.bars.items():
            if bar_info["count"] is None:
                continue
            print(f"[debug]   {group_name}.{name} = {bar_info['count']}")


def bind_late_barriers_with_default(dae, *insts, unresolved_count=None):
    bar_counts = dae.collect_barrier_release_counts(*insts)
    for group in dae.resource_groups.values():
        for name, bar_info in group.bars.items():
            if not bar_info["late_bind"] or bar_info["count"] is not None:
                continue

            matched_counts = {
                bar_counts[bar_id]
                for bar_id in group.bar_instances.get(name, [])
                if bar_id in bar_counts
            }
            if len(matched_counts) == 1:
                group.bindBarrier(name, matched_counts.pop())
                continue
            if len(matched_counts) == 0 and unresolved_count is not None:
                group.bindBarrier(name, unresolved_count)
                continue
            if len(matched_counts) > 1:
                raise ValueError(
                    f"Barrier {group.name}.{name} observed inconsistent release counts: {sorted(matched_counts)}"
                )
            raise ValueError(f"Could not infer release count for barrier {group.name}.{name}")
