DEBUG_STAGE_ORDER = (
    "embed",
    "q_proj",
    "q_rope",
    "k_proj",
    "k_rope",
    "v_proj",
    "attn",
    "out",
    "post_attn_rms",
    "gate_low",
    "gate_high",
    "up_low",
    "up_high",
    "silu_split",
    "gate_fused",
    "up_fused",
    "silu_fused",
    "down_low",
    "down_high",
    "final_rms",
    "logits",
    "argmax",
    "restore",
    "full",
)


def stage_enabled(stop_after: str, stage_name: str) -> bool:
    requested_idx = DEBUG_STAGE_ORDER.index(stop_after)
    stage_idx = DEBUG_STAGE_ORDER.index(stage_name)
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
