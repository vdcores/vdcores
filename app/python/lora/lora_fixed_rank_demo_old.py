import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from dae.launcher import *
from dae.schedule import SchedGemm, SchedGemv
from dae.util import dae_app, tensor_diff


torch.manual_seed(0)

gpu = torch.device("cuda")
dtype = torch.bfloat16

SHRINK_GEMM = Gemm_M64N64
EXPAND_GEMM = Gemm_M64N128K64
EXPAND_GEMV = globals().get("Gemv_M64N8K64")

HIDDEN = 4096
LORA_RANK = 64
GROUP_SIZES = [128] + [64] * 4 + [8] * 11
NUM_SMS = 128

if EXPAND_GEMV is None:
    raise RuntimeError("Gemv_M64N8K64 must be added before running app/python/lora/lora_fixed_rank_demo_ori.py")


@dataclass(frozen=True)
class StageChoice:
    num_sms: int
    latency: float


@dataclass
class StageSpec:
    group_id: int
    token_count: int
    stage_name: str
    bar_id: int | None
    choices: list[StageChoice]
    make_schedule: Callable[[int, int], object]

    @property
    def min_sms(self):
        return self.choices[0].num_sms

    @property
    def min_latency(self):
        return self.choices[-1].latency


def estimate_schedule_latency(schedule):
    tile_m, tile_n, _ = schedule.Atom.MNK
    tile_work = tile_m * tile_n * schedule.k_per_fold

    if isinstance(schedule, SchedGemm):
        rounds = math.ceil(schedule.total_workers / schedule.num_sms)
        return rounds * tile_work
    if isinstance(schedule, SchedGemv):
        return tile_work
    raise TypeError(f"Unsupported schedule type {type(schedule)}")


def barrier_release_count(schedule):
    if isinstance(schedule, SchedGemv):
        return schedule.num_sms
    if isinstance(schedule, SchedGemm):
        return schedule.total_workers
    raise TypeError(f"Unsupported schedule type {type(schedule)}")


def enumerate_stage_choices(schedule_ctor):
    choices = []
    best_latency = None
    for num_sms in range(1, NUM_SMS + 1):
        try:
            schedule = schedule_ctor(num_sms, 0)
        except (AssertionError, ValueError):
            continue

        latency = estimate_schedule_latency(schedule)
        if best_latency is not None and latency >= best_latency:
            continue

        choices.append(StageChoice(num_sms=num_sms, latency=latency))
        best_latency = latency

    if not choices:
        raise RuntimeError("No legal SM allocations found for schedule")
    return choices


def merge_segments(segments):
    if not segments:
        return []

    merged = []
    for base_sm, num_sms in sorted(segments):
        if not merged:
            merged.append([base_sm, num_sms])
            continue

        prev_base, prev_num = merged[-1]
        if prev_base + prev_num == base_sm:
            merged[-1][1] += num_sms
        else:
            merged.append([base_sm, num_sms])

    return [(base_sm, num_sms) for base_sm, num_sms in merged]


def reserve_segment(segments, num_sms):
    best_idx = None
    best_size = None
    for idx, (_, seg_size) in enumerate(segments):
        if seg_size < num_sms:
            continue
        if best_size is None or seg_size < best_size:
            best_idx = idx
            best_size = seg_size

    if best_idx is None:
        return None

    base_sm, seg_size = segments[best_idx]
    new_segments = list(segments)
    if seg_size == num_sms:
        new_segments.pop(best_idx)
    else:
        new_segments[best_idx] = (base_sm + num_sms, seg_size - num_sms)
    return base_sm, new_segments


def free_segment(segments, base_sm, num_sms):
    return merge_segments([*segments, (base_sm, num_sms)])


def consume_exact_segment(segments, base_sm, num_sms):
    new_segments = []
    matched = False

    for seg_base, seg_size in segments:
        seg_end = seg_base + seg_size
        req_end = base_sm + num_sms

        if req_end <= seg_base or base_sm >= seg_end:
            new_segments.append((seg_base, seg_size))
            continue

        if base_sm < seg_base or req_end > seg_end:
            raise RuntimeError("Requested placement exceeds free segment bounds")

        matched = True
        if seg_base < base_sm:
            new_segments.append((seg_base, base_sm - seg_base))
        if req_end < seg_end:
            new_segments.append((req_end, seg_end - req_end))

    if not matched:
        raise RuntimeError("Requested placement not found in free segments")

    return merge_segments(new_segments)


def critical_tail(spec, expand_specs):
    tail = spec.min_latency
    if spec.stage_name == "shrink":
        tail += expand_specs[spec.group_id].min_latency
    return tail


def tail_for_choice(spec, choice, expand_specs):
    tail = choice.latency
    if spec.stage_name == "shrink":
        tail += expand_specs[spec.group_id].min_latency
    return tail


def stage_priority(spec, expand_specs):
    expand_bonus = 1 if spec.stage_name == "expand" else 0
    return (
        expand_bonus,
        critical_tail(spec, expand_specs),
        spec.min_latency / spec.min_sms,
        spec.token_count,
    )


def current_choice(spec, choice_pos):
    return spec.choices[choice_pos]


def pack_batch(specs, choice_pos, free_segments):
    placements = []
    segments = list(free_segments)
    ordering = sorted(
        range(len(specs)),
        key=lambda idx: (
            -specs[idx].choices[choice_pos[idx]].num_sms,
            -specs[idx].group_id,
        ),
    )

    for idx in ordering:
        num_sms = specs[idx].choices[choice_pos[idx]].num_sms
        placed = reserve_segment(segments, num_sms)
        if placed is None:
            return None
        base_sm, segments = placed
        placements.append((idx, base_sm))

    placement_map = {idx: base_sm for idx, base_sm in placements}
    return [
        (
            specs[idx],
            specs[idx].choices[choice_pos[idx]].num_sms,
            placement_map[idx],
            specs[idx].choices[choice_pos[idx]].latency,
        )
        for idx in range(len(specs))
    ]


def choose_stage_batch(ready_specs, free_segments, expand_specs):
    if not ready_specs or not free_segments:
        return None

    total_free = sum(seg_size for _, seg_size in free_segments)
    ready_specs = sorted(
        ready_specs,
        key=lambda spec: (
            critical_tail(spec, expand_specs),
            1 if spec.stage_name == "expand" else 0,
            spec.token_count,
        ),
        reverse=True,
    )

    selected = []
    used_sms = 0
    for spec in ready_specs:
        if used_sms + spec.min_sms > total_free:
            continue
        selected.append(spec)
        used_sms += spec.min_sms

    while selected:
        choice_pos = [0] * len(selected)
        used_sms = sum(spec.choices[pos].num_sms for spec, pos in zip(selected, choice_pos))
        selected_tails = [
            tail_for_choice(spec, spec.choices[pos], expand_specs)
            for spec, pos in zip(selected, choice_pos)
        ]
        bottleneck_idx = max(range(len(selected)), key=lambda idx: selected_tails[idx])

        while True:
            spec = selected[bottleneck_idx]
            next_pos = choice_pos[bottleneck_idx] + 1
            if next_pos >= len(spec.choices):
                break

            extra_sms = spec.choices[next_pos].num_sms - spec.choices[choice_pos[bottleneck_idx]].num_sms
            if used_sms + extra_sms > total_free:
                break

            choice_pos[bottleneck_idx] = next_pos
            used_sms += extra_sms

        while True:
            best_upgrade = None
            best_score = None
            selected_tails = [
                tail_for_choice(spec, spec.choices[pos], expand_specs)
                for spec, pos in zip(selected, choice_pos)
            ]
            current_max_tail = max(selected_tails)
            for idx, spec in enumerate(selected):
                next_pos = choice_pos[idx] + 1
                if next_pos >= len(spec.choices):
                    continue

                extra_sms = spec.choices[next_pos].num_sms - spec.choices[choice_pos[idx]].num_sms
                if used_sms + extra_sms > total_free:
                    continue

                current_tail = tail_for_choice(spec, spec.choices[choice_pos[idx]], expand_specs)
                next_tail = tail_for_choice(spec, spec.choices[next_pos], expand_specs)
                tail_improvement = current_tail - next_tail
                if tail_improvement <= 0:
                    continue

                new_tails = list(selected_tails)
                new_tails[idx] = next_tail
                max_tail_improvement = current_max_tail - max(new_tails)
                is_bottleneck = 1 if current_tail == current_max_tail else 0
                expand_bias = 1.1 if spec.stage_name == "expand" else 1.0
                score = (
                    max_tail_improvement,
                    is_bottleneck,
                    tail_improvement * expand_bias / extra_sms,
                    tail_improvement,
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_upgrade = (idx, next_pos, extra_sms)

            if best_upgrade is None:
                break

            idx, next_pos, extra_sms = best_upgrade
            choice_pos[idx] = next_pos
            used_sms += extra_sms

        while True:
            packed = pack_batch(selected, choice_pos, free_segments)
            if packed is not None:
                return packed

            downgrade = None
            downgrade_score = None
            for idx, spec in enumerate(selected):
                if choice_pos[idx] == 0:
                    continue

                cur_choice = current_choice(spec, choice_pos[idx])
                prev_choice = current_choice(spec, choice_pos[idx] - 1)
                freed_sms = cur_choice.num_sms - prev_choice.num_sms
                tail_penalty = (
                    tail_for_choice(spec, prev_choice, expand_specs)
                    - tail_for_choice(spec, cur_choice, expand_specs)
                )
                score = (tail_penalty / freed_sms, tail_penalty, freed_sms)
                if downgrade_score is None or score < downgrade_score:
                    downgrade_score = score
                    downgrade = idx

            if downgrade is not None:
                choice_pos[downgrade] -= 1
                continue

            drop_idx = min(
                range(len(selected)),
                key=lambda idx: stage_priority(selected[idx], expand_specs),
            )
            selected.pop(drop_idx)
            break

    return None


def build_pipeline_schedule(shrink_specs, expand_specs):
    insts = []
    plan = []
    barrier_counts = {}

    ready_specs = list(shrink_specs)
    running = []
    free_segments = [(0, NUM_SMS)]
    current_time = 0.0
    total_stages = len(shrink_specs) + len(expand_specs)
    completed = 0

    while completed < total_stages:
        finished = [entry for entry in running if entry["end_time"] <= current_time + 1e-9]
        if finished:
            for entry in finished:
                running.remove(entry)
                free_segments = free_segment(free_segments, entry["base_sm"], entry["num_sms"])
                completed += 1
                if entry["spec"].stage_name == "shrink":
                    ready_specs.append(expand_specs[entry["spec"].group_id])

        started = False
        while True:
            batch = choose_stage_batch(ready_specs, free_segments, expand_specs)
            if batch is None:
                break

            started = True
            for spec, num_sms, base_sm, latency in batch:
                ready_specs.remove(spec)
                free_segments = consume_exact_segment(free_segments, base_sm, num_sms)

                schedule = spec.make_schedule(num_sms, base_sm)
                if spec.stage_name == "shrink":
                    schedule.bar("store", spec.bar_id)
                    barrier_counts[spec.bar_id] = barrier_release_count(schedule)
                else:
                    schedule.bar("load", spec.bar_id)

                insts.append(schedule)
                plan.append({
                    "group_id": spec.group_id,
                    "token_count": spec.token_count,
                    "stage_name": spec.stage_name,
                    "start_time": current_time,
                    "end_time": current_time + latency,
                    "latency": latency,
                    "num_sms": num_sms,
                    "base_sm": base_sm,
                })
                running.append({
                    "spec": spec,
                    "base_sm": base_sm,
                    "num_sms": num_sms,
                    "end_time": current_time + latency,
                })

        if completed >= total_stages:
            break
        if not running and not started:
            raise RuntimeError("Scheduler stalled with ready work but no runnable allocation")

        if running:
            current_time = min(entry["end_time"] for entry in running)

    return insts, plan, barrier_counts


def make_group_tensors():
    xs = []
    a_weights = []
    b_weights = []
    shrink_outs = []
    expand_outs = []

    for token_count in GROUP_SIZES:
        xs.append(torch.rand(token_count, HIDDEN, dtype=dtype, device=gpu) - 0.5)
        a_weights.append(torch.rand(LORA_RANK, HIDDEN, dtype=dtype, device=gpu) - 0.5)
        b_weights.append(torch.rand(HIDDEN, LORA_RANK, dtype=dtype, device=gpu) - 0.5)
        shrink_outs.append(torch.zeros(token_count, LORA_RANK, dtype=dtype, device=gpu))
        expand_outs.append(torch.zeros(token_count, HIDDEN, dtype=dtype, device=gpu))

    return xs, a_weights, b_weights, shrink_outs, expand_outs


def build_reference(xs, a_weights, b_weights):
    shrink_refs = []
    expand_refs = []
    for x, a_weight, b_weight in zip(xs, a_weights, b_weights):
        shrink_ref = x.float() @ a_weight.t().float()
        expand_ref = shrink_ref @ b_weight.t().float()
        shrink_refs.append(shrink_ref.to(dtype))
        expand_refs.append(expand_ref.to(dtype))
    return shrink_refs, expand_refs


def visualize_pipeline_plan(plan, output_path):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("matplotlib not available, skipping schedule visualization")
        return

    if not plan:
        print("empty pipeline plan, skipping schedule visualization")
        return

    fig, ax = plt.subplots(figsize=(15, 7))
    cmap = plt.get_cmap("tab20")

    for entry in plan:
        color_idx = (entry["group_id"] * 2) + (0 if entry["stage_name"] == "shrink" else 1)
        color = cmap(color_idx % cmap.N)
        width = entry["end_time"] - entry["start_time"]
        rect = Rectangle(
            (entry["start_time"], entry["base_sm"]),
            width,
            entry["num_sms"],
            facecolor=color,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.85,
        )
        ax.add_patch(rect)

        if width > 0:
            ax.text(
                entry["start_time"] + width / 2,
                entry["base_sm"] + entry["num_sms"] / 2,
                f"g{entry['group_id']} {entry['stage_name'][0].upper()}",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )

    ax.set_xlim(0, max(entry["end_time"] for entry in plan) * 1.02)
    ax.set_ylim(0, NUM_SMS)
    ax.set_xlabel("Proxy Time")
    ax.set_ylabel("SM ID")
    ax.set_title("LoRA Pipeline Schedule Plan (Original Scheduler)")
    ax.grid(True, axis="x", linestyle="--", alpha=0.25)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved schedule visualization to {output_path}")


matX, matA, matB, matShrink, matOut = make_group_tensors()
refShrink, refOut = build_reference(matX, matA, matB)

dae = Launcher(NUM_SMS, device=gpu)
bars = [None] * (len(GROUP_SIZES) + 1)


def make_shrink_spec(group_id, token_count):
    if token_count == 8:
        Atom = Gemv_M64N8B2
        Atom.n_batch = 2
        M, N, K = LORA_RANK, token_count, HIDDEN
        tile_m, tile_n, tile_k = Atom.MNK
        loadA = TmaTensor(dae, matA[group_id]).wgmma_load(tile_m, tile_k, Major.K)
        loadB = TmaTensor(dae, matX[group_id]).wgmma_load(tile_n, tile_k * Atom.n_batch, Major.K)
        reduceC = TmaTensor(dae, matShrink[group_id]).wgmma("reduce", tile_n, tile_m, Major.MN)
        op_class = SchedGemv
    elif token_count % 64 == 0:
        Atom = Gemm_M64N64
        M, N, K = token_count, LORA_RANK, HIDDEN
        tile_m, tile_n, tile_k = Atom.MNK
        loadA = TmaTensor(dae, matX[group_id]).wgmma_load(tile_m, tile_k, Major.K)
        loadB = TmaTensor(dae, matA[group_id]).wgmma_load(tile_n, tile_k, Major.K)
        reduceC = TmaTensor(dae, matShrink[group_id]).wgmma("reduce", tile_m, tile_n, Major.K)
        op_class = SchedGemm
    else:
        raise ValueError(f"Unsupported token count {token_count}")

    shrink_bar = dae.new_bar(None)
    bars[group_id] = shrink_bar

    def schedule_ctor(num_sms, base_sm):
        return op_class(
            Atom,
            MNK=(M, N, K),
            tmas=(loadA, loadB, reduceC),
        ).place(num_sms, base_sm)

    return StageSpec(
        group_id=group_id,
        token_count=token_count,
        stage_name="shrink",
        bar_id=shrink_bar,
        choices=enumerate_stage_choices(schedule_ctor),
        make_schedule=schedule_ctor,
    )


def make_expand_spec(group_id, token_count):
    if token_count == 8:
        Atom = Gemv_M64N8K64
        M, N, K = HIDDEN, token_count, LORA_RANK
        tile_m, tile_n, tile_k = Atom.MNK
        loadA = TmaTensor(dae, matB[group_id]).wgmma_load(tile_m, tile_k, Major.K)
        loadB = TmaTensor(dae, matShrink[group_id]).wgmma_load(tile_n, tile_k, Major.K)
        reduceC = TmaTensor(dae, matOut[group_id]).wgmma("reduce", tile_n, tile_m, Major.MN)
        op_class = SchedGemv
    elif token_count % 64 == 0:
        Atom = Gemm_M64N64K64
        M, N, K = token_count, HIDDEN, LORA_RANK
        tile_m, tile_n, tile_k = Atom.MNK
        loadA = TmaTensor(dae, matShrink[group_id]).wgmma_load(tile_m, tile_k, Major.K)
        loadB = TmaTensor(dae, matB[group_id]).wgmma_load(tile_n, tile_k, Major.K)
        reduceC = TmaTensor(dae, matOut[group_id]).wgmma("reduce", tile_m, tile_n, Major.K)
        op_class = SchedGemm
    else:
        raise ValueError(f"Unsupported token count {token_count}")

    def schedule_ctor(num_sms, base_sm):
        return op_class(
            Atom,
            MNK=(M, N, K),
            tmas=(loadA, loadB, reduceC),
        ).place(num_sms, base_sm)

    return StageSpec(
        group_id=group_id,
        token_count=token_count,
        stage_name="expand",
        bar_id=bars[group_id],
        choices=enumerate_stage_choices(schedule_ctor),
        make_schedule=schedule_ctor,
    )


shrink_specs = [make_shrink_spec(group_id, token_count) for group_id, token_count in enumerate(GROUP_SIZES)]
expand_specs = {
    group_id: make_expand_spec(group_id, token_count)
    for group_id, token_count in enumerate(GROUP_SIZES)
}
pipeline_insts, pipeline_plan, pipeline_barrier_counts = build_pipeline_schedule(shrink_specs, expand_specs)
for bar_id, count in pipeline_barrier_counts.items():
    dae.set_bar(bar_id, count)


dae.i(
    pipeline_insts,
    TerminateC(),
    TerminateM(),
)

print("LoRA fixed-rank mixed pipeline (original scheduler)")
print(f"group sizes: {GROUP_SIZES}, sms: {NUM_SMS}")
print("Predicted pipeline plan:")
for entry in pipeline_plan:
    print(
        "  "
        f"t=[{entry['start_time']:.0f}, {entry['end_time']:.0f}] "
        f"group {entry['group_id']} {entry['stage_name']} "
        f"tokens={entry['token_count']} sms={entry['num_sms']} base_sm={entry['base_sm']}"
    )
visualize_pipeline_plan(
    pipeline_plan,
    Path("build") / "plots" / "lora_fixed_rank_schedule_ori.png",
)

dae_app(dae)

# for group_id, token_count in enumerate(GROUP_SIZES):
#     tensor_diff(f"group{group_id}_shrink_{token_count}", refShrink[group_id], matShrink[group_id])
#     tensor_diff(f"group{group_id}_expand_{token_count}", refOut[group_id], matOut[group_id])
