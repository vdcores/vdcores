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
    raise RuntimeError("Gemv_M64N8K64 must be added before running app/python/lora_fixed_rank_demo.py")


@dataclass(frozen=True)
class StageChoice:
    num_sms: int
    latency: float


@dataclass
class StageSpec:
    key: str
    group_id: int
    token_count: int
    stage_name: str
    bar_id: int | None
    successors: list[str]
    downstream_latency: float
    chunk_label: str | None
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


def critical_tail(spec):
    return spec.min_latency + spec.downstream_latency


def tail_for_choice(spec, choice):
    return choice.latency + spec.downstream_latency


def stage_priority(spec):
    expand_bonus = 1 if spec.stage_name == "expand" else 0
    return (
        expand_bonus,
        critical_tail(spec),
        spec.min_latency / spec.min_sms,
        spec.token_count,
    )


def current_choice(spec, choice_pos):
    return spec.choices[choice_pos]


def group_tail_values(specs, choice_pos):
    tails = {}
    for idx, spec in enumerate(specs):
        tail = tail_for_choice(spec, spec.choices[choice_pos[idx]])
        tails[spec.group_id] = max(tails.get(spec.group_id, 0.0), tail)
    return tails


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


def choose_stage_batch(ready_specs, free_segments):
    if not ready_specs or not free_segments:
        return None

    total_free = sum(seg_size for _, seg_size in free_segments)
    ordered_specs = sorted(
        ready_specs,
        key=lambda spec: (
            critical_tail(spec),
            1 if spec.stage_name == "expand" else 0,
            spec.token_count,
        ),
        reverse=True,
    )

    def optimize_selected_batch(selected):
        working = list(selected)
        while working:
            choice_pos = [0] * len(working)
            used_sms = sum(spec.choices[pos].num_sms for spec, pos in zip(working, choice_pos))
            current_group_tails = group_tail_values(working, choice_pos)
            bottleneck_group = max(current_group_tails, key=current_group_tails.get)

            while True:
                candidate_idx = None
                candidate_score = None
                for idx, spec in enumerate(working):
                    if spec.group_id != bottleneck_group:
                        continue
                    next_pos = choice_pos[idx] + 1
                    if next_pos >= len(spec.choices):
                        continue

                    extra_sms = spec.choices[next_pos].num_sms - spec.choices[choice_pos[idx]].num_sms
                    if used_sms + extra_sms > total_free:
                        continue

                    new_choice_pos = list(choice_pos)
                    new_choice_pos[idx] = next_pos
                    new_group_tails = group_tail_values(working, new_choice_pos)
                    group_tail_improvement = (
                        current_group_tails[bottleneck_group] - new_group_tails[bottleneck_group]
                    )
                    current_tail = tail_for_choice(spec, spec.choices[choice_pos[idx]])
                    score = (
                        group_tail_improvement,
                        current_tail,
                        -(spec.choices[next_pos].num_sms),
                    )
                    if candidate_score is None or score > candidate_score:
                        candidate_score = score
                        candidate_idx = idx

                if candidate_idx is None:
                    break

                next_pos = choice_pos[candidate_idx] + 1
                candidate_spec = working[candidate_idx]
                extra_sms = (
                    candidate_spec.choices[next_pos].num_sms
                    - candidate_spec.choices[choice_pos[candidate_idx]].num_sms
                )
                choice_pos[candidate_idx] = next_pos
                used_sms += extra_sms
                current_group_tails = group_tail_values(working, choice_pos)

            while True:
                best_upgrade = None
                best_score = None
                current_group_tails = group_tail_values(working, choice_pos)
                current_max_group_tail = max(current_group_tails.values())
                for idx, spec in enumerate(working):
                    next_pos = choice_pos[idx] + 1
                    if next_pos >= len(spec.choices):
                        continue

                    extra_sms = spec.choices[next_pos].num_sms - spec.choices[choice_pos[idx]].num_sms
                    if used_sms + extra_sms > total_free:
                        continue

                    current_tail = tail_for_choice(spec, spec.choices[choice_pos[idx]])
                    next_tail = tail_for_choice(spec, spec.choices[next_pos])
                    tail_improvement = current_tail - next_tail
                    if tail_improvement <= 0:
                        continue

                    new_choice_pos = list(choice_pos)
                    new_choice_pos[idx] = next_pos
                    new_group_tails = group_tail_values(working, new_choice_pos)
                    max_tail_improvement = current_max_group_tail - max(new_group_tails.values())
                    is_bottleneck = 1 if current_group_tails[spec.group_id] == current_max_group_tail else 0
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

            current_max_group_tail = max(group_tail_values(working, choice_pos).values())

            while True:
                packed = pack_batch(working, choice_pos, free_segments)
                if packed is not None:
                    return packed, current_max_group_tail

                downgrade = None
                downgrade_score = None
                for idx, spec in enumerate(working):
                    if choice_pos[idx] == 0:
                        continue

                    cur_choice = current_choice(spec, choice_pos[idx])
                    prev_choice = current_choice(spec, choice_pos[idx] - 1)
                    freed_sms = cur_choice.num_sms - prev_choice.num_sms
                    tail_penalty = tail_for_choice(spec, prev_choice) - tail_for_choice(spec, cur_choice)
                    score = (tail_penalty / freed_sms, tail_penalty, freed_sms)
                    if downgrade_score is None or score < downgrade_score:
                        downgrade_score = score
                        downgrade = idx

                if downgrade is not None:
                    choice_pos[downgrade] -= 1
                    current_max_group_tail = max(group_tail_values(working, choice_pos).values())
                    continue

                drop_idx = min(
                    range(len(working)),
                    key=lambda idx: stage_priority(working[idx]),
                )
                working.pop(drop_idx)
                break

        return None, None

    selected = []
    best_packed = None
    best_max_tail = None

    for spec in ordered_specs:
        if sum(item.min_sms for item in selected) + spec.min_sms > total_free:
            continue

        packed, max_tail = optimize_selected_batch([*selected, spec])
        if packed is None:
            continue

        if best_packed is None or max_tail <= best_max_tail + 1e-9:
            selected.append(spec)
            best_packed = packed
            best_max_tail = max_tail

    return best_packed


def build_pipeline_schedule(stage_specs):
    insts = []
    plan = []
    barrier_counts = {}
    spec_by_key = {spec.key: spec for spec in stage_specs}
    remaining_deps = {spec.key: 0 for spec in stage_specs}
    for spec in stage_specs:
        for succ_key in spec.successors:
            remaining_deps[succ_key] += 1

    ready_specs = [spec for spec in stage_specs if remaining_deps[spec.key] == 0]
    running = []
    free_segments = [(0, NUM_SMS)]
    current_time = 0.0
    total_stages = len(stage_specs)
    completed = 0

    while completed < total_stages:
        finished = [entry for entry in running if entry["end_time"] <= current_time + 1e-9]
        if finished:
            for entry in finished:
                running.remove(entry)
                free_segments = free_segment(free_segments, entry["base_sm"], entry["num_sms"])
                completed += 1
                for succ_key in entry["spec"].successors:
                    remaining_deps[succ_key] -= 1
                    if remaining_deps[succ_key] == 0:
                        ready_specs.append(spec_by_key[succ_key])

        started = False
        batch = choose_stage_batch(ready_specs, free_segments)
        if batch is not None:
            started = True
            for spec, num_sms, base_sm, latency in batch:
                ready_specs.remove(spec)
                free_segments = consume_exact_segment(free_segments, base_sm, num_sms)

                schedule = spec.make_schedule(num_sms, base_sm)
                if spec.stage_name == "shrink":
                    schedule.bar("store", spec.bar_id)
                    barrier_counts[spec.bar_id] = barrier_release_count(schedule)
                else:
                    # schedule.bar("load", spec.bar_id)
                    ...

                insts.append(schedule)
                plan.append({
                    "key": spec.key,
                    "group_id": spec.group_id,
                    "token_count": spec.token_count,
                    "stage_name": spec.stage_name,
                    "chunk_label": spec.chunk_label,
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
                f"g{entry['group_id']} {entry['stage_name'][0].upper()}"
                + (f" {entry['chunk_label']}" if entry["chunk_label"] else ""),
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )

    ax.set_xlim(0, max(entry["end_time"] for entry in plan) * 1.02)
    ax.set_ylim(0, NUM_SMS)
    ax.set_xlabel("Proxy Time")
    ax.set_ylabel("SM ID")
    ax.set_title("LoRA Pipeline Schedule Plan")
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
bars = [None] * (len(GROUP_SIZES) + 1) # last one for global barrier baseline


def make_stage_spec(
    *,
    key,
    group_id,
    token_count,
    stage_name,
    bar_id,
    successors,
    chunk_label,
    schedule_ctor,
):
    return StageSpec(
        key=key,
        group_id=group_id,
        token_count=token_count,
        stage_name=stage_name,
        bar_id=bar_id,
        successors=list(successors),
        downstream_latency=0.0,
        chunk_label=chunk_label,
        choices=enumerate_stage_choices(schedule_ctor),
        make_schedule=schedule_ctor,
    )


def compute_downstream_latencies(stage_specs):
    spec_by_key = {spec.key: spec for spec in stage_specs}
    for spec in stage_specs:
        spec.downstream_latency = sum(spec_by_key[succ_key].min_latency for succ_key in spec.successors)

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

    return make_stage_spec(
        key=f"g{group_id}:shrink",
        group_id=group_id,
        token_count=token_count,
        stage_name="shrink",
        bar_id=shrink_bar,
        successors=[f"g{group_id}:expand"],
        chunk_label=None,
        schedule_ctor=schedule_ctor,
    )


def make_chunked_shrink_specs(group_id, token_count):
    if token_count == 8:
        return [make_shrink_spec(group_id, token_count)]

    if token_count % 64 != 0:
        raise ValueError(f"Unsupported token count {token_count}")

    num_chunks = token_count // 64
    if num_chunks == 1:
        return [make_shrink_spec(group_id, token_count)]

    Atom = Gemm_M64N64
    chunk_tokens = 64
    M, N, K = token_count, LORA_RANK, HIDDEN
    tile_m, tile_n, tile_k = Atom.MNK
    loadA = TmaTensor(dae, matX[group_id]).wgmma_load(tile_m, tile_k, Major.K)
    loadB = TmaTensor(dae, matA[group_id]).wgmma_load(tile_n, tile_k, Major.K)
    reduceC = TmaTensor(dae, matShrink[group_id]).wgmma("reduce", tile_m, tile_n, Major.K)

    specs = []
    for chunk_idx in range(num_chunks):
        chunk_base = chunk_idx * chunk_tokens
        shrink_bar = dae.new_bar(None)

        def schedule_ctor(num_sms, base_sm, chunk_base=chunk_base):
            return SchedGemm(
                Atom,
                MNK=((chunk_base, chunk_tokens), N, K),
                tmas=(loadA, loadB, reduceC),
            ).place(num_sms, base_sm)

        specs.append(make_stage_spec(
            key=f"g{group_id}:shrink:{chunk_idx}",
            group_id=group_id,
            token_count=chunk_tokens,
            stage_name="shrink",
            bar_id=shrink_bar,
            successors=[f"g{group_id}:expand:{chunk_idx}"],
            chunk_label=f"tok{chunk_idx + 1}/{num_chunks}",
            schedule_ctor=schedule_ctor,
        ))
    return specs

def base_shrink_sched():
    shrink_base_sm = 0
    def split_N(base_sm, num_sm, N, Atom):
        TileM, TileN, TileK = Atom.MNK
        insts = []
        loadA = TmaTensor(dae, matA[group_id]).wgmma_load(TileM, TileK, Major.K)
        for i in range(N // TileN):
            loadB = TmaTensor(dae, matX[group_id][i*TileN:(i+1)*TileN]).wgmma_load(TileN, TileK * Atom.n_batch, Major.K)
            reduceC = TmaTensor(dae, matShrink[group_id][i*TileN:(i+1)*TileN]).wgmma("reduce", TileN, TileM, Major.MN)

            inst = SchedGemv(
                Atom,
                MNK=(LORA_RANK, TileN, HIDDEN),
                tmas=(loadA, loadB, reduceC),
            ).place(num_sm, base_sm)
                # .bar("store", bars[-1])
            insts.append(inst)
            base_sm = (base_sm + num_sm) % NUM_SMS
        return insts, base_sm

    bar_cnt = sum(t for t in GROUP_SIZES) // 8
    bars[-1] = dae.new_bar(bar_cnt)
    for group_id, token_count in enumerate(GROUP_SIZES):
        insts, shrink_base_sm = split_N(shrink_base_sm, 1, token_count, Gemv_M64N8)
        shrink_insts.extend(insts)

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

    return make_stage_spec(
        key=f"g{group_id}:expand",
        group_id=group_id,
        token_count=token_count,
        stage_name="expand",
        bar_id=bars[group_id],
        successors=[],
        chunk_label=None,
        schedule_ctor=schedule_ctor,
    )


def make_chunked_expand_specs(group_id, token_count, shrink_specs):
    if token_count == 8:
        Atom = Gemv_M64N8K64
        chunk_m = HIDDEN // 4
        N, K = token_count, LORA_RANK
        tile_m, tile_n, tile_k = Atom.MNK
        loadA = TmaTensor(dae, matB[group_id]).wgmma_load(tile_m, tile_k, Major.K)
        loadB = TmaTensor(dae, matShrink[group_id]).wgmma_load(tile_n, tile_k, Major.K)
        reduceC = TmaTensor(dae, matOut[group_id]).wgmma("reduce", tile_n, tile_m, Major.MN)
        bar_id = shrink_specs[0].bar_id

        specs = []
        for chunk_idx in range(4):
            chunk_base = chunk_idx * chunk_m

            def schedule_ctor(num_sms, base_sm, chunk_base=chunk_base):
                return SchedGemv(
                    Atom,
                    MNK=((chunk_base, chunk_m), N, K),
                    tmas=(loadA, loadB, reduceC),
                ).place(num_sms, base_sm)

            specs.append(make_stage_spec(
                key=f"g{group_id}:expand:{chunk_idx}",
                group_id=group_id,
                token_count=token_count,
                stage_name="expand",
                bar_id=bar_id,
                successors=[],
                chunk_label=f"h{chunk_idx + 1}/4",
                schedule_ctor=schedule_ctor,
            ))
        shrink_specs[0].successors = [spec.key for spec in specs]
        return specs

    if token_count % 64 != 0:
        raise ValueError(f"Unsupported token count {token_count}")

    if len(shrink_specs) == 1:
        return [make_expand_spec(group_id, token_count)]

    Atom = Gemm_M64N64K64
    chunk_tokens = 64
    N, K = HIDDEN, LORA_RANK
    tile_m, tile_n, tile_k = Atom.MNK
    loadA = TmaTensor(dae, matShrink[group_id]).wgmma_load(tile_m, tile_k, Major.K)
    loadB = TmaTensor(dae, matB[group_id]).wgmma_load(tile_n, tile_k, Major.K)
    reduceC = TmaTensor(dae, matOut[group_id]).wgmma("reduce", tile_m, tile_n, Major.K)

    specs = []
    for chunk_idx, shrink_spec in enumerate(shrink_specs):
        chunk_base = chunk_idx * chunk_tokens

        def schedule_ctor(num_sms, base_sm, chunk_base=chunk_base):
            return SchedGemm(
                Atom,
                MNK=((chunk_base, chunk_tokens), N, K),
                tmas=(loadA, loadB, reduceC),
            ).place(num_sms, base_sm)

        specs.append(make_stage_spec(
            key=f"g{group_id}:expand:{chunk_idx}",
            group_id=group_id,
            token_count=chunk_tokens,
            stage_name="expand",
            bar_id=shrink_spec.bar_id,
            successors=[],
            chunk_label=f"tok{chunk_idx + 1}/{len(shrink_specs)}",
            schedule_ctor=schedule_ctor,
        ))
    return specs

def base_expand_sched():
    expand_base_sm = 0
    def split_N_M(base_sm, num_sm, N, Atom):
        TileM, TileN, TileK = Atom.MNK
        insts = []
        loadA = TmaTensor(dae, matB[group_id]).wgmma_load(TileM, TileK, Major.K)
        for i in range(N // TileN):
            loadB = TmaTensor(dae, matShrink[group_id][i*TileN:(i+1)*TileN]).wgmma_load(TileN, TileK * Atom.n_batch, Major.K)
            reduceC = TmaTensor(dae, matOut[group_id][i*TileN:(i+1)*TileN]).wgmma("reduce", TileN, TileM, Major.MN)

            inst = SchedGemv(
                Atom,
                MNK=(HIDDEN, TileN, LORA_RANK),
                tmas=(loadA, loadB, reduceC),
            ).place(num_sm, (base_sm + num_sm)).bar("load", bars[-1])
            insts.append(inst)
            base_sm = (base_sm + num_sm) % NUM_SMS
        return insts, base_sm
    
    for group_id, token_count in enumerate(GROUP_SIZES):
        num_sms = HIDDEN // 64
        insts, expand_base_sm = split_N_M(expand_base_sm, num_sms, token_count, Gemv_M64N8K64)
        shrink_insts.extend(insts)

shrink_specs = []
expand_specs = []
for group_id, token_count in enumerate(GROUP_SIZES):
    group_shrink_specs = make_chunked_shrink_specs(group_id, token_count)
    group_expand_specs = make_chunked_expand_specs(group_id, token_count, group_shrink_specs)
    shrink_specs.extend(group_shrink_specs)
    expand_specs.extend(group_expand_specs)

all_stage_specs = [*shrink_specs, *expand_specs]
compute_downstream_latencies(all_stage_specs)
pipeline_insts, pipeline_plan, pipeline_barrier_counts = build_pipeline_schedule(all_stage_specs)
for bar_id, count in pipeline_barrier_counts.items():
    dae.set_bar(bar_id, count)


dae.i(
    pipeline_insts,
    TerminateC(),
    TerminateM(),
)

print("LoRA fixed-rank mixed pipeline")
print(f"group sizes: {GROUP_SIZES}, sms: {NUM_SMS}")
print("Predicted pipeline plan:")
for entry in pipeline_plan:
    print(
        "  "
        f"t=[{entry['start_time']:.0f}, {entry['end_time']:.0f}] "
        f"group {entry['group_id']} {entry['stage_name']} "
        f"tokens={entry['token_count']} "
        f"{entry['chunk_label'] + ' ' if entry['chunk_label'] else ''}"
        f"sms={entry['num_sms']} base_sm={entry['base_sm']}"
    )
visualize_pipeline_plan(
    pipeline_plan,
    Path("build") / "plots" / "lora_fixed_rank_schedule.png",
)

dae_app(dae)

# for group_id, token_count in enumerate(GROUP_SIZES):
#     tensor_diff(f"group{group_id}_shrink_{token_count}", refShrink[group_id], matShrink[group_id])
#     tensor_diff(f"group{group_id}_expand_{token_count}", refOut[group_id], matOut[group_id])
