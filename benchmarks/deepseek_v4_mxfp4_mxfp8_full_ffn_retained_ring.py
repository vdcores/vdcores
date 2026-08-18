#!/usr/bin/env python3
"""One-launch retained-ring MXFP4/MXFP8 full-FFN benchmark."""

from __future__ import annotations

import argparse
import os
import statistics

import torch

from dae import runtime
from dae.instructions import ProfileEvent, TmaTensor
from dae.launcher import Launcher
from dae.runtime import opcode
from dae.schedule import (
    SchedMxfp4Mxfp8DownFixedRing,
    SchedMxfp4Mxfp8GateUpSiluFixedRing,
    SchedMxfp4Mxfp8ResidentFfn,
)
from deepseek_v4_cold_timing import cold_graph_timings_us, percentile_us


FP4_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def f32_bits(value: float) -> int:
    return int(
        torch.tensor(value, dtype=torch.float32).view(torch.int32)
    ) & 0xFFFFFFFF


def uniform_linear1_reference(
    weight_byte: int,
    weight_scale_byte: int,
    activation_row_bytes: torch.Tensor,
    activation_scale_byte: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight_scale = 2.0 ** (weight_scale_byte - 127)
    activation_scale = 2.0 ** (activation_scale_byte - 127)
    weight_sum = 2048.0 * (
        FP4_VALUES[weight_byte & 0xF]
        + FP4_VALUES[weight_byte >> 4]
    ) * weight_scale
    activation = activation_row_bytes.view(torch.float8_e4m3fn).float()
    gate = activation * activation_scale * weight_sum
    middle = gate / (1.0 + torch.exp(-gate)) * gate
    requested = (middle.abs() / 448.0).clamp_min(2.0**-127)
    exponents = torch.ceil(torch.log2(requested)).clamp(-127, 127)
    scales = torch.exp2(exponents)
    quantized = (middle / scales).clamp(-448.0, 448.0)
    return (
        quantized.to(torch.float8_e4m3fn).view(torch.uint8),
        (exponents.to(torch.int16) + 127).to(torch.uint8),
        quantized.to(torch.float8_e4m3fn).float() * scales,
    )


def span_us(profile: torch.Tensor, start: int, stop: int) -> float:
    values = profile.cpu().numpy()
    return (
        int(values[:, stop].max()) - int(values[:, start].min())
    ) / 1.0e3


def relative_finish_us(
    profile: torch.Tensor, start: int, finish: int
) -> tuple[float, float, float]:
    values = profile.cpu().numpy()
    origin = int(values[:, start].min())
    finishes = sorted((int(value) - origin) / 1.0e3 for value in values[:, finish])
    return finishes[0], statistics.median(finishes), finishes[-1]


def local_duration_us(
    profile: torch.Tensor, start: int, stop: int
) -> tuple[float, float, float]:
    values = profile.cpu().numpy()
    durations = sorted(
        (int(end) - int(begin)) / 1.0e3
        for begin, end in zip(values[:, start], values[:, stop])
    )
    return durations[0], statistics.median(durations), durations[-1]


def local_tail_us(
    profile: torch.Tensor, start: int, stop: int
) -> tuple[float, float, float]:
    values = profile.cpu().numpy()
    durations = [
        (int(end) - int(begin)) / 1.0e3
        for begin, end in zip(values[:, start], values[:, stop])
    ]
    median = statistics.median(durations)
    p95 = percentile(durations, 0.95)
    return statistics.pstdev(durations), p95, p95 - median


def event_skew_us(profile: torch.Tensor, event: int) -> float:
    values = profile.cpu().numpy()[:, event]
    return (int(values.max()) - int(values.min())) / 1.0e3


def report_track_counters(profile: torch.Tensor) -> None:
    values = profile.cpu().numpy()
    if not all(
        int(value) == 0x4454524B50524631 for value in values[:, 127]
    ):
        return
    base = runtime.config.track_profile_event_base
    span = int(values[:, 1].max()) - int(values[:, 0].min())
    envelope = span * values.shape[0]

    def total(offset: int) -> int:
        return sum(int(value) for value in values[:, base + offset])

    def percent(offset: int) -> float:
        return 100.0 * total(offset) / envelope if envelope else 0.0

    effective_ghz = [
        (int(clock_end) - int(clock_start))
        / (int(timer_end) - int(timer_start))
        for timer_start, timer_end, clock_start, clock_end in zip(
            values[:, 0], values[:, 1], values[:, 122], values[:, 123]
        )
        if int(timer_end) > int(timer_start)
    ]
    startup_us = [
        (int(post_init) - int(entry)) / 1.0e3
        for entry, post_init in zip(values[:, 124], values[:, 0])
    ]
    join_tail_us = [
        (int(join) - int(compute_end)) / 1.0e3
        for compute_end, join in zip(values[:, 1], values[:, 125])
    ]
    free_us = [
        (int(post_free) - int(join)) / 1.0e3
        for join, post_free in zip(values[:, 125], values[:, 126])
    ]
    device_envelope_us = [
        (int(post_free) - int(entry)) / 1.0e3
        for entry, post_free in zip(values[:, 124], values[:, 126])
    ]

    print(
        "DSV4_MXFP4_MXFP8_FULL_FFN_RETAINED_RING_COUNTERS "
        f"compute_m2c_wait_pct={percent(0):.3f} "
        f"allocator_slot_stall_pct={percent(3):.3f} "
        f"allocator_slot_stall_events={total(4)} "
        f"allocator_slot_retries={total(5)} "
        f"allocator_instructions={total(8)} "
        f"ldu0_queue_wait_pct={percent(9):.3f} "
        f"ldu0_dependency_wait_pct={percent(11):.3f} "
        f"ldu0_commands={total(13)} "
        f"ldu1_queue_wait_pct={percent(14):.3f} "
        f"ldu1_dependency_wait_pct={percent(16):.3f} "
        f"ldu1_commands={total(18)} "
        f"startup_median_us={statistics.median(startup_us):.6f} "
        f"join_tail_median_us={statistics.median(join_tail_us):.6f} "
        f"tmem_free_median_us={statistics.median(free_us):.6f} "
        "device_envelope_median_us="
        f"{statistics.median(device_envelope_us):.6f} "
        f"effective_sm_clock_median_ghz="
        f"{statistics.median(effective_ghz):.6f}",
        flush=True,
    )


def report_sm_timeline(
    profile: torch.Tensor,
    down_schedule: SchedMxfp4Mxfp8DownFixedRing,
) -> None:
    values = profile.cpu().numpy()
    origin = int(values[:, 2].min())
    rows = []
    for worker in range(values.shape[0]):
        linear1_local = (int(values[worker, 4]) - int(values[worker, 2])) / 1.0e3
        down_local = (int(values[worker, 5]) - int(values[worker, 4])) / 1.0e3
        task_local = (int(values[worker, 3]) - int(values[worker, 2])) / 1.0e3
        finish = (int(values[worker, 3]) - origin) / 1.0e3
        rows.append((worker, linear1_local, down_local, task_local, finish))

    slices_per_expert = 16
    for expert in range(len(rows) // slices_per_expert):
        group = rows[
            expert * slices_per_expert : (expert + 1) * slices_per_expert
        ]

        def group_stats(index: int) -> tuple[float, float]:
            samples = [row[index] for row in group]
            return statistics.median(samples), max(samples)

        linear1_stats = group_stats(1)
        down_stats = group_stats(2)
        task_stats = group_stats(3)
        print(
            "DSV4_MXFP4_MXFP8_FULL_FFN_SM_GROUP "
            f"linear1_expert={expert} "
            f"worker_start={expert * slices_per_expert} "
            f"linear1_local_median_us={linear1_stats[0]:.6f} "
            f"linear1_local_max_us={linear1_stats[1]:.6f} "
            f"down_local_median_us={down_stats[0]:.6f} "
            f"down_local_max_us={down_stats[1]:.6f} "
            f"task_local_median_us={task_stats[0]:.6f} "
            f"task_local_max_us={task_stats[1]:.6f}",
            flush=True,
        )

    # Down traverses K in Linear-1 record order, two records per K256 tile.
    # Group producer finishes by record index so a persistently late pair is
    # visible independently of the expert and physical worker placement.
    for local_slice in range(slices_per_expert):
        workers = range(local_slice, values.shape[0], slices_per_expert)
        local_samples = [
            (int(values[worker, 4]) - int(values[worker, 2])) / 1.0e3
            for worker in workers
        ]
        finish_samples = [
            (int(values[worker, 4]) - origin) / 1.0e3
            for worker in workers
        ]
        print(
            "DSV4_MXFP4_MXFP8_LINEAR1_RECORD_FINISH "
            f"record={local_slice} samples={len(local_samples)} "
            f"local_median_us={statistics.median(local_samples):.6f} "
            f"local_max_us={max(local_samples):.6f} "
            f"finish_median_us={statistics.median(finish_samples):.6f} "
            f"finish_max_us={max(finish_samples):.6f}",
            flush=True,
        )

    for worker, linear1_local, down_local, task_local, finish in sorted(
        rows, key=lambda row: row[3], reverse=True
    )[:16]:
        print(
            "DSV4_MXFP4_MXFP8_FULL_FFN_SM_TAIL "
            f"worker={worker} linear1_task={worker} "
            f"down_tasks={','.join(map(str, down_schedule.task_queues[worker]))} "
            f"linear1_local_us={linear1_local:.6f} "
            f"down_local_us={down_local:.6f} "
            f"task_local_us={task_local:.6f} "
            f"finish_us={finish:.6f}",
            flush=True,
        )


def down_task_timeline_samples(
    values, base: int
) -> list[list[float]]:
    samples = []
    for worker in range(values.shape[0]):
        entry = int(values[worker, base])
        end = int(values[worker, base + 14])
        if entry <= 0 or end < entry:
            continue
        samples.append(
            [
                (int(values[worker, event]) - entry) / 1.0e3
                for event in range(base, base + 15)
            ]
        )
    return samples


def report_down_task_timeline(profile: torch.Tensor) -> None:
    values = profile.cpu().numpy()
    first_samples = down_task_timeline_samples(values, 35)
    second_samples = down_task_timeline_samples(values, 20)
    if not first_samples or not second_samples:
        raise RuntimeError("Down timeline events were not populated")

    labels = [
        "entry",
        "compute_start",
        *[f"umma_issue_{tile}" for tile in range(8)],
        "umma_complete",
        "output_smem_ready",
        "reduction_ready",
        "reduce_complete",
        "task_end",
    ]
    for task_name, samples in (
        ("first", first_samples),
        ("second", second_samples),
    ):
        medians = [
            statistics.median(sample[index] for sample in samples)
            for index in range(len(labels))
        ]
        p95s = [
            percentile_us([sample[index] for sample in samples], 0.95)
            for index in range(len(labels))
        ]
        print(
            "DSV4_MXFP4_MXFP8_DOWN_TIMELINE "
            f"task={task_name} workers={len(samples)} "
            + " ".join(
                f"{label}_median_us={median:.6f} {label}_p95_us={p95:.6f}"
                for label, median, p95 in zip(labels, medians, p95s)
            ),
            flush=True,
        )
    handoff = [
        (int(values[worker, 20]) - int(values[worker, 49])) / 1.0e3
        for worker in range(values.shape[0])
    ]
    print(
        "DSV4_MXFP4_MXFP8_DOWN_TASK_HANDOFF "
        f"median_us={statistics.median(handoff):.6f} "
        f"p95_us={percentile_us(handoff, 0.95):.6f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--graph-inner", type=int, default=20)
    parser.add_argument("--cold-samples", type=int, default=0)
    parser.add_argument("--cold-l2-scrub-mib", type=int, default=260)
    parser.add_argument("--workers", type=int, default=112)
    parser.add_argument(
        "--resident-all-tma",
        action="store_true",
        help="use one fixed-layout compute/LDU FFN plan per worker",
    )
    parser.add_argument(
        "--down-task-limit",
        type=int,
        default=224,
        help=(
            "diagnostic: enqueue only this shared-first prefix of Down tasks; "
            "the full tensors and selected-op image remain unchanged"
        ),
    )
    parser.add_argument(
        "--report-sm-timeline",
        action="store_true",
        help="report producer-group summaries and the 16 slowest workers",
    )
    parser.add_argument(
        "--report-down-task-timeline",
        action="store_true",
        help="report diagnostic second-Down-task stage timestamps",
    )
    parser.add_argument(
        "--ring-handoff",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="rewrite adjacent retained gate/up and down rings as one lease",
    )
    parser.add_argument(
        "--blockwise-ready",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="let each down K bundle wait only for its Linear-1 records",
    )
    parser.add_argument(
        "--prebuilt-down-input",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "diagnostic control: make Down consume a separate already-ready "
            "native activation tensor while Linear-1 still publishes output"
        ),
    )
    parser.add_argument(
        "--weight-byte", type=lambda value: int(value, 0), default=0x66
    )
    parser.add_argument("--weight-scale", type=int, default=125)
    parser.add_argument(
        "--down-weight-byte", type=lambda value: int(value, 0), default=0x22
    )
    parser.add_argument("--down-weight-scale", type=int, default=124)
    parser.add_argument(
        "--activation-byte", type=lambda value: int(value, 0), default=0x60
    )
    parser.add_argument("--activation-scale", type=int, default=119)
    parser.add_argument(
        "--tma-l2-promotion", choices=("none", "64", "128", "256"),
        default="256",
    )
    args = parser.parse_args()
    os.environ["DAE_TMA_L2_PROMOTION"] = args.tma_l2_promotion
    if args.graph_inner <= 0:
        parser.error("--graph-inner must be positive")
    if not 32 <= args.workers <= 112:
        parser.error("--workers must be in [32,112]")
    weight_prefetch = bool(runtime.config.mxfp_weight_prefetch)
    weight_scale_tma = bool(runtime.config.mxfp_weight_scale_tma)
    gate_up_weight_scale_separate_barrier = bool(
        runtime.config.mxfp_gate_up_weight_scale_separate_barrier
    )
    down_weight_scale_separate_barrier = bool(
        runtime.config.mxfp_down_weight_scale_separate_barrier
    )
    overlap_down_prefetch = bool(
        runtime.config.mxfp_resident_ffn_overlap_down_prefetch
    )
    resident_down_pair_zero = bool(
        runtime.config.mxfp_resident_down_pair_zero
    )
    resident_down_split_ldu = bool(
        runtime.config.mxfp_resident_down_split_ldu
    )
    resident_fast_queue_init = bool(
        runtime.config.mxfp_resident_fast_queue_init
    )
    fast_memory_dispatch = bool(
        runtime.config.mxfp_resident_ffn_fast_memory_dispatch
    )
    ldu1_zero = bool(runtime.config.mxfp_resident_down_ldu1_zero)
    if not 0 <= args.down_task_limit <= 224:
        parser.error("--down-task-limit must be in [0,224]")
    for name, value in (
        ("weight-byte", args.weight_byte),
        ("weight-scale", args.weight_scale),
        ("down-weight-byte", args.down_weight_byte),
        ("down-weight-scale", args.down_weight_scale),
        ("activation-byte", args.activation_byte),
        ("activation-scale", args.activation_scale),
    ):
        if not 0 <= value <= 0xFF:
            parser.error(f"--{name} must fit uint8")
    if args.activation_byte > 0xF8:
        parser.error("--activation-byte must leave room for eight rows")
    if args.ring_handoff and not (
        bool(runtime.config.mxfp_gate_up_ldu_weight_ring)
        and bool(runtime.config.mxfp_down_ldu_weight_ring)
    ):
        parser.error("--ring-handoff requires both retained LDU ring paths")
    if args.resident_all_tma and args.ring_handoff:
        parser.error("resident all-TMA FFN does not use allocator ring handoff")
    if args.resident_all_tma and args.down_task_limit != 224:
        parser.error("resident all-TMA FFN currently requires all 224 Down tasks")
    device = torch.device("cuda")
    experts, linear1_slices, down_slices = 7, 16, 32
    linear1_tasks = experts * linear1_slices
    down_tasks = experts * down_slices
    tile_k = 512
    k_tiles = 4096 // tile_k
    k128_per_tile = tile_k // 128
    launcher = Launcher(args.workers, device=device)
    launcher.rewrite_retained_weight_ring_handoffs = args.ring_handoff

    ready_bars = [launcher.new_bar(1) for _ in range(linear1_tasks)]
    zero_ready = [launcher.new_bar(1) for _ in range(down_slices)]
    gate_weight = torch.full(
        (linear1_tasks, k_tiles, k128_per_tile, 128, 64),
        args.weight_byte,
        dtype=torch.uint8,
        device=device,
    )
    up_weight = torch.full_like(gate_weight, args.weight_byte)
    gate_scale = torch.full(
        (linear1_tasks, k_tiles, k128_per_tile * 512),
        args.weight_scale,
        dtype=torch.uint8,
        device=device,
    )
    up_scale = torch.full_like(gate_scale, args.weight_scale)
    activation_row_bytes = (
        torch.arange(8, dtype=torch.uint8) + args.activation_byte
    )
    expected_data, expected_scale, dequantized_middle = (
        uniform_linear1_reference(
            args.weight_byte,
            args.weight_scale,
            activation_row_bytes,
            args.activation_scale,
        )
    )
    activation_data = (
        activation_row_bytes.to(device)
        .reshape(1, 1, 8, 1)
        .expand(k_tiles, k128_per_tile, 8, 128)
        .contiguous()
        .reshape(k_tiles, 4096)
    )
    activation_scale = torch.full(
        (k_tiles, 2048),
        args.activation_scale,
        dtype=torch.uint8,
        device=device,
    )
    activation_records = torch.empty(
        (experts, linear1_slices, 1536), dtype=torch.uint8, device=device
    )
    activation_records_flat = activation_records.view(linear1_tasks, 1536)
    activation_output_data = activation_records_flat[:, :1024]
    activation_output_scale = activation_records_flat[:, 1024:]
    down_activation_records = activation_records
    if args.prebuilt_down_input:
        down_activation_records = torch.zeros_like(activation_records)
        prebuilt_data = (
            expected_data.reshape(8, 1)
            .expand(8, 128)
            .reshape(1, 1, -1)
            .to(device)
        )
        down_activation_records[..., :1024].copy_(prebuilt_data)
        active_scale_indices = (
            torch.arange(8, device=device).reshape(-1, 1) * 16
            + torch.arange(4, device=device).reshape(1, -1)
        ).reshape(-1)
        down_activation_records[
            ..., 1024 + active_scale_indices
        ] = expected_scale.repeat_interleave(4).to(device)

    gate_tma = TmaTensor(launcher, gate_weight).mxfp4_load(tile_k)
    up_tma = TmaTensor(launcher, up_weight).mxfp4_load(tile_k)
    linear1_records = torch.zeros(
        (linear1_tasks, 16), dtype=torch.int64, device="cpu"
    )
    for task in range(linear1_tasks):
        linear1_records[task, 0] = activation_data.data_ptr()
        linear1_records[task, 2] = gate_scale[task, 0].data_ptr()
        linear1_records[task, 3] = activation_scale[0].data_ptr()
        linear1_records[task, 4] = up_scale[task, 0].data_ptr()
        linear1_records[task, 5] = (
            gate_tma.arg | (up_tma.arg << 16) | (task << 32)
        )
        linear1_records[task, 6] = activation_records_flat[task].data_ptr()
        linear1_records[task, 7] = k128_per_tile * 512
        linear1_records[task, 8] = ready_bars[task]
    linear1_metadata = linear1_records.view(torch.uint8).to(device)

    down_weight = torch.full(
        (down_tasks, 8, 2, 128, 64),
        args.down_weight_byte,
        dtype=torch.uint8,
        device=device,
    )
    down_scale = torch.full(
        (down_tasks, 8, 1024),
        args.down_weight_scale,
        dtype=torch.uint8,
        device=device,
    )
    reduction_bf16 = bool(
        getattr(runtime.config, "mxfp_down_bf16_reduction", False)
    )
    reduction_dtype = torch.bfloat16 if reduction_bf16 else torch.float32
    final_output = torch.empty(
        (down_slices, 128, 8), dtype=reduction_dtype, device=device
    )
    down_tma = TmaTensor(launcher, down_weight).mxfp4_load(256)
    output_tma = TmaTensor(
        launcher, final_output.view(down_slices * 128, 8)
    ).rowmajor_2d("reduce", 128, 8)
    route_scales = [1.0, *([1.0 / 6.0] * 6)]
    down_records = torch.zeros(
        (down_tasks, 16), dtype=torch.int64, device="cpu"
    )
    for task in range(down_tasks):
        expert, output_tile = divmod(task, down_slices)
        down_records[task, 0] = down_scale[task, 0].data_ptr()
        down_records[task, 1] = down_activation_records[
            expert, 0
        ].data_ptr()
        down_records[task, 3] = (
            down_tma.arg | (output_tma.arg << 16) | (task << 32)
        )
        ready_bar = (
            0xFFFFFFFF
            if args.prebuilt_down_input
            else ready_bars[expert * linear1_slices]
        )
        down_records[task, 4] = ready_bar | (
            zero_ready[output_tile] << 32
        )
        down_records[task, 5] = f32_bits(route_scales[expert])
        down_records[task, 6] = final_output[output_tile].data_ptr()
        flags = 1 | (4 if args.blockwise_ready else 0)
        if resident_down_pair_zero and expert == 0:
            if output_tile < down_slices // 2:
                paired_output_tile = output_tile + down_slices // 2
                down_records[task, 9] = final_output[
                    paired_output_tile
                ].data_ptr()
                down_records[task, 10] = zero_ready[paired_output_tile]
                flags |= 8
            else:
                flags |= 16
        down_records[task, 8] = flags << 32
    down_metadata = down_records.view(torch.uint8).to(device)

    linear1_schedule_base = SchedMxfp4Mxfp8GateUpSiluFixedRing(
        gate_weight,
        gate_scale,
        up_weight,
        up_scale,
        activation_data,
        activation_scale,
        activation_output_data,
        activation_output_scale,
        gate_tma,
        up_tma,
        linear1_metadata,
        tile_k=tile_k,
    )
    down_schedule_base = SchedMxfp4Mxfp8DownFixedRing(
        down_weight,
        down_scale,
        down_activation_records,
        final_output,
        down_tma,
        down_metadata,
        retain_weight_ring_between_tasks=True,
    )
    if args.resident_all_tma:
        resident_schedule = SchedMxfp4Mxfp8ResidentFfn(
            linear1_schedule_base, down_schedule_base
        ).place(args.workers)
        linear1_schedule = resident_schedule.placed_linear1
        down_schedule = resident_schedule.placed_down
    else:
        linear1_schedule = linear1_schedule_base.place(args.workers)
        down_schedule = down_schedule_base.place(args.workers)
        if args.down_task_limit != down_tasks:
            for queue in down_schedule.task_queues:
                queue[:] = [
                    task for task in queue if task < args.down_task_limit
                ]
    queued_down_tasks = [
        task for queue in down_schedule.task_queues for task in queue
    ]
    down_active_workers = sum(
        bool(queue) for queue in down_schedule.task_queues
    )
    if args.resident_all_tma:
        launcher.s(resident_schedule)
    else:
        launcher.s(
            ProfileEvent(2),
            linear1_schedule,
            ProfileEvent(4),
            down_schedule,
            ProfileEvent(5),
            ProfileEvent(3),
        )
    launcher.build_instructions()
    flag_mask = (1 << 6) - 1
    handoff_source = (
        opcode.OP_ALLOC_TMA_LOAD_MX_WEIGHT_RING_HANDOFF_5D & ~flag_mask
    )
    handoff_target = (
        opcode.OP_TMA_LOAD_MX_DOWN_WEIGHT_RING_HANDOFF_5D & ~flag_mask
    )
    source_count = sum(
        (inst.opcode & ~flag_mask) == handoff_source
        for builder in launcher.builder
        for inst in builder.built_minsts
    )
    target_count = sum(
        (inst.opcode & ~flag_mask) == handoff_target
        for builder in launcher.builder
        for inst in builder.built_minsts
    )
    expected_handoffs = down_active_workers if args.ring_handoff else 0
    if source_count != expected_handoffs or target_count != expected_handoffs:
        raise RuntimeError(
            "unexpected retained-ring rewrite count: "
            f"sources={source_count} targets={target_count} "
            f"expected={expected_handoffs}"
        )

    stream = torch.cuda.Stream()

    def enqueue() -> None:
        with torch.cuda.stream(stream):
            launcher.launch(synchronize=False)

    stream.wait_stream(torch.cuda.current_stream())
    enqueue()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    expected_active_data = (
        expected_data.reshape(8, 1)
        .expand(8, 128)
        .reshape(1, -1)
        .to(device)
        .expand(linear1_tasks, -1)
    )
    torch.testing.assert_close(
        activation_output_data, expected_active_data, rtol=0, atol=0
    )
    active_scale_indices = (
        torch.arange(8, device=device).reshape(-1, 1) * 16
        + torch.arange(4, device=device).reshape(1, -1)
    ).reshape(-1)
    expected_active_scale = (
        expected_scale.repeat_interleave(4)
        .reshape(1, -1)
        .to(device)
        .expand(linear1_tasks, -1)
    )
    torch.testing.assert_close(
        activation_output_scale[:, active_scale_indices],
        expected_active_scale,
        rtol=0,
        atol=0,
    )
    down_weight_scale_value = 2.0 ** (args.down_weight_scale - 127)
    down_weight_sum = 1024.0 * (
        FP4_VALUES[args.down_weight_byte & 0xF]
        + FP4_VALUES[args.down_weight_byte >> 4]
    ) * down_weight_scale_value
    route_sum_by_output = torch.zeros(down_slices, dtype=torch.float32)
    for task in queued_down_tasks:
        expert, output_tile = divmod(task, down_slices)
        route_sum_by_output[output_tile] += route_scales[expert]
    expected_final = (
        (dequantized_middle * down_weight_sum)
        .reshape(1, 1, 8)
        .to(device)
        * route_sum_by_output.to(device).reshape(down_slices, 1, 1)
    ).expand(down_slices, 128, 8)
    checked_outputs = route_sum_by_output != 0
    checked_outputs_device = checked_outputs.to(device)
    if bool(checked_outputs.any()):
        torch.testing.assert_close(
            final_output[checked_outputs_device].float(),
            expected_final[checked_outputs_device],
            rtol=3e-2 if reduction_bf16 else 2e-5,
            atol=1e-1 if reduction_bf16 else 1e-3,
        )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(args.graph_inner):
            enqueue()
    for _ in range(args.warmup):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(args.iterations):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop) * 1.0e3 / args.graph_inner)

    cold_times = None
    if args.cold_samples:
        cold_times = cold_graph_timings_us(
            enqueue,
            stream=stream,
            warmup=args.warmup,
            samples=args.cold_samples,
            l2_scrub_mib=args.cold_l2_scrub_mib,
        )

    if bool(checked_outputs.any()):
        torch.testing.assert_close(
            final_output[checked_outputs_device].float(),
            expected_final[checked_outputs_device],
            rtol=3e-2 if reduction_bf16 else 2e-5,
            atol=1e-1 if reduction_bf16 else 1e-3,
        )
    profile = launcher.profile
    linear1_profile = profile[:linear1_tasks]
    linear1_finish = relative_finish_us(linear1_profile, 2, 4)
    down_finish = relative_finish_us(profile, 2, 5)
    linear1_local = local_duration_us(linear1_profile, 2, 4)
    down_local = local_duration_us(profile, 4, 5)
    task_local = local_duration_us(profile, 2, 3)
    linear1_tail = local_tail_us(linear1_profile, 2, 4)
    down_tail = local_tail_us(profile, 4, 5)
    if bool(checked_outputs.any()):
        error = (
            final_output[checked_outputs_device].float()
            - expected_final[checked_outputs_device]
        ).abs()
        relative_error = error / expected_final[
            checked_outputs_device
        ].abs().clamp_min(torch.finfo(torch.float32).tiny)
        max_abs_error = float(error.max())
        max_rel_error = float(relative_error.max())
    else:
        max_abs_error = 0.0
        max_rel_error = 0.0
    report_track_counters(profile)
    if args.report_sm_timeline:
        report_sm_timeline(profile, down_schedule)
    if args.report_down_task_timeline:
        report_down_task_timeline(profile)
    profile_values = profile.cpu().numpy()
    linear1_samples = [
        (int(end) - int(begin)) / 1.0e3
        for begin, end in zip(
            profile_values[:linear1_tasks, 2],
            profile_values[:linear1_tasks, 4],
        )
    ]
    exposed_linear1 = [
        value
        for value, queue in zip(
            linear1_samples, down_schedule.task_queues[:linear1_tasks]
        )
        if queue
    ]
    unexposed_linear1 = [
        value
        for value, queue in zip(
            linear1_samples, down_schedule.task_queues[:linear1_tasks]
        )
        if not queue
    ]

    def exposure_fields(name: str, samples: list[float]) -> str:
        if not samples:
            return (
                f"{name}_count=0 {name}_median_us=-1.000000 "
                f"{name}_stddev_us=-1.000000 {name}_p95_us=-1.000000"
            )
        return (
            f"{name}_count={len(samples)} "
            f"{name}_median_us={statistics.median(samples):.6f} "
            f"{name}_stddev_us={statistics.pstdev(samples):.6f} "
            f"{name}_p95_us={percentile(samples, 0.95):.6f}"
        )

    print(
        "DSV4_MXFP4_MXFP8_FULL_FFN_DOWN_EXPOSURE "
        f"down_tasks_queued={len(queued_down_tasks)} "
        f"down_active_workers={down_active_workers} "
        f"{exposure_fields('exposed_linear1', exposed_linear1)} "
        f"{exposure_fields('unexposed_linear1', unexposed_linear1)}",
        flush=True,
    )
    if cold_times is not None:
        print(
            "DSV4_MXFP4_MXFP8_FULL_FFN_RETAINED_RING_COLD_RESULT "
            f"workers={args.workers} linear1_tasks={linear1_tasks} "
            f"down_tasks={down_tasks} "
            "timing=cold_data_one_ffn_graph "
            f"l2_scrub_mib={args.cold_l2_scrub_mib} "
            f"samples={args.cold_samples} "
            f"weight_prefetch={str(weight_prefetch).lower()} "
            f"weight_scale_tma={str(weight_scale_tma).lower()} "
            "gate_up_weight_scale_separate_barrier="
            f"{str(gate_up_weight_scale_separate_barrier).lower()} "
            "down_weight_scale_separate_barrier="
            f"{str(down_weight_scale_separate_barrier).lower()} "
            f"min_us={min(cold_times):.6f} "
            f"median_us={statistics.median(cold_times):.6f} "
            f"p90_us={percentile_us(cold_times, 0.90):.6f} "
            f"stddev_us={statistics.pstdev(cold_times):.6f} "
            f"max_us={max(cold_times):.6f} "
            "output_correct=true",
            flush=True,
        )
    print(
        "DSV4_MXFP4_MXFP8_FULL_FFN_RETAINED_RING_RESULT "
        f"workers={args.workers} linear1_tasks={linear1_tasks} "
        f"down_tasks={down_tasks} down_tasks_queued={len(queued_down_tasks)} "
        f"down_active_workers={down_active_workers} "
        "cuda_kernel_launches=1 vdcores_launches=1 "
        "persistent=true "
        f"resident_all_tma={str(args.resident_all_tma).lower()} "
        "kernel=dae2 "
        "linear1_ldu_weight_ring=true "
        "down_ldu_weight_ring=true "
        "down_ring_stages="
        f"{int(runtime.config.mxfp_down_ldu_weight_ring_stages)} "
        "activation_scales_task_owned="
        f"{str(not args.resident_all_tma).lower()} "
        f"down_bf16_reduction={str(reduction_bf16).lower()} "
        f"overlap_down_prefetch={str(overlap_down_prefetch).lower()} "
        f"resident_down_pair_zero={str(resident_down_pair_zero).lower()} "
        f"resident_down_split_ldu={str(resident_down_split_ldu).lower()} "
        f"resident_fast_queue_init={str(resident_fast_queue_init).lower()} "
        f"fast_memory_dispatch={str(fast_memory_dispatch).lower()} "
        f"ldu1_zero={str(ldu1_zero).lower()} "
        f"ring_handoff={str(args.ring_handoff).lower()} "
        f"handoff_sources={source_count} handoff_targets={target_count} "
        f"blockwise_ready={str(args.blockwise_ready).lower()} "
        f"weight_prefetch={str(weight_prefetch).lower()} "
        f"weight_scale_tma={str(weight_scale_tma).lower()} "
        "gate_up_weight_scale_separate_barrier="
        f"{str(gate_up_weight_scale_separate_barrier).lower()} "
        "down_weight_scale_separate_barrier="
        f"{str(down_weight_scale_separate_barrier).lower()} "
        f"down_input={'prebuilt' if args.prebuilt_down_input else 'linear1'} "
        f"allocator_slots={runtime.config.num_slots} "
        f"linear1_span_us={span_us(linear1_profile, 2, 4):.6f} "
        f"down_span_us={span_us(profile, 4, 5):.6f} "
        f"task_span_us={span_us(profile, 2, 3):.6f} "
        f"kernel_us={span_us(profile, 0, 1):.6f} "
        f"linear1_start_skew_us={event_skew_us(linear1_profile, 2):.6f} "
        f"linear1_local_min_us={linear1_local[0]:.6f} "
        f"linear1_local_median_us={linear1_local[1]:.6f} "
        f"linear1_local_stddev_us={linear1_tail[0]:.6f} "
        f"linear1_local_p95_us={linear1_tail[1]:.6f} "
        f"linear1_local_p95_tail_us={linear1_tail[2]:.6f} "
        f"linear1_local_max_us={linear1_local[2]:.6f} "
        f"down_local_min_us={down_local[0]:.6f} "
        f"down_local_median_us={down_local[1]:.6f} "
        f"down_local_stddev_us={down_tail[0]:.6f} "
        f"down_local_p95_us={down_tail[1]:.6f} "
        f"down_local_p95_tail_us={down_tail[2]:.6f} "
        f"down_local_max_us={down_local[2]:.6f} "
        f"task_local_min_us={task_local[0]:.6f} "
        f"task_local_median_us={task_local[1]:.6f} "
        f"task_local_max_us={task_local[2]:.6f} "
        f"linear1_finish_min_us={linear1_finish[0]:.6f} "
        f"linear1_finish_median_us={linear1_finish[1]:.6f} "
        f"linear1_finish_max_us={linear1_finish[2]:.6f} "
        f"down_finish_min_us={down_finish[0]:.6f} "
        f"down_finish_median_us={down_finish[1]:.6f} "
        f"down_finish_max_us={down_finish[2]:.6f} "
        f"end_to_end_min_us={min(times):.6f} "
        f"end_to_end_median_us={statistics.median(times):.6f} "
        f"end_to_end_max_us={max(times):.6f} "
        f"max_abs_error={max_abs_error:.8f} "
        f"max_rel_error={max_rel_error:.8f} "
        f"down_output_checked={str(bool(checked_outputs.any())).lower()} "
        "output_correct=true",
        flush=True,
    )


if __name__ == "__main__":
    main()
