#!/usr/bin/env python3
"""Standalone retained-weight-ring MXFP4/MXFP8 down benchmark."""

from __future__ import annotations

import argparse
import os
import statistics

import torch

from dae import runtime
from dae.instructions import ProfileEvent, TmaTensor
from dae.launcher import Launcher
from dae.schedule import (
    SchedMxfp4Mxfp8DownFixedRing,
    SchedMxfp4Mxfp8DownResident,
)


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


def profile_span_us(profile: torch.Tensor, start: int, stop: int) -> float:
    values = profile.cpu().numpy()
    return (int(values[:, stop].max()) - int(values[:, start].min())) / 1.0e3


def report_track_profile(profile: torch.Tensor) -> None:
    values = profile.cpu().numpy()
    if not all(int(value) == 0x4454524B50524631 for value in values[:, 127]):
        return
    base = runtime.config.track_profile_event_base
    span = max(int(value) for value in values[:, 1]) - min(
        int(value) for value in values[:, 0]
    )
    envelope = span * values.shape[0]

    def total(offset: int) -> int:
        return sum(int(value) for value in values[:, base + offset])

    def percent(offset: int) -> float:
        return 100.0 * total(offset) / envelope if envelope > 0 else 0.0

    print(
        "DSV4_MXFP4_MXFP8_DOWN_COUNTERS "
        f"internal_span_us={span / 1.0e3:.6f} "
        f"compute_m2c_wait_pct={percent(0):.3f} "
        f"allocator_slot_stall_pct={percent(3):.3f} "
        f"ldu0_queue_wait_pct={percent(9):.3f} "
        f"ldu0_dependency_wait_pct={percent(11):.3f} "
        f"ldu1_queue_wait_pct={percent(14):.3f} "
        f"ldu1_dependency_wait_pct={percent(16):.3f} "
        f"allocator_instructions={total(8)} "
        f"allocator_slot_stall_events={total(4)} "
        f"ldu0_commands={total(13)} ldu1_commands={total(18)}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--graph-inner", type=int, default=20)
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--resident-all-tma",
        action="store_true",
        help="use one fixed-layout resident LDU plan per worker",
    )
    parser.add_argument(
        "--dependency-bars",
        action="store_true",
        help=(
            "wait on already-ready per-record activation barriers while "
            "leaving retained-weight commands independent"
        ),
    )
    parser.add_argument(
        "--blockwise-ready",
        action="store_true",
        help="check each K256 bundle's two activation-record barriers",
    )
    parser.add_argument(
        "--weight-ring-chain",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="allocate one memory-owned ring per worker and reuse it across tasks",
    )
    parser.add_argument(
        "--weight-byte", type=lambda value: int(value, 0), default=0x22
    )
    parser.add_argument("--weight-scale", type=int, default=124)
    parser.add_argument(
        "--activation-byte", type=lambda value: int(value, 0), default=0x38
    )
    parser.add_argument("--activation-scale", type=int, default=127)
    parser.add_argument(
        "--tma-l2-promotion", choices=("none", "64", "128", "256"),
        default="256",
    )
    args = parser.parse_args()
    os.environ["DAE_TMA_L2_PROMOTION"] = args.tma_l2_promotion
    if args.graph_inner <= 0:
        parser.error("--graph-inner must be positive")
    if args.blockwise_ready and not args.dependency_bars:
        parser.error("--blockwise-ready requires --dependency-bars")
    for name, value in (
        ("weight-byte", args.weight_byte),
        ("weight-scale", args.weight_scale),
        ("activation-byte", args.activation_byte),
        ("activation-scale", args.activation_scale),
    ):
        if not 0 <= value <= 0xFF:
            parser.error(f"--{name} must fit uint8")
    ldu_weight_ring = bool(
        getattr(runtime.config, "mxfp_down_ldu_weight_ring", False)
    )
    weight_ring_chain = (
        ldu_weight_ring
        if args.weight_ring_chain is None
        else args.weight_ring_chain
    )
    if weight_ring_chain and not ldu_weight_ring:
        parser.error("--weight-ring-chain requires retained LDU down weights")

    device = torch.device("cuda")
    experts, activation_slices, output_slices = 7, 16, 32
    tasks = experts * output_slices
    physical_sms = torch.cuda.get_device_properties(device).multi_processor_count
    workers = min(physical_sms, tasks) if args.workers is None else args.workers
    if not output_slices <= workers <= min(physical_sms, tasks):
        parser.error(
            f"--workers must be in [{output_slices},{min(physical_sms, tasks)}]"
        )

    launcher = Launcher(workers, device=device)
    dependency_bars = (
        [
            launcher.new_bar(0)
            for _ in range(experts * activation_slices)
        ]
        if args.dependency_bars
        else None
    )
    zero_ready = [launcher.new_bar(1) for _ in range(output_slices)]
    weight_data = torch.full(
        (tasks, 8, 2, 128, 64),
        args.weight_byte,
        dtype=torch.uint8,
        device=device,
    )
    weight_scales = torch.full(
        (tasks, 8, 1024),
        args.weight_scale,
        dtype=torch.uint8,
        device=device,
    )
    activation_records = torch.empty(
        (experts, activation_slices, 1536), dtype=torch.uint8, device=device
    )
    activation_records[..., :1024].fill_(args.activation_byte)
    activation_records[..., 1024:].fill_(args.activation_scale)
    reduction_bf16 = bool(
        getattr(runtime.config, "mxfp_down_bf16_reduction", False)
    )
    reduction_dtype = torch.bfloat16 if reduction_bf16 else torch.float32
    final_output = torch.empty(
        (output_slices, 128, 8), dtype=reduction_dtype, device=device
    )

    weight_tma = TmaTensor(launcher, weight_data).mxfp4_load(256)
    output_tma = TmaTensor(
        launcher, final_output.view(output_slices * 128, 8)
    ).rowmajor_2d("reduce", 128, 8)
    route_scales = [1.0, *([1.0 / 6.0] * 6)]
    records = torch.zeros((tasks, 16), dtype=torch.int64, device="cpu")
    for task in range(tasks):
        expert, output_tile = divmod(task, output_slices)
        records[task, 0] = weight_scales[task, 0].data_ptr()
        records[task, 1] = activation_records[expert, 0].data_ptr()
        records[task, 3] = (
            weight_tma.arg | (output_tma.arg << 16) | (task << 32)
        )
        activation_bar = (
            dependency_bars[expert * activation_slices]
            if dependency_bars is not None
            else 0xFFFFFFFF
        )
        records[task, 4] = activation_bar | (zero_ready[output_tile] << 32)
        records[task, 5] = f32_bits(route_scales[expert])
        records[task, 6] = final_output[output_tile].data_ptr()
        flags = 1 | (4 if args.blockwise_ready else 0)
        records[task, 8] = flags << 32
    metadata = records.view(torch.uint8).to(device)
    schedule_args = {}
    if ldu_weight_ring:
        schedule_args["retain_weight_ring_between_tasks"] = (
            weight_ring_chain
        )
    schedule_type = (
        SchedMxfp4Mxfp8DownResident
        if args.resident_all_tma
        else SchedMxfp4Mxfp8DownFixedRing
    )
    schedule = schedule_type(
        weight_data,
        weight_scales,
        activation_records,
        final_output,
        weight_tma,
        metadata,
        **schedule_args,
    ).place(workers)
    launcher.s(ProfileEvent(2), schedule, ProfileEvent(3))

    stream = torch.cuda.Stream()

    def enqueue() -> None:
        with torch.cuda.stream(stream):
            launcher.launch(synchronize=False)

    stream.wait_stream(torch.cuda.current_stream())
    enqueue()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    activation_value = float(
        torch.tensor([args.activation_byte], dtype=torch.uint8)
        .view(torch.float8_e4m3fn)
        .float()[0]
    )
    activation_scale = 2.0 ** (args.activation_scale - 127)
    packed_weight_sum = (
        FP4_VALUES[args.weight_byte & 0xF]
        + FP4_VALUES[args.weight_byte >> 4]
    )
    weight_scale_value = 2.0 ** (args.weight_scale - 127)
    expected_value = (
        activation_value
        * activation_scale
        * 1024.0
        * packed_weight_sum
        * weight_scale_value
        * sum(route_scales)
    )
    expected = torch.full_like(final_output, expected_value, dtype=torch.float32)
    rtol = 3e-2 if reduction_bf16 else 2e-5
    atol = 1e-1 if reduction_bf16 else 1e-3
    torch.testing.assert_close(final_output.float(), expected, rtol=rtol, atol=atol)

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

    torch.testing.assert_close(final_output.float(), expected, rtol=rtol, atol=atol)
    profile = launcher.profile
    profile_values = profile.cpu().numpy()
    task_local_us = (profile_values[:, 3] - profile_values[:, 2]) / 1.0e3
    task_local_samples = [float(value) for value in task_local_us]
    task_local_median = statistics.median(task_local_samples)
    task_local_p95 = percentile(task_local_samples, 0.95)
    error = (final_output.float() - expected).abs()
    queue_depths = [len(queue) for queue in schedule.task_queues]
    report_track_profile(profile)
    print(
        "DSV4_MXFP4_MXFP8_DOWN_RETAINED_RING_RESULT "
        f"workers={workers} tasks={tasks} max_tasks_per_worker={max(queue_depths)} "
        "cuda_kernel_launches=1 vdcores_launches=1 persistent=true "
        f"resident_all_tma={str(args.resident_all_tma).lower()} "
        f"down_ldu_weight_ring={str(ldu_weight_ring).lower()} "
        "down_ring_stages="
        f"{int(runtime.config.mxfp_down_ldu_weight_ring_stages)} "
        "weight_scale_tma="
        f"{str(bool(runtime.config.mxfp_weight_scale_tma)).lower()} "
        "weight_scale_separate_barrier="
        f"{str(bool(runtime.config.mxfp_down_weight_scale_separate_barrier)).lower()} "
        "activation_scales_task_owned="
        f"{str(not args.resident_all_tma).lower()} "
        f"weight_ring_chain={str(weight_ring_chain).lower()} "
        f"dependency={'activation_records_ready_zero' if args.dependency_bars else 'none'} "
        f"blockwise_ready={str(args.blockwise_ready).lower()} "
        f"allocator_slots={runtime.config.num_slots} "
        f"task_us={profile_span_us(profile, 2, 3):.6f} "
        f"task_local_min_us={float(task_local_us.min()):.6f} "
        f"task_local_median_us={task_local_median:.6f} "
        f"task_local_stddev_us="
        f"{statistics.pstdev(task_local_samples):.6f} "
        f"task_local_p95_us={task_local_p95:.6f} "
        f"task_local_p95_tail_us="
        f"{task_local_p95 - task_local_median:.6f} "
        f"task_local_max_us={float(task_local_us.max()):.6f} "
        f"kernel_us={profile_span_us(profile, 0, 1):.6f} "
        f"end_to_end_min_us={min(times):.6f} "
        f"end_to_end_median_us={statistics.median(times):.6f} "
        f"end_to_end_max_us={max(times):.6f} "
        f"max_abs_error={float(error.max()):.8f} output_correct=true",
        flush=True,
    )


if __name__ == "__main__":
    main()
