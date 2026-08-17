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
)


FP4_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


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
        f"ldu1_commands={total(18)}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--graph-inner", type=int, default=20)
    parser.add_argument("--workers", type=int, default=112)
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
    final_output = torch.empty(
        (down_slices, 128, 8), dtype=torch.float32, device=device
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
        down_records[task, 1] = activation_records[expert, 0].data_ptr()
        down_records[task, 3] = (
            down_tma.arg | (output_tma.arg << 16) | (task << 32)
        )
        down_records[task, 4] = (
            ready_bars[expert * linear1_slices]
            | (zero_ready[output_tile] << 32)
        )
        down_records[task, 5] = f32_bits(route_scales[expert])
        down_records[task, 6] = final_output[output_tile].data_ptr()
        flags = 1 | (4 if args.blockwise_ready else 0)
        down_records[task, 8] = flags << 32
    down_metadata = down_records.view(torch.uint8).to(device)

    linear1_schedule = SchedMxfp4Mxfp8GateUpSiluFixedRing(
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
    ).place(args.workers)
    down_schedule = SchedMxfp4Mxfp8DownFixedRing(
        down_weight,
        down_scale,
        activation_records,
        final_output,
        down_tma,
        down_metadata,
        retain_weight_ring_between_tasks=True,
    ).place(args.workers)
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
    expected_handoffs = args.workers if args.ring_handoff else 0
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

    expected_data, expected_scale, dequantized_middle = (
        uniform_linear1_reference(
            args.weight_byte,
            args.weight_scale,
            activation_row_bytes,
            args.activation_scale,
        )
    )
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
    expected_final = (
        (dequantized_middle * down_weight_sum * sum(route_scales))
        .reshape(1, 1, 8)
        .expand(down_slices, 128, 8)
        .to(device)
    )
    torch.testing.assert_close(
        final_output, expected_final, rtol=2e-5, atol=1e-3
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

    torch.testing.assert_close(
        final_output, expected_final, rtol=2e-5, atol=1e-3
    )
    profile = launcher.profile
    linear1_finish = relative_finish_us(profile, 2, 4)
    down_finish = relative_finish_us(profile, 2, 5)
    error = (final_output - expected_final).abs()
    relative_error = error / expected_final.abs().clamp_min(
        torch.finfo(torch.float32).tiny
    )
    report_track_counters(profile)
    print(
        "DSV4_MXFP4_MXFP8_FULL_FFN_RETAINED_RING_RESULT "
        f"workers={args.workers} linear1_tasks={linear1_tasks} "
        f"down_tasks={down_tasks} cuda_kernel_launches=1 vdcores_launches=1 "
        "persistent=true linear1_ldu_weight_ring=true "
        "down_ldu_weight_ring=true down_scales_task_owned=true "
        f"ring_handoff={str(args.ring_handoff).lower()} "
        f"handoff_sources={source_count} handoff_targets={target_count} "
        f"blockwise_ready={str(args.blockwise_ready).lower()} "
        f"allocator_slots={runtime.config.num_slots} "
        f"linear1_span_us={span_us(profile, 2, 4):.6f} "
        f"down_span_us={span_us(profile, 4, 5):.6f} "
        f"task_span_us={span_us(profile, 2, 3):.6f} "
        f"kernel_us={span_us(profile, 0, 1):.6f} "
        f"linear1_finish_min_us={linear1_finish[0]:.6f} "
        f"linear1_finish_median_us={linear1_finish[1]:.6f} "
        f"linear1_finish_max_us={linear1_finish[2]:.6f} "
        f"down_finish_min_us={down_finish[0]:.6f} "
        f"down_finish_median_us={down_finish[1]:.6f} "
        f"down_finish_max_us={down_finish[2]:.6f} "
        f"end_to_end_min_us={min(times):.6f} "
        f"end_to_end_median_us={statistics.median(times):.6f} "
        f"end_to_end_max_us={max(times):.6f} "
        f"max_abs_error={float(error.max()):.8f} "
        f"max_rel_error={float(relative_error.max()):.8f} "
        "output_correct=true",
        flush=True,
    )


if __name__ == "__main__":
    main()
