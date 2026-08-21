#!/usr/bin/env python3
"""One-launch retained-ring MXFP4/MXFP8 full-FFN benchmark."""

from __future__ import annotations

import argparse
import statistics

import torch

from dae import runtime
from dae.instructions import TmaLoadMxfpCoupledStream, TmaTensor
from dae.launcher import Launcher
from dae.runtime import opcode
from dae.schedule import (
    SchedDsv4RouteTop6,
    SchedMxfp4Mxfp8DownFixedRing,
    SchedMxfp4Mxfp8GateUpSiluFixedRing,
    SchedMxfp4Mxfp8ResidentFfn,
    SchedMxfp4Mxfp8RoutedResidentFfn,
)
from dae.sequential import SequentialProgram, SequentialStage
from deepseek_v4_cold_timing import percentile_us


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


def device_counter_span_us(profile: torch.Tensor) -> float:
    """Return dae2 event-0 to event-1 grid span from GPU globaltimer."""

    values = profile[:, :2].cpu().numpy()
    return (int(values[:, 1].max()) - int(values[:, 0].min())) / 1.0e3


def cold_graph_counter_timings_us(
    run,
    *,
    profile: torch.Tensor,
    stream: torch.cuda.Stream,
    warmup: int,
    samples: int,
    l2_scrub_mib: int,
) -> tuple[list[float], list[float]]:
    """Collect cold CUDA-event and in-dae2 globaltimer spans together."""

    current = torch.cuda.current_stream()
    stream.wait_stream(current)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            graph.replay()
    current.wait_stream(stream)
    torch.cuda.synchronize()

    scrub = torch.zeros(
        l2_scrub_mib * 1024 * 1024,
        dtype=torch.uint8,
        device="cuda",
    )
    stream.wait_stream(current)
    event_times: list[float] = []
    counter_times: list[float] = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            scrub.add_(1)
            start.record(stream)
            graph.replay()
            stop.record(stream)
        stop.synchronize()
        event_times.append(start.elapsed_time(stop) * 1.0e3)
        counter_times.append(device_counter_span_us(profile))
    return event_times, counter_times


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
        f"store_queue_wait_pct={percent(19):.3f} "
        f"store_service_pct={percent(21):.3f} "
        f"store_commands={total(23)} "
        f"effective_sm_clock_median_ghz="
        f"{statistics.median(effective_ghz):.6f}",
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
        "--dynamic-routing",
        action="store_true",
        help=(
            "select routed MX weights through a prepared route record and "
            "write contiguous BF16 [8,4096]"
        ),
    )
    parser.add_argument(
        "--route-with-topk",
        action="store_true",
        help="produce the dynamic route record with top-k in the same dae2 launch",
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
    args = parser.parse_args()
    if args.graph_inner <= 0:
        parser.error("--graph-inner must be positive")
    if args.workers != 112:
        parser.error("the production resident FFN uses exactly 112 workers")
    if args.route_with_topk and not args.dynamic_routing:
        parser.error("--route-with-topk requires --dynamic-routing")
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
    device = torch.device("cuda")
    experts, linear1_slices, down_slices = 7, 16, 32
    linear1_tasks = experts * linear1_slices
    down_tasks = experts * down_slices
    tile_k = 512
    k_tiles = 4096 // tile_k
    k128_per_tile = tile_k // 128
    launcher = Launcher(args.workers, device=device)

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
    route_record = None
    route_indices = None
    route_weights = None
    linear1_stream_weights = None
    linear1_stream_scales = None
    down_stream_weights = None
    down_stream_scales = None
    prepared_route_scores = None
    route_hash_indices = None
    if args.dynamic_routing:
        checkpoint_experts = 256
        stream_experts = checkpoint_experts + 1
        selected_experts = torch.tensor(
            (5, 17, 33, 65, 129, 255), dtype=torch.int32
        )
        selected_weights = torch.arange(1, 7, dtype=torch.float32) / 21.0
        route_record_host = torch.empty((128,), dtype=torch.uint8)
        if not args.route_with_topk:
            route_record_host[:32].view(torch.int32)[:6].copy_(
                selected_experts
            )
            route_record_host[32:64].view(torch.float32)[:6].copy_(
                selected_weights
            )
            route_record_host[64:96].view(torch.int32)[:6].copy_(
                (selected_experts + 1) * linear1_slices
            )
            route_record_host[96:128].view(torch.int32)[:6].copy_(
                (selected_experts + 1) * down_slices
            )
        route_record = route_record_host.to(device)
        route_indices = route_record[:32].view(torch.int32)
        route_weights = route_record[32:64].view(torch.float32)
        if args.route_with_topk:
            prepared_route_scores = torch.empty(
                (256, 2), dtype=torch.float32, device=device
            )
            prepared_route_scores[:, 0] = 1.0
            prepared_route_scores[:, 1] = -1.0e6
            for rank, expert in enumerate(selected_experts.tolist()):
                prepared_route_scores[expert, 0] = float(rank + 1)
                prepared_route_scores[expert, 1] = float(6 - rank)
            route_hash_indices = torch.zeros(
                (8,), dtype=torch.int32, device=device
            )

        linear1_stream_weights = torch.zeros(
            (
                stream_experts * linear1_slices,
                2 * k_tiles,
                k128_per_tile,
                128,
                64,
            ),
            dtype=torch.uint8,
            device=device,
        )
        linear1_stream_scales = torch.full(
            (
                stream_experts * linear1_slices,
                2 * k_tiles,
                k128_per_tile * 512,
            ),
            args.weight_scale,
            dtype=torch.uint8,
            device=device,
        )
        active_stream_experts = (0,) + tuple(
            int(expert) + 1 for expert in selected_experts
        )
        for stream_expert in active_stream_experts:
            task_start = stream_expert * linear1_slices
            linear1_stream_weights[
                task_start : task_start + linear1_slices
            ].fill_(args.weight_byte)
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

    if args.dynamic_routing:
        down_stream_weights = torch.zeros(
            (stream_experts * down_slices, 8, 2, 128, 64),
            dtype=torch.uint8,
            device=device,
        )
        down_stream_scales = torch.full(
            (stream_experts * down_slices, 8, 1024),
            args.down_weight_scale,
            dtype=torch.uint8,
            device=device,
        )
        for stream_expert in active_stream_experts:
            task_start = stream_expert * down_slices
            for output_tile in range(down_slices):
                row_nibbles = 1 + (
                    torch.arange(128, device=device) // 16 + output_tile
                ) % 6
                row_bytes = (row_nibbles | (row_nibbles << 4)).to(
                    torch.uint8
                )
                down_stream_weights[
                    task_start + output_tile
                ].copy_(
                    row_bytes.reshape(1, 1, 128, 1).expand(8, 2, 128, 64)
                )
        down_weight = down_stream_weights[:down_tasks]
        down_scale = down_stream_scales[:down_tasks]
        final_output = torch.empty(
            (8, 4096), dtype=torch.bfloat16, device=device
        )
    else:
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
            (down_slices, 128, 8), dtype=torch.bfloat16, device=device
        )
    down_tma = TmaTensor(launcher, down_weight).mxfp4_load(256)
    if args.dynamic_routing:
        output_tma = TmaTensor(
            launcher, final_output
        ).m128n8_output("reduce")
        route_scales = [
            1.0,
            *(
                selected_weights.tolist()
                if args.route_with_topk
                else route_weights[:6].cpu().tolist()
            ),
        ]
    else:
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
        ready_bar = ready_bars[expert * linear1_slices]
        down_records[task, 4] = ready_bar | (
            zero_ready[output_tile] << 32
        )
        down_records[task, 5] = f32_bits(route_scales[expert])
        down_records[task, 6] = final_output.data_ptr()
        down_records[task, 8] = (1 | 4) << 32
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
        output_n_major=args.dynamic_routing,
    )
    if args.dynamic_routing:
        resident_schedule_base = SchedMxfp4Mxfp8RoutedResidentFfn(
            linear1_schedule_base,
            down_schedule_base,
            route_record,
            linear1_stream_weights,
            linear1_stream_scales,
            down_stream_weights,
            down_stream_scales,
        )
        if args.route_with_topk:
            route_schedule = SchedDsv4RouteTop6(
                prepared_route_scores,
                None,
                route_hash_indices,
                route_indices,
                route_weights,
                route_scale=1.0,
                pretransformed=True,
                packed_output=route_record,
            )
            program = SequentialProgram(
                launcher,
                (
                    SequentialStage(
                        "router_top6",
                        route_schedule,
                        1,
                        release_group="mx_route_ready",
                    ),
                    SequentialStage(
                        "routed_mx_ffn",
                        resident_schedule_base,
                        args.workers,
                        wait_group_roles=(("mx_route_ready", "input"),),
                    ),
                ),
            )
            resident_schedule = program.placed_schedules[1]
            launcher.s(program)
        else:
            resident_schedule = resident_schedule_base.place(args.workers)
            launcher.s(resident_schedule)
    else:
        resident_schedule = SchedMxfp4Mxfp8ResidentFfn(
            linear1_schedule_base, down_schedule_base
        ).place(args.workers)
        launcher.s(resident_schedule)
    linear1_schedule = resident_schedule.placed_linear1
    down_schedule = resident_schedule.placed_down
    queued_down_tasks = [
        task for queue in down_schedule.task_queues for task in queue
    ]
    down_active_workers = sum(
        bool(queue) for queue in down_schedule.task_queues
    )
    launcher.build_instructions()
    flag_mask = (1 << 6) - 1
    coupled_opcode = opcode.OP_TMA_LOAD_MX_COUPLED_STREAM & ~flag_mask
    coupled_commands = [
        inst
        for builder in launcher.builder
        for inst in builder.built_minsts
        if (inst.opcode & ~flag_mask) == coupled_opcode
    ]
    coupled_chain_sources = sum(
        bool(inst.arg & TmaLoadMxfpCoupledStream.LOCAL_CHAIN)
        for inst in coupled_commands
    )
    expected_coupled_commands = 3 * args.workers
    expected_coupled_chains = args.workers
    if (
        len(coupled_commands) != expected_coupled_commands
        or coupled_chain_sources != expected_coupled_chains
    ):
        raise RuntimeError(
            "unexpected coupled-stream lowering: "
            f"commands={len(coupled_commands)} "
            f"chains={coupled_chain_sources} "
            f"expected_commands={expected_coupled_commands} "
            f"expected_chains={expected_coupled_chains}"
        )

    stream = torch.cuda.Stream()

    def enqueue() -> None:
        with torch.cuda.stream(stream):
            final_output.zero_()
            launcher.launch(synchronize=False)

    stream.wait_stream(torch.cuda.current_stream())
    enqueue()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    if args.route_with_topk:
        torch.testing.assert_close(
            route_indices[:6], selected_experts.to(device), rtol=0, atol=0
        )
        torch.testing.assert_close(
            route_weights[:6], selected_weights.to(device), rtol=2e-5, atol=2e-5
        )
        torch.testing.assert_close(
            route_record[64:96].view(torch.int32)[:6],
            (selected_experts.to(device) + 1) * linear1_slices,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            route_record[96:128].view(torch.int32)[:6],
            (selected_experts.to(device) + 1) * down_slices,
            rtol=0,
            atol=0,
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
    if args.dynamic_routing:
        down_weight_sum = torch.empty(
            (down_slices, 128), dtype=torch.float32, device=device
        )
        for output_tile in range(down_slices):
            row_nibbles = 1 + (
                torch.arange(128, device=device) // 16 + output_tile
            ) % 6
            fp4_rows = torch.tensor(
                FP4_VALUES, dtype=torch.float32, device=device
            )[row_nibbles]
            down_weight_sum[output_tile] = (
                2048.0 * fp4_rows * down_weight_scale_value
            )
    else:
        down_weight_sum = 1024.0 * (
            FP4_VALUES[args.down_weight_byte & 0xF]
            + FP4_VALUES[args.down_weight_byte >> 4]
        ) * down_weight_scale_value
    route_sum_by_output = torch.zeros(down_slices, dtype=torch.float32)
    for task in queued_down_tasks:
        expert, output_tile = divmod(task, down_slices)
        route_sum_by_output[output_tile] += route_scales[expert]
    if args.dynamic_routing:
        expected_final_mmajor = (
            down_weight_sum.reshape(down_slices, 128, 1)
            * dequantized_middle.to(device).reshape(1, 1, 8)
            * route_sum_by_output.to(device).reshape(down_slices, 1, 1)
        )
    else:
        expected_final_mmajor = (
            (dequantized_middle * down_weight_sum)
            .reshape(1, 1, 8)
            .to(device)
            * route_sum_by_output.to(device).reshape(down_slices, 1, 1)
        ).expand(down_slices, 128, 8)
    expected_final = (
        expected_final_mmajor.permute(2, 0, 1).reshape(8, 4096)
        if args.dynamic_routing
        else expected_final_mmajor
    )
    checked_outputs = route_sum_by_output != 0
    checked_outputs_device = checked_outputs.to(device)
    if args.dynamic_routing:
        torch.testing.assert_close(
            final_output.float(), expected_final, rtol=3e-2, atol=1e-1
        )
    elif bool(checked_outputs.any()):
        torch.testing.assert_close(
            final_output[checked_outputs_device].float(),
            expected_final[checked_outputs_device],
            rtol=3e-2,
            atol=1e-1,
        )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(args.graph_inner):
            enqueue()
    for _ in range(args.warmup):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    device_counter_times = []
    for _ in range(args.iterations):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop) * 1.0e3 / args.graph_inner)
        device_counter_times.append(device_counter_span_us(launcher.profile))

    hot_profile = launcher.profile.clone()

    cold_times = None
    cold_device_counter_times = None
    if args.cold_samples:
        cold_times, cold_device_counter_times = cold_graph_counter_timings_us(
            enqueue,
            profile=launcher.profile,
            stream=stream,
            warmup=args.warmup,
            samples=args.cold_samples,
            l2_scrub_mib=args.cold_l2_scrub_mib,
        )

    if args.dynamic_routing:
        torch.testing.assert_close(
            final_output.float(), expected_final, rtol=3e-2, atol=1e-1
        )
    elif bool(checked_outputs.any()):
        torch.testing.assert_close(
            final_output[checked_outputs_device].float(),
            expected_final[checked_outputs_device],
            rtol=3e-2,
            atol=1e-1,
        )
    profile = hot_profile
    linear1_profile = profile[:linear1_tasks]
    linear1_finish = relative_finish_us(linear1_profile, 2, 4)
    down_finish = relative_finish_us(profile, 2, 5)
    linear1_local = local_duration_us(linear1_profile, 2, 4)
    down_local = local_duration_us(profile, 4, 5)
    task_local = local_duration_us(profile, 2, 3)
    linear1_tail = local_tail_us(linear1_profile, 2, 4)
    down_tail = local_tail_us(profile, 4, 5)
    if args.dynamic_routing:
        error = (final_output.float() - expected_final).abs()
        relative_error = error / expected_final.abs().clamp_min(
            torch.finfo(torch.float32).tiny
        )
        max_abs_error = float(error.max())
        max_rel_error = float(relative_error.max())
    elif bool(checked_outputs.any()):
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
            f"dynamic_routing={str(args.dynamic_routing).lower()} "
            f"route_with_topk={str(args.route_with_topk).lower()} "
            f"output_layout={'n8_m4096' if args.dynamic_routing else 'm4096_n8'} "
            "timing=cold_data_one_ffn_graph "
            f"l2_scrub_mib={args.cold_l2_scrub_mib} "
            f"samples={args.cold_samples} "
            f"min_us={min(cold_times):.6f} "
            f"median_us={statistics.median(cold_times):.6f} "
            f"p90_us={percentile_us(cold_times, 0.90):.6f} "
            f"stddev_us={statistics.pstdev(cold_times):.6f} "
            f"max_us={max(cold_times):.6f} "
            "device_counter_scope=post_queue_init_to_compute_terminate "
            f"device_counter_min_us={min(cold_device_counter_times):.6f} "
            "device_counter_median_us="
            f"{statistics.median(cold_device_counter_times):.6f} "
            "device_counter_p90_us="
            f"{percentile_us(cold_device_counter_times, 0.90):.6f} "
            "device_counter_stddev_us="
            f"{statistics.pstdev(cold_device_counter_times):.6f} "
            f"device_counter_max_us={max(cold_device_counter_times):.6f} "
            "output_correct=true",
            flush=True,
        )
    print(
        "DSV4_MXFP4_MXFP8_FULL_FFN_RETAINED_RING_RESULT "
        f"workers={args.workers} linear1_tasks={linear1_tasks} "
        f"down_tasks={down_tasks} down_tasks_queued={len(queued_down_tasks)} "
        f"down_active_workers={down_active_workers} "
        f"dynamic_routing={str(args.dynamic_routing).lower()} "
        f"route_with_topk={str(args.route_with_topk).lower()} "
        f"output_layout={'n8_m4096' if args.dynamic_routing else 'm4096_n8'} "
        "cuda_kernel_launches=1 vdcores_launches=1 "
        "persistent=true "
        "resident_all_tma=true "
        "resident_load_operator=coupled_stream "
        f"coupled_stream_commands={len(coupled_commands)} "
        f"coupled_stream_local_chains={coupled_chain_sources} "
        "kernel=dae2 "
        "linear1_ldu_weight_ring=true "
        "down_ldu_weight_ring=true "
        "down_ring_stages=2 "
        "activation_scales_task_owned=false "
        "down_bf16_reduction=true "
        "weight_prefetch_distance=1 "
        "queue_init=generic "
        "stu_reduction=false "
        "ldu1_enabled=true "
        "python_output_zero=true "
        "blockwise_ready=true "
        "weight_scale_tma=true "
        "down_input=linear1 "
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
        "device_counter_scope=post_queue_init_to_compute_terminate "
        f"device_counter_min_us={min(device_counter_times):.6f} "
        "device_counter_median_us="
        f"{statistics.median(device_counter_times):.6f} "
        f"device_counter_p90_us={percentile(device_counter_times, 0.90):.6f} "
        "device_counter_stddev_us="
        f"{statistics.pstdev(device_counter_times):.6f} "
        f"device_counter_max_us={max(device_counter_times):.6f} "
        f"max_abs_error={max_abs_error:.8f} "
        f"max_rel_error={max_rel_error:.8f} "
        f"down_output_checked={str(bool(checked_outputs.any())).lower()} "
        "output_correct=true",
        flush=True,
    )


if __name__ == "__main__":
    main()
