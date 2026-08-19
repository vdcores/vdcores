#!/usr/bin/env python3
"""Allocator-compatible native-record MXFP4/MXFP8 full-FFN benchmark.

Linear-1 publishes sixteen 1536-byte MXFP8 records per expert. The stream-
ordered down projection consumes them directly and reduces all seven expert
outputs into FP32 with TMA reduce-add. Task-local scratch is disjoint from the
normal allocator layout; the focused down launch gives that scratch a standalone
arena. The down kernel runs two disjoint four-warp tasks inside one resident CTA
per SM. The timed graph contains no conversion, repack, or row replication.
"""

from __future__ import annotations

import argparse
import os
import statistics

import torch

from dae import runtime
from dae.instructions import TmaTensor
from dae.launcher import Launcher
from deepseek_v4_cold_timing import percentile_us


FP4_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def f32_bits(value: float) -> int:
    return int(torch.tensor(value, dtype=torch.float32).view(torch.int32)) & 0xFFFFFFFF


def uniform_linear1_reference(
    weight_byte: int,
    weight_scale_byte: int,
    activation_row_bytes: torch.Tensor,
    activation_scale_byte: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight_scale = 2.0 ** (weight_scale_byte - 127)
    activation_scale = 2.0 ** (activation_scale_byte - 127)
    weight_sum = 2048.0 * (
        FP4_VALUES[weight_byte & 0xF] + FP4_VALUES[weight_byte >> 4]
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


def kernel_span_us(launcher: Launcher) -> float:
    profile = launcher.profile[:, :2].cpu().numpy()
    return (int(profile[:, 1].max()) - int(profile[:, 0].min())) / 1.0e3


def direct_counter_spans_us(
    linear1_profile: torch.Tensor,
    down_profile: torch.Tensor,
) -> tuple[float, float, float, float]:
    """Return both kernels, their launch gap, and the combined grid envelope."""

    linear1 = linear1_profile[:, :2].cpu().numpy()
    down = down_profile[:, :2].cpu().numpy()
    linear1_start = int(linear1[:, 0].min())
    linear1_end = int(linear1[:, 1].max())
    down_start = int(down[:, 0].min())
    down_end = int(down[:, 1].max())
    return (
        (linear1_end - linear1_start) / 1.0e3,
        (down_start - linear1_end) / 1.0e3,
        (down_end - down_start) / 1.0e3,
        (down_end - linear1_start) / 1.0e3,
    )


def cold_graph_direct_counter_timings_us(
    run,
    *,
    linear1_profile: torch.Tensor,
    down_profile: torch.Tensor,
    stream: torch.cuda.Stream,
    warmup: int,
    samples: int,
    l2_scrub_mib: int,
) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    """Collect cold CUDA-event and both direct-kernel counter spans."""

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
    linear1_counter_times: list[float] = []
    inter_kernel_gap_times: list[float] = []
    down_counter_times: list[float] = []
    combined_counter_times: list[float] = []
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
        linear1_us, inter_kernel_gap_us, down_us, combined_us = (
            direct_counter_spans_us(
                linear1_profile, down_profile
            )
        )
        linear1_counter_times.append(linear1_us)
        inter_kernel_gap_times.append(inter_kernel_gap_us)
        down_counter_times.append(down_us)
        combined_counter_times.append(combined_us)
    return (
        event_times,
        linear1_counter_times,
        inter_kernel_gap_times,
        down_counter_times,
        combined_counter_times,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--graph-inner", type=int, default=20)
    parser.add_argument("--cold-samples", type=int, default=0)
    parser.add_argument("--cold-l2-scrub-mib", type=int, default=260)
    parser.add_argument("--weight-byte", type=lambda value: int(value, 0), default=0x66)
    parser.add_argument("--weight-scale", type=int, default=125)
    parser.add_argument("--down-weight-byte", type=lambda value: int(value, 0), default=0x22)
    parser.add_argument("--down-weight-scale", type=int, default=124)
    parser.add_argument("--activation-byte", type=lambda value: int(value, 0), default=0x60)
    parser.add_argument("--activation-scale", type=int, default=119)
    parser.add_argument(
        "--tma-l2-promotion", choices=("none", "64", "128", "256"),
        default="256",
    )
    parser.add_argument("--vllm-us", type=float, default=30.480000)
    args = parser.parse_args()
    os.environ["DAE_TMA_L2_PROMOTION"] = args.tma_l2_promotion
    if args.graph_inner <= 0:
        parser.error("--graph-inner must be positive")
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

    workers = 124
    linear1_launcher = Launcher(linear1_tasks, device=device)
    down_launcher = Launcher(workers, device=device)
    zero_ready = [down_launcher.new_bar(1) for _ in range(down_slices)]
    # Both stream-ordered kernels share the reduction-sense barrier arena.
    linear1_launcher.bars = down_launcher.bars

    linear1_weight_shape = (
        k_tiles, linear1_tasks, k128_per_tile, 128, 64
    )
    weight_plane_bytes = (
        linear1_tasks * k_tiles * k128_per_tile * 128 * 64
    )
    weight_arena = torch.empty(
        (3, weight_plane_bytes), dtype=torch.uint8, device=device
    )
    gate_weight = weight_arena[0].view(linear1_weight_shape)
    up_weight = weight_arena[1].view(linear1_weight_shape)
    gate_weight.fill_(args.weight_byte)
    up_weight.fill_(args.weight_byte)
    linear1_scale_shape = (k_tiles, linear1_tasks, k128_per_tile * 512)
    weight_scale_plane_bytes = (
        linear1_tasks * k_tiles * k128_per_tile * 512
    )
    weight_scale_arena = torch.empty(
        (3, weight_scale_plane_bytes), dtype=torch.uint8, device=device
    )
    gate_scale = weight_scale_arena[0].view(linear1_scale_shape)
    up_scale = weight_scale_arena[1].view(linear1_scale_shape)
    gate_scale.fill_(args.weight_scale)
    up_scale.fill_(args.weight_scale)
    activation_row_bytes = torch.arange(8, dtype=torch.uint8) + args.activation_byte
    activation_data = (
        activation_row_bytes.to(device)
        .reshape(1, 1, 8, 1)
        .expand(k_tiles, k128_per_tile, 8, 128)
        .contiguous()
        .reshape(k_tiles, 4096)
    )
    activation_scale = torch.full(
        (k_tiles, 2048), args.activation_scale, dtype=torch.uint8, device=device
    )
    activation_record = torch.empty(
        (experts, linear1_slices, 1536), dtype=torch.uint8, device=device
    )
    activation_record_flat = activation_record.view(linear1_tasks, 1536)
    activation_output_data = activation_record_flat[:, :1024]
    activation_output_scale = activation_record_flat[:, 1024:]

    gate_tma = TmaTensor(linear1_launcher, gate_weight).mxfp4_kmajor_load(tile_k)
    up_tma = TmaTensor(linear1_launcher, up_weight).mxfp4_kmajor_load(tile_k)
    weight_scale_tile_stride = linear1_tasks * k128_per_tile * 512
    linear1_records = torch.zeros(
        (linear1_tasks, 16), dtype=torch.int64, device="cpu"
    )
    for task in range(linear1_tasks):
        expert, output_slice = divmod(task, linear1_slices)
        linear1_records[task, 0] = activation_data.data_ptr()
        linear1_records[task, 2] = gate_scale[0, task].data_ptr()
        linear1_records[task, 3] = activation_scale[0].data_ptr()
        linear1_records[task, 4] = up_scale[0, task].data_ptr()
        linear1_records[task, 5] = (
            gate_tma.arg | (up_tma.arg << 16) | (task << 32)
        )
        linear1_records[task, 6] = activation_record_flat[task].data_ptr()
        linear1_records[task, 7] = (
            weight_scale_tile_stride | (1 << 32)
        )
        linear1_records[task, 8] = 0xFFFFFFFF
    linear1_metadata = linear1_records.view(torch.uint8).to(device)

    down_weight = weight_arena[2].view(down_tasks, 8, 2, 128, 64)
    down_weight.fill_(args.down_weight_byte)
    down_weight_scale = weight_scale_arena[2].view(down_tasks, 8, 1024)
    down_weight_scale.fill_(args.down_weight_scale)
    final_output = torch.empty(
        (down_slices, 128, 8), dtype=torch.bfloat16, device=device
    )
    down_tma = TmaTensor(down_launcher, down_weight).mxfp4_load(256)
    output_tma = TmaTensor(
        down_launcher, final_output.view(down_slices * 128, 8)
    ).rowmajor_2d("reduce", 128, 8)
    route_scales = [1.0, *([1.0 / 6.0] * 6)]
    down_records = torch.zeros(
        (down_tasks, 16), dtype=torch.int64, device="cpu"
    )
    for task in range(down_tasks):
        expert, m_tile = divmod(task, down_slices)
        down_records[task, 0] = down_weight_scale[task, 0].data_ptr()
        down_records[task, 1] = activation_record[expert, 0].data_ptr()
        down_records[task, 3] = (
            down_tma.arg | (output_tma.arg << 16) | (task << 32)
        )
        down_records[task, 4] = 0xFFFFFFFF | (zero_ready[m_tile] << 32)
        down_records[task, 5] = f32_bits(route_scales[expert])
        down_records[task, 6] = final_output[m_tile].data_ptr()
        # Offset 68 selects zero-initialized all-expert TMA reduce-add.
        down_records[task, 8] = 1 << 32
    # Bank zero holds tasks 0..123. Its first 24 workers stay single-group;
    # bank one maps tasks 124..223 to workers 24..123.
    down_launch_records = torch.zeros(
        (2 * workers, 16), dtype=torch.int64, device="cpu"
    )
    down_launch_records[:workers].copy_(down_records[:workers])
    dual_group_workers = down_tasks - workers
    single_group_workers = workers - dual_group_workers
    down_launch_records[
        workers + single_group_workers :
        workers + single_group_workers + dual_group_workers
    ].copy_(down_records[workers:])
    down_metadata = down_launch_records.view(torch.uint8).to(device)
    _, _, linear1_tmas = linear1_launcher.prepare_launch()
    _, _, down_tmas = down_launcher.prepare_launch()
    linear1_metadata_bytes = linear1_metadata.reshape(-1, 1)
    down_metadata_bytes = down_metadata.reshape(-1, 1)
    linear1_profile_bytes = linear1_launcher.profile.view(torch.uint8).view(-1, 8)
    down_profile_bytes = down_launcher.profile.view(torch.uint8).view(-1, 8)
    root_stream = torch.cuda.Stream()
    down_launcher.bars.zero_()

    def launch_linear1() -> None:
        stream = torch.cuda.current_stream().cuda_stream
        runtime.launch_dae_ffn_linear1_direct(
            linear1_tasks,
            linear1_launcher.smem_size,
            linear1_metadata_bytes,
            linear1_tmas,
            down_launcher.bars,
            zero_ready[0],
            len(zero_ready),
            linear1_profile_bytes,
            stream,
        )

    def launch_down() -> None:
        stream = torch.cuda.current_stream().cuda_stream
        runtime.launch_dae_ffn_down_direct(
            workers,
            160 * 1024,
            down_metadata_bytes,
            down_tmas,
            down_launcher.bars,
            down_profile_bytes,
            stream,
        )

    def enqueue_full_ffn() -> None:
        with torch.cuda.stream(root_stream):
            launch_linear1()
            launch_down()

    root_stream.wait_stream(torch.cuda.current_stream())
    enqueue_full_ffn()
    torch.cuda.current_stream().wait_stream(root_stream)
    torch.cuda.synchronize()

    expected_data, expected_scale, dequantized_middle = uniform_linear1_reference(
        args.weight_byte,
        args.weight_scale,
        activation_row_bytes,
        args.activation_scale,
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
    expected_unscaled = dequantized_middle * down_weight_sum
    expected_final = (
        (expected_unscaled * sum(route_scales))
        .reshape(1, 1, 8)
        .expand(down_slices, 128, 8)
        .to(device)
    )
    rtol = 3e-2
    atol = 1e-1
    torch.testing.assert_close(
        final_output.float(), expected_final, rtol=rtol, atol=atol
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=root_stream):
        for _ in range(args.graph_inner):
            enqueue_full_ffn()

    for _ in range(args.warmup):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    linear1_counter_times = []
    inter_kernel_gap_times = []
    down_counter_times = []
    combined_counter_times = []
    for _ in range(args.iterations):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop) * 1.0e3 / args.graph_inner)
        linear1_us, inter_kernel_gap_us, down_us, combined_us = (
            direct_counter_spans_us(
                linear1_launcher.profile, down_launcher.profile
            )
        )
        linear1_counter_times.append(linear1_us)
        inter_kernel_gap_times.append(inter_kernel_gap_us)
        down_counter_times.append(down_us)
        combined_counter_times.append(combined_us)

    cold_times = None
    cold_linear1_counter_times = None
    cold_inter_kernel_gap_times = None
    cold_down_counter_times = None
    cold_combined_counter_times = None
    if args.cold_samples:
        (
            cold_times,
            cold_linear1_counter_times,
            cold_inter_kernel_gap_times,
            cold_down_counter_times,
            cold_combined_counter_times,
        ) = cold_graph_direct_counter_timings_us(
            enqueue_full_ffn,
            linear1_profile=linear1_launcher.profile,
            down_profile=down_launcher.profile,
            stream=root_stream,
            warmup=args.warmup,
            samples=args.cold_samples,
            l2_scrub_mib=args.cold_l2_scrub_mib,
        )

    torch.testing.assert_close(
        final_output.float(), expected_final, rtol=rtol, atol=atol
    )
    median_us = statistics.median(times)
    speedup = args.vllm_us / median_us
    improvement = (args.vllm_us - median_us) / args.vllm_us * 100.0
    output_fp32 = final_output.float()
    max_abs_error = float((output_fp32 - expected_final).abs().max())
    max_rel_error = float(
        ((output_fp32 - expected_final).abs() /
         expected_final.abs().clamp_min(torch.finfo(torch.float32).tiny)).max()
    )
    print(
        "DSV4_MXFP4_MXFP8_FULL_FFN_RESULT "
        f"experts={experts} shared=1 routed=6 rows=8 "
        "native_handoff=true conversion=false repack=false replication=false "
        "reduction=tma_bf16 "
        "kernels=2 focused_entrypoints=true "
        "allocator_compatible=true one_cta_per_sm=true "
        "linear1_tmem_epilogue=late_register "
        f"workers={workers} allocator_slots={runtime.config.num_slots} "
        f"slot_bytes={runtime.config.slot_size} "
        f"dual_group_workers={dual_group_workers} "
        f"single_group_workers={single_group_workers} "
        f"shared_publishers_single_group={min(single_group_workers, down_slices)} "
        "down_compute_groups=2 "
        f"tma_l2_promotion={args.tma_l2_promotion} graph=true "
        f"graph_inner={args.graph_inner} "
        f"linear1_kernel_us={kernel_span_us(linear1_launcher):.6f} "
        f"down_kernel_us={kernel_span_us(down_launcher):.6f} "
        "device_counter_scope=linear1_kernel_entry_to_down_kernel_exit "
        f"linear1_device_counter_min_us={min(linear1_counter_times):.6f} "
        "linear1_device_counter_median_us="
        f"{statistics.median(linear1_counter_times):.6f} "
        "linear1_device_counter_p90_us="
        f"{percentile_us(linear1_counter_times, 0.90):.6f} "
        "linear1_device_counter_stddev_us="
        f"{statistics.pstdev(linear1_counter_times):.6f} "
        f"linear1_device_counter_max_us={max(linear1_counter_times):.6f} "
        f"inter_kernel_gap_min_us={min(inter_kernel_gap_times):.6f} "
        f"inter_kernel_gap_median_us={statistics.median(inter_kernel_gap_times):.6f} "
        "inter_kernel_gap_p90_us="
        f"{percentile_us(inter_kernel_gap_times, 0.90):.6f} "
        f"inter_kernel_gap_stddev_us={statistics.pstdev(inter_kernel_gap_times):.6f} "
        f"inter_kernel_gap_max_us={max(inter_kernel_gap_times):.6f} "
        f"down_device_counter_min_us={min(down_counter_times):.6f} "
        "down_device_counter_median_us="
        f"{statistics.median(down_counter_times):.6f} "
        f"down_device_counter_p90_us={percentile_us(down_counter_times, 0.90):.6f} "
        f"down_device_counter_stddev_us={statistics.pstdev(down_counter_times):.6f} "
        f"down_device_counter_max_us={max(down_counter_times):.6f} "
        f"device_counter_min_us={min(combined_counter_times):.6f} "
        f"device_counter_median_us={statistics.median(combined_counter_times):.6f} "
        f"device_counter_p90_us={percentile_us(combined_counter_times, 0.90):.6f} "
        f"device_counter_stddev_us={statistics.pstdev(combined_counter_times):.6f} "
        f"device_counter_max_us={max(combined_counter_times):.6f} "
        f"end_to_end_min_us={min(times):.6f} "
        f"end_to_end_median_us={median_us:.6f} "
        f"end_to_end_max_us={max(times):.6f} "
        f"vllm_us={args.vllm_us:.6f} speedup={speedup:.4f}x "
        f"improvement_pct={improvement:.3f} target_met={str(improvement >= 10.0).lower()} "
        f"max_abs_error={max_abs_error:.8f} max_rel_error={max_rel_error:.8f} "
        "output_correct=true",
        flush=True,
    )
    if cold_times is not None:
        print(
            "DSV4_MXFP4_MXFP8_FULL_FFN_COLD_RESULT "
            f"experts={experts} shared=1 routed=6 rows=8 "
            "implementation=task_direct kernels=2 "
            "timing=cold_data_one_ffn_graph "
            f"l2_scrub_mib={args.cold_l2_scrub_mib} "
            f"samples={args.cold_samples} "
            f"min_us={min(cold_times):.6f} "
            f"median_us={statistics.median(cold_times):.6f} "
            f"p90_us={percentile_us(cold_times, 0.90):.6f} "
            f"stddev_us={statistics.pstdev(cold_times):.6f} "
            f"max_us={max(cold_times):.6f} "
            "device_counter_scope=linear1_kernel_entry_to_down_kernel_exit "
            "linear1_device_counter_min_us="
            f"{min(cold_linear1_counter_times):.6f} "
            "linear1_device_counter_median_us="
            f"{statistics.median(cold_linear1_counter_times):.6f} "
            "linear1_device_counter_p90_us="
            f"{percentile_us(cold_linear1_counter_times, 0.90):.6f} "
            "linear1_device_counter_stddev_us="
            f"{statistics.pstdev(cold_linear1_counter_times):.6f} "
            "linear1_device_counter_max_us="
            f"{max(cold_linear1_counter_times):.6f} "
            f"inter_kernel_gap_min_us={min(cold_inter_kernel_gap_times):.6f} "
            "inter_kernel_gap_median_us="
            f"{statistics.median(cold_inter_kernel_gap_times):.6f} "
            "inter_kernel_gap_p90_us="
            f"{percentile_us(cold_inter_kernel_gap_times, 0.90):.6f} "
            "inter_kernel_gap_stddev_us="
            f"{statistics.pstdev(cold_inter_kernel_gap_times):.6f} "
            f"inter_kernel_gap_max_us={max(cold_inter_kernel_gap_times):.6f} "
            f"down_device_counter_min_us={min(cold_down_counter_times):.6f} "
            "down_device_counter_median_us="
            f"{statistics.median(cold_down_counter_times):.6f} "
            "down_device_counter_p90_us="
            f"{percentile_us(cold_down_counter_times, 0.90):.6f} "
            "down_device_counter_stddev_us="
            f"{statistics.pstdev(cold_down_counter_times):.6f} "
            f"down_device_counter_max_us={max(cold_down_counter_times):.6f} "
            f"device_counter_min_us={min(cold_combined_counter_times):.6f} "
            "device_counter_median_us="
            f"{statistics.median(cold_combined_counter_times):.6f} "
            "device_counter_p90_us="
            f"{percentile_us(cold_combined_counter_times, 0.90):.6f} "
            "device_counter_stddev_us="
            f"{statistics.pstdev(cold_combined_counter_times):.6f} "
            f"device_counter_max_us={max(cold_combined_counter_times):.6f} "
            "output_correct=true",
            flush=True,
        )


if __name__ == "__main__":
    main()
