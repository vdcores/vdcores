#!/usr/bin/env python3
"""Fused MXFP4 gate/up -> SiLU -> native MXFP8 Linear-1 benchmark.

The timed region starts from final native HBM operands. One task owns one
expert's M128 intermediate slice and writes one contiguous N8/K128 MXFP8 data
and UE8M0-scale record. By default the task-major HBM layout is shared-first:
16 slices for the shared expert followed by 16 slices for each of six routed
experts. Every UMMA consumes eight pre-laid-out activation rows and preserves
all eight outputs; the kernel performs no row replication. All 112 unsplit
tasks run as one persistent-kernel wave.
"""

from __future__ import annotations

import argparse
import statistics

import torch

from dae import runtime
from dae.instructions import ProfileEvent, TmaTensor
from dae.launcher import Launcher
from dae.schedule import SchedMxfp4Mxfp8GateUpSiluFixedRing


FP4_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def expected_uniform_row_output(
    weight_byte: int,
    weight_scale_byte: int,
    activation_row_bytes: torch.Tensor,
    activation_scale_byte: int,
    bf16_epilogue: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Python-only reference for eight distinct constant activation rows."""
    weight_scale = 2.0 ** (weight_scale_byte - 127)
    activation_scale = 2.0 ** (activation_scale_byte - 127)
    weight_sum = 2048.0 * (
        FP4_VALUES[weight_byte & 0xF] + FP4_VALUES[weight_byte >> 4]
    ) * weight_scale
    activation = activation_row_bytes.view(torch.float8_e4m3fn).float()
    gate = activation * activation_scale * weight_sum
    if bf16_epilogue:
        rounded_gate = gate.to(torch.bfloat16)
        gate_silu = (
            rounded_gate.float()
            / (1.0 + torch.exp(-rounded_gate.float()))
        ).to(torch.bfloat16)
        middle = (
            gate_silu.float() * rounded_gate.float()
        ).to(torch.bfloat16).float()
    else:
        middle = gate / (1.0 + torch.exp(-gate)) * gate
    requested = (middle.abs() / 448.0).clamp_min(2.0**-127)
    exponents = torch.ceil(torch.log2(requested)).clamp(-127, 127)
    scales = torch.exp2(exponents)
    quantized = (middle / scales).clamp(-448.0, 448.0)
    return (
        quantized.to(torch.float8_e4m3fn).view(torch.uint8),
        (exponents.to(torch.int16) + 127).to(torch.uint8),
        middle,
    )


def report_mxfp_timeline(launcher: Launcher) -> None:
    profile = launcher.profile.cpu().numpy()
    task_entry = profile[:, 4]
    active = task_entry != 0
    if not active.any():
        return

    def relative_us(event: int) -> float:
        values = (profile[active, event] - task_entry[active]) / 1.0e3
        return float(statistics.median(values))

    def delta_us(start: int, stop: int) -> float:
        values = (profile[active, stop] - profile[active, start]) / 1.0e3
        return float(statistics.median(values))

    for tile in range(8):
        print(
            "DSV4_MXFP4_MXFP8_GATE_UP_SILU_TILE "
            f"tile={tile} "
            f"activation_ready_us={relative_us(5 + tile):.3f} "
            f"scale_weight_ready_us={relative_us(13 + tile):.3f} "
            f"gate_issue_us={relative_us(29 + tile):.3f} "
            f"gate_complete_us={relative_us(37 + tile):.3f} "
            f"up_ready_us={relative_us(45 + tile):.3f} "
            f"up_issue_us={relative_us(53 + tile):.3f} "
            f"up_complete_us={relative_us(61 + tile):.3f}",
            flush=True,
        )
    print(
        "DSV4_MXFP4_MXFP8_GATE_UP_SILU_EPILOGUE "
        f"gate_silu_start_us={relative_us(77):.3f} "
        f"gate_silu_helper_done_us={relative_us(78):.3f} "
        f"all_umma_sync_us={relative_us(85):.3f} "
        f"up_tmem_mul_done_us={relative_us(86):.3f} "
        f"quant_scale_done_us={relative_us(87):.3f} "
        f"pack_done_us={relative_us(88):.3f} "
        f"task_end_us={relative_us(94):.3f} "
        f"gate_silu_overlap_us={delta_us(77, 78):.3f} "
        f"post_umma_epilogue_us={delta_us(85, 94):.3f}",
        flush=True,
    )


def report_track_profile(launcher: Launcher) -> None:
    profile = launcher.profile.cpu().numpy()
    if not all(int(value) == 0x4454524B50524631 for value in profile[:, 127]):
        return
    base = runtime.config.track_profile_event_base
    span = max(int(value) for value in profile[:, 1]) - min(
        int(value) for value in profile[:, 0]
    )
    envelope = span * profile.shape[0]

    def total(offset: int) -> int:
        return sum(int(value) for value in profile[:, base + offset])

    def percent(offset: int) -> float:
        return 100.0 * total(offset) / envelope if envelope > 0 else 0.0

    print(
        "DSV4_MXFP4_MXFP8_GATE_UP_SILU_COUNTERS "
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


def build_metadata(
    gate_scale: torch.Tensor,
    activation_scale: torch.Tensor,
    up_scale: torch.Tensor,
    output_record: torch.Tensor | None = None,
    *,
    gate_tma_index: int | None = None,
    up_tma_index: int | None = None,
) -> torch.Tensor:
    records = torch.zeros(
        (gate_scale.shape[0], 16), dtype=torch.int64, device="cpu"
    )
    for tile in range(gate_scale.shape[0]):
        records[tile, 2] = gate_scale[tile, 0].data_ptr()
        records[tile, 3] = activation_scale[0].data_ptr()
        records[tile, 4] = up_scale[tile, 0].data_ptr()
        if gate_tma_index is not None and up_tma_index is not None:
            records[tile, 5] = (
                int(gate_tma_index)
                | (int(up_tma_index) << 16)
                | (tile << 32)
            )
        if output_record is not None:
            records[tile, 6] = output_record[tile].data_ptr()
    return records.view(torch.uint8).to(gate_scale.device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        type=int,
        default=None,
        help="override the shared+routed expert task count",
    )
    parser.add_argument("--shared-experts", type=int, default=1)
    parser.add_argument("--routed-experts", type=int, default=6)
    parser.add_argument("--slices-per-expert", type=int, default=16)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--tile-k", type=int, choices=(128, 512), default=512)
    parser.add_argument("--diagnostic-output", action="store_true")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument(
        "--weight-byte", type=lambda value: int(value, 0), default=0x66
    )
    parser.add_argument("--weight-scale", type=int, default=125)
    parser.add_argument(
        "--activation-byte", type=lambda value: int(value, 0), default=0x60
    )
    parser.add_argument("--activation-scale", type=int, default=119)
    args = parser.parse_args()
    if args.shared_experts < 0 or args.routed_experts < 0:
        parser.error("expert counts must be non-negative")
    if args.shared_experts + args.routed_experts <= 0:
        parser.error("at least one expert is required")
    if args.slices_per_expert <= 0:
        parser.error("--slices-per-expert must be positive")
    if not 0 <= args.weight_byte <= 0xFF:
        parser.error("--weight-byte must fit uint8")
    if not 0 <= args.weight_scale <= 0xFF:
        parser.error("--weight-scale must fit uint8")
    if not 0 <= args.activation_byte <= 0xF8:
        parser.error("--activation-byte must leave room for eight uint8 rows")
    if not 0 <= args.activation_scale <= 0xFF:
        parser.error("--activation-scale must fit uint8")
    mixed_tasks = (
        (args.shared_experts + args.routed_experts)
        * args.slices_per_expert
    )
    if args.tasks is None:
        args.tasks = mixed_tasks
    mixed_layout = args.tasks == mixed_tasks
    if args.tasks <= 0:
        parser.error("--tasks must be positive")
    workers = args.tasks if args.workers is None else args.workers
    if not 1 <= workers <= args.tasks:
        parser.error("--workers must be in [1,tasks]")

    print(
        "DSV4_MXFP4_MXFP8_GATE_UP_SILU_LAYOUT "
        f"shared_experts={args.shared_experts} "
        f"routed_experts={args.routed_experts} "
        f"slices_per_expert={args.slices_per_expert} "
        f"tasks={args.tasks} "
        f"shared_first={str(mixed_layout).lower()}",
        flush=True,
    )

    schedule_type = SchedMxfp4Mxfp8GateUpSiluFixedRing
    k_tiles = 4096 // args.tile_k
    weight_k128_tiles = args.tile_k // 128
    weight_scale_bytes = weight_k128_tiles * 512
    activation_data_bytes = weight_k128_tiles * 1024
    activation_scale_bytes = weight_k128_tiles * 512
    device = torch.device("cuda")
    weight_shape = (
        args.tasks,
        k_tiles,
        weight_k128_tiles,
        schedule_type.TILE_M,
        schedule_type.WEIGHT_PACKED_K128_BYTES,
    )
    gate_weight = torch.full(
        weight_shape, args.weight_byte, dtype=torch.uint8, device=device
    )
    up_weight = torch.full_like(gate_weight, args.weight_byte)
    scale_shape = (
        args.tasks,
        k_tiles,
        weight_scale_bytes,
    )
    gate_scale = torch.full(
        scale_shape, args.weight_scale, dtype=torch.uint8, device=device
    )
    up_scale = torch.full_like(gate_scale, args.weight_scale)
    activation_row_bytes = (
        torch.arange(8, dtype=torch.uint8) + args.activation_byte
    )
    # HBM already contains eight physical activation rows for every K128
    # member. The rows are intentionally distinct; their within-row values are
    # constant, so the native SW128 chunk permutation is observable at output
    # without requiring any timed conversion or replication.
    activation_data = (
        activation_row_bytes.to(device)
        .reshape(1, 1, 8, 1)
        .expand(k_tiles, weight_k128_tiles, 8, 128)
        .contiguous()
        .reshape(k_tiles, activation_data_bytes)
    )
    activation_scale = torch.full(
        (k_tiles, activation_scale_bytes),
        args.activation_scale,
        dtype=torch.uint8,
        device=device,
    )
    output_record = torch.full(
        (
            args.tasks,
            schedule_type.OUTPUT_DATA_BYTES
            + schedule_type.OUTPUT_SCALE_BYTES,
        ),
        0xFF,
        dtype=torch.uint8,
        device=device,
    )
    output_data = output_record[:, : schedule_type.OUTPUT_DATA_BYTES]
    output_scale = output_record[:, schedule_type.OUTPUT_DATA_BYTES :]
    launcher = Launcher(workers, device=device)
    gate_tma = TmaTensor(launcher, gate_weight).mxfp4_load(args.tile_k)
    up_tma = TmaTensor(launcher, up_weight).mxfp4_load(args.tile_k)
    metadata = build_metadata(
        gate_scale,
        activation_scale,
        up_scale,
        output_record,
        gate_tma_index=gate_tma.arg,
        up_tma_index=up_tma.arg,
    )
    schedule = schedule_type(
        gate_weight,
        gate_scale,
        up_weight,
        up_scale,
        activation_data,
        activation_scale,
        output_data,
        output_scale,
        gate_tma,
        up_tma,
        metadata,
        tile_k=args.tile_k,
    ).place(workers)
    launcher.s(ProfileEvent(2), schedule, ProfileEvent(3))
    print(
        "DSV4_MXFP4_MXFP8_FIXED_RING_INSTRUCTIONS "
        f"compute={len(launcher.builder[0].cinsts)} "
        f"memory={len(launcher.builder[0].minsts)} "
        f"capacity={launcher.max_insts}",
        flush=True,
    )

    launcher.launch()
    torch.cuda.synchronize()
    output_rows = runtime.config.mxfp_gate_up_fixed_output_rows
    bf16_epilogue = bool(
        runtime.config.mxfp_gate_up_fixed_bf16_epilogue
    )
    if args.diagnostic_output:
        diagnostic_rows = output_data[0].reshape(8, -1)
        print(
            "DSV4_MXFP4_MXFP8_DIAGNOSTIC "
            f"data_unique={output_data.unique().cpu().tolist()} "
            f"data_row_unique="
            f"{[row.unique().cpu().tolist() for row in diagnostic_rows]} "
            f"scale_unique={output_scale.unique().cpu().tolist()}",
            flush=True,
        )
    (
        expected_row_bytes,
        expected_scale_codes,
        expected_middle,
    ) = expected_uniform_row_output(
        args.weight_byte,
        args.weight_scale,
        activation_row_bytes,
        args.activation_scale,
        bf16_epilogue,
    )
    (
        fp32_row_bytes,
        fp32_scale_codes,
        fp32_middle,
    ) = expected_uniform_row_output(
        args.weight_byte,
        args.weight_scale,
        activation_row_bytes,
        args.activation_scale,
        False,
    )
    epilogue_abs_error = float(
        (expected_middle - fp32_middle).abs().max().item()
    )
    epilogue_rel_error = float(
        (
            (expected_middle - fp32_middle).abs()
            / fp32_middle.abs().clamp_min(torch.finfo(torch.float32).tiny)
        ).max().item()
    )
    serialized_matches_fp32 = bool(
        torch.equal(expected_row_bytes, fp32_row_bytes)
        and torch.equal(expected_scale_codes, fp32_scale_codes)
    )
    expected_active_data = (
        expected_row_bytes[:output_rows]
        .reshape(-1, 1)
        .expand(output_rows, schedule_type.TILE_M)
        .reshape(1, -1)
        .to(device)
        .expand(args.tasks, -1)
    )
    torch.testing.assert_close(
        output_data[:, : output_rows * schedule_type.TILE_M],
        expected_active_data,
        rtol=0,
        atol=0,
    )
    active_scale_indices = (
        torch.arange(output_rows, device=device).reshape(-1, 1) * 16
        + torch.arange(4, device=device).reshape(1, -1)
    ).reshape(-1)
    expected_active_scales = (
        expected_scale_codes[:output_rows]
        .repeat_interleave(4)
        .reshape(1, -1)
        .to(device)
        .expand(args.tasks, -1)
    )
    torch.testing.assert_close(
        output_scale[:, active_scale_indices],
        expected_active_scales,
        rtol=0,
        atol=0,
    )

    for _ in range(args.warmup):
        launcher.launch()
    torch.cuda.synchronize()
    task_times = []
    kernel_times = []
    for _ in range(args.iterations):
        launcher.launch()
        profile = launcher.profile[:, :4].cpu().numpy()
        task_times.append(
            (profile[:, 3].max() - profile[:, 2].min()) / 1.0e3
        )
        kernel_times.append(
            (profile[:, 1].max() - profile[:, 0].min()) / 1.0e3
        )

    print(
        "DSV4_MXFP4_MXFP8_GATE_UP_SILU_RESULT "
        f"tasks={args.tasks} workers={workers} "
        f"mixed_layout={str(mixed_layout).lower()} "
        f"shared_experts={args.shared_experts} "
        f"routed_experts={args.routed_experts} "
        f"slices_per_expert={args.slices_per_expert} "
        "fixed_ring=true "
        f"tile_k={args.tile_k} "
        f"activation_tiles_per_load={k_tiles} "
        f"activation_rows={activation_row_bytes.tolist()} "
        f"output_row_count={output_rows} "
        f"bf16_epilogue={str(bf16_epilogue).lower()} "
        f"serialized_matches_fp32={str(serialized_matches_fp32).lower()} "
        f"epilogue_max_abs_error={epilogue_abs_error:.6f} "
        f"epilogue_max_rel_error={epilogue_rel_error:.8f} "
        f"output_rows={expected_row_bytes[:output_rows].tolist()} "
        f"task_min_us={min(task_times):.6f} "
        f"task_median_us={statistics.median(task_times):.6f} "
        f"task_max_us={max(task_times):.6f} "
        f"kernel_median_us={statistics.median(kernel_times):.6f} "
        "output_exact=true",
        flush=True,
    )
    report_mxfp_timeline(launcher)
    report_track_profile(launcher)


if __name__ == "__main__":
    main()
