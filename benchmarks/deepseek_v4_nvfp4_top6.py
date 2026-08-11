#!/usr/bin/env python3
"""One-launch native NVFP4 top-6 expert-flow benchmark."""

from __future__ import annotations

import argparse
import statistics

import torch

from dae.deepseek_v4 import (
    DeepSeekV4FlashConfig,
    bounded_swiglu,
    route_top6_reference,
)
from dae.deepseek_v4_quant import dequantize_nvfp4, quantize_nvfp4
from dae.instructions import ProfileEvent, TmaLoad1D, TmaTensor
from dae.launcher import Launcher
from dae.routing import RoutedAddressTable
from dae.runtime import config as runtime_config
from dae.schedule import (
    Schedule,
    SchedDsv4ExpertReduce,
    SchedDsv4Nvfp4QuantUmmaB,
    SchedDsv4RouteTop6,
    SchedDsv4SwiGluShard128,
    SchedNvfp4UmmaPrepack,
    SchedRoutedDsv4Nvfp4QuantUmmaB,
    SchedRoutedNvfp4GemvUmmaStream,
    SubgridSchedule,
)
from dae.sequential import SequentialProgram, SequentialStage
from dae.tma_utils import Major


def _row_pointer(tensor: torch.Tensor, row: int) -> int:
    return tensor.data_ptr() + row * tensor.stride(0) * tensor.element_size()


class _BarrierProfileSchedule(Schedule):
    """Record a diagnostic timestamp after one memory dependency completes."""

    def __init__(self, event_id: int, operand: torch.Tensor):
        super().__init__()
        self.event_id = event_id
        self.operand = operand

    def _on_place(self):
        if self.num_sms <= 0:
            raise ValueError("barrier profile marker requires at least one SM")

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        return [
            ProfileEvent(self.event_id, wait_for_memory=True),
            TmaLoad1D(self.operand, bytes=16),
        ]


def _prepack_weight(
    source: torch.Tensor,
    output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize one setup-only weight and emit the combined native layout."""
    rows, k = source.shape
    m_tiles = rows // 128
    k_tiles = k // 256
    weight, scale, global_scale = quantize_nvfp4(source)
    scale_tiles = (
        scale.view(m_tiles, 128, k_tiles, 16)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    launcher = Launcher(m_tiles, device=source.device)
    data_tma = TmaTensor(launcher, weight).wgmma_load(128, 128, Major.K)
    launcher.s(
        SchedNvfp4UmmaPrepack(
            SchedNvfp4UmmaPrepack.WEIGHT,
            weight,
            scale_tiles,
            output,
            data_tma,
        ).place(m_tiles)
    )
    launcher.launch()
    return weight, scale, global_scale


def _native_activation_oracle(
    source: torch.Tensor,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    """Build a setup oracle for the direct native activation quantizer."""
    k_tiles = source.numel() // 256
    packed, scale, _ = quantize_nvfp4(source, global_scale)
    rows = packed.reshape(1, -1).expand(8, -1).contiguous()
    scale_tiles = scale.view(k_tiles, 16)
    output = torch.empty(
        (k_tiles, SchedNvfp4UmmaPrepack.ACTIVATION_TILE_BYTES),
        dtype=torch.uint8,
        device=source.device,
    )
    launcher = Launcher(1, device=source.device)
    data_tma = TmaTensor(launcher, rows).wgmma_load(8, 128, Major.K)
    launcher.s(
        SchedNvfp4UmmaPrepack(
            SchedNvfp4UmmaPrepack.ACTIVATION,
            rows,
            scale_tiles,
            output,
            data_tma,
        ).place(1)
    )
    launcher.launch()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--trace-setup", action="store_true")
    parser.add_argument(
        "--profile-phases",
        action="store_true",
        help="insert dependency-fed internal timestamps on the otherwise idle SM",
    )
    parser.add_argument(
        "--hash-routing",
        action="store_true",
        help="use direct hash expert ids while retaining transformed route weights",
    )
    args = parser.parse_args()

    cfg = DeepSeekV4FlashConfig()
    device = torch.device("cuda")
    device_sms = torch.cuda.get_device_properties(device).multi_processor_count
    branch_sms = device_sms // cfg.experts_per_token
    if branch_sms < 25:
        raise ValueError("native top-6 benchmark requires at least 150 SMs")
    branch_sms = 25
    queued_sms = branch_sms * cfg.experts_per_token
    reducer_sm = queued_sms
    profile_sm = reducer_sm + 1
    if profile_sm >= device_sms:
        raise ValueError(
            "native top-6 benchmark needs reducer and profile SMs after branches"
        )

    generator = torch.Generator(device=device).manual_seed(args.seed)

    def setup_stage(name: str) -> None:
        if args.trace_setup:
            torch.cuda.synchronize()
            print(f"DSV4_NVFP4_TOP6_SETUP {name}", flush=True)

    hidden = torch.randn(
        (cfg.hidden_size,),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ) * 0.1
    hidden_packed, hidden_scale, hidden_global_scale = quantize_nvfp4(hidden)
    hidden_dequant = dequantize_nvfp4(
        hidden_packed, hidden_scale, hidden_global_scale
    )
    hidden_native_oracle = _native_activation_oracle(
        hidden, hidden_global_scale
    )
    setup_stage("hidden_ready")

    shapes = {
        "w1": (cfg.expert_intermediate_size, cfg.hidden_size),
        "w3": (cfg.expert_intermediate_size, cfg.hidden_size),
        "w2": (cfg.hidden_size, cfg.expert_intermediate_size),
    }
    packed_weights = {
        tag: torch.empty(
            (
                cfg.experts_per_token,
                rows // 128,
                k // 256,
                SchedNvfp4UmmaPrepack.WEIGHT_TILE_BYTES,
            ),
            dtype=torch.uint8,
            device=device,
        )
        for tag, (rows, k) in shapes.items()
    }
    alpha = {
        tag: torch.zeros(
            (cfg.experts_per_token, 4), dtype=torch.float32, device=device
        )
        for tag in shapes
    }
    up_input_scale = torch.zeros(
        (cfg.experts_per_token, 4), dtype=torch.float32, device=device
    )
    down_input_scale = torch.zeros_like(up_input_scale)
    up_input_scale[:, 0] = hidden_global_scale
    expected_gate = torch.empty(
        (cfg.experts_per_token, cfg.expert_intermediate_size),
        dtype=torch.bfloat16,
        device=device,
    )
    expected_up = torch.empty_like(expected_gate)
    expected_middle = torch.empty_like(expected_gate)
    expected_down = torch.empty(
        (cfg.experts_per_token, cfg.hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )
    middle_native_oracle = torch.empty(
        (
            cfg.experts_per_token,
            cfg.expert_intermediate_size // 256,
            SchedNvfp4UmmaPrepack.ACTIVATION_TILE_BYTES,
        ),
        dtype=torch.uint8,
        device=device,
    )

    for expert in range(cfg.experts_per_token):
        references = {}
        for tag in ("w1", "w3"):
            rows, k = shapes[tag]
            source = torch.randn(
                (rows, k),
                generator=generator,
                dtype=torch.bfloat16,
                device=device,
            ) * 0.05
            weight, scale, global_scale = _prepack_weight(
                source, packed_weights[tag][expert]
            )
            references[tag] = (
                dequantize_nvfp4(weight, scale, global_scale) @ hidden_dequant
            ).to(torch.bfloat16)
            alpha[tag][expert, 0] = global_scale * hidden_global_scale
        expected_gate[expert].copy_(references["w1"])
        expected_up[expert].copy_(references["w3"])
        expected_middle[expert].copy_(
            bounded_swiglu(
                references["w1"], references["w3"], cfg.swiglu_limit
            )
        )

        middle_packed, middle_scale, middle_global = quantize_nvfp4(
            expected_middle[expert]
        )
        middle_dequant = dequantize_nvfp4(
            middle_packed, middle_scale, middle_global
        )
        down_input_scale[expert, 0] = middle_global
        middle_native_oracle[expert].copy_(
            _native_activation_oracle(expected_middle[expert], middle_global)
        )

        rows, k = shapes["w2"]
        source = torch.randn(
            (rows, k),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        ) * 0.05
        weight, scale, global_scale = _prepack_weight(
            source, packed_weights["w2"][expert]
        )
        expected_down[expert].copy_(
            (
                dequantize_nvfp4(weight, scale, global_scale)
                @ middle_dequant
            ).to(torch.bfloat16)
        )
        alpha["w2"][expert, 0] = global_scale * middle_global
        setup_stage(f"expert_{expert}_ready")

    columns: dict[str, list[int]] = {}
    for tag, (rows, _) in shapes.items():
        for m_tile in range(rows // 128):
            columns[f"{tag}.m{m_tile}"] = [
                packed_weights[tag][expert, m_tile, 0].data_ptr()
                for expert in range(cfg.experts_per_token)
            ]
        columns[f"{tag}.alpha"] = [
            _row_pointer(alpha[tag], expert)
            for expert in range(cfg.experts_per_token)
        ]
    columns["up.input_scale"] = [
        _row_pointer(up_input_scale, expert)
        for expert in range(cfg.experts_per_token)
    ]
    columns["down.input_scale"] = [
        _row_pointer(down_input_scale, expert)
        for expert in range(cfg.experts_per_token)
    ]
    owners = (
        *packed_weights.values(),
        *alpha.values(),
        up_input_scale,
        down_input_scale,
    )
    table = RoutedAddressTable.from_pointer_columns(
        columns, device=device, owners=owners
    )

    desired_routes = torch.tensor(
        [5, 2, 4, 1, 3, 0], dtype=torch.int64, device=device
    )
    logits = torch.full(
        (cfg.num_experts,), -20.0, dtype=torch.bfloat16, device=device
    )
    logits[desired_routes] = torch.arange(
        6.0, 0.0, -1.0, dtype=torch.bfloat16, device=device
    )
    bias = torch.zeros((cfg.num_experts,), dtype=torch.float32, device=device)
    hash_indices = torch.zeros((8,), dtype=torch.int32, device=device)
    if args.hash_routing:
        hash_indices[:6].copy_(desired_routes.to(torch.int32))
    route_weights = torch.empty((8,), dtype=torch.float32, device=device)

    native_input = torch.empty_like(hidden_native_oracle).repeat(
        cfg.experts_per_token, 1, 1
    )
    middle = torch.empty_like(expected_middle)
    native_middle = torch.empty_like(middle_native_oracle)
    down = torch.empty_like(expected_down)
    shared = torch.zeros(
        (cfg.hidden_size,), dtype=torch.bfloat16, device=device
    )
    reduced = torch.empty_like(shared)
    profile_operand = torch.zeros((16,), dtype=torch.uint8, device=device)

    launcher = Launcher(device_sms, device=device)
    stages = [
        SequentialStage(
            "route",
            SchedDsv4RouteTop6(
                logits,
                bias,
                hash_indices,
                table.route_indices_storage,
                route_weights,
                hash_routing=args.hash_routing,
                route_scale=cfg.route_scale,
            ),
            1,
            release_group="route_ready",
        )
    ]

    def phase_marker(name: str, event_id: int, wait_group: str):
        return SequentialStage(
            f"profile.{name}",
            _BarrierProfileSchedule(event_id, profile_operand),
            1,
            base_sm=profile_sm,
            wait_group=wait_group,
        )

    if args.profile_phases:
        stages.append(phase_marker("route", 4, "route_ready"))

    def lane(schedule, count: int, base: int) -> SubgridSchedule:
        return SubgridSchedule(schedule, count, base)

    for rank in range(cfg.experts_per_token):
        base = rank * branch_sms
        input_ready = f"expert{rank}.input_ready"
        middle_ready = f"expert{rank}.middle_ready"
        down_ready = f"expert{rank}.down_ready"
        branch_stages = (
                SequentialStage(
                    f"expert{rank}.input_quant",
                    lane(
                        SchedRoutedDsv4Nvfp4QuantUmmaB(
                            table.state,
                            rank,
                            table.field("up.input_scale"),
                            hidden,
                            native_input[rank],
                        ),
                        cfg.hidden_size // 256,
                        base,
                    ),
                    queued_sms,
                    input_role="route",
                    wait_group="route_ready",
                    release_group=input_ready,
                ),
                SequentialStage(
                    f"expert{rank}.w1",
                    lane(
                        SchedRoutedNvfp4GemvUmmaStream(
                            table.state,
                            rank,
                            tuple(
                                table.field(f"w1.m{tile}")
                                for tile in range(cfg.expert_intermediate_size // 128)
                            ),
                            table.field("w1.alpha"),
                            native_input[rank],
                            None,
                            activation_mode="retain",
                            output_mode="retain",
                            output_register=1,
                            output_port=0,
                        ),
                        cfg.expert_intermediate_size // 128,
                        base,
                    ),
                    queued_sms,
                    input_role="route",
                    wait_group=input_ready,
                ),
                SequentialStage(
                    f"expert{rank}.w3",
                    lane(
                        SchedRoutedNvfp4GemvUmmaStream(
                            table.state,
                            rank,
                            tuple(
                                table.field(f"w3.m{tile}")
                                for tile in range(cfg.expert_intermediate_size // 128)
                            ),
                            table.field("w3.alpha"),
                            native_input[rank],
                            None,
                            route_ready=True,
                            activation_mode="reuse",
                            output_mode="retain",
                            output_register=1,
                            output_port=1,
                        ),
                        cfg.expert_intermediate_size // 128,
                        base,
                    ),
                    queued_sms,
                    wait_for_previous=False,
                ),
                SequentialStage(
                    f"expert{rank}.swiglu_shards",
                    lane(
                        SchedDsv4SwiGluShard128(
                            1,
                            0,
                            1,
                            1,
                            middle[rank],
                            swiglu_limit=cfg.swiglu_limit,
                        ),
                        cfg.expert_intermediate_size // 128,
                        base,
                    ),
                    queued_sms,
                    wait_for_previous=False,
                    release_group=middle_ready,
                ),
                SequentialStage(
                    f"expert{rank}.middle_quant",
                    lane(
                        SchedRoutedDsv4Nvfp4QuantUmmaB(
                            table.state,
                            rank,
                            table.field("down.input_scale"),
                            middle[rank],
                            native_middle[rank],
                        ),
                        cfg.expert_intermediate_size // 256,
                        base,
                    ),
                    queued_sms,
                    input_role="route",
                    wait_group=middle_ready,
                    release_group=down_ready,
                ),
                SequentialStage(
                    f"expert{rank}.w2",
                    lane(
                        SchedRoutedNvfp4GemvUmmaStream(
                            table.state,
                            rank,
                            tuple(
                                table.field(f"w2.m{tile}")
                                for tile in range(cfg.hidden_size // 128)
                            ),
                            table.field("w2.alpha"),
                            native_middle[rank],
                            down[rank],
                            activation_mode="load",
                        ),
                        branch_sms,
                        base,
                    ),
                    queued_sms,
                    input_role="route",
                    wait_group=down_ready,
                    release_group="expert_join",
                ),
            )
        for stage_index, stage in enumerate(branch_stages):
            stages.append(stage)
            if args.profile_phases and rank == 0:
                marker = {
                    0: ("input_quant", 5, input_ready),
                    3: ("gate_up_fused", 6, middle_ready),
                    4: ("middle_quant", 7, down_ready),
                }.get(stage_index)
                if marker is not None:
                    stages.append(phase_marker(*marker))

    if args.profile_phases:
        stages.append(phase_marker("expert_join", 8, "expert_join"))

    stages.append(
        SequentialStage(
            "expert_reduce",
            SchedDsv4ExpertReduce(
                down, route_weights[:6], shared, reduced
            ),
            1,
            base_sm=reducer_sm,
            wait_group="expert_join",
            release_group="flow_done" if args.profile_phases else None,
        )
    )
    if args.profile_phases:
        stages.append(phase_marker("expert_reduce", 9, "flow_done"))
    program = SequentialProgram(
        launcher, stages, balance_load_ports=True
    )
    launcher.s(
        _BarrierProfileSchedule(2, profile_operand).place(device_sms),
        program,
        ProfileEvent(3),
    )
    setup_stage("launcher_ready")

    launcher.launch()
    torch.cuda.synchronize()
    expected_weights, expected_indices = route_top6_reference(
        logits,
        bias,
        hash_indices=hash_indices[:6] if args.hash_routing else None,
    )
    torch.testing.assert_close(
        table.route_indices_storage[:6], expected_indices, rtol=0, atol=0
    )
    torch.testing.assert_close(
        route_weights[:6], expected_weights, rtol=1.0e-5, atol=1.0e-5
    )
    for rank, expert in enumerate(expected_indices.tolist()):
        torch.testing.assert_close(
            native_input[rank], hidden_native_oracle, rtol=0, atol=0
        )
        torch.testing.assert_close(
            middle[rank], expected_middle[expert], rtol=3.0e-2, atol=6.0e-2
        )
        torch.testing.assert_close(
            native_middle[rank], middle_native_oracle[expert], rtol=0, atol=0
        )
        torch.testing.assert_close(
            down[rank], expected_down[expert], rtol=5.0e-2, atol=1.0e-1
        )
    expected_reduced = torch.sum(
        expected_down[expected_indices.long()].float()
        * expected_weights[:, None],
        dim=0,
    ).to(torch.bfloat16)
    torch.testing.assert_close(
        reduced, expected_reduced, rtol=6.0e-2, atol=2.0e-1
    )
    setup_stage("correctness_passed")

    for _ in range(args.warmup):
        launcher.launch()
    torch.cuda.synchronize()
    kernel_timings = []
    flow_timings = []
    for _ in range(args.iterations):
        launcher.launch()
        profile = launcher.profile[:, :4].cpu().numpy()
        kernel_timings.append(
            (profile[:, 1].max() - profile[:, 0].min()) / 1.0e3
        )
        flow_timings.append(
            (profile[:, 3].max() - profile[:, 2].min()) / 1.0e3
        )

    max_compute = program.max_compute_instructions
    max_memory = program.max_memory_instructions
    print(
        "DSV4_NVFP4_TOP6_RESULT "
        "status=PASS launches=1 routes=6 "
        f"sms={device_sms} branch_sms={branch_sms} "
        f"flow_min_us={min(flow_timings):.6f} "
        f"flow_median_us={statistics.median(flow_timings):.6f} "
        f"flow_max_us={max(flow_timings):.6f} "
        f"kernel_median_us={statistics.median(kernel_timings):.6f} "
        f"compute_insts={max_compute} memory_insts={max_memory}",
        flush=True,
    )

    profile = launcher.profile.cpu()
    if args.profile_phases:
        previous = int(profile[profile_sm, 2])
        for label, event_id in (
            ("route", 4),
            ("input_quant_rank0", 5),
            ("gate_up_fused_rank0", 6),
            ("middle_quant_rank0", 7),
            ("expert_join", 8),
            ("expert_reduce", 9),
        ):
            current = int(profile[profile_sm, event_id])
            if current <= previous:
                raise RuntimeError(
                    f"phase profile event {label!r} is missing or non-monotonic"
                )
            print(
                "DSV4_NVFP4_TOP6_PHASE "
                f"name={label} elapsed_us={(current - previous) / 1.0e3:.6f} "
                f"from_start_us={(current - int(profile[profile_sm, 2])) / 1.0e3:.6f}",
                flush=True,
            )
            previous = current

    track_magic = 0x4454524B50524631
    if all(int(value) == track_magic for value in profile[:, 127]):
        counter_base = runtime_config.track_profile_event_base
        internal_span = max(int(value) for value in profile[:, 1]) - min(
            int(value) for value in profile[:, 0]
        )
        grid_envelope = internal_span * profile.shape[0]

        def counter_sum(offset: int) -> int:
            return sum(int(value) for value in profile[:, counter_base + offset])

        def grid_percent(offset: int) -> float:
            if grid_envelope <= 0:
                return 0.0
            return 100.0 * counter_sum(offset) / grid_envelope

        print(
            "DSV4_NVFP4_TOP6_COUNTERS "
            f"internal_span_us={internal_span / 1.0e3:.6f} "
            f"compute_m2c_wait_grid_pct={grid_percent(0):.3f} "
            f"allocator_slot_stall_grid_pct={grid_percent(3):.3f} "
            f"ldu0_queue_wait_grid_pct={grid_percent(9):.3f} "
            f"ldu0_dependency_wait_grid_pct={grid_percent(11):.3f} "
            f"ldu1_queue_wait_grid_pct={grid_percent(14):.3f} "
            f"ldu1_dependency_wait_grid_pct={grid_percent(16):.3f} "
            f"store_queue_wait_grid_pct={grid_percent(19):.3f} "
            f"store_service_grid_pct={grid_percent(21):.3f} "
            f"allocator_instructions={counter_sum(8)} "
            f"ldu0_commands={counter_sum(13)} "
            f"ldu1_commands={counter_sum(18)} "
            f"store_commands={counter_sum(23)} "
            f"ldu0_dependency_contended={counter_sum(12)} "
            f"ldu1_dependency_contended={counter_sum(17)}",
            flush=True,
        )
        for rank in range(cfg.experts_per_token):
            start = rank * branch_sms
            stop = start + branch_sms
            dependency_ns = [
                int(profile[sm, counter_base + 11])
                + int(profile[sm, counter_base + 16])
                for sm in range(start, stop)
            ]
            queue_ns = [
                int(profile[sm, counter_base + 9])
                + int(profile[sm, counter_base + 14])
                for sm in range(start, stop)
            ]
            print(
                "DSV4_NVFP4_TOP6_BRANCH_COUNTERS "
                f"rank={rank} "
                f"dependency_wait_median_us={statistics.median(dependency_ns) / 1.0e3:.6f} "
                f"dependency_wait_max_us={max(dependency_ns) / 1.0e3:.6f} "
                f"queue_wait_median_us={statistics.median(queue_ns) / 1.0e3:.6f} "
                f"queue_wait_max_us={max(queue_ns) / 1.0e3:.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
