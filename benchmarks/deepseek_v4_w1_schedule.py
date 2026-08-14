#!/usr/bin/env python3
"""Schedule-only frontier search for every DeepSeek-V4 W1 expert.

The timed region contains six dynamically selected NVFP4 W1 projections and
the FP8 shared-expert W1 projection.  Routing ids and native activations are
published before timing.  This benchmark deliberately changes no compute
kernel: candidates differ only in task placement, split-K, and queue order.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass

import torch

from dae import runtime
from dae.instructions import ProfileEvent, TmaLoad1D, TmaTensor
from dae.launcher import Launcher
from dae.routing import RoutedAddressTable
from dae.schedule import (
    ListSchedule,
    Schedule,
    SchedFp8GemvUmmaSplitK,
    SchedFp8GemvUmmaStream,
    SchedNvfp4GemvUmmaStream,
    SchedRoutedNvfp4ExpertGroupSplitK,
    SchedRoutedNvfp4GemvUmmaStream,
    SubgridSchedule,
)
from dae.sequential import SequentialProgram, SequentialStage


HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 2048
ROUTED_EXPERTS = 6
NVFP4_K_TILE = 256
FP8_K_TILE = 128
NVFP4_M_TILES = INTERMEDIATE_SIZE // 128
NVFP4_K_TILES = HIDDEN_SIZE // NVFP4_K_TILE
FP8_M_TILES = INTERMEDIATE_SIZE // 128
FP8_K_TILES = HIDDEN_SIZE // FP8_K_TILE
SCALE_PACK = 2


class _BarrierProfileSchedule(Schedule):
    """Timestamp completion after a barrier-gated memory operation."""

    def __init__(self, event_id: int, operand: torch.Tensor):
        super().__init__()
        self.event_id = int(event_id)
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


@dataclass(frozen=True)
class Candidate:
    name: str
    routed_split_k: int
    routed_sms: int
    shared_split_k: int
    shared_sms: int
    shared_group: int
    shared_base: int
    shared_first: bool = True

    @property
    def routed_work(self) -> int:
        return ROUTED_EXPERTS * NVFP4_M_TILES * self.routed_split_k

    @property
    def shared_work(self) -> int:
        return FP8_M_TILES * self.shared_split_k


@dataclass
class Inputs:
    table: RoutedAddressTable
    activation_nvfp4: torch.Tensor
    weight_fields: dict[int, tuple[tuple[int, ...], ...]]
    alpha_field: int
    weight_fp8: torch.Tensor
    activation_fp8: torch.Tensor
    owners: tuple[object, ...]


@dataclass
class BuiltCandidate:
    launcher: Launcher
    routed_output: torch.Tensor
    shared_output: torch.Tensor
    zero_before_launch: tuple[torch.Tensor, ...]
    marker_sm: int
    program: SequentialProgram

    def reset_and_launch(self):
        for tensor in self.zero_before_launch:
            tensor.zero_()
        self.launcher.launch()


def _row_pointer(tensor: torch.Tensor, row: int) -> int:
    return tensor.data_ptr() + row * tensor.stride(0) * tensor.element_size()


def _make_inputs(device: torch.device, seed: int) -> Inputs:
    def stage(name: str):
        print(f"DSV4_W1_SETUP_STAGE name={name}", flush=True)

    generator = torch.Generator(device=device).manual_seed(seed)
    # The experiment begins after routing and activation conversion.  A zero
    # native operand is valid for both formats and leaves instruction issue,
    # TMA traffic, UMMA work, and writeback unchanged while avoiding any
    # setup-only compute operator in the selective image.
    activation_nvfp4 = torch.zeros(
        (
            ROUTED_EXPERTS,
            NVFP4_K_TILES,
            SchedNvfp4GemvUmmaStream.ACTIVATION_TILE_BYTES,
        ),
        dtype=torch.uint8,
        device=device,
    )
    stage("nvfp4_activation_ready")

    activation_fp8 = torch.zeros(
        (
            FP8_K_TILES,
            SchedFp8GemvUmmaStream.ACTIVATION_TILE_BYTES,
        ),
        dtype=torch.uint8,
        device=device,
    )
    stage("fp8_activation_ready")

    packed_nvfp4 = torch.empty(
        (
            ROUTED_EXPERTS,
            NVFP4_M_TILES,
            NVFP4_K_TILES,
            SchedNvfp4GemvUmmaStream.WEIGHT_TILE_BYTES,
        ),
        dtype=torch.uint8,
        device=device,
    )
    checkpoint_scale = torch.ones(
        (INTERMEDIATE_SIZE, HIDDEN_SIZE // 16),
        dtype=torch.float32,
        device=device,
    ).to(torch.float8_e4m3fn)
    for expert in range(ROUTED_EXPERTS):
        checkpoint_weight = torch.randint(
            0,
            256,
            (INTERMEDIATE_SIZE, HIDDEN_SIZE // 2),
            generator=generator,
            dtype=torch.uint8,
            device=device,
        )
        runtime.prepack_nvfp4_checkpoint(
            checkpoint_weight, checkpoint_scale, packed_nvfp4[expert]
        )
        stage(f"routed_weight_{expert}_ready")

    alpha = torch.zeros(
        (ROUTED_EXPERTS, 4), dtype=torch.float32, device=device
    )
    # The packed random checkpoint uses unit E4M3 scales.  A small synthetic
    # checkpoint-global scale keeps validation outputs finite and well-sized.
    alpha[:, 0] = torch.linspace(
        0.025,
        0.035,
        ROUTED_EXPERTS,
        dtype=torch.float32,
        device=device,
    )

    columns: dict[str, list[int]] = {}
    for m_tile in range(NVFP4_M_TILES):
        for k_start in range(NVFP4_K_TILES):
            columns[f"w1.m{m_tile}.k{k_start}"] = [
                packed_nvfp4[expert, m_tile, k_start].data_ptr()
                for expert in range(ROUTED_EXPERTS)
            ]
    columns["w1.alpha"] = [
        _row_pointer(alpha, expert) for expert in range(ROUTED_EXPERTS)
    ]
    table = RoutedAddressTable.from_pointer_columns(
        columns,
        device=device,
        owners=(packed_nvfp4, alpha),
    )
    # A nontrivial fixed selection proves that W1 uses the routed address
    # path while keeping the routing task and its dependency outside timing.
    table.route_indices_storage[:ROUTED_EXPERTS].copy_(
        torch.tensor([5, 2, 4, 1, 3, 0], dtype=torch.int32, device=device)
    )
    stage("route_table_ready")

    # Byte codes [0,126] are finite positive E4M3 values.  Generating the
    # checkpoint representation directly avoids a large verification-only
    # BF16 -> FP8 conversion while preserving realistic independent bytes.
    checkpoint_fp8 = torch.randint(
        0,
        127,
        (INTERMEDIATE_SIZE, HIDDEN_SIZE),
        generator=generator,
        dtype=torch.uint8,
        device=device,
    ).view(torch.float8_e4m3fn)
    checkpoint_fp8_scale = torch.ones(
        (FP8_M_TILES, FP8_K_TILES),
        dtype=torch.float32,
        device=device,
    ).to(torch.float8_e8m0fnu)
    stage("shared_checkpoint_ready")
    weight_fp8 = torch.empty(
        (
            FP8_M_TILES,
            FP8_K_TILES,
            SchedFp8GemvUmmaStream.WEIGHT_TILE_BYTES,
        ),
        dtype=torch.uint8,
        device=device,
    )
    runtime.prepack_fp8_checkpoint(
        checkpoint_fp8, checkpoint_fp8_scale, weight_fp8, SCALE_PACK
    )
    torch.cuda.synchronize(device)
    stage("shared_weight_ready")

    weight_fields = {}
    for split_k in (2, 4, 8, 16):
        tiles_per_split = NVFP4_K_TILES // split_k
        weight_fields[split_k] = tuple(
            tuple(
                table.field(
                    f"w1.m{m_tile}.k{split * tiles_per_split}"
                )
                for split in range(split_k)
            )
            for m_tile in range(NVFP4_M_TILES)
        )

    return Inputs(
        table=table,
        activation_nvfp4=activation_nvfp4,
        weight_fields=weight_fields,
        alpha_field=table.field("w1.alpha"),
        weight_fp8=weight_fp8,
        activation_fp8=activation_fp8,
        owners=(
            packed_nvfp4,
            alpha,
            weight_fp8,
            activation_nvfp4,
            activation_fp8,
            checkpoint_fp8,
            checkpoint_fp8_scale,
        ),
    )


def _candidate_list(device_sms: int) -> list[Candidate]:
    if device_sms != 152:
        raise ValueError(
            "the checked W1 scheduling frontier is defined for 152 SMs; "
            f"got {device_sms}"
        )
    candidates: list[Candidate] = []

    def add(
        name,
        routed_split,
        routed_sms,
        shared_split,
        shared_sms,
        shared_group,
        shared_base,
        shared_first=True,
    ):
        candidates.append(
            Candidate(
                name,
                routed_split,
                routed_sms,
                shared_split,
                shared_sms,
                shared_group,
                shared_base,
                shared_first,
            )
        )

    # Conventional parallel-expert placement: routed experts own SMs 0..95
    # and shared W1 uses only otherwise idle SMs.
    add("stream96_shared_stream8_g2", 1, 96, 1, 8, 2, 96)
    add("stream96_shared_stream16_g1", 1, 96, 1, 16, 1, 96)
    add("stream96_shared_stream16_g2", 1, 96, 1, 16, 2, 96)
    for shared_sms in (16, 24, 32):
        for group in (1, 2):
            add(
                f"stream96_shared_split2_{shared_sms}_g{group}",
                1,
                96,
                2,
                shared_sms,
                group,
                96,
            )
    for shared_sms in (32, 40, 48, 56):
        for group in (1, 2):
            add(
                f"stream96_shared_split4_{shared_sms}_g{group}",
                1,
                96,
                4,
                shared_sms,
                group,
                96,
            )

    # Disjoint split-K partitions.  Each pair consumes all 152 SMs (except
    # the 96/56 case, which also consumes all SMs).
    disjoint = (
        (96, 56, 4),
        (104, 48, 4),
        (112, 40, 4),
        (120, 32, 2),
        (128, 24, 2),
        (136, 16, 2),
        (144, 8, 1),
    )
    for routed_sms, shared_sms, shared_split in disjoint:
        for group in (1, 2):
            add(
                f"split2_{routed_sms}_shared_split{shared_split}_"
                f"{shared_sms}_g{group}",
                2,
                routed_sms,
                shared_split,
                shared_sms,
                group,
                routed_sms,
            )

    # With 192 routed split-2 tasks on 152 SMs, SMs 0..39 receive two tasks
    # and SMs 40..151 receive one.  Put shared work on the shorter queues.
    for shared_split, shared_sms, group in (
        (2, 16, 2),
        (2, 24, 2),
        (2, 32, 1),
        (4, 32, 2),
        (4, 48, 2),
        (4, 64, 1),
        (8, 64, 2),
        (8, 96, 1),
    ):
        add(
            f"split2_152_shared_split{shared_split}_{shared_sms}_g{group}_first",
            2,
            152,
            shared_split,
            shared_sms,
            group,
            40,
            True,
        )
    add(
        "split2_152_shared_split4_64_g1_last",
        2,
        152,
        4,
        64,
        1,
        40,
        False,
    )

    # Split-4 has 384 tasks: SMs 0..79 own three and SMs 80..151 own two.
    # The same queue-balancing experiment checks whether smaller K shards win
    # after their extra reductions and task setup are counted.
    for shared_split, shared_sms, group in (
        (2, 16, 2),
        (2, 32, 1),
        (4, 32, 2),
        (4, 64, 1),
    ):
        add(
            f"split4_152_shared_split{shared_split}_{shared_sms}_g{group}_first",
            4,
            152,
            shared_split,
            shared_sms,
            group,
            80,
            True,
        )
    add("split4_144_shared_stream8_g2", 4, 144, 1, 8, 2, 144)
    return candidates


def _validate_candidate(candidate: Candidate, device_sms: int):
    if candidate.routed_split_k == 1:
        if candidate.routed_sms != ROUTED_EXPERTS * NVFP4_M_TILES:
            raise ValueError("non-split routed W1 needs exactly 96 SMs")
    elif candidate.routed_work < candidate.routed_sms:
        raise ValueError("routed split-K has fewer work items than SMs")
    if FP8_K_TILES % candidate.shared_split_k:
        raise ValueError("shared split-K must divide K/128")
    if candidate.shared_split_k > 1 and (
        FP8_K_TILES // candidate.shared_split_k
    ) % SCALE_PACK:
        raise ValueError("shared split-K shard must preserve scale packing")
    if candidate.shared_sms > candidate.shared_work:
        raise ValueError("shared W1 has fewer work items than SMs")
    if candidate.shared_group not in (1, 2):
        raise ValueError("shared output grouping must be one or two")
    if candidate.shared_base < 0 or (
        candidate.shared_base + candidate.shared_sms > device_sms
    ):
        raise ValueError("shared placement exceeds device")


def _build_candidate(
    candidate: Candidate,
    inputs: Inputs,
    device: torch.device,
    device_sms: int,
) -> BuiltCandidate:
    _validate_candidate(candidate, device_sms)
    launcher = Launcher(device_sms, device=device)
    zero_before_launch: list[torch.Tensor] = []
    routed_items: list[Schedule] = []

    if candidate.routed_split_k == 1:
        routed_output = torch.empty(
            (ROUTED_EXPERTS, INTERMEDIATE_SIZE),
            dtype=torch.bfloat16,
            device=device,
        )
        fields = tuple(
            inputs.table.field(f"w1.m{m_tile}.k0")
            for m_tile in range(NVFP4_M_TILES)
        )
        for rank in range(ROUTED_EXPERTS):
            schedule = SchedRoutedNvfp4GemvUmmaStream(
                inputs.table.state,
                rank,
                fields,
                inputs.alpha_field,
                inputs.activation_nvfp4[rank],
                routed_output[rank],
                route_ready=True,
                activation_mode="load",
                output_mode="store",
                pipeline=True,
                activation_tiles_per_load=4,
            )
            routed_items.append(
                SubgridSchedule(
                    schedule,
                    NVFP4_M_TILES,
                    rank * NVFP4_M_TILES,
                )
            )
    else:
        routed_output = torch.zeros(
            (ROUTED_EXPERTS, INTERMEDIATE_SIZE),
            dtype=torch.float32,
            device=device,
        )
        zero_before_launch.append(routed_output)
        routed_reduce = TmaTensor(
            launcher, routed_output
        ).rowmajor_2d("reduce", 1, 128)
        grouped = SchedRoutedNvfp4ExpertGroupSplitK(
            inputs.table.state,
            (inputs.weight_fields[candidate.routed_split_k],),
            (inputs.alpha_field,),
            inputs.activation_nvfp4,
            (routed_reduce,),
            torch.ones((1,), dtype=torch.float32, device=device),
            candidate.routed_split_k,
            route_ready=True,
            pipeline=True,
            activation_tiles_per_load=min(
                4, NVFP4_K_TILES // candidate.routed_split_k
            ),
        )
        routed_items.append(
            SubgridSchedule(grouped, candidate.routed_sms, 0)
        )

    if candidate.shared_split_k == 1:
        shared_output = torch.empty(
            (INTERMEDIATE_SIZE,), dtype=torch.bfloat16, device=device
        )
        shared_schedule = SchedFp8GemvUmmaStream(
            inputs.weight_fp8,
            inputs.activation_fp8,
            shared_output,
            SCALE_PACK,
            candidate.shared_group,
        )
    else:
        shared_accumulator = torch.zeros(
            (1, INTERMEDIATE_SIZE), dtype=torch.float32, device=device
        )
        zero_before_launch.append(shared_accumulator)
        shared_output = shared_accumulator[0]
        shared_reduce = TmaTensor(
            launcher, shared_accumulator
        ).rowmajor_2d("reduce", 1, 128)
        shared_schedule = SchedFp8GemvUmmaSplitK(
            inputs.weight_fp8,
            inputs.activation_fp8,
            shared_reduce,
            candidate.shared_split_k,
            SCALE_PACK,
            candidate.shared_group,
        )
    shared_item = SubgridSchedule(
        shared_schedule, candidate.shared_sms, candidate.shared_base
    )

    items = (
        [shared_item, *routed_items]
        if candidate.shared_first
        else [*routed_items, shared_item]
    )
    frontier = ListSchedule(items)
    marker_operand = torch.zeros((16,), dtype=torch.uint8, device=device)
    marker_sm = device_sms - 1
    program = SequentialProgram(
        launcher,
        (
            SequentialStage(
                "all_w1",
                frontier,
                device_sms,
                release_group="all_w1_done",
            ),
            SequentialStage(
                "profile.all_w1",
                _BarrierProfileSchedule(3, marker_operand),
                1,
                base_sm=marker_sm,
                wait_group="all_w1_done",
            ),
        ),
        balance_load_ports=False,
    )
    launcher.s(ProfileEvent(2), program)
    # Preserve every device allocation referenced by encoded instructions.
    launcher._w1_schedule_owners = (inputs, marker_operand, program)
    return BuiltCandidate(
        launcher=launcher,
        routed_output=routed_output,
        shared_output=shared_output,
        zero_before_launch=tuple(zero_before_launch),
        marker_sm=marker_sm,
        program=program,
    )


def _profile_times_us(built: BuiltCandidate) -> tuple[float, float]:
    profile = built.launcher.profile.cpu().numpy()
    begin = min(int(value) for value in profile[:, 2])
    end = int(profile[built.marker_sm, 3])
    kernel_begin = min(int(value) for value in profile[:, 0])
    kernel_end = max(int(value) for value in profile[:, 1])
    return (end - begin) / 1.0e3, (kernel_end - kernel_begin) / 1.0e3


def _check_output(
    candidate: Candidate,
    built: BuiltCandidate,
    golden_routed: torch.Tensor,
    golden_shared: torch.Tensor,
):
    actual_routed = built.routed_output.float()
    actual_shared = built.shared_output.float()
    try:
        torch.testing.assert_close(
            actual_routed,
            golden_routed,
            rtol=6.0e-2,
            atol=2.5e-1,
        )
        torch.testing.assert_close(
            actual_shared,
            golden_shared,
            rtol=6.0e-2,
            atol=2.5e-1,
        )
    except AssertionError as error:
        routed_error = (actual_routed - golden_routed).abs().max().item()
        shared_error = (actual_shared - golden_shared).abs().max().item()
        raise AssertionError(
            f"{candidate.name} validation failed: "
            f"routed_max_abs={routed_error:.6f} "
            f"shared_max_abs={shared_error:.6f}"
        ) from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--candidates",
        default="",
        help="comma-separated candidate names; empty runs the full frontier",
    )
    parser.add_argument("--list-candidates", action="store_true")
    parser.add_argument(
        "--device-sms",
        type=int,
        default=0,
        help="zero selects min(device SMs,152)",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be nonnegative and iterations positive")

    device = torch.device("cuda")
    available_sms = torch.cuda.get_device_properties(device).multi_processor_count
    device_sms = args.device_sms or min(available_sms, 152)
    candidates = _candidate_list(device_sms)
    if args.list_candidates:
        for candidate in candidates:
            print(candidate.name)
        return
    if args.candidates:
        requested = [name.strip() for name in args.candidates.split(",")]
        by_name = {candidate.name: candidate for candidate in candidates}
        missing = [name for name in requested if name not in by_name]
        if missing:
            parser.error(f"unknown candidates: {missing}")
        candidates = [by_name[name] for name in requested]

    print(
        "DSV4_W1_SETUP "
        f"sms={device_sms} routed_experts={ROUTED_EXPERTS} "
        f"shape={INTERMEDIATE_SIZE}x{HIDDEN_SIZE} candidates={len(candidates)}",
        flush=True,
    )
    inputs = _make_inputs(device, args.seed)

    golden_routed = None
    golden_shared = None
    results = []
    for index, candidate in enumerate(candidates):
        built = _build_candidate(candidate, inputs, device, device_sms)
        built.reset_and_launch()
        if golden_routed is None:
            if candidate.name != "stream96_shared_stream8_g2":
                baseline = next(
                    item
                    for item in _candidate_list(device_sms)
                    if item.name == "stream96_shared_stream8_g2"
                )
                golden = _build_candidate(baseline, inputs, device, device_sms)
                golden.reset_and_launch()
                golden_routed = golden.routed_output.float().clone()
                golden_shared = golden.shared_output.float().clone()
            else:
                golden_routed = built.routed_output.float().clone()
                golden_shared = built.shared_output.float().clone()
        _check_output(
            candidate, built, golden_routed, golden_shared
        )

        for _ in range(args.warmup):
            built.reset_and_launch()
        frontier_samples = []
        kernel_samples = []
        for _ in range(args.iterations):
            built.reset_and_launch()
            frontier_us, kernel_us = _profile_times_us(built)
            frontier_samples.append(frontier_us)
            kernel_samples.append(kernel_us)
        median_us = statistics.median(frontier_samples)
        p10_us = sorted(frontier_samples)[
            max(0, int(0.1 * len(frontier_samples)) - 1)
        ]
        kernel_median_us = statistics.median(kernel_samples)
        results.append((median_us, candidate, p10_us, kernel_median_us))
        overlap = max(
            0,
            min(candidate.routed_sms, candidate.shared_base + candidate.shared_sms)
            - candidate.shared_base,
        )
        print(
            "DSV4_W1_RESULT "
            f"index={index} name={candidate.name} "
            f"frontier_median_us={median_us:.6f} "
            f"frontier_p10_us={p10_us:.6f} "
            f"kernel_median_us={kernel_median_us:.6f} "
            f"routed_split_k={candidate.routed_split_k} "
            f"routed_sms={candidate.routed_sms} "
            f"routed_work={candidate.routed_work} "
            f"shared_split_k={candidate.shared_split_k} "
            f"shared_sms={candidate.shared_sms} "
            f"shared_work={candidate.shared_work} "
            f"shared_group={candidate.shared_group} "
            f"shared_base={candidate.shared_base} "
            f"overlap_sms={overlap} "
            f"shared_first={int(candidate.shared_first)}",
            flush=True,
        )

    results.sort(key=lambda item: item[0])
    for rank, (median_us, candidate, p10_us, kernel_us) in enumerate(
        results[: min(10, len(results))], 1
    ):
        print(
            "DSV4_W1_RANK "
            f"rank={rank} name={candidate.name} "
            f"frontier_median_us={median_us:.6f} "
            f"frontier_p10_us={p10_us:.6f} "
            f"kernel_median_us={kernel_us:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
