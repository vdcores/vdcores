#!/usr/bin/env python3
"""Schedule the 4,096 native UMMA tiles in DeepSeek-V4 Linear-1.

The timed graph is deliberately limited to gate and up.  It contains the two
FP8 shared-expert projections (1,024 K128 tiles) and the two NVFP4 projections
for six routed experts (3,072 K256 tiles).  Routing, packing, allocation, and
zeroing happen before the timed frontier.  Candidate plans change only the
worker queues and routed split-K spans; the already-compiled group-1 UMMA
handlers are used unchanged.

Every one of the 152 queues starts with exactly one shared task.  This gives
shared work strict queue priority without a global barrier: each worker can
start its routed tail as soon as its own shared head finishes.
"""

from __future__ import annotations

import argparse
import heapq
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass

import torch

from dae import runtime
from dae.instructions import (
    Fp8GemvUmmaSplitKSm100,
    Nvfp4GemvUmmaPipelineFp32Sm100,
    ProfileEvent,
    RoutedTmaLoad1D,
    RoutedTmaLoadBase1D,
    TmaLoad1D,
    TmaLoadAddressReg1D,
    TmaTensor,
)
from dae.launcher import Launcher
from dae.routing import RoutedAddressTable
from dae.schedule import Schedule
from dae.sequential import SequentialProgram, SequentialStage

from deepseek_v4_w1_schedule import _BarrierProfileSchedule


WORKERS = 152
PROJECTIONS = 2
ROUTED_EXPERTS = 6
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 2048
M_TILES = INTERMEDIATE_SIZE // 128
ROUTED_K_TILES = HIDDEN_SIZE // 256
SHARED_K_TILES = HIDDEN_SIZE // 128
SCALE_PACK = 2

SHARED_ATOMIC_TILES = PROJECTIONS * M_TILES * SHARED_K_TILES
ROUTED_ATOMIC_TILES = (
    PROJECTIONS * ROUTED_EXPERTS * M_TILES * ROUTED_K_TILES
)
TOTAL_ATOMIC_TILES = SHARED_ATOMIC_TILES + ROUTED_ATOMIC_TILES

NVFP4_WEIGHT_TILE_BYTES = 18432
FP8_WEIGHT_DATA_BYTES = 128 * 128
FP8_WEIGHT_SCALE_BYTES = 512
TASK_PROFILE_EVENT_BASE = 4

# Placement estimates are used only to choose a queue.  The reported result
# always comes from the device timestamps.  Shared K6/K8 values come from the
# matched group-1 task profile; the routed model is refined by the sweep.
DEFAULT_SHARED_COST_US = {
    2: 2.20,
    4: 3.25,
    6: 4.256,
    8: 5.184,
    10: 6.432,
    12: 7.296,
    14: 8.30,
    16: 9.25,
}
DEFAULT_ROUTED_OVERHEAD_US = 0.288
DEFAULT_ROUTED_TILE_US = 0.736
MEASURED_ROUTED_COST_US = {
    4: 3.232,
    5: 4.064,
    6: 4.704,
    16: 13.056,
}


@dataclass(frozen=True)
class Linear1Inputs:
    table: RoutedAddressTable
    activation_nvfp4: torch.Tensor
    alpha_fields: tuple[int, ...]
    weight_fp8: torch.Tensor
    activation_fp8: torch.Tensor
    owners: tuple[object, ...]


@dataclass(frozen=True)
class TileChunk:
    kind: str
    projection: int
    m_tile: int
    k_start: int
    k_tiles: int
    route_rank: int = -1

    @property
    def row_key(self) -> tuple[int, int, int]:
        return self.projection, self.route_rank, self.m_tile

    @property
    def label(self) -> str:
        projection = "gate" if self.projection == 0 else "up"
        if self.kind == "shared":
            return (
                f"shared.{projection}.m{self.m_tile}."
                f"k{self.k_start}+{self.k_tiles}"
            )
        return (
            f"r{self.route_rank}.{projection}.m{self.m_tile}."
            f"k{self.k_start}+{self.k_tiles}"
        )


@dataclass(frozen=True)
class PlanSpec:
    name: str
    split_rows: int
    split_spans: tuple[int, ...]
    mixed_spans: tuple[tuple[int, ...], ...] = ()

    @property
    def routed_tasks(self) -> int:
        if self.mixed_spans:
            return (
                PROJECTIONS * ROUTED_EXPERTS * M_TILES
                + sum(len(spans) - 1 for spans in self.mixed_spans)
            )
        return (
            PROJECTIONS * ROUTED_EXPERTS * M_TILES
            + self.split_rows * (len(self.split_spans) - 1)
        )

    @property
    def routed_extra_reductions(self) -> int:
        return self.routed_tasks - PROJECTIONS * ROUTED_EXPERTS * M_TILES


@dataclass
class BuiltPlan:
    spec: PlanSpec
    shared_plan: str
    placement: str
    queues: tuple[tuple[TileChunk, ...], ...]
    launcher: Launcher
    accumulator: torch.Tensor
    marker_sm: int

    def reset_and_launch(self) -> None:
        self.accumulator.zero_()
        self.launcher.launch()

    def sample_us(self) -> tuple[float, float]:
        return _profile_times_us(self.launcher, self.marker_sm)


def _row_pointer(tensor: torch.Tensor, *indices: int) -> int:
    view = tensor
    for index in indices:
        view = view[index]
    return view.data_ptr()


def _make_inputs(device: torch.device, seed: int) -> Linear1Inputs:
    def stage(name: str) -> None:
        print(f"DSV4_LINEAR1_SETUP_STAGE name={name}", flush=True)

    generator = torch.Generator(device=device).manual_seed(seed)
    activation_nvfp4 = torch.zeros(
        (ROUTED_EXPERTS, ROUTED_K_TILES, 3072),
        dtype=torch.uint8,
        device=device,
    )
    activation_fp8 = torch.zeros(
        (SHARED_K_TILES, 2048), dtype=torch.uint8, device=device
    )
    stage("activations_ready")

    packed_nvfp4 = torch.empty(
        (
            PROJECTIONS,
            ROUTED_EXPERTS,
            M_TILES,
            ROUTED_K_TILES,
            NVFP4_WEIGHT_TILE_BYTES,
        ),
        dtype=torch.uint8,
        device=device,
    )
    checkpoint_scale = torch.ones(
        (INTERMEDIATE_SIZE, HIDDEN_SIZE // 16),
        dtype=torch.float32,
        device=device,
    ).to(torch.float8_e4m3fn)
    for projection in range(PROJECTIONS):
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
                checkpoint_weight,
                checkpoint_scale,
                packed_nvfp4[projection, expert],
            )
        stage(f"routed_projection_{projection}_ready")

    alpha = torch.zeros(
        (PROJECTIONS, ROUTED_EXPERTS, 4),
        dtype=torch.float32,
        device=device,
    )
    alpha[:, :, 0] = torch.linspace(
        0.025,
        0.035,
        ROUTED_EXPERTS,
        dtype=torch.float32,
        device=device,
    )
    columns: dict[str, list[int]] = {}
    for projection in range(PROJECTIONS):
        for m_tile in range(M_TILES):
            for k_tile in range(ROUTED_K_TILES):
                columns[f"p{projection}.m{m_tile}.k{k_tile}"] = [
                    packed_nvfp4[
                        projection, expert, m_tile, k_tile
                    ].data_ptr()
                    for expert in range(ROUTED_EXPERTS)
                ]
        columns[f"p{projection}.alpha"] = [
            _row_pointer(alpha, projection, expert)
            for expert in range(ROUTED_EXPERTS)
        ]
    table = RoutedAddressTable.from_pointer_columns(
        columns, device=device, owners=(packed_nvfp4, alpha)
    )
    table.route_indices_storage[:ROUTED_EXPERTS].copy_(
        torch.tensor([5, 2, 4, 1, 3, 0], dtype=torch.int32, device=device)
    )
    stage("routing_ready")

    weight_fp8 = torch.empty(
        (
            PROJECTIONS,
            M_TILES,
            SHARED_K_TILES,
            FP8_WEIGHT_DATA_BYTES + FP8_WEIGHT_SCALE_BYTES,
        ),
        dtype=torch.uint8,
        device=device,
    )
    checkpoint_fp8_scale = torch.ones(
        (M_TILES, SHARED_K_TILES),
        dtype=torch.float32,
        device=device,
    ).to(torch.float8_e8m0fnu)
    for projection in range(PROJECTIONS):
        checkpoint_fp8 = torch.randint(
            0,
            127,
            (INTERMEDIATE_SIZE, HIDDEN_SIZE),
            generator=generator,
            dtype=torch.uint8,
            device=device,
        ).view(torch.float8_e4m3fn)
        runtime.prepack_fp8_checkpoint(
            checkpoint_fp8,
            checkpoint_fp8_scale,
            weight_fp8[projection],
            SCALE_PACK,
        )
        stage(f"shared_projection_{projection}_ready")
    torch.cuda.synchronize(device)

    return Linear1Inputs(
        table=table,
        activation_nvfp4=activation_nvfp4,
        alpha_fields=tuple(
            table.field(f"p{projection}.alpha")
            for projection in range(PROJECTIONS)
        ),
        weight_fp8=weight_fp8,
        activation_fp8=activation_fp8,
        owners=(
            packed_nvfp4,
            alpha,
            weight_fp8,
            activation_nvfp4,
            activation_fp8,
            checkpoint_scale,
            checkpoint_fp8_scale,
        ),
    )


def _parse_plan(name: str) -> PlanSpec:
    if name == "u16":
        return PlanSpec(name, 0, (ROUTED_K_TILES,))
    if name == "jointmix":
        mixed_spans = (
            *((2, 6, 8),) * 10,
            *((4, 6, 6),) * 4,
            *((5, 5, 6),) * 8,
            *((4, 4, 8),) * 5,
            *((3, 5, 8),) * 7,
            (2, 7, 7),
            *((4, 5, 7),) * 2,
            (3, 6, 7),
            *((1, 7, 8),) * 2,
        )
        return PlanSpec(name, len(mixed_spans), (), mixed_spans)
    match = re.fullmatch(r"n(\d+)_p(\d+(?:-\d+)*)", name)
    if match is None:
        raise ValueError(
            f"invalid plan {name!r}; use u16 or n<rows>_p<span-span>"
        )
    split_rows = int(match.group(1))
    spans = tuple(int(value) for value in match.group(2).split("-"))
    if split_rows <= 0 or split_rows > (
        PROJECTIONS * ROUTED_EXPERTS * M_TILES
    ):
        raise ValueError("split-row count must be in [1,192]")
    if len(spans) < 2 or any(span <= 0 for span in spans):
        raise ValueError("a split plan requires at least two positive spans")
    if sum(spans) != ROUTED_K_TILES:
        raise ValueError(
            f"routed spans must cover K{ROUTED_K_TILES}, got {spans}"
        )
    return PlanSpec(name, split_rows, spans)


def _shared_chunks(plan: str) -> list[TileChunk]:
    """Return a feasible 152-chunk partition of all 32 shared rows."""
    chunks: list[TileChunk] = []
    row = 0
    for projection in range(PROJECTIONS):
        for m_tile in range(M_TILES):
            if plan == "uniform68":
                # Smallest possible maximum span when every shared head is
                # optimized in isolation.
                spans = (
                    (8, 6, 6, 6, 6) if row < 24 else (8, 8, 8, 8)
                )
            elif plan == "comp664":
                # Histogram {K4:80,K8:40,K12:32}.  Weighted LPT pairs the
                # K4 heads with K16+K6 tails, K8 with K16+K4, and K12 with
                # K16-only tails.
                spans = (
                    (12, 8, 8, 4)
                    if row < 8
                    else (12, 8, 4, 4, 4)
                )
            elif plan == "comp655":
                # Feasible complement for K6/K5/K5 routed shards.  The 24
                # five-way and eight four-way rows produce
                # {K4:48,K6:72,K12:24,K14:8}.
                spans = (
                    (12, 6, 6, 4, 4)
                    if row < 24
                    else (14, 6, 6, 6)
                )
            elif plan == "comp44":
                # Complement for 44 K6/K5/K5 split rows when routed tails
                # are packed independently and paired longest-to-shortest.
                if row < 2:
                    spans = (10, 10, 12)
                elif row < 14:
                    spans = (6, 12, 14)
                elif row == 14:
                    spans = (4, 4, 12, 12)
                elif row < 30:
                    spans = (4, 4, 6, 6, 6, 6)
                else:
                    spans = (4, 4, 4, 4, 4, 4, 4, 4)
            elif plan == "jointmix":
                row_spans = (
                    *((2, 2, 4, 12, 12),) * 3,
                    *((2, 4, 6, 10, 10),) * 2,
                    *((2, 2, 8, 8, 12),) * 4,
                    *((2, 4, 6, 8, 12),) * 2,
                    (4, 6, 6, 6, 10),
                    (4, 4, 6, 8, 10),
                    *((2, 6, 6, 6, 12),) * 2,
                    *((2, 2, 6, 10, 12),) * 3,
                    (2, 4, 6, 6, 14),
                    *((4, 4, 4, 10, 10),) * 2,
                    (2, 6, 6, 8, 10),
                    (4, 4, 4, 8, 12),
                    (4, 4, 4, 6, 14),
                    *((4, 8, 8, 12),) * 3,
                    (6, 6, 8, 12),
                    (4, 6, 10, 12),
                    (2, 8, 10, 12),
                    (2, 6, 12, 12),
                    (2, 8, 8, 14),
                )
                if len(row_spans) != PROJECTIONS * M_TILES:
                    raise AssertionError("joint mixed shared plan needs 32 rows")
                spans = row_spans[row]
            else:
                raise ValueError(f"unknown shared plan: {plan}")
            k_start = 0
            for span in spans:
                chunks.append(
                    TileChunk(
                        "shared", projection, m_tile, k_start, span
                    )
                )
                k_start += span
            if k_start != SHARED_K_TILES:
                raise AssertionError("shared row coverage is incomplete")
            row += 1
    if len(chunks) != WORKERS:
        raise AssertionError("shared head must contain exactly 152 tasks")
    return chunks


def _routed_chunks(spec: PlanSpec) -> list[TileChunk]:
    chunks: list[TileChunk] = []
    row = 0
    total_rows = PROJECTIONS * ROUTED_EXPERTS * M_TILES
    # Spread split rows through gate/up and all six experts rather than giving
    # one projection or expert a systematically different queue shape.
    split_row_ids = {
        (index * total_rows) // spec.split_rows
        for index in range(spec.split_rows)
    } if spec.split_rows else set()
    if len(split_row_ids) != spec.split_rows:
        raise AssertionError("failed to select the requested routed rows")
    split_span_by_row = {
        row_id: spec.mixed_spans[index]
        for index, row_id in enumerate(sorted(split_row_ids))
    } if spec.mixed_spans else {}
    for projection in range(PROJECTIONS):
        for route_rank in range(ROUTED_EXPERTS):
            for m_tile in range(M_TILES):
                spans = (
                    split_span_by_row.get(row, spec.split_spans)
                    if row in split_row_ids
                    else (ROUTED_K_TILES,)
                )
                k_start = 0
                for span in spans:
                    chunks.append(
                        TileChunk(
                            "routed",
                            projection,
                            m_tile,
                            k_start,
                            span,
                            route_rank,
                        )
                    )
                    k_start += span
                if k_start != ROUTED_K_TILES:
                    raise AssertionError("routed row coverage is incomplete")
                row += 1
    if len(chunks) != spec.routed_tasks:
        raise AssertionError("unexpected routed task count")
    return chunks


def _routed_cost(
    chunk: TileChunk, routed_overhead_us: float, routed_tile_us: float
) -> float:
    return MEASURED_ROUTED_COST_US.get(
        chunk.k_tiles,
        routed_overhead_us + routed_tile_us * chunk.k_tiles,
    )


def _pack_queues(
    spec: PlanSpec,
    *,
    shared_plan: str,
    placement: str,
    routed_overhead_us: float,
    routed_tile_us: float,
) -> tuple[tuple[TileChunk, ...], ...]:
    """Place all tasks with strict shared heads and weighted routed tails."""
    shared = sorted(
        _shared_chunks(shared_plan),
        key=lambda chunk: (
            -chunk.k_tiles,
            chunk.projection,
            chunk.m_tile,
            chunk.k_start,
        ),
    )
    if placement == "head_lpt":
        queues: list[list[TileChunk]] = [[chunk] for chunk in shared]
        loads = [DEFAULT_SHARED_COST_US[chunk.k_tiles] for chunk in shared]
    elif placement == "joint":
        queues = [[] for _ in range(WORKERS)]
        loads = [0.0] * WORKERS
    else:
        raise ValueError(f"unknown placement: {placement}")
    heap = [(load, worker) for worker, load in enumerate(loads)]
    heapq.heapify(heap)
    used_workers: dict[tuple[int, int, int], set[int]] = defaultdict(set)
    routed = sorted(
        _routed_chunks(spec),
        key=lambda chunk: (
            -_routed_cost(chunk, routed_overhead_us, routed_tile_us),
            chunk.projection,
            chunk.route_rank,
            chunk.m_tile,
            chunk.k_start,
        ),
    )
    for chunk in routed:
        skipped: list[tuple[float, int]] = []
        used = used_workers[chunk.row_key]
        while True:
            load, worker = heapq.heappop(heap)
            if worker not in used:
                break
            skipped.append((load, worker))
        for item in skipped:
            heapq.heappush(heap, item)
        queues[worker].append(chunk)
        used.add(worker)
        load += _routed_cost(chunk, routed_overhead_us, routed_tile_us)
        loads[worker] = load
        heapq.heappush(heap, (load, worker))

    if placement == "joint":
        # First make the routed-only bins as equal as possible.  Then exploit
        # the freedom to assign any shared output shard to any worker: pair
        # the longest routed tail with the shortest shared head.  This avoids
        # stranding a multi-task routed tail behind a long shared chunk.
        tails = sorted(
            zip(loads, queues),
            key=lambda item: (
                -item[0],
                tuple(chunk.label for chunk in item[1]),
            ),
        )
        shared_by_cost = sorted(
            shared,
            key=lambda chunk: (
                DEFAULT_SHARED_COST_US[chunk.k_tiles],
                chunk.projection,
                chunk.m_tile,
                chunk.k_start,
            ),
        )
        queues = [
            [head, *tail]
            for head, (_, tail) in zip(shared_by_cost, tails)
        ]

    packed = tuple(tuple(queue) for queue in queues)
    _validate_queues(packed)
    return packed


def _validate_queues(queues: tuple[tuple[TileChunk, ...], ...]) -> None:
    if len(queues) != WORKERS:
        raise AssertionError("Linear-1 schedule must use all 152 workers")
    if any(not queue or queue[0].kind != "shared" for queue in queues):
        raise AssertionError("every worker queue must start with shared")
    if any(
        chunk.kind == "shared"
        for queue in queues
        for chunk in queue[1:]
    ):
        raise AssertionError("shared work may only appear at queue heads")

    coverage: dict[tuple[str, int, int, int], list[int]] = defaultdict(list)
    atomic_tiles = 0
    for queue in queues:
        for chunk in queue:
            key = (
                chunk.kind,
                chunk.projection,
                chunk.route_rank,
                chunk.m_tile,
            )
            coverage[key].extend(
                range(chunk.k_start, chunk.k_start + chunk.k_tiles)
            )
            atomic_tiles += chunk.k_tiles
    expected_rows = PROJECTIONS * M_TILES + (
        PROJECTIONS * ROUTED_EXPERTS * M_TILES
    )
    if len(coverage) != expected_rows:
        raise AssertionError("Linear-1 schedule lost output rows")
    for key, k_tiles in coverage.items():
        expected = SHARED_K_TILES if key[0] == "shared" else ROUTED_K_TILES
        if sorted(k_tiles) != list(range(expected)):
            raise AssertionError(f"invalid K coverage for {key}: {k_tiles}")
    if atomic_tiles != TOTAL_ATOMIC_TILES:
        raise AssertionError(
            f"expected {TOTAL_ATOMIC_TILES} atomic tiles, got {atomic_tiles}"
        )


class SchedLinear1Workers(Schedule):
    """Render one strict-priority Linear-1 queue per SM."""

    def __init__(
        self,
        inputs: Linear1Inputs,
        output_reduce: TmaTensor,
        output_scale: torch.Tensor,
        queues: tuple[tuple[TileChunk, ...], ...],
        *,
        profile_tasks: bool = False,
    ):
        super().__init__()
        self.inputs = inputs
        self.output_reduce = output_reduce
        self.output_scale = output_scale
        self.queues = queues
        self.profile_tasks = bool(profile_tasks)

    def _on_place(self) -> None:
        if self.num_sms != WORKERS or len(self.queues) != WORKERS:
            raise ValueError("Linear-1 worker schedule requires 152 SMs")
        output = getattr(self.output_reduce, "mat", None)
        if (
            getattr(self.output_reduce, "mode", None) != "reduce"
            or output is None
            or output.dtype != torch.float32
            or tuple(output.shape)
            != (PROJECTIONS * (ROUTED_EXPERTS + 1), INTERMEDIATE_SIZE)
        ):
            raise ValueError("Linear-1 requires FP32 reduce [14,2048]")

    def _routed_task(self, chunk: TileChunk) -> list[object]:
        batch = min(4, chunk.k_tiles)
        instructions: list[object] = [
            Nvfp4GemvUmmaPipelineFp32Sm100(
                chunk.k_tiles, activation_tiles_per_load=batch
            ),
            RoutedTmaLoad1D(
                self.inputs.table.state,
                chunk.route_rank,
                self.inputs.alpha_fields[chunk.projection],
                16,
            ).fixed_port(1),
            TmaLoad1D(self.output_scale).fixed_port(1),
        ]
        for local_start in range(0, chunk.k_tiles, batch):
            local_stop = min(local_start + batch, chunk.k_tiles)
            absolute_start = chunk.k_start + local_start
            absolute_stop = chunk.k_start + local_stop
            instructions.append(
                TmaLoad1D(
                    self.inputs.activation_nvfp4[
                        chunk.route_rank, absolute_start:absolute_stop
                    ].reshape(-1)
                ).fixed_port(1)
            )
            for local_k in range(local_start, local_stop):
                if local_k == 0:
                    load = RoutedTmaLoadBase1D(
                        self.inputs.table.state,
                        chunk.route_rank,
                        self.inputs.table.field(
                            f"p{chunk.projection}.m{chunk.m_tile}."
                            f"k{chunk.k_start}"
                        ),
                        NVFP4_WEIGHT_TILE_BYTES,
                    )
                else:
                    load = TmaLoadAddressReg1D(
                        RoutedTmaLoadBase1D.ADDRESS_REGISTER,
                        local_k * NVFP4_WEIGHT_TILE_BYTES,
                        NVFP4_WEIGHT_TILE_BYTES,
                    )
                instructions.append(load.fixed_port(0))
        output_row = (
            chunk.projection * (ROUTED_EXPERTS + 1) + chunk.route_rank
        )
        instructions.append(
            self.output_reduce.cord(output_row, chunk.m_tile * 128)
        )
        return instructions

    def _shared_task(self, chunk: TileChunk) -> list[object]:
        if chunk.k_start % SCALE_PACK or chunk.k_tiles % SCALE_PACK:
            raise ValueError("shared chunks must preserve scale-pack pairs")
        instructions: list[object] = [
            Fp8GemvUmmaSplitKSm100(
                chunk.k_tiles, 4, SCALE_PACK, 1
            )
        ]
        for activation_start in range(
            chunk.k_start, chunk.k_start + chunk.k_tiles, 8
        ):
            activation_stop = min(
                chunk.k_start + chunk.k_tiles, activation_start + 8
            )
            instructions.append(
                TmaLoad1D(
                    self.inputs.activation_fp8[
                        activation_start:activation_stop
                    ].reshape(-1)
                ).fixed_port(1)
            )
            for scale_start in range(
                activation_start, activation_stop, SCALE_PACK
            ):
                for k_tile in range(scale_start, scale_start + SCALE_PACK):
                    weight = self.inputs.weight_fp8[
                        chunk.projection, chunk.m_tile, k_tile
                    ].reshape(-1)
                    if k_tile % SCALE_PACK:
                        weight = weight[:FP8_WEIGHT_DATA_BYTES]
                    instructions.append(TmaLoad1D(weight).fixed_port(0))
        output_row = (
            chunk.projection * (ROUTED_EXPERTS + 1) + ROUTED_EXPERTS
        )
        instructions.append(
            self.output_reduce.cord(output_row, chunk.m_tile * 128)
        )
        return instructions

    def schedule(self, sm: int) -> list[object]:
        if sm < 0 or sm >= self.num_sms:
            return []
        instructions: list[object] = []
        queue = self.queues[sm]
        for task_index, chunk in enumerate(queue):
            task = (
                self._shared_task(chunk)
                if chunk.kind == "shared"
                else self._routed_task(chunk)
            )
            if task_index + 1 == len(queue):
                task[-1].bar(self._bar("output"))
            instructions.extend(task)
            if self.profile_tasks:
                instructions.append(
                    ProfileEvent(TASK_PROFILE_EVENT_BASE + task_index)
                )
        return instructions

    def bar_release_count(self, role: str) -> int:
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


def _profile_times_us(launcher: Launcher, marker_sm: int) -> tuple[float, float]:
    profile = launcher.profile.cpu().numpy()
    begin = min(int(value) for value in profile[:, 2])
    end = int(profile[marker_sm, 3])
    kernel_begin = min(int(value) for value in profile[:, 0])
    kernel_end = max(int(value) for value in profile[:, 1])
    return (end - begin) / 1.0e3, (kernel_end - kernel_begin) / 1.0e3


def _predicted_cost(
    queue: tuple[TileChunk, ...],
    routed_overhead_us: float,
    routed_tile_us: float,
) -> float:
    value = DEFAULT_SHARED_COST_US[queue[0].k_tiles]
    return value + sum(
        _routed_cost(chunk, routed_overhead_us, routed_tile_us)
        for chunk in queue[1:]
    )


def _print_plan(
    spec: PlanSpec,
    queues: tuple[tuple[TileChunk, ...], ...],
    *,
    shared_plan: str,
    placement: str,
    routed_overhead_us: float,
    routed_tile_us: float,
    print_queues: bool,
) -> None:
    all_chunks = [chunk for queue in queues for chunk in queue]
    shared_chunks = [chunk for chunk in all_chunks if chunk.kind == "shared"]
    routed_chunks = [chunk for chunk in all_chunks if chunk.kind == "routed"]
    queue_tasks = [len(queue) for queue in queues]
    routed_tiles = [
        sum(chunk.k_tiles for chunk in queue if chunk.kind == "routed")
        for queue in queues
    ]
    predicted = [
        _predicted_cost(queue, routed_overhead_us, routed_tile_us)
        for queue in queues
    ]
    print(
        "DSV4_LINEAR1_PLAN "
        f"name={spec.name} shared_plan={shared_plan} placement={placement} "
        f"workers={WORKERS} "
        f"atomic_tiles={TOTAL_ATOMIC_TILES} "
        f"shared_atomic_tiles={SHARED_ATOMIC_TILES} "
        f"routed_atomic_tiles={ROUTED_ATOMIC_TILES} "
        f"shared_tasks={len(shared_chunks)} routed_tasks={len(routed_chunks)} "
        f"routed_extra_reductions={spec.routed_extra_reductions} "
        f"shared_span_hist={dict(sorted(Counter(c.k_tiles for c in shared_chunks).items()))} "
        f"routed_span_hist={dict(sorted(Counter(c.k_tiles for c in routed_chunks).items()))} "
        f"queue_task_hist={dict(sorted(Counter(queue_tasks).items()))} "
        f"routed_tile_load_hist={dict(sorted(Counter(routed_tiles).items()))} "
        f"predicted_min_us={min(predicted):.3f} "
        f"predicted_max_us={max(predicted):.3f}",
        flush=True,
    )
    if print_queues:
        for worker, queue in enumerate(queues):
            print(
                "DSV4_LINEAR1_QUEUE "
                f"name={spec.name} shared_plan={shared_plan} "
                f"placement={placement} worker={worker} "
                f"predicted_us={predicted[worker]:.3f} "
                f"tasks={','.join(chunk.label for chunk in queue)}",
                flush=True,
            )


def _record_task_profiles(
    profile,
    queues: tuple[tuple[TileChunk, ...], ...],
    elapsed_samples: dict[tuple[int, str, int], list[float]],
) -> None:
    for sm, queue in enumerate(queues):
        previous = int(profile[sm, 2])
        for task_index, chunk in enumerate(queue):
            current = int(profile[sm, TASK_PROFILE_EVENT_BASE + task_index])
            elapsed_samples[(task_index, chunk.kind, chunk.k_tiles)].append(
                (current - previous) / 1.0e3
            )
            previous = current


def _build_plan(
    spec: PlanSpec,
    inputs: Linear1Inputs,
    device: torch.device,
    *,
    shared_plan: str,
    placement: str,
    routed_overhead_us: float,
    routed_tile_us: float,
    profile_tasks: bool,
    print_queues: bool,
) -> BuiltPlan:
    queues = _pack_queues(
        spec,
        shared_plan=shared_plan,
        placement=placement,
        routed_overhead_us=routed_overhead_us,
        routed_tile_us=routed_tile_us,
    )
    _print_plan(
        spec,
        queues,
        shared_plan=shared_plan,
        placement=placement,
        routed_overhead_us=routed_overhead_us,
        routed_tile_us=routed_tile_us,
        print_queues=print_queues,
    )
    launcher = Launcher(WORKERS, device=device)
    accumulator = torch.zeros(
        (PROJECTIONS * (ROUTED_EXPERTS + 1), INTERMEDIATE_SIZE),
        dtype=torch.float32,
        device=device,
    )
    output_reduce = TmaTensor(launcher, accumulator).rowmajor_2d(
        "reduce", 1, 128
    )
    output_scale = torch.ones((4,), dtype=torch.float32, device=device)
    schedule = SchedLinear1Workers(
        inputs,
        output_reduce,
        output_scale,
        queues,
        profile_tasks=profile_tasks,
    )
    marker_operand = torch.zeros((16,), dtype=torch.uint8, device=device)
    marker_sm = WORKERS - 1
    program = SequentialProgram(
        launcher,
        (
            SequentialStage(
                "linear1_gate_up",
                schedule,
                WORKERS,
                release_group="linear1_done",
            ),
            SequentialStage(
                "profile.linear1_gate_up",
                _BarrierProfileSchedule(3, marker_operand),
                1,
                base_sm=marker_sm,
                wait_group="linear1_done",
            ),
        ),
        balance_load_ports=False,
    )
    launcher.s(ProfileEvent(2), program)
    launcher._linear1_owners = (
        inputs,
        accumulator,
        output_scale,
        marker_operand,
        program,
    )

    built = BuiltPlan(
        spec,
        shared_plan,
        placement,
        queues,
        launcher,
        accumulator,
        marker_sm,
    )
    built.reset_and_launch()
    if not bool(torch.isfinite(accumulator).all().item()):
        raise AssertionError("Linear-1 output contains non-finite values")
    return built


def _report_plan_samples(
    built: BuiltPlan,
    frontier_samples: list[float],
    kernel_samples: list[float],
    task_samples: dict[tuple[int, str, int], list[float]] | None = None,
) -> tuple[float, float]:
    spec = built.spec
    shared_plan = built.shared_plan
    placement = built.placement
    ordered = sorted(frontier_samples)
    median = statistics.median(frontier_samples)
    kernel_median = statistics.median(kernel_samples)
    print(
        "DSV4_LINEAR1_RESULT "
        f"name={spec.name} shared_plan={shared_plan} placement={placement} "
        f"frontier_median_us={median:.6f} "
        f"frontier_p10_us={ordered[max(0, len(ordered) // 10 - 1)]:.6f} "
        f"frontier_min_us={min(frontier_samples):.6f} "
        f"kernel_median_us={kernel_median:.6f}",
        flush=True,
    )
    if task_samples:
        for key in sorted(task_samples):
            task_index, kind, k_tiles = key
            samples = task_samples[key]
            print(
                "DSV4_LINEAR1_TASK_PROFILE "
                f"name={spec.name} shared_plan={shared_plan} "
                f"placement={placement} "
                f"queue_index={task_index} kind={kind} "
                f"k_tiles={k_tiles} samples={len(samples)} "
                f"median_us={statistics.median(samples):.6f}",
                flush=True,
            )
    return median, kernel_median


def _run_plan(
    spec: PlanSpec,
    inputs: Linear1Inputs,
    device: torch.device,
    *,
    shared_plan: str,
    placement: str,
    warmup: int,
    iterations: int,
    routed_overhead_us: float,
    routed_tile_us: float,
    profile_tasks: bool,
    print_queues: bool,
) -> tuple[float, float]:
    built = _build_plan(
        spec,
        inputs,
        device,
        shared_plan=shared_plan,
        placement=placement,
        routed_overhead_us=routed_overhead_us,
        routed_tile_us=routed_tile_us,
        profile_tasks=profile_tasks,
        print_queues=print_queues,
    )
    for _ in range(warmup):
        built.reset_and_launch()
    frontier_samples: list[float] = []
    kernel_samples: list[float] = []
    task_samples: dict[tuple[int, str, int], list[float]] = defaultdict(list)
    for _ in range(iterations):
        built.reset_and_launch()
        frontier, kernel = built.sample_us()
        frontier_samples.append(frontier)
        kernel_samples.append(kernel)
        if profile_tasks:
            _record_task_profiles(
                built.launcher.profile.cpu().numpy(),
                built.queues,
                task_samples,
            )
    return _report_plan_samples(
        built,
        frontier_samples,
        kernel_samples,
        task_samples if profile_tasks else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--plans",
        default="n40_p6-5-5",
        help="comma-separated u16 or n<split rows>_p<K spans>",
    )
    parser.add_argument(
        "--routed-overhead-us",
        type=float,
        default=DEFAULT_ROUTED_OVERHEAD_US,
    )
    parser.add_argument(
        "--routed-tile-us", type=float, default=DEFAULT_ROUTED_TILE_US
    )
    parser.add_argument(
        "--shared-plans",
        default="comp655",
        help=(
            "comma-separated uniform68, comp664, comp655, comp44, "
            "or jointmix"
        ),
    )
    parser.add_argument(
        "--placement", choices=("joint", "head_lpt"), default="joint"
    )
    parser.add_argument(
        "--interleave",
        action="store_true",
        help="round-robin candidate warmups and samples on one GPU",
    )
    parser.add_argument("--profile-tasks", action="store_true")
    parser.add_argument("--print-queues", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be nonnegative and iterations positive")
    if args.routed_overhead_us < 0 or args.routed_tile_us <= 0:
        parser.error("routed cost coefficients must be nonnegative/positive")
    if args.interleave and args.profile_tasks:
        parser.error("task profiling and candidate interleave are exclusive")
    try:
        specs = tuple(
            _parse_plan(name.strip())
            for name in args.plans.split(",")
            if name.strip()
        )
    except ValueError as error:
        parser.error(str(error))
    if not specs:
        parser.error("at least one plan is required")
    shared_plans = tuple(
        name.strip() for name in args.shared_plans.split(",") if name.strip()
    )
    if not shared_plans or any(
        name
        not in ("uniform68", "comp664", "comp655", "comp44", "jointmix")
        for name in shared_plans
    ):
        parser.error(
            "shared plans must be uniform68, comp664, comp655, comp44, "
            "and/or jointmix"
        )

    device = torch.device("cuda")
    device_sms = min(
        torch.cuda.get_device_properties(device).multi_processor_count,
        WORKERS,
    )
    if device_sms != WORKERS:
        parser.error(f"this schedule requires 152 SMs, got {device_sms}")
    inputs = _make_inputs(device, args.seed)
    results = []
    candidate_keys = [
        (spec, shared_plan)
        for shared_plan in shared_plans
        for spec in specs
    ]
    if args.interleave:
        built_plans = [
            _build_plan(
                spec,
                inputs,
                device,
                shared_plan=shared_plan,
                placement=args.placement,
                routed_overhead_us=args.routed_overhead_us,
                routed_tile_us=args.routed_tile_us,
                profile_tasks=False,
                print_queues=args.print_queues,
            )
            for spec, shared_plan in candidate_keys
        ]
        # One round is one launch of every candidate.  This keeps clock,
        # thermal, and cache-state drift common to all candidates.
        for _ in range(args.warmup):
            for built in built_plans:
                built.reset_and_launch()
        frontier_samples = [[] for _ in built_plans]
        kernel_samples = [[] for _ in built_plans]
        for _ in range(args.iterations):
            for index, built in enumerate(built_plans):
                built.reset_and_launch()
                frontier, kernel = built.sample_us()
                frontier_samples[index].append(frontier)
                kernel_samples[index].append(kernel)
        for built, frontiers, kernels in zip(
            built_plans, frontier_samples, kernel_samples
        ):
            median, kernel_median = _report_plan_samples(
                built, frontiers, kernels
            )
            results.append(
                (
                    median,
                    built.spec.routed_extra_reductions,
                    built.spec.name,
                    built.shared_plan,
                    built.placement,
                    kernel_median,
                )
            )
    else:
        for spec, shared_plan in candidate_keys:
            median, kernel_median = _run_plan(
                spec,
                inputs,
                device,
                shared_plan=shared_plan,
                placement=args.placement,
                warmup=args.warmup,
                iterations=args.iterations,
                routed_overhead_us=args.routed_overhead_us,
                routed_tile_us=args.routed_tile_us,
                profile_tasks=args.profile_tasks,
                print_queues=args.print_queues,
            )
            results.append(
                (
                    median,
                    spec.routed_extra_reductions,
                    spec.name,
                    shared_plan,
                    args.placement,
                    kernel_median,
                )
            )
    print("DSV4_LINEAR1_RANKING", flush=True)
    for rank, (
        median,
        extra,
        name,
        shared_plan,
        placement,
        kernel_median,
    ) in enumerate(
        sorted(results), start=1
    ):
        print(
            "DSV4_LINEAR1_RANK "
            f"rank={rank} name={name} shared_plan={shared_plan} "
            f"placement={placement} "
            f"frontier_median_us={median:.6f} "
            f"routed_extra_reductions={extra} "
            f"kernel_median_us={kernel_median:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
