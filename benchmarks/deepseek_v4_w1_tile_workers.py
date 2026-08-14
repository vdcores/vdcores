#!/usr/bin/env python3
"""Compare one-wave and two-wave all-UMMA W1 worker schedules.

This is a schedule-only experiment.  It uses the already-compiled NVFP4 and
FP8 UMMA compute operators with arbitrary contiguous K spans; no compute
kernel or operator image needs to be rebuilt.  Wave counters describe queue
entries and never insert a barrier between waves.
"""

from __future__ import annotations

import argparse
import heapq
import statistics
from collections import defaultdict
from dataclasses import dataclass

import torch

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
from dae.schedule import Schedule
from dae.sequential import SequentialProgram, SequentialStage

from deepseek_v4_w1_schedule import (
    FP8_K_TILES,
    FP8_M_TILES,
    INTERMEDIATE_SIZE,
    NVFP4_K_TILES,
    NVFP4_M_TILES,
    ROUTED_EXPERTS,
    SCALE_PACK,
    _BarrierProfileSchedule,
    _make_inputs,
)


WORKERS = 152
NVFP4_WEIGHT_TILE_BYTES = 18432
NVFP4_ACTIVATION_TILE_BYTES = 3072
FP8_WEIGHT_DATA_BYTES = 128 * 128
FP8_WEIGHT_SCALE_BYTES = 512
FP8_WEIGHT_TILE_BYTES = FP8_WEIGHT_DATA_BYTES + FP8_WEIGHT_SCALE_BYTES
FP8_ACTIVATION_TILE_BYTES = 2048
FP32_OUTPUT_TILE_BYTES = 128 * 4
TASK_PROFILE_EVENT_BASE = 4

SCHEDULES = ("one_wave152", "two_wave160")
EXPECTED_LOAD_HISTOGRAM = {8: 32, 10: 16, 12: 8, 16: 96}


@dataclass(frozen=True)
class TileChunk:
    kind: str
    m_tile: int
    k_start: int
    k_tiles: int
    route_rank: int = -1

    @property
    def label(self) -> str:
        if self.kind == "routed":
            return (
                f"r{self.route_rank}.m{self.m_tile}."
                f"k{self.k_start}+{self.k_tiles}"
            )
        return f"shared.m{self.m_tile}.k{self.k_start}+{self.k_tiles}"


def _pack_chunks(
    chunks: list[TileChunk],
) -> tuple[tuple[TileChunk, ...], ...]:
    """LPT-pack explicit contiguous K chunks onto 152 worker queues."""
    # Longest-processing-time list scheduling.  Shared chunks win equal-size
    # ties, matching the desired shared-first queue order.  All K spans stay
    # contiguous, so one chunk means one UMMA accumulation and one reduction.
    chunks.sort(
        key=lambda chunk: (
            -chunk.k_tiles,
            0 if chunk.kind == "shared" else 1,
            chunk.route_rank,
            chunk.m_tile,
            chunk.k_start,
        )
    )
    queues: list[list[TileChunk]] = [[] for _ in range(WORKERS)]
    heap = [(0, worker) for worker in range(WORKERS)]
    heapq.heapify(heap)
    for chunk in chunks:
        load, worker = heapq.heappop(heap)
        queues[worker].append(chunk)
        heapq.heappush(heap, (load + chunk.k_tiles, worker))

    queue_loads = [sum(chunk.k_tiles for chunk in queue) for queue in queues]
    total_tiles = (
        ROUTED_EXPERTS * NVFP4_M_TILES * NVFP4_K_TILES
        + FP8_M_TILES * FP8_K_TILES
    )
    if sum(queue_loads) != total_tiles:
        raise AssertionError("tile queue lost work")
    return tuple(tuple(queue) for queue in queues)


def _append_uniform_chunks(
    chunks: list[TileChunk],
    kind: str,
    m_tile: int,
    spans: tuple[int, ...],
    *,
    route_rank: int = -1,
) -> None:
    k_start = 0
    for span in spans:
        if span <= 0:
            raise ValueError("chunk sizes must be positive")
        if kind == "shared" and span % SCALE_PACK:
            raise ValueError(
                "shared chunk sizes must preserve scale-pack pairs"
            )
        chunks.append(
            TileChunk(kind, m_tile, k_start, span, route_rank)
        )
        k_start += span
    expected = FP8_K_TILES if kind == "shared" else NVFP4_K_TILES
    if k_start != expected:
        raise ValueError(f"{kind} chunk plan must cover K{expected}")


def _build_routed_k16_chunks(chunks: list[TileChunk]) -> None:
    for route_rank in range(ROUTED_EXPERTS):
        for m_tile in range(NVFP4_M_TILES):
            _append_uniform_chunks(
                chunks,
                "routed",
                m_tile,
                (NVFP4_K_TILES,),
                route_rank=route_rank,
            )


def build_wave_compare_queues(
    name: str,
) -> tuple[tuple[TileChunk, ...], ...]:
    """Build the only two schedules admitted by this comparison."""
    chunks: list[TileChunk] = []
    _build_routed_k16_chunks(chunks)
    if name == "one_wave152":
        # 96 routed tasks plus exactly 56 shared tasks gives one queue entry
        # on every worker.  The heterogeneous shared spans preserve the same
        # final worker-load histogram as the two-wave schedule.
        for m_tile in range(FP8_M_TILES):
            shared_spans = (
                (12, 10, 10) if m_tile < 8 else (8, 8, 8, 8)
            )
            _append_uniform_chunks(
                chunks, "shared", m_tile, shared_spans
            )
    elif name == "two_wave160":
        # The final 16 K6 chunks pack two-deep on eight workers.  There is no
        # barrier between their first and second queue entries.
        for m_tile in range(FP8_M_TILES):
            _append_uniform_chunks(
                chunks, "shared", m_tile, (10, 8, 8, 6)
            )
    else:
        raise ValueError(f"unknown comparison schedule: {name}")
    return _pack_chunks(chunks)


def _wave_counters(
    queues: tuple[tuple[TileChunk, ...], ...],
) -> tuple[dict[str, int], ...]:
    counters = []
    for wave in range(max(len(queue) for queue in queues)):
        chunks = [queue[wave] for queue in queues if len(queue) > wave]
        routed = [chunk for chunk in chunks if chunk.kind == "routed"]
        shared = [chunk for chunk in chunks if chunk.kind == "shared"]
        counters.append(
            {
                "wave": wave,
                "tasks": len(chunks),
                "tiles": sum(chunk.k_tiles for chunk in chunks),
                "routed_tasks": len(routed),
                "routed_tiles": sum(
                    chunk.k_tiles for chunk in routed
                ),
                "shared_tasks": len(shared),
                "shared_tiles": sum(
                    chunk.k_tiles for chunk in shared
                ),
            }
        )
    return tuple(counters)


def _add_operator(
    operators: dict[str, dict[str, int]],
    name: str,
    *,
    count: int = 1,
    bytes: int = 0,
) -> None:
    entry = operators.setdefault(name, {"count": 0, "bytes": 0})
    entry["count"] += count
    entry["bytes"] += bytes


def _operator_counters(
    queues: tuple[tuple[TileChunk, ...], ...],
) -> tuple[dict[str, dict[str, int]], ...]:
    """Count every scheduled compute/memory operator by queue wave."""
    wave_operators = []
    for wave in range(max(len(queue) for queue in queues)):
        operators: dict[str, dict[str, int]] = {}
        for queue in queues:
            if len(queue) <= wave:
                continue
            chunk = queue[wave]
            if chunk.kind == "routed":
                batch = min(4, chunk.k_tiles)
                _add_operator(
                    operators,
                    f"C.NVFP4_UMMA_PIPELINE_FP32_K{chunk.k_tiles}",
                )
                _add_operator(
                    operators,
                    "M.ROUTED_TMA_ALPHA",
                    bytes=16,
                )
                _add_operator(
                    operators,
                    "M.TMA_OUTPUT_SCALE",
                    bytes=16,
                )
                _add_operator(
                    operators,
                    "M.TMA_NVFP4_ACTIVATION",
                    count=(chunk.k_tiles + batch - 1) // batch,
                    bytes=(
                        chunk.k_tiles * NVFP4_ACTIVATION_TILE_BYTES
                    ),
                )
                _add_operator(
                    operators,
                    "M.ROUTED_TMA_NVFP4_WEIGHT_BASE",
                    bytes=NVFP4_WEIGHT_TILE_BYTES,
                )
                _add_operator(
                    operators,
                    "M.TMA_NVFP4_WEIGHT_ADDRESS_REG",
                    count=chunk.k_tiles - 1,
                    bytes=(
                        (chunk.k_tiles - 1) * NVFP4_WEIGHT_TILE_BYTES
                    ),
                )
            else:
                _add_operator(
                    operators,
                    f"C.FP8_UMMA_SPLITK_PACK2_FP32_K{chunk.k_tiles}",
                )
                _add_operator(
                    operators,
                    "M.TMA_FP8_ACTIVATION",
                    count=(chunk.k_tiles + 7) // 8,
                    bytes=chunk.k_tiles * FP8_ACTIVATION_TILE_BYTES,
                )
                scale_tiles = chunk.k_tiles // SCALE_PACK
                _add_operator(
                    operators,
                    "M.TMA_FP8_WEIGHT_DATA_SCALE",
                    count=scale_tiles,
                    bytes=scale_tiles * FP8_WEIGHT_TILE_BYTES,
                )
                _add_operator(
                    operators,
                    "M.TMA_FP8_WEIGHT_DATA_ONLY",
                    count=scale_tiles,
                    bytes=scale_tiles * FP8_WEIGHT_DATA_BYTES,
                )
            _add_operator(
                operators,
                "M.TMA_REDUCE_ADD_FP32_M128",
                bytes=FP32_OUTPUT_TILE_BYTES,
            )
        wave_operators.append(operators)
    return tuple(wave_operators)


class SchedAllW1TileWorkers(Schedule):
    """Render an explicit queue of contiguous K partials on every worker."""

    def __init__(
        self,
        inputs,
        output_reduce,
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

    def _on_place(self):
        if self.num_sms != WORKERS or len(self.queues) != WORKERS:
            raise ValueError("tile-worker schedule requires exactly 152 SMs")
        output = getattr(self.output_reduce, "mat", None)
        if (
            getattr(self.output_reduce, "mode", None) != "reduce"
            or output is None
            or output.dtype != torch.float32
            or tuple(output.shape)
            != (ROUTED_EXPERTS + 1, INTERMEDIATE_SIZE)
        ):
            raise ValueError("tile workers require FP32 reduce [7,2048]")
        if (
            self.output_scale.dtype != torch.float32
            or self.output_scale.numel() != 4
            or not self.output_scale.is_contiguous()
        ):
            raise ValueError("tile workers require 16-byte FP32 output scale")

    def _routed_task(self, chunk: TileChunk):
        batch = min(4, chunk.k_tiles)
        instructions = [
            Nvfp4GemvUmmaPipelineFp32Sm100(
                chunk.k_tiles,
                activation_tiles_per_load=batch,
            ),
            RoutedTmaLoad1D(
                self.inputs.table.state,
                chunk.route_rank,
                self.inputs.alpha_field,
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
                            f"w1.m{chunk.m_tile}.k{chunk.k_start}"
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
        instructions.append(
            self.output_reduce.cord(
                chunk.route_rank, chunk.m_tile * 128
            )
        )
        return instructions

    def _shared_task(self, chunk: TileChunk):
        if chunk.k_start % SCALE_PACK or chunk.k_tiles % SCALE_PACK:
            raise ValueError("shared chunks must preserve FP8 scale-pack pairs")
        instructions = [
            Fp8GemvUmmaSplitKSm100(
                chunk.k_tiles,
                4,
                SCALE_PACK,
                1,
            )
        ]
        activation_tiles_per_chunk = 8
        for chunk_start in range(
            chunk.k_start,
            chunk.k_start + chunk.k_tiles,
            activation_tiles_per_chunk,
        ):
            chunk_stop = min(
                chunk.k_start + chunk.k_tiles,
                chunk_start + activation_tiles_per_chunk,
            )
            instructions.append(
                TmaLoad1D(
                    self.inputs.activation_fp8[
                        chunk_start:chunk_stop
                    ].reshape(-1)
                ).fixed_port(1)
            )
            for scale_start in range(
                chunk_start, chunk_stop, SCALE_PACK
            ):
                for k_tile in range(
                    scale_start, scale_start + SCALE_PACK
                ):
                    weight = self.inputs.weight_fp8[
                        chunk.m_tile, k_tile
                    ].reshape(-1)
                    if k_tile % SCALE_PACK:
                        weight = weight[:FP8_WEIGHT_DATA_BYTES]
                    instructions.append(TmaLoad1D(weight).fixed_port(0))
        instructions.append(
            self.output_reduce.cord(
                ROUTED_EXPERTS, chunk.m_tile * 128
            )
        )
        return instructions

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        instructions = []
        queue = self.queues[sm]
        for task_index, chunk in enumerate(queue):
            task = (
                self._routed_task(chunk)
                if chunk.kind == "routed"
                else self._shared_task(chunk)
            )
            if task_index + 1 == len(queue):
                task[-1].bar(self._bar("output"))
            instructions.extend(task)
            if self.profile_tasks:
                instructions.append(
                    ProfileEvent(TASK_PROFILE_EVENT_BASE + task_index)
                )
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


def _profile_times_us(launcher: Launcher, marker_sm: int):
    profile = launcher.profile.cpu().numpy()
    begin = min(int(value) for value in profile[:, 2])
    end = int(profile[marker_sm, 3])
    kernel_begin = min(int(value) for value in profile[:, 0])
    kernel_end = max(int(value) for value in profile[:, 1])
    return (end - begin) / 1.0e3, (kernel_end - kernel_begin) / 1.0e3


def _record_task_profiles(
    profile,
    queues: tuple[tuple[TileChunk, ...], ...],
    elapsed_samples: dict[tuple[int, str, int], list[float]],
    finish_samples: dict[tuple[int, str, int], list[float]],
) -> None:
    global_begin = min(int(value) for value in profile[:, 2])
    launch_finishes: dict[tuple[int, str, int], list[float]] = defaultdict(
        list
    )
    for sm, queue in enumerate(queues):
        previous = int(profile[sm, 2])
        for wave, chunk in enumerate(queue):
            event = TASK_PROFILE_EVENT_BASE + wave
            current = int(profile[sm, event])
            key = (wave, chunk.kind, chunk.k_tiles)
            elapsed_samples[key].append((current - previous) / 1.0e3)
            launch_finishes[key].append((current - global_begin) / 1.0e3)
            previous = current
    for key, values in launch_finishes.items():
        finish_samples[key].append(max(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--plan", choices=SCHEDULES, default="one_wave152"
    )
    parser.add_argument("--print-queues", action="store_true")
    parser.add_argument("--operator-breakdown", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be nonnegative and iterations positive")

    device = torch.device("cuda")
    device_sms = min(
        torch.cuda.get_device_properties(device).multi_processor_count,
        WORKERS,
    )
    if device_sms != WORKERS:
        parser.error(f"this schedule requires 152 SMs, got {device_sms}")
    queues = build_wave_compare_queues(args.plan)
    queue_loads = [sum(chunk.k_tiles for chunk in queue) for queue in queues]
    queue_tasks = [len(queue) for queue in queues]
    if any(load == 0 for load in queue_loads):
        parser.error(
            f"plan {args.plan} leaves one or more of the 152 workers idle"
        )
    histogram = {
        load: queue_loads.count(load) for load in sorted(set(queue_loads))
    }
    if histogram != EXPECTED_LOAD_HISTOGRAM:
        raise AssertionError(
            f"unexpected worker-load histogram: {histogram}"
        )
    wave_counters = _wave_counters(queues)
    expected_wave_tasks = (
        (152,) if args.plan == "one_wave152" else (152, 8)
    )
    if tuple(counter["tasks"] for counter in wave_counters) != (
        expected_wave_tasks
    ):
        raise AssertionError(
            f"unexpected wave task counts: {wave_counters}"
        )
    description = (
        "routed=all_K16 shared=m0-7_K12+K10+K10,m8-15_4xK8"
        if args.plan == "one_wave152"
        else "routed=all_K16 shared=all_K10+K8+K8+K6"
    )
    print(
        "DSV4_W1_TILE_PLAN "
        f"name={args.plan} workers={WORKERS} "
        f"description={description} "
        f"total_tiles={sum(queue_loads)} lower_bound=14 "
        f"max_queue={max(queue_loads)} "
        f"load_histogram={histogram} "
        f"task_count_min={min(queue_tasks)} "
        f"task_count_max={max(queue_tasks)}",
        flush=True,
    )
    for counter in wave_counters:
        print(
            "DSV4_W1_WAVE_COUNTER "
            f"name={args.plan} wave={counter['wave']} "
            f"tasks={counter['tasks']} tiles={counter['tiles']} "
            f"routed_tasks={counter['routed_tasks']} "
            f"routed_tiles={counter['routed_tiles']} "
            f"shared_tasks={counter['shared_tasks']} "
            f"shared_tiles={counter['shared_tiles']}",
            flush=True,
        )
    if args.operator_breakdown:
        for wave, operators in enumerate(_operator_counters(queues)):
            for operator_name in sorted(operators):
                counter = operators[operator_name]
                print(
                    "DSV4_W1_OPERATOR_COUNTER "
                    f"name={args.plan} wave={wave} "
                    f"operator={operator_name} "
                    f"count={counter['count']} "
                    f"bytes={counter['bytes']}",
                    flush=True,
                )
    if args.print_queues:
        for worker, queue in enumerate(queues):
            print(
                "DSV4_W1_TILE_QUEUE "
                f"worker={worker} load={queue_loads[worker]} "
                f"tasks={','.join(chunk.label for chunk in queue)}",
                flush=True,
            )

    inputs = _make_inputs(device, args.seed)
    launcher = Launcher(WORKERS, device=device)
    accumulator = torch.zeros(
        (ROUTED_EXPERTS + 1, INTERMEDIATE_SIZE),
        dtype=torch.float32,
        device=device,
    )
    output_reduce = TmaTensor(
        launcher, accumulator
    ).rowmajor_2d("reduce", 1, 128)
    output_scale = torch.ones((4,), dtype=torch.float32, device=device)
    schedule = SchedAllW1TileWorkers(
        inputs,
        output_reduce,
        output_scale,
        queues,
        profile_tasks=args.operator_breakdown,
    )
    marker_operand = torch.zeros((16,), dtype=torch.uint8, device=device)
    marker_sm = WORKERS - 1
    program = SequentialProgram(
        launcher,
        (
            SequentialStage(
                "all_w1_tiles",
                schedule,
                WORKERS,
                release_group="all_w1_tiles_done",
            ),
            SequentialStage(
                "profile.all_w1_tiles",
                _BarrierProfileSchedule(3, marker_operand),
                1,
                base_sm=marker_sm,
                wait_group="all_w1_tiles_done",
            ),
        ),
    )
    launcher.s(ProfileEvent(2), program)
    launcher._w1_tile_owners = (
        inputs,
        accumulator,
        output_scale,
        marker_operand,
        program,
    )

    def reset_and_launch():
        accumulator.zero_()
        launcher.launch()

    reset_and_launch()
    if not bool(torch.isfinite(accumulator).all().item()):
        raise AssertionError("tile-worker output contains non-finite values")
    for _ in range(args.warmup):
        reset_and_launch()
    frontier_samples = []
    kernel_samples = []
    task_elapsed_samples: dict[
        tuple[int, str, int], list[float]
    ] = defaultdict(list)
    task_finish_samples: dict[
        tuple[int, str, int], list[float]
    ] = defaultdict(list)
    for _ in range(args.iterations):
        reset_and_launch()
        frontier, kernel = _profile_times_us(launcher, marker_sm)
        frontier_samples.append(frontier)
        kernel_samples.append(kernel)
        if args.operator_breakdown:
            _record_task_profiles(
                launcher.profile.cpu().numpy(),
                queues,
                task_elapsed_samples,
                task_finish_samples,
            )
    ordered = sorted(frontier_samples)
    print(
        "DSV4_W1_TILE_RESULT "
        f"name={args.plan} "
        f"frontier_median_us={statistics.median(frontier_samples):.6f} "
        f"frontier_p10_us={ordered[max(0, len(ordered) // 10 - 1)]:.6f} "
        f"frontier_min_us={min(frontier_samples):.6f} "
        f"kernel_median_us={statistics.median(kernel_samples):.6f}",
        flush=True,
    )
    if args.operator_breakdown:
        for key in sorted(task_elapsed_samples):
            wave, kind, k_tiles = key
            elapsed = task_elapsed_samples[key]
            finishes = task_finish_samples[key]
            print(
                "DSV4_W1_COMPUTE_OPERATOR_PROFILE "
                f"name={args.plan} wave={wave} kind={kind} "
                f"k_tiles={k_tiles} samples={len(elapsed)} "
                f"task_median_us={statistics.median(elapsed):.6f} "
                f"class_finish_median_us="
                f"{statistics.median(finishes):.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
