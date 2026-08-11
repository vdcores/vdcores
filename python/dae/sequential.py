"""Flatten placed schedules into one dependency-ordered vdcores launch."""

from __future__ import annotations

from dataclasses import dataclass

from .instructions import (
    ComputeInstruction,
    LduProfileLayer,
    LduReloadBarriers,
    LoopC,
    LoopM,
    MemoryInstruction,
    ResetIndirectLayer,
)
from .runtime import config
from .schedule import Schedule


_MEM_ALLOCATE = 0x1
_MEM_WRITEBACK = 0x2
_MEM_BARRIER = 0x10
_MEM_PORT1 = 0x20


@dataclass(frozen=True)
class SequentialStage:
    """One independently placed schedule in a sequential launch program."""

    name: str
    schedule: Schedule
    num_sms: int
    base_sm: int = 0
    input_role: str | None = None
    profile_after: bool = False
    wait_for_previous: bool = True


@dataclass(frozen=True)
class SequentialBlock:
    """A queue body that may repeat before advancing to the next body."""

    name: str
    stages: tuple[SequentialStage, ...] | list[SequentialStage]
    repeat: int = 1
    reload_after: bool = True
    barrier_banks: int = 1


def _flatten(item, sm: int, output: list) -> None:
    if item is None:
        return
    if isinstance(item, (ComputeInstruction, MemoryInstruction)):
        output.append(item.copy() if isinstance(item, MemoryInstruction) else item)
        return
    if isinstance(item, (list, tuple)):
        for child in item:
            _flatten(child, sm, output)
        return
    if hasattr(item, "expand_instructions"):
        _flatten(item.expand_instructions(), sm, output)
        return
    if callable(item):
        _flatten(item(sm), sm, output)
        return
    raise TypeError(f"unsupported sequential instruction item {type(item).__name__}")


def _bar_id(inst: MemoryInstruction) -> int | None:
    if not inst.opcode & _MEM_BARRIER:
        return None
    return inst.num_slots >> 6


def _attach_bar(inst: MemoryInstruction, bar_id: int, *, stage: str) -> None:
    existing = _bar_id(inst)
    if existing is not None and existing != bar_id:
        raise ValueError(
            f"stage {stage!r} instruction already uses barrier {existing}; "
            f"cannot also attach sequential barrier {bar_id}"
        )
    if existing is None:
        inst.bar(bar_id)


def _writeback_tail(per_sm: list[list], stage: str) -> tuple[int, list[MemoryInstruction]]:
    tails = []
    for instructions in per_sm:
        if not instructions:
            continue
        writebacks = [
            inst
            for inst in instructions
            if isinstance(inst, MemoryInstruction) and inst.opcode & _MEM_WRITEBACK
        ]
        if not writebacks:
            raise ValueError(
                f"sequential stage {stage!r} has work but no writeback boundary"
            )
        tails.append(writebacks[-1])
    if not tails:
        raise ValueError(f"sequential stage {stage!r} has no active SMs")
    return len(tails), tails


def _gate_load_ports(per_sm: list[list], bar_id: int, stage: str) -> None:
    active = False
    for instructions in per_sm:
        if not instructions:
            continue
        active = True
        first_load_by_port = {}
        for inst in instructions:
            if not isinstance(inst, MemoryInstruction):
                continue
            if not inst.opcode & _MEM_ALLOCATE or inst.opcode & _MEM_WRITEBACK:
                continue
            port = 1 if inst.opcode & _MEM_PORT1 else 0
            first_load_by_port.setdefault(port, inst)
        if not first_load_by_port:
            raise ValueError(
                f"sequential stage {stage!r} has work but no allocating load boundary"
            )
        for inst in first_load_by_port.values():
            _attach_bar(inst, bar_id, stage=stage)
    if not active:
        raise ValueError(f"sequential stage {stage!r} has no active SMs")


def _balance_load_ports(per_sm: list[list]) -> None:
    """Distribute stage operands over both LDU FIFOs by encoded byte size."""

    for instructions in per_sm:
        port_bytes = [0, 0]
        for inst in instructions:
            if not isinstance(inst, MemoryInstruction):
                continue
            if not inst.opcode & _MEM_ALLOCATE or inst.opcode & _MEM_WRITEBACK:
                continue
            fixed_port = inst.annotation.get("fixed_port")
            if fixed_port is not None:
                if fixed_port not in (0, 1):
                    raise ValueError(f"invalid fixed LDU port {fixed_port!r}")
                encoded_port = 1 if inst.opcode & _MEM_PORT1 else 0
                if encoded_port != fixed_port:
                    raise ValueError(
                        "fixed LDU port annotation disagrees with encoded port"
                    )
                port = fixed_port
            elif inst.opcode & _MEM_PORT1:
                port = 1
            else:
                port = 1 if port_bytes[1] < port_bytes[0] else 0
                if port == 1:
                    inst.port(1)
            port_bytes[port] += max(1, inst.size)


class SequentialProgram:
    """Render a strict stage chain into per-SM compute/memory queues.

    Every edge is released by the producer's final STU writeback on each
    active SM.  The first LDU allocation on every memory port waits on that
    edge, which preserves ordering even when a stage uses both load ports.
    """

    def __init__(
        self,
        launcher,
        stages: list[SequentialStage] | tuple[SequentialStage, ...],
        *,
        completion_barrier: bool = False,
        profile_event_count: int | None = None,
        balance_load_ports: bool = False,
    ):
        self.launcher = launcher
        self.stages = tuple(stages)
        if not self.stages:
            raise ValueError("sequential program requires at least one stage")
        if profile_event_count is None:
            profile_event_count = sum(stage.profile_after for stage in self.stages)
        self.profile_event_count = int(profile_event_count)
        layer_profile_capacity = (
            config.reload_profile_event_base - config.layer_profile_event_base
        )
        if not 0 <= self.profile_event_count <= layer_profile_capacity:
            raise ValueError("layer profile events exceed the runtime profile row")

        self.instructions = [[] for _ in range(launcher.num_sms)]
        self.barriers = []
        self.barrier_start = launcher.num_bars
        self.completion_barrier = None
        self.stage_stats = []
        # Placement may materialize device-side routing/index tables or padded
        # scalar storage referenced by encoded memory instructions.  Retain
        # every placed schedule for at least as long as the launch program.
        self.placed_schedules = []

        previous = None
        previous_stage = None
        previous_name = None
        previous_profile_after = False
        for stage in self.stages:
            if stage.num_sms <= 0:
                raise ValueError(f"stage {stage.name!r} must use at least one SM")
            if stage.base_sm < 0 or stage.base_sm + stage.num_sms > launcher.num_sms:
                raise ValueError(
                    f"stage {stage.name!r} placement [{stage.base_sm}, "
                    f"{stage.base_sm + stage.num_sms}) exceeds {launcher.num_sms} SMs"
                )

            input_bar = None
            if previous is not None and stage.wait_for_previous:
                count, tails = _writeback_tail(previous, previous_name)
                if launcher.num_bars >= config.max_bars - 2:
                    raise ValueError("sequential program exceeds the runtime barrier capacity")
                input_bar = launcher.new_bar(count)
                self.barriers.append(input_bar)
                for tail in tails:
                    _attach_bar(tail, input_bar, stage=previous_name)
                if previous_profile_after:
                    if self.profile_event_count == 0:
                        raise ValueError("profiled stage requires profile event capacity")
                    marker = LduProfileLayer(
                        config.layer_profile_event_base, self.profile_event_count
                    ).bar(input_bar)
                    for instructions in self.instructions:
                        instructions.append(marker.copy())
            elif previous is not None:
                if previous_profile_after:
                    raise ValueError(
                        "a profiled stage cannot elide its completion edge"
                    )
                if (
                    stage.base_sm != previous_stage.base_sm
                    or stage.num_sms != previous_stage.num_sms
                ):
                    raise ValueError(
                        f"independent stage {stage.name!r} must match the "
                        "previous stage placement so its queue tail dominates"
                    )

            schedule = stage.schedule._clone()
            if input_bar is not None and stage.input_role is not None:
                schedule.bar(stage.input_role, input_bar)
            placed = schedule.place(stage.num_sms, stage.base_sm)
            self.placed_schedules.append(placed)
            rendered = []
            for sm in range(launcher.num_sms):
                instructions = []
                _flatten(placed(sm), sm, instructions)
                rendered.append(instructions)
            if balance_load_ports:
                _balance_load_ports(rendered)
            if input_bar is not None:
                _gate_load_ports(rendered, input_bar, stage.name)

            max_compute = max(
                sum(isinstance(inst, ComputeInstruction) for inst in instructions)
                for instructions in rendered
            )
            max_memory = max(
                sum(isinstance(inst, MemoryInstruction) for inst in instructions)
                for instructions in rendered
            )
            self.stage_stats.append((stage.name, max_compute, max_memory))
            for sm, instructions in enumerate(rendered):
                self.instructions[sm].extend(instructions)
            previous = rendered
            previous_stage = stage
            previous_name = stage.name
            previous_profile_after = stage.profile_after

        if completion_barrier:
            count, tails = _writeback_tail(previous, previous_name)
            if launcher.num_bars >= config.max_bars - 2:
                raise ValueError("sequential program exceeds the runtime barrier capacity")
            self.completion_barrier = launcher.new_bar(count)
            self.barriers.append(self.completion_barrier)
            for tail in tails:
                _attach_bar(tail, self.completion_barrier, stage=previous_name)
            if previous_profile_after:
                if self.profile_event_count == 0:
                    raise ValueError("profiled stage requires profile event capacity")
                marker = LduProfileLayer(
                    config.layer_profile_event_base, self.profile_event_count
                ).bar(self.completion_barrier)
                for instructions in self.instructions:
                    instructions.append(marker.copy())
        elif previous_profile_after:
            raise ValueError(
                "a profiled final stage requires a completion barrier"
            )
        self.barrier_stop = launcher.num_bars

        max_compute = max(
            sum(isinstance(inst, ComputeInstruction) for inst in instructions)
            for instructions in self.instructions
        )
        max_memory = max(
            sum(isinstance(inst, MemoryInstruction) for inst in instructions)
            for instructions in self.instructions
        )
        # Launcher.s() appends one terminator to each stream.
        if max_compute + 1 > launcher.max_insts or max_memory + 1 > launcher.max_insts:
            raise ValueError(
                "sequential program exceeds the resident instruction image: "
                f"compute={max_compute + 1}/{launcher.max_insts}, "
                f"memory={max_memory + 1}/{launcher.max_insts}"
            )
        self.max_compute_instructions = max_compute + 1
        self.max_memory_instructions = max_memory + 1

    def __call__(self, sm: int):
        return self.instructions[sm]


class LoopedSequentialProgram:
    """Compose compact nested loops with shifted dependency-barrier banks."""

    def __init__(
        self,
        launcher,
        blocks: list[SequentialBlock] | tuple[SequentialBlock, ...],
        *,
        balance_load_ports: bool = False,
    ):
        self.launcher = launcher
        self.blocks = tuple(blocks)
        if not self.blocks:
            raise ValueError("looped sequential program requires at least one block")

        self.instructions = [[] for _ in range(launcher.num_sms)]
        self.barriers = []
        self.stage_stats = []
        self.segments = []
        self.placed_schedules = []
        self.profile_event_count = sum(
            sum(stage.profile_after for stage in block.stages) * block.repeat
            for block in self.blocks
        )
        layer_profile_capacity = (
            config.reload_profile_event_base - config.layer_profile_event_base
        )
        if self.profile_event_count > layer_profile_capacity:
            raise ValueError("layer profile events exceed the runtime profile row")
        compute_base = launcher.copy_cptrs()
        memory_base = launcher.copy_mptrs()

        for block in self.blocks:
            if not 1 <= block.repeat <= 0xFFFF:
                raise ValueError(f"block {block.name!r} repeat must fit in uint16")
            if block.repeat > 1 and not block.reload_after:
                raise ValueError(
                    f"repeated block {block.name!r} must reload its dependencies"
                )
            if block.barrier_banks <= 0:
                raise ValueError(f"block {block.name!r} barrier_banks must be positive")
            bank_count = min(block.barrier_banks, block.repeat)
            while block.repeat % bank_count:
                bank_count -= 1
            outer_count = block.repeat // bank_count
            required_counters = int(bank_count > 1) + int(outer_count > 1)
            if required_counters > config.num_loop_counters:
                raise ValueError("looped program exceeds runtime loop-counter capacity")
            if block.repeat > 1:
                for sm in range(launcher.num_sms):
                    self.instructions[sm].append(ResetIndirectLayer())
            compute_start = [
                (compute_base[sm] + sum(
                    isinstance(inst, ComputeInstruction)
                    for inst in self.instructions[sm]
                )) % launcher.max_insts
                for sm in range(launcher.num_sms)
            ]
            memory_start = [
                (memory_base[sm] + sum(
                    isinstance(inst, MemoryInstruction)
                    for inst in self.instructions[sm]
                )) % launcher.max_insts
                for sm in range(launcher.num_sms)
            ]

            segment = SequentialProgram(
                launcher,
                block.stages,
                completion_barrier=block.reload_after,
                profile_event_count=self.profile_event_count,
                balance_load_ports=balance_load_ports,
            )
            self.segments.append(segment)
            barriers_per_bank = segment.barrier_stop - segment.barrier_start
            if bank_count > 1:
                if not block.reload_after or barriers_per_bank <= 0:
                    raise ValueError(
                        f"block {block.name!r} needs a completion barrier for shifting"
                    )
                base_values = [
                    launcher.bar_values[bar_id]
                    for bar_id in range(segment.barrier_start, segment.barrier_stop)
                ]
                for _ in range(1, bank_count):
                    for value in base_values:
                        if launcher.num_bars >= config.max_bars - 2:
                            raise ValueError(
                                "looped program exceeds shifted barrier capacity"
                            )
                        launcher.new_bar(value)
                for instructions in segment.instructions:
                    for inst in instructions:
                        if not isinstance(inst, MemoryInstruction):
                            continue
                        if not inst.opcode & _MEM_BARRIER:
                            continue
                        bar_id = _bar_id(inst)
                        if segment.barrier_start <= bar_id < segment.barrier_stop:
                            inst.group()
            self.barriers.extend(
                range(
                    segment.barrier_start,
                    segment.barrier_start + barriers_per_bank * bank_count,
                )
            )
            self.stage_stats.extend(segment.stage_stats)
            self.placed_schedules.extend(segment.placed_schedules)
            for sm in range(launcher.num_sms):
                self.instructions[sm].extend(segment.instructions[sm])

            inner_reg = 0
            outer_reg = int(bank_count > 1)
            if block.reload_after:
                # Keep the reload in the repeated memory body. It waits for
                # the current bank's final STU completion and sits ahead of
                # every following LDU command in both port FIFOs, providing
                # the loop-carried dependency without an IssueBarrier.
                reload = LduReloadBarriers(
                    launcher.bars_src,
                    segment.barrier_start,
                    barriers_per_bank,
                    2 if self.profile_event_count else 0,
                ).bar(segment.completion_barrier)
                if bank_count > 1:
                    reload.group()
                for sm in range(launcher.num_sms):
                    self.instructions[sm].append(reload.copy())
            if bank_count > 1:
                for sm in range(launcher.num_sms):
                    self.instructions[sm].extend(
                        (
                            LoopC(bank_count, compute_start[sm], reg=inner_reg),
                            LoopM(
                                bank_count,
                                memory_start[sm],
                                reg=inner_reg,
                                bar_shift=barriers_per_bank,
                                advance_indirect_layer=True,
                            ),
                        )
                    )
            if outer_count > 1:
                for sm in range(launcher.num_sms):
                    self.instructions[sm].extend(
                        (
                            LoopC(outer_count, compute_start[sm], reg=outer_reg),
                            LoopM(
                                outer_count,
                                memory_start[sm],
                                reg=outer_reg,
                                advance_indirect_layer=bank_count == 1,
                            ),
                        )
                    )

        max_compute = max(
            sum(isinstance(inst, ComputeInstruction) for inst in instructions)
            for instructions in self.instructions
        )
        max_memory = max(
            sum(isinstance(inst, MemoryInstruction) for inst in instructions)
            for instructions in self.instructions
        )
        if max_compute + 1 > launcher.max_insts or max_memory + 1 > launcher.max_insts:
            raise ValueError(
                "looped sequential program exceeds the instruction queue: "
                f"compute={max_compute + 1}/{launcher.max_insts}, "
                f"memory={max_memory + 1}/{launcher.max_insts}"
            )
        self.max_compute_instructions = max_compute + 1
        self.max_memory_instructions = max_memory + 1

    def __call__(self, sm: int):
        return self.instructions[sm]


__all__ = [
    "LoopedSequentialProgram",
    "SequentialBlock",
    "SequentialProgram",
    "SequentialStage",
]
