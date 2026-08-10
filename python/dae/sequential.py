"""Flatten placed schedules into one dependency-ordered vdcores launch."""

from __future__ import annotations

from dataclasses import dataclass

from .instructions import ComputeInstruction, MemoryInstruction
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


class SequentialProgram:
    """Render a strict stage chain into per-SM compute/memory queues.

    Every edge is released by the producer's final STU writeback on each
    active SM.  The first LDU allocation on every memory port waits on that
    edge, which preserves ordering even when a stage uses both load ports.
    """

    def __init__(self, launcher, stages: list[SequentialStage] | tuple[SequentialStage, ...]):
        self.launcher = launcher
        self.stages = tuple(stages)
        if not self.stages:
            raise ValueError("sequential program requires at least one stage")

        self.instructions = [[] for _ in range(launcher.num_sms)]
        self.barriers = []
        self.stage_stats = []
        # Placement may materialize device-side routing/index tables or padded
        # scalar storage referenced by encoded memory instructions.  Retain
        # every placed schedule for at least as long as the launch program.
        self.placed_schedules = []

        previous = None
        previous_name = None
        for stage in self.stages:
            if stage.num_sms <= 0:
                raise ValueError(f"stage {stage.name!r} must use at least one SM")
            if stage.base_sm < 0 or stage.base_sm + stage.num_sms > launcher.num_sms:
                raise ValueError(
                    f"stage {stage.name!r} placement [{stage.base_sm}, "
                    f"{stage.base_sm + stage.num_sms}) exceeds {launcher.num_sms} SMs"
                )

            input_bar = None
            if previous is not None:
                count, tails = _writeback_tail(previous, previous_name)
                if launcher.num_bars >= config.max_bars:
                    raise ValueError("sequential program exceeds the runtime barrier capacity")
                input_bar = launcher.new_bar(count)
                self.barriers.append(input_bar)
                for tail in tails:
                    _attach_bar(tail, input_bar, stage=previous_name)

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
            previous_name = stage.name

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


__all__ = ["SequentialProgram", "SequentialStage"]
