"""Flatten placed schedules into one dependency-ordered vdcores launch."""

from __future__ import annotations

from dataclasses import dataclass

from .instructions import (
    ComputeInstruction,
    Fp8GemvUmmaCoupledSm100,
    LduProfileLayer,
    LduReloadBarriers,
    LoopC,
    LoopM,
    MemoryInstruction,
    ProfileAggregate,
    ProfileStep,
    ResetIndirectLayer,
    TmaLoadMxfpCoupledStream,
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
    wait_group: str | None = None
    release_group: str | None = None
    profile_step_event: int | None = None
    profile_aggregate_events: tuple[int, int] | None = None
    profile_span_begin: tuple[int, int] | None = None
    profile_span_end: tuple[int, int] | None = None
    prefetch_before_wait: bool = False
    wait_group_roles: tuple[tuple[str, str], ...] = ()
    release_group_roles: tuple[tuple[str, str], ...] = ()


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
        if not any(
            not isinstance(inst, ProfileAggregate) for inst in instructions
        ):
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
        if not any(
            not isinstance(inst, ProfileAggregate) for inst in instructions
        ):
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


def _validate_prefetch_gate(per_sm: list[list], bar_id: int, stage: str) -> None:
    """Require an explicit LDU dependency when earlier loads may prefetch."""

    active = False
    for instructions in per_sm:
        if not any(
            not isinstance(inst, ProfileAggregate) for inst in instructions
        ):
            continue
        active = True
        gated_loads = [
            inst
            for inst in instructions
            if isinstance(inst, MemoryInstruction)
            and inst.opcode & _MEM_ALLOCATE
            and not inst.opcode & _MEM_WRITEBACK
            and _bar_id(inst) == bar_id
        ]
        if not gated_loads:
            raise ValueError(
                f"prefetching stage {stage!r} has no explicitly gated LDU load"
            )
    if not active:
        raise ValueError(f"sequential stage {stage!r} has no active SMs")


def _validate_explicit_role_bar(
    per_sm: list[list], bar_id: int, stage: str, role: str
) -> None:
    if any(
        isinstance(inst, MemoryInstruction) and _bar_id(inst) == bar_id
        for instructions in per_sm
        for inst in instructions
    ):
        return
    raise ValueError(
        f"sequential stage {stage!r} role {role!r} emitted no "
        "memory command for its dependency barrier"
    )


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


def _rebase_coupled_fp8_phases(
    instructions: list, initial_phase: int
) -> int:
    """Carry the persistent two-stage ring phase across schedule boundaries."""

    computes = [
        inst
        for inst in instructions
        if isinstance(inst, Fp8GemvUmmaCoupledSm100)
    ]
    loads = [
        inst
        for inst in instructions
        if isinstance(inst, MemoryInstruction)
        and inst.annotation.get("coupled_stream_kind")
        == TmaLoadMxfpCoupledStream.FP8_GEMV
        and inst.annotation.get("coupled_stream_allocator_lease")
    ]
    if len(computes) != len(loads):
        raise ValueError(
            "coupled FP8 compute/load command counts do not match"
        )

    phase = int(initial_phase) % (2 * TmaLoadMxfpCoupledStream.FP8_STAGES)
    local_phase = 0
    phase_mask = (
        TmaLoadMxfpCoupledStream.MAX_PHASE_BASE
        << TmaLoadMxfpCoupledStream.PHASE_BASE_SHIFT
    )
    for compute, load in zip(computes, loads):
        pair_count = int(compute.args[0])
        encoded_phase = (
            load.arg >> TmaLoadMxfpCoupledStream.PHASE_BASE_SHIFT
        ) & TmaLoadMxfpCoupledStream.MAX_PHASE_BASE
        if (
            pair_count != int(load.size)
            or int(compute.args[2]) != encoded_phase
            or encoded_phase
            % (2 * TmaLoadMxfpCoupledStream.FP8_STAGES)
            != local_phase
        ):
            raise ValueError(
                "coupled FP8 stage has inconsistent local phase progression"
            )
        compute.args[2] = phase
        load.arg = (
            (load.arg & ~phase_mask)
            | (phase << TmaLoadMxfpCoupledStream.PHASE_BASE_SHIFT)
        )
        phase = (
            phase + pair_count
        ) % (2 * TmaLoadMxfpCoupledStream.FP8_STAGES)
        local_phase = (
            local_phase + pair_count
        ) % (2 * TmaLoadMxfpCoupledStream.FP8_STAGES)
    return phase


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
        profile_special_slot: int = 0,
        balance_load_ports: bool = False,
        coupled_fp8_initial_phases: list[int] | tuple[int, ...] | None = None,
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
        step_events = [
            stage.profile_step_event
            for stage in self.stages
            if stage.profile_step_event is not None
        ]
        if len(step_events) != len(set(step_events)):
            raise ValueError("step profile events must be unique")
        if any(
            event < config.layer_profile_event_base
            or event >= config.reload_profile_event_base
            for event in step_events
        ):
            raise ValueError("step profile events exceed the layer-profile range")
        aggregate_events = [
            event
            for stage in self.stages
            for pair in (
                stage.profile_aggregate_events,
                stage.profile_span_begin,
                stage.profile_span_end,
            )
            if pair is not None
            for event in pair
        ]
        if any(
            event < config.layer_profile_event_base
            or event >= config.reload_profile_event_base
            for event in aggregate_events
        ):
            raise ValueError(
                "aggregate profile events exceed the layer-profile range"
            )

        self.instructions = [[] for _ in range(launcher.num_sms)]
        self.barriers = []
        self.barrier_start = launcher.num_bars
        self.completion_barrier = None
        self.stage_stats = []
        # Placement may materialize device-side routing/index tables or padded
        # scalar storage referenced by encoded memory instructions.  Retain
        # every placed schedule for at least as long as the launch program.
        self.placed_schedules = []
        if coupled_fp8_initial_phases is None:
            coupled_fp8_initial_phases = (0,) * launcher.num_sms
        if len(coupled_fp8_initial_phases) != launcher.num_sms:
            raise ValueError(
                "coupled FP8 phase state must cover every resident SM"
            )
        self.coupled_fp8_initial_phases = tuple(
            int(phase) % (2 * TmaLoadMxfpCoupledStream.FP8_STAGES)
            for phase in coupled_fp8_initial_phases
        )
        coupled_fp8_phases = list(self.coupled_fp8_initial_phases)

        release_groups = []
        for stage in self.stages:
            stage_groups = []
            if stage.release_group is not None:
                stage_groups.append(stage.release_group)
            stage_groups.extend(
                group for group, _ in stage.release_group_roles
            )
            for group in stage_groups:
                if group not in release_groups:
                    release_groups.append(group)
        wait_groups = set()
        for stage in self.stages:
            if stage.wait_group is not None:
                wait_groups.add(stage.wait_group)
            wait_groups.update(
                group for group, _ in stage.wait_group_roles
            )
        missing_groups = wait_groups.difference(release_groups)
        if missing_groups:
            raise ValueError(
                "sequential stage waits on groups with no producers: "
                f"{sorted(missing_groups)}"
            )
        group_barriers = {}
        group_release_counts = {group: 0 for group in release_groups}
        for group in release_groups:
            if launcher.num_bars >= config.max_bars - 2:
                raise ValueError(
                    "sequential program exceeds the runtime barrier capacity"
                )
            group_barriers[group] = launcher.new_bar(None)
            self.barriers.append(group_barriers[group])

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
            if stage.prefetch_before_wait and stage.input_role is None:
                raise ValueError(
                    f"prefetching stage {stage.name!r} requires an input_role"
                )
            if stage.wait_group is not None and stage.wait_group_roles:
                raise ValueError(
                    f"stage {stage.name!r} cannot mix wait_group with "
                    "wait_group_roles"
                )
            role_groups = {}
            for group, role in (
                *stage.wait_group_roles,
                *stage.release_group_roles,
            ):
                previous_group = role_groups.setdefault(role, group)
                if previous_group != group:
                    raise ValueError(
                        f"stage {stage.name!r} binds role {role!r} to "
                        "multiple dependency groups"
                    )

            input_bar = None
            if stage.wait_group is not None:
                input_bar = group_barriers[stage.wait_group]
            elif stage.wait_group_roles:
                # The schedule binds each consuming memory command to its
                # own named edge below; no stage-wide LDU gate is needed.
                pass
            elif previous is not None and stage.wait_for_previous:
                count, tails = _writeback_tail(previous, previous_name)
                if launcher.num_bars >= config.max_bars - 2:
                    raise ValueError("sequential program exceeds the runtime barrier capacity")
                input_bar = launcher.new_bar(count)
                self.barriers.append(input_bar)
                for tail in tails:
                    _attach_bar(tail, input_bar, stage=previous_name)
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

            if previous_profile_after:
                profile_input_bar = input_bar
                if profile_input_bar is None and stage.wait_group_roles:
                    profile_bars = {
                        group_barriers[group]
                        for group, _ in stage.wait_group_roles
                    }
                    if len(profile_bars) == 1:
                        profile_input_bar = next(iter(profile_bars))
                if profile_input_bar is None:
                    raise ValueError(
                        "a profiled stage requires a following dependency"
                    )
                if self.profile_event_count == 0:
                    raise ValueError("profiled stage requires profile event capacity")
                marker = LduProfileLayer(
                    config.layer_profile_event_base,
                    self.profile_event_count,
                    special_slot=profile_special_slot,
                ).bar(profile_input_bar)
                for instructions in self.instructions:
                    instructions.append(marker.copy())

            schedule = stage.schedule._clone()
            if input_bar is not None and stage.input_role is not None:
                schedule.bar(stage.input_role, input_bar)
            for group, role in stage.wait_group_roles:
                schedule.bar(role, group_barriers[group])
            for group, role in stage.release_group_roles:
                schedule.bar(role, group_barriers[group])
            placed = schedule.place(stage.num_sms, stage.base_sm)
            self.placed_schedules.append(placed)
            rendered = []
            for sm in range(launcher.num_sms):
                instructions = []
                _flatten(placed(sm), sm, instructions)
                if (
                    stage.profile_step_event is not None
                    and any(
                        isinstance(inst, ComputeInstruction)
                        for inst in instructions
                    )
                ):
                    instructions.insert(
                        0,
                        ProfileStep(stage.profile_step_event, begin=True),
                    )
                    instructions.append(
                        ProfileStep(stage.profile_step_event, begin=False)
                    )
                if (
                    stage.profile_aggregate_events is not None
                    and any(
                        isinstance(inst, ComputeInstruction)
                        for inst in instructions
                    )
                ):
                    begin_event, aggregate_event = (
                        stage.profile_aggregate_events
                    )
                    instructions.insert(
                        0,
                        ProfileAggregate(
                            begin_event, aggregate_event, begin=True
                        ),
                    )
                    instructions.append(
                        ProfileAggregate(
                            begin_event, aggregate_event, begin=False
                        )
                    )
                if stage.profile_span_begin is not None:
                    begin_event, aggregate_event = stage.profile_span_begin
                    instructions.insert(
                        0,
                        ProfileAggregate(
                            begin_event, aggregate_event, begin=True
                        ),
                    )
                if stage.profile_span_end is not None:
                    begin_event, aggregate_event = stage.profile_span_end
                    instructions.append(
                        ProfileAggregate(
                            begin_event, aggregate_event, begin=False
                        )
                    )
                rendered.append(instructions)
            if balance_load_ports:
                _balance_load_ports(rendered)
            for sm, instructions in enumerate(rendered):
                coupled_fp8_phases[sm] = _rebase_coupled_fp8_phases(
                    instructions, coupled_fp8_phases[sm]
                )
            if input_bar is not None:
                if stage.prefetch_before_wait:
                    _validate_prefetch_gate(rendered, input_bar, stage.name)
                else:
                    _gate_load_ports(rendered, input_bar, stage.name)
            for group, role in stage.wait_group_roles:
                _validate_explicit_role_bar(
                    rendered,
                    group_barriers[group],
                    stage.name,
                    role,
                )

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
            if stage.release_group is not None:
                count, tails = _writeback_tail(rendered, stage.name)
                release_bar = group_barriers[stage.release_group]
                for tail in tails:
                    _attach_bar(tail, release_bar, stage=stage.name)
                group_release_counts[stage.release_group] += count
            for group, role in stage.release_group_roles:
                release_bar = group_barriers[group]
                _validate_explicit_role_bar(
                    rendered, release_bar, stage.name, role
                )
                count = placed.bar_release_count(role)
                if count <= 0:
                    raise ValueError(
                        f"sequential stage {stage.name!r} role {role!r} "
                        "has no barrier release count"
                    )
                group_release_counts[group] += count
            previous = rendered
            previous_stage = stage
            previous_name = stage.name
            previous_profile_after = stage.profile_after

        for group, bar_id in group_barriers.items():
            count = group_release_counts[group]
            if count <= 0:
                raise ValueError(f"release group {group!r} has no active producers")
            launcher.set_bar(bar_id, count)

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
                    config.layer_profile_event_base,
                    self.profile_event_count,
                    special_slot=profile_special_slot,
                ).bar(self.completion_barrier)
                for instructions in self.instructions:
                    instructions.append(marker.copy())
        elif previous_profile_after:
            raise ValueError(
                "a profiled final stage requires a completion barrier"
            )
        self.barrier_stop = launcher.num_bars
        self.coupled_fp8_final_phases = tuple(coupled_fp8_phases)

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
        coupled_fp8_phases = [0] * launcher.num_sms
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
                coupled_fp8_initial_phases=coupled_fp8_phases,
            )
            if block.repeat > 1:
                incompatible_sms = [
                    sm
                    for sm, (initial, final) in enumerate(zip(
                        segment.coupled_fp8_initial_phases,
                        segment.coupled_fp8_final_phases,
                    ))
                    if initial != final
                ]
                if incompatible_sms:
                    raise ValueError(
                        "repeated coupled FP8 block does not return its "
                        "persistent ring phase to the entry state on SMs "
                        f"{incompatible_sms}"
                    )
            coupled_fp8_phases = list(segment.coupled_fp8_final_phases)
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
        self.coupled_fp8_final_phases = tuple(coupled_fp8_phases)

    def __call__(self, sm: int):
        return self.instructions[sm]


__all__ = [
    "LoopedSequentialProgram",
    "SequentialBlock",
    "SequentialProgram",
    "SequentialStage",
]
