"""Flatten placed schedules into one dependency-ordered vdcores launch."""

from __future__ import annotations

from dataclasses import dataclass

from .instructions import (
    ComputeInstruction,
    Fp8GemvUmmaCoupledSm100,
    LduAsyncReloadBarriers,
    LduReloadBarriers,
    LduWaitBarrier,
    LoopC,
    LoopM,
    MemoryInstruction,
    ProfileAggregate,
    ProfileEvent,
    ProfileLayer,
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
    parallel_with_previous: bool = False
    wait_group: str | None = None
    release_group: str | None = None
    profile_step_event: int | None = None
    profile_step_begin_event: int | None = None
    profile_aggregate_events: tuple[int, int] | None = None
    profile_span_begin: tuple[int, int] | None = None
    profile_span_end: tuple[int, int] | None = None
    prefetch_before_wait: bool = False
    prefetch_before_resident_reset: bool = False
    wait_group_roles: tuple[tuple[str, str], ...] = ()
    release_group_roles: tuple[tuple[str, str], ...] = ()
    reset_mxfp_resident_after: bool = False
    join_completion: bool = False


@dataclass(frozen=True)
class SequentialBlock:
    """A queue body that may repeat before advancing to the next body."""

    name: str
    stages: tuple[SequentialStage, ...] | list[SequentialStage]
    repeat: int = 1
    reload_after: bool = True
    barrier_banks: int = 1
    reload_barrier_start: int | None = None
    reload_mxfp_resident: bool = False
    elide_terminal_reload: bool = False
    async_reload_after: bool = False
    async_reload_worker_base: int = 32


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
            if not inst.opcode & _MEM_ALLOCATE:
                continue
            if inst.opcode & _MEM_WRITEBACK and not inst.annotation.get(
                "readwrite_load"
            ):
                continue
            port = 1 if inst.opcode & _MEM_PORT1 else 0
            first_load_by_port.setdefault(port, inst)
        if not first_load_by_port:
            memory_commands = [
                inst
                for inst in instructions
                if isinstance(inst, MemoryInstruction)
            ]
            if memory_commands and all(
                inst.annotation.get("input_independent_writeback")
                for inst in memory_commands
            ):
                continue
            raise ValueError(
                f"sequential stage {stage!r} has work but no allocating load boundary"
            )
        for inst in first_load_by_port.values():
            if inst.annotation.get("readwrite_load"):
                coord = inst.annotation.get("input_bar_coord")
                if coord is None:
                    raise ValueError("read/write load has no input-barrier field")
                if inst.cords[coord] not in (0xFFFF, bar_id):
                    raise ValueError(
                        f"stage {stage!r} read/write instruction already waits "
                        f"on barrier {inst.cords[coord]}; cannot also wait on {bar_id}"
                    )
                inst.cords[coord] = bar_id
            else:
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
            memory_commands = [
                inst
                for inst in instructions
                if isinstance(inst, MemoryInstruction)
            ]
            if memory_commands and all(
                inst.annotation.get("input_independent_writeback")
                for inst in memory_commands
            ):
                continue
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
    for command_index, (compute, load) in enumerate(zip(computes, loads)):
        pair_count = int(compute.args[0])
        encoded_phase = (
            load.arg >> TmaLoadMxfpCoupledStream.PHASE_BASE_SHIFT
        ) & TmaLoadMxfpCoupledStream.MAX_PHASE_BASE
        stream_length = (
            int(load.size) & TmaLoadMxfpCoupledStream.STREAM_LENGTH_MASK
        )
        if (
            pair_count != stream_length
            or int(compute.args[2]) != encoded_phase
            or encoded_phase
            % (2 * TmaLoadMxfpCoupledStream.FP8_STAGES)
            != local_phase
        ):
            raise ValueError(
                "coupled FP8 stage has inconsistent local phase progression: "
                f"command={command_index} pairs={pair_count} "
                f"stream_length={stream_length} raw_size={int(load.size)} "
                f"compute_phase={int(compute.args[2])} "
                f"encoded_phase={encoded_phase} expected={local_phase}"
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
        initial_barrier: int | None = None,
        profile_event_count: int | None = None,
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
        self.initial_barrier = initial_barrier
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
        completion_join_tails: list[tuple[str, MemoryInstruction]] = []
        for stage_index, stage in enumerate(self.stages):
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
            if stage.parallel_with_previous and (
                stage.wait_for_previous
                or stage.wait_group is not None
                or stage.wait_group_roles
            ):
                raise ValueError(
                    f"parallel stage {stage.name!r} cannot also wait on an edge"
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

            reset_mxfp_before_stage = (
                previous_stage is not None
                and previous_stage.reset_mxfp_resident_after
            )
            if stage.prefetch_before_resident_reset and (
                not reset_mxfp_before_stage
                or not stage.prefetch_before_wait
            ):
                raise ValueError(
                    f"stage {stage.name!r} can prefetch across a resident "
                    "reset only with an explicit prefetched input dependency"
                )
            if reset_mxfp_before_stage and (
                not stage.wait_for_previous
                or stage.parallel_with_previous
                or stage.wait_group is not None
                or stage.wait_group_roles
            ):
                raise ValueError(
                    f"stage {previous_name!r} resident reset requires a "
                    "direct dependency on the following stage"
                )

            input_bar = None
            if stage_index == 0 and initial_barrier is not None:
                if (
                    not stage.wait_for_previous
                    or stage.parallel_with_previous
                    or stage.wait_group is not None
                    or stage.wait_group_roles
                ):
                    raise ValueError(
                        f"initial dependency for stage {stage.name!r} "
                        "requires its direct input edge"
                    )
                input_bar = initial_barrier
            elif stage.wait_group is not None:
                input_bar = group_barriers[stage.wait_group]
            elif stage.wait_group_roles:
                # The schedule binds each consuming memory command to its
                # own named edge below; no stage-wide LDU gate is needed.
                pass
            elif previous is not None and stage.wait_for_previous:
                count, tails = _writeback_tail(previous, previous_name)
                named_tails = [
                    (previous_name, tail) for tail in tails
                ]
                if reset_mxfp_before_stage and completion_join_tails:
                    named_tails.extend(completion_join_tails)
                    completion_join_tails = []
                    unique_named_tails = []
                    seen_tail_ids = set()
                    for stage_name, tail in named_tails:
                        if id(tail) in seen_tail_ids:
                            continue
                        seen_tail_ids.add(id(tail))
                        unique_named_tails.append((stage_name, tail))
                    named_tails = unique_named_tails
                    count = len(named_tails)
                if launcher.num_bars >= config.max_bars - 2:
                    raise ValueError("sequential program exceeds the runtime barrier capacity")
                input_bar = launcher.new_bar(count)
                self.barriers.append(input_bar)
                for stage_name, tail in named_tails:
                    _attach_bar(tail, input_bar, stage=stage_name)
            elif previous is not None:
                if previous_profile_after:
                    raise ValueError(
                        "a profiled stage cannot elide its completion edge"
                    )
                placement_differs = (
                    stage.base_sm != previous_stage.base_sm
                    or stage.num_sms != previous_stage.num_sms
                )
                if placement_differs and not stage.parallel_with_previous:
                    raise ValueError(
                        f"independent stage {stage.name!r} must match the "
                        "previous stage placement so its queue tail dominates"
                    )
                if stage.parallel_with_previous:
                    previous_stop = (
                        previous_stage.base_sm + previous_stage.num_sms
                    )
                    stage_stop = stage.base_sm + stage.num_sms
                    if not (
                        stage_stop <= previous_stage.base_sm
                        or previous_stop <= stage.base_sm
                    ):
                        raise ValueError(
                            f"parallel stage {stage.name!r} must use a "
                            "disjoint SM placement"
                        )

            if previous_profile_after:
                if self.profile_event_count == 0:
                    raise ValueError("profiled stage requires profile event capacity")
                marker = ProfileLayer(
                    config.layer_profile_event_base,
                    self.profile_event_count,
                )
                for instructions in self.instructions:
                    instructions.append(marker)

            resident_reset_reload = None
            if reset_mxfp_before_stage:
                if input_bar is None:
                    raise ValueError(
                        f"stage {previous_name!r} resident reset has no "
                        "completion dependency"
                    )
                # The control command waits directly on the producer's STU
                # completion edge, drains both LDU queues, and resets the
                # one-shot resident ring state.  Allocator publication does
                # not advance to the next stage until both LDUs acknowledge,
                # so the following loads need no second wait on a barrier that
                # this command has just reinitialized.
                resident_reset_reload = LduReloadBarriers(
                    launcher.bars_src,
                    input_bar,
                    1,
                    # Slot two identifies loop-tail reloads to the optional
                    # profiler.  A distinct immutable mailbox keeps this
                    # intra-body reset out of that bounded event stream.
                    special_slot=7,
                    reset_mxfp_resident=True,
                ).bar(input_bar)
                if not stage.prefetch_before_resident_reset:
                    input_bar = None

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
                    profile_prefix = []
                    if stage.profile_step_begin_event is not None:
                        profile_prefix.append(
                            ProfileEvent(stage.profile_step_begin_event)
                        )
                    profile_prefix.append(
                        ProfileStep(stage.profile_step_event, begin=True)
                    )
                    instructions[0:0] = profile_prefix
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
                try:
                    coupled_fp8_phases[sm] = _rebase_coupled_fp8_phases(
                        instructions, coupled_fp8_phases[sm]
                    )
                except ValueError as error:
                    raise ValueError(
                        f"sequential stage {stage.name!r} SM {sm}: {error}"
                    ) from error
            if input_bar is not None:
                if stage.prefetch_before_wait:
                    _validate_prefetch_gate(rendered, input_bar, stage.name)
                else:
                    _gate_load_ports(rendered, input_bar, stage.name)
            if resident_reset_reload is not None:
                found_reset_prefetch = False
                if any(
                    isinstance(inst, MemoryInstruction)
                    and inst.annotation.get("profile_hc_global_bar")
                    for instructions in rendered
                    for inst in instructions
                ):
                    # Bits 12--13 are unused by the ordinary reload opcode.
                    # Tag the matching reset so both LDU ports can trace the
                    # first restore of the same global-counter generation.
                    resident_reset_reload.arg |= 1 << 13
                for sm, instructions in enumerate(rendered):
                    prefetch_positions = [
                        index
                        for index, inst in enumerate(instructions)
                        if isinstance(inst, MemoryInstruction)
                        and inst.annotation.get(
                            "prefetch_before_resident_reset"
                        )
                    ]
                    if prefetch_positions:
                        found_reset_prefetch = True
                        prefix_stop = prefetch_positions[-1] + 1
                        if any(
                            isinstance(inst, MemoryInstruction)
                            and not inst.annotation.get(
                                "prefetch_before_resident_reset"
                            )
                            for inst in instructions[:prefix_stop]
                        ):
                            raise ValueError(
                                f"stage {stage.name!r} resident-reset "
                                "prefetch is not a memory-stream prefix"
                            )
                        self.instructions[sm].extend(
                            inst
                            for inst in instructions[:prefix_stop]
                            if isinstance(inst, MemoryInstruction)
                            and inst.annotation.get(
                                "prefetch_before_resident_reset"
                            )
                        )
                        rendered[sm] = [
                            inst
                            for index, inst in enumerate(instructions)
                            if index not in prefetch_positions
                        ]
                    self.instructions[sm].append(
                        resident_reset_reload.copy()
                    )
                if (
                    stage.prefetch_before_resident_reset
                    and not found_reset_prefetch
                ):
                    raise ValueError(
                        f"stage {stage.name!r} declared resident-reset "
                        "prefetching but emitted no annotated memory operand"
                    )
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
            if stage.join_completion:
                if not completion_barrier:
                    raise ValueError(
                        f"stage {stage.name!r} joins a disabled completion barrier"
                    )
                _, tails = _writeback_tail(rendered, stage.name)
                completion_join_tails.extend(
                    (stage.name, tail) for tail in tails
                )
            previous = rendered
            previous_stage = stage
            previous_name = stage.name
            previous_profile_after = stage.profile_after

        if previous_stage.reset_mxfp_resident_after:
            raise ValueError(
                f"final stage {previous_name!r} cannot reset resident MXFP "
                "state without a following stage"
            )

        for group, bar_id in group_barriers.items():
            count = group_release_counts[group]
            if count <= 0:
                raise ValueError(f"release group {group!r} has no active producers")
            launcher.set_bar(bar_id, count)

        if completion_barrier:
            _, tails = _writeback_tail(previous, previous_name)
            completion_tails = [
                (previous_name, tail) for tail in tails
            ] + completion_join_tails
            # A final stage may itself request a completion join.  Count and
            # annotate each physical STU command once in that case.
            unique_tails = []
            seen_tail_ids = set()
            for stage_name, tail in completion_tails:
                if id(tail) in seen_tail_ids:
                    continue
                seen_tail_ids.add(id(tail))
                unique_tails.append((stage_name, tail))
            count = len(unique_tails)
            if launcher.num_bars >= config.max_bars - 2:
                raise ValueError("sequential program exceeds the runtime barrier capacity")
            self.completion_barrier = launcher.new_bar(count)
            self.barriers.append(self.completion_barrier)
            for stage_name, tail in unique_tails:
                _attach_bar(tail, self.completion_barrier, stage=stage_name)
            if previous_profile_after:
                if self.profile_event_count == 0:
                    raise ValueError("profiled stage requires profile event capacity")
                marker = ProfileLayer(
                    config.layer_profile_event_base,
                    self.profile_event_count,
                )
                for instructions in self.instructions:
                    instructions.append(marker)
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

        carried_barrier = None
        for block_index, block in enumerate(self.blocks):
            if not 1 <= block.repeat <= 0xFFFF:
                raise ValueError(f"block {block.name!r} repeat must fit in uint16")
            if block.repeat > 1 and not block.reload_after:
                raise ValueError(
                    f"repeated block {block.name!r} must reload its dependencies"
                )
            if block.async_reload_after and not block.reload_after:
                raise ValueError(
                    f"block {block.name!r} async reload needs a completion barrier"
                )
            if block.reload_mxfp_resident and not block.reload_after:
                raise ValueError(
                    f"block {block.name!r} cannot reset resident MXFP state "
                    "without a loop-tail reload"
                )
            if block.barrier_banks <= 0:
                raise ValueError(f"block {block.name!r} barrier_banks must be positive")
            if block.elide_terminal_reload:
                if not block.reload_after:
                    raise ValueError(
                        f"block {block.name!r} cannot elide a disabled reload"
                    )
                if block_index + 1 >= len(self.blocks):
                    raise ValueError(
                        f"block {block.name!r} terminal dependency has no consumer"
                    )
            bank_count = min(block.barrier_banks, block.repeat)
            while block.repeat % bank_count:
                bank_count -= 1
            if block.reload_barrier_start is not None:
                if block.async_reload_after:
                    raise ValueError(
                        f"block {block.name!r} async reload owns its local bank range"
                    )
                if bank_count != 1:
                    raise ValueError(
                        f"block {block.name!r} cannot combine an external "
                        "reload range with shifted barrier banks"
                    )
                if (
                    block.reload_barrier_start < 0
                    or block.reload_barrier_start > launcher.num_bars
                ):
                    raise ValueError(
                        f"block {block.name!r} has an invalid reload barrier start"
                    )
            if (
                block.elide_terminal_reload
                and bank_count != 1
                and not block.async_reload_after
            ):
                raise ValueError(
                    f"block {block.name!r} terminal reload elision requires one barrier bank"
                )
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

            bank_barrier_start = launcher.num_bars
            async_ready_bar = None
            async_worker_bar = None
            if block.async_reload_after:
                if launcher.num_bars >= config.max_bars - 3:
                    raise ValueError(
                        "looped program exceeds async reload barrier capacity"
                    )
                async_ready_bar = launcher.new_bar(0)
                async_worker_bar = launcher.new_bar(0)
            segment = SequentialProgram(
                launcher,
                block.stages,
                completion_barrier=block.reload_after,
                initial_barrier=carried_barrier,
                profile_event_count=self.profile_event_count,
                balance_load_ports=balance_load_ports,
                coupled_fp8_initial_phases=coupled_fp8_phases,
            )
            carried_barrier = None
            # Coupled FP8 phase is owned by persistent per-SM LDU/compute
            # counters. Repeated command bodies therefore advance naturally;
            # they need not return to their encoded entry phase per iteration.
            coupled_fp8_phases = list(segment.coupled_fp8_final_phases)
            self.segments.append(segment)
            bank_barrier_stop = launcher.num_bars
            barriers_per_bank = bank_barrier_stop - bank_barrier_start
            if bank_count > 1:
                if not block.reload_after or barriers_per_bank <= 0:
                    raise ValueError(
                        f"block {block.name!r} needs a completion barrier for shifting"
                    )
                base_values = [
                    launcher.bar_values[bar_id]
                    for bar_id in range(bank_barrier_start, bank_barrier_stop)
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
                        if bank_barrier_start <= bar_id < bank_barrier_stop:
                            inst.group()
            inner_reg = 0
            outer_reg = int(bank_count > 1)
            if block.async_reload_after:
                workers = config.async_barrier_reload_workers
                worker_base = block.async_reload_worker_base
                if (
                    worker_base < 0
                    or worker_base + workers > launcher.num_sms
                ):
                    raise ValueError(
                        f"block {block.name!r} has no disjoint async reload workers"
                )
                clear_count = (
                    segment.completion_barrier + 1 - segment.barrier_start
                )
                if clear_count <= 0:
                    raise ValueError(
                        f"block {block.name!r} has no barrier range to clear"
                )
                width, remainder = divmod(clear_count, workers)
                for sm in range(launcher.num_sms):
                    segment.instructions[sm].insert(
                        0,
                        LduWaitBarrier(
                            outer_reg,
                            special_slot=3,
                        )
                        .bar(async_ready_bar)
                        .group(),
                    )
                    worker = sm - worker_base
                    if 0 <= worker < workers:
                        local_count = width + int(worker < remainder)
                        local_offset = (
                            worker * width + min(worker, remainder)
                        )
                        segment.instructions[sm].append(
                            LduAsyncReloadBarriers(
                                launcher.bars_src,
                                segment.barrier_start + local_offset,
                                local_count,
                                segment.completion_barrier,
                                special_slot=2,
                                shift_target=True,
                                bank_ready_completion=True,
                                bank_ready_leader=worker == 0,
                            )
                            .bar(async_worker_bar)
                            .group()
                        )
            self.barriers.extend(
                range(
                    bank_barrier_start,
                    bank_barrier_start + barriers_per_bank * bank_count,
                )
            )
            self.stage_stats.extend(segment.stage_stats)
            self.placed_schedules.extend(segment.placed_schedules)
            for sm in range(launcher.num_sms):
                self.instructions[sm].extend(segment.instructions[sm])

            if block.reload_after and not (
                block.elide_terminal_reload and block.repeat == 1
            ) and not block.async_reload_after:
                # Keep the reload in the repeated memory body. It waits for
                # the current bank's final STU completion and sits ahead of
                # every following LDU command in both port FIFOs, providing
                # the loop-carried dependency without an IssueBarrier.
                reload_start = (
                    segment.barrier_start
                    if block.reload_barrier_start is None
                    else block.reload_barrier_start
                )
                reload_count = segment.completion_barrier + 1 - reload_start
                if reload_count <= 0:
                    raise ValueError(
                        f"block {block.name!r} reload range does not reach "
                        "its completion barrier"
                    )
                reload = LduReloadBarriers(
                    launcher.bars_src,
                    reload_start,
                    reload_count,
                    2,
                    reset_mxfp_resident=block.reload_mxfp_resident,
                    skip_final_loop_reg=outer_reg
                    if block.elide_terminal_reload
                    else None,
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
            if block.elide_terminal_reload:
                final_bank = (block.repeat - 1) % bank_count
                carried_barrier = (
                    segment.completion_barrier
                    + final_bank * barriers_per_bank
                )

        if carried_barrier is not None:
            raise ValueError("terminal dependency barrier was not consumed")

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
