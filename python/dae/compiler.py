from __future__ import annotations

from dataclasses import dataclass, replace
import copy
from pathlib import Path
import hashlib
from typing import Iterable

from .instruction_utils import decode_opcode
from .instructions import ComputeInstruction, MemoryInstruction
from .op_families import family_spec_by_name
from .tma_utils import cords2addr


MEM_FLAG_ALLOCATE = 0x1
MEM_FLAG_WRITEBACK = 0x2
MEM_FLAG_GROUP = 0x4
MEM_FLAG_JUMP = 0x8
MEM_FLAG_BARRIER = 0x10
MEM_FLAG_PORT = 0x20
MEM_FLAG_MASK = 0x3F
MEM_DYNAMIC_FLAG_MASK = MEM_FLAG_GROUP | MEM_FLAG_JUMP | MEM_FLAG_BARRIER | MEM_FLAG_PORT


class CompileModeError(RuntimeError):
    pass


def _clone_instruction(inst):
    return copy.copy(inst)


def _clear_jump(inst: MemoryInstruction) -> MemoryInstruction:
    out = _clone_instruction(inst)
    out.opcode &= ~MEM_FLAG_JUMP
    return out


def _set_jump(inst: MemoryInstruction, enabled: bool) -> MemoryInstruction:
    out = _clone_instruction(inst)
    out.opcode &= ~MEM_FLAG_JUMP
    if enabled:
        out.opcode |= MEM_FLAG_JUMP
    return out


def _base_memory_opcode_value(raw_opcode_value: int) -> int:
    return raw_opcode_value & ~MEM_DYNAMIC_FLAG_MASK


def _memory_opcode_name(raw_opcode_value: int) -> str:
    return decode_opcode(_base_memory_opcode_value(raw_opcode_value))


def _address_kind_for_opcode(opcode_name: str) -> str:
    coord_tokens = (
        "_2D",
        "_3D",
        "_4D",
        "_5D",
        "_REDUCE_ADD_2D",
        "_REDUCE_ADD_3D",
    )
    if "TENSOR_1D" in opcode_name:
        return "address"
    if any(token in opcode_name for token in coord_tokens):
        return "coords"
    if opcode_name in {"OP_LOOP", "OP_REPEAT"}:
        return "control"
    return "address"


@dataclass(frozen=True)
class AddressRecipe:
    kind: str
    address: int
    coords: tuple[int, int, int, int]

    def structural_key(self) -> tuple[object, ...]:
        if self.kind == "coords":
            return (self.kind, *self.coords)
        return (self.kind, self.address)


@dataclass(frozen=True)
class ComputeOpIR:
    opcode_name: str
    args: tuple[int, ...]
    op_family_name: str | None
    original: ComputeInstruction
    kind: str = "compute_op"

    def emit(self) -> list[ComputeInstruction]:
        return [_clone_instruction(self.original)]


@dataclass(frozen=True)
class LoopCIR:
    count: int
    target_pc: int
    opcode_name: str
    original: ComputeInstruction
    kind: str = "loopc"

    def emit(self) -> list[ComputeInstruction]:
        return [_clone_instruction(self.original)]


@dataclass(frozen=True)
class TerminateComputeIR:
    opcode_name: str
    original: ComputeInstruction
    kind: str = "terminate_compute"

    def emit(self) -> list[ComputeInstruction]:
        return [_clone_instruction(self.original)]


ComputeIRNode = ComputeOpIR | LoopCIR | TerminateComputeIR


@dataclass(frozen=True)
class MemoryOpIR:
    opcode_name: str
    base_opcode_value: int
    raw_opcode_value: int
    raw_num_slots: int
    slot_request: int | None
    arg: int
    size: int
    barrier_id: int | None
    address_recipe: AddressRecipe
    allocate: bool
    writeback: bool
    group: bool
    jump: bool
    port: int
    queue_role: str
    original: MemoryInstruction
    kind: str = "memory_op"

    def emit(self, jump_override: bool | None = None) -> list[MemoryInstruction]:
        if jump_override is None:
            return [_clone_instruction(self.original)]
        return [_set_jump(self.original, jump_override)]

    def structural_key(self, ignore_jump: bool = False) -> tuple[object, ...]:
        return (
            self.opcode_name,
            self.slot_request,
            self.arg,
            self.size,
            self.barrier_id,
            self.allocate,
            self.writeback,
            self.group,
            False if ignore_jump else self.jump,
            self.port,
            self.queue_role,
            self.address_recipe.structural_key(),
        )


@dataclass(frozen=True)
class RepeatControlIR:
    count: int
    reg_start: int
    reg_end: int
    delta_cords: tuple[int, int, int, int]
    original: MemoryInstruction
    kind: str = "repeat_control"

    def emit(self, count_override: int | None = None) -> list[MemoryInstruction]:
        inst = _clone_instruction(self.original)
        inst.size = self.count if count_override is None else count_override
        return [inst]

    def structural_key(self) -> tuple[object, ...]:
        return (self.reg_start, self.reg_end, self.delta_cords)


@dataclass(frozen=True)
class RepeatRegionIR:
    count: int
    controls: tuple[RepeatControlIR, ...]
    body: tuple[MemoryOpIR, ...]
    kind: str = "repeat_region"

    def structural_key(self) -> tuple[object, ...]:
        return (
            tuple(control.structural_key() for control in self.controls),
            tuple(node.structural_key(ignore_jump=True) for node in self.body),
        )

    def emit(self) -> list[MemoryInstruction]:
        if len(self.body) == 0:
            return []

        if self.count <= 1:
            return [
                emitted
                for index, node in enumerate(self.body)
                for emitted in node.emit(jump_override=False if index == len(self.body) - 1 else None)
            ]

        emitted: list[MemoryInstruction] = []
        for index, control in enumerate(self.controls):
            emitted.extend(control.emit(count_override=self.count if index == len(self.controls) - 1 else 0))
        for index, node in enumerate(self.body):
            emitted.extend(node.emit(jump_override=index == len(self.body) - 1))
        return emitted

    def can_merge_with(self, other: object) -> bool:
        if not isinstance(other, RepeatRegionIR):
            return False
        return self.structural_key() == other.structural_key()

    def merged_with(self, other: "RepeatRegionIR") -> "RepeatRegionIR":
        if not self.can_merge_with(other):
            raise ValueError("cannot merge mismatched repeat regions")
        return replace(self, count=self.count + other.count)


@dataclass(frozen=True)
class LoopMIR:
    count: int
    target_pc: int
    reg: int
    bar_shift: int
    tma_shift: int
    opcode_name: str
    original: MemoryInstruction
    kind: str = "loopm"

    def emit(self) -> list[MemoryInstruction]:
        return [_clone_instruction(self.original)]


@dataclass(frozen=True)
class BarrierIssueIR:
    barrier_id: int | None
    opcode_name: str
    original: MemoryInstruction
    kind: str = "barrier_issue"

    def emit(self) -> list[MemoryInstruction]:
        return [_clone_instruction(self.original)]


@dataclass(frozen=True)
class TerminateMemoryIR:
    opcode_name: str
    original: MemoryInstruction
    kind: str = "terminate_memory"

    def emit(self) -> list[MemoryInstruction]:
        return [_clone_instruction(self.original)]


MemoryIRNode = MemoryOpIR | RepeatControlIR | RepeatRegionIR | LoopMIR | BarrierIssueIR | TerminateMemoryIR


@dataclass(frozen=True)
class SMProgramIR:
    sm_id: int
    compute_ops: tuple[ComputeIRNode, ...]
    memory_ops: tuple[MemoryIRNode, ...]


@dataclass(frozen=True)
class ProgramIR:
    sms: tuple[SMProgramIR, ...]


@dataclass(frozen=True)
class GeneratedCudaSource:
    path: Path
    source: str
    tag: str


@dataclass(frozen=True)
class CompileArtifacts:
    mode: str
    original_program: ProgramIR
    normalized_program: ProgramIR
    emitted_compute: tuple[tuple[ComputeInstruction, ...], ...]
    emitted_memory: tuple[tuple[MemoryInstruction, ...], ...]
    split_unit_program: "SplitUnitProgramIR | None" = None
    generated_cuda: GeneratedCudaSource | None = None
    generated_runtime: GeneratedCudaSource | None = None


@dataclass(frozen=True)
class SplitLoopSpanIR:
    start: int
    count: int
    trip_count: int


@dataclass(frozen=True)
class SplitAllocOpIR:
    sm_id: int
    opcode_name: str
    slot_request: int
    queue_role: str
    writeback: bool
    group: bool
    barrier_id: int | None
    port: int


@dataclass(frozen=True)
class SplitMemOpIR:
    sm_id: int
    opcode_name: str
    queue_role: str
    arg: int
    size: int
    barrier_id: int | None
    port: int
    group: bool
    address_recipe: AddressRecipe


@dataclass(frozen=True)
class SplitComputeOpIR:
    sm_id: int
    opcode_name: str
    kind: str
    args: tuple[int, ...]
    target_pc: int | None = None
    trip_count: int | None = None


@dataclass(frozen=True)
class SMSplitUnitIR:
    sm_id: int
    alloc_ops: tuple[SplitAllocOpIR, ...]
    alloc_spans: tuple[SplitLoopSpanIR, ...]
    ldu_ops: tuple[SplitMemOpIR, ...]
    ldu_spans: tuple[SplitLoopSpanIR, ...]
    stu_ops: tuple[SplitMemOpIR, ...]
    stu_spans: tuple[SplitLoopSpanIR, ...]
    compute_ops: tuple[SplitComputeOpIR, ...]


@dataclass(frozen=True)
class SplitUnitProgramIR:
    sms: tuple[SMSplitUnitIR, ...]


def _build_compute_node(inst: ComputeInstruction) -> ComputeIRNode:
    opcode_name = inst.compute_operator_name()
    if opcode_name.startswith("UNKNOWN_OPCODE"):
        raise CompileModeError(f"Unsupported compute opcode in compile mode: {opcode_name}")
    if opcode_name == "OP_LOOPC":
        return LoopCIR(count=inst.args[0], target_pc=inst.args[1], opcode_name=opcode_name, original=inst)
    if opcode_name == "OP_TERMINATEC":
        return TerminateComputeIR(opcode_name=opcode_name, original=inst)
    return ComputeOpIR(
        opcode_name=opcode_name,
        args=tuple(inst.args),
        op_family_name=getattr(inst, "op_family_name", None),
        original=inst,
    )


def _build_memory_node(inst: MemoryInstruction) -> MemoryIRNode:
    opcode_name = _memory_opcode_name(inst.opcode)
    if opcode_name.startswith("UNKNOWN_OPCODE"):
        raise CompileModeError(f"Unsupported memory opcode in compile mode: 0x{inst.opcode:04x}")

    if opcode_name == "OP_REPEAT":
        reg_start = inst.num_slots & 0xFF
        reg_end = inst.num_slots >> 8
        return RepeatControlIR(
            count=inst.size,
            reg_start=reg_start,
            reg_end=reg_end,
            delta_cords=tuple(int(value) for value in inst.cords),
            original=inst,
        )

    if opcode_name == "OP_LOOP":
        bar_shift = inst.cords[2] >> 6
        return LoopMIR(
            count=inst.size,
            target_pc=inst.cords[0],
            reg=inst.num_slots,
            bar_shift=bar_shift,
            tma_shift=inst.cords[3],
            opcode_name=opcode_name,
            original=inst,
        )

    if opcode_name == "OP_ISSUE_BARRIER":
        return BarrierIssueIR(
            barrier_id=inst.num_slots >> 6 if inst.opcode & MEM_FLAG_BARRIER else None,
            opcode_name=opcode_name,
            original=inst,
        )

    if opcode_name == "OP_TERMINATE":
        return TerminateMemoryIR(opcode_name=opcode_name, original=inst)

    allocate = bool(inst.opcode & MEM_FLAG_ALLOCATE)
    writeback = bool(inst.opcode & MEM_FLAG_WRITEBACK)
    barrier = bool(inst.opcode & MEM_FLAG_BARRIER)
    slot_request = (inst.num_slots & 0x3F) if allocate else None
    queue_role = "control"
    if allocate:
        queue_role = "store" if writeback else "load"

    return MemoryOpIR(
        opcode_name=opcode_name,
        base_opcode_value=_base_memory_opcode_value(inst.opcode),
        raw_opcode_value=inst.opcode,
        raw_num_slots=inst.num_slots,
        slot_request=slot_request,
        arg=inst.arg,
        size=inst.size,
        barrier_id=(inst.num_slots >> 6) if barrier else None,
        address_recipe=AddressRecipe(
            kind=_address_kind_for_opcode(opcode_name),
            address=int(cords2addr(inst.cords)),
            coords=tuple(int(value) for value in inst.cords),
        ),
        allocate=allocate,
        writeback=writeback,
        group=bool(inst.opcode & MEM_FLAG_GROUP),
        jump=bool(inst.opcode & MEM_FLAG_JUMP),
        port=1 if inst.opcode & MEM_FLAG_PORT else 0,
        queue_role=queue_role,
        original=inst,
    )


def build_program_ir(builders: Iterable[object]) -> ProgramIR:
    sms = []
    for builder in builders:
        compute_ops = tuple(_build_compute_node(inst) for inst in builder.cinsts)
        memory_ops = tuple(_build_memory_node(inst) for inst in builder.minsts)
        sms.append(SMProgramIR(sm_id=builder.sm_id, compute_ops=compute_ops, memory_ops=memory_ops))
    return ProgramIR(sms=tuple(sms))


def _collapse_repeat_controls(memory_ops: tuple[MemoryIRNode, ...]) -> tuple[MemoryIRNode, ...]:
    collapsed: list[MemoryIRNode] = []
    index = 0
    while index < len(memory_ops):
        node = memory_ops[index]
        if not isinstance(node, RepeatControlIR):
            collapsed.append(node)
            index += 1
            continue

        controls: list[RepeatControlIR] = []
        while index < len(memory_ops) and isinstance(memory_ops[index], RepeatControlIR):
            controls.append(memory_ops[index])
            index += 1

        body: list[MemoryOpIR] = []
        while index < len(memory_ops) and isinstance(memory_ops[index], MemoryOpIR):
            body.append(memory_ops[index])
            index += 1
            if body[-1].jump:
                break

        if len(body) == 0:
            collapsed.extend(controls)
            continue

        last_control = controls[-1]
        count = max(last_control.count, 1)
        collapsed.append(
            RepeatRegionIR(
                count=count,
                controls=tuple(controls),
                body=tuple(body),
            )
        )
    return tuple(collapsed)


def _merge_adjacent_repeat_regions(memory_ops: tuple[MemoryIRNode, ...]) -> tuple[MemoryIRNode, ...]:
    merged: list[MemoryIRNode] = []
    for node in memory_ops:
        if isinstance(node, RepeatRegionIR) and merged and isinstance(merged[-1], RepeatRegionIR) and merged[-1].can_merge_with(node):
            merged[-1] = merged[-1].merged_with(node)
            continue
        merged.append(node)
    return tuple(merged)


def normalize_program_ir(program: ProgramIR) -> ProgramIR:
    normalized_sms: list[SMProgramIR] = []
    for sm in program.sms:
        memory_ops = _collapse_repeat_controls(sm.memory_ops)
        memory_ops = _merge_adjacent_repeat_regions(memory_ops)
        normalized_sms.append(
            SMProgramIR(
                sm_id=sm.sm_id,
                compute_ops=sm.compute_ops,
                memory_ops=memory_ops,
            )
        )
    return ProgramIR(sms=tuple(normalized_sms))


def _validate_program(program: ProgramIR, mode: str) -> None:
    supported_memory_ops_compile_cuda = {
        "OP_ALLOC_TMA_LOAD_1D",
        "OP_ALLOC_TMA_LOAD_TENSOR_1D",
        "OP_ALLOC_TMA_LOAD_2D",
        "OP_ALLOC_TMA_LOAD_3D",
        "OP_ALLOC_TMA_LOAD_4D",
        "OP_ALLOC_TMA_LOAD_5D_FIX0",
        "OP_ALLOC_WB_TMA_STORE_1D",
        "OP_ALLOC_WB_TMA_STORE_2D",
        "OP_ALLOC_WB_TMA_STORE_3D",
        "OP_ALLOC_WB_TMA_STORE_4D",
        "OP_ALLOC_WB_TMA_STORE_5D_FIX0",
        "OP_ALLOC_WB_TMA_REDUCE_ADD_2D",
        "OP_ALLOC_WB_TMA_REDUCE_ADD_3D",
        "OP_TERMINATE",
    }
    supported_compute_prefixes_compile_cuda = ("OP_GEMV_WGMMA__",)
    supported_compute_exact_compile_cuda = {"OP_TERMINATEC", "OP_DUMMY", "OP_COPY"}

    for sm in program.sms:
        for node in sm.compute_ops:
            if isinstance(node, ComputeOpIR) and node.opcode_name.startswith("UNKNOWN_OPCODE"):
                raise CompileModeError(f"Unsupported compute opcode on SM {sm.sm_id}: {node.opcode_name}")
            if mode != "compile_cuda":
                continue
            opcode_name = node.opcode_name
            if opcode_name in supported_compute_exact_compile_cuda:
                continue
            if any(opcode_name.startswith(prefix) for prefix in supported_compute_prefixes_compile_cuda):
                continue
            raise CompileModeError(
                f"compile_cuda currently supports the GEMV_WGMMA subset only; "
                f"SM {sm.sm_id} uses compute op {opcode_name}"
            )

        for node in sm.memory_ops:
            if isinstance(node, MemoryOpIR):
                if node.opcode_name.startswith("UNKNOWN_OPCODE"):
                    raise CompileModeError(f"Unsupported memory opcode on SM {sm.sm_id}: {node.opcode_name}")
                if mode == "compile_cuda" and node.opcode_name not in supported_memory_ops_compile_cuda:
                    raise CompileModeError(
                        f"compile_cuda does not yet support memory op {node.opcode_name} on SM {sm.sm_id}"
                    )
            elif isinstance(node, RepeatRegionIR):
                if mode == "compile_cuda":
                    for body_op in node.body:
                        if body_op.opcode_name not in supported_memory_ops_compile_cuda:
                            raise CompileModeError(
                                f"compile_cuda does not yet support repeated memory op "
                                f"{body_op.opcode_name} on SM {sm.sm_id}"
                            )
            elif isinstance(node, (BarrierIssueIR, LoopMIR, TerminateMemoryIR)):
                if mode == "compile_cuda" and not isinstance(node, TerminateMemoryIR):
                    raise CompileModeError(
                        f"compile_cuda does not yet support control-only memory op {node.opcode_name} on SM {sm.sm_id}"
                    )
            elif isinstance(node, RepeatControlIR):
                raise CompileModeError(f"Unnormalized repeat control reached validation on SM {sm.sm_id}")


def emit_program_ir(program: ProgramIR) -> tuple[tuple[ComputeInstruction, ...], tuple[tuple[MemoryInstruction, ...], ...]]:
    compute_by_sm: list[tuple[ComputeInstruction, ...]] = []
    memory_by_sm: list[tuple[MemoryInstruction, ...]] = []
    for sm in program.sms:
        compute_ops: list[ComputeInstruction] = []
        for node in sm.compute_ops:
            compute_ops.extend(node.emit())

        memory_ops: list[MemoryInstruction] = []
        for node in sm.memory_ops:
            memory_ops.extend(node.emit())

        compute_by_sm.append(tuple(compute_ops))
        memory_by_sm.append(tuple(memory_ops))
    return tuple(compute_by_sm), tuple(memory_by_sm)


def lower_to_split_units(program: ProgramIR) -> SplitUnitProgramIR:
    sms: list[SMSplitUnitIR] = []
    for sm in program.sms:
        alloc_ops: list[SplitAllocOpIR] = []
        alloc_spans: list[SplitLoopSpanIR] = []
        ldu_ops: list[SplitMemOpIR] = []
        ldu_spans: list[SplitLoopSpanIR] = []
        stu_ops: list[SplitMemOpIR] = []
        stu_spans: list[SplitLoopSpanIR] = []

        def append_mem_ops(body: tuple[MemoryOpIR, ...], trip_count: int):
            alloc_start = len(alloc_ops)
            ldu_start = len(ldu_ops)
            stu_start = len(stu_ops)

            for body_op in body:
                if body_op.allocate:
                    alloc_ops.append(
                        SplitAllocOpIR(
                            sm_id=sm.sm_id,
                            opcode_name=body_op.opcode_name,
                            slot_request=body_op.slot_request or 0,
                            queue_role=body_op.queue_role,
                            writeback=body_op.writeback,
                            group=body_op.group,
                            barrier_id=body_op.barrier_id,
                            port=body_op.port,
                        )
                    )
                mem_op = SplitMemOpIR(
                    sm_id=sm.sm_id,
                    opcode_name=body_op.opcode_name,
                    queue_role=body_op.queue_role,
                    arg=body_op.arg,
                    size=body_op.size,
                    barrier_id=body_op.barrier_id,
                    port=body_op.port,
                    group=body_op.group,
                    address_recipe=body_op.address_recipe,
                )
                if body_op.queue_role == "load":
                    ldu_ops.append(mem_op)
                elif body_op.queue_role == "store":
                    stu_ops.append(mem_op)

            alloc_count = len(alloc_ops) - alloc_start
            ldu_count = len(ldu_ops) - ldu_start
            stu_count = len(stu_ops) - stu_start
            if alloc_count:
                alloc_spans.append(SplitLoopSpanIR(start=alloc_start, count=alloc_count, trip_count=trip_count))
            if ldu_count:
                ldu_spans.append(SplitLoopSpanIR(start=ldu_start, count=ldu_count, trip_count=trip_count))
            if stu_count:
                stu_spans.append(SplitLoopSpanIR(start=stu_start, count=stu_count, trip_count=trip_count))

        for node in sm.memory_ops:
            if isinstance(node, RepeatRegionIR):
                append_mem_ops(node.body, node.count)
            elif isinstance(node, MemoryOpIR):
                append_mem_ops((node,), 1)

        compute_ops: list[SplitComputeOpIR] = []
        for node in sm.compute_ops:
            if isinstance(node, ComputeOpIR):
                compute_ops.append(
                    SplitComputeOpIR(
                        sm_id=sm.sm_id,
                        opcode_name=node.opcode_name,
                        kind=node.kind,
                        args=node.args,
                    )
                )
            elif isinstance(node, LoopCIR):
                compute_ops.append(
                    SplitComputeOpIR(
                        sm_id=sm.sm_id,
                        opcode_name=node.opcode_name,
                        kind=node.kind,
                        args=(node.count, node.target_pc),
                        target_pc=node.target_pc,
                        trip_count=node.count,
                    )
                )
            elif isinstance(node, TerminateComputeIR):
                compute_ops.append(
                    SplitComputeOpIR(
                        sm_id=sm.sm_id,
                        opcode_name=node.opcode_name,
                        kind=node.kind,
                        args=(),
                    )
                )

        sms.append(
            SMSplitUnitIR(
                sm_id=sm.sm_id,
                alloc_ops=tuple(alloc_ops),
                alloc_spans=tuple(alloc_spans),
                ldu_ops=tuple(ldu_ops),
                ldu_spans=tuple(ldu_spans),
                stu_ops=tuple(stu_ops),
                stu_spans=tuple(stu_spans),
                compute_ops=tuple(compute_ops),
            )
        )
    return SplitUnitProgramIR(sms=tuple(sms))


def _indent(lines: list[str], level: int = 1) -> list[str]:
    prefix = "  " * level
    return [prefix + line if line else "" for line in lines]


def _repeat_delta_for_index(node: RepeatRegionIR, body_index: int) -> tuple[int, int, int, int]:
    for control in node.controls:
        if control.reg_start <= body_index < control.reg_end:
            return control.delta_cords
    return (0, 0, 0, 0)


def _coord_expr(base: int, delta: int, loop_var: str | None) -> str:
    if loop_var is None or delta == 0:
        return str(base)
    return f"static_cast<uint16_t>({base} + {loop_var} * {delta})"


def _addr_expr(base: int, delta: int, loop_var: str | None) -> str:
    if loop_var is None or delta == 0:
        return f"{base}ULL"
    return f"({base}ULL + static_cast<uint64_t>({loop_var}) * {delta}ULL)"


def _memory_op_code(node: MemoryOpIR, loop_var: str | None, delta: tuple[int, int, int, int], unit: str) -> list[str]:
    if unit == "alloc":
        if node.queue_role == "load":
            return [
                f"compiled_alloc_load(lane_id, alloc, slot_avail, {node.slot_request}, {node.port}, m2c, ldq);"
            ]
        if node.queue_role == "store":
            return [
                f"compiled_alloc_store(lane_id, alloc, slot_avail, {node.slot_request}, m2c);"
            ]
        return []

    if unit == "ldu":
        if node.queue_role != "load":
            return []
        if node.port not in (0, 1):
            raise CompileModeError(f"Unsupported LDU port {node.port} for {node.opcode_name}")
        call = ""
        c0 = _coord_expr(node.address_recipe.coords[0], delta[0], loop_var)
        c1 = _coord_expr(node.address_recipe.coords[1], delta[1], loop_var)
        c2 = _coord_expr(node.address_recipe.coords[2], delta[2], loop_var)
        c3 = _coord_expr(node.address_recipe.coords[3], delta[3], loop_var)
        addr_delta = cords2addr(list(delta))
        addr = _addr_expr(node.address_recipe.address, addr_delta, loop_var)
        if node.opcode_name == "OP_ALLOC_TMA_LOAD_1D":
            call = f"compiled_load_1d(cmd, {addr}, {node.size}, smem_base, m2c);"
        elif node.opcode_name == "OP_ALLOC_TMA_LOAD_TENSOR_1D":
            call = f"compiled_tma_load_tensor_1d(cmd, tma_descs, {node.arg}, {addr}, {node.size}, smem_base, m2c);"
        elif node.opcode_name == "OP_ALLOC_TMA_LOAD_2D":
            call = f"compiled_tma_load_2d(cmd, tma_descs, {node.arg}, {node.size}, {c0}, {c1}, smem_base, m2c);"
        elif node.opcode_name == "OP_ALLOC_TMA_LOAD_3D":
            call = f"compiled_tma_load_3d(cmd, tma_descs, {node.arg}, {node.size}, {c0}, {c1}, {c2}, smem_base, m2c);"
        elif node.opcode_name == "OP_ALLOC_TMA_LOAD_4D":
            call = f"compiled_tma_load_4d(cmd, tma_descs, {node.arg}, {node.size}, {c0}, {c1}, {c2}, {c3}, smem_base, m2c);"
        else:
            raise CompileModeError(f"Unsupported direct LDU opcode {node.opcode_name}")
        return [
            f"cmd = ldq[{node.port}].pop();",
            call,
        ]

    if unit == "stu":
        if node.queue_role != "store":
            return []
        c0 = _coord_expr(node.address_recipe.coords[0], delta[0], loop_var)
        c1 = _coord_expr(node.address_recipe.coords[1], delta[1], loop_var)
        addr_delta = cords2addr(list(delta))
        addr = _addr_expr(node.address_recipe.address, addr_delta, loop_var)
        if node.opcode_name == "OP_ALLOC_WB_TMA_STORE_1D":
            return [f"compiled_store_1d(c2m, {addr}, {node.size}, smem_base);"]
        if node.opcode_name == "OP_ALLOC_WB_TMA_STORE_2D":
            return [f"compiled_tma_store_2d(c2m, tma_descs, {node.arg}, {c0}, {c1}, smem_base);"]
        if node.opcode_name == "OP_ALLOC_WB_TMA_REDUCE_ADD_2D":
            return [f"compiled_tma_reduce_add_2d(c2m, tma_descs, {node.arg}, {c0}, {c1}, smem_base);"]
        raise CompileModeError(f"Unsupported direct STU opcode {node.opcode_name}")

    raise ValueError(f"Unknown unit {unit}")


def _emit_memory_unit(program: SMProgramIR, unit: str, port: int | None = None, enable_store_mask: list[bool] | None = None) -> list[str]:
    lines: list[str] = []
    loop_counter = 0
    store_flat_index = 0

    for node in program.memory_ops:
        if isinstance(node, RepeatRegionIR):
            loop_var = f"repeat_idx_{loop_counter}"
            loop_counter += 1
            inner: list[str] = []
            for body_index, body in enumerate(node.body):
                if unit == "ldu" and body.port != port:
                    if body.queue_role == "store":
                        store_flat_index += 1
                    continue
                delta = _repeat_delta_for_index(node, body_index)
                if unit == "stu" and body.queue_role == "store":
                    enabled = enable_store_mask[store_flat_index] if enable_store_mask is not None else True
                    store_flat_index += 1
                    if not enabled:
                        continue
                inner.extend(_memory_op_code(body, loop_var, delta, unit))
            if inner:
                lines.append(f"for (int {loop_var} = 0; {loop_var} < {node.count}; ++{loop_var}) {{")
                lines.extend(_indent(inner))
                lines.append("}")
            continue

        if isinstance(node, MemoryOpIR):
            if unit == "ldu" and node.port != port:
                if node.queue_role == "store":
                    store_flat_index += 1
                continue
            if unit == "stu" and node.queue_role == "store":
                enabled = enable_store_mask[store_flat_index] if enable_store_mask is not None else True
                store_flat_index += 1
                if not enabled:
                    continue
            lines.extend(_memory_op_code(node, None, (0, 0, 0, 0), unit))
    return lines


def _store_writeback_mask(program: SMProgramIR) -> list[bool]:
    mask: list[bool] = []
    store_count = 0
    for node in program.memory_ops:
        if isinstance(node, RepeatRegionIR):
            for body in node.body:
                if body.queue_role == "store":
                    store_count += 1
        elif isinstance(node, MemoryOpIR) and node.queue_role == "store":
            store_count += 1

    if store_count == 0:
        return []

    compute_names = [
        node.opcode_name
        for node in program.compute_ops
        if isinstance(node, (ComputeOpIR, LoopCIR, TerminateComputeIR))
    ]
    if any(name.startswith("OP_GEMV_WGMMA__") for name in compute_names):
        mask = [False] * store_count
        mask[-1] = True
        return mask
    if "OP_COPY" in compute_names:
        return [True] * store_count
    return [False] * store_count


def _emit_compute_op(node: ComputeIRNode) -> list[str]:
    if isinstance(node, TerminateComputeIR):
        return [
            "if (thread_id == 0) {",
            "  int event_base = compiled_sm_id * numProfileEvents;",
            "  g_events[event_base + 1] = cuda::ptx::get_sreg_globaltimer();",
            "}",
            "return;",
        ]

    if isinstance(node, LoopCIR):
        raise CompileModeError("Direct compile runtime does not yet support LoopC")

    opcode_name = node.opcode_name
    if opcode_name == "OP_DUMMY":
        return [
            f"for (int i = 0; i < {node.args[0]}; ++i) {{",
            "  auto slot_id = m2c.pop();",
            f"  __nanosleep({node.args[1] if len(node.args) > 1 else 0});",
            "  c2m.push(thread_id, slot_id);",
            "}",
        ]
    if opcode_name == "OP_COPY":
        return [
            f"for (int i = 0; i < {node.args[0]}; ++i) {{",
            "  auto read_slot = m2c.pop();",
            "  uint32_t *read_data = (uint32_t *)get_slot_address(smem_base, extract(read_slot));",
            "  auto write_slot = m2c.pop();",
            "  uint32_t *write_data = (uint32_t *)get_slot_address(smem_base, extract(write_slot));",
            f"  for (int j = thread_id; j < {node.args[1]}; j += 128) {{",
            "    write_data[j] = read_data[j];",
            "  }",
            "  c2m.template push<0, true>(thread_id, write_slot);",
            "  c2m.push(thread_id, read_slot);",
            "}",
        ]

    family = family_spec_by_name(opcode_name)
    if family is None or family["family"] != "gemv_wgmma":
        raise CompileModeError(f"Unsupported direct compute op {opcode_name}")
    residual = "true" if int(family["residual"]) else "false"
    return [
        "using compiled_gemv_atom = cute::SM90_64x8x16_F32BF16BF16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;",
        (
            f"task_gemv<compiled_gemv_atom, {family['m']}, {family['k']}, {family['bload']}, {residual}>"
            f"({node.args[0]}, {node.args[1]}, smem_base, m2c, c2m);"
        ),
    ]


def _emit_compute_unit(program: SMProgramIR) -> list[str]:
    lines: list[str] = []
    for node in program.compute_ops:
        lines.extend(_emit_compute_op(node))
    return lines


def _emit_sm_functions(program: SMProgramIR) -> list[str]:
    sm_id = program.sm_id
    store_mask = _store_writeback_mask(program)
    alloc_lines = _emit_memory_unit(program, "alloc")
    ldu0_lines = _emit_memory_unit(program, "ldu", port=0)
    ldu1_lines = _emit_memory_unit(program, "ldu", port=1)
    stu_lines = _emit_memory_unit(program, "stu", enable_store_mask=store_mask)
    compute_lines = _emit_compute_unit(program)
    uses_ldu0 = bool(ldu0_lines)
    uses_ldu1 = bool(ldu1_lines)

    lines = [
        f"template <typename M2CQueue, typename C2MQueue, typename LDQueue>",
        f"__device__ __forceinline__ void compiled_alloc_sm_{sm_id}(",
        "    int lane_id,",
        "    int *slot_avail,",
        "    M2CQueue &m2c,",
        "    C2MQueue &c2m,",
        "    LDQueue ldq[2]) {",
        "  (void)c2m;",
        "  SharedMemoryAllocator<numSlots> alloc;",
    ]
    lines.extend(_indent(alloc_lines))
    if uses_ldu0 or uses_ldu1:
        lines.append("  if (lane_id == 0) {")
        if uses_ldu0:
            lines.append("    ldq[0].push(CompiledLdCmd{0, 0});")
        if uses_ldu1:
            lines.append("    ldq[1].push(CompiledLdCmd{0, 0});")
        lines.append("  }")
    lines.extend(
        [
            "}",
            "",
            f"template <typename M2CQueue, typename LDQueue>",
            f"__device__ __forceinline__ void compiled_ldu_sm_{sm_id}_0(",
            "    const void *smem_base,",
            "    const CUtensorMap *tma_descs,",
            "    M2CQueue &m2c,",
            "    LDQueue ldq[2]) {",
        ]
    )
    if uses_ldu0:
        lines.append("  CompiledLdCmd cmd{};")
        lines.extend(_indent(ldu0_lines))
    else:
        lines.extend(
            [
                "  (void)smem_base;",
                "  (void)tma_descs;",
                "  (void)m2c;",
                "  (void)ldq;",
            ]
        )
    lines.extend(
        [
            "}",
            "",
            f"template <typename M2CQueue, typename LDQueue>",
            f"__device__ __forceinline__ void compiled_ldu_sm_{sm_id}_1(",
            "    const void *smem_base,",
            "    const CUtensorMap *tma_descs,",
            "    M2CQueue &m2c,",
            "    LDQueue ldq[2]) {",
        ]
    )
    if uses_ldu1:
        lines.append("  CompiledLdCmd cmd{};")
        lines.extend(_indent(ldu1_lines))
    else:
        lines.extend(
            [
                "  (void)smem_base;",
                "  (void)tma_descs;",
                "  (void)m2c;",
                "  (void)ldq;",
            ]
        )
    lines.extend(
        [
            "}",
            "",
            f"template <typename C2MQueue>",
            f"__device__ __forceinline__ void compiled_stu_sm_{sm_id}(",
            "    const void *smem_base,",
            "    const CUtensorMap *tma_descs,",
            "    C2MQueue &c2m) {",
        ]
    )
    lines.extend(_indent(stu_lines))
    lines.extend(
        [
            "}",
            "",
            f"template <typename M2CQueue, typename C2MQueue>",
            f"__device__ __forceinline__ void compiled_compute_sm_{sm_id}(",
            "    int compiled_sm_id,",
            "    int thread_id,",
            "    void *smem_base,",
            "    uint64_t *g_events,",
            "    M2CQueue &m2c,",
            "    C2MQueue &c2m) {",
        ]
    )
    lines.extend(_indent(compute_lines))
    lines.extend(
        [
            "}",
            "",
        ]
    )
    return lines


def _emit_kernel_body_wrapper(programs: tuple[SMProgramIR, ...]) -> list[str]:
    sm_entries = []
    smem_attr_entries = []
    stream_decl_entries = []
    stream_setup_entries = []
    launch_entries = []
    sync_entries = []
    destroy_entries = []
    for program in programs:
        sm_id = program.sm_id
        sm_entries.extend(_emit_sm_functions(program))
        sm_entries.extend(
            [
                f"__global__ void dae_compiled_sm_kernel_{sm_id}(",
                "    const CUtensorMap *__restrict__ tma_descs,",
                "    int *__restrict__ bars,",
                "    uint64_t *__restrict__ g_events) {",
                "  int thread_id = threadIdx.x;",
                "  int warp_id = (thread_id % 128) / 32;",
                "  int lane_id = thread_id % 32;",
                "",
                "  constexpr int numQueueElements = 32;",
                "  __shared__ int slot_avail;",
                "  if (thread_id == 0) {",
                f"    int event_base = {sm_id} * numProfileEvents;",
                "    g_events[event_base + 0] = cuda::ptx::get_sreg_globaltimer();",
                "    slot_avail = (1U << numSlots) - 1;",
                "  }",
                "  #pragma nv_diag_suppress static_var_with_dynamic_init",
                "  __shared__ cuda::barrier<cuda::thread_scope_block> barriers[4][numQueueElements];",
                "  if (threadIdx.x < numQueueElements) {",
                "    init(&barriers[0][threadIdx.x], numThreadsM2CBarrier);",
                "    init(&barriers[1][threadIdx.x], numThreadsC2MBarrier);",
                "    init(&barriers[2][threadIdx.x], numThreadsLDBarrier);",
                "    init(&barriers[3][threadIdx.x], numThreadsLDBarrier);",
                "  }",
                "  __shared__ int m2c_data[numQueueElements];",
                "  __shared__ int c2m_data[numQueueElements];",
                "  __shared__ CompiledLdCmd ldq_data[2][numQueueElements];",
                "  SizeBoundedBarrierQueue<int, numQueueElements> m2c{ .barriers = barriers[0], .data = m2c_data, .ptr = 0 };",
                "  SizeBoundedBarrierAllocQueue<numQueueElements> c2m{ barriers[1], c2m_data, 0, &slot_avail };",
                "  SizeBoundedBarrierQueue<CompiledLdCmd, numQueueElements> ldq[2] = {",
                "    { .barriers = barriers[2], .data = ldq_data[0], .ptr = 0 },",
                "    { .barriers = barriers[3], .data = ldq_data[1], .ptr = 0 },",
                "  };",
                "  extern __shared__ uint8_t shared_mem[];",
                "  void *smem_base = compiled_align_to((void*)shared_mem, 1024);",
                "  __syncthreads();",
                "  if (threadIdx.x < numComputeWarps * 32) {",
                f"    compiled_compute_sm_{sm_id}({sm_id}, thread_id, smem_base, g_events, m2c, c2m);",
                "    return;",
                "  }",
                "  if (warp_id == 0) {",
                f"    compiled_alloc_sm_{sm_id}(lane_id, &slot_avail, m2c, c2m, ldq);",
                "  } else if (warp_id == 1) {",
                "    if (lane_id == 0) {",
                f"      compiled_stu_sm_{sm_id}(smem_base, tma_descs, c2m);",
                "    }",
                "  } else if (warp_id == 2) {",
                "    if (lane_id == 0) {",
                f"      compiled_ldu_sm_{sm_id}_0(smem_base, tma_descs, m2c, ldq);",
                "    }",
                "  } else if (warp_id == 3) {",
                "    if (lane_id == 0) {",
                f"      compiled_ldu_sm_{sm_id}_1(smem_base, tma_descs, m2c, ldq);",
                "    }",
                "  }",
                "}",
                "",
            ]
        )
        smem_attr_entries.extend(
            [
                f"  err = cudaFuncSetAttribute(",
                f"      dae_compiled_sm_kernel_{sm_id},",
                "      cudaFuncAttributeMaxDynamicSharedMemorySize,",
                "      static_cast<int>(smem_size));",
                "  if (err != cudaSuccess) {",
                "    return err;",
                "  }",
            ]
        )
        launch_entries.append(
            f"  dae_compiled_sm_kernel_{sm_id}<<<1, numThreads, smem_size, launch_stream_{sm_id}>>>(tma_descs, bars, profile);"
        )
        stream_decl_entries.append(f"  cudaStream_t launch_stream_{sm_id} = nullptr;")
        stream_setup_entries.extend(
            [
                f"  err = cudaStreamCreateWithFlags(&launch_stream_{sm_id}, cudaStreamNonBlocking);",
                "  if (err != cudaSuccess) {",
                "    goto cleanup;",
                "  }",
                f"  err = cudaStreamWaitEvent(launch_stream_{sm_id}, launch_ready, 0);",
                "  if (err != cudaSuccess) {",
                "    goto cleanup;",
                "  }",
            ]
        )
        sync_entries.extend(
            [
                f"  if (launch_stream_{sm_id} != nullptr) {{",
                f"    cudaError_t stream_err = cudaStreamSynchronize(launch_stream_{sm_id});",
                "    if (err == cudaSuccess && stream_err != cudaSuccess) {",
                "      err = stream_err;",
                "    }",
                "  }",
            ]
        )
        destroy_entries.extend(
            [
                f"  if (launch_stream_{sm_id} != nullptr) {{",
                f"    cudaStreamDestroy(launch_stream_{sm_id});",
                "  }",
            ]
        )

    wrapper = [
        "static constexpr bool kGeneratedRuntimeAvailable = true;",
        "",
        *sm_entries,
        "static __host__ inline cudaError_t set_generated_compiled_runtime_smem_size(size_t smem_size) {",
        "  cudaError_t err = cudaSuccess;",
        *smem_attr_entries,
        "  return cudaSuccess;",
        "}",
        "",
        "static __host__ inline cudaError_t launch_generated_compiled_runtime(",
        "    int num_sms,",
        "    size_t smem_size,",
        "    CUtensorMap *tma_descs,",
        "    int *bars,",
        "    uint64_t *profile,",
        "    int64_t stream) {",
        f"  if (num_sms != {len(programs)}) {{",
        f'    fprintf(stderr, "compiled runtime expected {len(programs)} SM programs but got %d\\n", num_sms);',
        "    return cudaErrorInvalidValue;",
        "  }",
        "  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);",
        "  cudaError_t err = cudaSuccess;",
        "  cudaEvent_t launch_ready = nullptr;",
        *stream_decl_entries,
        "  err = cudaEventCreateWithFlags(&launch_ready, cudaEventDisableTiming);",
        "  if (err != cudaSuccess) {",
        "    return err;",
        "  }",
        "  err = cudaEventRecord(launch_ready, cuda_stream);",
        "  if (err != cudaSuccess) {",
        "    cudaEventDestroy(launch_ready);",
        "    return err;",
        "  }",
        *stream_setup_entries,
        *launch_entries,
        "  err = cudaGetLastError();",
        "  if (err != cudaSuccess) {",
        "    goto cleanup;",
        "  }",
        *sync_entries,
        "cleanup:",
        "  if (launch_ready != nullptr) {",
        "    cudaEventDestroy(launch_ready);",
        "  }",
        *destroy_entries,
        "  return err == cudaSuccess ? cudaGetLastError() : err;",
        "}",
    ]
    return wrapper


def _render_direct_compiled_program(program: ProgramIR, tag: str) -> str:
    return "\n".join(
        [
            "// Generated by dae.compiler.emit_split_loop_cuda",
            "#pragma once",
            "#include <cstdio>",
            '#include "dae/compiled_runtime_support.cuh"',
            '#include "task/gemv.cuh"',
            "",
            "namespace dae::compiled {",
            f'static constexpr const char *kCompiledProgramTag = "{tag}";',
            "// Generated direct loops: no instruction tables are retained here.",
            "// Address generation stays in the LDU/STU loops rather than ALLOC.",
            "",
            *_emit_kernel_body_wrapper(program.sms),
            "",
            "} // namespace dae::compiled",
            "",
        ]
    )


def emit_split_loop_cuda(
    program: ProgramIR,
    output_path: str | Path,
    runtime_header_path: str | Path | None = None,
) -> tuple[GeneratedCudaSource, GeneratedCudaSource]:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if runtime_header_path is None:
        runtime_header_path = Path("build/generated/dae/generated_compiled_runtime.cuh")
    runtime_path = Path(runtime_header_path)
    runtime_path.parent.mkdir(parents=True, exist_ok=True)

    placeholder = "__dae_compiled__"
    source = _render_direct_compiled_program(program, placeholder)
    tag = hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]
    source = _render_direct_compiled_program(program, tag)

    path.write_text(source, encoding="utf-8")
    runtime_path.write_text(source, encoding="utf-8")
    generated = GeneratedCudaSource(path=path, source=source, tag=tag)
    runtime_generated = GeneratedCudaSource(path=runtime_path, source=source, tag=tag)
    return generated, runtime_generated


class SemanticSimulator:
    def __init__(self, program: ProgramIR):
        self.program = program

    def simulate_memory(self, sm_id: int) -> list[dict[str, object]]:
        sm = self.program.sms[sm_id]
        events: list[dict[str, object]] = []
        for pc, node in enumerate(sm.memory_ops):
            if isinstance(node, MemoryOpIR):
                events.append(
                    {
                        "pc": pc,
                        "kind": node.kind,
                        "opcode": node.opcode_name,
                        "queue_role": node.queue_role,
                        "slot_request": node.slot_request,
                    }
                )
            elif isinstance(node, RepeatRegionIR):
                events.append(
                    {
                        "pc": pc,
                        "kind": node.kind,
                        "trip_count": node.count,
                        "body_ops": tuple(body.opcode_name for body in node.body),
                    }
                )
            else:
                events.append({"pc": pc, "kind": node.kind, "opcode": getattr(node, "opcode_name", node.kind)})
        return events


def compile_builders(
    builders: Iterable[object],
    *,
    mode: str,
    cuda_output_path: str | Path | None = None,
) -> CompileArtifacts:
    original = build_program_ir(builders)
    normalized = normalize_program_ir(original)
    _validate_program(normalized, mode)
    emitted_compute, emitted_memory = emit_program_ir(normalized)

    split_unit_program = None
    generated_cuda = None
    generated_runtime = None
    if mode == "compile_cuda":
        split_unit_program = lower_to_split_units(normalized)
        if cuda_output_path is None:
            cuda_output_path = Path("build/generated/dae_compiled_program.cu")
        generated_cuda, generated_runtime = emit_split_loop_cuda(normalized, cuda_output_path)

    return CompileArtifacts(
        mode=mode,
        original_program=original,
        normalized_program=normalized,
        emitted_compute=emitted_compute,
        emitted_memory=emitted_memory,
        split_unit_program=split_unit_program,
        generated_cuda=generated_cuda,
        generated_runtime=generated_runtime,
    )
