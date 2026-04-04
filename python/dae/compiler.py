from __future__ import annotations

from dataclasses import dataclass, replace
import copy
from pathlib import Path
import hashlib
import re
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


def _format_address_recipe(recipe: AddressRecipe) -> str:
    if recipe.kind == "coords":
        active_rank = 4
        while active_rank > 1 and recipe.coords[active_rank - 1] == 0:
            active_rank -= 1
        coords = ", ".join(str(value) for value in recipe.coords[:active_rank])
        return f"coords[{coords}]"
    return f"addr=0x{recipe.address:x}"


def _render_memory_node(node: MemoryIRNode, prefix: str = "") -> list[str]:
    if isinstance(node, MemoryOpIR):
        pieces = [
            f"{prefix}{node.opcode_name}",
            f"queue={node.queue_role}",
            f"size={node.size}",
        ]
        if node.slot_request is not None:
            pieces.append(f"slots={node.slot_request}")
        if node.arg:
            pieces.append(f"arg={node.arg}")
        if node.port:
            pieces.append(f"port={node.port}")
        if node.barrier_id is not None:
            pieces.append(f"bar={node.barrier_id}")
        pieces.append(_format_address_recipe(node.address_recipe))
        return [" ".join(pieces)]

    if isinstance(node, RepeatRegionIR):
        lines = [f"{prefix}repeat x{node.count} delta={node.controls[-1].delta_cords}"]
        for body in node.body:
            lines.extend(_render_memory_node(body, prefix=f"{prefix}  "))
        return lines

    if isinstance(node, LoopMIR):
        return [
            (
                f"{prefix}{node.opcode_name} count={node.count} target_pc={node.target_pc} "
                f"reg={node.reg} bar_shift={node.bar_shift} tma_shift={node.tma_shift}"
            )
        ]

    if isinstance(node, BarrierIssueIR):
        return [f"{prefix}{node.opcode_name} bar={node.barrier_id}"]

    if isinstance(node, RepeatControlIR):
        return [f"{prefix}{node.kind} count={node.count} regs=[{node.reg_start},{node.reg_end}) delta={node.delta_cords}"]

    return [f"{prefix}{node.opcode_name}"]


def _render_compute_node(node: ComputeIRNode, prefix: str = "") -> str:
    if isinstance(node, ComputeOpIR):
        args = ", ".join(str(arg) for arg in node.args)
        return f"{prefix}{node.opcode_name}({args})" if args else f"{prefix}{node.opcode_name}"
    if isinstance(node, LoopCIR):
        return f"{prefix}{node.opcode_name} count={node.count} target_pc={node.target_pc}"
    return f"{prefix}{node.opcode_name}"


def render_program_ir(program: ProgramIR, sm_ids: Iterable[int] | None = None) -> str:
    selected = set(sm_ids) if sm_ids is not None else None
    lines: list[str] = []
    for sm in program.sms:
        if selected is not None and sm.sm_id not in selected:
            continue
        lines.append(f"SM {sm.sm_id}")
        lines.append("  Compute:")
        for node in sm.compute_ops:
            lines.append(_render_compute_node(node, prefix="    "))
        lines.append("  Memory:")
        for node in sm.memory_ops:
            lines.extend(_render_memory_node(node, prefix="    "))
        lines.append("")
    return "\n".join(lines).rstrip()


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
        compute_ops = tuple(
            node
            for node in sm.compute_ops
            if not isinstance(node, LoopCIR) or node.count > 1
        )
        memory_ops = _collapse_repeat_controls(sm.memory_ops)
        memory_ops = _merge_adjacent_repeat_regions(memory_ops)
        memory_ops = tuple(
            node
            for node in memory_ops
            if not isinstance(node, LoopMIR) or node.count > 1
        )
        normalized_sms.append(
            SMProgramIR(
                sm_id=sm.sm_id,
                compute_ops=compute_ops,
                memory_ops=memory_ops,
            )
        )
    return ProgramIR(sms=tuple(normalized_sms))


def _validate_program(program: ProgramIR, mode: str) -> None:
    supported_memory_ops_compile_cuda = {
        "OP_CC0",
        "OP_CC0_ROW_BYTES",
        "OP_ALLOC_REG_LOAD",
        "OP_ALLOC_TMA_LOAD_1D",
        "OP_ALLOC_TMA_LOAD_TENSOR_1D",
        "OP_ALLOC_TMA_LOAD_2D",
        "OP_ALLOC_TMA_LOAD_3D",
        "OP_ALLOC_TMA_LOAD_4D",
        "OP_ALLOC_TMA_LOAD_5D_FIX0",
        "OP_ALLOC_WB_REG_STORE",
        "OP_ALLOC_WB_RAW_ADDRESS",
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
    supported_compute_exact_compile_cuda = {
        "OP_TERMINATEC",
        "OP_DUMMY",
        "OP_COPY",
        "OP_ROPE_INTERLEAVE_512",
        "OP_RMS_NORM_F16_K_128_SMEM",
        "OP_RMS_NORM_F16_K_2048_SMEM",
        "OP_RMS_NORM_F16_K_4096_SMEM",
        "OP_RMS_NORM_F16_K_5120_SMEM",
        "OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim",
        "OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim64",
        "OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim_split",
        "OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim_MMA",
        "OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim64_MMA",
        "OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim_split_MMA",
        "OP_ATTN_SPLIT_POST_REDUCE",
        "OP_SILU_MUL_SHARED_BF16_K_4096_INTER",
        "OP_SILU_MUL_SHARED_BF16_K_64_SW128",
        "OP_ARGMAX_PARTIAL_bf16_1152_50688_132",
        "OP_ARGMAX_REDUCE_bf16_1152_132",
        "OP_ARGMAX_PARTIAL_bf16_1024_65536_128",
        "OP_ARGMAX_REDUCE_bf16_1024_128",
    }
    unsupported_compute: dict[str, set[int]] = {}
    unsupported_memory: dict[str, set[int]] = {}
    unsupported_control: dict[str, set[int]] = {}

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
            unsupported_compute.setdefault(opcode_name, set()).add(sm.sm_id)

        for node in sm.memory_ops:
            if isinstance(node, MemoryOpIR):
                if node.opcode_name.startswith("UNKNOWN_OPCODE"):
                    raise CompileModeError(f"Unsupported memory opcode on SM {sm.sm_id}: {node.opcode_name}")
                if mode == "compile_cuda" and node.opcode_name not in supported_memory_ops_compile_cuda:
                    unsupported_memory.setdefault(node.opcode_name, set()).add(sm.sm_id)
            elif isinstance(node, RepeatRegionIR):
                if mode == "compile_cuda":
                    for body_op in node.body:
                        if body_op.opcode_name not in supported_memory_ops_compile_cuda:
                            unsupported_memory.setdefault(body_op.opcode_name, set()).add(sm.sm_id)
            elif isinstance(node, (BarrierIssueIR, LoopMIR, TerminateMemoryIR)):
                if mode == "compile_cuda" and not isinstance(node, TerminateMemoryIR):
                    unsupported_control.setdefault(node.opcode_name, set()).add(sm.sm_id)
            elif isinstance(node, RepeatControlIR):
                raise CompileModeError(f"Unnormalized repeat control reached validation on SM {sm.sm_id}")

    if mode == "compile_cuda" and (unsupported_compute or unsupported_memory or unsupported_control):
        lines = ["compile_cuda unsupported ops summary:"]
        for opcode_name, sm_ids in sorted(unsupported_compute.items()):
            lines.append(f"  compute {opcode_name}: SMs {sorted(sm_ids)}")
        for opcode_name, sm_ids in sorted(unsupported_memory.items()):
            lines.append(f"  memory {opcode_name}: SMs {sorted(sm_ids)}")
        for opcode_name, sm_ids in sorted(unsupported_control.items()):
            lines.append(f"  control {opcode_name}: SMs {sorted(sm_ids)}")
        raise CompileModeError("\n".join(lines))


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


@dataclass(frozen=True)
class TemplateParamField:
    name: str
    ctype: str


@dataclass(frozen=True)
class SharedProgramTemplate:
    template_id: int
    representative: SMProgramIR
    members: tuple[SMProgramIR, ...]
    shape_key: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class AllocRunEntry:
    kind: str
    slot_request: int
    port: int
    repeat_count: int


def _sanitize_param_prefix(prefix: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", prefix).strip("_")
    return sanitized or "value"


def _format_scalar_literal(value: int, ctype: str) -> str:
    intval = int(value)
    if ctype == "uint64_t":
        return f"{intval}ULL"
    return str(intval)


class _LiteralValueContext:
    def scalar(self, prefix: str, value: int, ctype: str) -> str:
        del prefix
        return _format_scalar_literal(value, ctype)


class _SchemaValueContext:
    def __init__(self):
        self.fields: list[TemplateParamField] = []
        self._counter = 0

    def scalar(self, prefix: str, value: int, ctype: str) -> str:
        del value
        name = f"{_sanitize_param_prefix(prefix)}_{self._counter}"
        self._counter += 1
        self.fields.append(TemplateParamField(name=name, ctype=ctype))
        return f"params.{name}"


class _CollectValueContext:
    def __init__(self, schema: tuple[TemplateParamField, ...]):
        self.schema = schema
        self.values: list[int] = []
        self._index = 0

    def scalar(self, prefix: str, value: int, ctype: str) -> str:
        del prefix
        if self._index >= len(self.schema):
            raise CompileModeError("Template parameter collection overflow")
        field = self.schema[self._index]
        if field.ctype != ctype:
            raise CompileModeError(
                f"Template parameter type mismatch: expected {field.ctype}, got {ctype}"
            )
        self.values.append(int(value))
        self._index += 1
        return _format_scalar_literal(0, ctype)

    def finish(self) -> tuple[int, ...]:
        if self._index != len(self.schema):
            raise CompileModeError(
                f"Template parameter collection incomplete: {self._index} / {len(self.schema)}"
            )
        return tuple(self.values)


def _coord_expr(base_expr: str, terms: Iterable[tuple[str | None, str | None]]) -> str:
    active_terms = [(var, delta_expr) for var, delta_expr in terms if var is not None and delta_expr is not None]
    if not active_terms:
        return base_expr
    expr = base_expr
    for var, delta_expr in active_terms:
        expr = f"{expr} + static_cast<int>({var}) * static_cast<int>({delta_expr})"
    return f"static_cast<uint16_t>({expr})"


def _addr_expr(base_expr: str, terms: Iterable[tuple[str | None, str | None]], extra_terms: Iterable[str] = ()) -> str:
    active_terms = [(var, delta_expr) for var, delta_expr in terms if var is not None and delta_expr is not None]
    active_extra_terms = [term for term in extra_terms if term]
    if not active_terms and not active_extra_terms:
        return base_expr

    expr = base_expr
    for var, delta_expr in active_terms:
        expr = f"{expr} + static_cast<uint64_t>({var}) * static_cast<uint64_t>({delta_expr})"
    for term in active_extra_terms:
        expr = f"{expr} + {term}"
    return f"({expr})"


def _delta_expr(
    value_ctx,
    prefix: str,
    delta: int,
    ctype: str = "int",
) -> str | None:
    if delta == 0:
        return None
    return value_ctx.scalar(prefix, delta, ctype)


def _cc0_offset_expr_parts(opcode_name: str, address_expr: str, arg_expr: str, size_expr: str) -> str:
    token_ptr = f"reinterpret_cast<const int *>({address_expr})"
    if opcode_name == "OP_CC0":
        return f"(static_cast<uint64_t>(*{token_ptr}) << {arg_expr})"
    if opcode_name == "OP_CC0_ROW_BYTES":
        return f"(static_cast<uint64_t>(*{token_ptr}) * static_cast<uint64_t>({size_expr}))"
    raise CompileModeError(f"Unsupported CC0 control opcode {opcode_name}")


def _memory_op_code(
    node: MemoryOpIR,
    unit: str,
    outer_loop_var: str | None = None,
    outer_delta: tuple[int, int, int, int] = (0, 0, 0, 0),
    inner_loop_var: str | None = None,
    inner_delta: tuple[int, int, int, int] | int | None = None,
    extra_addr_terms: tuple[str, ...] = (),
    value_ctx=None,
) -> list[str]:
    if value_ctx is None:
        value_ctx = _LiteralValueContext()

    slot_request_expr = value_ctx.scalar(f"{unit}_{node.opcode_name}_slot_request", node.slot_request or 0, "int")
    arg_expr = value_ctx.scalar(f"{unit}_{node.opcode_name}_arg", node.arg, "int")
    size_expr = value_ctx.scalar(f"{unit}_{node.opcode_name}_size", node.size, "int")
    addr_base_expr = value_ctx.scalar(f"{unit}_{node.opcode_name}_address", node.address_recipe.address, "uint64_t")
    barrier_expr = None
    if node.barrier_id is not None:
        barrier_expr = value_ctx.scalar(f"{unit}_{node.opcode_name}_barrier", node.barrier_id, "int")
    outer_coord_delta_exprs = tuple(
        _delta_expr(value_ctx, f"{unit}_{node.opcode_name}_outer_delta_{idx}", delta)
        if outer_loop_var is not None
        else None
        for idx, delta in enumerate(outer_delta)
    )
    inner_coord_delta_exprs = (None, None, None, None)
    if isinstance(inner_delta, tuple):
        inner_coord_delta_exprs = tuple(
            _delta_expr(value_ctx, f"{unit}_{node.opcode_name}_inner_delta_{idx}", delta)
            if inner_loop_var is not None
            else None
            for idx, delta in enumerate(inner_delta)
        )

    if unit == "alloc":
        if node.opcode_name in {"OP_ALLOC_WB_RAW_ADDRESS", "OP_ALLOC_REG_LOAD"}:
            return [
                "if (lane_id == 0) {",
                f"  m2c.put({slot_request_expr});",
                f"  ldq[{node.port}].put(CompiledLdCmd{{{slot_request_expr}, static_cast<int>(m2c.ptr)}});",
                "  m2c.advance();",
                f"  ldq[{node.port}].commit();",
                f"  ldq[{node.port}].advance();",
                "}",
            ]
        if node.queue_role == "load":
            return [
                f"compiled_alloc_load(lane_id, alloc, slot_avail, {slot_request_expr}, {node.port}, m2c, ldq);"
            ]
        if node.queue_role == "store":
            return [
                f"compiled_alloc_store(lane_id, alloc, slot_avail, {slot_request_expr}, m2c);"
            ]
        return []

    if unit == "ldu":
        if node.queue_role != "load" and node.opcode_name != "OP_ALLOC_WB_RAW_ADDRESS":
            return []
        if node.port not in (0, 1):
            raise CompileModeError(f"Unsupported LDU port {node.port} for {node.opcode_name}")
        call = ""
        c0 = _coord_expr(
            value_ctx.scalar(f"{unit}_{node.opcode_name}_coord_0", node.address_recipe.coords[0], "int"),
            ((outer_loop_var, outer_coord_delta_exprs[0]), (inner_loop_var, inner_coord_delta_exprs[0])),
        )
        c1 = _coord_expr(
            value_ctx.scalar(f"{unit}_{node.opcode_name}_coord_1", node.address_recipe.coords[1], "int"),
            ((outer_loop_var, outer_coord_delta_exprs[1]), (inner_loop_var, inner_coord_delta_exprs[1])),
        )
        c2 = _coord_expr(
            value_ctx.scalar(f"{unit}_{node.opcode_name}_coord_2", node.address_recipe.coords[2], "int"),
            ((outer_loop_var, outer_coord_delta_exprs[2]), (inner_loop_var, inner_coord_delta_exprs[2])),
        )
        c3 = _coord_expr(
            value_ctx.scalar(f"{unit}_{node.opcode_name}_coord_3", node.address_recipe.coords[3], "int"),
            ((outer_loop_var, outer_coord_delta_exprs[3]), (inner_loop_var, inner_coord_delta_exprs[3])),
        )
        addr_terms = [
            (
                outer_loop_var,
                _delta_expr(
                    value_ctx,
                    f"{unit}_{node.opcode_name}_outer_addr_delta",
                    cords2addr(list(outer_delta)),
                    ctype="uint64_t",
                ),
            ),
        ]
        if isinstance(inner_delta, int):
            addr_terms.append(
                (
                    inner_loop_var,
                    _delta_expr(
                        value_ctx,
                        f"{unit}_{node.opcode_name}_inner_addr_delta",
                        inner_delta,
                        ctype="uint64_t",
                    ),
                )
            )
        addr = _addr_expr(addr_base_expr, addr_terms, extra_terms=extra_addr_terms)
        wait_lines = []
        if barrier_expr is not None:
            wait_lines.append(f"compiled_wait_global_barrier(bars, {barrier_expr});")
        if node.opcode_name == "OP_ALLOC_TMA_LOAD_1D":
            call = f"compiled_load_1d(cmd, {addr}, {size_expr}, smem_base, m2c);"
        elif node.opcode_name == "OP_ALLOC_TMA_LOAD_TENSOR_1D":
            call = f"compiled_tma_load_tensor_1d(cmd, tma_descs, {arg_expr}, {addr}, {size_expr}, smem_base, m2c);"
        elif node.opcode_name == "OP_ALLOC_TMA_LOAD_2D":
            call = f"compiled_tma_load_2d(cmd, tma_descs, {arg_expr}, {size_expr}, {c0}, {c1}, smem_base, m2c);"
        elif node.opcode_name == "OP_ALLOC_TMA_LOAD_3D":
            call = f"compiled_tma_load_3d(cmd, tma_descs, {arg_expr}, {size_expr}, {c0}, {c1}, {c2}, smem_base, m2c);"
        elif node.opcode_name == "OP_ALLOC_TMA_LOAD_4D":
            call = f"compiled_tma_load_4d(cmd, tma_descs, {arg_expr}, {size_expr}, {c0}, {c1}, {c2}, {c3}, smem_base, m2c);"
        elif node.opcode_name == "OP_ALLOC_TMA_LOAD_5D_FIX0":
            call = f"compiled_tma_load_5d_fix0(cmd, tma_descs, {arg_expr}, {size_expr}, {c0}, {c1}, {c2}, {c3}, smem_base, m2c);"
        elif node.opcode_name == "OP_ALLOC_WB_RAW_ADDRESS":
            call = f"compiled_raw_address_ready(cmd, {addr}, st_insts, m2c);"
        elif node.opcode_name == "OP_ALLOC_WB_REG_STORE":
            call = f"compiled_reg_store_ready(cmd, {size_expr}, compiled_reg_file, m2c);"
        elif node.opcode_name == "OP_ALLOC_REG_LOAD":
            call = f"compiled_reg_load_ready(cmd, {size_expr}, compiled_reg_file, m2c);"
        else:
            raise CompileModeError(f"Unsupported direct LDU opcode {node.opcode_name}")
        return [f"cmd = ldq[{node.port}].pop();", *wait_lines, call]

    if unit == "stu":
        if node.queue_role != "store":
            return []
        c0 = _coord_expr(
            value_ctx.scalar(f"{unit}_{node.opcode_name}_coord_0", node.address_recipe.coords[0], "int"),
            ((outer_loop_var, outer_coord_delta_exprs[0]), (inner_loop_var, inner_coord_delta_exprs[0])),
        )
        c1 = _coord_expr(
            value_ctx.scalar(f"{unit}_{node.opcode_name}_coord_1", node.address_recipe.coords[1], "int"),
            ((outer_loop_var, outer_coord_delta_exprs[1]), (inner_loop_var, inner_coord_delta_exprs[1])),
        )
        addr_terms = [
            (
                outer_loop_var,
                _delta_expr(
                    value_ctx,
                    f"{unit}_{node.opcode_name}_outer_addr_delta",
                    cords2addr(list(outer_delta)),
                    ctype="uint64_t",
                ),
            ),
        ]
        if isinstance(inner_delta, int):
            addr_terms.append(
                (
                    inner_loop_var,
                    _delta_expr(
                        value_ctx,
                        f"{unit}_{node.opcode_name}_inner_addr_delta",
                        inner_delta,
                        ctype="uint64_t",
                    ),
                )
            )
        addr = _addr_expr(addr_base_expr, addr_terms, extra_terms=extra_addr_terms)
        lines: list[str]
        if node.opcode_name == "OP_ALLOC_WB_TMA_STORE_1D":
            lines = [f"compiled_store_1d(c2m, {addr}, {size_expr}, smem_base);"]
        elif node.opcode_name == "OP_ALLOC_WB_RAW_ADDRESS":
            raw_barrier_expr = barrier_expr if barrier_expr is not None else value_ctx.scalar(
                f"{unit}_{node.opcode_name}_barrier_none",
                -1,
                "int",
            )
            lines = [f"compiled_raw_address_writeback(c2m, bars, {raw_barrier_expr});"]
        elif node.opcode_name == "OP_ALLOC_WB_TMA_STORE_2D":
            lines = [f"compiled_tma_store_2d(c2m, tma_descs, {arg_expr}, {c0}, {c1}, smem_base);"]
        elif node.opcode_name == "OP_ALLOC_WB_TMA_STORE_3D":
            c2 = _coord_expr(
                value_ctx.scalar(f"{unit}_{node.opcode_name}_coord_2", node.address_recipe.coords[2], "int"),
                ((outer_loop_var, outer_coord_delta_exprs[2]), (inner_loop_var, inner_coord_delta_exprs[2])),
            )
            lines = [f"compiled_tma_store_3d(c2m, tma_descs, {arg_expr}, {c0}, {c1}, {c2}, smem_base);"]
        elif node.opcode_name == "OP_ALLOC_WB_TMA_STORE_4D":
            c2 = _coord_expr(
                value_ctx.scalar(f"{unit}_{node.opcode_name}_coord_2", node.address_recipe.coords[2], "int"),
                ((outer_loop_var, outer_coord_delta_exprs[2]), (inner_loop_var, inner_coord_delta_exprs[2])),
            )
            c3 = _coord_expr(
                value_ctx.scalar(f"{unit}_{node.opcode_name}_coord_3", node.address_recipe.coords[3], "int"),
                ((outer_loop_var, outer_coord_delta_exprs[3]), (inner_loop_var, inner_coord_delta_exprs[3])),
            )
            lines = [f"compiled_tma_store_4d(c2m, tma_descs, {arg_expr}, {c0}, {c1}, {c2}, {c3}, smem_base);"]
        elif node.opcode_name == "OP_ALLOC_WB_TMA_STORE_5D_FIX0":
            c2 = _coord_expr(
                value_ctx.scalar(f"{unit}_{node.opcode_name}_coord_2", node.address_recipe.coords[2], "int"),
                ((outer_loop_var, outer_coord_delta_exprs[2]), (inner_loop_var, inner_coord_delta_exprs[2])),
            )
            c3 = _coord_expr(
                value_ctx.scalar(f"{unit}_{node.opcode_name}_coord_3", node.address_recipe.coords[3], "int"),
                ((outer_loop_var, outer_coord_delta_exprs[3]), (inner_loop_var, inner_coord_delta_exprs[3])),
            )
            lines = [f"compiled_tma_store_5d_fix0(c2m, tma_descs, {arg_expr}, {c0}, {c1}, {c2}, {c3}, smem_base);"]
        elif node.opcode_name == "OP_ALLOC_WB_TMA_REDUCE_ADD_2D":
            lines = [f"compiled_tma_reduce_add_2d(c2m, tma_descs, {arg_expr}, {c0}, {c1}, smem_base);"]
        elif node.opcode_name == "OP_ALLOC_WB_TMA_REDUCE_ADD_3D":
            c2 = _coord_expr(
                value_ctx.scalar(f"{unit}_{node.opcode_name}_coord_2", node.address_recipe.coords[2], "int"),
                ((outer_loop_var, outer_coord_delta_exprs[2]), (inner_loop_var, inner_coord_delta_exprs[2])),
            )
            lines = [f"compiled_tma_reduce_add_3d(c2m, tma_descs, {arg_expr}, {c0}, {c1}, {c2}, smem_base);"]
        else:
            raise CompileModeError(f"Unsupported direct STU opcode {node.opcode_name}")
        if barrier_expr is not None and node.opcode_name != "OP_ALLOC_WB_RAW_ADDRESS":
            lines.append(f"compiled_arrive_global_barrier(bars, {barrier_expr});")
        return lines

    raise ValueError(f"Unknown unit {unit}")


def _alloc_run_kind(node: MemoryOpIR) -> str | None:
    if node.opcode_name in {"OP_ALLOC_WB_RAW_ADDRESS", "OP_ALLOC_REG_LOAD"}:
        return "ready"
    if node.queue_role == "load":
        return "load"
    if node.queue_role == "store":
        return "store"
    return None


def _collect_alloc_run_entries(nodes: list[MemoryOpIR]) -> tuple[AllocRunEntry, ...]:
    entries: list[AllocRunEntry] = []
    index = 0
    while index < len(nodes):
        node = nodes[index]
        kind = _alloc_run_kind(node)
        if kind is None:
            index += 1
            continue

        slot_request = int(node.slot_request or 0)
        port = int(node.port)
        run_length = 1
        while index + run_length < len(nodes):
            cur = nodes[index + run_length]
            if (
                _alloc_run_kind(cur) != kind
                or int(cur.slot_request or 0) != slot_request
                or int(cur.port) != port
            ):
                break
            run_length += 1

        entries.append(
            AllocRunEntry(
                kind=kind,
                slot_request=slot_request,
                port=port,
                repeat_count=run_length,
            )
        )
        index += run_length
    return tuple(entries)


def _alloc_kind_literal(kind: str) -> str:
    kind_map = {
        "load": "CompiledAllocOpKind::Load",
        "store": "CompiledAllocOpKind::Store",
        "ready": "CompiledAllocOpKind::Ready",
    }
    try:
        return kind_map[kind]
    except KeyError as exc:
        raise CompileModeError(f"Unsupported alloc run kind {kind}") from exc


def _emit_alloc_run_entries(
    entries: tuple[AllocRunEntry, ...],
    sequence_id: int,
    value_ctx,
) -> list[str]:
    if not entries:
        return []

    if len(entries) == 1:
        entry = entries[0]
        slot_request_expr = value_ctx.scalar(
            f"alloc_seq_{sequence_id}_{entry.kind}_slot_request",
            entry.slot_request,
            "int",
        )
        port_expr = value_ctx.scalar(
            f"alloc_seq_{sequence_id}_{entry.kind}_port",
            entry.port,
            "int",
        )
        repeat_expr = value_ctx.scalar(
            f"alloc_seq_{sequence_id}_{entry.kind}_repeat",
            entry.repeat_count,
            "int",
        )
        return [
            (
                "compiled_run_alloc_op("
                "lane_id, alloc, slot_avail, "
                f"{_alloc_kind_literal(entry.kind)}, "
                f"static_cast<uint16_t>({slot_request_expr}), "
                f"{port_expr}, {repeat_expr}, m2c, ldq);"
            )
        ]

    kind_literals = ", ".join(_alloc_kind_literal(entry.kind) for entry in entries)
    slot_request_exprs = ", ".join(
        value_ctx.scalar(
            f"alloc_seq_{sequence_id}_{entry.kind}_slot_request_{idx}",
            entry.slot_request,
            "int",
        )
        for idx, entry in enumerate(entries)
    )
    port_exprs = ", ".join(
        value_ctx.scalar(
            f"alloc_seq_{sequence_id}_{entry.kind}_port_{idx}",
            entry.port,
            "int",
        )
        for idx, entry in enumerate(entries)
    )
    repeat_exprs = ", ".join(
        value_ctx.scalar(
            f"alloc_seq_{sequence_id}_{entry.kind}_repeat_{idx}",
            entry.repeat_count,
            "int",
        )
        for idx, entry in enumerate(entries)
    )
    op_var = f"alloc_op_idx_{sequence_id}"
    count_literal = str(len(entries))
    return [
        f"const CompiledAllocOpKind alloc_kind_seq_{sequence_id}[{count_literal}] = {{ {kind_literals} }};",
        f"const int alloc_slot_request_seq_{sequence_id}[{count_literal}] = {{ {slot_request_exprs} }};",
        f"const int alloc_port_seq_{sequence_id}[{count_literal}] = {{ {port_exprs} }};",
        f"const int alloc_repeat_seq_{sequence_id}[{count_literal}] = {{ {repeat_exprs} }};",
        f"for (int {op_var} = 0; {op_var} < {count_literal}; ++{op_var}) {{",
        (
            "  compiled_run_alloc_op("
            "lane_id, alloc, slot_avail, "
            f"alloc_kind_seq_{sequence_id}[{op_var}], "
            f"static_cast<uint16_t>(alloc_slot_request_seq_{sequence_id}[{op_var}]), "
            f"alloc_port_seq_{sequence_id}[{op_var}], "
            f"alloc_repeat_seq_{sequence_id}[{op_var}], m2c, ldq);"
        ),
        "}",
    ]


def _memory_run_key(node: MemoryOpIR, unit: str, extra_addr_terms: tuple[str, ...] = ()) -> tuple[object, ...]:
    if unit == "alloc":
        return ("alloc", node.queue_role, node.slot_request, node.port if node.queue_role == "load" else None)
    return (
        unit,
        node.opcode_name,
        node.queue_role,
        node.slot_request,
        node.arg,
        node.size,
        node.port,
        node.address_recipe.kind,
        extra_addr_terms,
    )


def _memory_run_step(prev: MemoryOpIR, cur: MemoryOpIR, unit: str) -> tuple[int, int, int, int] | int | None:
    if unit == "alloc":
        return 0
    if prev.address_recipe.kind != cur.address_recipe.kind:
        return None
    if prev.address_recipe.kind == "coords":
        return tuple(cur.address_recipe.coords[i] - prev.address_recipe.coords[i] for i in range(4))
    return cur.address_recipe.address - prev.address_recipe.address


def _emit_memory_sequence(
    items: list[tuple[MemoryOpIR, tuple[int, int, int, int], tuple[str, ...]]],
    unit: str,
    outer_loop_var: str | None,
    fold_counter: int,
    value_ctx=None,
) -> tuple[list[str], int]:
    if value_ctx is None:
        value_ctx = _LiteralValueContext()
    lines: list[str] = []
    index = 0

    while index < len(items):
        node, outer_delta, extra_addr_terms = items[index]
        run_length = 1
        run_step: tuple[int, int, int, int] | int | None = None
        run_key = _memory_run_key(node, unit, extra_addr_terms)

        if index + 1 < len(items):
            next_node, next_outer_delta, next_extra_addr_terms = items[index + 1]
            if run_key == _memory_run_key(next_node, unit, next_extra_addr_terms) and outer_delta == next_outer_delta:
                run_step = _memory_run_step(node, next_node, unit)
                if run_step is not None:
                    run_length = 2
                    while index + run_length < len(items):
                        prev_node, _, _ = items[index + run_length - 1]
                        cur_node, cur_outer_delta, cur_extra_addr_terms = items[index + run_length]
                        if run_key != _memory_run_key(cur_node, unit, cur_extra_addr_terms) or outer_delta != cur_outer_delta:
                            break
                        if _memory_run_step(prev_node, cur_node, unit) != run_step:
                            break
                        run_length += 1

        if run_length > 1 and run_step is not None:
            inner_var = f"fold_idx_{fold_counter}"
            fold_counter += 1
            fold_count_expr = value_ctx.scalar(f"{unit}_{node.opcode_name}_fold_count", run_length, "int")
            lines.append(f"for (int {inner_var} = 0; {inner_var} < {fold_count_expr}; ++{inner_var}) {{")
            lines.extend(
                _indent(
                    _memory_op_code(
                        node,
                        unit,
                        outer_loop_var=outer_loop_var,
                        outer_delta=outer_delta,
                        inner_loop_var=inner_var,
                        inner_delta=run_step,
                        extra_addr_terms=extra_addr_terms,
                        value_ctx=value_ctx,
                    )
                )
            )
            lines.append("}")
            index += run_length
            continue

        lines.extend(
            _memory_op_code(
                node,
                unit,
                outer_loop_var=outer_loop_var,
                outer_delta=outer_delta,
                extra_addr_terms=extra_addr_terms,
                value_ctx=value_ctx,
            )
        )
        index += 1

    return lines, fold_counter


def _emit_memory_unit(
    program: SMProgramIR,
    unit: str,
    port: int | None = None,
    enable_store_mask: list[bool] | None = None,
    value_ctx=None,
) -> list[str]:
    if value_ctx is None:
        value_ctx = _LiteralValueContext()
    if unit == "alloc":
        lines: list[str] = []
        sequence_id = 0
        loop_counter = 0
        linear_nodes: list[MemoryOpIR] = []

        def flush_linear_nodes() -> None:
            nonlocal sequence_id
            if not linear_nodes:
                return
            entries = _collect_alloc_run_entries(linear_nodes)
            if entries:
                lines.extend(_emit_alloc_run_entries(entries, sequence_id, value_ctx))
                sequence_id += 1
            linear_nodes.clear()

        for node in program.memory_ops:
            if isinstance(node, RepeatRegionIR):
                flush_linear_nodes()
                entries = _collect_alloc_run_entries(list(node.body))
                if not entries:
                    continue
                loop_var = f"repeat_idx_{loop_counter}"
                loop_counter += 1
                loop_count_expr = value_ctx.scalar(f"alloc_repeat_count_{sequence_id}", node.count, "int")
                lines.append(f"for (int {loop_var} = 0; {loop_var} < {loop_count_expr}; ++{loop_var}) {{")
                lines.extend(_indent(_emit_alloc_run_entries(entries, sequence_id, value_ctx)))
                lines.append("}")
                sequence_id += 1
                continue
            if isinstance(node, MemoryOpIR):
                linear_nodes.append(node)
        flush_linear_nodes()
        return lines

    lines: list[str] = []
    loop_counter = 0
    fold_counter = 0
    store_flat_index = 0
    pending_addr_terms: tuple[str, ...] = ()

    for node in program.memory_ops:
        if isinstance(node, RepeatRegionIR):
            loop_var = f"repeat_idx_{loop_counter}"
            loop_counter += 1
            selected_ops: list[tuple[MemoryOpIR, tuple[int, int, int, int], tuple[str, ...]]] = []
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
                extra_addr_terms = pending_addr_terms if pending_addr_terms and not selected_ops else ()
                if extra_addr_terms:
                    pending_addr_terms = ()
                selected_ops.append((body, delta, extra_addr_terms))
            inner, fold_counter = _emit_memory_sequence(selected_ops, unit, loop_var, fold_counter, value_ctx=value_ctx)
            if inner:
                loop_count_expr = value_ctx.scalar(f"{unit}_repeat_count", node.count, "int")
                lines.append(f"for (int {loop_var} = 0; {loop_var} < {loop_count_expr}; ++{loop_var}) {{")
                lines.extend(_indent(inner))
                lines.append("}")
            continue

        if isinstance(node, MemoryOpIR):
            if node.queue_role == "control":
                if unit == "ldu" and node.opcode_name in {"OP_CC0", "OP_CC0_ROW_BYTES"}:
                    cc0_addr_expr = value_ctx.scalar(f"{unit}_{node.opcode_name}_address", node.address_recipe.address, "uint64_t")
                    cc0_arg_expr = value_ctx.scalar(f"{unit}_{node.opcode_name}_arg", node.arg, "int")
                    cc0_size_expr = value_ctx.scalar(f"{unit}_{node.opcode_name}_size", node.size, "int")
                    pending_addr_terms = (
                        _cc0_offset_expr_parts(node.opcode_name, cc0_addr_expr, cc0_arg_expr, cc0_size_expr),
                    )
                continue
            if unit == "ldu" and node.port != port:
                if node.queue_role == "store":
                    store_flat_index += 1
                continue
            if unit == "stu" and node.queue_role == "store":
                enabled = enable_store_mask[store_flat_index] if enable_store_mask is not None else True
                store_flat_index += 1
                if not enabled:
                    continue
            extra_addr_terms = pending_addr_terms
            if extra_addr_terms:
                pending_addr_terms = ()
            emitted, fold_counter = _emit_memory_sequence(
                [(node, (0, 0, 0, 0), extra_addr_terms)],
                unit,
                None,
                fold_counter,
                value_ctx=value_ctx,
            )
            lines.extend(emitted)
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
    return [True] * store_count


def _emit_fixed_compute_handler_call(node: ComputeOpIR, value_ctx=None) -> list[str]:
    if value_ctx is None:
        value_ctx = _LiteralValueContext()
    padded_args = [
        value_ctx.scalar(f"compute_{node.opcode_name}_arg_{idx}", arg, "int")
        for idx, arg in enumerate(node.args[:3])
    ] + ["0"] * (3 - len(node.args))
    inst_args = [f"static_cast<uint16_t>({arg})" for arg in padded_args[:3]]
    return [
        "{",
        f"CInst inst{{{node.opcode_name}, {{{inst_args[0]}, {inst_args[1]}, {inst_args[2]}}}}};",
        (
            f"dae_compute_handler_{node.opcode_name}("
            "compiled_sm_id, thread_id, compiled_pc, compiled_count, compiled_finish, "
            "inst, smem_base, scratch_space, st_insts, m2c, c2m, g_events);"
        ),
        "}",
    ]


def _emit_compute_op(node: ComputeIRNode, value_ctx=None) -> list[str]:
    if value_ctx is None:
        value_ctx = _LiteralValueContext()
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
    if opcode_name in {
        "OP_ROPE_INTERLEAVE_512",
        "OP_RMS_NORM_F16_K_128_SMEM",
        "OP_RMS_NORM_F16_K_2048_SMEM",
        "OP_RMS_NORM_F16_K_4096_SMEM",
        "OP_RMS_NORM_F16_K_5120_SMEM",
        "OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim",
        "OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim64",
        "OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim_split",
        "OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim_MMA",
        "OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim64_MMA",
        "OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim_split_MMA",
        "OP_ATTN_SPLIT_POST_REDUCE",
        "OP_SILU_MUL_SHARED_BF16_K_4096_INTER",
        "OP_SILU_MUL_SHARED_BF16_K_64_SW128",
        "OP_ARGMAX_PARTIAL_bf16_1152_50688_132",
        "OP_ARGMAX_REDUCE_bf16_1152_132",
        "OP_ARGMAX_PARTIAL_bf16_1024_65536_128",
        "OP_ARGMAX_REDUCE_bf16_1024_128",
    }:
        return _emit_fixed_compute_handler_call(node, value_ctx=value_ctx)
    if opcode_name == "OP_DUMMY":
        loop_count_expr = value_ctx.scalar(f"compute_{opcode_name}_count", node.args[0], "int")
        sleep_expr = value_ctx.scalar(
            f"compute_{opcode_name}_sleep",
            node.args[1] if len(node.args) > 1 else 0,
            "int",
        )
        return [
            f"for (int i = 0; i < {loop_count_expr}; ++i) {{",
            "  auto slot_id = m2c.pop();",
            f"  __nanosleep({sleep_expr});",
            "  c2m.push(thread_id, slot_id);",
            "}",
        ]
    if opcode_name == "OP_COPY":
        copy_count_expr = value_ctx.scalar(f"compute_{opcode_name}_count", node.args[0], "int")
        copy_words_expr = value_ctx.scalar(f"compute_{opcode_name}_words", node.args[1], "int")
        return [
            f"for (int i = 0; i < {copy_count_expr}; ++i) {{",
            "  auto read_slot = m2c.pop();",
            "  uint32_t *read_data = (uint32_t *)get_slot_address(smem_base, extract(read_slot));",
            "  auto write_slot = m2c.pop();",
            "  uint32_t *write_data = (uint32_t *)get_slot_address(smem_base, extract(write_slot));",
            f"  for (int j = thread_id; j < {copy_words_expr}; j += 128) {{",
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
    gemv_count_expr = value_ctx.scalar(f"compute_{opcode_name}_count", node.args[0], "int")
    gemv_flag_expr = value_ctx.scalar(f"compute_{opcode_name}_flag", node.args[1], "int")
    return [
        "using compiled_gemv_atom = cute::SM90_64x8x16_F32BF16BF16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;",
        (
            f"task_gemv<compiled_gemv_atom, {family['m']}, {family['k']}, {family['bload']}, {residual}>"
            f"({gemv_count_expr}, {gemv_flag_expr}, smem_base, m2c, c2m);"
        ),
    ]


def _emit_compute_unit(program: SMProgramIR, value_ctx=None) -> list[str]:
    if value_ctx is None:
        value_ctx = _LiteralValueContext()
    lines: list[str] = [
        "uint32_t compiled_pc = 0;",
        "[[maybe_unused]] uint32_t compiled_count = 0;",
        "[[maybe_unused]] bool compiled_finish = false;",
    ]
    for node in program.compute_ops:
        lines.extend(_emit_compute_op(node, value_ctx=value_ctx))
        if not isinstance(node, TerminateComputeIR):
            lines.append("++compiled_pc;")
    return lines


def _emit_program_units(program: SMProgramIR, value_ctx=None) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    if value_ctx is None:
        value_ctx = _LiteralValueContext()
    store_mask = _store_writeback_mask(program)
    alloc_lines = _emit_memory_unit(program, "alloc", value_ctx=value_ctx)
    ldu0_lines = _emit_memory_unit(program, "ldu", port=0, value_ctx=value_ctx)
    ldu1_lines = _emit_memory_unit(program, "ldu", port=1, value_ctx=value_ctx)
    stu_lines = _emit_memory_unit(program, "stu", enable_store_mask=store_mask, value_ctx=value_ctx)
    compute_lines = _emit_compute_unit(program, value_ctx=value_ctx)
    return alloc_lines, ldu0_lines, ldu1_lines, stu_lines, compute_lines


def _scrub_shape_lines(lines: Iterable[str]) -> tuple[str, ...]:
    return tuple(re.sub(r"\b\d+ULL\b|\b\d+\b", "#", line) for line in lines)


def _shared_template_shape_key(program: SMProgramIR) -> tuple[tuple[str, ...], ...]:
    alloc_lines, ldu0_lines, ldu1_lines, stu_lines, compute_lines = _emit_program_units(program)
    return (
        _scrub_shape_lines(alloc_lines),
        _scrub_shape_lines(ldu0_lines),
        _scrub_shape_lines(ldu1_lines),
        _scrub_shape_lines(stu_lines),
        _scrub_shape_lines(compute_lines),
    )


def _group_shared_program_templates(programs: tuple[SMProgramIR, ...]) -> tuple[SharedProgramTemplate, ...]:
    grouped: dict[tuple[tuple[str, ...], ...], list[SMProgramIR]] = {}
    for program in programs:
        grouped.setdefault(_shared_template_shape_key(program), []).append(program)

    templates: list[SharedProgramTemplate] = []
    for template_id, (shape_key, members) in enumerate(sorted(grouped.items(), key=lambda item: item[1][0].sm_id)):
        ordered_members = tuple(sorted(members, key=lambda program: program.sm_id))
        templates.append(
            SharedProgramTemplate(
                template_id=template_id,
                representative=ordered_members[0],
                members=ordered_members,
                shape_key=shape_key,
            )
        )
    return tuple(templates)


def _format_param_initializer(values: tuple[int, ...], schema: tuple[TemplateParamField, ...]) -> str:
    if not schema:
        return "{0}"
    return "{ " + ", ".join(_format_scalar_literal(value, field.ctype) for value, field in zip(values, schema, strict=True)) + " }"


def _template_param_schema_and_lines(
    template: SharedProgramTemplate,
) -> tuple[tuple[TemplateParamField, ...], tuple[list[str], list[str], list[str], list[str], list[str]]]:
    schema_ctx = _SchemaValueContext()
    alloc_lines, ldu0_lines, ldu1_lines, stu_lines, compute_lines = _emit_program_units(
        template.representative,
        value_ctx=schema_ctx,
    )
    return tuple(schema_ctx.fields), (alloc_lines, ldu0_lines, ldu1_lines, stu_lines, compute_lines)


def _template_param_values(program: SMProgramIR, schema: tuple[TemplateParamField, ...]) -> tuple[int, ...]:
    collector = _CollectValueContext(schema)
    _emit_program_units(program, value_ctx=collector)
    return collector.finish()


def _emit_shared_template_functions(
    template: SharedProgramTemplate,
    schema: tuple[TemplateParamField, ...],
    unit_lines: tuple[list[str], list[str], list[str], list[str], list[str]],
) -> list[str]:
    template_id = template.template_id
    params_name = f"CompiledProgramTemplate_{template_id}Params"
    params_symbol = f"kCompiledProgramTemplate_{template_id}Params"
    sm_ids_symbol = f"kCompiledProgramTemplate_{template_id}SMIds"
    alloc_lines, ldu0_lines, ldu1_lines, stu_lines, compute_lines = unit_lines
    uses_ldu0 = bool(ldu0_lines)
    uses_ldu1 = bool(ldu1_lines)
    param_values = [_template_param_values(program, schema) for program in template.members]
    sm_ids = [program.sm_id for program in template.members]
    field_specs = schema if schema else (TemplateParamField(name="unused", ctype="int"),)
    memory_unit_line_count = len(alloc_lines) + len(ldu0_lines) + len(ldu1_lines) + len(stu_lines)
    memory_helper_qualifier = "__device__ __forceinline__" if memory_unit_line_count <= 24 else "__device__ __noinline__"

    lines = [
        f"struct {params_name} {{",
        *[f"  {field.ctype} {field.name};" for field in field_specs],
        "};",
        "",
        f"static __device__ const {params_name} {params_symbol}[{len(template.members)}] = {{",
        *[
            f"  {_format_param_initializer(values, schema)},"
            for values in param_values
        ],
        "};",
        "",
        f"static __device__ const int {sm_ids_symbol}[{len(template.members)}] = {{",
        f"  {', '.join(str(sm_id) for sm_id in sm_ids)}",
        "};",
        "",
        f"struct CompiledProgramTemplate_{template_id} {{",
        f"  using Params = {params_name};",
        "  __device__ __forceinline__ static const Params &params(int program_index) {",
        f"    return {params_symbol}[program_index];",
        "  }",
        "  __device__ __forceinline__ static int sm_id(int program_index) {",
        f"    return {sm_ids_symbol}[program_index];",
        "  }",
        "};",
        "",
        "template <typename M2CQueue, typename C2MQueue, typename LDQueue>",
        f"{memory_helper_qualifier} void compiled_alloc_template_{template_id}(",
        f"    const {params_name} &params,",
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
            "template <typename M2CQueue, typename LDQueue>",
            f"{memory_helper_qualifier} void compiled_ldu_template_{template_id}_0(",
            f"    const {params_name} &params,",
            "    const void *smem_base,",
            "    const CUtensorMap *tma_descs,",
            "    int *bars,",
            "    MInst *st_insts,",
            "    int compiled_reg_file[32],",
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
                "  (void)params;",
                "  (void)smem_base;",
                "  (void)tma_descs;",
                "  (void)bars;",
                "  (void)st_insts;",
                "  (void)compiled_reg_file;",
                "  (void)m2c;",
                "  (void)ldq;",
            ]
        )
    lines.extend(
        [
            "}",
            "",
            "template <typename M2CQueue, typename LDQueue>",
            f"{memory_helper_qualifier} void compiled_ldu_template_{template_id}_1(",
            f"    const {params_name} &params,",
            "    const void *smem_base,",
            "    const CUtensorMap *tma_descs,",
            "    int *bars,",
            "    MInst *st_insts,",
            "    int compiled_reg_file[32],",
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
                "  (void)params;",
                "  (void)smem_base;",
                "  (void)tma_descs;",
                "  (void)bars;",
                "  (void)st_insts;",
                "  (void)compiled_reg_file;",
                "  (void)m2c;",
                "  (void)ldq;",
            ]
        )
    lines.extend(
        [
            "}",
            "",
            "template <typename C2MQueue>",
            f"{memory_helper_qualifier} void compiled_stu_template_{template_id}(",
            f"    const {params_name} &params,",
            "    const void *smem_base,",
            "    const CUtensorMap *tma_descs,",
            "    int *bars,",
            "    C2MQueue &c2m) {",
        ]
    )
    if stu_lines:
        lines.extend(_indent(stu_lines))
    else:
        lines.extend(
            [
                "  (void)params;",
                "  (void)smem_base;",
                "  (void)tma_descs;",
                "  (void)bars;",
                "  (void)c2m;",
            ]
        )
    lines.extend(
        [
            "}",
            "",
            "template <typename M2CQueue, typename C2MQueue>",
            f"__device__ __forceinline__ void compiled_compute_template_{template_id}(",
            f"    const {params_name} &params,",
            "    int compiled_sm_id,",
            "    int thread_id,",
            "    void *smem_base,",
            "    uint64_t *scratch_space,",
            "    MInst *st_insts,",
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
            f"struct CompiledProgramTemplateRuntime_{template_id} {{",
            f"  using Params = {params_name};",
            "  __device__ __forceinline__ static const Params &load_params(int program_index) {",
            f"    return CompiledProgramTemplate_{template_id}::params(program_index);",
            "  }",
            "  __device__ __forceinline__ static int load_sm_id(int program_index) {",
            f"    return CompiledProgramTemplate_{template_id}::sm_id(program_index);",
            "  }",
            "  template <typename M2CQueue, typename C2MQueue, typename LDQueue>",
            "  __device__ __forceinline__ static void alloc(",
            "      const Params &params,",
            "      int lane_id,",
            "      int *slot_avail,",
            "      M2CQueue &m2c,",
            "      C2MQueue &c2m,",
            "      LDQueue ldq[2]) {",
            f"    compiled_alloc_template_{template_id}(params, lane_id, slot_avail, m2c, c2m, ldq);",
            "  }",
            "  template <typename M2CQueue, typename LDQueue>",
            "  __device__ __forceinline__ static void ldu0(",
            "      const Params &params,",
            "      const void *smem_base,",
            "      const CUtensorMap *tma_descs,",
            "      int *bars,",
            "      MInst *st_insts,",
            "      int compiled_reg_file[32],",
            "      M2CQueue &m2c,",
            "      LDQueue ldq[2]) {",
            f"    compiled_ldu_template_{template_id}_0(params, smem_base, tma_descs, bars, st_insts, compiled_reg_file, m2c, ldq);",
            "  }",
            "  template <typename M2CQueue, typename LDQueue>",
            "  __device__ __forceinline__ static void ldu1(",
            "      const Params &params,",
            "      const void *smem_base,",
            "      const CUtensorMap *tma_descs,",
            "      int *bars,",
            "      MInst *st_insts,",
            "      int compiled_reg_file[32],",
            "      M2CQueue &m2c,",
            "      LDQueue ldq[2]) {",
            f"    compiled_ldu_template_{template_id}_1(params, smem_base, tma_descs, bars, st_insts, compiled_reg_file, m2c, ldq);",
            "  }",
            "  template <typename C2MQueue>",
            "  __device__ __forceinline__ static void stu(",
            "      const Params &params,",
            "      const void *smem_base,",
            "      const CUtensorMap *tma_descs,",
            "      int *bars,",
            "      C2MQueue &c2m) {",
            f"    compiled_stu_template_{template_id}(params, smem_base, tma_descs, bars, c2m);",
            "  }",
            "  template <typename M2CQueue, typename C2MQueue>",
            "  __device__ __forceinline__ static void compute(",
            "      const Params &params,",
            "      int compiled_sm_id,",
            "      int thread_id,",
            "      void *smem_base,",
            "      uint64_t *scratch_space,",
            "      MInst *st_insts,",
            "      uint64_t *g_events,",
            "      M2CQueue &m2c,",
            "      C2MQueue &c2m) {",
            f"    compiled_compute_template_{template_id}(params, compiled_sm_id, thread_id, smem_base, scratch_space, st_insts, g_events, m2c, c2m);",
            "  }",
            "};",
            "",
        ]
    )
    return lines


def _emit_kernel_body_wrapper(programs: tuple[SMProgramIR, ...]) -> list[str]:
    templates = _group_shared_program_templates(programs)
    template_entries = []
    smem_attr_entries = []
    launch_entries = []
    for template in templates:
        schema, unit_lines = _template_param_schema_and_lines(template)
        template_entries.extend(_emit_shared_template_functions(template, schema, unit_lines))
        smem_attr_entries.extend(
            [
                "  err = cudaFuncSetAttribute(",
                f"      dae_compiled_sm_kernel<CompiledProgramTemplateRuntime_{template.template_id}>,",
                "      cudaFuncAttributeMaxDynamicSharedMemorySize,",
                "      static_cast<int>(smem_size));",
                "  if (err != cudaSuccess) {",
                "    return err;",
                "  }",
            ]
        )
        launch_entries.append(
            "  "
            f"dae_compiled_sm_kernel<CompiledProgramTemplateRuntime_{template.template_id}>"
            f"<<<{len(template.members)}, numThreads, smem_size, cuda_stream>>>"
            f"(tma_descs, bars, profile);"
        )

    wrapper = [
        "static constexpr bool kGeneratedRuntimeAvailable = true;",
        "",
        *template_entries,
        "template <typename Program>",
        "__global__ void dae_compiled_sm_kernel(",
        "    const CUtensorMap *__restrict__ tma_descs,",
        "    int *__restrict__ bars,",
        "    uint64_t *__restrict__ g_events) {",
        "  int program_index = static_cast<int>(blockIdx.x);",
        "  int compiled_sm_id = Program::load_sm_id(program_index);",
        "  int thread_id = threadIdx.x;",
        "  int warp_id = (thread_id % 128) / 32;",
        "  int lane_id = thread_id % 32;",
        "  const auto &params = Program::load_params(program_index);",
        "",
        "  constexpr int numQueueElements = 32;",
        "  __shared__ int slot_avail;",
        "  __shared__ MInst st_insts[numSlots + numSpecialSlots];",
        "  __shared__ uint64_t scratch_space[32];",
        "  __shared__ int compiled_reg_file[32];",
        "  if (thread_id == 0) {",
        "    slot_avail = (1U << numSlots) - 1;",
        "  }",
        "  if (thread_id < numSlots + numSpecialSlots) {",
        "    st_insts[thread_id].opcode = 0;",
        "    st_insts[thread_id].size = 0;",
        "    st_insts[thread_id].num_slots = 0;",
        "    st_insts[thread_id].arg = 0;",
        "    st_insts[thread_id].address = 0;",
        "  }",
        "  if (thread_id < 32) {",
        "    compiled_reg_file[thread_id] = 0;",
        "    scratch_space[thread_id] = 0;",
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
        "  if (thread_id == 0) {",
        "    int event_base = compiled_sm_id * numProfileEvents;",
        "    g_events[event_base + 0] = cuda::ptx::get_sreg_globaltimer();",
        "  }",
        "  if (threadIdx.x < numComputeWarps * 32) {",
        "    Program::compute(params, compiled_sm_id, thread_id, smem_base, scratch_space, st_insts, g_events, m2c, c2m);",
        "    return;",
        "  }",
        "  if (warp_id == 0) {",
        "    Program::alloc(params, lane_id, &slot_avail, m2c, c2m, ldq);",
        "  } else if (warp_id == 1) {",
        "    if (lane_id == 0) {",
        "      Program::stu(params, smem_base, tma_descs, bars, c2m);",
        "    }",
        "  } else if (warp_id == 2) {",
        "    if (lane_id == 0) {",
        "      Program::ldu0(params, smem_base, tma_descs, bars, st_insts, compiled_reg_file, m2c, ldq);",
        "    }",
        "  } else if (warp_id == 3) {",
        "    if (lane_id == 0) {",
        "      Program::ldu1(params, smem_base, tma_descs, bars, st_insts, compiled_reg_file, m2c, ldq);",
        "    }",
        "  }",
        "}",
        "",
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
        *launch_entries,
        "  return cudaGetLastError();",
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
            '#include "dae/compute_dispatch.cuh"',
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
    runtime_header_path: str | Path | None = None,
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
        generated_cuda, generated_runtime = emit_split_loop_cuda(
            normalized,
            cuda_output_path,
            runtime_header_path=runtime_header_path,
        )

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
