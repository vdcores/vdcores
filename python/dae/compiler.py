from __future__ import annotations

from dataclasses import dataclass, replace
import copy
from pathlib import Path
from typing import Iterable

from .instruction_utils import decode_opcode
from .instructions import ComputeInstruction, MemoryInstruction


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


@dataclass(frozen=True)
class CompileArtifacts:
    mode: str
    original_program: ProgramIR
    normalized_program: ProgramIR
    emitted_compute: tuple[tuple[ComputeInstruction, ...], ...]
    emitted_memory: tuple[tuple[MemoryInstruction, ...], ...]
    generated_cuda: GeneratedCudaSource | None = None


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
            address=int(getattr(inst, "address", 0)),
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
    supported_compute_exact_compile_cuda = {"OP_TERMINATEC"}

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


def _format_cpp_array(name: str, rows: list[list[str]]) -> str:
    rendered_rows = []
    for row in rows:
        rendered_rows.append("  { " + ", ".join(row) + " }")
    return f"static constexpr const char *{name}[][12] = {{\n" + ",\n".join(rendered_rows) + "\n};"


def emit_split_loop_cuda(program: ProgramIR, output_path: str | Path) -> GeneratedCudaSource:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    alloc_rows: list[list[str]] = []
    ldu_rows: list[list[str]] = []
    stu_rows: list[list[str]] = []
    compute_rows: list[list[str]] = []

    for sm in program.sms:
        alloc_count = 0
        ldu_count = 0
        stu_count = 0
        for node in sm.memory_ops:
            if isinstance(node, RepeatRegionIR):
                alloc_count += sum(1 for body in node.body if body.allocate)
                ldu_count += sum(1 for body in node.body if body.queue_role == "load")
                stu_count += sum(1 for body in node.body if body.queue_role == "store")
            elif isinstance(node, MemoryOpIR):
                alloc_count += 1 if node.allocate else 0
                ldu_count += 1 if node.queue_role == "load" else 0
                stu_count += 1 if node.queue_role == "store" else 0

        alloc_rows.append(
            [
                f"\"sm={sm.sm_id}\"",
                f"\"alloc_ops={alloc_count}\"",
                f"\"memory_nodes={len(sm.memory_ops)}\"",
            ]
        )
        ldu_rows.append(
            [
                f"\"sm={sm.sm_id}\"",
                f"\"ldu_ops={ldu_count}\"",
                "\"addr_gen=inside_ldu_loop\"",
            ]
        )
        stu_rows.append(
            [
                f"\"sm={sm.sm_id}\"",
                f"\"stu_ops={stu_count}\"",
                "\"addr_gen=inside_stu_loop\"",
            ]
        )
        compute_rows.append(
            [
                f"\"sm={sm.sm_id}\"",
                f"\"compute_nodes={len(sm.compute_ops)}\"",
                "\"direct_task_calls=true\"",
            ]
        )

    source = "\n".join(
        [
            "// Generated by dae.compiler.emit_split_loop_cuda",
            "#include <cstdint>",
            "#include <cstdio>",
            "",
            "namespace dae::compiled {",
            "",
            _format_cpp_array("kAllocSummary", alloc_rows),
            "",
            _format_cpp_array("kLduSummary", ldu_rows),
            "",
            _format_cpp_array("kStuSummary", stu_rows),
            "",
            _format_cpp_array("kComputeSummary", compute_rows),
            "",
            "template <typename State>",
            "__device__ __forceinline__ void run_alloc_loop(State &state) {",
            "  for (int sm = 0; sm < state.num_sms; ++sm) {",
            "    // ALLOC owns slot allocation and queue/order state only.",
            "    // It must not dispatch full decoded memory instructions to LDU/STU.",
            "    (void)kAllocSummary[sm];",
            "  }",
            "}",
            "",
            "template <typename State>",
            "__device__ __forceinline__ void run_ldu_loop(State &state) {",
            "  for (int sm = 0; sm < state.num_sms; ++sm) {",
            "    // LDU reconstructs address generation and load opcode execution inside this loop.",
            "    (void)kLduSummary[sm];",
            "  }",
            "}",
            "",
            "template <typename State>",
            "__device__ __forceinline__ void run_stu_loop(State &state) {",
            "  for (int sm = 0; sm < state.num_sms; ++sm) {",
            "    // STU reconstructs address generation and store opcode execution inside this loop.",
            "    (void)kStuSummary[sm];",
            "  }",
            "}",
            "",
            "template <typename State>",
            "__device__ __forceinline__ void run_compute_loop(State &state) {",
            "  for (int sm = 0; sm < state.num_sms; ++sm) {",
            "    // Compute lowers IR nodes to direct task/helper calls.",
            "    (void)kComputeSummary[sm];",
            "  }",
            "}",
            "",
            "} // namespace dae::compiled",
            "",
        ]
    )

    path.write_text(source, encoding="utf-8")
    return GeneratedCudaSource(path=path, source=source)


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

    generated_cuda = None
    if mode == "compile_cuda":
        if cuda_output_path is None:
            cuda_output_path = Path("build/generated/dae_compiled_program.cu")
        generated_cuda = emit_split_loop_cuda(normalized, cuda_output_path)

    return CompileArtifacts(
        mode=mode,
        original_program=original,
        normalized_program=normalized,
        emitted_compute=emitted_compute,
        emitted_memory=emitted_memory,
        generated_cuda=generated_cuda,
    )
