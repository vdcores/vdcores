from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .instruction_utils import decode_opcode
from .instructions import ComputeInstruction, MemoryInstruction


DEFAULT_COMPILED_SPEC_FILE = "dae_compiled_program.vdcore.json"
_MEM_DYNAMIC_FLAG_MASK = 4 | 8 | 16 | 32

_SUPPORTED_MEMORY_BASE_OPS = {
    "OP_TERMINATE",
    "OP_REPEAT",
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
}

_SUPPORTED_COMPUTE_OPS = {
    "OP_DUMMY",
    "OP_COPY",
    "OP_TERMINATEC",
}


def _memory_base_opcode_name(inst: MemoryInstruction) -> str:
    return decode_opcode(inst.opcode & ~_MEM_DYNAMIC_FLAG_MASK)


def _memory_flags(inst: MemoryInstruction) -> dict[str, bool]:
    return {
        "jump": bool(inst.opcode & 8),
        "group": bool(inst.opcode & 4),
        "barrier": bool(inst.opcode & 16),
        "port1": bool(inst.opcode & 32),
    }


def _compute_is_supported(name: str) -> bool:
    return (
        name in _SUPPORTED_COMPUTE_OPS
        or name.startswith("OP_GEMV_WGMMA__")
        or name.startswith("OP_GEMV_MMA__")
    )


def _memory_base_is_supported(name: str) -> bool:
    return name in _SUPPORTED_MEMORY_BASE_OPS


def _builder_stream(builder) -> tuple[list[ComputeInstruction], list[MemoryInstruction]]:
    return (
        [*builder.built_cinsts, *builder.cinsts],
        [*builder.built_minsts, *builder.minsts],
    )


def _validate_compute_stream(cinsts: list[ComputeInstruction]) -> list[dict[str, object]]:
    compute = []
    for index, inst in enumerate(cinsts):
        name = inst.compute_operator_name()
        if not _compute_is_supported(name):
            raise ValueError(f"Compiled mode does not support compute op {name} at cinst[{index}]")
        compute.append(
            {
                "index": index,
                "name": name,
                "args_len": len(inst.args),
            }
        )
    return compute


def _make_linear_step(inst_index: int, inst: MemoryInstruction) -> dict[str, object]:
    base_name = _memory_base_opcode_name(inst)
    flags = _memory_flags(inst)
    if not _memory_base_is_supported(base_name):
        raise ValueError(f"Compiled mode does not support memory op {base_name} at minst[{inst_index}]")
    if flags["group"] or flags["barrier"]:
        raise ValueError(
            f"Compiled mode does not support GROUP/BARRIER flags at minst[{inst_index}] ({base_name})"
        )
    if flags["jump"]:
        raise ValueError(f"Compiled mode only supports JUMP inside RepeatM blocks: minst[{inst_index}]")
    if flags["port1"]:
        raise ValueError(f"Compiled mode does not yet support PORT1 loads: minst[{inst_index}]")
    return {
        "kind": "op",
        "inst_index": inst_index,
        "base_name": base_name,
        "writeback": "WB_" in base_name or "_WB_" in base_name or base_name.startswith("OP_ALLOC_WB_"),
    }


def _parse_repeat_block(minsts: list[MemoryInstruction], start: int) -> tuple[dict[str, object], int]:
    seed_indices: list[int] = []
    seed_ranges: list[tuple[int, int, int]] = []
    cursor = start
    while cursor < len(minsts):
        inst = minsts[cursor]
        if _memory_base_opcode_name(inst) != "OP_REPEAT":
            break
        flags = _memory_flags(inst)
        if any(flags.values()):
            raise ValueError(f"Compiled mode does not support flags on OP_REPEAT at minst[{cursor}]")
        reg_start = inst.num_slots & 0xFF
        reg_end = inst.num_slots >> 8
        seed_indices.append(cursor)
        seed_ranges.append((cursor, reg_start, reg_end))
        cursor += 1

    if not seed_indices:
        raise ValueError("internal error: repeat block without seeds")

    count_seed_index = next((idx for idx in seed_indices if minsts[idx].size > 0), None)
    if count_seed_index is None:
        raise ValueError(f"Compiled mode repeat block at minst[{start}] has no loop-count seed")

    steps: list[dict[str, object]] = []
    repeat_step_index = 0
    while cursor < len(minsts):
        inst = minsts[cursor]
        base_name = _memory_base_opcode_name(inst)
        flags = _memory_flags(inst)
        if base_name == "OP_REPEAT":
            raise ValueError(f"Compiled mode does not support nested RepeatM blocks near minst[{cursor}]")
        if base_name == "OP_TERMINATE":
            raise ValueError(f"RepeatM block starting at minst[{start}] is missing a JUMP-marked final step")
        if not inst.opcode & 1:
            raise ValueError(
                f"Compiled mode RepeatM blocks may only contain allocating ops, got {base_name} at minst[{cursor}]"
            )
        if flags["group"] or flags["barrier"] or flags["port1"]:
            raise ValueError(
                f"Compiled mode RepeatM blocks do not support GROUP/BARRIER/PORT1 at minst[{cursor}] ({base_name})"
            )
        delta_seed_index = None
        for seed_index, reg_start, reg_end in seed_ranges:
            if reg_start <= repeat_step_index < reg_end:
                delta_seed_index = seed_index
                break
        if delta_seed_index is None:
            raise ValueError(
                f"RepeatM block starting at minst[{start}] has no delta seed for step {repeat_step_index}"
            )
        steps.append(
            {
                "inst_index": cursor,
                "base_name": base_name,
                "delta_seed_index": delta_seed_index,
                "writeback": "WB_" in base_name or "_WB_" in base_name or base_name.startswith("OP_ALLOC_WB_"),
            }
        )
        cursor += 1
        repeat_step_index += 1
        if flags["jump"]:
            return (
                {
                    "kind": "repeat",
                    "seed_indices": seed_indices,
                    "count_seed_index": count_seed_index,
                    "steps": steps,
                },
                cursor,
            )

    raise ValueError(f"RepeatM block starting at minst[{start}] reached end of stream without a JUMP step")


def _validate_memory_stream(minsts: list[MemoryInstruction]) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    cursor = 0
    while cursor < len(minsts):
        inst = minsts[cursor]
        base_name = _memory_base_opcode_name(inst)
        if base_name == "OP_REPEAT":
            block, cursor = _parse_repeat_block(minsts, cursor)
            blocks.append(block)
            continue
        if base_name == "OP_TERMINATE":
            flags = _memory_flags(inst)
            if any(flags.values()):
                raise ValueError(f"Compiled mode does not support flags on TerminateM at minst[{cursor}]")
            blocks.append({"kind": "terminate", "inst_index": cursor})
            cursor += 1
            continue
        blocks.append(_make_linear_step(cursor, inst))
        cursor += 1
    if not blocks or blocks[-1]["kind"] != "terminate":
        raise ValueError("Compiled mode requires a terminating memory instruction")
    return blocks


def _program_structure(builder) -> dict[str, object]:
    cinsts, minsts = _builder_stream(builder)
    return {
        "compute": _validate_compute_stream(cinsts),
        "memory": _validate_memory_stream(minsts),
    }


def _canonical_program_key(program: dict[str, object]) -> str:
    return json.dumps(program, sort_keys=True, separators=(",", ":"))


def build_compiled_spec(launcher) -> dict[str, object]:
    programs: list[dict[str, object]] = []
    sm_program_ids: list[int] = []
    program_key_to_id: dict[str, int] = {}
    for builder in launcher.builder:
        program = _program_structure(builder)
        key = _canonical_program_key(program)
        program_id = program_key_to_id.get(key)
        if program_id is None:
            program_id = len(programs)
            program_key_to_id[key] = program_id
            programs.append(
                {
                    "program_id": program_id,
                    **program,
                }
            )
        sm_program_ids.append(program_id)

    spec = {
        "version": 1,
        "num_sms": launcher.num_sms,
        "compute_ops": launcher.compute_operator_names(),
        "sm_program_ids": sm_program_ids,
        "programs": programs,
    }
    spec["hash"] = compiled_spec_hash(spec)
    return spec


def compiled_spec_hash(spec: dict[str, object]) -> str:
    payload = {k: v for k, v in spec.items() if k != "hash"}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_compiled_spec(launcher, path: str = DEFAULT_COMPILED_SPEC_FILE) -> str:
    spec = build_compiled_spec(launcher)
    path_obj = Path(path)
    path_obj.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path_obj)

