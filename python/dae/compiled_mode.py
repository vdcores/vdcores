from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .instruction_utils import decode_opcode
from .instructions import ComputeInstruction, MemoryInstruction
from .tma_utils import cords2addr


DEFAULT_COMPILED_SPEC_FILE = "dae_compiled_program.vdcore.json"
_MEM_DYNAMIC_FLAG_MASK = 4 | 8 | 16 | 32
_MEM_WRITEBACK_FLAG = 2

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
    "OP_ALLOC_WB_REG_STORE",
    "OP_ALLOC_REG_LOAD",
    "OP_ALLOC_WB_RAW_ADDRESS",
}

_SUPPORTED_COMPUTE_OPS = {
    "OP_DUMMY",
    "OP_COPY",
    "OP_TERMINATEC",
    "OP_SILU_MUL_SHARED_BF16_K_4096_INTER",
    "OP_SILU_MUL_SHARED_BF16_K_64_SW128",
    "OP_RMS_NORM_F16_K_4096",
    "OP_RMS_NORM_F16_K_4096_SMEM",
    "OP_RMS_NORM_F16_K_2048_SMEM",
    "OP_RMS_NORM_F16_K_5120_SMEM",
    "OP_RMS_NORM_F16_K_128_SMEM",
    "OP_ARGMAX_PARTIAL_bf16_1152_50688_132",
    "OP_ARGMAX_REDUCE_bf16_1152_132",
    "OP_ARGMAX_PARTIAL_bf16_1024_65536_128",
    "OP_ARGMAX_REDUCE_bf16_1024_128",
}

_SCALAR_ADDRESS_FIELD_BASE_OPS = {
    "OP_ALLOC_TMA_LOAD_1D",
    "OP_ALLOC_TMA_LOAD_TENSOR_1D",
    "OP_ALLOC_WB_TMA_STORE_1D",
    "OP_ALLOC_WB_RAW_ADDRESS",
}

_NO_ADDRESS_MEMORY_BASE_OPS = {
    "OP_ALLOC_WB_REG_STORE",
    "OP_ALLOC_REG_LOAD",
}

def _memory_base_opcode_name(inst: MemoryInstruction) -> str:
    masked_opcode = inst.opcode & ~_MEM_DYNAMIC_FLAG_MASK
    name = decode_opcode(masked_opcode)
    if name.startswith("UNKNOWN_OPCODE") and (masked_opcode & _MEM_WRITEBACK_FLAG):
        name = decode_opcode(masked_opcode & ~_MEM_WRITEBACK_FLAG)
    return name


def _memory_flags(inst: MemoryInstruction) -> dict[str, bool]:
    return {
        "writeback": bool(inst.opcode & _MEM_WRITEBACK_FLAG),
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


def _st_consumes(base_name: str, flags: dict[str, bool]) -> bool:
    if not flags["writeback"]:
        return False
    return base_name != "OP_ALLOC_WB_REG_STORE"


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
                "args": [int(arg) for arg in inst.args],
            }
        )
    return compute


def _encode_memory_fields(
    inst_index: int,
    inst: MemoryInstruction,
    base_name: str,
    payload_values: list[int],
) -> dict[str, object]:
    encoded = {
        "source_index": inst_index,
        "base_name": base_name,
        "opcode_value": int(inst.opcode),
        "num_slots_value": int(inst.num_slots),
        "nslot": int(inst.num_slots & 0x3F),
        "bar_id": int(inst.num_slots >> 6),
        "arg": int(inst.arg),
        "size": int(inst.size),
        "flags": _memory_flags(inst),
        "writeback": bool(inst.opcode & _MEM_WRITEBACK_FLAG),
        "st_consumes": _st_consumes(base_name, _memory_flags(inst)),
    }

    if base_name in _NO_ADDRESS_MEMORY_BASE_OPS:
        return encoded

    if base_name in _SCALAR_ADDRESS_FIELD_BASE_OPS:
        raw_value = int(cords2addr(inst.cords))
        encoded["address_payload_index"] = len(payload_values)
        payload_values.append(raw_value)
        return encoded

    encoded["coords_payload_index"] = len(payload_values)
    payload_values.extend(int(coord) for coord in inst.cords)
    return encoded


def _make_linear_step(
    inst_index: int,
    inst: MemoryInstruction,
    payload_values: list[int],
) -> dict[str, object]:
    base_name = _memory_base_opcode_name(inst)
    flags = _memory_flags(inst)
    if not _memory_base_is_supported(base_name):
        raise ValueError(f"Compiled mode does not support memory op {base_name} at minst[{inst_index}]")
    if flags["jump"]:
        raise ValueError(f"Compiled mode only supports JUMP inside RepeatM blocks: minst[{inst_index}]")
    return {
        "kind": "op",
        **_encode_memory_fields(inst_index, inst, base_name, payload_values),
    }


def _repeat_delta(seed_inst: MemoryInstruction, base_name: str) -> dict[str, object]:
    if base_name in _SCALAR_ADDRESS_FIELD_BASE_OPS:
        return {"delta_address": int(cords2addr(seed_inst.cords))}
    return {"delta_coords": [int(coord) for coord in seed_inst.cords]}


def _parse_repeat_block(
    minsts: list[MemoryInstruction],
    start: int,
    payload_values: list[int],
) -> tuple[dict[str, object], int]:
    seed_ranges: list[tuple[MemoryInstruction, int, int, int]] = []
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
        seed_ranges.append((inst, cursor, reg_start, reg_end))
        cursor += 1

    if not seed_ranges:
        raise ValueError("internal error: repeat block without seeds")

    count_seed = next((seed for seed in seed_ranges if seed[0].size > 0), None)
    if count_seed is None:
        raise ValueError(f"Compiled mode repeat block at minst[{start}] has no loop-count seed")

    block = {
        "kind": "repeat",
        "count": int(count_seed[0].size),
        "steps": [],
    }

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
        matched_seed = None
        for seed_inst, seed_index, reg_start, reg_end in seed_ranges:
            if reg_start <= repeat_step_index < reg_end:
                matched_seed = (seed_inst, seed_index)
                break
        if matched_seed is None:
            raise ValueError(
                f"RepeatM block starting at minst[{start}] has no delta seed for step {repeat_step_index}"
            )

        seed_inst, seed_index = matched_seed
        step = {
            **_encode_memory_fields(cursor, inst, base_name, payload_values),
            "delta_source_index": seed_index,
            **_repeat_delta(seed_inst, base_name),
        }
        block["steps"].append(step)

        cursor += 1
        repeat_step_index += 1
        if flags["jump"]:
            return block, cursor

    raise ValueError(f"RepeatM block starting at minst[{start}] reached end of stream without a JUMP step")


def _validate_memory_stream(
    minsts: list[MemoryInstruction],
    payload_values: list[int],
) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    cursor = 0
    while cursor < len(minsts):
        inst = minsts[cursor]
        base_name = _memory_base_opcode_name(inst)
        if base_name == "OP_REPEAT":
            block, cursor = _parse_repeat_block(minsts, cursor, payload_values)
            blocks.append(block)
            continue
        if base_name == "OP_TERMINATE":
            flags = _memory_flags(inst)
            if any(flags.values()):
                raise ValueError(f"Compiled mode does not support flags on TerminateM at minst[{cursor}]")
            blocks.append({"kind": "terminate", "source_index": cursor})
            cursor += 1
            continue
        blocks.append(_make_linear_step(cursor, inst, payload_values))
        cursor += 1
    if not blocks or blocks[-1]["kind"] != "terminate":
        raise ValueError("Compiled mode requires a terminating memory instruction")
    return blocks


def _iter_linear_memory_steps(
    blocks: list[dict[str, object]],
) -> list[tuple[int, str, dict[str, object]]]:
    steps: list[tuple[int, str, dict[str, object]]] = []
    order = 0
    for block in blocks:
        if block["kind"] == "repeat":
            for step in block["steps"]:
                steps.append((order, "repeat", step))
                order += 1
            continue
        if block["kind"] == "op":
            steps.append((order, "op", block))
            order += 1
    return steps


def _annotate_reg_pairs(blocks: list[dict[str, object]]) -> dict[str, object]:
    reg_ops: dict[int, list[tuple[int, str, dict[str, object]]]] = {}
    for order, block_kind, step in _iter_linear_memory_steps(blocks):
        base_name = step["base_name"]
        if base_name not in _NO_ADDRESS_MEMORY_BASE_OPS:
            continue
        reg_ops.setdefault(int(step["size"]), []).append((order, block_kind, step))

    reg_pairs: list[dict[str, object]] = []
    for reg_id, entries in reg_ops.items():
        stores = [entry for entry in entries if entry[2]["base_name"] == "OP_ALLOC_WB_REG_STORE"]
        loads = [entry for entry in entries if entry[2]["base_name"] == "OP_ALLOC_REG_LOAD"]
        if len(stores) != 1 or len(loads) != 1:
            continue
        store_order, store_block_kind, store_step = stores[0]
        load_order, load_block_kind, load_step = loads[0]
        if store_block_kind != "op" or load_block_kind != "op":
            continue
        if store_order >= load_order:
            continue
        if bool(store_step["flags"]["port1"]) != bool(load_step["flags"]["port1"]):
            continue
        pair_id = len(reg_pairs)
        store_step["reg_pair_id"] = pair_id
        load_step["reg_pair_id"] = pair_id
        reg_pairs.append(
            {
                "pair_id": pair_id,
                "reg_id": reg_id,
                "port1": bool(store_step["flags"]["port1"]),
            }
        )

    needs_generic_reg_file = any(
        step["base_name"] in _NO_ADDRESS_MEMORY_BASE_OPS and "reg_pair_id" not in step
        for _, _, step in _iter_linear_memory_steps(blocks)
    )
    return {
        "memory": blocks,
        "reg_pairs": reg_pairs,
        "needs_generic_reg_file": needs_generic_reg_file,
    }


def _program_structure(builder) -> tuple[dict[str, object], list[int]]:
    cinsts, minsts = _builder_stream(builder)
    payload_values: list[int] = []
    memory = _validate_memory_stream(minsts, payload_values)
    reg_analysis = _annotate_reg_pairs(memory)
    program = {
        "compute": _validate_compute_stream(cinsts),
        "memory": reg_analysis["memory"],
        "reg_pairs": reg_analysis["reg_pairs"],
        "needs_generic_reg_file": reg_analysis["needs_generic_reg_file"],
        "num_live_values": len(payload_values),
    }
    return program, payload_values


def _canonical_program_key(program: dict[str, object]) -> str:
    return json.dumps(program, sort_keys=True, separators=(",", ":"))


def _analyze_launcher(launcher) -> dict[str, object]:
    programs: list[dict[str, object]] = []
    sm_program_ids: list[int] = []
    sm_live_offsets: list[int] = []
    live_values: list[int] = []
    program_key_to_id: dict[str, int] = {}

    for builder in launcher.builder:
        program, sm_live_values = _program_structure(builder)
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
        sm_live_offsets.append(len(live_values))
        live_values.extend(sm_live_values)

    return {
        "programs": programs,
        "sm_program_ids": sm_program_ids,
        "sm_live_offsets": sm_live_offsets,
        "live_values": live_values,
    }


def build_compiled_runtime_bundle(launcher) -> dict[str, object]:
    analysis = _analyze_launcher(launcher)
    spec = {
        "version": 3,
        "num_sms": launcher.num_sms,
        "compute_ops": launcher.compute_operator_names(),
        "sm_program_ids": analysis["sm_program_ids"],
        "sm_live_offsets": analysis["sm_live_offsets"],
        "num_live_values": len(analysis["live_values"]),
        "programs": analysis["programs"],
    }
    spec["hash"] = compiled_spec_hash(spec)
    return {
        "spec": spec,
        "live_values": analysis["live_values"],
    }


def build_compiled_spec(launcher) -> dict[str, object]:
    return build_compiled_runtime_bundle(launcher)["spec"]


def build_compiled_live_values(launcher) -> list[int]:
    return build_compiled_runtime_bundle(launcher)["live_values"]


def compiled_spec_hash(spec: dict[str, object]) -> str:
    payload = {k: v for k, v in spec.items() if k != "hash"}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_compiled_spec(launcher, path: str = DEFAULT_COMPILED_SPEC_FILE) -> str:
    spec = build_compiled_spec(launcher)
    path_obj = Path(path)
    path_obj.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path_obj)
