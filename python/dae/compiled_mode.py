from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from .instruction_utils import decode_opcode
from .instructions import ComputeInstruction, MemoryInstruction
from .tma_utils import cords2addr


DEFAULT_COMPILED_SPEC_FILE = "dae_compiled_program.vdcore.json"
_MEM_DYNAMIC_FLAG_MASK = 4 | 8 | 16 | 32
_MEM_WRITEBACK_FLAG = 2
_COORD_DELTA_LIMIT = 0xFFFF
_ADDRESS_STRUCTURAL_DELTA_MULTIPLIER = 64

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

_MEMORY_OP_COORD_COUNTS = {
    "OP_ALLOC_TMA_LOAD_2D": 2,
    "OP_ALLOC_TMA_LOAD_3D": 3,
    "OP_ALLOC_TMA_LOAD_4D": 4,
    "OP_ALLOC_TMA_LOAD_5D_FIX0": 4,
    "OP_ALLOC_WB_TMA_STORE_2D": 2,
    "OP_ALLOC_WB_TMA_STORE_3D": 3,
    "OP_ALLOC_WB_TMA_STORE_4D": 4,
    "OP_ALLOC_WB_TMA_STORE_5D_FIX0": 4,
    "OP_ALLOC_WB_TMA_REDUCE_ADD_2D": 2,
    "OP_ALLOC_WB_TMA_REDUCE_ADD_3D": 3,
}


def _memory_coord_count(base_name: str) -> int:
    return _MEMORY_OP_COORD_COUNTS.get(base_name, 0)


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
        encoded["address_value"] = int(cords2addr(inst.cords))
        return encoded

    coord_count = _memory_coord_count(base_name)
    encoded["coords_values"] = [int(coord) for coord in inst.cords[:coord_count]]
    return encoded


def _make_linear_step(
    inst_index: int,
    inst: MemoryInstruction,
) -> dict[str, object]:
    base_name = _memory_base_opcode_name(inst)
    flags = _memory_flags(inst)
    if not _memory_base_is_supported(base_name):
        raise ValueError(f"Compiled mode does not support memory op {base_name} at minst[{inst_index}]")
    if flags["jump"]:
        raise ValueError(f"Compiled mode only supports JUMP inside RepeatM blocks: minst[{inst_index}]")
    return {
        "kind": "op",
        **_encode_memory_fields(inst_index, inst, base_name),
    }


def _repeat_delta(seed_inst: MemoryInstruction, base_name: str) -> dict[str, object]:
    if base_name in _SCALAR_ADDRESS_FIELD_BASE_OPS:
        return {"delta_address": int(cords2addr(seed_inst.cords))}
    return {"delta_coords": [int(coord) for coord in seed_inst.cords]}


def _parse_repeat_block(
    minsts: list[MemoryInstruction],
    start: int,
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
            **_encode_memory_fields(cursor, inst, base_name),
            "delta_source_index": seed_index,
            **_repeat_delta(seed_inst, base_name),
        }
        block["steps"].append(step)

        cursor += 1
        repeat_step_index += 1
        if flags["jump"]:
            return block, cursor

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
            blocks.append({"kind": "terminate", "source_index": cursor})
            cursor += 1
            continue
        blocks.append(_make_linear_step(cursor, inst))
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


def _program_structure_raw(builder) -> dict[str, object]:
    cinsts, minsts = _builder_stream(builder)
    memory = _validate_memory_stream(minsts)
    reg_analysis = _annotate_reg_pairs(memory)
    return {
        "compute": _validate_compute_stream(cinsts),
        "memory": reg_analysis["memory"],
        "reg_pairs": reg_analysis["reg_pairs"],
        "needs_generic_reg_file": reg_analysis["needs_generic_reg_file"],
    }


def _strip_dynamic_fields(value):
    if isinstance(value, list):
        return [_strip_dynamic_fields(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _strip_dynamic_fields(item)
            for key, item in value.items()
            if key not in {"address_value", "coords_values", "address_source", "coord_sources", "num_live_values"}
        }
    return value


def _canonical_program_template_key(program: dict[str, object]) -> str:
    return json.dumps(_strip_dynamic_fields(program), sort_keys=True, separators=(",", ":"))


def _get_program_step(program: dict[str, object], occurrence: dict[str, object]) -> dict[str, object]:
    block = program["memory"][int(occurrence["block_index"])]
    if block["kind"] == "repeat":
        return block["steps"][int(occurrence["step_index"])]
    return block


def _iter_program_occurrences(program: dict[str, object]) -> list[dict[str, object]]:
    occurrences: list[dict[str, object]] = []
    for block_index, block in enumerate(program["memory"]):
        if block["kind"] not in {"repeat", "op"}:
            continue
        steps = block["steps"] if block["kind"] == "repeat" else [block]
        for step_index, step in enumerate(steps):
            shared = {
                "block_index": block_index,
                "step_index": step_index if block["kind"] == "repeat" else None,
                "base_name": step["base_name"],
                "size": int(step["size"]),
            }
            if "address_value" in step:
                occurrences.append(
                    {
                        **shared,
                        "domain": "address",
                    }
                )
            for coord_index, _ in enumerate(step.get("coords_values", [])):
                occurrences.append(
                    {
                        **shared,
                        "domain": "coord",
                        "coord_index": coord_index,
                    }
                )
    return occurrences


def _occurrence_values(raw_programs: list[dict[str, object]], occurrence: dict[str, object]) -> list[int]:
    values: list[int] = []
    for program in raw_programs:
        step = _get_program_step(program, occurrence)
        if occurrence["domain"] == "address":
            values.append(int(step["address_value"]))
        else:
            values.append(int(step["coords_values"][int(occurrence["coord_index"])]))
    return values


def _all_same(values: list[int]) -> bool:
    return all(value == values[0] for value in values[1:])


def _invariant_delta(values: list[int], base_values: list[int]) -> int | None:
    delta = int(values[0]) - int(base_values[0])
    for value, base_value in zip(values[1:], base_values[1:]):
        if int(value) - int(base_value) != delta:
            return None
    return delta


def _coord_constant_source(values: list[int]) -> dict[str, int] | None:
    if not _all_same(values):
        return None
    value = int(values[0])
    if value == 0 or len(values) > 1:
        return {"kind": "const", "value": value}
    return None


def _coord_sm_affine_const_source(values: list[int], sm_indices: list[int]) -> dict[str, int] | None:
    if len(values) != len(sm_indices) or not values:
        return None
    if len(values) == 1:
        return None

    base_sm = int(sm_indices[0])
    base_value = int(values[0])
    stride: int | None = None
    for sm_index, value in zip(sm_indices[1:], values[1:]):
        sm_delta = int(sm_index) - base_sm
        value_delta = int(value) - base_value
        if sm_delta == 0:
            if value_delta != 0:
                return None
            continue
        if value_delta % sm_delta != 0:
            return None
        current_stride = value_delta // sm_delta
        if stride is None:
            stride = current_stride
        elif current_stride != stride:
            return None

    return {
        "kind": "sm_affine_const",
        "base_sm": base_sm,
        "value": base_value,
        "stride": 0 if stride is None else int(stride),
    }


def _address_constant_source(values: list[int]) -> dict[str, int] | None:
    if _all_same(values) and int(values[0]) == 0:
        return {"kind": "const", "value": 0}
    return None


def _coord_addend_is_structural(delta: int) -> bool:
    return -_COORD_DELTA_LIMIT <= int(delta) <= _COORD_DELTA_LIMIT


def _address_addend_is_structural(delta: int, occurrence: dict[str, object]) -> bool:
    if delta == 0:
        return True
    size = int(occurrence["size"])
    if size <= 0:
        return False
    if delta % size != 0:
        return False
    return abs(delta) <= size * _ADDRESS_STRUCTURAL_DELTA_MULTIPLIER


def _compose_source(
    source: dict[str, int],
    delta: int,
    *,
    domain: str,
) -> dict[str, int] | None:
    if source["kind"] == "const":
        value = int(source["value"]) + int(delta)
        if domain == "coord":
            return {"kind": "const", "value": value}
        if value == 0:
            return {"kind": "const", "value": 0}
        return None
    if source["kind"] == "sm_affine_const":
        value = int(source["value"]) + int(delta)
        return {
            "kind": "sm_affine_const",
            "base_sm": int(source["base_sm"]),
            "value": value,
            "stride": int(source["stride"]),
        }

    add = int(source.get("add", 0)) + int(delta)
    result = {
        "kind": "live",
        "index": int(source["index"]),
    }
    if add != 0:
        result["add"] = add
    return result


def _candidate_score(source: dict[str, int], delta: int) -> tuple[int, int, int]:
    return (
        0 if source["kind"] == "sm_affine_const" else (1 if source["kind"] == "const" else 2),
        abs(int(delta)),
        int(source.get("index", 0)),
    )


def _set_occurrence_source(
    compiled_program: dict[str, object],
    occurrence: dict[str, object],
    source: dict[str, int],
) -> None:
    step = _get_program_step(compiled_program, occurrence)
    if occurrence["domain"] == "address":
        step["address_source"] = source
        step.pop("address_value", None)
        return

    coord_sources = step.setdefault("coord_sources", [None] * len(step.get("coords_values", [])))
    coord_sources[int(occurrence["coord_index"])] = source
    if all(item is not None for item in coord_sources):
        step.pop("coords_values", None)


def _optimize_program_group(
    raw_programs: list[dict[str, object]],
    sm_indices: list[int],
) -> tuple[dict[str, object], list[list[int]]]:
    if not raw_programs:
        raise ValueError("internal error: empty raw program group")

    template = raw_programs[0]
    occurrences = _iter_program_occurrences(template)
    occurrence_values = [_occurrence_values(raw_programs, occurrence) for occurrence in occurrences]
    normalized_sources: list[dict[str, int]] = []
    live_occurrences: list[int] = []

    for occ_index, occurrence in enumerate(occurrences):
        values = occurrence_values[occ_index]
        if occurrence["domain"] == "coord":
            const_source = _coord_constant_source(values)
        else:
            const_source = _address_constant_source(values)
        if const_source is not None:
            normalized_sources.append(const_source)
            continue
        if occurrence["domain"] == "coord":
            sm_affine_source = _coord_sm_affine_const_source(values, sm_indices)
            if sm_affine_source is not None:
                normalized_sources.append(sm_affine_source)
                continue

        exact_candidate: tuple[tuple[int, int, int], dict[str, int]] | None = None
        add_candidate: tuple[tuple[int, int, int], dict[str, int]] | None = None

        for prev_index in range(occ_index):
            delta = _invariant_delta(values, occurrence_values[prev_index])
            if delta is None:
                continue
            if delta == 0:
                candidate = _compose_source(
                    normalized_sources[prev_index],
                    0,
                    domain=str(occurrence["domain"]),
                )
                if candidate is None:
                    continue
                score = _candidate_score(candidate, 0)
                if exact_candidate is None or score < exact_candidate[0]:
                    exact_candidate = (score, candidate)
                continue

            if occurrence["domain"] == "coord":
                if not _coord_addend_is_structural(delta):
                    continue
            else:
                if not _address_addend_is_structural(delta, occurrence):
                    continue

            candidate = _compose_source(
                normalized_sources[prev_index],
                delta,
                domain=str(occurrence["domain"]),
            )
            if candidate is None:
                continue
            score = _candidate_score(candidate, delta)
            if add_candidate is None or score < add_candidate[0]:
                add_candidate = (score, candidate)

        if exact_candidate is not None:
            normalized_sources.append(exact_candidate[1])
            continue
        if add_candidate is not None:
            normalized_sources.append(add_candidate[1])
            continue

        live_index = len(live_occurrences)
        live_occurrences.append(occ_index)
        normalized_sources.append({"kind": "live", "index": live_index})

    compiled_program = copy.deepcopy(template)
    for occurrence, source in zip(occurrences, normalized_sources):
        _set_occurrence_source(compiled_program, occurrence, source)
    compiled_program["num_live_values"] = len(live_occurrences)

    per_builder_live_values: list[list[int]] = []
    for builder_index in range(len(raw_programs)):
        live_values: list[int] = []
        for occ_index in live_occurrences:
            live_values.append(int(occurrence_values[occ_index][builder_index]))
        per_builder_live_values.append(live_values)
    return compiled_program, per_builder_live_values


def _analyze_launcher(launcher) -> dict[str, object]:
    grouped_programs: dict[str, dict[str, object]] = {}
    ordered_group_keys: list[str] = []
    sm_program_ids = [0] * len(launcher.builder)
    sm_live_offsets = [0] * len(launcher.builder)
    live_values: list[int] = []
    programs: list[dict[str, object]] = []

    for sm_index, builder in enumerate(launcher.builder):
        raw_program = _program_structure_raw(builder)
        key = _canonical_program_template_key(raw_program)
        group = grouped_programs.get(key)
        if group is None:
            group = {
                "raw_programs": [],
                "sm_indices": [],
            }
            grouped_programs[key] = group
            ordered_group_keys.append(key)
        group["raw_programs"].append(raw_program)
        group["sm_indices"].append(sm_index)

    for group_key in ordered_group_keys:
        group = grouped_programs[group_key]
        optimized_program, group_live_values = _optimize_program_group(group["raw_programs"], group["sm_indices"])
        program_id = len(programs)
        programs.append(
            {
                "program_id": program_id,
                **optimized_program,
            }
        )
        for sm_index, sm_live_values in zip(group["sm_indices"], group_live_values):
            sm_program_ids[sm_index] = program_id
            sm_live_offsets[sm_index] = len(live_values)
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
        "version": 5,
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
