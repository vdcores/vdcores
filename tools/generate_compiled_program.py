#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPILED_SPEC_FILE = "dae_compiled_program.vdcore.json"
COMPILED_SPEC_ENV = "DAE_COMPILED_SPEC_FILE"
ALLOC_TABLE_MODE_ENV = "DAE_COMPILED_ALLOC_TABLE_MODE"
LIVE_VALUE_MODE_ENV = "DAE_COMPILED_LIVE_VALUE_MODE"

_LOAD_OPS = {
    "OP_ALLOC_TMA_LOAD_1D",
    "OP_ALLOC_TMA_LOAD_TENSOR_1D",
    "OP_ALLOC_TMA_LOAD_2D",
    "OP_ALLOC_TMA_LOAD_3D",
    "OP_ALLOC_TMA_LOAD_4D",
    "OP_ALLOC_TMA_LOAD_5D_FIX0",
}
_WRITEBACK_OPS = {
    "OP_ALLOC_WB_TMA_STORE_1D",
    "OP_ALLOC_WB_TMA_STORE_2D",
    "OP_ALLOC_WB_TMA_STORE_3D",
    "OP_ALLOC_WB_TMA_STORE_4D",
    "OP_ALLOC_WB_TMA_STORE_5D_FIX0",
    "OP_ALLOC_WB_TMA_REDUCE_ADD_2D",
    "OP_ALLOC_WB_TMA_REDUCE_ADD_3D",
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

_SLOT_INST_REQUIRED_BASE_OPS = {
    "OP_ALLOC_WB_RAW_ADDRESS",
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

_ALLOC_TABLE_MODE_DISABLED = "disabled"
_ALLOC_TABLE_MODE_CONSTANT = "constant"
_ALLOC_TABLE_MODE_SHARED = "shared"
_ALLOC_TABLE_MODE_GLOBAL = "global"
_ALLOC_TABLE_MODES = {
    _ALLOC_TABLE_MODE_DISABLED,
    _ALLOC_TABLE_MODE_CONSTANT,
    _ALLOC_TABLE_MODE_SHARED,
    _ALLOC_TABLE_MODE_GLOBAL,
}

_ALLOC_CMD_FLAG_DIRECT_READY = 1

_LIVE_VALUE_MODE_GLOBAL = "global"
_LIVE_VALUE_MODE_SHARED = "shared"
_LIVE_VALUE_MODE_CONSTANT = "constant"
_LIVE_VALUE_MODES = {
    _LIVE_VALUE_MODE_GLOBAL,
    _LIVE_VALUE_MODE_SHARED,
    _LIVE_VALUE_MODE_CONSTANT,
}


def _emit_int_array_initializer(values: list[int], *, indent: str) -> list[str]:
    if not values:
        return [f"{indent}}};"]

    lines: list[str] = []
    row: list[str] = []
    for index, value in enumerate(values, start=1):
        row.append(str(int(value)))
        if len(row) == 16 or index == len(values):
            suffix = "," if index != len(values) else ""
            lines.append(f"{indent}{', '.join(row)}{suffix}")
            row = []
    lines.append(f"{indent}}};")
    return lines


def _dense_affine_segments(values: list[int]) -> list[tuple[int, int, int, int]]:
    if not values:
        return []
    if len(values) == 1:
        return [(0, 1, int(values[0]), 0)]

    segments: list[tuple[int, int, int, int]] = []
    start = 0
    delta = int(values[1]) - int(values[0])
    for index in range(1, len(values) - 1):
        next_delta = int(values[index + 1]) - int(values[index])
        if next_delta != delta:
            segments.append((start, index + 1, int(values[start]), delta))
            start = index + 1
            delta = next_delta
    segments.append((start, len(values), int(values[start]), delta))
    return segments


def _dense_constant_runs(values: list[int]) -> list[tuple[int, int, int]]:
    if not values:
        return []

    runs: list[tuple[int, int, int]] = []
    start = 0
    current = int(values[0])
    for index in range(1, len(values)):
        value = int(values[index])
        if value != current:
            runs.append((start, index, current))
            start = index
            current = value
    runs.append((start, len(values), current))
    return runs


def _emit_dense_lookup(
    lines: list[str],
    *,
    storage_name: str,
    func_name: str,
    values: list[int],
    default_value: int,
) -> None:
    num_values = len(values)
    if num_values == 0:
        lines.append(f"static __device__ __forceinline__ int {func_name}(int) {{")
        lines.append(f"  return {int(default_value)};")
        lines.append("}")
        lines.append("")
        return

    runs = _dense_constant_runs(values)
    segments = _dense_affine_segments(values)
    use_runs = 0 < len(runs) <= 4 and len(runs) <= len(segments)
    use_affine = not use_runs and 0 < len(segments) <= 4

    if not use_runs and not use_affine:
        lines.append(f"static __device__ __constant__ int {storage_name}[{num_values}] = {{")
        lines.extend(_emit_int_array_initializer(values, indent="  "))
        lines.append("")

    lines.append(f"static __device__ __forceinline__ int {func_name}(int sm_id) {{")
    lines.append(f"  if (static_cast<unsigned int>(sm_id) >= {num_values}U) return {int(default_value)};")
    if use_runs:
        if len(runs) == 1:
            lines.append(f"  return {runs[0][2]};")
        else:
            for _, end, value in runs[:-1]:
                lines.append(f"  if (sm_id < {end}) return {value};")
            lines.append(f"  return {runs[-1][2]};")
    elif use_affine:
        if len(segments) == 1:
            start, _, base, delta = segments[0]
            if delta == 0:
                lines.append(f"  return {base};")
            elif start == 0:
                lines.append(f"  return {base} + sm_id * {delta};")
            else:
                lines.append(f"  return {base} + (sm_id - {start}) * {delta};")
        else:
            for start, end, base, delta in segments[:-1]:
                lines.append(f"  if (sm_id < {end}) return {base} + (sm_id - {start}) * {delta};")
            start, _, base, delta = segments[-1]
            lines.append(f"  return {base} + (sm_id - {start}) * {delta};")
    else:
        lines.append(f"  return {storage_name}[sm_id];")
    lines.append("}")
    lines.append("")


def resolve_spec_path(base_dir: Path) -> tuple[Path | None, str]:
    if COMPILED_SPEC_ENV in os.environ:
        spec_path = Path(os.environ[COMPILED_SPEC_ENV]).expanduser()
        if not spec_path.is_absolute():
            spec_path = base_dir / spec_path
        return spec_path, f"env:{COMPILED_SPEC_ENV}"

    spec_path = base_dir / DEFAULT_COMPILED_SPEC_FILE
    if spec_path.exists():
        return spec_path, f"file:{spec_path}"
    return None, "default:disabled"


def load_spec(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_alloc_table_mode() -> str:
    mode = os.environ.get(ALLOC_TABLE_MODE_ENV, _ALLOC_TABLE_MODE_GLOBAL).strip().lower()
    if mode not in _ALLOC_TABLE_MODES:
        raise ValueError(
            f"Unsupported {ALLOC_TABLE_MODE_ENV}={mode!r}; expected one of {sorted(_ALLOC_TABLE_MODES)}"
        )
    return mode


def load_live_value_mode() -> str:
    mode = os.environ.get(LIVE_VALUE_MODE_ENV, _LIVE_VALUE_MODE_SHARED).strip().lower()
    if mode not in _LIVE_VALUE_MODES:
        raise ValueError(
            f"Unsupported {LIVE_VALUE_MODE_ENV}={mode!r}; expected one of {sorted(_LIVE_VALUE_MODES)}"
        )
    return mode


def _u64_literal(value: int) -> str:
    return f"{int(value)}ULL"


def _compute_opcode_expr(name: str) -> str:
    return name if "__" not in name else "0"


def _emit_compute_inst_expr(entry: dict[str, object], indent: str) -> list[str]:
    args = [0, 0, 0]
    raw_args = [int(arg) for arg in entry.get("args", [])]
    args[: len(raw_args)] = raw_args
    return [
        f"{indent}CInst inst = dae_make_compiled_cinst({_compute_opcode_expr(entry['name'])}, {args[0]}, {args[1]}, {args[2]});"
    ]


def _compute_entry_signature(entry: dict[str, object]) -> tuple[object, ...]:
    return (
        entry["name"],
        tuple(int(arg) for arg in entry.get("args", [])),
    )


def _group_compute_entries(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    index = 0
    while index < len(entries):
        entry = entries[index]
        if entry["name"] == "OP_TERMINATEC":
            groups.append({"kind": "single", "entry": entry})
            index += 1
            continue

        signature = _compute_entry_signature(entry)
        run_end = index + 1
        while run_end < len(entries):
            candidate = entries[run_end]
            if candidate["name"] == "OP_TERMINATEC":
                break
            if _compute_entry_signature(candidate) != signature:
                break
            run_end += 1

        if run_end - index > 1:
            groups.append(
                {
                    "kind": "repeat",
                    "entry": entry,
                    "count": run_end - index,
                    "start_index": index,
                }
            )
        else:
            groups.append({"kind": "single", "entry": entry})
        index = run_end
    return groups


def _payload_scalar_var(index: int) -> str:
    return f"__payload_u64_{index}"


def _declare_local(indent: str, decl: str, init: str, *, maybe_unused: bool = True) -> str:
    prefix = "[[maybe_unused]] " if maybe_unused else ""
    return f"{indent}{prefix}{decl} = {init};"


def _step_sources(step: dict[str, object]) -> list[dict[str, object]]:
    sources: list[dict[str, object]] = []
    if "address_source" in step:
        sources.append(dict(step["address_source"]))
    for source in step.get("coord_sources", []):
        sources.append(dict(source))
    return sources


def _payload_indices_for_step(step: dict[str, object]) -> list[int]:
    indices: list[int] = []
    for source in _step_sources(step):
        if source["kind"] == "live":
            indices.append(int(source["index"]))
    return indices


def _source_signature(source: dict[str, object]) -> tuple[object, ...]:
    if source["kind"] == "const":
        return ("const",)
    if source["kind"] == "sm_affine_const":
        return ("sm_affine_const", int(source["base_sm"]), int(source["stride"]))
    return ("live", int(source["index"]))


def _source_base_add(source: dict[str, object]) -> int:
    if source["kind"] == "const":
        return int(source["value"])
    if source["kind"] == "sm_affine_const":
        return int(source["value"])
    return int(source.get("add", 0))


def _memory_step_fold_signature(step: dict[str, object]) -> tuple[object, ...]:
    flags = {name: bool(value) for name, value in step["flags"].items() if name != "jump"}
    opcode_value = (
        int(step["opcode_value"])
        if step["base_name"] in _SLOT_INST_REQUIRED_BASE_OPS
        else 0
    )
    return (
        step["base_name"],
        opcode_value,
        int(step["size"]),
        int(step["num_slots_value"]),
        int(step["nslot"]),
        int(step["bar_id"]),
        int(step["arg"]),
        tuple(sorted(flags.items())),
        tuple(int(value) for value in step.get("delta_coords", [])),
        int(step.get("delta_address", 0)),
        tuple(_source_signature(source) for source in _step_sources(step)),
        bool(step.get("st_consumes", False)),
    )


def _memory_op_coord_count(base_name: str) -> int:
    return _MEMORY_OP_COORD_COUNTS.get(base_name, 0)


def _group_foldable_steps(steps: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    index = 0
    while index < len(steps):
        step = steps[index]
        sources = _step_sources(step)
        if not sources:
            groups.append({"kind": "single", "step": step})
            index += 1
            continue

        signature = _memory_step_fold_signature(step)
        run_end = index + 1
        source_strides: list[int | None] = [None] * len(sources)
        while run_end < len(steps):
            candidate = steps[run_end]
            if _memory_step_fold_signature(candidate) != signature:
                break
            candidate_sources = _step_sources(candidate)
            if len(candidate_sources) != len(sources):
                break
            previous_sources = _step_sources(steps[run_end - 1])
            current_valid = True
            for source_index, (current_source, previous_source) in enumerate(zip(candidate_sources, previous_sources)):
                current_stride = _source_base_add(current_source) - _source_base_add(previous_source)
                if source_strides[source_index] is None:
                    source_strides[source_index] = current_stride
                elif current_stride != source_strides[source_index]:
                    current_valid = False
                    break
            if not current_valid:
                break
            run_end += 1

        if run_end - index > 1:
            groups.append(
                {
                    "kind": "fold",
                    "step": step,
                    "count": run_end - index,
                    "source_strides": [int(stride or 0) for stride in source_strides],
                }
            )
        else:
            groups.append({"kind": "single", "step": step})
        index = run_end
    return groups


def _collect_memory_sequence_items(
    blocks: list[dict[str, object]],
    *,
    include_step,
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for block in blocks:
        if block["kind"] == "repeat":
            steps = [step for step in block["steps"] if include_step(step)]
            if steps:
                items.append(
                    {
                        "kind": "repeat",
                        "count": int(block["count"]),
                        "steps": steps,
                    }
                )
            continue
        if block["kind"] == "op" and include_step(block):
            items.append({"kind": "op", "step": block})
    return items


def _flatten_memory_sequence_items(items: list[dict[str, object]]) -> list[dict[str, object]]:
    steps: list[dict[str, object]] = []
    for item in items:
        if item["kind"] == "repeat":
            steps.extend(item["steps"])
        else:
            steps.append(item["step"])
    return steps


def _emit_grouped_step_sequence(
    step_list: list[dict[str, object]],
    lines: list[str],
    indent: str,
    emit_step,
) -> None:
    for group in _group_foldable_steps(step_list):
        if group["kind"] == "fold":
            lines.append(f"{indent}for (int __step = 0; __step < {int(group['count'])}; ++__step) {{")
            emit_step(
                group["step"],
                lines,
                indent + "  ",
                step_offset_expr="__step",
                source_strides=[int(stride) for stride in group["source_strides"]],
            )
            lines.append(f"{indent}}}")
        else:
            emit_step(group["step"], lines, indent)


def _emit_grouped_sequence_items(
    items: list[dict[str, object]],
    lines: list[str],
    indent: str,
    emit_step,
) -> None:
    index = 0
    while index < len(items):
        item = items[index]
        if item["kind"] == "repeat":
            lines.append(f"{indent}for (int __rep = 0; __rep < {int(item['count'])}; ++__rep) {{")
            _emit_grouped_step_sequence(item["steps"], lines, indent + "  ", emit_step)
            lines.append(f"{indent}}}")
            index += 1
            continue

        run_end = index + 1
        op_steps = [item["step"]]
        while run_end < len(items) and items[run_end]["kind"] == "op":
            op_steps.append(items[run_end]["step"])
            run_end += 1
        _emit_grouped_step_sequence(op_steps, lines, indent, emit_step)
        index = run_end


def _flatten_alloc_steps(blocks: list[dict[str, object]]) -> list[dict[str, object]]:
    alloc_items = _collect_memory_sequence_items(blocks, include_step=lambda _step: True)
    steps: list[dict[str, object]] = []
    for item in alloc_items:
        if item["kind"] == "repeat":
            repeat_count = int(item["count"])
            for _ in range(repeat_count):
                steps.extend(item["steps"])
        else:
            steps.append(item["step"])
    return steps


def _alloc_terminate_count(blocks: list[dict[str, object]]) -> int:
    return sum(1 for block in blocks if block["kind"] == "terminate")


def _program_uses_table_driven_alloc(program: dict[str, object]) -> bool:
    return all(step["base_name"] not in _SLOT_INST_REQUIRED_BASE_OPS for step in _flatten_alloc_steps(program["memory"]))


def _encode_alloc_cmd(step: dict[str, object]) -> int:
    flags = _ALLOC_CMD_FLAG_DIRECT_READY if step["base_name"] in _WRITEBACK_OPS else 0
    port = 1 if step["flags"]["port1"] else 0
    return int(step["nslot"]) | (port << 8) | (flags << 9)


def _emit_alloc_cmd_array(
    lines: list[str],
    *,
    name: str,
    storage: str,
    values: list[int],
) -> None:
    array_size = max(len(values), 1)
    if storage == "constant":
        lines.append(f"static __device__ __constant__ uint32_t {name}[{array_size}] = {{")
    elif storage == "global":
        lines.append(f"static __device__ uint32_t {name}[{array_size}] = {{")
    else:
        raise ValueError(f"unsupported alloc cmd storage {storage}")
    if values:
        lines.extend(_emit_int_array_initializer(values, indent="  "))
    else:
        lines.append("  0")
        lines.append("};")
    lines.append("")


def _alloc_table_storage_kind(mode: str) -> str:
    if mode in {_ALLOC_TABLE_MODE_CONSTANT, _ALLOC_TABLE_MODE_SHARED}:
        return "constant"
    if mode == _ALLOC_TABLE_MODE_GLOBAL:
        return "global"
    raise ValueError(f"alloc table storage is undefined for mode {mode}")


def _sm_live_counts(spec: dict[str, object]) -> list[int]:
    offsets = [int(offset) for offset in spec.get("sm_live_offsets", [])]
    total = int(spec.get("num_live_values", 0))
    if not offsets:
        return []
    counts: list[int] = []
    for index, offset in enumerate(offsets):
        next_offset = offsets[index + 1] if index + 1 < len(offsets) else total
        counts.append(max(0, next_offset - offset))
    return counts


def _emit_payload_aliases(step_list: list[dict[str, object]], lines: list[str], indent: str) -> None:
    seen: set[int] = set()
    for step in step_list:
        for index in _payload_indices_for_step(step):
            if index in seen:
                continue
            seen.add(index)
            lines.append(
                f"{indent}[[maybe_unused]] const uint64_t &{_payload_scalar_var(index)} = live_values[{index}];"
            )


def _reg_pair_var(pair_id: int) -> str:
    return f"__reg_pair_{pair_id}"


def _program_port_reg_pairs(program: dict[str, object], port_id: int) -> list[int]:
    pair_ids = [
        int(pair["pair_id"])
        for pair in program.get("reg_pairs", [])
        if int(bool(pair["port1"])) == port_id
    ]
    return sorted(pair_ids)


def _program_port_uses_generic_reg_file(program: dict[str, object], port_id: int) -> bool:
    if not program.get("needs_generic_reg_file", False):
        return False
    for block in program["memory"]:
        steps = block["steps"] if block["kind"] == "repeat" else [block]
        for step in steps:
            if step.get("base_name") not in _NO_ADDRESS_MEMORY_BASE_OPS:
                continue
            if int(step["flags"]["port1"]) != port_id:
                continue
            if "reg_pair_id" not in step:
                return True
    return False


def _emit_ld_locals(program: dict[str, object], lines: list[str], indent: str, port_id: int) -> None:
    for pair_id in _program_port_reg_pairs(program, port_id):
        lines.append(f"{indent}int {_reg_pair_var(pair_id)} = 0;")
    if _program_port_uses_generic_reg_file(program, port_id):
        lines.append(f"{indent}int __reg_file[32] = {{}};")


def _source_expr(
    source: dict[str, object],
    *,
    step_offset_expr: str | None,
    stride: int,
    coord: bool,
) -> str:
    if source["kind"] == "const":
        base_expr = str(int(source["value"]))
        if step_offset_expr is not None and stride != 0:
            return f"({base_expr} + ({step_offset_expr}) * {stride})"
        return base_expr
    if source["kind"] == "sm_affine_const":
        base_sm = int(source["base_sm"])
        base_expr = str(int(source["value"]))
        stride_expr = int(source["stride"]) + int(stride)
        if stride_expr == 0:
            return base_expr
        return f"({base_expr} + (sm_id - {base_sm}) * {stride_expr})"

    live_expr = _payload_scalar_var(int(source["index"]))
    add = int(source.get("add", 0))
    delta_terms: list[str] = []
    if add != 0:
        delta_terms.append(str(add))
    if step_offset_expr is not None and stride != 0:
        delta_terms.append(f"({step_offset_expr}) * {stride}")
    if coord:
        base_expr = f"static_cast<int>({live_expr} & 0xFFFFULL)"
        if delta_terms:
            return f"({base_expr} + {' + '.join(delta_terms)})"
        return base_expr
    if not delta_terms:
        return f"static_cast<uint64_t>({live_expr})"
    return (
        f"static_cast<uint64_t>(static_cast<int64_t>({live_expr}) + "
        f"static_cast<int64_t>({' + '.join(delta_terms)}))"
    )


def _emit_memory_field_locals(
    step: dict[str, object],
    lines: list[str],
    indent: str,
    *,
    step_offset_expr: str | None = None,
    source_strides: list[int] | None = None,
    emit_address: bool = False,
    emit_coord_count: int = 0,
) -> None:
    base_name = step["base_name"]
    lines.append(_declare_local(indent, "constexpr uint16_t __opcode", str(int(step["opcode_value"])) ))
    lines.append(_declare_local(indent, "constexpr uint16_t __size", str(int(step["size"])) ))
    lines.append(_declare_local(indent, "constexpr uint16_t __num_slots", str(int(step["num_slots_value"])) ))
    lines.append(_declare_local(indent, "constexpr uint8_t __nslot", str(int(step["nslot"])) ))
    lines.append(_declare_local(indent, "constexpr uint8_t __bar", str(int(step["bar_id"])) ))
    lines.append(_declare_local(indent, "constexpr uint16_t __arg", str(int(step["arg"])) ))
    if base_name in _NO_ADDRESS_MEMORY_BASE_OPS:
        return
    if base_name in _SCALAR_ADDRESS_FIELD_BASE_OPS:
        if not emit_address:
            return
        address_source = step["address_source"]
        address_stride = 0 if not source_strides else int(source_strides[0])
        base_expr = _source_expr(
            address_source,
            step_offset_expr=step_offset_expr,
            stride=address_stride,
            coord=False,
        )
        if "delta_address" in step and int(step["delta_address"]) != 0:
            lines.append(_declare_local(indent, "uint64_t __address", base_expr))
            lines.append(
                f"{indent}if (__rep != 0) __address += static_cast<uint64_t>(__rep) * {_u64_literal(int(step['delta_address']))};"
            )
        else:
            lines.append(_declare_local(indent, "const uint64_t __address", base_expr))
        return

    if emit_coord_count <= 0:
        return

    delta_coords = step.get("delta_coords")
    coord_sources = step.get("coord_sources", [])
    coord_source_strides = []
    if source_strides:
        start_index = 1 if "address_source" in step else 0
        coord_source_strides = [int(stride) for stride in source_strides[start_index: start_index + emit_coord_count]]
    while len(coord_source_strides) < emit_coord_count:
        coord_source_strides.append(0)
    for coord_index in range(emit_coord_count):
        coord_name = f"__coord{coord_index}"
        coord_expr = _source_expr(
            coord_sources[coord_index],
            step_offset_expr=step_offset_expr,
            stride=coord_source_strides[coord_index],
            coord=True,
        )
        delta = 0 if delta_coords is None else int(delta_coords[coord_index])
        if delta != 0:
            lines.append(_declare_local(indent, f"uint16_t {coord_name}", f"static_cast<uint16_t>({coord_expr})"))
            lines.append(
                f"{indent}if (__rep != 0) {coord_name} = static_cast<uint16_t>({coord_name} + static_cast<uint16_t>(__rep * {delta}));"
            )
        else:
            lines.append(_declare_local(indent, f"const uint16_t {coord_name}", f"static_cast<uint16_t>({coord_expr})"))


def _emit_store_slot_inst(step: dict[str, object], lines: list[str], indent: str, slot_expr: str) -> None:
    if step["base_name"] not in _SLOT_INST_REQUIRED_BASE_OPS:
        return
    lines.append(f"{indent}auto &__slot_inst = st_insts[{slot_expr}];")
    lines.append(f"{indent}__slot_inst.opcode = __opcode;")
    lines.append(f"{indent}__slot_inst.size = __size;")
    lines.append(f"{indent}__slot_inst.num_slots = __num_slots;")
    lines.append(f"{indent}__slot_inst.arg = __arg;")
    if step["base_name"] in _NO_ADDRESS_MEMORY_BASE_OPS:
        lines.append(f"{indent}__slot_inst.address = 0;")
        return
    if step["base_name"] in _SCALAR_ADDRESS_FIELD_BASE_OPS:
        lines.append(f"{indent}__slot_inst.address = __address;")
        return
    for coord_index in range(_memory_op_coord_count(step["base_name"])):
        lines.append(f"{indent}__slot_inst.coords[{coord_index}] = __coord{coord_index};")


def _emit_alloc_step(
    step: dict[str, object],
    lines: list[str],
    indent: str,
    *,
    step_offset_expr: str | None = None,
    source_strides: list[int] | None = None,
) -> None:
    lines.append(f"{indent}{{")
    _emit_memory_field_locals(
        step,
        lines,
        indent + "  ",
        step_offset_expr=step_offset_expr,
        source_strides=source_strides,
        emit_address=step["base_name"] in _SLOT_INST_REQUIRED_BASE_OPS and step["base_name"] in _SCALAR_ADDRESS_FIELD_BASE_OPS,
        emit_coord_count=_memory_op_coord_count(step["base_name"]) if step["base_name"] in _SLOT_INST_REQUIRED_BASE_OPS else 0,
    )
    lines.append(f"{indent}  int alloc_mask = 0;")
    lines.append(f"{indent}  int slot_alloc = -1;")
    lines.append(f"{indent}  while (true) {{")
    lines.append(f"{indent}    slot_alloc = alloc.allocate(lane_id, flags, __nslot, alloc_mask);")
    lines.append(f"{indent}    if (slot_alloc >= 0) break;")
    lines.append(f"{indent}    __nanosleep(allocRetrySleepCycles);")
    lines.append(f"{indent}  }}")
    lines.append(f"{indent}  if (lane_id == 0) {{")
    _emit_store_slot_inst(step, lines, indent + "    ", "slot_alloc")
    lines.append(f"{indent}    m2c.put(alloc_mask);")
    if step["base_name"] in _WRITEBACK_OPS:
        lines.append(
            f'{indent}    __mprint("[compiled alloc] {step["base_name"]} direct-ready slot=%d m2c=%d", '
            f'slot_alloc, m2c.ptr);'
        )
        lines.append(f"{indent}    m2c.commit();")
        lines.append(f"{indent}    m2c.advance();")
    else:
        lines.append(f"{indent}    CompiledLdCmd ld;")
        lines.append(f"{indent}    ld.init(static_cast<uint8_t>(slot_alloc), static_cast<uint8_t>(m2c.ptr));")
        lines.append(f"{indent}    auto &curld = m2ld[{1 if step['flags']['port1'] else 0}];")
        lines.append(
            f'{indent}    __mprint("[compiled alloc] {step["base_name"]} enqueue slot=%d m2c=%d ldq={1 if step["flags"]["port1"] else 0} ldptr=%d", '
            f'slot_alloc, m2c.ptr, curld.ptr);'
        )
        lines.append(f"{indent}    curld.put(ld.raw);")
        lines.append(f"{indent}    m2c.advance();")
        lines.append(f"{indent}    curld.commit();")
        lines.append(f"{indent}    curld.advance();")
    lines.append(f"{indent}  }}")
    lines.append(f"{indent}}}")


def _emit_alloc_terminate(lines: list[str], indent: str) -> None:
    lines.append(f"{indent}if (lane_id == 0) {{")
    lines.append(f'{indent}  __mprint("[compiled alloc] terminate ld0_ptr=%d ld1_ptr=%d", m2ld[0].ptr, m2ld[1].ptr);')
    lines.append(f"{indent}}}")


def _emit_alloc_program(blocks: list[dict[str, object]], lines: list[str], indent: str) -> None:
    index = 0
    while index < len(blocks):
        if blocks[index]["kind"] == "terminate":
            _emit_alloc_terminate(lines, indent)
            index += 1
            continue

        run_end = index
        while run_end < len(blocks) and blocks[run_end]["kind"] != "terminate":
            run_end += 1
        items = _collect_memory_sequence_items(blocks[index:run_end], include_step=lambda _step: True)
        _emit_grouped_sequence_items(items, lines, indent, _emit_alloc_step)
        index = run_end


def _emit_ld_barrier_wait(step: dict[str, object], lines: list[str], indent: str) -> None:
    if not step["flags"]["barrier"] or step["writeback"]:
        return
    lines.append(f"{indent}volatile int *bar = bars + __bar;")
    lines.append(
        f'{indent}__mprint("[compiled ld] {step["base_name"]} wait bar=%d value=%d slot=%d m2c=%d", '
        f'__bar, *bar, cmd.slot, cmd.m2c_slot);'
    )
    lines.append(f"{indent}while (*bar != 0) {{")
    lines.append(f"{indent}  __nanosleep(barrierPollSleepCycles);")
    lines.append(f"{indent}}}")
    lines.append(
        f'{indent}__mprint("[compiled ld] {step["base_name"]} passed bar=%d slot=%d m2c=%d", '
        f'__bar, cmd.slot, cmd.m2c_slot);'
    )


def _emit_ld_step(
    step: dict[str, object],
    lines: list[str],
    indent: str,
    *,
    step_offset_expr: str | None = None,
    source_strides: list[int] | None = None,
) -> None:
    base_name = step["base_name"]
    lines.append(f"{indent}{{")
    lines.append(f"{indent}  CompiledLdCmd cmd {{}};")
    lines.append(f"{indent}  cmd.raw = m2ld.pop();")
    lines.append(
        f'{indent}  __mprint("[compiled ld] {base_name} pop slot=%d m2c=%d", cmd.slot, cmd.m2c_slot);'
    )
    if base_name in _LOAD_OPS:
        _emit_memory_field_locals(
            step,
            lines,
            indent + "  ",
            step_offset_expr=step_offset_expr,
            source_strides=source_strides,
            emit_address=base_name in _SCALAR_ADDRESS_FIELD_BASE_OPS,
            emit_coord_count=_memory_op_coord_count(base_name),
        )
        _emit_ld_barrier_wait(step, lines, indent + "  ")
        if base_name == "OP_ALLOC_TMA_LOAD_1D":
            lines.append(f"{indent}  asm volatile(")
            lines.append(
                f'{indent}  "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "'
            )
            lines.append(f'{indent}  "[%0], [%1], %2, [%3];\\n"')
            lines.append(f"{indent}  :")
            lines.append(
                f"{indent}  : \"r\"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, cmd.slot))),"
            )
            lines.append(f"{indent}    \"l\"(reinterpret_cast<const void *>(__address)),")
            lines.append(f"{indent}    \"r\"((uint32_t)__size),")
            lines.append(
                f"{indent}    \"r\"((uint32_t)__cvta_generic_to_shared(m2c.native_bar(cmd.m2c_slot)))"
            )
            lines.append(f"{indent}  : \"memory\");")
            lines.append(f"{indent}  cuda::device::barrier_expect_tx(m2c.barriers[cmd.m2c_slot], cuda::aligned_size_t<16>(__size));")
        elif base_name == "OP_ALLOC_TMA_LOAD_TENSOR_1D":
            lines.append(f"{indent}  asm volatile(")
            lines.append(f'{indent}  "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes"')
            lines.append(f'{indent}  "[%0], [%1, {{%2}}], [%3];\\n"')
            lines.append(f"{indent}  :")
            lines.append(f"{indent}  : \"r\"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, cmd.slot))),")
            lines.append(f"{indent}    \"l\"((void *)(tma_descs + __arg)),")
            lines.append(f"{indent}    \"r\"((uint32_t)__address),")
            lines.append(f"{indent}    \"r\"((uint32_t)__cvta_generic_to_shared(m2c.native_bar(cmd.m2c_slot)))")
            lines.append(f"{indent}  : \"memory\");")
            lines.append(f"{indent}  cuda::device::barrier_expect_tx(m2c.barriers[cmd.m2c_slot], cuda::aligned_size_t<16>(__size));")
        elif base_name == "OP_ALLOC_TMA_LOAD_2D":
            lines.append(f"{indent}  asm volatile(")
            lines.append(f'{indent}  "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"')
            lines.append(f'{indent}  "[%0], [%1, {{%2, %3}}], [%4];\\n"')
            lines.append(f"{indent}  :")
            lines.append(f"{indent}  : \"r\"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, cmd.slot))),")
            lines.append(f"{indent}    \"l\"((void *)(tma_descs + __arg)),")
            lines.append(f"{indent}    \"r\"((int)__coord0),")
            lines.append(f"{indent}    \"r\"((int)__coord1),")
            lines.append(f"{indent}    \"r\"((uint32_t)__cvta_generic_to_shared(m2c.native_bar(cmd.m2c_slot)))")
            lines.append(f"{indent}  : \"memory\");")
            lines.append(f"{indent}  cuda::device::barrier_expect_tx(m2c.barriers[cmd.m2c_slot], cuda::aligned_size_t<16>(__size));")
        elif base_name == "OP_ALLOC_TMA_LOAD_3D":
            lines.append(f"{indent}  asm volatile(")
            lines.append(f'{indent}  "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"')
            lines.append(f'{indent}  "[%0], [%1, {{%2, %3, %4}}], [%5];\\n"')
            lines.append(f"{indent}  :")
            lines.append(f"{indent}  : \"r\"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, cmd.slot))),")
            lines.append(f"{indent}    \"l\"((void *)(tma_descs + __arg)),")
            lines.append(f"{indent}    \"r\"((int)__coord0),")
            lines.append(f"{indent}    \"r\"((int)__coord1),")
            lines.append(f"{indent}    \"r\"((int)__coord2),")
            lines.append(f"{indent}    \"r\"((uint32_t)__cvta_generic_to_shared(m2c.native_bar(cmd.m2c_slot)))")
            lines.append(f"{indent}  : \"memory\");")
            lines.append(f"{indent}  cuda::device::barrier_expect_tx(m2c.barriers[cmd.m2c_slot], cuda::aligned_size_t<16>(__size));")
        elif base_name == "OP_ALLOC_TMA_LOAD_4D":
            lines.append(f"{indent}  asm volatile(")
            lines.append(f'{indent}  "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes"')
            lines.append(f'{indent}  "[%0], [%1, {{%2, %3, %4, %5}}], [%6];\\n"')
            lines.append(f"{indent}  :")
            lines.append(f"{indent}  : \"r\"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, cmd.slot))),")
            lines.append(f"{indent}    \"l\"((void *)(tma_descs + __arg)),")
            lines.append(f"{indent}    \"r\"((int)__coord0),")
            lines.append(f"{indent}    \"r\"((int)__coord1),")
            lines.append(f"{indent}    \"r\"((int)__coord2),")
            lines.append(f"{indent}    \"r\"((int)__coord3),")
            lines.append(f"{indent}    \"r\"((uint32_t)__cvta_generic_to_shared(m2c.native_bar(cmd.m2c_slot)))")
            lines.append(f"{indent}  : \"memory\");")
            lines.append(f"{indent}  cuda::device::barrier_expect_tx(m2c.barriers[cmd.m2c_slot], cuda::aligned_size_t<16>(__size));")
        elif base_name == "OP_ALLOC_TMA_LOAD_5D_FIX0":
            lines.append(f"{indent}  asm volatile(")
            lines.append(f'{indent}  "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes"')
            lines.append(f'{indent}  "[%0], [%1, {{0, %2, %3, %4, %5}}], [%6];\\n"')
            lines.append(f"{indent}  :")
            lines.append(f"{indent}  : \"r\"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, cmd.slot))),")
            lines.append(f"{indent}    \"l\"((void *)(tma_descs + __arg)),")
            lines.append(f"{indent}    \"r\"((int)__coord0),")
            lines.append(f"{indent}    \"r\"((int)__coord1),")
            lines.append(f"{indent}    \"r\"((int)__coord2),")
            lines.append(f"{indent}    \"r\"((int)__coord3),")
            lines.append(f"{indent}    \"r\"((uint32_t)__cvta_generic_to_shared(m2c.native_bar(cmd.m2c_slot)))")
            lines.append(f"{indent}  : \"memory\");")
            lines.append(f"{indent}  cuda::device::barrier_expect_tx(m2c.barriers[cmd.m2c_slot], cuda::aligned_size_t<16>(__size));")
    elif base_name == "OP_ALLOC_WB_REG_STORE":
        lines.append(f"{indent}  const int slotMask = mkSlotMask(cmd.slot, {int(step['nslot'])});")
        lines.append(f"{indent}  m2c.data[cmd.m2c_slot] = slotMask | 0x80000000U;")
        if "reg_pair_id" in step:
            lines.append(f"{indent}  {_reg_pair_var(int(step['reg_pair_id']))} = slotMask;")
        else:
            lines.append(f"{indent}  __reg_file[{int(step['size'])}] = slotMask;")
    elif base_name == "OP_ALLOC_REG_LOAD":
        if "reg_pair_id" in step:
            pair_var = _reg_pair_var(int(step["reg_pair_id"]))
            lines.append(f"{indent}  m2c.data[cmd.m2c_slot] = {pair_var};")
            lines.append(f"{indent}  {pair_var} = 0;")
        else:
            lines.append(f"{indent}  m2c.data[cmd.m2c_slot] = __reg_file[{int(step['size'])}];")
    elif base_name == "OP_ALLOC_WB_RAW_ADDRESS":
        pass
    elif base_name in _WRITEBACK_OPS:
        pass
    else:
        raise ValueError(f"Unsupported LDU step {base_name}")
    lines.append(f"{indent}  (void)m2c.barriers[cmd.m2c_slot].arrive();")
    lines.append(
        f'{indent}  __mprint("[compiled ld] {base_name} arrive slot=%d m2c=%d", cmd.slot, cmd.m2c_slot);'
    )
    lines.append(f"{indent}}}")
    return


def _emit_ld_program(program: dict[str, object], lines: list[str], indent: str, port_id: int) -> None:
    blocks = program["memory"]

    def _is_ld_step(step: dict[str, object]) -> bool:
        return (
            step["base_name"] in _LOAD_OPS
            or step["base_name"] in _NO_ADDRESS_MEMORY_BASE_OPS
            or step["base_name"] == "OP_ALLOC_WB_RAW_ADDRESS"
        )

    filtered_items = _collect_memory_sequence_items(
        blocks,
        include_step=lambda step: int(step["flags"]["port1"]) == port_id and _is_ld_step(step),
    )
    step_list = _flatten_memory_sequence_items(filtered_items)
    _emit_ld_locals(program, lines, indent, port_id)
    _emit_payload_aliases(step_list, lines, indent)
    _emit_grouped_sequence_items(filtered_items, lines, indent, _emit_ld_step)


def _emit_st_step(
    step: dict[str, object],
    lines: list[str],
    indent: str,
    *,
    step_offset_expr: str | None = None,
    source_strides: list[int] | None = None,
) -> None:
    base_name = step["base_name"]
    lines.append(f"{indent}{{")
    lines.append(f"{indent}  int slot_token = c2m.pop();")
    if base_name == "OP_ALLOC_WB_RAW_ADDRESS":
        lines.append(f"{indent}  int slot = slot_token;")
    else:
        lines.append(f"{indent}  int slot = extract(slot_token);")
    lines.append(
        f'{indent}  __mprint("[compiled st] {base_name} pop token=%x slot=%d", slot_token, slot);'
    )
    _emit_memory_field_locals(
        step,
        lines,
        indent + "  ",
        step_offset_expr=step_offset_expr,
        source_strides=source_strides,
        emit_address=base_name == "OP_ALLOC_WB_TMA_STORE_1D",
        emit_coord_count=_memory_op_coord_count(base_name),
    )
    if base_name == "OP_ALLOC_WB_TMA_STORE_1D":
        lines.append(f"{indent}  cuda::ptx::cp_async_bulk(")
        lines.append(f"{indent}    cuda::ptx::space_global,")
        lines.append(f"{indent}    cuda::ptx::space_shared,")
        lines.append(f"{indent}    (void *)(__address),")
        lines.append(f"{indent}    (const void *)(get_slot_address(smem_base, slot)),")
        lines.append(f"{indent}    __size);")
        lines.append(f"{indent}  cuda::ptx::cp_async_bulk_commit_group();")
    elif base_name == "OP_ALLOC_WB_TMA_STORE_2D":
        lines.append(f"{indent}  asm volatile(")
        lines.append(f'{indent}  "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "')
        lines.append(f'{indent}  "[%0, {{%1, %2}}], [%3];\\n"')
        lines.append(f"{indent}  :")
        lines.append(f"{indent}  : \"l\"((void *)(tma_descs + __arg)),")
        lines.append(f"{indent}    \"r\"((int)__coord0),")
        lines.append(f"{indent}    \"r\"((int)__coord1),")
        lines.append(f"{indent}    \"r\"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))")
        lines.append(f"{indent}  : \"memory\");")
        lines.append(f"{indent}  cuda::ptx::cp_async_bulk_commit_group();")
    elif base_name == "OP_ALLOC_WB_TMA_STORE_3D":
        lines.append(f"{indent}  asm volatile(")
        lines.append(f'{indent}  "cp.async.bulk.tensor.3d.global.shared::cta.bulk_group "')
        lines.append(f'{indent}  "[%0, {{%1, %2, %3}}], [%4];\\n"')
        lines.append(f"{indent}  :")
        lines.append(f"{indent}  : \"l\"((void *)(tma_descs + __arg)),")
        lines.append(f"{indent}    \"r\"((int)__coord0),")
        lines.append(f"{indent}    \"r\"((int)__coord1),")
        lines.append(f"{indent}    \"r\"((int)__coord2),")
        lines.append(f"{indent}    \"r\"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))")
        lines.append(f"{indent}  : \"memory\");")
        lines.append(f"{indent}  cuda::ptx::cp_async_bulk_commit_group();")
    elif base_name == "OP_ALLOC_WB_TMA_STORE_4D":
        lines.append(f"{indent}  asm volatile(")
        lines.append(f'{indent}  "cp.async.bulk.tensor.4d.global.shared::cta.bulk_group "')
        lines.append(f'{indent}  "[%0, {{%1, %2, %3, %4}}], [%5];\\n"')
        lines.append(f"{indent}  :")
        lines.append(f"{indent}  : \"l\"((void *)(tma_descs + __arg)),")
        lines.append(f"{indent}    \"r\"((int)__coord0),")
        lines.append(f"{indent}    \"r\"((int)__coord1),")
        lines.append(f"{indent}    \"r\"((int)__coord2),")
        lines.append(f"{indent}    \"r\"((int)__coord3),")
        lines.append(f"{indent}    \"r\"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))")
        lines.append(f"{indent}  : \"memory\");")
        lines.append(f"{indent}  cuda::ptx::cp_async_bulk_commit_group();")
    elif base_name == "OP_ALLOC_WB_TMA_STORE_5D_FIX0":
        lines.append(f"{indent}  asm volatile(")
        lines.append(f'{indent}  "cp.async.bulk.tensor.5d.global.shared::cta.bulk_group "')
        lines.append(f'{indent}  "[%0, {{0, %1, %2, %3, %4}}], [%5];\\n"')
        lines.append(f"{indent}  :")
        lines.append(f"{indent}  : \"l\"((void *)(tma_descs + __arg)),")
        lines.append(f"{indent}    \"r\"((int)__coord0),")
        lines.append(f"{indent}    \"r\"((int)__coord1),")
        lines.append(f"{indent}    \"r\"((int)__coord2),")
        lines.append(f"{indent}    \"r\"((int)__coord3),")
        lines.append(f"{indent}    \"r\"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))")
        lines.append(f"{indent}  : \"memory\");")
        lines.append(f"{indent}  cuda::ptx::cp_async_bulk_commit_group();")
    elif base_name == "OP_ALLOC_WB_TMA_REDUCE_ADD_2D":
        lines.append(f"{indent}  asm volatile(")
        lines.append(f'{indent}  "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group "')
        lines.append(f'{indent}  "[%0, {{%1, %2}}], [%3];\\n"')
        lines.append(f"{indent}  :")
        lines.append(f"{indent}  : \"l\"((void *)(tma_descs + __arg)),")
        lines.append(f"{indent}    \"r\"((int)__coord0),")
        lines.append(f"{indent}    \"r\"((int)__coord1),")
        lines.append(f"{indent}    \"r\"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))")
        lines.append(f"{indent}  : \"memory\");")
        lines.append(f"{indent}  cuda::ptx::cp_async_bulk_commit_group();")
    elif base_name == "OP_ALLOC_WB_TMA_REDUCE_ADD_3D":
        lines.append(f"{indent}  asm volatile(")
        lines.append(f'{indent}  "cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.bulk_group "')
        lines.append(f'{indent}  "[%0, {{%1, %2, %3}}], [%4];\\n"')
        lines.append(f"{indent}  :")
        lines.append(f"{indent}  : \"l\"((void *)(tma_descs + __arg)),")
        lines.append(f"{indent}    \"r\"((int)__coord0),")
        lines.append(f"{indent}    \"r\"((int)__coord1),")
        lines.append(f"{indent}    \"r\"((int)__coord2),")
        lines.append(f"{indent}    \"r\"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))")
        lines.append(f"{indent}  : \"memory\");")
        lines.append(f"{indent}  cuda::ptx::cp_async_bulk_commit_group();")
    elif base_name == "OP_ALLOC_WB_RAW_ADDRESS":
        lines.append(f"{indent}  (void)slot;")
    else:
        raise ValueError(f"Unsupported STU step {base_name}")
    if step["flags"]["barrier"]:
        if base_name != "OP_ALLOC_WB_RAW_ADDRESS":
            lines.append(f"{indent}  cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{{}});")
        lines.append(f"{indent}  cuda::std::atomic_ref<int> bar {{bars[__bar]}};")
        lines.append(f"{indent}  int current_cnt = bar.fetch_sub(1, cuda::std::memory_order_release);")
        lines.append(
            f'{indent}  __mprint("[compiled st] {base_name} barrier=%d remaining=%d", __bar, current_cnt - 1);'
        )
    elif base_name != "OP_ALLOC_WB_RAW_ADDRESS":
        lines.append(f"{indent}  cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<0>{{}});")
    if base_name != "OP_ALLOC_WB_RAW_ADDRESS":
        lines.append(f"{indent}  c2m.reset(slot_token);")
        lines.append(
            f'{indent}  __mprint("[compiled st] {base_name} reset token=%x slot=%d", slot_token, slot);'
        )
    lines.append(f"{indent}}}")


def _emit_st_program(blocks: list[dict[str, object]], lines: list[str], indent: str) -> None:
    write_items = _collect_memory_sequence_items(
        blocks,
        include_step=lambda step: bool(step["st_consumes"]),
    )
    step_list = _flatten_memory_sequence_items(write_items)
    _emit_payload_aliases(step_list, lines, indent)
    if not write_items:
        lines.append(f"{indent}int __done = c2m.pop();")
        lines.append(f"{indent}assert(__done == 0 && \"compiled st expected terminate sentinel\");")
        return
    _emit_grouped_sequence_items(write_items, lines, indent, _emit_st_step)
    lines.append(f"{indent}int __done = c2m.pop();")
    lines.append(f"{indent}assert(__done == 0 && \"compiled st expected terminate sentinel\");")


def generate_enabled(
    spec: dict[str, object], *, debug: bool, alloc_table_mode: str, live_value_mode: str
) -> str:
    sm_program_ids = [int(program_id) for program_id in spec["sm_program_ids"]]
    sm_live_offsets = [int(live_offset) for live_offset in spec.get("sm_live_offsets", [])]
    sm_live_counts = _sm_live_counts(spec)
    max_live_values_per_sm = max(sm_live_counts, default=0)
    table_driven_programs = {
        int(program["program_id"]): _flatten_alloc_steps(program["memory"])
        for program in spec["programs"]
        if alloc_table_mode != _ALLOC_TABLE_MODE_DISABLED and _program_uses_table_driven_alloc(program)
    }
    table_storage_kind = (
        _alloc_table_storage_kind(alloc_table_mode)
        if alloc_table_mode != _ALLOC_TABLE_MODE_DISABLED
        else None
    )
    max_alloc_cmds = max((len(steps) for steps in table_driven_programs.values()), default=0)
    lines = [
        "// Generated by tools/generate_compiled_program.py.",
        f'// compiled_hash={spec["hash"]}',
        f"// alloc_table_mode={alloc_table_mode}",
        f"// live_value_mode={live_value_mode}",
        "",
        "static constexpr bool daeCompiledProgramEnabled = true;",
        f"static constexpr bool daeCompiledProgramDebug = {'true' if debug else 'false'};",
        f'static constexpr const char *daeCompiledProgramHash = "{spec["hash"]}";',
        f'static constexpr int daeCompiledProgramNumSms = {int(spec["num_sms"])};',
        f'static constexpr int daeCompiledProgramLiveValueCount = {int(spec.get("num_live_values", 0))};',
        f"static constexpr int daeCompiledLiveValueMaxPerSm = {max_live_values_per_sm};",
        f"static constexpr int daeCompiledAllocMaxCmdCount = {max_alloc_cmds};",
        "",
        f"static constexpr int daeCompiledLiveValueModeGlobal = 0;",
        f"static constexpr int daeCompiledLiveValueModeShared = 1;",
        f"static constexpr int daeCompiledLiveValueModeConstant = 2;",
        f"static constexpr int daeCompiledLiveValueMode = "
        f"{'daeCompiledLiveValueModeGlobal' if live_value_mode == _LIVE_VALUE_MODE_GLOBAL else ('daeCompiledLiveValueModeShared' if live_value_mode == _LIVE_VALUE_MODE_SHARED else 'daeCompiledLiveValueModeConstant')};",
        "",
        "static __device__ __constant__ uint64_t "
        "daeCompiledLiveValuesConst[(daeCompiledProgramLiveValueCount > 0) ? daeCompiledProgramLiveValueCount : 1] = {};",
        "",
    ]
    _emit_dense_lookup(
        lines,
        storage_name="daeCompiledProgramIdsBySm",
        func_name="dae_compiled_program_id_for_sm",
        values=sm_program_ids,
        default_value=-1,
    )
    _emit_dense_lookup(
        lines,
        storage_name="daeCompiledLiveOffsetsBySm",
        func_name="dae_compiled_live_offset_for_sm",
        values=sm_live_offsets,
        default_value=0,
    )
    _emit_dense_lookup(
        lines,
        storage_name="daeCompiledLiveCountsBySm",
        func_name="dae_compiled_live_count_for_sm",
        values=sm_live_counts,
        default_value=0,
    )
    if table_storage_kind is not None:
        for program in spec["programs"]:
            program_id = int(program["program_id"])
            if program_id not in table_driven_programs:
                continue
            _emit_alloc_cmd_array(
                lines,
                name=f"daeCompiledAllocCmdsProgram{program_id}",
                storage=table_storage_kind,
                values=[_encode_alloc_cmd(step) for step in table_driven_programs[program_id]],
            )
    lines.extend(
        [
            "template <typename M2CQueue, typename C2MQueue>",
            "static __device__ __forceinline__ void dae_compiled_compute_execute(",
            "  int sm_id,",
            "  int thread_id,",
            "  void *smem_base,",
            "  uint64_t *scratch_space,",
            "  MInst *st_insts,",
            "  M2CQueue &m2c,",
            "  C2MQueue &c2m,",
            "  uint64_t *g_events",
            ") {",
            "  switch (dae_compiled_program_id_for_sm(sm_id)) {",
        ]
    )
    if debug:
        lines.insert(-1, "  uint32_t pc = 0;")
    for program in spec["programs"]:
        lines.append(f"    case {int(program['program_id'])}: {{")
        for group in _group_compute_entries(program["compute"]):
            entry = group["entry"]
            if group["kind"] == "repeat":
                lines.append(f"      for (int __ci = 0; __ci < {int(group['count'])}; ++__ci) {{")
                lines.append("        {")
                if debug:
                    lines.append(f"          pc = {int(group['start_index'])} + __ci + 1;")
                lines.extend(_emit_compute_inst_expr(entry, "          "))
                lines.append(
                    f"          dae_compute_handler_{entry['name']}(sm_id, thread_id, "
                    f"{'&pc' if debug else 'nullptr'}, nullptr, nullptr, inst, smem_base, scratch_space, st_insts, m2c, c2m, g_events);"
                )
                lines.append("        }")
                lines.append("      }")
                continue

            lines.append("      {")
            if debug:
                lines.append(f"        pc = {int(entry['index']) + 1};")
            if entry["name"] == "OP_TERMINATEC":
                lines.extend(
                    [
                        "        c2m.template push<0, true>(thread_id, 0);",
                        "        if (thread_id == 0) {",
                        "          int event_base = sm_id * numProfileEvents;",
                        "          g_events[event_base + 1] = cuda::ptx::get_sreg_globaltimer();",
                        "        }",
                        '        __cprint("TERMINATE from comptue: c2m.ptr=%d", c2m.ptr);',
                    ]
                )
            else:
                lines.extend(_emit_compute_inst_expr(entry, "        "))
                lines.append(
                    f"        dae_compute_handler_{entry['name']}(sm_id, thread_id, "
                    f"{'&pc' if debug else 'nullptr'}, nullptr, nullptr, inst, smem_base, scratch_space, st_insts, m2c, c2m, g_events);"
                )
            lines.append("      }")
        lines.append("      break;")
        lines.append("    }")
    lines.extend(
        [
            '    default: assert(false && "missing compiled compute program"); break;',
            "  }",
            "}",
            "",
            "template <typename M2CQueue, typename M2LDQueue>",
            "static __device__ __forceinline__ void dae_compiled_alloc_execute(",
            "  int sm_id,",
            "  int lane_id,",
            "  M2CQueue &m2c,",
            "  M2LDQueue m2ld[2],",
            "  MInst *st_insts,",
            "  const uint64_t *live_values,",
            "  uint32_t *shared_alloc_cmds,",
            "  int *flags",
            ") {",
            "  SharedMemoryAllocator<numSlots> alloc;",
            "  __syncwarp();",
            "  const uint32_t *alloc_cmds = nullptr;",
            "  int alloc_cmd_count = 0;",
            "  int alloc_terminate_count = 0;",
            "  switch (dae_compiled_program_id_for_sm(sm_id)) {",
        ]
    )
    for program in spec["programs"]:
        program_id = int(program["program_id"])
        lines.append(f"    case {int(program['program_id'])}: {{")
        if program_id in table_driven_programs:
            lines.append(f"      alloc_cmds = daeCompiledAllocCmdsProgram{program_id};")
            lines.append(f"      alloc_cmd_count = {len(table_driven_programs[program_id])};")
            lines.append(f"      alloc_terminate_count = {_alloc_terminate_count(program['memory'])};")
            lines.append("      break;")
        else:
            alloc_steps: list[dict[str, object]] = []
            for block in program["memory"]:
                if block["kind"] == "repeat":
                    alloc_steps.extend(block["steps"])
                elif block["kind"] == "op":
                    alloc_steps.append(block)
            _emit_payload_aliases(alloc_steps, lines, "      ")
            _emit_alloc_program(program["memory"], lines, "      ")
            lines.append("      return;")
        lines.append("    }")
    lines.extend(
        [
            '    default: assert(false && "missing compiled alloc program"); break;',
            "  }",
            f"  if (alloc_cmds != nullptr && alloc_cmd_count > 0 && {'true' if alloc_table_mode == _ALLOC_TABLE_MODE_SHARED else 'false'}) {{",
            "    for (int __copy = lane_id; __copy < alloc_cmd_count; __copy += 32) {",
            "      shared_alloc_cmds[__copy] = alloc_cmds[__copy];",
            "    }",
            "    __syncwarp();",
            "    alloc_cmds = shared_alloc_cmds;",
            "  }",
            f"  if (alloc_cmds != nullptr && alloc_cmd_count > 0 && {'true' if alloc_table_mode == _ALLOC_TABLE_MODE_GLOBAL else 'false'}) {{",
            "    if (lane_id < alloc_cmd_count && lane_id < 32) {",
            "      prefetch_l1(alloc_cmds + lane_id);",
            "    }",
            "    __syncwarp();",
            "  }",
            "  for (int __ai = 0; __ai < alloc_cmd_count; ++__ai) {",
            f"    if ({'true' if alloc_table_mode == _ALLOC_TABLE_MODE_GLOBAL else 'false'} && __ai + 32 < alloc_cmd_count && lane_id == (__ai & 31)) {{",
            "      prefetch_l1(alloc_cmds + __ai + 32);",
            "    }",
            "    const uint32_t __cmd = alloc_cmds[__ai];",
            "    const uint8_t __nslot = static_cast<uint8_t>(__cmd & 0xFFU);",
            "    const uint8_t __ld_port = static_cast<uint8_t>((__cmd >> 8) & 0x1U);",
            "    const uint8_t __direct_ready = static_cast<uint8_t>((__cmd >> 9) & 0x1U);",
            "    int alloc_mask = 0;",
            "    int slot_alloc = -1;",
            "    while (true) {",
            "      slot_alloc = alloc.allocate(lane_id, flags, __nslot, alloc_mask);",
            "      if (slot_alloc >= 0) break;",
            "      __nanosleep(allocRetrySleepCycles);",
            "    }",
            "    if (lane_id == 0) {",
            "      m2c.put(alloc_mask);",
            "      if (__direct_ready != 0) {",
            "        m2c.commit();",
            "        m2c.advance();",
            "      } else {",
            "        CompiledLdCmd ld;",
            "        ld.init(static_cast<uint8_t>(slot_alloc), static_cast<uint8_t>(m2c.ptr));",
            "        auto &curld = m2ld[__ld_port];",
            "        curld.put(ld.raw);",
            "        m2c.advance();",
            "        curld.commit();",
            "        curld.advance();",
            "      }",
            "    }",
            "  }",
            "  for (int __ti = 0; __ti < alloc_terminate_count; ++__ti) {",
            "    if (lane_id == 0) {",
            '      __mprint("[compiled alloc] terminate ld0_ptr=%d ld1_ptr=%d", m2ld[0].ptr, m2ld[1].ptr);',
            "    }",
            "  }",
            "}",
            "",
            "template <typename M2LDQueue, typename M2CQueue>",
            "static __device__ __forceinline__ void dae_compiled_ld0_execute(",
            "  int sm_id,",
            "  M2LDQueue &m2ld,",
            "  M2CQueue &m2c,",
            "  const uint64_t *live_values,",
            "  const void *smem_base,",
            "  const CUtensorMap *tma_descs,",
            "  int *bars",
            ") {",
            "  switch (dae_compiled_program_id_for_sm(sm_id)) {",
        ]
    )
    for program in spec["programs"]:
        lines.append(f"    case {int(program['program_id'])}: {{")
        _emit_ld_program(program, lines, "      ", 0)
        lines.append("      break;")
        lines.append("    }")
    lines.extend(
        [
            '    default: assert(false && "missing compiled ld0 program"); break;',
            "  }",
            "}",
            "",
            "template <typename M2LDQueue, typename M2CQueue>",
            "static __device__ __forceinline__ void dae_compiled_ld1_execute(",
            "  int sm_id,",
            "  M2LDQueue &m2ld,",
            "  M2CQueue &m2c,",
            "  const uint64_t *live_values,",
            "  const void *smem_base,",
            "  const CUtensorMap *tma_descs,",
            "  int *bars",
            ") {",
            "  switch (dae_compiled_program_id_for_sm(sm_id)) {",
        ]
    )
    for program in spec["programs"]:
        lines.append(f"    case {int(program['program_id'])}: {{")
        _emit_ld_program(program, lines, "      ", 1)
        lines.append("      break;")
        lines.append("    }")
    lines.extend(
        [
            '    default: assert(false && "missing compiled ld1 program"); break;',
            "  }",
            "}",
            "",
            "template <typename C2MQueue>",
            "static __device__ __forceinline__ void dae_compiled_st_execute(",
            "  int sm_id,",
            "  C2MQueue &c2m,",
            "  const uint64_t *live_values,",
            "  const void *smem_base,",
            "  const CUtensorMap *tma_descs,",
            "  int *bars",
            ") {",
            "  (void)bars;",
            "  switch (dae_compiled_program_id_for_sm(sm_id)) {",
        ]
    )
    for program in spec["programs"]:
        lines.append(f"    case {int(program['program_id'])}: {{")
        _emit_st_program(program["memory"], lines, "      ")
        lines.append("      break;")
        lines.append("    }")
    lines.extend(
        [
            '    default: assert(false && "missing compiled st program"); break;',
            "  }",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def generate_disabled(source: str, *, debug: bool, alloc_table_mode: str, live_value_mode: str) -> str:
    return "\n".join(
        [
            "// Generated by tools/generate_compiled_program.py.",
            f"// source={source}",
            f"// alloc_table_mode={alloc_table_mode}",
            f"// live_value_mode={live_value_mode}",
            "",
            "static constexpr bool daeCompiledProgramEnabled = false;",
            f"static constexpr bool daeCompiledProgramDebug = {'true' if debug else 'false'};",
            'static constexpr const char *daeCompiledProgramHash = "";',
            "static constexpr int daeCompiledProgramNumSms = 0;",
            "static constexpr int daeCompiledProgramLiveValueCount = 0;",
            "static constexpr int daeCompiledLiveValueMaxPerSm = 0;",
            "static constexpr int daeCompiledAllocMaxCmdCount = 0;",
            "static constexpr int daeCompiledLiveValueModeGlobal = 0;",
            "static constexpr int daeCompiledLiveValueModeShared = 1;",
            "static constexpr int daeCompiledLiveValueModeConstant = 2;",
            "static constexpr int daeCompiledLiveValueMode = daeCompiledLiveValueModeGlobal;",
            "",
            "static __device__ __forceinline__ int dae_compiled_live_offset_for_sm(int) {",
            "  return 0;",
            "}",
            "static __device__ __forceinline__ int dae_compiled_live_count_for_sm(int) {",
            "  return 0;",
            "}",
            "static __device__ __constant__ uint64_t daeCompiledLiveValuesConst[1] = {};",
            "",
            "template <typename... Args>",
            "static __device__ __forceinline__ void dae_compiled_compute_execute(Args&&...) {",
            '  assert(false && "compiled mode was not built into this runtime");',
            "}",
            "template <typename... Args>",
            "static __device__ __forceinline__ void dae_compiled_alloc_execute(Args&&...) {",
            '  assert(false && "compiled mode was not built into this runtime");',
            "}",
            "template <typename... Args>",
            "static __device__ __forceinline__ void dae_compiled_ld0_execute(Args&&...) {",
            '  assert(false && "compiled mode was not built into this runtime");',
            "}",
            "template <typename... Args>",
            "static __device__ __forceinline__ void dae_compiled_ld1_execute(Args&&...) {",
            '  assert(false && "compiled mode was not built into this runtime");',
            "}",
            "template <typename... Args>",
            "static __device__ __forceinline__ void dae_compiled_st_execute(Args&&...) {",
            '  assert(false && "compiled mode was not built into this runtime");',
            "}",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    spec_path, source = resolve_spec_path(ROOT)
    alloc_table_mode = load_alloc_table_mode()
    live_value_mode = load_live_value_mode()
    if spec_path is None or not spec_path.exists():
        output = generate_disabled(
            source,
            debug=args.debug,
            alloc_table_mode=alloc_table_mode,
            live_value_mode=live_value_mode,
        )
    else:
        spec = load_spec(spec_path)
        output = generate_enabled(
            spec,
            debug=args.debug,
            alloc_table_mode=alloc_table_mode,
            live_value_mode=live_value_mode,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
