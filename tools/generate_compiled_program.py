#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPILED_SPEC_FILE = "dae_compiled_program.vdcore.json"
COMPILED_SPEC_ENV = "DAE_COMPILED_SPEC_FILE"

_ADDR_OPS = {
    "OP_ALLOC_TMA_LOAD_1D",
    "OP_ALLOC_WB_TMA_STORE_1D",
}
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


def _emit_repeat_inst_expr(step: dict[str, object], indent: str) -> list[str]:
    inst_index = step["inst_index"]
    delta_seed_index = step["delta_seed_index"]
    base_name = step["base_name"]
    lines = [f"{indent}MInst inst = minsts[{inst_index}];"]
    lines.append(f"{indent}const MInst &delta_inst = minsts[{delta_seed_index}];")
    lines.append(f"{indent}if (__rep != 0) {{")
    if base_name in _ADDR_OPS:
        lines.append(f"{indent}  inst.address += static_cast<uint64_t>(__rep) * delta_inst.address;")
    else:
        lines.append(f"{indent}  inst.coords[0] += static_cast<uint16_t>(__rep * delta_inst.coords[0]);")
        lines.append(f"{indent}  inst.coords[1] += static_cast<uint16_t>(__rep * delta_inst.coords[1]);")
        lines.append(f"{indent}  inst.coords[2] += static_cast<uint16_t>(__rep * delta_inst.coords[2]);")
        lines.append(f"{indent}  inst.coords[3] += static_cast<uint16_t>(__rep * delta_inst.coords[3]);")
    lines.append(f"{indent}}}")
    return lines


def _emit_linear_inst_expr(inst_index: int, indent: str) -> list[str]:
    return [f"{indent}const MInst &inst = minsts[{inst_index}];"]


def _emit_alloc_block(block: dict[str, object], lines: list[str], indent: str) -> None:
    if block["kind"] == "repeat":
        count_seed_index = block["count_seed_index"]
        lines.append(f"{indent}for (int __rep = 0; __rep < static_cast<int>(minsts[{count_seed_index}].size); ++__rep) {{")
        for step in block["steps"]:
            lines.append(f"{indent}  {{")
            inst_index = step["inst_index"]
            lines.extend(_emit_linear_inst_expr(inst_index, indent + "    "))
            lines.append(f"{indent}    int alloc_mask = 0;")
            lines.append(f"{indent}    int slot_alloc = -1;")
            lines.append(f"{indent}    while (true) {{")
            lines.append(f"{indent}      slot_alloc = alloc.allocate(lane_id, flags, inst.nslot(), alloc_mask);")
            lines.append(f"{indent}      if (slot_alloc >= 0) break;")
            lines.append(f"{indent}      __nanosleep(allocRetrySleepCycles);")
            lines.append(f"{indent}    }}")
            lines.append(f"{indent}    if (lane_id == 0) {{")
            lines.append(f"{indent}      m2c.put(alloc_mask);")
            lines.append(f"{indent}      CompiledLdCmd ld;")
            lines.append(f"{indent}      ld.init(static_cast<uint8_t>(slot_alloc), static_cast<uint8_t>(m2c.ptr));")
            lines.append(f"{indent}      auto &curld = m2ld[(inst.opcode & MEM_OP_FLAGS_PORT) ? 1 : 0];")
            lines.append(f"{indent}      curld.put(ld.raw);")
            lines.append(f"{indent}      m2c.advance();")
            lines.append(f"{indent}      curld.commit();")
            lines.append(f"{indent}      curld.advance();")
            lines.append(f"{indent}    }}")
            lines.append(f"{indent}  }}")
        lines.append(f"{indent}}}")
        return

    if block["kind"] == "op":
        lines.append(f"{indent}{{")
        inst_index = block["inst_index"]
        lines.extend(_emit_linear_inst_expr(inst_index, indent + "  "))
        lines.append(f"{indent}  int alloc_mask = 0;")
        lines.append(f"{indent}  int slot_alloc = -1;")
        lines.append(f"{indent}  while (true) {{")
        lines.append(f"{indent}    slot_alloc = alloc.allocate(lane_id, flags, inst.nslot(), alloc_mask);")
        lines.append(f"{indent}    if (slot_alloc >= 0) break;")
        lines.append(f"{indent}    __nanosleep(allocRetrySleepCycles);")
        lines.append(f"{indent}  }}")
        lines.append(f"{indent}  if (lane_id == 0) {{")
        lines.append(f"{indent}    m2c.put(alloc_mask);")
        lines.append(f"{indent}    CompiledLdCmd ld;")
        lines.append(f"{indent}    ld.init(static_cast<uint8_t>(slot_alloc), static_cast<uint8_t>(m2c.ptr));")
        lines.append(f"{indent}    auto &curld = m2ld[(inst.opcode & MEM_OP_FLAGS_PORT) ? 1 : 0];")
        lines.append(f"{indent}    curld.put(ld.raw);")
        lines.append(f"{indent}    m2c.advance();")
        lines.append(f"{indent}    curld.commit();")
        lines.append(f"{indent}    curld.advance();")
        lines.append(f"{indent}  }}")
        lines.append(f"{indent}}}")
        return

    if block["kind"] == "terminate":
        lines.append(f"{indent}if (lane_id == 0) {{")
        lines.append(f"{indent}  m2ld[0].push(CompiledLdCmd::end().raw);")
        lines.append(f"{indent}  m2ld[1].push(CompiledLdCmd::end().raw);")
        lines.append(f"{indent}}}")
        return

    raise ValueError(f"Unknown alloc block kind {block['kind']}")


def _emit_ld_step(step: dict[str, object], lines: list[str], indent: str) -> None:
    base_name = step["base_name"]
    lines.append(f"{indent}{{")
    if base_name in _LOAD_OPS:
        lines.append(f"{indent}  CompiledLdCmd cmd {{}};")
        lines.append(f"{indent}  cmd.raw = m2ld.pop();")
        lines.append(f"{indent}  if (cmd.slot == SLOT_END) {{ assert(false && \"compiled ld step terminated early\"); return; }}")
        lines.extend(_emit_repeat_inst_expr(step, indent + "  ") if "delta_seed_index" in step else _emit_linear_inst_expr(step["inst_index"], indent + "  "))
        if base_name == "OP_ALLOC_TMA_LOAD_1D":
            lines.append(f"{indent}  cuda::device::memcpy_async_tx(")
            lines.append(f"{indent}      static_cast<char *>(get_slot_address(smem_base, cmd.slot)),")
            lines.append(f"{indent}      reinterpret_cast<char *>(inst.address),")
            lines.append(f"{indent}      cuda::aligned_size_t<16>(inst.size),")
            lines.append(f"{indent}      m2c.barriers[cmd.m2c_slot]);")
            lines.append(f"{indent}  cuda::device::barrier_expect_tx(m2c.barriers[cmd.m2c_slot], cuda::aligned_size_t<16>(inst.size));")
        elif base_name == "OP_ALLOC_TMA_LOAD_TENSOR_1D":
            lines.append(f"{indent}  asm volatile(")
            lines.append(f'{indent}  "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes"')
            lines.append(f'{indent}  "[%0], [%1, {{%2}}], [%3];\\n"')
            lines.append(f"{indent}  :")
            lines.append(f"{indent}  : \"r\"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, cmd.slot))),")
            lines.append(f"{indent}    \"l\"((void *)(tma_descs + inst.arg)),")
            lines.append(f"{indent}    \"r\"((uint32_t)inst.address),")
            lines.append(f"{indent}    \"r\"((uint32_t)__cvta_generic_to_shared(m2c.native_bar(cmd.m2c_slot)))")
            lines.append(f"{indent}  : \"memory\");")
            lines.append(f"{indent}  cuda::device::barrier_expect_tx(m2c.barriers[cmd.m2c_slot], cuda::aligned_size_t<16>(inst.size));")
        elif base_name == "OP_ALLOC_TMA_LOAD_2D":
            lines.append(f"{indent}  asm volatile(")
            lines.append(f'{indent}  "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"')
            lines.append(f'{indent}  "[%0], [%1, {{%2, %3}}], [%4];\\n"')
            lines.append(f"{indent}  :")
            lines.append(f"{indent}  : \"r\"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, cmd.slot))),")
            lines.append(f"{indent}    \"l\"((void *)(tma_descs + inst.arg)),")
            lines.append(f"{indent}    \"r\"((int)inst.coords[0]),")
            lines.append(f"{indent}    \"r\"((int)inst.coords[1]),")
            lines.append(f"{indent}    \"r\"((uint32_t)__cvta_generic_to_shared(m2c.native_bar(cmd.m2c_slot)))")
            lines.append(f"{indent}  : \"memory\");")
            lines.append(f"{indent}  cuda::device::barrier_expect_tx(m2c.barriers[cmd.m2c_slot], cuda::aligned_size_t<16>(inst.size));")
        elif base_name == "OP_ALLOC_TMA_LOAD_3D":
            lines.append(f"{indent}  asm volatile(")
            lines.append(f'{indent}  "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"')
            lines.append(f'{indent}  "[%0], [%1, {{%2, %3, %4}}], [%5];\\n"')
            lines.append(f"{indent}  :")
            lines.append(f"{indent}  : \"r\"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, cmd.slot))),")
            lines.append(f"{indent}    \"l\"((void *)(tma_descs + inst.arg)),")
            lines.append(f"{indent}    \"r\"((int)inst.coords[0]),")
            lines.append(f"{indent}    \"r\"((int)inst.coords[1]),")
            lines.append(f"{indent}    \"r\"((int)inst.coords[2]),")
            lines.append(f"{indent}    \"r\"((uint32_t)__cvta_generic_to_shared(m2c.native_bar(cmd.m2c_slot)))")
            lines.append(f"{indent}  : \"memory\");")
            lines.append(f"{indent}  cuda::device::barrier_expect_tx(m2c.barriers[cmd.m2c_slot], cuda::aligned_size_t<16>(inst.size));")
        elif base_name == "OP_ALLOC_TMA_LOAD_4D":
            lines.append(f"{indent}  asm volatile(")
            lines.append(f'{indent}  "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes"')
            lines.append(f'{indent}  "[%0], [%1, {{%2, %3, %4, %5}}], [%6];\\n"')
            lines.append(f"{indent}  :")
            lines.append(f"{indent}  : \"r\"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, cmd.slot))),")
            lines.append(f"{indent}    \"l\"((void *)(tma_descs + inst.arg)),")
            lines.append(f"{indent}    \"r\"((int)inst.coords[0]),")
            lines.append(f"{indent}    \"r\"((int)inst.coords[1]),")
            lines.append(f"{indent}    \"r\"((int)inst.coords[2]),")
            lines.append(f"{indent}    \"r\"((int)inst.coords[3]),")
            lines.append(f"{indent}    \"r\"((uint32_t)__cvta_generic_to_shared(m2c.native_bar(cmd.m2c_slot)))")
            lines.append(f"{indent}  : \"memory\");")
            lines.append(f"{indent}  cuda::device::barrier_expect_tx(m2c.barriers[cmd.m2c_slot], cuda::aligned_size_t<16>(inst.size));")
        elif base_name == "OP_ALLOC_TMA_LOAD_5D_FIX0":
            lines.append(f"{indent}  asm volatile(")
            lines.append(f'{indent}  "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes"')
            lines.append(f'{indent}  "[%0], [%1, {{0, %2, %3, %4, %5}}], [%6];\\n"')
            lines.append(f"{indent}  :")
            lines.append(f"{indent}  : \"r\"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, cmd.slot))),")
            lines.append(f"{indent}    \"l\"((void *)(tma_descs + inst.arg)),")
            lines.append(f"{indent}    \"r\"((int)inst.coords[0]),")
            lines.append(f"{indent}    \"r\"((int)inst.coords[1]),")
            lines.append(f"{indent}    \"r\"((int)inst.coords[2]),")
            lines.append(f"{indent}    \"r\"((int)inst.coords[3]),")
            lines.append(f"{indent}    \"r\"((uint32_t)__cvta_generic_to_shared(m2c.native_bar(cmd.m2c_slot)))")
            lines.append(f"{indent}  : \"memory\");")
            lines.append(f"{indent}  cuda::device::barrier_expect_tx(m2c.barriers[cmd.m2c_slot], cuda::aligned_size_t<16>(inst.size));")
        lines.append(f"{indent}  (void)m2c.barriers[cmd.m2c_slot].arrive();")
        lines.append(f"{indent}}}")
        return

    if base_name in _WRITEBACK_OPS:
        lines.append(f"{indent}  CompiledLdCmd cmd {{}};")
        lines.append(f"{indent}  cmd.raw = m2ld.pop();")
        lines.append(f"{indent}  if (cmd.slot == SLOT_END) {{ assert(false && \"compiled ld step terminated early\"); return; }}")
        lines.append(f"{indent}  (void)m2c.barriers[cmd.m2c_slot].arrive();")
        lines.append(f"{indent}}}")
        return

    raise ValueError(f"Unsupported LDU step {base_name}")


def _emit_ld_program(blocks: list[dict[str, object]], port_id: int, lines: list[str], indent: str) -> None:
    for block in blocks:
        if block["kind"] == "repeat":
            port_steps = [step for step in block["steps"] if step.get("port", 0) == port_id]
            if not port_steps:
                continue
            count_seed_index = block["count_seed_index"]
            lines.append(f"{indent}for (int __rep = 0; __rep < static_cast<int>(minsts[{count_seed_index}].size); ++__rep) {{")
            for step in port_steps:
                _emit_ld_step(step, lines, indent + "  ")
            lines.append(f"{indent}}}")
            continue
        if block["kind"] == "op":
            if block.get("port", 0) == port_id:
                _emit_ld_step(block, lines, indent)
            continue
    lines.append(f"{indent}CompiledLdCmd __done {{}};")
    lines.append(f"{indent}__done.raw = m2ld.pop();")
    lines.append(f"{indent}assert(__done.slot == SLOT_END && \"compiled ld expected SLOT_END\");")


def _emit_st_step(step: dict[str, object], lines: list[str], indent: str) -> None:
    base_name = step["base_name"]
    lines.append(f"{indent}{{")
    lines.append(f"{indent}  int slot_mask = c2m.pop();")
    lines.append(f"{indent}  if (!slot_mask) {{ assert(false && \"compiled st step terminated early\"); return; }}")
    lines.append(f"{indent}  int slot = extract(slot_mask);")
    lines.extend(_emit_repeat_inst_expr(step, indent + "  ") if "delta_seed_index" in step else _emit_linear_inst_expr(step["inst_index"], indent + "  "))
    if base_name == "OP_ALLOC_WB_TMA_STORE_1D":
        lines.append(f"{indent}  cuda::ptx::cp_async_bulk(")
        lines.append(f"{indent}    cuda::ptx::space_global,")
        lines.append(f"{indent}    cuda::ptx::space_shared,")
        lines.append(f"{indent}    (void *)(inst.address),")
        lines.append(f"{indent}    (const void *)(get_slot_address(smem_base, slot)),")
        lines.append(f"{indent}    inst.size);")
        lines.append(f"{indent}  cuda::ptx::cp_async_bulk_commit_group();")
    elif base_name == "OP_ALLOC_WB_TMA_STORE_2D":
        lines.append(f"{indent}  asm volatile(")
        lines.append(f'{indent}  "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "')
        lines.append(f'{indent}  "[%0, {{%1, %2}}], [%3];\\n"')
        lines.append(f"{indent}  :")
        lines.append(f"{indent}  : \"l\"((void *)(tma_descs + inst.arg)),")
        lines.append(f"{indent}    \"r\"((int)inst.coords[0]),")
        lines.append(f"{indent}    \"r\"((int)inst.coords[1]),")
        lines.append(f"{indent}    \"r\"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))")
        lines.append(f"{indent}  : \"memory\");")
        lines.append(f"{indent}  cuda::ptx::cp_async_bulk_commit_group();")
    elif base_name == "OP_ALLOC_WB_TMA_STORE_3D":
        lines.append(f"{indent}  asm volatile(")
        lines.append(f'{indent}  "cp.async.bulk.tensor.3d.global.shared::cta.bulk_group "')
        lines.append(f'{indent}  "[%0, {{%1, %2, %3}}], [%4];\\n"')
        lines.append(f"{indent}  :")
        lines.append(f"{indent}  : \"l\"((void *)(tma_descs + inst.arg)),")
        lines.append(f"{indent}    \"r\"((int)inst.coords[0]),")
        lines.append(f"{indent}    \"r\"((int)inst.coords[1]),")
        lines.append(f"{indent}    \"r\"((int)inst.coords[2]),")
        lines.append(f"{indent}    \"r\"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))")
        lines.append(f"{indent}  : \"memory\");")
        lines.append(f"{indent}  cuda::ptx::cp_async_bulk_commit_group();")
    elif base_name == "OP_ALLOC_WB_TMA_STORE_4D":
        lines.append(f"{indent}  asm volatile(")
        lines.append(f'{indent}  "cp.async.bulk.tensor.4d.global.shared::cta.bulk_group "')
        lines.append(f'{indent}  "[%0, {{%1, %2, %3, %4}}], [%5];\\n"')
        lines.append(f"{indent}  :")
        lines.append(f"{indent}  : \"l\"((void *)(tma_descs + inst.arg)),")
        lines.append(f"{indent}    \"r\"((int)inst.coords[0]),")
        lines.append(f"{indent}    \"r\"((int)inst.coords[1]),")
        lines.append(f"{indent}    \"r\"((int)inst.coords[2]),")
        lines.append(f"{indent}    \"r\"((int)inst.coords[3]),")
        lines.append(f"{indent}    \"r\"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))")
        lines.append(f"{indent}  : \"memory\");")
        lines.append(f"{indent}  cuda::ptx::cp_async_bulk_commit_group();")
    elif base_name == "OP_ALLOC_WB_TMA_STORE_5D_FIX0":
        lines.append(f"{indent}  asm volatile(")
        lines.append(f'{indent}  "cp.async.bulk.tensor.5d.global.shared::cta.bulk_group "')
        lines.append(f'{indent}  "[%0, {{0, %1, %2, %3, %4}}], [%5];\\n"')
        lines.append(f"{indent}  :")
        lines.append(f"{indent}  : \"l\"((void *)(tma_descs + inst.arg)),")
        lines.append(f"{indent}    \"r\"((int)inst.coords[0]),")
        lines.append(f"{indent}    \"r\"((int)inst.coords[1]),")
        lines.append(f"{indent}    \"r\"((int)inst.coords[2]),")
        lines.append(f"{indent}    \"r\"((int)inst.coords[3]),")
        lines.append(f"{indent}    \"r\"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))")
        lines.append(f"{indent}  : \"memory\");")
        lines.append(f"{indent}  cuda::ptx::cp_async_bulk_commit_group();")
    elif base_name == "OP_ALLOC_WB_TMA_REDUCE_ADD_2D":
        lines.append(f"{indent}  asm volatile(")
        lines.append(f'{indent}  "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group "')
        lines.append(f'{indent}  "[%0, {{%1, %2}}], [%3];\\n"')
        lines.append(f"{indent}  :")
        lines.append(f"{indent}  : \"l\"((void *)(tma_descs + inst.arg)),")
        lines.append(f"{indent}    \"r\"((int)inst.coords[0]),")
        lines.append(f"{indent}    \"r\"((int)inst.coords[1]),")
        lines.append(f"{indent}    \"r\"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))")
        lines.append(f"{indent}  : \"memory\");")
        lines.append(f"{indent}  cuda::ptx::cp_async_bulk_commit_group();")
    elif base_name == "OP_ALLOC_WB_TMA_REDUCE_ADD_3D":
        lines.append(f"{indent}  asm volatile(")
        lines.append(f'{indent}  "cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.bulk_group "')
        lines.append(f'{indent}  "[%0, {{%1, %2, %3}}], [%4];\\n"')
        lines.append(f"{indent}  :")
        lines.append(f"{indent}  : \"l\"((void *)(tma_descs + inst.arg)),")
        lines.append(f"{indent}    \"r\"((int)inst.coords[0]),")
        lines.append(f"{indent}    \"r\"((int)inst.coords[1]),")
        lines.append(f"{indent}    \"r\"((int)inst.coords[2]),")
        lines.append(f"{indent}    \"r\"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))")
        lines.append(f"{indent}  : \"memory\");")
        lines.append(f"{indent}  cuda::ptx::cp_async_bulk_commit_group();")
    else:
        raise ValueError(f"Unsupported STU step {base_name}")
    lines.append(f"{indent}  cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<0>{{}});")
    lines.append(f"{indent}  c2m.reset(slot_mask);")
    lines.append(f"{indent}}}")


def _emit_st_program(blocks: list[dict[str, object]], lines: list[str], indent: str) -> None:
    write_blocks: list[dict[str, object]] = []
    for block in blocks:
        if block["kind"] == "repeat":
            steps = [step for step in block["steps"] if step["writeback"]]
            if steps:
                write_blocks.append(
                    {
                        "kind": "repeat",
                        "count_seed_index": block["count_seed_index"],
                        "steps": steps,
                    }
                )
        elif block["kind"] == "op" and block["writeback"]:
            write_blocks.append(block)
    if not write_blocks:
        lines.append(f"{indent}int __done = c2m.pop();")
        lines.append(f"{indent}assert(__done == 0 && \"compiled st expected terminate sentinel\");")
        return
    for block in write_blocks:
        if block["kind"] == "repeat":
            lines.append(f"{indent}for (int __rep = 0; __rep < static_cast<int>(minsts[{block['count_seed_index']}].size); ++__rep) {{")
            for step in block["steps"]:
                _emit_st_step(step, lines, indent + "  ")
            lines.append(f"{indent}}}")
        else:
            _emit_st_step(block, lines, indent)
    lines.append(f"{indent}int __done = c2m.pop();")
    lines.append(f"{indent}assert(__done == 0 && \"compiled st expected terminate sentinel\");")


def generate_enabled(spec: dict[str, object]) -> str:
    lines = [
        "// Generated by tools/generate_compiled_program.py.",
        f'// compiled_hash={spec["hash"]}',
        "",
        "static constexpr bool daeCompiledProgramEnabled = true;",
        f'static constexpr const char *daeCompiledProgramHash = "{spec["hash"]}";',
        f'static constexpr int daeCompiledProgramNumSms = {spec["num_sms"]};',
        "",
        "static __device__ __forceinline__ int dae_compiled_program_id_for_sm(int sm_id) {",
        "  switch (sm_id) {",
    ]
    for sm_id, program_id in enumerate(spec["sm_program_ids"]):
        lines.append(f"    case {sm_id}: return {program_id};")
    lines.extend(
        [
            "    default: return -1;",
            "  }",
            "}",
            "",
            "template <typename M2CQueue, typename C2MQueue>",
            "static __device__ __forceinline__ void dae_compiled_compute_execute(",
            "  int sm_id,",
            "  int thread_id,",
            "  const CInst *cinsts,",
            "  void *smem_base,",
            "  uint64_t *scratch_space,",
            "  MInst *st_insts,",
            "  M2CQueue &m2c,",
            "  C2MQueue &c2m,",
            "  uint64_t *g_events",
            ") {",
            "  uint32_t pc = 0;",
            "  uint32_t count = 0;",
            "  bool finish = false;",
            "  switch (dae_compiled_program_id_for_sm(sm_id)) {",
        ]
    )
    for program in spec["programs"]:
        lines.append(f'    case {program["program_id"]}: {{')
        for entry in program["compute"]:
            idx = entry["index"]
            name = entry["name"]
            lines.append(f"      pc = {idx + 1};")
            lines.append(f"      {{ const CInst &inst = cinsts[{idx}]; dae_compute_handler_{name}(sm_id, thread_id, pc, count, finish, inst, smem_base, scratch_space, st_insts, m2c, c2m, g_events); }}")
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
            "  const MInst *minsts,",
            "  int *flags",
            ") {",
            "  SharedMemoryAllocator<numSlots> alloc;",
            "  __syncwarp();",
            "  switch (dae_compiled_program_id_for_sm(sm_id)) {",
        ]
    )
    for program in spec["programs"]:
        lines.append(f'    case {program["program_id"]}: {{')
        for block in program["memory"]:
            _emit_alloc_block(block, lines, "      ")
        lines.append("      break;")
        lines.append("    }")
    lines.extend(
        [
            '    default: assert(false && "missing compiled alloc program"); break;',
            "  }",
            "}",
            "",
            "template <typename M2LDQueue, typename M2CQueue>",
            "static __device__ __forceinline__ void dae_compiled_ld_execute(",
            "  int sm_id,",
            "  int port_id,",
            "  M2LDQueue &m2ld,",
            "  M2CQueue &m2c,",
            "  const MInst *minsts,",
            "  const void *smem_base,",
            "  const CUtensorMap *tma_descs,",
            "  int *bars",
            ") {",
            "  (void)bars;",
            "  switch (dae_compiled_program_id_for_sm(sm_id)) {",
        ]
    )
    for program in spec["programs"]:
        lines.append(f'    case {program["program_id"]}: {{')
        port_blocks = []
        for block in program["memory"]:
            if block["kind"] == "repeat":
                port_blocks.append(
                    {
                        "kind": "repeat",
                        "count_seed_index": block["count_seed_index"],
                        "steps": [
                            {
                                **step,
                                "port": 1 if step.get("port1") else 0,
                            }
                            for step in block["steps"]
                        ],
                    }
                )
            else:
                port_blocks.append({**block, "port": 1 if block.get("port1") else 0})
        lines.append("      if (port_id == 0) {")
        _emit_ld_program(port_blocks, 0, lines, "        ")
        lines.append("      } else {")
        _emit_ld_program(port_blocks, 1, lines, "        ")
        lines.append("      }")
        lines.append("      break;")
        lines.append("    }")
    lines.extend(
        [
            '    default: assert(false && "missing compiled ld program"); break;',
            "  }",
            "}",
            "",
            "template <typename C2MQueue>",
            "static __device__ __forceinline__ void dae_compiled_st_execute(",
            "  int sm_id,",
            "  C2MQueue &c2m,",
            "  const MInst *minsts,",
            "  const void *smem_base,",
            "  const CUtensorMap *tma_descs,",
            "  int *bars",
            ") {",
            "  (void)bars;",
            "  switch (dae_compiled_program_id_for_sm(sm_id)) {",
        ]
    )
    for program in spec["programs"]:
        lines.append(f'    case {program["program_id"]}: {{')
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


def generate_disabled(source: str) -> str:
    return "\n".join(
        [
            "// Generated by tools/generate_compiled_program.py.",
            f"// source={source}",
            "",
            "static constexpr bool daeCompiledProgramEnabled = false;",
            'static constexpr const char *daeCompiledProgramHash = "";',
            "static constexpr int daeCompiledProgramNumSms = 0;",
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
            "static __device__ __forceinline__ void dae_compiled_ld_execute(Args&&...) {",
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
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    spec_path, source = resolve_spec_path(ROOT)
    if spec_path is None or not spec_path.exists():
        output = generate_disabled(source)
    else:
        spec = load_spec(spec_path)
        output = generate_enabled(spec)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
