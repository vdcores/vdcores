from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import torch

from dae.ep_pool import (
    EP_BATCH_BYTES,
    EP_CONFIG_BYTES,
    EP_PROFILE_DISPATCH_READY,
    EP_PROFILE_DONE,
    EP_PROFILE_START,
    ExpertPoolBatch,
    ExpertPoolBatchFlags,
    ExpertPoolConfig,
    ExpertPoolProgram,
    group_routes_by_expert,
)


ROOT = Path(__file__).resolve().parents[1]


def _config(**updates) -> ExpertPoolConfig:
    values = dict(
        source_address=1,
        packed_source_address=2,
        expert_input_address=3,
        expert_output_address=4,
        return_inbox_address=5,
        returned_address=6,
        send_offsets_address=7,
        send_rows_address=8,
        send_origin_rows_address=9,
        send_batches_address=10,
        receive_batches_address=11,
        expert_tails_address=12,
        sequence_address=13,
        control_address=14,
        row_bytes=1024,
        source_stride=1024,
        expert_row_stride=1024,
        return_stride=1024,
        expert_stride=16 * 1024,
        active_rows=8,
        route_capacity=8,
        expert_capacity_rows=16,
        num_experts=4,
        experts_per_pe=2,
        num_pes=2,
        my_pe=0,
        dispatch_signal_base=0,
        return_signal_base=8,
        reset_signal_base=12,
        signal_count=14,
        source_capacity_rows=8,
        return_capacity_rows=8,
    )
    values.update(updates)
    return ExpertPoolConfig(**values)


def test_expert_pool_batch_abi_round_trips():
    batch = ExpertPoolBatch(
        sequence=17,
        base_row=123,
        source_base=7,
        row_count=9,
        source_pe=3,
        local_expert=2,
        flags=ExpertPoolBatchFlags.ERROR,
    )
    payload = batch.pack()
    assert len(payload) == EP_BATCH_BYTES
    assert ExpertPoolBatch.unpack(payload) == batch


def test_expert_pool_config_abi_and_ranges():
    assert len(_config().pack()) == EP_CONFIG_BYTES

    try:
        _config(return_signal_base=11).pack()
    except ValueError as error:
        assert "return signal range" in str(error)
    else:
        raise AssertionError("invalid signal range was accepted")

    try:
        _config(reset_signal_base=13).pack()
    except ValueError as error:
        assert "reset signal range" in str(error)
    else:
        raise AssertionError("invalid reset signal range was accepted")

    try:
        _config(expert_stride=1023).pack()
    except ValueError as error:
        assert "expert_stride" in str(error)
    else:
        raise AssertionError("short expert stride was accepted")

    try:
        _config(active_rows=9, route_capacity=8).pack()
    except ValueError as error:
        assert "route_capacity" in str(error)
    else:
        raise AssertionError("active rows beyond route capacity were accepted")

    try:
        _config(return_capacity_rows=0).pack()
    except ValueError as error:
        assert "return_capacity_rows" in str(error)
    else:
        raise AssertionError("zero return capacity was accepted")

    try:
        _config(row_bytes=1008).pack()
    except ValueError as error:
        assert "row_bytes" in str(error)
    else:
        raise AssertionError("sub-1KiB rows were accepted")


def test_group_routes_is_stable_and_emits_compressed_offsets():
    offsets, rows, origins = group_routes_by_expert(
        [2, 0, 2, 1, 0, 3],
        num_experts=4,
        source_rows=[10, 11, 12, 13, 14, 15],
        origin_rows=[20, 21, 22, 23, 24, 25],
    )

    assert offsets.dtype == torch.uint32
    assert offsets.tolist() == [0, 2, 3, 5, 6]
    assert rows.tolist() == [11, 14, 13, 10, 12, 15]
    assert origins.tolist() == [21, 24, 23, 20, 22, 25]


def test_group_routes_rejects_invalid_expert():
    try:
        group_routes_by_expert([0, 4], num_experts=4)
    except ValueError as error:
        assert "expert_ids" in str(error)
    else:
        raise AssertionError("invalid expert id was accepted")


def test_group_routes_supports_repeated_topk_source_rows():
    offsets, rows, origins = group_routes_by_expert(
        [0, 1, 1, 2],
        num_experts=3,
        source_rows=[0, 0, 1, 1],
        origin_rows=[0, 1, 2, 3],
    )
    assert offsets.tolist() == [0, 1, 3, 4]
    assert rows.tolist() == [0, 0, 1, 1]
    assert origins.tolist() == [0, 1, 2, 3]


def test_expert_pool_timing_uses_internal_profile_events():
    profile = torch.zeros((3, 128), dtype=torch.uint64)
    profile[:, EP_PROFILE_START] = torch.tensor([100, 110, 120])
    profile[:, EP_PROFILE_DISPATCH_READY] = torch.tensor([150, 160, 170])
    profile[:, EP_PROFILE_DONE] = torch.tensor([220, 230, 240])
    program = ExpertPoolProgram(
        launcher=SimpleNamespace(profile=profile),
        reset_barrier=0,
        dispatch_barriers=(),
        compute_barriers=(),
        chunk_rows=1,
    )

    assert program.timing_ns() == (70, 70, 140)


def _function_source(path: Path, function_name: str) -> str:
    source = path.read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    segment = ast.get_source_segment(source, function)
    assert segment is not None
    return segment


def test_nccl_comparison_is_external_to_vdcores_sources():
    source_roots = (
        ROOT / "src",
        ROOT / "include",
        ROOT / "python" / "dae",
        ROOT / "app" / "python" / "memory_pool",
    )
    forbidden = ("NCCL_ALGO", "torch.distributed", "torch.cuda.Event")
    source_suffixes = {".cc", ".cpp", ".cu", ".cuh", ".h", ".py"}
    for source_root in source_roots:
        for path in source_root.rglob("*"):
            if path.suffix not in source_suffixes:
                continue
            text = path.read_text()
            for marker in forbidden:
                assert marker not in text, f"{marker} leaked into {path.relative_to(ROOT)}"

    benchmark = ROOT / "benchmarks" / "ep_pool_nccl_compare.py"
    pool_source = _function_source(benchmark, "run_pool")
    nccl_source = _function_source(benchmark, "run_nccl")
    assert "program.timing_ns()" in pool_source
    assert "torch.cuda.Event" not in pool_source
    assert "elapsed_time" not in pool_source
    assert "event_quadruple()" in nccl_source
