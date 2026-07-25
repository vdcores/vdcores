from __future__ import annotations

import pytest
import torch

from dae.instructions import (
    MemoryPoolRun,
    MemoryPoolSubmit,
    MemoryPoolWait,
    NvshmemPut,
    NvshmemWait,
)
from dae.memory_pool import (
    CONFIG_BYTES,
    NO_DEPENDENCY,
    REQUEST_BYTES,
    MemoryPoolConfig,
    MemoryPoolDependencyDeadlock,
    MemoryPoolFlags,
    MemoryPoolOpcode,
    MemoryPoolRequest,
    reference_gather_rows,
    reference_scatter_rows,
    resolve_dependency_order,
)
from dae.runtime import opcode


def _fields(instruction) -> list[int]:
    return instruction.tensor().view(torch.uint16).tolist()


def test_request_abi_is_128_bytes_and_round_trips():
    request = MemoryPoolRequest(
        sequence=17,
        opcode=MemoryPoolOpcode.SCATTER,
        source_address=0x123456789ABCDEF0,
        destination_address=0x1111222233334444,
        route_address=0xAAAABBBBCCCCDDDD,
        pool_offset=4096,
        bytes=8192,
        wait_value=16,
        signal_delta=2,
        flags=MemoryPoolFlags.REDUCE_SUM_F32,
        source_pe=3,
        target_pe=4,
        completion_pe=3,
        completion_signal=41,
        wait_slot=7,
        signal_slot=8,
        row_bytes=256,
        row_count=12,
        source_stride=512,
        destination_stride=1024,
        user_tag=99,
    )

    payload = request.pack()
    assert len(payload) == REQUEST_BYTES
    assert MemoryPoolRequest.unpack(payload) == request


def test_config_abi_is_128_bytes_and_checks_submit_signal_span():
    config = MemoryPoolConfig(
        mailboxes_address=1,
        pool_data_address=2,
        data_scratch_address=3,
        route_scratch_address=4,
        dependencies_address=5,
        consumed_sequences_address=6,
        control_address=7,
        pool_bytes=4096,
        data_scratch_bytes=1024,
        mailbox_count=16,
        dependency_count=8,
        submit_signal_base=4,
        signal_count=64,
        route_capacity=32,
    )
    assert len(config.pack()) == CONFIG_BYTES

    with pytest.raises(ValueError, match="submit signal range"):
        MemoryPoolConfig(
            **{**config.__dict__, "submit_signal_base": 60}
        ).pack()


def test_dependency_scheduler_holds_early_read_until_sixteen_writes():
    read = MemoryPoolRequest(
        sequence=1,
        opcode=MemoryPoolOpcode.READ,
        wait_slot=3,
        wait_value=16,
        user_tag=1000,
    )
    writes = [
        MemoryPoolRequest(
            sequence=1,
            opcode=MemoryPoolOpcode.WRITE,
            signal_slot=3,
            signal_delta=1,
            user_tag=index,
        )
        for index in range(16)
    ]

    order, dependencies = resolve_dependency_order([read, *writes])

    assert order == [*range(1, 17), 0]
    assert dependencies[3] == 16


def test_dependency_scheduler_reports_unsatisfied_cycle():
    request = MemoryPoolRequest(
        sequence=1,
        opcode=MemoryPoolOpcode.READ,
        wait_slot=0,
        wait_value=1,
        user_tag=77,
    )
    with pytest.raises(MemoryPoolDependencyDeadlock, match="77"):
        resolve_dependency_order([request])


def test_reference_scatter_and_gather_restore_token_order():
    source = torch.arange(24, dtype=torch.float32).view(6, 4)
    routes = torch.tensor([4, 1, 5, 0, 3, 2])
    pool = torch.full((6, 4), -1.0)

    reference_scatter_rows(source, routes, pool)
    gathered = reference_gather_rows(pool, routes)

    assert torch.equal(gathered, source)
    assert torch.equal(pool[4], source[0])
    assert torch.equal(pool[0], source[3])


def test_reference_top1_ep_routes_across_four_pes():
    num_pes = 4
    tokens_per_pe = 3
    hidden_size = 2
    total_tokens = num_pes * tokens_per_pe
    pool = torch.zeros(num_pes * total_tokens, hidden_size)
    sources = [
        torch.arange(
            pe * tokens_per_pe * hidden_size,
            (pe + 1) * tokens_per_pe * hidden_size,
            dtype=torch.float32,
        ).view(tokens_per_pe, hidden_size)
        for pe in range(num_pes)
    ]

    origin_routes = []
    for pe, source in enumerate(sources):
        global_ids = [pe * tokens_per_pe + token for token in range(tokens_per_pe)]
        routes = [
            (global_id % num_pes) * total_tokens + global_id
            for global_id in global_ids
        ]
        origin_routes.append(routes)
        reference_scatter_rows(source, routes, pool)

    for expert in range(num_pes):
        global_ids = [
            global_id
            for global_id in range(total_tokens)
            if global_id % num_pes == expert
        ]
        routes = [expert * total_tokens + global_id for global_id in global_ids]
        expert_input = reference_gather_rows(pool, routes)
        expert_output = expert_input * (expert + 1) + expert * 0.25
        reference_scatter_rows(expert_output, routes, pool)

    for pe, source in enumerate(sources):
        result = reference_gather_rows(pool, origin_routes[pe])
        global_ids = [pe * tokens_per_pe + token for token in range(tokens_per_pe)]
        expert_ids = torch.tensor(
            [global_id % num_pes for global_id in global_ids], dtype=torch.float32
        ).unsqueeze(1)
        expected = source * (expert_ids + 1) + expert_ids * 0.25
        assert torch.equal(result, expected)


def test_memory_pool_instructions_are_nonallocating_and_encode_operands():
    submit = MemoryPoolSubmit(0x123456789ABCDEF0, pool_pe=9, submit_signal=257)
    wait = MemoryPoolWait(0x123456789ABCDEF0)
    run = MemoryPoolRun(0x1111222233334444, expected_requests=0x12345678)

    submit_fields = _fields(submit)
    wait_fields = _fields(wait)
    run_fields = _fields(run)

    assert submit_fields[:4] == [opcode.OP_MEMORY_POOL_SUBMIT, 257, 0, 9]
    assert wait_fields[:4] == [opcode.OP_MEMORY_POOL_WAIT, 0, 0, 0]
    assert run_fields[:4] == [opcode.OP_MEMORY_POOL_RUN, 0x5678, 0x1234, 0]
    assert submit.opcode & 1 == 0
    assert wait.opcode & 1 == 0
    assert run.opcode & 1 == 0
    assert submit.requires_signal_array
    assert wait.requires_signal_array
    assert run.requires_signal_array


def test_corrected_issue_25_put_wait_encoding_includes_signal_id():
    put = NvshmemPut(
        0x123456789ABCDEF0,
        nbytes=0x12345678,
        target_pe=7,
        signal_id=11,
    )
    wait = NvshmemWait(signal_id=11, value=23)

    assert _fields(put)[:4] == [
        opcode.OP_NVSHMEM_PUT,
        0x5678,
        0x1234,
        (11 << 8) | 7,
    ]
    assert _fields(wait)[:4] == [opcode.OP_NVSHMEM_WAIT, 11, 0, 0]
    assert put.opcode & 1 == 0
    assert wait.opcode & 1 == 0
    assert put.requires_signal_array
    assert wait.requires_signal_array


def test_control_pointer_can_contain_full_uint16_address_words():
    address = 0xFFFF0000FFFF0000
    fields = _fields(MemoryPoolWait(address))
    assert fields[4:] == [0x0000, 0xFFFF, 0x0000, 0xFFFF]


def test_no_dependency_sentinel_is_uint32_max():
    assert NO_DEPENDENCY == 0xFFFFFFFF
