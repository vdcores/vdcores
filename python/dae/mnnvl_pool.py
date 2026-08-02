"""Rank-per-GPU CUDA Fabric allocation for one NVL72 PoolSlice domain.

MPI is used only to exchange opaque CUDA Fabric handles and to coordinate
setup.  PoolInst metadata, activation, signal, and multimem traffic use the
NVLink fabric through process-local CUDA VMM mappings.
"""

from __future__ import annotations

from typing import Any

import torch

from . import runtime
from .local_pool import _bytes, select_local_group_limit, select_local_pool_blocks
from .pool_slice import (
    POOL_SLICE_CONFIG_BYTES,
    POOL_SLICE_CONTROL_WORDS,
    POOL_SLICE_MAX_LOCAL_READERS,
    POOL_SLICE_MAX_PES,
    POOL_SLICE_MAX_POOL_BLOCKS,
    POOL_SLICE_MAX_TMA_BYTES,
    POOL_SLICE_METADATA_ENVELOPE_BYTES,
    POOL_SLICE_RECEIVE_BYTES,
    PoolSliceBuffers,
)


def allocate_mnnvl_pool_slice(
    *,
    comm: Any,
    device: int,
    local_readers: int,
    token_capacity: int,
    expert_capacity_rows: int,
    hidden_size: int,
    route_capacity: int | None = None,
    pool_blocks: int | None = None,
    group_limit: int = 0,
    reduction_backend: str = "multimem",
    static_routes: bool = False,
    in_place_expert_output: bool = False,
) -> PoolSliceBuffers:
    """Allocate one rank's slice and map every peer through Fabric handles."""

    rank = int(comm.Get_rank())
    num_pes = int(comm.Get_size())
    device = int(device)
    if not 1 <= num_pes <= POOL_SLICE_MAX_PES:
        raise ValueError("invalid MNNVL rank count")
    if not 0 <= rank < num_pes:
        raise ValueError("invalid MNNVL rank")
    if not 0 <= device < torch.cuda.device_count():
        raise ValueError(f"invalid local CUDA device {device}")
    if not 1 <= local_readers <= POOL_SLICE_MAX_LOCAL_READERS:
        raise ValueError("invalid local_readers")
    if reduction_backend not in (
        "forward",
        "multimem",
        "source_gather",
    ):
        raise ValueError(
            "reduction_backend must be 'forward', 'multimem', or "
            "'source_gather'; peer-direct return is not a valid global "
            "top-k reduction"
        )
    if route_capacity is None:
        route_capacity = token_capacity
    if expert_capacity_rows < num_pes * token_capacity:
        raise ValueError("expert capacity must contain one segment per source")

    torch.cuda.set_device(device)
    if pool_blocks is None:
        local_sm_count = torch.cuda.get_device_properties(
            device
        ).multi_processor_count
        sm_counts = [int(value) for value in comm.allgather(local_sm_count)]
        pool_blocks = select_local_pool_blocks(
            num_pes=num_pes,
            token_capacity=token_capacity,
            local_readers=local_readers,
            sm_counts=sm_counts,
        )
    if not 1 <= pool_blocks <= POOL_SLICE_MAX_POOL_BLOCKS:
        raise ValueError("invalid pool_blocks")
    if pool_blocks < num_pes:
        raise ValueError("reduction requires at least one PoolInst block per GPU")

    row_bytes = hidden_size * torch.empty(
        (), dtype=torch.bfloat16
    ).element_size()
    if row_bytes < 1024 or row_bytes % 16:
        raise ValueError("BF16 row width must be >=1024 and 16-byte aligned")
    write_chunk_rows = POOL_SLICE_MAX_TMA_BYTES // row_bytes
    write_chunks = (token_capacity + write_chunk_rows - 1) // write_chunk_rows
    if group_limit == 0:
        group_limit = select_local_group_limit(token_capacity)

    metadata_packet_bytes = (
        POOL_SLICE_METADATA_ENVELOPE_BYTES + 2 * route_capacity * 4 + 15
    ) // 16 * 16
    delivery_planes = num_pes * (2 if reduction_backend == "forward" else 1)
    specs: list[tuple[str, tuple[int, ...], torch.dtype]] = [
        ("signals", (num_pes,), torch.uint64),
        ("send_offsets", (num_pes * local_readers + 1,), torch.uint32),
        ("send_rows", (route_capacity,), torch.uint32),
        ("send_origin_rows", (route_capacity,), torch.uint32),
        ("send_token_rows", (2 * num_pes, token_capacity), torch.uint32),
        ("send_token_counts", (num_pes,), torch.uint32),
        ("token_pool", (token_capacity, hidden_size), torch.bfloat16),
        (
            "delivery_pool",
            (delivery_planes * token_capacity, hidden_size),
            torch.bfloat16,
        ),
        (
            "expert_input",
            (local_readers, expert_capacity_rows, hidden_size),
            torch.bfloat16,
        ),
        (
            "return_inbox",
            (num_pes * token_capacity, hidden_size),
            torch.bfloat16,
        ),
        ("send_batches", (num_pes * metadata_packet_bytes,), torch.uint8),
        ("receive_batches", (num_pes * metadata_packet_bytes,), torch.uint8),
        (
            "combine_rows",
            (local_readers, num_pes, token_capacity),
            torch.uint64,
        ),
        (
            "receive_routes",
            (local_readers, num_pes, POOL_SLICE_RECEIVE_BYTES),
            torch.uint8,
        ),
        ("sequence", (1,), torch.uint64),
        ("control", (POOL_SLICE_CONTROL_WORDS,), torch.uint64),
    ]
    if not in_place_expert_output:
        specs.insert(
            9,
            (
                "expert_output",
                (local_readers, expert_capacity_rows, hidden_size),
                torch.bfloat16,
            ),
        )
    offsets: dict[str, tuple[int, tuple[int, ...], torch.dtype]] = {}
    cursor = 0
    for name, shape, dtype in specs:
        cursor = (cursor + 255) // 256 * 256
        offsets[name] = (cursor, shape, dtype)
        cursor += _bytes(shape, dtype)
    arena_bytes = (cursor + 4095) // 4096 * 4096

    from . import _local_pool_runtime

    arena, arena_handle, mapped_arena_bytes = (
        _local_pool_runtime.allocate_fabric_arena(device, arena_bytes)
    )
    arena_descriptors = comm.allgather(
        (bytes(arena_handle), int(mapped_arena_bytes))
    )
    if any(size != int(mapped_arena_bytes) for _, size in arena_descriptors):
        raise RuntimeError("MNNVL ranks computed different arena sizes")
    peer_bases: list[int] = []
    for pe, (peer_handle, peer_bytes) in enumerate(arena_descriptors):
        if pe == rank:
            peer_bases.append(arena.data_ptr())
        else:
            peer_bases.append(
                int(
                    _local_pool_runtime.import_fabric_arena(
                        device, peer_handle, peer_bytes
                    )
                )
            )

    multicast_control_bytes = 4096
    multicast_plane_bytes = num_pes * token_capacity * row_bytes
    multicast_bytes = multicast_control_bytes + multicast_plane_bytes
    multicast_descriptor = None
    multicast_allocation_id = None
    if rank == 0:
        (
            multicast_allocation_id,
            multicast_handle,
            mapped_multicast_bytes,
            multicast_alignment,
        ) = _local_pool_runtime.create_fabric_multicast(
            device, num_pes, multicast_bytes
        )
        multicast_descriptor = (
            bytes(multicast_handle),
            int(mapped_multicast_bytes),
            int(multicast_alignment),
        )
    multicast_handle, mapped_multicast_bytes, multicast_alignment = comm.bcast(
        multicast_descriptor, root=0
    )
    if rank != 0:
        multicast_allocation_id = (
            _local_pool_runtime.import_fabric_multicast(
                device,
                multicast_handle,
                mapped_multicast_bytes,
                multicast_alignment,
            )
        )
    assert multicast_allocation_id is not None
    _local_pool_runtime.add_fabric_multicast_device(
        multicast_allocation_id, device
    )
    comm.Barrier()
    multicast_unicast, multicast_alias = (
        _local_pool_runtime.bind_fabric_multicast(
            multicast_allocation_id, device, multicast_bytes
        )
    )
    torch.cuda.synchronize(device)
    comm.Barrier()

    views: dict[str, torch.Tensor] = {}
    for name, (offset, shape, dtype) in offsets.items():
        views[name] = (
            arena.narrow(0, offset, _bytes(shape, dtype)).view(dtype).view(shape)
        )
    if in_place_expert_output:
        views["expert_output"] = views["expert_input"]
    runtime.configure_local_pool_runtime(
        arena.data_ptr(),
        peer_bases,
        multicast_unicast.data_ptr() + multicast_control_bytes,
        int(multicast_alias) + multicast_control_bytes,
    )
    config_tensor = torch.empty(
        (pool_blocks, POOL_SLICE_CONFIG_BYTES),
        dtype=torch.uint8,
        device=device,
    )
    buffers = PoolSliceBuffers(
        **views,
        config_tensor=config_tensor,
        num_pes=num_pes,
        my_pe=rank,
        local_readers=local_readers,
        token_capacity=token_capacity,
        route_capacity=route_capacity,
        expert_capacity_rows=expert_capacity_rows,
        signal_base=0,
        group_limit=group_limit,
        write_chunks=write_chunks,
        write_chunk_rows=write_chunk_rows,
        pool_count=pool_blocks,
        weighted_return=True,
        transport="local",
        reduction_backend=reduction_backend,
        static_routes=bool(static_routes),
        data_plane_arena=arena,
    )
    buffers.set_static_routes(static_routes)
    buffers._local_multicast_backing = multicast_unicast
    buffers._mnnvl_peer_bases = tuple(peer_bases)
    buffers._mnnvl_multicast_allocation_id = int(multicast_allocation_id)
    buffers._mnnvl_arena_mapped_bytes = int(mapped_arena_bytes)
    buffers._mnnvl_multicast_mapped_bytes = int(mapped_multicast_bytes)
    if reduction_backend == "multimem":
        result_offset = (
            multicast_control_bytes + rank * token_capacity * row_bytes
        )
        buffers._local_reduction_output = (
            multicast_unicast.narrow(
                0, result_offset, token_capacity * row_bytes
            )
            .view(torch.bfloat16)
            .view(token_capacity, hidden_size)
        )
    return buffers


__all__ = ["allocate_mnnvl_pool_slice"]
