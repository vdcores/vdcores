"""Single-process, NVSHMEM-free local-NVLink pool allocation."""

from __future__ import annotations

import math
import torch

from . import runtime
from .pool_slice import (
    POOL_SLICE_CONFIG_BYTES,
    POOL_SLICE_CONTROL_WORDS,
    POOL_SLICE_MAX_DATA_GROUPS,
    POOL_SLICE_MAX_LOCAL_READERS,
    POOL_SLICE_MAX_PES,
    POOL_SLICE_MAX_POOL_BLOCKS,
    POOL_SLICE_MAX_TMA_BYTES,
    POOL_SLICE_METADATA_ENVELOPE_BYTES,
    POOL_SLICE_RECEIVE_BYTES,
    PoolSliceBuffers,
)


def _bytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    return math.prod(shape) * torch.empty((), dtype=dtype).element_size()


def select_local_pool_blocks(
    *,
    num_pes: int,
    token_capacity: int,
    local_readers: int,
    sm_counts: list[int] | tuple[int, ...],
) -> int:
    """Choose the resident PoolInst CTA count for local GB300 reduction.

    Twenty CTAs minimize fixed queue/arbitration latency for a 32-token batch.
    On two PEs the four-vector GB300 reduction kernel favors about 3.5 source
    rows per aggregate pool CTA (37 CTAs at 128 tokens). Three peers create
    more independent work, where the retained 11-row source-shard policy is
    better. Four peers use one CTA beyond the 64-CTA symmetric boundary once
    the row-derived policy saturates. Always leave room for the configured
    reader CTAs.
    """

    if len(sm_counts) != num_pes or num_pes <= 0:
        raise ValueError("sm_counts must contain one entry per PE")
    resident_limit = min(int(count) - local_readers for count in sm_counts)
    resident_limit = min(resident_limit, POOL_SLICE_MAX_POOL_BLOCKS)
    resident_limit -= resident_limit % num_pes
    if resident_limit < num_pes:
        raise ValueError("insufficient SMs for one pool CTA per PE")
    if num_pes == 2:
        target = max(20, math.ceil(2 * token_capacity / 7))
    else:
        # Three peers create more independent send/gather work per source CTA,
        # so fewer reduction shards are needed than on the two-peer path.
        # Across rank-per-GPU Fabric mappings, 128-token sweeps at 8 and 16
        # PEs show that executor counts above 64 only lengthen ordered dispatch
        # retirement.  Cap the larger-topology policy before the resident
        # limit; smaller token shapes can still select fewer CTAs.
        rows_per_shard = 11 if num_pes == 3 else 8
        row_target = num_pes * math.ceil(token_capacity / rows_per_shard)
        target = max(20, row_target)
        target = math.ceil(target / num_pes) * num_pes
        if num_pes >= 4:
            target = min(target, 64)
        # Four peers sit on a sharp scheduler/executor boundary at 64 CTAs:
        # rank zero is scheduler-only, leaving 63 reduction executors.  One
        # extra CTA removes the final uneven shard without changing the
        # publisher/route/send role layout.  Keep smaller four-peer batches on
        # their row-derived policy and retain the profiled 64-CTA cap above
        # four peers.
        if num_pes == 4 and target == 64 and resident_limit >= 65:
            target = 65
    return min(target, resident_limit)


def select_local_group_limit(token_capacity: int) -> int:
    """Cap NVLink streaming groups at one group per 64 source tokens."""

    if token_capacity <= 0:
        raise ValueError("token_capacity must be positive")
    return min(
        token_capacity,
        POOL_SLICE_MAX_DATA_GROUPS,
        max(1, math.ceil(token_capacity / 64)),
    )


def allocate_local_pool_slices(
    *,
    devices: list[int] | tuple[int, ...],
    local_readers: int,
    token_capacity: int,
    expert_capacity_rows: int,
    hidden_size: int,
    route_capacity: int | None = None,
    pool_blocks: int | None = None,
    group_limit: int = 0,
    reduction_backend: str = "multimem",
    in_place_expert_output: bool = False,
) -> list[PoolSliceBuffers]:
    """Allocate identical per-GPU arenas and configure direct peer mappings."""

    devices = [int(device) for device in devices]
    num_pes = len(devices)
    if not 1 <= num_pes <= POOL_SLICE_MAX_PES:
        raise ValueError("invalid local GPU count")
    if len(set(devices)) != num_pes:
        raise ValueError("devices must be unique")
    if not 1 <= local_readers <= POOL_SLICE_MAX_LOCAL_READERS:
        raise ValueError("invalid local_readers")
    if pool_blocks is None:
        sm_counts = [
            torch.cuda.get_device_properties(device).multi_processor_count
            for device in devices
        ]
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
    if reduction_backend not in ("forward", "multimem"):
        raise ValueError("reduction_backend must be 'forward' or 'multimem'")
    if route_capacity is None:
        route_capacity = token_capacity
    if expert_capacity_rows < num_pes * token_capacity:
        raise ValueError("expert capacity must contain one segment per source")
    row_bytes = hidden_size * 2
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
        ("delivery_pool", (delivery_planes * token_capacity, hidden_size), torch.bfloat16),
        ("expert_input", (local_readers, expert_capacity_rows, hidden_size), torch.bfloat16),
        ("return_inbox", (num_pes * token_capacity, hidden_size), torch.bfloat16),
        ("send_batches", (num_pes * metadata_packet_bytes,), torch.uint8),
        ("receive_batches", (num_pes * metadata_packet_bytes,), torch.uint8),
        ("combine_rows", (local_readers, num_pes, token_capacity), torch.uint64),
        ("receive_routes", (local_readers, num_pes, POOL_SLICE_RECEIVE_BYTES), torch.uint8),
        ("sequence", (1,), torch.uint64),
        ("control", (POOL_SLICE_CONTROL_WORDS,), torch.uint64),
    ]
    if not in_place_expert_output:
        specs.insert(
            9,
            ("expert_output", (local_readers, expert_capacity_rows, hidden_size), torch.bfloat16),
        )

    offsets: dict[str, tuple[int, tuple[int, ...], torch.dtype]] = {}
    cursor = 0
    for name, shape, dtype in specs:
        cursor = (cursor + 255) // 256 * 256
        offsets[name] = (cursor, shape, dtype)
        cursor += _bytes(shape, dtype)
    arena_bytes = (cursor + 4095) // 4096 * 4096

    from . import _local_pool_runtime

    _local_pool_runtime.enable_peer_access(devices)
    arenas: list[torch.Tensor] = []
    fields: list[dict[str, torch.Tensor]] = []
    for device in devices:
        with torch.cuda.device(device):
            arena = torch.zeros(arena_bytes, dtype=torch.uint8, device=device)
        arenas.append(arena)
        views: dict[str, torch.Tensor] = {}
        for name, (offset, shape, dtype) in offsets.items():
            views[name] = arena.narrow(0, offset, _bytes(shape, dtype)).view(dtype).view(shape)
        if in_place_expert_output:
            views["expert_output"] = views["expert_input"]
        fields.append(views)

    # Reserve a prefix for multicast reduction semaphores. The configured
    # device pointers address the data plane immediately after this prefix.
    multicast_control_bytes = 4096
    multicast_plane_bytes = num_pes * token_capacity * row_bytes
    multicast_bytes = multicast_control_bytes + multicast_plane_bytes
    multicast_unicast, multicast_aliases, _ = _local_pool_runtime.allocate_multicast(
        devices, multicast_bytes
    )
    peer_bases = [arena.data_ptr() for arena in arenas]
    buffers: list[PoolSliceBuffers] = []
    for pe, device in enumerate(devices):
        views = fields[pe]
        with torch.cuda.device(device):
            runtime.configure_local_pool_runtime(
                arenas[pe].data_ptr(),
                peer_bases,
                multicast_unicast[pe].data_ptr() + multicast_control_bytes,
                multicast_aliases[pe] + multicast_control_bytes,
            )
            config_tensor = torch.empty(
                (pool_blocks, POOL_SLICE_CONFIG_BYTES),
                dtype=torch.uint8,
                device=device,
            )
        buffers.append(PoolSliceBuffers(
            **views,
            config_tensor=config_tensor,
            num_pes=num_pes,
            my_pe=pe,
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
            data_plane_arena=arenas[pe],
        ))
    # Retain multicast tensors for the life of the buffer set.
    for buffer, backing in zip(buffers, multicast_unicast):
        buffer._local_multicast_backing = backing
        if reduction_backend == "multimem":
            result_offset = (
                multicast_control_bytes
                + buffer.my_pe * token_capacity * row_bytes
            )
            buffer._local_reduction_output = (
                backing.narrow(
                    0, result_offset, token_capacity * row_bytes
                )
                .view(torch.bfloat16)
                .view(token_capacity, hidden_size)
            )
    return buffers


__all__ = [
    "allocate_local_pool_slices",
    "select_local_group_limit",
    "select_local_pool_blocks",
]
