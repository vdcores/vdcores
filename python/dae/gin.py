"""NCCL GIN lifecycle and single-window allocation for PoolInst.

The module is optional and imported only by the compile-time GIN runtime. It
keeps NCCL communicator/window setup on the host while every timed dispatch and
combine operation remains one VDCores PoolInst program.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
import os
from typing import Any

import torch

from . import runtime as _dae_runtime
from .pool_slice import (
    POOL_SLICE_CONFIG_BYTES,
    POOL_SLICE_CONTROL_WORDS,
    POOL_SLICE_MAX_DATA_GROUPS,
    POOL_SLICE_MAX_LOCAL_READERS,
    POOL_SLICE_MAX_PES,
    POOL_SLICE_MAX_POOL_BLOCKS,
    POOL_SLICE_METADATA_ENVELOPE_BYTES,
    POOL_SLICE_RECEIVE_BYTES,
    PoolSliceBuffers,
)


_ALIGNMENT = 4096


def _align(value: int) -> int:
    return (int(value) + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT


def _dtype_bytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


@dataclass(frozen=True)
class GinPoolAllocation:
    buffers: PoolSliceBuffers
    returned: torch.Tensor
    arena: torch.Tensor
    actual_contexts: int


@dataclass
class GinRuntime:
    """One MPI/NCCL rank owning a GDAKI device communicator."""

    mpi: Any
    communicator: Any
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    window: Any | None = None
    dev_comm: Any | None = None
    arena: torch.Tensor | None = None
    actual_contexts: int = 0

    @property
    def num_pes(self) -> int:
        return self.world_size

    @property
    def pe(self) -> int:
        return self.rank

    def benchmark_barrier(self) -> None:
        torch.cuda.synchronize(self.device)
        self.mpi.Barrier()

    @classmethod
    def init(cls) -> "GinRuntime":
        if not bool(getattr(_dae_runtime.config, "nccl_gin_enabled", False)):
            raise RuntimeError("dae.runtime was not built with NCCL GIN enabled")

        # GDAKI is the direct GPU/NIC backend used by this compiled PoolInst.
        os.environ.setdefault("NCCL_GIN_TYPE", "3")
        os.environ.setdefault("NCCL_SOCKET_IFNAME", "ibP2s2")
        from mpi4py import MPI
        import nccl.core as nccl

        world = MPI.COMM_WORLD
        rank = world.Get_rank()
        world_size = world.Get_size()
        local = world.Split_type(MPI.COMM_TYPE_SHARED, key=rank)
        local_rank = local.Get_rank()
        if local.Get_size() > torch.cuda.device_count():
            raise RuntimeError("MPI local ranks exceed visible CUDA devices")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        unique_id = nccl.get_unique_id() if rank == 0 else None
        unique_id = world.bcast(unique_id, root=0)
        communicator = nccl.Communicator.init(
            nranks=world_size, rank=rank, unique_id=unique_id
        )
        gin_type = communicator.gin_type
        if int(gin_type) != int(nccl.NcclGinType.GDAKI):
            communicator.destroy()
            raise RuntimeError(
                f"NCCL communicator selected GIN type {gin_type}, "
                "but the PoolInst build requires GDAKI"
            )
        return cls(
            mpi=world,
            communicator=communicator,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            device=device,
        )

    def install_arena(
        self,
        arena_bytes: int,
        *,
        context_count: int,
        queue_depth: int,
    ) -> torch.Tensor:
        if self.arena is not None:
            raise RuntimeError("a GIN arena is already installed")
        if arena_bytes <= 0 or context_count <= 0 or queue_depth <= 0:
            raise ValueError("GIN arena, context count, and queue depth must be positive")

        import nccl.core as nccl
        from nccl.core.interop.torch import empty as nccl_empty

        arena_bytes = _align(arena_bytes)
        arena = nccl_empty(
            arena_bytes, dtype=torch.uint8, device=self.device
        )
        arena.zero_()
        window = self.communicator.register_window(arena)
        if window is None or not window.is_valid:
            raise RuntimeError("NCCL could not register the PoolInst HBM window")
        requirements = nccl.NCCLDevCommRequirements(
            gin_force_enable=True,
            gin_connection_type=nccl.NcclGinConnectionType.FULL,
            gin_context_count=context_count,
            gin_exclusive_contexts=True,
            gin_queue_depth=queue_depth,
            gin_strong_signals_required=True,
            gin_va_signals_required=True,
        )
        # NCCL4Py 0.3.1 was compiled with 2.30.4 headers, but its requirement
        # POD is ABI-compatible with 2.30.x.  GDAKI's context layout changed in
        # 2.30.5 and NCCL selects that layout from this version field.  Stamp
        # the requirements with the loaded libnccl version so its host-side
        # context layout matches the device headers used to build VDCores.
        lib_version = nccl.get_lib_version()
        requirements._lowpp.version = (
            lib_version.major * 10000
            + lib_version.minor * 100
            + lib_version.micro
        )
        dev_comm = self.communicator.create_dev_comm(requirements=requirements)
        actual_contexts = int(
            _dae_runtime._configure_pool_gin_transport(
                int(dev_comm.ptr),
                int(window.handle),
                int(arena.data_ptr()),
                int(arena.numel()),
            )
        )
        if actual_contexts <= 0:
            raise RuntimeError("NCCL created no usable GIN contexts")
        self.arena = arena
        self.window = window
        self.dev_comm = dev_comm
        self.actual_contexts = actual_contexts
        self.mpi.Barrier()
        return arena

    def close(self) -> None:
        if self.communicator is None:
            return
        torch.cuda.synchronize(self.device)
        self.mpi.Barrier()
        if self.dev_comm is not None:
            self.dev_comm.close()
            self.dev_comm = None
        if self.window is not None:
            self.window.close()
            self.window = None
        self.communicator.destroy()
        self.communicator = None
        self.arena = None


def _pool_specs(
    *,
    num_pes: int,
    local_readers: int,
    token_capacity: int,
    route_capacity: int,
    expert_capacity_rows: int,
    hidden_size: int,
    dtype: torch.dtype,
    in_place_expert_output: bool,
) -> list[tuple[str, tuple[int, ...], torch.dtype]]:
    weighted_return = True
    token_row_planes = num_pes * 2
    delivery_rows = num_pes * token_capacity * 2
    return_inbox_rows = num_pes * token_capacity
    metadata_packet_bytes = _align(
        POOL_SLICE_METADATA_ENVELOPE_BYTES + route_capacity * 4
    )
    specs: list[tuple[str, tuple[int, ...], torch.dtype]] = [
        ("signals", (max(64, num_pes),), torch.uint64),
        ("send_offsets", (num_pes * local_readers + 1,), torch.uint32),
        ("send_rows", (route_capacity,), torch.uint32),
        ("send_origin_rows", (route_capacity,), torch.uint32),
        ("send_token_rows", (token_row_planes, token_capacity), torch.uint32),
        ("send_token_counts", (num_pes,), torch.uint32),
        ("token_pool", (token_capacity, hidden_size), dtype),
        ("delivery_pool", (delivery_rows, hidden_size), dtype),
        (
            "expert_input",
            (local_readers, expert_capacity_rows, hidden_size),
            dtype,
        ),
        ("return_inbox", (return_inbox_rows, hidden_size), dtype),
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
        ("returned", (token_capacity, hidden_size), dtype),
    ]
    if not in_place_expert_output:
        specs.insert(
            9,
            (
                "expert_output",
                (local_readers, expert_capacity_rows, hidden_size),
                dtype,
            ),
        )
    return specs


def _layout_bytes(
    specs: list[tuple[str, tuple[int, ...], torch.dtype]]
) -> int:
    cursor = 0
    for _, shape, dtype in specs:
        cursor = _align(cursor)
        cursor += prod(shape) * _dtype_bytes(dtype)
    return _align(cursor)


def _arena_views(
    arena: torch.Tensor,
    specs: list[tuple[str, tuple[int, ...], torch.dtype]],
) -> dict[str, torch.Tensor]:
    cursor = 0
    result: dict[str, torch.Tensor] = {}
    for name, shape, dtype in specs:
        cursor = _align(cursor)
        byte_count = prod(shape) * _dtype_bytes(dtype)
        result[name] = arena.narrow(0, cursor, byte_count).view(dtype).view(shape)
        cursor += byte_count
    if _align(cursor) > arena.numel():
        raise AssertionError("GIN pool layout exceeds its registered arena")
    return result


def allocate_pool_slice_gin(
    transport: GinRuntime,
    *,
    local_readers: int,
    token_capacity: int,
    route_capacity: int | None = None,
    expert_capacity_rows: int,
    hidden_size: int,
    dtype: torch.dtype = torch.bfloat16,
    signal_base: int = 0,
    group_limit: int = 0,
    pool_blocks: int = 64,
    in_place_expert_output: bool = True,
    context_count: int = 16,
    queue_depth: int = 1024,
) -> GinPoolAllocation:
    """Collectively allocate and install one weighted GIN logical pool."""

    num_pes = int(transport.world_size)
    my_pe = int(transport.rank)
    if not 1 <= num_pes <= POOL_SLICE_MAX_PES:
        raise ValueError("GIN pool PE count is outside the protocol range")
    if not 1 <= local_readers <= POOL_SLICE_MAX_LOCAL_READERS:
        raise ValueError("local_readers is outside the protocol range")
    if route_capacity is None:
        route_capacity = token_capacity
    for name, value in (
        ("token_capacity", token_capacity),
        ("route_capacity", route_capacity),
        ("expert_capacity_rows", expert_capacity_rows),
        ("hidden_size", hidden_size),
    ):
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    if not num_pes <= pool_blocks <= POOL_SLICE_MAX_POOL_BLOCKS:
        raise ValueError("weighted GIN requires pool_blocks in [num_pes, 132]")
    if expert_capacity_rows < num_pes * token_capacity:
        raise ValueError("expert capacity must reserve one source segment per PE")
    row_bytes = hidden_size * _dtype_bytes(dtype)
    if dtype != torch.bfloat16 or row_bytes < 1024 or row_bytes % 16:
        raise ValueError("weighted GIN requires aligned BF16 rows of at least 1 KiB")
    if signal_base < 0:
        raise ValueError("signal_base must be nonnegative")
    if group_limit == 0:
        payload_ctas = max(1, pool_blocks - 1)
        remote_targets = max(1, num_pes - 1)
        group_limit = min(
            token_capacity,
            POOL_SLICE_MAX_DATA_GROUPS,
            max(1, payload_ctas // remote_targets),
        )
    if not 1 <= group_limit <= POOL_SLICE_MAX_DATA_GROUPS:
        raise ValueError("group_limit is outside the protocol range")

    specs = _pool_specs(
        num_pes=num_pes,
        local_readers=local_readers,
        token_capacity=token_capacity,
        route_capacity=route_capacity,
        expert_capacity_rows=expert_capacity_rows,
        hidden_size=hidden_size,
        dtype=dtype,
        in_place_expert_output=in_place_expert_output,
    )
    arena = transport.install_arena(
        _layout_bytes(specs),
        context_count=context_count,
        queue_depth=queue_depth,
    )
    tensors = _arena_views(arena, specs)
    expert_output = (
        tensors["expert_input"]
        if in_place_expert_output
        else tensors["expert_output"]
    )
    write_chunk_rows = ((1 << 16) - 1) // row_bytes
    write_chunks = (token_capacity + write_chunk_rows - 1) // write_chunk_rows
    config_tensor = torch.empty(
        (pool_blocks, POOL_SLICE_CONFIG_BYTES),
        dtype=torch.uint8,
        device=transport.device,
    )
    buffers = PoolSliceBuffers(
        signals=tensors["signals"],
        send_offsets=tensors["send_offsets"],
        send_rows=tensors["send_rows"],
        send_origin_rows=tensors["send_origin_rows"],
        send_token_rows=tensors["send_token_rows"],
        send_token_counts=tensors["send_token_counts"],
        token_pool=tensors["token_pool"],
        delivery_pool=tensors["delivery_pool"],
        expert_input=tensors["expert_input"],
        expert_output=expert_output,
        return_inbox=tensors["return_inbox"],
        send_batches=tensors["send_batches"],
        receive_batches=tensors["receive_batches"],
        combine_rows=tensors["combine_rows"],
        receive_routes=tensors["receive_routes"],
        sequence=tensors["sequence"],
        control=tensors["control"],
        config_tensor=config_tensor,
        num_pes=num_pes,
        my_pe=my_pe,
        local_readers=local_readers,
        token_capacity=token_capacity,
        route_capacity=route_capacity,
        expert_capacity_rows=expert_capacity_rows,
        signal_base=signal_base,
        group_limit=group_limit,
        write_chunks=write_chunks,
        write_chunk_rows=write_chunk_rows,
        pool_count=pool_blocks,
        weighted_return=True,
        transport="nccl_gin",
        transport_arena=arena,
        transport_owner=transport,
        data_plane_arena=arena,
    )
    transport.mpi.Barrier()
    return GinPoolAllocation(
        buffers=buffers,
        returned=tensors["returned"],
        arena=arena,
        actual_contexts=transport.actual_contexts,
    )


__all__ = [
    "GinPoolAllocation",
    "GinRuntime",
    "allocate_pool_slice_gin",
]
