"""Unified pool-slice dependent-read protocol and VDCores program builder.

Applications write source activations and reader-grouped route metadata in
HBM. A VDCores writer stores each activation once in source-owned token slots.
In parallel, a communication-specialized VDCores macro operator publishes
metadata, packs unique `(token, target PE)` delivery batches, resolves
dynamic-read placement, moves contiguous batches, and returns reader output to
its origins.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, IntFlag
import struct
from typing import Sequence, TYPE_CHECKING

import torch

from .runtime import config as runtime_config

if TYPE_CHECKING:
    from .launcher import Launcher


POOL_SLICE_MAX_PES = 32
POOL_SLICE_MAX_LOCAL_READERS = 8
POOL_SLICE_MAX_POOL_BLOCKS = 32
POOL_SLICE_MAX_EXTERNAL_REDUCERS = 32
POOL_SLICE_AVAILABLE_WORKER_WARPS = (
    int(getattr(runtime_config, "pool_slice_warps", 8)) - 1
)
POOL_SLICE_LARGE_PACK_BYTES = 512 * 1024
POOL_SLICE_SIGNAL_PHASES = 3
POOL_SLICE_PUBLISH_BYTES = 64
POOL_SLICE_RECEIVE_BYTES = 48
POOL_SLICE_CONFIG_BYTES = 256
POOL_SLICE_CONTROL_WORDS = 166
POOL_SLICE_CONTROL_READER_ROW_COUNT = 158
POOL_SLICE_MAX_TMA_BYTES = (1 << 16) - 1
POOL_SLICE_PROFILE_START = 5
POOL_SLICE_PROFILE_GATHER_READY = 6
POOL_SLICE_PROFILE_DONE = 7
POOL_SLICE_PROFILE_DATA_PUBLISHED = 8
POOL_SLICE_PROFILE_FIRST_PAYLOAD = 9
POOL_SLICE_PROFILE_METADATA_CLOSED = 10
POOL_SLICE_PROFILE_PAYLOAD_DONE = 11
POOL_SLICE_PROFILE_FIRST_DATA_PUBLISHED = 12
POOL_SLICE_PROFILE_COMPUTE_READY = 13
POOL_SLICE_PROFILE_RETURN_PAYLOAD_DONE = 14
POOL_SLICE_PROFILE_RETURN_SIGNALS_CLOSED = 15
POOL_SLICE_PROFILE_SCATTER_DONE = 16
POOL_SLICE_PROFILE_EXTERNAL_REDUCE_START = 17
POOL_SLICE_PROFILE_EXTERNAL_REDUCE_DONE = 18
POOL_SLICE_PROFILE_EXTERNAL_ZERO_START = 19
POOL_SLICE_PROFILE_EXTERNAL_ZERO_DONE = 20
POOL_SLICE_PROFILE_RETURN_REDUCE_START = 21
POOL_SLICE_PROFILE_RETURN_REDUCE_DONE = 22
POOL_SLICE_PROFILE_FIRST_RETURN_PUT = 23
POOL_SLICE_PROFILE_RETURN_CTA_DONE = 24

_PUBLISH_STRUCT = struct.Struct("<Q6I8I")
_RECEIVE_STRUCT = struct.Struct("<QQ5I3I")
_CONFIG_STRUCT = struct.Struct("<19Q24I8x")
assert _PUBLISH_STRUCT.size == POOL_SLICE_PUBLISH_BYTES
assert _RECEIVE_STRUCT.size == POOL_SLICE_RECEIVE_BYTES
assert _CONFIG_STRUCT.size == POOL_SLICE_CONFIG_BYTES


class PoolSliceStatus(IntEnum):
    OK = 0
    BAD_CONFIG = 1
    SEQUENCE = 2
    BATCH = 3
    ROUTE_RANGE = 4
    CAPACITY = 5
    SIGNAL_RANGE = 6


class PoolSliceBatchFlags(IntFlag):
    NONE = 0
    ERROR = 1 << 0


class PoolSliceDispatchMode(IntEnum):
    POOL_GATHER = 0


class PoolSliceFlags(IntFlag):
    NONE = 0
    DEDICATED_COORDINATOR = 1 << 0
    PUT_PHASE_WORDS = 1 << 1
    PIPELINED_RETURN = 1 << 2
    READER_PIPELINE = 1 << 3
    WEIGHTED_RETURN = 1 << 4
    EXTERNAL_WEIGHTED_REDUCER = 1 << 5
    EXTERNAL_TOKEN_REDUCER = 1 << 6


def _uint(name: str, value: int, bits: int) -> int:
    value = int(value)
    if not 0 <= value < 1 << bits:
        raise ValueError(f"{name} must fit in uint{bits}")
    return value


def _positive_uint(name: str, value: int, bits: int) -> int:
    value = _uint(name, value, bits)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _symmetric_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
    from . import nvshmem

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError(f"{name} must be a contiguous CUDA tensor")
    if not nvshmem.is_symmetric_tensor(tensor):
        raise ValueError(f"{name} must be allocated from NVSHMEM symmetric HBM")
    return tensor


def _copy_packed(destination: torch.Tensor, payload: bytes) -> None:
    if destination.dtype != torch.uint8 or destination.ndim != 1:
        raise ValueError("packed destination must be a rank-1 uint8 tensor")
    if destination.numel() != len(payload):
        raise ValueError("packed destination size does not match payload")
    destination.copy_(torch.tensor(list(payload), dtype=torch.uint8))


@dataclass(frozen=True)
class PoolSlicePublishBatch:
    sequence: int
    source_pe: int
    target_pe: int
    active_rows: int
    route_begin: int
    route_end: int
    reader_counts: tuple[int, ...] = (0,) * POOL_SLICE_MAX_LOCAL_READERS
    flags: PoolSliceBatchFlags | int = PoolSliceBatchFlags.NONE

    def pack(self) -> bytes:
        if len(self.reader_counts) != POOL_SLICE_MAX_LOCAL_READERS:
            raise ValueError(
                f"reader_counts must contain {POOL_SLICE_MAX_LOCAL_READERS} values"
            )
        return _PUBLISH_STRUCT.pack(
            _uint("sequence", self.sequence, 64),
            _uint("source_pe", self.source_pe, 32),
            _uint("target_pe", self.target_pe, 32),
            _uint("active_rows", self.active_rows, 32),
            _uint("flags", int(self.flags), 32),
            _uint("route_begin", self.route_begin, 32),
            _uint("route_end", self.route_end, 32),
            *(
                _uint(f"reader_counts[{index}]", count, 32)
                for index, count in enumerate(self.reader_counts)
            ),
        )

    @classmethod
    def unpack(
        cls, payload: bytes | bytearray | memoryview
    ) -> "PoolSlicePublishBatch":
        if len(payload) != POOL_SLICE_PUBLISH_BYTES:
            raise ValueError(
                f"publish payload must be {POOL_SLICE_PUBLISH_BYTES} bytes"
            )
        values = _PUBLISH_STRUCT.unpack(payload)
        return cls(
            sequence=values[0],
            source_pe=values[1],
            target_pe=values[2],
            active_rows=values[3],
            flags=PoolSliceBatchFlags(values[4]),
            route_begin=values[5],
            route_end=values[6],
            reader_counts=tuple(values[7:]),
        )


@dataclass(frozen=True)
class PoolSliceReceiveBatch:
    sequence: int
    base_row: int
    source_begin: int
    row_count: int
    source_pe: int
    local_reader: int
    flags: PoolSliceBatchFlags | int = PoolSliceBatchFlags.NONE

    def pack(self) -> bytes:
        return _RECEIVE_STRUCT.pack(
            _uint("sequence", self.sequence, 64),
            _uint("base_row", self.base_row, 64),
            _uint("source_begin", self.source_begin, 32),
            _uint("row_count", self.row_count, 32),
            _uint("source_pe", self.source_pe, 32),
            _uint("local_reader", self.local_reader, 32),
            _uint("flags", int(self.flags), 32),
            0,
            0,
            0,
        )

    @classmethod
    def unpack(
        cls, payload: bytes | bytearray | memoryview
    ) -> "PoolSliceReceiveBatch":
        if len(payload) != POOL_SLICE_RECEIVE_BYTES:
            raise ValueError(
                f"receive payload must be {POOL_SLICE_RECEIVE_BYTES} bytes"
            )
        values = _RECEIVE_STRUCT.unpack(payload)
        return cls(
            sequence=values[0],
            base_row=values[1],
            source_begin=values[2],
            row_count=values[3],
            source_pe=values[4],
            local_reader=values[5],
            flags=PoolSliceBatchFlags(values[6]),
        )


@dataclass(frozen=True)
class PoolSliceConfig:
    combine_rows_address: int
    token_pool_address: int
    delivery_pool_address: int
    expert_input_address: int
    expert_output_address: int
    return_inbox_address: int
    returned_address: int
    send_offsets_address: int
    send_rows_address: int
    send_origin_rows_address: int
    send_token_rows_address: int
    send_token_counts_address: int
    send_batches_address: int
    receive_batches_address: int
    receive_rows_address: int
    receive_routes_address: int
    sequence_address: int
    group_ready_address: int
    control_address: int
    row_bytes: int
    reducer_count: int
    pool_stride: int
    delivery_stride: int
    expert_row_stride: int
    return_stride: int
    expert_stride: int
    active_rows: int
    token_capacity: int
    route_capacity: int
    expert_capacity_rows: int
    local_readers: int
    num_pes: int
    my_pe: int
    signal_base: int
    signal_count: int
    return_capacity_rows: int
    pack_warps: int
    write_chunks: int
    write_chunk_rows: int
    pool_rank: int = 0
    pool_count: int = 1
    dispatch_mode: PoolSliceDispatchMode | int = PoolSliceDispatchMode.POOL_GATHER
    flags: int = 0

    @property
    def num_readers(self) -> int:
        return self.local_readers * self.num_pes

    def pack(self) -> bytes:
        pointers = (
            self.combine_rows_address,
            self.token_pool_address,
            self.delivery_pool_address,
            self.expert_input_address,
            self.expert_output_address,
            self.return_inbox_address,
            self.returned_address,
            self.send_offsets_address,
            self.send_rows_address,
            self.send_origin_rows_address,
            self.send_token_rows_address,
            self.send_token_counts_address,
            self.send_batches_address,
            self.receive_batches_address,
            self.receive_rows_address,
            self.receive_routes_address,
            self.sequence_address,
            self.group_ready_address,
            self.control_address,
        )
        pointer_values = tuple(
            _positive_uint(f"pointer[{index}]", value, 64)
            for index, value in enumerate(pointers)
        )
        values = (
            _positive_uint("row_bytes", self.row_bytes, 32),
            _positive_uint("reducer_count", self.reducer_count, 32),
            _positive_uint("pool_stride", self.pool_stride, 32),
            _positive_uint("delivery_stride", self.delivery_stride, 32),
            _positive_uint("expert_row_stride", self.expert_row_stride, 32),
            _positive_uint("return_stride", self.return_stride, 32),
            _positive_uint("expert_stride", self.expert_stride, 32),
            _uint("active_rows", self.active_rows, 32),
            _positive_uint("token_capacity", self.token_capacity, 32),
            _positive_uint("route_capacity", self.route_capacity, 32),
            _positive_uint(
                "expert_capacity_rows", self.expert_capacity_rows, 32
            ),
            _positive_uint("local_readers", self.local_readers, 32),
            _positive_uint("num_pes", self.num_pes, 32),
            _uint("my_pe", self.my_pe, 32),
            _uint("signal_base", self.signal_base, 32),
            _positive_uint("signal_count", self.signal_count, 32),
            _positive_uint(
                "return_capacity_rows", self.return_capacity_rows, 32
            ),
            _positive_uint("pack_warps", self.pack_warps, 32),
            _positive_uint("write_chunks", self.write_chunks, 32),
            _positive_uint("write_chunk_rows", self.write_chunk_rows, 32),
            _uint("pool_rank", self.pool_rank, 32),
            _positive_uint("pool_count", self.pool_count, 32),
            _uint("dispatch_mode", int(self.dispatch_mode), 32),
            _uint("flags", self.flags, 32),
        )
        if self.my_pe >= self.num_pes:
            raise ValueError("my_pe is outside the PE range")
        if self.num_pes > POOL_SLICE_MAX_PES:
            raise ValueError(f"a pool slice supports at most {POOL_SLICE_MAX_PES} PEs")
        if not 1 <= self.local_readers <= POOL_SLICE_MAX_LOCAL_READERS:
            raise ValueError(
                "local_readers must be in "
                f"[1, {POOL_SLICE_MAX_LOCAL_READERS}]"
            )
        if not 1 <= self.pool_count <= POOL_SLICE_MAX_POOL_BLOCKS:
            raise ValueError(
                f"pool_count must be in [1, {POOL_SLICE_MAX_POOL_BLOCKS}]"
            )
        if self.pool_rank >= self.pool_count:
            raise ValueError("pool_rank is outside the pool block range")
        try:
            dispatch_mode = PoolSliceDispatchMode(self.dispatch_mode)
        except ValueError as error:
            raise ValueError("unsupported pool-slice dispatch mode") from error
        try:
            flags = PoolSliceFlags(self.flags)
        except ValueError as error:
            raise ValueError("unsupported pool-slice flags") from error
        supported_flags = (
            PoolSliceFlags.DEDICATED_COORDINATOR
            | PoolSliceFlags.PUT_PHASE_WORDS
            | PoolSliceFlags.PIPELINED_RETURN
            | PoolSliceFlags.READER_PIPELINE
            | PoolSliceFlags.WEIGHTED_RETURN
            | PoolSliceFlags.EXTERNAL_WEIGHTED_REDUCER
            | PoolSliceFlags.EXTERNAL_TOKEN_REDUCER
        )
        if int(flags) & ~int(supported_flags):
            raise ValueError("unsupported pool-slice flags")
        if (
            flags & PoolSliceFlags.DEDICATED_COORDINATOR
            and self.pool_count == 1
        ):
            raise ValueError("a dedicated coordinator requires multiple pool blocks")
        if (
            flags & PoolSliceFlags.EXTERNAL_WEIGHTED_REDUCER
            and not flags & PoolSliceFlags.WEIGHTED_RETURN
        ):
            raise ValueError("external weighted reducer requires weighted return")
        if (
            flags & PoolSliceFlags.PIPELINED_RETURN
            and flags & PoolSliceFlags.WEIGHTED_RETURN
        ):
            raise ValueError(
                "weighted return and per-reader pipelined return are mutually exclusive"
            )
        if (
            flags & PoolSliceFlags.EXTERNAL_TOKEN_REDUCER
            and not flags & PoolSliceFlags.EXTERNAL_WEIGHTED_REDUCER
        ):
            raise ValueError("external token reducer requires external reduction")
        if self.row_bytes < 1024 or self.row_bytes % 16:
            raise ValueError("row_bytes must be at least 1024 and a multiple of 16")
        if not 1 <= self.reducer_count <= POOL_SLICE_MAX_EXTERNAL_REDUCERS:
            raise ValueError(
                "reducer_count must be in "
                f"[1, {POOL_SLICE_MAX_EXTERNAL_REDUCERS}]"
            )
        if (
            not flags & PoolSliceFlags.EXTERNAL_TOKEN_REDUCER
            and self.reducer_count != self.local_readers
        ):
            raise ValueError("non-sharded reducer count must equal local_readers")
        for name, stride in (
            ("pool_stride", self.pool_stride),
            ("return_stride", self.return_stride),
        ):
            if stride < self.row_bytes or stride % 16:
                raise ValueError(f"{name} must cover an aligned row")
        if self.delivery_stride != self.row_bytes:
            raise ValueError("delivery rows must be contiguous")
        if self.pool_stride != self.row_bytes:
            raise ValueError("pool-gather source rows must be contiguous")
        if self.expert_row_stride != self.row_bytes:
            raise ValueError("expert rows must be contiguous")
        if self.expert_stride < self.expert_capacity_rows * self.row_bytes:
            raise ValueError("expert_stride does not cover expert capacity")
        if self.expert_capacity_rows < self.num_pes * self.token_capacity:
            raise ValueError(
                "slot-put expert capacity must provide one token-capacity "
                "segment per source PE"
            )
        if self.active_rows > self.route_capacity:
            raise ValueError("active_rows exceeds route_capacity")
        if self.signal_base + self.num_pes > self.signal_count:
            raise ValueError("pool-slice signal range exceeds signal_count")
        if not 1 <= self.pack_warps <= POOL_SLICE_AVAILABLE_WORKER_WARPS:
            raise ValueError("dispatch shards exceed worker warps")
        expected_chunks = (
            self.token_capacity + self.write_chunk_rows - 1
        ) // self.write_chunk_rows
        if self.write_chunks != expected_chunks:
            raise ValueError("write_chunks does not cover token_capacity")
        return _CONFIG_STRUCT.pack(*pointer_values, *values)


def group_routes_by_reader(
    reader_ids: Sequence[int] | torch.Tensor,
    *,
    num_readers: int,
    source_rows: Sequence[int] | torch.Tensor | None = None,
    origin_rows: Sequence[int] | torch.Tensor | None = None,
    route_weights: Sequence[float] | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return stable reader-grouped source and provenance metadata on CPU."""

    num_readers = _positive_uint("num_readers", num_readers, 32)
    readers = torch.as_tensor(reader_ids, dtype=torch.int64, device="cpu").view(-1)
    count = readers.numel()
    if count and (
        readers.min().item() < 0 or readers.max().item() >= num_readers
    ):
        raise ValueError("reader_ids contains an id outside the configured range")

    if source_rows is None:
        source = torch.arange(count, dtype=torch.int64)
    else:
        source = torch.as_tensor(
            source_rows, dtype=torch.int64, device="cpu"
        ).view(-1)
    if origin_rows is None:
        origins = torch.arange(count, dtype=torch.int64)
    else:
        origins = torch.as_tensor(
            origin_rows, dtype=torch.int64, device="cpu"
        ).view(-1)
    if route_weights is None:
        weights = torch.ones(count, dtype=torch.float32)
    else:
        weights = torch.as_tensor(
            route_weights, dtype=torch.float32, device="cpu"
        ).view(-1)
    if source.numel() != count or origins.numel() != count or weights.numel() != count:
        raise ValueError(
            "reader, source, origin, and weight arrays must have equal length"
        )
    if not torch.isfinite(weights).all().item():
        raise ValueError("route_weights must be finite")
    for name, values in (("source_rows", source), ("origin_rows", origins)):
        if values.numel() and (
            values.min().item() < 0 or values.max().item() >= 1 << 32
        ):
            raise ValueError(f"{name} must fit in uint32")

    if count:
        positions = torch.arange(count, dtype=torch.int64)
        order = torch.argsort(readers * (count + 1) + positions)
        grouped_rows = source.index_select(0, order).to(torch.uint32)
        grouped_origins = origins.index_select(0, order).to(torch.uint32)
        grouped_weights = weights.index_select(0, order).to(torch.bfloat16)
        counts = torch.bincount(readers, minlength=num_readers)
    else:
        grouped_rows = torch.empty(0, dtype=torch.uint32)
        grouped_origins = torch.empty(0, dtype=torch.uint32)
        grouped_weights = torch.empty(0, dtype=torch.bfloat16)
        counts = torch.zeros(num_readers, dtype=torch.int64)
    offsets = torch.zeros(num_readers + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(counts, dim=0)
    if offsets[-1].item() >= 1 << 32:
        raise ValueError("route count must fit in uint32")
    return offsets.to(torch.uint32), grouped_rows, grouped_origins, grouped_weights


@dataclass
class PoolSliceBuffers:
    signals: torch.Tensor
    send_offsets: torch.Tensor
    send_rows: torch.Tensor
    send_origin_rows: torch.Tensor
    send_token_rows: torch.Tensor
    send_token_counts: torch.Tensor
    token_pool: torch.Tensor
    delivery_pool: torch.Tensor
    expert_input: torch.Tensor
    expert_output: torch.Tensor
    return_inbox: torch.Tensor
    send_batches: torch.Tensor
    receive_batches: torch.Tensor
    receive_rows: torch.Tensor
    combine_rows: torch.Tensor
    receive_routes: torch.Tensor
    sequence: torch.Tensor
    group_ready: torch.Tensor
    control: torch.Tensor
    config_tensor: torch.Tensor
    num_pes: int
    my_pe: int
    local_readers: int
    token_capacity: int
    route_capacity: int
    expert_capacity_rows: int
    signal_base: int
    pack_warps: int
    write_chunks: int
    write_chunk_rows: int
    pool_count: int
    reducer_count: int
    flags: PoolSliceFlags
    active_rows: int = 0
    _source: torch.Tensor | None = None
    _returned: torch.Tensor | None = None
    _required_return_rows: int = 0
    _last_sequence: int = 0

    @property
    def num_readers(self) -> int:
        return self.num_pes * self.local_readers

    @property
    def row_bytes(self) -> int:
        return self.token_pool.shape[-1] * self.token_pool.element_size()

    def write_routes(
        self,
        reader_ids: Sequence[int] | torch.Tensor,
        *,
        source_rows: Sequence[int] | torch.Tensor | None = None,
        origin_rows: Sequence[int] | torch.Tensor | None = None,
        route_weights: Sequence[float] | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        offsets, rows, origins, weights = group_routes_by_reader(
            reader_ids,
            num_readers=self.num_readers,
            source_rows=source_rows,
            origin_rows=origin_rows,
            route_weights=route_weights,
        )
        if rows.numel() > self.route_capacity:
            raise ValueError("active route count exceeds route_capacity")
        if rows.numel() and rows.to(torch.int64).max().item() >= self.token_capacity:
            raise ValueError("source row exceeds token_capacity")
        if self.flags & PoolSliceFlags.WEIGHTED_RETURN:
            if origins.numel() and not torch.equal(origins, rows):
                raise ValueError(
                    "weighted return currently requires origin_rows == source_rows"
                )
            self._required_return_rows = self.token_capacity
        else:
            self._required_return_rows = (
                int(origins.to(torch.int64).max().item()) + 1
                if origins.numel()
                else 0
            )
        self.send_offsets.copy_(offsets)
        self.send_rows.zero_()
        self.send_origin_rows.zero_()
        self.send_token_rows.zero_()
        self.send_token_rows[self.num_pes :].fill_((1 << 32) - 1)
        self.send_token_counts.zero_()
        if rows.numel():
            self.send_origin_rows[: origins.numel()].copy_(origins)
            compact_rows = torch.empty_like(rows)
            for target_pe in range(self.num_pes):
                reader_begin = target_pe * self.local_readers
                route_begin = int(offsets[reader_begin].item())
                route_end = int(
                    offsets[reader_begin + self.local_readers].item()
                )
                if route_begin == route_end:
                    continue
                unique_rows, inverse = torch.unique(
                    rows[route_begin:route_end].to(torch.int64),
                    sorted=True,
                    return_inverse=True,
                )
                self.send_token_counts[target_pe] = unique_rows.numel()
                self.send_token_rows[
                    target_pe, : unique_rows.numel()
                ].copy_(unique_rows.to(torch.uint32))
                inverse_rows = torch.full(
                    (self.token_capacity,),
                    (1 << 32) - 1,
                    dtype=torch.int64,
                )
                inverse_rows[unique_rows] = torch.arange(
                    unique_rows.numel(), dtype=torch.int64
                )
                self.send_token_rows[self.num_pes + target_pe].copy_(
                    inverse_rows.to(torch.uint32)
                )
                compact_rows[route_begin:route_end].copy_(
                    inverse.to(torch.uint32)
                )
            weight_bits = weights.view(torch.uint16).to(torch.int64)
            route_words = compact_rows.to(torch.int64) | (weight_bits << 32)
            self.send_rows[: rows.numel()].copy_(route_words.to(torch.uint64))
        self.active_rows = rows.numel()
        return (
            self.send_offsets,
            self.send_rows[: self.active_rows],
            self.send_origin_rows[: self.active_rows],
        )

    def config(
        self,
        source: torch.Tensor,
        returned: torch.Tensor,
        *,
        pool_rank: int = 0,
    ) -> PoolSliceConfig:
        source = _symmetric_tensor(source, "source")
        returned = _symmetric_tensor(returned, "returned")
        if source.ndim != 2 or returned.ndim != 2:
            raise ValueError("source and returned must be rank-2 row tensors")
        if source.dtype != self.token_pool.dtype or returned.dtype != source.dtype:
            raise ValueError("source, pool, and returned tensors must share dtype")
        if source.shape[1] != self.token_pool.shape[1]:
            raise ValueError("source and pool row widths must match")
        if returned.shape[1] != source.shape[1]:
            raise ValueError("returned and source row widths must match")
        if source.shape[0] < self.token_capacity:
            raise ValueError("source must cover token_capacity rows")
        if returned.shape[0] < self._required_return_rows:
            raise ValueError("returned does not cover the largest origin row")
        element_bytes = source.element_size()
        return PoolSliceConfig(
            combine_rows_address=self.combine_rows.data_ptr(),
            token_pool_address=self.token_pool.data_ptr(),
            delivery_pool_address=self.delivery_pool.data_ptr(),
            expert_input_address=self.expert_input.data_ptr(),
            expert_output_address=self.expert_output.data_ptr(),
            return_inbox_address=self.return_inbox.data_ptr(),
            returned_address=returned.data_ptr(),
            send_offsets_address=self.send_offsets.data_ptr(),
            send_rows_address=self.send_rows.data_ptr(),
            send_origin_rows_address=self.send_origin_rows.data_ptr(),
            send_token_rows_address=self.send_token_rows.data_ptr(),
            send_token_counts_address=self.send_token_counts.data_ptr(),
            send_batches_address=self.send_batches.data_ptr(),
            receive_batches_address=self.receive_batches.data_ptr(),
            receive_rows_address=self.receive_rows.data_ptr(),
            receive_routes_address=self.receive_routes.data_ptr(),
            sequence_address=self.sequence.data_ptr(),
            group_ready_address=self.group_ready.data_ptr(),
            control_address=self.control.data_ptr(),
            row_bytes=self.row_bytes,
            reducer_count=self.reducer_count,
            pool_stride=self.token_pool.stride(0) * element_bytes,
            delivery_stride=self.delivery_pool.stride(0) * element_bytes,
            expert_row_stride=self.expert_input.stride(1) * element_bytes,
            return_stride=returned.stride(0) * element_bytes,
            expert_stride=self.expert_input.stride(0) * element_bytes,
            active_rows=self.active_rows,
            token_capacity=self.token_capacity,
            route_capacity=self.route_capacity,
            expert_capacity_rows=self.expert_capacity_rows,
            local_readers=self.local_readers,
            num_pes=self.num_pes,
            my_pe=self.my_pe,
            signal_base=self.signal_base,
            signal_count=self.signals.numel(),
            return_capacity_rows=returned.shape[0],
            pack_warps=self.pack_warps,
            write_chunks=self.write_chunks,
            write_chunk_rows=self.write_chunk_rows,
            pool_rank=pool_rank,
            pool_count=self.pool_count,
            dispatch_mode=PoolSliceDispatchMode.POOL_GATHER,
            flags=self.flags,
        )

    def prepare(self, source: torch.Tensor, returned: torch.Tensor) -> torch.Tensor:
        self._source = source
        self._returned = returned
        for pool_rank in range(self.pool_count):
            _copy_packed(
                self.config_tensor[pool_rank],
                self.config(source, returned, pool_rank=pool_rank).pack(),
            )
        return self.config_tensor

    def set_sequence(self, sequence: int) -> None:
        sequence = _positive_uint("sequence", sequence, 64)
        maximum = ((1 << 64) - POOL_SLICE_SIGNAL_PHASES) // (
            POOL_SLICE_SIGNAL_PHASES
        ) + 1
        if sequence > maximum:
            raise ValueError("sequence is too large for merged phase signals")
        if sequence <= self._last_sequence:
            raise ValueError("sequence must increase monotonically")
        self.sequence.fill_(sequence)
        self._last_sequence = sequence

    def control_state(self) -> tuple[PoolSliceStatus, int, int, int, int, int]:
        values = self.control.cpu().tolist()
        return (
            PoolSliceStatus(values[0]),
            values[1],
            values[2],
            values[3],
            values[4],
            int(self.group_ready[0].item()),
        )

    def performance_state(self) -> tuple[int, int, int]:
        """Return nonempty sources, dispatch batches, and worker configuration."""

        values = self.control[5:8].cpu().tolist()
        return tuple(int(value) for value in values)

    def read_receive_routes(self) -> list[list[PoolSliceReceiveBatch]]:
        raw = self.receive_routes.cpu()
        result: list[list[PoolSliceReceiveBatch]] = []
        for local_reader in range(self.local_readers):
            per_source = []
            for source_pe in range(self.num_pes):
                per_source.append(
                    PoolSliceReceiveBatch.unpack(
                        bytes(raw[local_reader, source_pe].tolist())
                    )
                )
            result.append(per_source)
        return result


@dataclass(frozen=True)
class PoolSliceProgram:
    launcher: "Launcher"
    write_barrier: int
    dispatch_barriers: tuple[int, ...]
    compute_barriers: tuple[int, ...]
    chunk_rows: int
    communication_block: int = 1
    pool_blocks: tuple[int, ...] = (1,)
    expert_barriers: tuple[int, ...] = ()
    zero_barrier: int | None = None
    reducer_blocks: tuple[int, ...] = ()

    def launch(self) -> None:
        self.launcher.launch()

    def timing_ns(self) -> tuple[int, int, int]:
        events = self.launcher.profile[
            self.communication_block,
            POOL_SLICE_PROFILE_START : POOL_SLICE_PROFILE_DONE + 1,
        ].cpu().to(torch.int64)
        start, gather_ready, done = (int(value) for value in events.tolist())
        if start == 0 or gather_ready < start or done < gather_ready:
            raise RuntimeError("pool-slice profile events are incomplete")
        return gather_ready - start, done - gather_ready, done - start

    def overlap_timing_ns(self) -> dict[str, int | None]:
        """Return macro-operator phase offsets from its internal start event."""

        indices = (
            POOL_SLICE_PROFILE_START,
            POOL_SLICE_PROFILE_FIRST_DATA_PUBLISHED,
            POOL_SLICE_PROFILE_DATA_PUBLISHED,
            POOL_SLICE_PROFILE_FIRST_PAYLOAD,
            POOL_SLICE_PROFILE_METADATA_CLOSED,
            POOL_SLICE_PROFILE_PAYLOAD_DONE,
            POOL_SLICE_PROFILE_GATHER_READY,
            POOL_SLICE_PROFILE_COMPUTE_READY,
            POOL_SLICE_PROFILE_RETURN_PAYLOAD_DONE,
            POOL_SLICE_PROFILE_RETURN_SIGNALS_CLOSED,
            POOL_SLICE_PROFILE_SCATTER_DONE,
        )
        profile = self.launcher.profile[self.communication_block].cpu().to(torch.int64)
        events = profile[list(indices)]
        (
            start,
            first_data_published,
            data_published,
            first_payload,
            metadata_closed,
            payload_done,
            gather_ready,
            compute_ready,
            return_payload_done,
            return_signals_closed,
            scatter_done,
        ) = (int(value) for value in events.tolist())
        required = (
            first_data_published,
            data_published,
            metadata_closed,
            payload_done,
            gather_ready,
            compute_ready,
            return_payload_done,
            return_signals_closed,
            scatter_done,
        )
        if start == 0 or any(value < start for value in required):
            raise RuntimeError("pool-slice overlap events are incomplete")
        if gather_ready < payload_done:
            raise RuntimeError("pool-slice overlap events are out of order")
        return {
            "first_data_published": first_data_published - start,
            "data_published": data_published - start,
            "first_payload": (
                first_payload - start if first_payload >= start else None
            ),
            "metadata_closed": metadata_closed - start,
            "payload_done": payload_done - start,
            "gather_ready": gather_ready - start,
            "compute_ready": compute_ready - start,
            "return_payload_done": return_payload_done - start,
            "return_signals_closed": return_signals_closed - start,
            "scatter_done": scatter_done - start,
        }

    def weighted_return_timing_ns(self) -> dict[str, int | None]:
        """Return cross-CTA weighted-return stage boundaries.

        PoolInst CTAs own disjoint token shards, so the useful boundaries are
        the earliest CTA start/PUT and the latest CTA reduction/quiet. Values
        are offsets from the rank-zero PoolInst start event.
        """

        profile = self.launcher.profile.cpu().to(torch.int64)
        start = int(
            profile[self.communication_block, POOL_SLICE_PROFILE_START].item()
        )
        block_profile = profile[list(self.pool_blocks)]

        def offset(index: int, *, take_min: bool) -> int | None:
            values = block_profile[:, index]
            values = values[values >= start]
            if start == 0 or values.numel() == 0:
                return None
            selected = values.min() if take_min else values.max()
            return int(selected.item()) - start

        return {
            "return_reduce_start": offset(
                POOL_SLICE_PROFILE_RETURN_REDUCE_START, take_min=True
            ),
            "return_reduce_done": offset(
                POOL_SLICE_PROFILE_RETURN_REDUCE_DONE, take_min=False
            ),
            "first_return_put": offset(
                POOL_SLICE_PROFILE_FIRST_RETURN_PUT, take_min=True
            ),
            "return_cta_done": offset(
                POOL_SLICE_PROFILE_RETURN_CTA_DONE, take_min=False
            ),
        }

    def weighted_return_cta_timing_ns(self) -> list[dict[str, int | None]]:
        """Return raw per-PoolInst CTA offsets for profiler diagnosis."""

        profile = self.launcher.profile.cpu().to(torch.int64)
        start = int(
            profile[self.communication_block, POOL_SLICE_PROFILE_START].item()
        )
        indices = {
            "reduce_start": POOL_SLICE_PROFILE_RETURN_REDUCE_START,
            "reduce_done": POOL_SLICE_PROFILE_RETURN_REDUCE_DONE,
            "first_put": POOL_SLICE_PROFILE_FIRST_RETURN_PUT,
            "cta_done": POOL_SLICE_PROFILE_RETURN_CTA_DONE,
        }
        result: list[dict[str, int | None]] = []
        for block in self.pool_blocks:
            timings: dict[str, int | None] = {"block": block}
            for name, index in indices.items():
                value = int(profile[block, index].item())
                timings[name] = value - start if start and value >= start else None
            result.append(timings)
        return result

    def external_reducer_timing_ns(self) -> dict[str, int]:
        """Return ordinary-reducer events relative to the PoolInst start."""

        if not self.reducer_blocks:
            raise RuntimeError("program does not contain external reducer blocks")
        profile = self.launcher.profile.cpu().to(torch.int64)
        pool_start = int(
            profile[self.communication_block, POOL_SLICE_PROFILE_START].item()
        )
        starts = profile[
            list(self.reducer_blocks), POOL_SLICE_PROFILE_EXTERNAL_REDUCE_START
        ]
        dones = profile[
            list(self.reducer_blocks), POOL_SLICE_PROFILE_EXTERNAL_REDUCE_DONE
        ]
        if (
            pool_start == 0
            or torch.any(starts == 0).item()
            or torch.any(dones < starts).item()
        ):
            raise RuntimeError("external reducer profile events are incomplete")
        first_start = int(starts.min().item())
        last_start = int(starts.max().item())
        first_done = int(dones.min().item())
        all_done = int(dones.max().item())
        result = {
            "first_external_reduce_start": first_start - pool_start,
            "last_external_reduce_start": last_start - pool_start,
            "first_external_reduce_done": first_done - pool_start,
            "all_external_reduce_done": all_done - pool_start,
            "external_reduce_span": all_done - first_start,
        }
        if self.zero_barrier is not None:
            zero_block = self.reducer_blocks[0] - 1
            zero_start = int(
                profile[
                    zero_block, POOL_SLICE_PROFILE_EXTERNAL_ZERO_START
                ].item()
            )
            zero_done = int(
                profile[
                    zero_block, POOL_SLICE_PROFILE_EXTERNAL_ZERO_DONE
                ].item()
            )
            if zero_start == 0 or zero_done < zero_start:
                raise RuntimeError("external zero profile events are incomplete")
            result["external_zero_start"] = zero_start - pool_start
            result["external_zero_done"] = zero_done - pool_start
        return result


def build_pool_slice_copy_program(
    buffers: PoolSliceBuffers,
    *,
    benchmark_barrier=None,
    in_place_identity: bool = False,
    source_preloaded: bool = False,
    reader_rms_weights: torch.Tensor | None = None,
    reader_rms_epsilon: float = 1.0e-5,
) -> PoolSliceProgram:
    """Build writer, PoolInst, reader, and optional reducer VDCores ops."""

    from .instructions import (
        Copy,
        IssueBarrier,
        PoolRawAddress,
        POOL_EXPERT_ATOMIC_REDUCE_BF16,
        POOL_RMS_NORM_F16_K_4096,
        POOL_TOKEN_REDUCE_BF16,
        POOL_ZERO_WEIGHTED_RETURN,
        PoolTmaStore1D,
        PoolWaitSignal,
        PoolSliceExchange,
        RawAddress,
        TerminateC,
        TerminateM,
        TmaLoad1D,
    )
    from .launcher import Launcher

    if buffers._source is None or buffers._returned is None:
        raise RuntimeError("call buffers.prepare(source, returned) before building")
    external_reducer = bool(
        buffers.flags & PoolSliceFlags.EXTERNAL_WEIGHTED_REDUCER
    )
    token_reducer = bool(
        buffers.flags & PoolSliceFlags.EXTERNAL_TOKEN_REDUCER
    )
    device = buffers.signals.device
    properties = torch.cuda.get_device_properties(device)
    if in_place_identity and (
        buffers.expert_input.data_ptr() != buffers.expert_output.data_ptr()
    ):
        raise ValueError(
            "in_place_identity requires expert_input and expert_output to alias"
        )
    if in_place_identity and reader_rms_weights is not None:
        raise ValueError("reader RMS requires a separate expert output buffer")
    if reader_rms_weights is not None:
        if buffers.expert_input.dtype != torch.bfloat16:
            raise ValueError("reader RMS currently requires bfloat16 buffers")
        if buffers.token_pool.shape[-1] != 4096:
            raise ValueError("reader RMS currently requires hidden_size=4096")
        if (
            not isinstance(reader_rms_weights, torch.Tensor)
            or not reader_rms_weights.is_cuda
            or reader_rms_weights.dtype != torch.bfloat16
            or not reader_rms_weights.is_contiguous()
            or reader_rms_weights.numel() != 4096
        ):
            raise ValueError(
                "reader_rms_weights must be a contiguous CUDA bfloat16 row "
                "with 4096 elements"
            )
    if source_preloaded and (
        buffers._source.data_ptr() != buffers.token_pool.data_ptr()
    ):
        raise ValueError(
            "source_preloaded requires source and token_pool to alias"
        )
    reader_blocks = 0 if in_place_identity else buffers.local_readers
    reducer_blocks_count = (
        buffers.reducer_count + (0 if token_reducer else 1)
        if external_reducer
        else 0
    )
    num_sms = 1 + buffers.pool_count + reader_blocks + reducer_blocks_count
    if num_sms > properties.multi_processor_count:
        raise ValueError(
            "pool writer, PoolInst blocks, and local readers exceed the GPU "
            f"SM count ({properties.multi_processor_count})"
        )

    row_bytes = buffers.row_bytes
    chunk_rows = POOL_SLICE_MAX_TMA_BYTES // row_bytes
    if chunk_rows == 0:
        raise ValueError(
            f"row_bytes={row_bytes} exceeds the 16-bit VDCores TMA size field"
        )
    token_chunks = (
        buffers.token_capacity + chunk_rows - 1
    ) // chunk_rows
    reader_chunks = (
        buffers.expert_capacity_rows + chunk_rows - 1
    ) // chunk_rows

    launcher = Launcher(
        num_sms=num_sms,
        device=device,
        signal_array=buffers.signals,
        benchmark_barrier=benchmark_barrier,
    )
    write_barriers = tuple(
        launcher.new_bar(0 if source_preloaded else 1)
        for _ in range(buffers.write_chunks)
    )
    write_barrier = write_barriers[0]
    dispatch_barriers = tuple(
        launcher.new_bar(1) for _ in range(buffers.local_readers)
    )
    compute_barriers = tuple(
        launcher.new_bar(1 if external_reducer else (0 if in_place_identity else 1))
        for _ in range(buffers.reducer_count)
    )
    expert_barriers = (
        tuple(launcher.new_bar(1) for _ in range(buffers.local_readers))
        if external_reducer and not in_place_identity
        else ()
    )
    zero_barrier = (
        launcher.new_bar(1)
        if external_reducer and not token_reducer
        else None
    )
    dispatch_barrier_base = dispatch_barriers[0]
    compute_barrier_base = compute_barriers[0]
    if dispatch_barriers != tuple(
        range(dispatch_barrier_base, dispatch_barrier_base + buffers.local_readers)
    ):
        raise AssertionError("dispatch barriers must be contiguous")
    if compute_barriers != tuple(
        range(compute_barrier_base, compute_barrier_base + buffers.reducer_count)
    ):
        raise AssertionError("compute barriers must be contiguous")

    writer_builder = launcher.builder[0]
    for pool_rank in range(buffers.pool_count):
        pool_builder = launcher.builder[1 + pool_rank]
        pool_builder.add_pool(
            PoolSliceExchange(
                buffers.config_tensor[pool_rank],
                write_barrier=write_barrier,
                dispatch_barrier_base=dispatch_barrier_base,
                compute_barrier_base=compute_barrier_base,
            )
        )

    hidden_size = buffers.token_pool.shape[-1]
    if not source_preloaded:
        source_flat = buffers._source.view(-1)
        token_pool_flat = buffers.token_pool.view(-1)
        for chunk in range(token_chunks):
            row_begin = chunk * chunk_rows
            rows = min(chunk_rows, buffers.token_capacity - row_begin)
            elements = rows * hidden_size
            offset = row_begin * hidden_size
            source = source_flat.narrow(0, offset, elements)
            destination = token_pool_flat.narrow(0, offset, elements)
            nbytes = rows * row_bytes
            writer_builder.add_memory(TmaLoad1D(source, bytes=nbytes))
            store = PoolTmaStore1D(destination, bytes=nbytes)
            store.bar(write_barriers[chunk])
            writer_builder.add_memory(store)
            writer_builder.add_compute(Copy(1, nbytes))

    if not in_place_identity:
        reader_base = 1 + buffers.pool_count
        for local_reader in range(buffers.local_readers):
            builder = launcher.builder[reader_base + local_reader]
            builder.add_memory(PoolWaitSignal(dispatch_barriers[local_reader]))
            if reader_rms_weights is not None:
                builder.add_memory(RawAddress(reader_rms_weights, 26))
                builder.add_memory(
                    RawAddress(buffers.expert_input[local_reader], 24)
                )
                rms_output = PoolRawAddress(
                    buffers.expert_output[local_reader], 25
                )
                rms_output.bar(
                    expert_barriers[local_reader]
                    if external_reducer
                    else compute_barriers[local_reader]
                )
                builder.add_memory(rms_output)
                builder.add_memory(
                    RawAddress(
                        buffers.control[
                            POOL_SLICE_CONTROL_READER_ROW_COUNT + local_reader :
                            POOL_SLICE_CONTROL_READER_ROW_COUNT + local_reader + 1
                        ],
                        27,
                    )
                )
                builder.add_compute(
                    POOL_RMS_NORM_F16_K_4096(
                        buffers.expert_capacity_rows,
                        reader_rms_epsilon,
                    )
                )
                continue
            input_flat = buffers.expert_input[local_reader].view(-1)
            output_flat = buffers.expert_output[local_reader].view(-1)
            for chunk in range(reader_chunks):
                row_begin = chunk * chunk_rows
                rows = min(
                    chunk_rows,
                    buffers.expert_capacity_rows - row_begin,
                )
                elements = rows * hidden_size
                offset = row_begin * hidden_size
                source = input_flat.narrow(0, offset, elements)
                destination = output_flat.narrow(0, offset, elements)
                nbytes = rows * row_bytes
                builder.add_memory(TmaLoad1D(source, bytes=nbytes))
                store = PoolTmaStore1D(destination, bytes=nbytes)
                if chunk + 1 == reader_chunks:
                    store.bar(
                        expert_barriers[local_reader]
                        if external_reducer
                        else compute_barriers[local_reader]
                    )
                builder.add_memory(store)
                builder.add_compute(Copy(1, nbytes))

    reducer_blocks: tuple[int, ...] = ()
    if external_reducer:
        reducer_base = 1 + buffers.pool_count + reader_blocks
        staging_row = buffers.num_pes * buffers.token_capacity
        staging = buffers.return_inbox.narrow(0, staging_row, staging_row)
        if token_reducer:
            reducer_blocks = tuple(
                reducer_base + reducer
                for reducer in range(buffers.reducer_count)
            )
            ready_barriers = (
                dispatch_barriers if in_place_identity else expert_barriers
            )
            for reducer_rank, reducer_block in enumerate(reducer_blocks):
                builder = launcher.builder[reducer_block]
                for reader_ready in ready_barriers:
                    builder.add_memory(PoolWaitSignal(reader_ready))
                builder.add_memory(RawAddress(buffers.config_tensor[0], 24))
                reduce_output = PoolRawAddress(staging, 25)
                reduce_output.bar(compute_barriers[reducer_rank])
                builder.add_memory(reduce_output)
                builder.add_compute(
                    POOL_TOKEN_REDUCE_BF16(
                        reducer_rank, buffers.reducer_count
                    )
                )
        else:
            assert zero_barrier is not None
            zero_block = reducer_base
            zero_builder = launcher.builder[zero_block]
            zero_builder.add_memory(RawAddress(buffers.config_tensor[0], 24))
            zero_output = PoolRawAddress(staging, 25)
            zero_output.bar(zero_barrier)
            zero_builder.add_memory(zero_output)
            zero_builder.add_compute(POOL_ZERO_WEIGHTED_RETURN())

            reducer_blocks = tuple(
                zero_block + 1 + reader
                for reader in range(buffers.local_readers)
            )
            for local_reader, reducer_block in enumerate(reducer_blocks):
                builder = launcher.builder[reducer_block]
                builder.add_memory(PoolWaitSignal(zero_barrier))
                reader_ready = (
                    dispatch_barriers[local_reader]
                    if in_place_identity
                    else expert_barriers[local_reader]
                )
                builder.add_memory(PoolWaitSignal(reader_ready))
                builder.add_memory(RawAddress(buffers.config_tensor[0], 24))
                reduce_output = PoolRawAddress(staging, 25)
                reduce_output.bar(compute_barriers[local_reader])
                builder.add_memory(reduce_output)
                builder.add_compute(
                    POOL_EXPERT_ATOMIC_REDUCE_BF16(local_reader)
                )

    for builder in launcher.builder:
        builder.add_memory(TerminateM())
        builder.add_compute(TerminateC())

    max_memory = max(
        1 if source_preloaded else 2 * token_chunks + 1,
        1
        if in_place_identity
        else (
            6
            if reader_rms_weights is not None
            else 2 * reader_chunks + 2
        ),
        (
            buffers.local_readers + 3
            if token_reducer
            else (5 if external_reducer else 1)
        ),
    )
    max_compute = max(
        1 if source_preloaded else token_chunks + 1,
        1
        if in_place_identity
        else (2 if reader_rms_weights is not None else reader_chunks + 1),
        2 if external_reducer else 1,
    )
    if max_memory > launcher.max_insts or max_compute > launcher.max_insts:
        raise ValueError("pool capacity requires too many VDCores instructions")
    return PoolSliceProgram(
        launcher=launcher,
        write_barrier=write_barrier,
        dispatch_barriers=dispatch_barriers,
        compute_barriers=compute_barriers,
        chunk_rows=chunk_rows,
        communication_block=1,
        pool_blocks=tuple(range(1, 1 + buffers.pool_count)),
        expert_barriers=expert_barriers,
        zero_barrier=zero_barrier,
        reducer_blocks=reducer_blocks,
    )


def allocate_pool_slice(
    signals: torch.Tensor,
    *,
    num_pes: int,
    my_pe: int,
    local_readers: int,
    token_capacity: int,
    route_capacity: int | None = None,
    expert_capacity_rows: int,
    hidden_size: int,
    dtype: torch.dtype = torch.bfloat16,
    signal_base: int = 0,
    pack_warps: int = 0,
    pool_blocks: int = 1,
    in_place_expert_output: bool = False,
    dedicated_coordinator: bool = False,
    put_phase_words: bool = False,
    pipelined_return: bool = False,
    reader_pipeline: bool = False,
    weighted_return: bool = False,
    external_weighted_reducer: bool = False,
    external_reducer_mode: str = "expert_atomic",
    external_reducer_blocks: int = 0,
) -> PoolSliceBuffers:
    """Collectively allocate one batched logical pool slice on every PE."""

    from . import nvshmem

    signals = _symmetric_tensor(signals, "signals")
    if signals.dtype != torch.uint64 or signals.ndim != 1:
        raise ValueError("signals must be a contiguous rank-1 uint64 tensor")
    num_pes = _positive_uint("num_pes", num_pes, 32)
    my_pe = _uint("my_pe", my_pe, 32)
    local_readers = _positive_uint("local_readers", local_readers, 32)
    token_capacity = _positive_uint("token_capacity", token_capacity, 32)
    if route_capacity is None:
        route_capacity = token_capacity
    route_capacity = _positive_uint("route_capacity", route_capacity, 32)
    expert_capacity_rows = _positive_uint(
        "expert_capacity_rows", expert_capacity_rows, 32
    )
    hidden_size = _positive_uint("hidden_size", hidden_size, 32)
    signal_base = _uint("signal_base", signal_base, 32)
    pack_warps = _uint("pack_warps", pack_warps, 32)
    pool_blocks = _positive_uint("pool_blocks", pool_blocks, 32)
    if num_pes > POOL_SLICE_MAX_PES:
        raise ValueError(f"a pool slice supports at most {POOL_SLICE_MAX_PES} PEs")
    if my_pe >= num_pes:
        raise ValueError("my_pe is outside the PE range")
    if local_readers > POOL_SLICE_MAX_LOCAL_READERS:
        raise ValueError(
            f"local_readers cannot exceed {POOL_SLICE_MAX_LOCAL_READERS}"
        )
    if pool_blocks > POOL_SLICE_MAX_POOL_BLOCKS:
        raise ValueError(
            f"pool_blocks cannot exceed {POOL_SLICE_MAX_POOL_BLOCKS}"
        )
    if dedicated_coordinator and pool_blocks == 1:
        raise ValueError("a dedicated coordinator requires multiple pool blocks")
    if weighted_return and dtype != torch.bfloat16:
        raise ValueError("weighted return currently requires bfloat16 rows")
    if pipelined_return and weighted_return:
        raise ValueError(
            "weighted return and per-reader pipelined return are mutually exclusive"
        )
    if external_weighted_reducer and not weighted_return:
        raise ValueError("external weighted reducer requires weighted return")
    if external_reducer_mode not in {"expert_atomic", "token_sharded"}:
        raise ValueError(
            "external_reducer_mode must be expert_atomic or token_sharded"
        )
    if not external_weighted_reducer and external_reducer_mode != "expert_atomic":
        raise ValueError("external reducer mode requires external reduction")
    external_reducer_blocks = _uint(
        "external_reducer_blocks", external_reducer_blocks, 32
    )
    if external_reducer_blocks > POOL_SLICE_MAX_EXTERNAL_REDUCERS:
        raise ValueError(
            "external_reducer_blocks exceeds the supported reducer warp"
        )
    if external_reducer_blocks and (
        not external_weighted_reducer
        or external_reducer_mode != "token_sharded"
    ):
        raise ValueError(
            "external_reducer_blocks is only valid for token-sharded reduction"
        )
    reducer_count = (
        (external_reducer_blocks or local_readers)
        if external_weighted_reducer
        and external_reducer_mode == "token_sharded"
        else local_readers
    )
    if expert_capacity_rows < num_pes * token_capacity:
        raise ValueError(
            "slot-put expert capacity must provide one token-capacity "
            "segment per source PE"
        )
    row_bytes = hidden_size * torch.empty((), dtype=dtype).element_size()
    if row_bytes < 1024 or row_bytes % 16:
        raise ValueError("a pool row must be at least 1024 bytes and 16-byte aligned")
    if pack_warps == 0:
        if token_capacity * row_bytes >= POOL_SLICE_LARGE_PACK_BYTES:
            pack_warps = POOL_SLICE_AVAILABLE_WORKER_WARPS
        else:
            pack_warps = min(4, POOL_SLICE_AVAILABLE_WORKER_WARPS)
    elif pack_warps > POOL_SLICE_AVAILABLE_WORKER_WARPS:
        raise ValueError("dispatch shards exceed worker warps")
    if signal_base + num_pes > signals.numel():
        raise ValueError("pool-slice signal range exceeds the signal tensor")
    write_chunk_rows = POOL_SLICE_MAX_TMA_BYTES // row_bytes
    write_chunks = (token_capacity + write_chunk_rows - 1) // write_chunk_rows

    num_readers = num_pes * local_readers
    send_offsets = nvshmem.zeros(num_readers + 1, dtype=torch.uint32)
    # One route metadata word carries the compact activation row in bits 0:31
    # and the BF16 route weight in bits 32:47. Keeping them together preserves
    # one metadata message and one visibility signal per source/target pair.
    send_rows = nvshmem.zeros(route_capacity, dtype=torch.uint64)
    send_origin_rows = nvshmem.zeros(route_capacity, dtype=torch.uint32)
    send_token_rows = nvshmem.zeros(
        (2 * num_pes, token_capacity), dtype=torch.uint32
    )
    send_token_counts = nvshmem.zeros(num_pes, dtype=torch.uint32)
    token_pool = nvshmem.zeros((token_capacity, hidden_size), dtype=dtype)
    delivery_rows = max(route_capacity, 2 * num_pes * token_capacity)
    delivery_pool = nvshmem.zeros((delivery_rows, hidden_size), dtype=dtype)
    expert_input = nvshmem.zeros(
        (local_readers, expert_capacity_rows, hidden_size), dtype=dtype
    )
    expert_output = (
        expert_input
        if in_place_expert_output
        else nvshmem.zeros(
            (local_readers, expert_capacity_rows, hidden_size), dtype=dtype
        )
    )
    weighted_inbox_rows = num_pes * token_capacity * (
        2 if external_weighted_reducer else 1
    )
    return_inbox = nvshmem.zeros(
        (max(route_capacity, weighted_inbox_rows), hidden_size), dtype=dtype
    )
    send_batches = nvshmem.zeros(
        (num_pes, POOL_SLICE_PUBLISH_BYTES), dtype=torch.uint8
    )
    receive_batches = nvshmem.zeros(
        (num_pes, POOL_SLICE_PUBLISH_BYTES), dtype=torch.uint8
    )
    receive_rows = nvshmem.zeros(
        (num_pes, route_capacity), dtype=torch.uint64
    )
    combine_rows = nvshmem.zeros(
        (local_readers, num_pes, token_capacity), dtype=torch.uint64
    )
    receive_routes = nvshmem.zeros(
        (local_readers, num_pes, POOL_SLICE_RECEIVE_BYTES), dtype=torch.uint8
    )
    sequence = nvshmem.zeros(1, dtype=torch.uint64)
    # Word zero is the public gathered-read generation. The remaining words
    # are per-global-reader return-batch signals used by the compact pool-gather
    # pipeline; all are symmetric and single-writer for a given sequence.
    group_ready = nvshmem.zeros(1 + num_readers, dtype=torch.uint64)
    control = nvshmem.zeros(POOL_SLICE_CONTROL_WORDS, dtype=torch.uint64)
    config_tensor = torch.empty(
        (pool_blocks, POOL_SLICE_CONFIG_BYTES),
        dtype=torch.uint8,
        device=signals.device,
    )

    buffers = PoolSliceBuffers(
        signals=signals,
        send_offsets=send_offsets,
        send_rows=send_rows,
        send_origin_rows=send_origin_rows,
        send_token_rows=send_token_rows,
        send_token_counts=send_token_counts,
        token_pool=token_pool,
        delivery_pool=delivery_pool,
        expert_input=expert_input,
        expert_output=expert_output,
        return_inbox=return_inbox,
        send_batches=send_batches,
        receive_batches=receive_batches,
        receive_rows=receive_rows,
        combine_rows=combine_rows,
        receive_routes=receive_routes,
        sequence=sequence,
        group_ready=group_ready,
        control=control,
        config_tensor=config_tensor,
        num_pes=num_pes,
        my_pe=my_pe,
        local_readers=local_readers,
        token_capacity=token_capacity,
        route_capacity=route_capacity,
        expert_capacity_rows=expert_capacity_rows,
        signal_base=signal_base,
        pack_warps=pack_warps,
        write_chunks=write_chunks,
        write_chunk_rows=write_chunk_rows,
        pool_count=pool_blocks,
        reducer_count=reducer_count,
        flags=(
            (PoolSliceFlags.DEDICATED_COORDINATOR if dedicated_coordinator else 0)
            | (PoolSliceFlags.PUT_PHASE_WORDS if put_phase_words else 0)
            | (PoolSliceFlags.PIPELINED_RETURN if pipelined_return else 0)
            | (PoolSliceFlags.READER_PIPELINE if reader_pipeline else 0)
            | (PoolSliceFlags.WEIGHTED_RETURN if weighted_return else 0)
            | (
                PoolSliceFlags.EXTERNAL_WEIGHTED_REDUCER
                if external_weighted_reducer
                else 0
            )
            | (
                PoolSliceFlags.EXTERNAL_TOKEN_REDUCER
                if external_weighted_reducer
                and external_reducer_mode == "token_sharded"
                else 0
            )
        ),
    )
    nvshmem.barrier()
    return buffers


__all__ = [
    "POOL_SLICE_MAX_PES",
    "POOL_SLICE_MAX_LOCAL_READERS",
    "POOL_SLICE_MAX_POOL_BLOCKS",
    "POOL_SLICE_MAX_EXTERNAL_REDUCERS",
    "POOL_SLICE_SIGNAL_PHASES",
    "POOL_SLICE_PUBLISH_BYTES",
    "POOL_SLICE_RECEIVE_BYTES",
    "POOL_SLICE_CONFIG_BYTES",
    "POOL_SLICE_PROFILE_START",
    "POOL_SLICE_PROFILE_GATHER_READY",
    "POOL_SLICE_PROFILE_DONE",
    "POOL_SLICE_PROFILE_DATA_PUBLISHED",
    "POOL_SLICE_PROFILE_FIRST_PAYLOAD",
    "POOL_SLICE_PROFILE_METADATA_CLOSED",
    "POOL_SLICE_PROFILE_PAYLOAD_DONE",
    "POOL_SLICE_PROFILE_FIRST_DATA_PUBLISHED",
    "POOL_SLICE_PROFILE_COMPUTE_READY",
    "POOL_SLICE_PROFILE_RETURN_PAYLOAD_DONE",
    "POOL_SLICE_PROFILE_RETURN_SIGNALS_CLOSED",
    "POOL_SLICE_PROFILE_SCATTER_DONE",
    "POOL_SLICE_PROFILE_EXTERNAL_REDUCE_START",
    "POOL_SLICE_PROFILE_EXTERNAL_REDUCE_DONE",
    "POOL_SLICE_PROFILE_EXTERNAL_ZERO_START",
    "POOL_SLICE_PROFILE_EXTERNAL_ZERO_DONE",
    "POOL_SLICE_PROFILE_RETURN_REDUCE_START",
    "POOL_SLICE_PROFILE_RETURN_REDUCE_DONE",
    "POOL_SLICE_PROFILE_FIRST_RETURN_PUT",
    "POOL_SLICE_PROFILE_RETURN_CTA_DONE",
    "PoolSliceStatus",
    "PoolSliceBatchFlags",
    "PoolSliceDispatchMode",
    "PoolSliceFlags",
    "PoolSlicePublishBatch",
    "PoolSliceReceiveBatch",
    "PoolSliceConfig",
    "PoolSliceBuffers",
    "PoolSliceProgram",
    "group_routes_by_reader",
    "allocate_pool_slice",
    "build_pool_slice_copy_program",
]
