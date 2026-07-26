"""Unified pool-slice dependent-read protocol and VDCores program builder.

Applications write source activations and reader-grouped route metadata in
HBM. A VDCores writer stores each activation once in source-owned token slots.
In parallel, a communication-specialized VDCores macro operator publishes
metadata, packs route-major delivery batches, resolves dynamic-read placement,
moves contiguous batches, and returns reader output to its origins.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, IntFlag
import struct
from typing import Sequence, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .launcher import Launcher


POOL_SLICE_MAX_PES = 32
POOL_SLICE_MAX_LOCAL_READERS = 8
POOL_SLICE_AVAILABLE_WORKER_WARPS = 8
POOL_SLICE_LARGE_PACK_BYTES = 512 * 1024
POOL_SLICE_SIGNAL_PHASES = 3
POOL_SLICE_PUBLISH_BYTES = 64
POOL_SLICE_RECEIVE_BYTES = 48
POOL_SLICE_CONFIG_BYTES = 208
POOL_SLICE_CONTROL_WORDS = 8
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

_PUBLISH_STRUCT = struct.Struct("<Q6I8I")
_RECEIVE_STRUCT = struct.Struct("<QQ5I3I")
_CONFIG_STRUCT = struct.Struct("<16Q20I")
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
    source_address: int
    token_pool_address: int
    delivery_pool_address: int
    expert_input_address: int
    expert_output_address: int
    return_inbox_address: int
    returned_address: int
    send_offsets_address: int
    send_rows_address: int
    send_origin_rows_address: int
    send_batches_address: int
    receive_batches_address: int
    receive_routes_address: int
    sequence_address: int
    group_ready_address: int
    control_address: int
    row_bytes: int
    source_stride: int
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

    @property
    def num_readers(self) -> int:
        return self.local_readers * self.num_pes

    def pack(self) -> bytes:
        pointers = (
            self.source_address,
            self.token_pool_address,
            self.delivery_pool_address,
            self.expert_input_address,
            self.expert_output_address,
            self.return_inbox_address,
            self.returned_address,
            self.send_offsets_address,
            self.send_rows_address,
            self.send_origin_rows_address,
            self.send_batches_address,
            self.receive_batches_address,
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
            _positive_uint("source_stride", self.source_stride, 32),
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
        if self.row_bytes < 1024 or self.row_bytes % 16:
            raise ValueError("row_bytes must be at least 1024 and a multiple of 16")
        for name, stride in (
            ("source_stride", self.source_stride),
            ("pool_stride", self.pool_stride),
            ("return_stride", self.return_stride),
        ):
            if stride < self.row_bytes or stride % 16:
                raise ValueError(f"{name} must cover an aligned row")
        if self.delivery_stride != self.row_bytes:
            raise ValueError("delivery rows must be contiguous")
        if self.expert_row_stride != self.row_bytes:
            raise ValueError("expert rows must be contiguous")
        if self.expert_stride < self.expert_capacity_rows * self.row_bytes:
            raise ValueError("expert_stride does not cover expert capacity")
        if self.active_rows > self.route_capacity:
            raise ValueError("active_rows exceeds route_capacity")
        if self.signal_base + self.num_pes > self.signal_count:
            raise ValueError("pool-slice signal range exceeds signal_count")
        if not 1 <= self.pack_warps < POOL_SLICE_AVAILABLE_WORKER_WARPS:
            raise ValueError(
                "pack_warps must leave at least one receive worker warp"
            )
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    if source.numel() != count or origins.numel() != count:
        raise ValueError("reader, source, and origin arrays must have equal length")
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
        counts = torch.bincount(readers, minlength=num_readers)
    else:
        grouped_rows = torch.empty(0, dtype=torch.uint32)
        grouped_origins = torch.empty(0, dtype=torch.uint32)
        counts = torch.zeros(num_readers, dtype=torch.int64)
    offsets = torch.zeros(num_readers + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(counts, dim=0)
    if offsets[-1].item() >= 1 << 32:
        raise ValueError("route count must fit in uint32")
    return offsets.to(torch.uint32), grouped_rows, grouped_origins


def select_pool_slice_pack_warps(
    *,
    num_pes: int,
    route_capacity: int,
    row_bytes: int,
    requested: int = 0,
) -> int:
    """Choose pack/receive warp balance; zero requests the measured policy."""

    if requested:
        if not 1 <= requested < POOL_SLICE_AVAILABLE_WORKER_WARPS:
            raise ValueError(
                "pack_warps must leave at least one receive worker warp"
            )
        return requested
    pack_bytes = route_capacity * row_bytes
    if pack_bytes < POOL_SLICE_LARGE_PACK_BYTES:
        return 4
    receive_warps = min(num_pes, 4)
    return max(4, POOL_SLICE_AVAILABLE_WORKER_WARPS - receive_warps)


@dataclass
class PoolSliceBuffers:
    signals: torch.Tensor
    send_offsets: torch.Tensor
    send_rows: torch.Tensor
    send_origin_rows: torch.Tensor
    token_pool: torch.Tensor
    delivery_pool: torch.Tensor
    expert_input: torch.Tensor
    expert_output: torch.Tensor
    return_inbox: torch.Tensor
    send_batches: torch.Tensor
    receive_batches: torch.Tensor
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        offsets, rows, origins = group_routes_by_reader(
            reader_ids,
            num_readers=self.num_readers,
            source_rows=source_rows,
            origin_rows=origin_rows,
        )
        if rows.numel() > self.route_capacity:
            raise ValueError("active route count exceeds route_capacity")
        if rows.numel() and rows.to(torch.int64).max().item() >= self.token_capacity:
            raise ValueError("source row exceeds token_capacity")
        self._required_return_rows = (
            int(origins.to(torch.int64).max().item()) + 1 if origins.numel() else 0
        )
        self.send_offsets.copy_(offsets)
        self.send_rows.zero_()
        self.send_origin_rows.zero_()
        if rows.numel():
            self.send_rows[: rows.numel()].copy_(rows)
            self.send_origin_rows[: origins.numel()].copy_(origins)
        self.active_rows = rows.numel()
        return (
            self.send_offsets,
            self.send_rows[: self.active_rows],
            self.send_origin_rows[: self.active_rows],
        )

    def config(self, source: torch.Tensor, returned: torch.Tensor) -> PoolSliceConfig:
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
            source_address=source.data_ptr(),
            token_pool_address=self.token_pool.data_ptr(),
            delivery_pool_address=self.delivery_pool.data_ptr(),
            expert_input_address=self.expert_input.data_ptr(),
            expert_output_address=self.expert_output.data_ptr(),
            return_inbox_address=self.return_inbox.data_ptr(),
            returned_address=returned.data_ptr(),
            send_offsets_address=self.send_offsets.data_ptr(),
            send_rows_address=self.send_rows.data_ptr(),
            send_origin_rows_address=self.send_origin_rows.data_ptr(),
            send_batches_address=self.send_batches.data_ptr(),
            receive_batches_address=self.receive_batches.data_ptr(),
            receive_routes_address=self.receive_routes.data_ptr(),
            sequence_address=self.sequence.data_ptr(),
            group_ready_address=self.group_ready.data_ptr(),
            control_address=self.control.data_ptr(),
            row_bytes=self.row_bytes,
            source_stride=source.stride(0) * element_bytes,
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
        )

    def prepare(self, source: torch.Tensor, returned: torch.Tensor) -> torch.Tensor:
        self._source = source
        self._returned = returned
        _copy_packed(self.config_tensor, self.config(source, returned).pack())
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
            int(self.group_ready.item()),
        )

    def performance_state(self) -> tuple[int, int, int]:
        """Return nonempty sources, dispatch batches, and active pack warps."""

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


def build_pool_slice_copy_program(
    buffers: PoolSliceBuffers,
    *,
    benchmark_barrier=None,
) -> PoolSliceProgram:
    """Build one writer, one communication core, and local reader VDCores ops."""

    from .instructions import (
        Copy,
        IssueBarrier,
        PoolSliceExchange,
        TerminateC,
        TerminateComm,
        TerminateM,
        TmaLoad1D,
        TmaStore1D,
    )
    from .launcher import Launcher

    if buffers._source is None or buffers._returned is None:
        raise RuntimeError("call buffers.prepare(source, returned) before building")
    device = buffers.signals.device
    properties = torch.cuda.get_device_properties(device)
    num_sms = buffers.local_readers + 2
    if num_sms > properties.multi_processor_count:
        raise ValueError(
            "pool writer, communication core, and local readers exceed the GPU "
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
        launcher.new_bar(1) for _ in range(buffers.write_chunks)
    )
    write_barrier = write_barriers[0]
    dispatch_barriers = tuple(
        launcher.new_bar(1) for _ in range(buffers.local_readers)
    )
    compute_barriers = tuple(
        launcher.new_bar(1) for _ in range(buffers.local_readers)
    )
    dispatch_barrier_base = dispatch_barriers[0]
    compute_barrier_base = compute_barriers[0]
    if dispatch_barriers != tuple(
        range(dispatch_barrier_base, dispatch_barrier_base + buffers.local_readers)
    ):
        raise AssertionError("dispatch barriers must be contiguous")
    if compute_barriers != tuple(
        range(compute_barrier_base, compute_barrier_base + buffers.local_readers)
    ):
        raise AssertionError("compute barriers must be contiguous")

    writer_builder = launcher.builder[0]
    communication_builder = launcher.builder[1]
    communication_builder.add_communication(
        PoolSliceExchange(
            buffers.config_tensor,
            write_barrier=write_barrier,
            dispatch_barrier_base=dispatch_barrier_base,
            compute_barrier_base=compute_barrier_base,
        )
    )

    source_flat = buffers._source.view(-1)
    token_pool_flat = buffers.token_pool.view(-1)
    hidden_size = buffers.token_pool.shape[-1]
    for chunk in range(token_chunks):
        row_begin = chunk * chunk_rows
        rows = min(chunk_rows, buffers.token_capacity - row_begin)
        elements = rows * hidden_size
        offset = row_begin * hidden_size
        source = source_flat.narrow(0, offset, elements)
        destination = token_pool_flat.narrow(0, offset, elements)
        nbytes = rows * row_bytes
        writer_builder.add_memory(TmaLoad1D(source, bytes=nbytes))
        store = TmaStore1D(destination, bytes=nbytes)
        store.bar(write_barriers[chunk])
        writer_builder.add_memory(store)
        writer_builder.add_compute(Copy(1, nbytes))

    for local_reader in range(buffers.local_readers):
        builder = launcher.builder[local_reader + 2]
        builder.add_memory(IssueBarrier(dispatch_barriers[local_reader]))
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
            store = TmaStore1D(destination, bytes=nbytes)
            if chunk + 1 == reader_chunks:
                store.bar(compute_barriers[local_reader])
            builder.add_memory(store)
            builder.add_compute(Copy(1, nbytes))

    for builder in launcher.builder:
        builder.add_communication(TerminateComm())
        builder.add_memory(TerminateM())
        builder.add_compute(TerminateC())

    max_memory = max(2 * token_chunks + 1, 2 * reader_chunks + 2)
    max_compute = max(token_chunks + 1, reader_chunks + 1)
    if max_memory > launcher.max_insts or max_compute > launcher.max_insts:
        raise ValueError("pool capacity requires too many VDCores instructions")
    return PoolSliceProgram(
        launcher=launcher,
        write_barrier=write_barrier,
        dispatch_barriers=dispatch_barriers,
        compute_barriers=compute_barriers,
        chunk_rows=chunk_rows,
        communication_block=1,
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
    if num_pes > POOL_SLICE_MAX_PES:
        raise ValueError(f"a pool slice supports at most {POOL_SLICE_MAX_PES} PEs")
    if my_pe >= num_pes:
        raise ValueError("my_pe is outside the PE range")
    if local_readers > POOL_SLICE_MAX_LOCAL_READERS:
        raise ValueError(
            f"local_readers cannot exceed {POOL_SLICE_MAX_LOCAL_READERS}"
        )
    row_bytes = hidden_size * torch.empty((), dtype=dtype).element_size()
    if row_bytes < 1024 or row_bytes % 16:
        raise ValueError("a pool row must be at least 1024 bytes and 16-byte aligned")
    pack_warps = select_pool_slice_pack_warps(
        num_pes=num_pes,
        route_capacity=route_capacity,
        row_bytes=row_bytes,
        requested=pack_warps,
    )
    if signal_base + num_pes > signals.numel():
        raise ValueError("pool-slice signal range exceeds the signal tensor")
    write_chunk_rows = POOL_SLICE_MAX_TMA_BYTES // row_bytes
    write_chunks = (token_capacity + write_chunk_rows - 1) // write_chunk_rows

    num_readers = num_pes * local_readers
    send_offsets = nvshmem.zeros(num_readers + 1, dtype=torch.uint32)
    send_rows = nvshmem.zeros(route_capacity, dtype=torch.uint32)
    send_origin_rows = nvshmem.zeros(route_capacity, dtype=torch.uint32)
    token_pool = nvshmem.zeros((token_capacity, hidden_size), dtype=dtype)
    delivery_pool = nvshmem.zeros((route_capacity, hidden_size), dtype=dtype)
    expert_input = nvshmem.zeros(
        (local_readers, expert_capacity_rows, hidden_size), dtype=dtype
    )
    expert_output = nvshmem.zeros(
        (local_readers, expert_capacity_rows, hidden_size), dtype=dtype
    )
    return_inbox = nvshmem.zeros((route_capacity, hidden_size), dtype=dtype)
    send_batches = nvshmem.zeros(
        (num_pes, POOL_SLICE_PUBLISH_BYTES), dtype=torch.uint8
    )
    receive_batches = nvshmem.zeros(
        (num_pes, POOL_SLICE_PUBLISH_BYTES), dtype=torch.uint8
    )
    receive_routes = nvshmem.zeros(
        (local_readers, num_pes, POOL_SLICE_RECEIVE_BYTES), dtype=torch.uint8
    )
    sequence = nvshmem.zeros(1, dtype=torch.uint64)
    group_ready = nvshmem.zeros(1, dtype=torch.uint64)
    control = nvshmem.zeros(POOL_SLICE_CONTROL_WORDS, dtype=torch.uint64)
    config_tensor = torch.empty(
        POOL_SLICE_CONFIG_BYTES, dtype=torch.uint8, device=signals.device
    )

    buffers = PoolSliceBuffers(
        signals=signals,
        send_offsets=send_offsets,
        send_rows=send_rows,
        send_origin_rows=send_origin_rows,
        token_pool=token_pool,
        delivery_pool=delivery_pool,
        expert_input=expert_input,
        expert_output=expert_output,
        return_inbox=return_inbox,
        send_batches=send_batches,
        receive_batches=receive_batches,
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
    )
    nvshmem.barrier()
    return buffers


__all__ = [
    "POOL_SLICE_MAX_PES",
    "POOL_SLICE_MAX_LOCAL_READERS",
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
    "PoolSliceStatus",
    "PoolSliceBatchFlags",
    "PoolSlicePublishBatch",
    "PoolSliceReceiveBatch",
    "PoolSliceConfig",
    "PoolSliceBuffers",
    "PoolSliceProgram",
    "group_routes_by_reader",
    "select_pool_slice_pack_warps",
    "allocate_pool_slice",
    "build_pool_slice_copy_program",
]
