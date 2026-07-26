"""Pool-owned distributed slices and dynamic gathered reads.

The device ABI lives in ``include/dae/pool_slice_abi.cuh``. A producer only
publishes a compact route batch into every target slice's per-source mailbox.
The target pool communication warp consumes all metadata, resolves token-slot
readiness, gathers rows into local reader buffers, and owns the return path.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from numbers import Integral
from typing import TYPE_CHECKING, Sequence

import torch

if TYPE_CHECKING:
    from .launcher import Launcher

POOL_SLICE_PUBLISH_BYTES = 32
POOL_SLICE_RECEIVE_BYTES = 48
POOL_SLICE_CONFIG_BYTES = 240
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

_PUBLISH_STRUCT = struct.Struct("<Q6I")
_RECEIVE_STRUCT = struct.Struct("<2Q8I")
_CONFIG_STRUCT = struct.Struct("<18Q22I8x")
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


class PoolSliceFlags(IntFlag):
    NONE = 0
    STREAMING_GATHER = 1 << 0


def _uint(name: str, value: int, bits: int) -> int:
    if not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
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
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be CUDA")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    from . import nvshmem

    if not nvshmem.is_symmetric_tensor(tensor):
        raise ValueError(f"{name} must be in the NVSHMEM symmetric heap")
    return tensor


def _copy_packed(destination: torch.Tensor, payload: bytes) -> None:
    if (
        destination.dtype != torch.uint8
        or destination.numel() != len(payload)
        or not destination.is_contiguous()
    ):
        raise ValueError("packed destination has the wrong dtype or size")
    destination.copy_(torch.tensor(list(payload), dtype=torch.uint8))


@dataclass(frozen=True)
class PoolSlicePublishBatch:
    sequence: int
    source_pe: int
    target_pe: int
    active_rows: int
    route_begin: int = 0
    route_end: int = 0
    flags: PoolSliceBatchFlags | int = PoolSliceBatchFlags.NONE

    def pack(self) -> bytes:
        return _PUBLISH_STRUCT.pack(
            _positive_uint("sequence", self.sequence, 64),
            _uint("source_pe", self.source_pe, 32),
            _uint("target_pe", self.target_pe, 32),
            _uint("active_rows", self.active_rows, 32),
            _uint("flags", int(self.flags), 32),
            _uint("route_begin", self.route_begin, 32),
            _uint("route_end", self.route_end, 32),
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
            route_begin=values[5],
            route_end=values[6],
            flags=PoolSliceBatchFlags(values[4]),
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
            _positive_uint("sequence", self.sequence, 64),
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
    expert_input_address: int
    expert_output_address: int
    return_inbox_address: int
    returned_address: int
    send_offsets_address: int
    send_rows_address: int
    send_origin_rows_address: int
    send_batches_address: int
    receive_batches_address: int
    offsets_inbox_address: int
    rows_inbox_address: int
    receive_routes_address: int
    reader_tails_address: int
    sequence_address: int
    group_ready_address: int
    control_address: int
    row_bytes: int
    source_stride: int
    pool_stride: int
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
    queue_signal_base: int
    data_signal_base: int
    return_signal_base: int
    signal_count: int
    return_capacity_rows: int
    flags: int = 0
    data_stages: int = 1
    early_ready_rows: int = 0

    @property
    def num_readers(self) -> int:
        return self.local_readers * self.num_pes

    def pack(self) -> bytes:
        pointers = (
            self.source_address,
            self.token_pool_address,
            self.expert_input_address,
            self.expert_output_address,
            self.return_inbox_address,
            self.returned_address,
            self.send_offsets_address,
            self.send_rows_address,
            self.send_origin_rows_address,
            self.send_batches_address,
            self.receive_batches_address,
            self.offsets_inbox_address,
            self.rows_inbox_address,
            self.receive_routes_address,
            self.reader_tails_address,
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
            _uint("queue_signal_base", self.queue_signal_base, 32),
            _uint("data_signal_base", self.data_signal_base, 32),
            _uint("return_signal_base", self.return_signal_base, 32),
            _positive_uint("signal_count", self.signal_count, 32),
            _positive_uint(
                "return_capacity_rows", self.return_capacity_rows, 32
            ),
            _uint("flags", self.flags, 32),
            _positive_uint("data_stages", self.data_stages, 32),
            _uint("early_ready_rows", self.early_ready_rows, 32),
            0,
        )
        if self.my_pe >= self.num_pes:
            raise ValueError("my_pe is outside the PE range")
        if self.num_pes > 32:
            raise ValueError("a pool slice supports at most 32 PEs")
        if self.local_readers >= 132:
            raise ValueError("local_readers must leave one SM for the pool core")
        if self.row_bytes < 1024 or self.row_bytes % 16:
            raise ValueError("row_bytes must be at least 1024 and a multiple of 16")
        for name, stride in (
            ("source_stride", self.source_stride),
            ("pool_stride", self.pool_stride),
            ("return_stride", self.return_stride),
        ):
            if stride < self.row_bytes or stride % 16:
                raise ValueError(f"{name} must cover an aligned row")
        if self.expert_row_stride != self.row_bytes:
            raise ValueError("expert rows must be contiguous")
        if self.expert_stride < self.expert_capacity_rows * self.row_bytes:
            raise ValueError("expert_stride does not cover expert capacity")
        if self.active_rows > self.route_capacity:
            raise ValueError("active_rows exceeds route_capacity")
        if self.flags & ~int(PoolSliceFlags.STREAMING_GATHER):
            raise ValueError("flags contains an unsupported pool-slice flag")
        if self.data_stages not in (1, 2):
            raise ValueError("data_stages must be one or two")
        if self.data_stages == 1 and self.early_ready_rows != 0:
            raise ValueError("one data stage cannot have an early-ready prefix")
        if self.data_stages == 2 and not (
            0 < self.early_ready_rows < self.token_capacity
        ):
            raise ValueError(
                "two data stages require a nonempty strict row prefix"
            )
        if self.data_stages == 2 and not (
            self.flags & int(PoolSliceFlags.STREAMING_GATHER)
        ):
            raise ValueError("two data stages require streaming gather")
        signal_ranges = (
            ("queue", self.queue_signal_base),
            ("data", self.data_signal_base),
            ("return", self.return_signal_base),
        )
        for name, base in signal_ranges:
            if base + self.num_pes > self.signal_count:
                raise ValueError(f"{name} signal range exceeds signal_count")
        for index, (left_name, left_base) in enumerate(signal_ranges):
            for right_name, right_base in signal_ranges[index + 1 :]:
                if not (
                    left_base + self.num_pes <= right_base
                    or right_base + self.num_pes <= left_base
                ):
                    raise ValueError(
                        f"{left_name} and {right_name} signal ranges overlap"
                    )
        return _CONFIG_STRUCT.pack(*pointer_values, *values)


def group_routes_by_reader(
    reader_ids: Sequence[int] | torch.Tensor,
    *,
    num_readers: int,
    source_rows: Sequence[int] | torch.Tensor | None = None,
    origin_rows: Sequence[int] | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return stable reader-grouped slot and provenance metadata on CPU."""

    num_readers = _positive_uint("num_readers", num_readers, 32)
    readers = torch.as_tensor(reader_ids, dtype=torch.int64, device="cpu").view(-1)
    count = readers.numel()
    if count and (
        readers.min().item() < 0 or readers.max().item() >= num_readers
    ):
        raise ValueError("reader_ids contains an id outside the configured range")
    sources = (
        torch.arange(count, dtype=torch.int64)
        if source_rows is None
        else torch.as_tensor(source_rows, dtype=torch.int64, device="cpu").view(-1)
    )
    origins = (
        torch.arange(count, dtype=torch.int64)
        if origin_rows is None
        else torch.as_tensor(origin_rows, dtype=torch.int64, device="cpu").view(-1)
    )
    if sources.numel() != count or origins.numel() != count:
        raise ValueError("reader_ids, source_rows, and origin_rows must match")
    if count and (sources.min().item() < 0 or origins.min().item() < 0):
        raise ValueError("row ids must be non-negative")
    if count and (
        sources.max().item() >= 2**32 or origins.max().item() >= 2**32
    ):
        raise ValueError("row ids must fit in uint32")

    order = torch.argsort(readers, stable=True)
    counts = torch.bincount(readers, minlength=num_readers)
    offsets = torch.empty(num_readers + 1, dtype=torch.int64)
    offsets[0] = 0
    offsets[1:] = counts.cumsum(0)
    return (
        offsets.to(torch.uint32),
        sources.index_select(0, order).to(torch.uint32),
        origins.index_select(0, order).to(torch.uint32),
    )


@dataclass
class PoolSliceBuffers:
    signals: torch.Tensor
    send_offsets: torch.Tensor
    send_rows: torch.Tensor
    send_origin_rows: torch.Tensor
    token_pool: torch.Tensor
    expert_input: torch.Tensor
    expert_output: torch.Tensor
    return_inbox: torch.Tensor
    send_batches: torch.Tensor
    receive_batches: torch.Tensor
    offsets_inbox: torch.Tensor
    rows_inbox: torch.Tensor
    receive_routes: torch.Tensor
    reader_tails: torch.Tensor
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
    queue_signal_base: int
    data_signal_base: int
    return_signal_base: int
    protocol_flags: PoolSliceFlags
    data_stages: int
    early_ready_rows: int
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
            expert_input_address=self.expert_input.data_ptr(),
            expert_output_address=self.expert_output.data_ptr(),
            return_inbox_address=self.return_inbox.data_ptr(),
            returned_address=returned.data_ptr(),
            send_offsets_address=self.send_offsets.data_ptr(),
            send_rows_address=self.send_rows.data_ptr(),
            send_origin_rows_address=self.send_origin_rows.data_ptr(),
            send_batches_address=self.send_batches.data_ptr(),
            receive_batches_address=self.receive_batches.data_ptr(),
            offsets_inbox_address=self.offsets_inbox.data_ptr(),
            rows_inbox_address=self.rows_inbox.data_ptr(),
            receive_routes_address=self.receive_routes.data_ptr(),
            reader_tails_address=self.reader_tails.data_ptr(),
            sequence_address=self.sequence.data_ptr(),
            group_ready_address=self.group_ready.data_ptr(),
            control_address=self.control.data_ptr(),
            row_bytes=self.row_bytes,
            source_stride=source.stride(0) * element_bytes,
            pool_stride=self.token_pool.stride(0) * element_bytes,
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
            queue_signal_base=self.queue_signal_base,
            data_signal_base=self.data_signal_base,
            return_signal_base=self.return_signal_base,
            signal_count=self.signals.numel(),
            return_capacity_rows=returned.shape[0],
            flags=int(self.protocol_flags),
            data_stages=self.data_stages,
            early_ready_rows=self.early_ready_rows,
        )

    def prepare(self, source: torch.Tensor, returned: torch.Tensor) -> torch.Tensor:
        self._source = source
        self._returned = returned
        _copy_packed(self.config_tensor, self.config(source, returned).pack())
        return self.config_tensor

    def set_sequence(self, sequence: int) -> None:
        sequence = _positive_uint("sequence", sequence, 64)
        if sequence > ((1 << 64) - 1) // self.data_stages:
            raise ValueError("sequence is too large for staged data signals")
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

    def streaming_state(self) -> tuple[int, int, int]:
        """Return metadata waves, payload sources, and peak source batches."""

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
    streaming_gather: bool = False

    def launch(self) -> None:
        self.launcher.launch()

    def timing_ns(self) -> tuple[int, int, int]:
        events = self.launcher.profile[
            0,
            POOL_SLICE_PROFILE_START : POOL_SLICE_PROFILE_DONE + 1,
        ].cpu().to(torch.int64)
        start, gather_ready, done = (int(value) for value in events.tolist())
        if start == 0 or gather_ready < start or done < gather_ready:
            raise RuntimeError("pool-slice profile events are incomplete")
        return gather_ready - start, done - gather_ready, done - start

    def overlap_timing_ns(self) -> dict[str, int | None]:
        """Return streaming-gather event offsets from the program start."""

        if not self.streaming_gather:
            raise RuntimeError("overlap events require streaming gather")
        indices = (
            POOL_SLICE_PROFILE_START,
            POOL_SLICE_PROFILE_FIRST_DATA_PUBLISHED,
            POOL_SLICE_PROFILE_DATA_PUBLISHED,
            POOL_SLICE_PROFILE_FIRST_PAYLOAD,
            POOL_SLICE_PROFILE_METADATA_CLOSED,
            POOL_SLICE_PROFILE_PAYLOAD_DONE,
            POOL_SLICE_PROFILE_GATHER_READY,
        )
        profile = self.launcher.profile[0].cpu().to(torch.int64)
        events = profile[list(indices)]
        (
            start,
            first_data_published,
            data_published,
            first_payload,
            metadata_closed,
            payload_done,
            gather_ready,
        ) = (int(value) for value in events.tolist())
        required = (
            first_data_published,
            data_published,
            metadata_closed,
            payload_done,
            gather_ready,
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
        }


def build_pool_slice_copy_program(
    buffers: PoolSliceBuffers,
    *,
    benchmark_barrier=None,
) -> PoolSliceProgram:
    """Build one pool-owned write/gather/identity/return VDCores program."""

    from .instructions import (
        CommRecordEvent,
        Copy,
        IssueBarrier,
        PoolSliceGather,
        PoolSlicePublish,
        PoolSliceReturn,
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
    num_sms = buffers.local_readers + 1
    if num_sms > properties.multi_processor_count:
        raise ValueError(
            "pool core plus local readers exceeds the GPU SM count "
            f"({properties.multi_processor_count})"
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
        launcher.new_bar(1) for _ in range(buffers.data_stages)
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

    config_tensor = buffers.config_tensor
    pool_builder = launcher.builder[0]
    pool_builder.add_communication(CommRecordEvent(POOL_SLICE_PROFILE_START))
    # Source publication is also a pool-core action.  Keeping publish and
    # receive on block 0 makes the physical ownership match the protocol: the
    # application produces only HBM route metadata, while the pool PE moves
    # that metadata into every remote slice and then scans its local queues.
    pool_builder.add_communication(PoolSlicePublish(config_tensor))
    pool_builder.add_communication(
        PoolSliceGather(
            config_tensor,
            write_barrier=write_barrier,
            dispatch_barrier_base=dispatch_barrier_base,
        )
    )
    pool_builder.add_communication(
        CommRecordEvent(POOL_SLICE_PROFILE_GATHER_READY)
    )
    pool_builder.add_communication(
        PoolSliceReturn(
            config_tensor,
            compute_barrier_base=compute_barrier_base,
        )
    )
    pool_builder.add_communication(CommRecordEvent(POOL_SLICE_PROFILE_DONE))
    pool_builder.add_communication(TerminateComm())

    # Reader blocks contain no communication operations.  They are released
    # only by barriers resolved by the pool communication warp.
    for builder in launcher.builder[1:]:
        builder.add_communication(TerminateComm())

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
        pool_builder.add_memory(TmaLoad1D(source, bytes=nbytes))
        store = TmaStore1D(destination, bytes=nbytes)
        if (
            buffers.data_stages == 2
            and row_begin + rows == buffers.early_ready_rows
        ):
            store.bar(write_barriers[0])
        if chunk + 1 == token_chunks:
            store.bar(write_barriers[-1])
        pool_builder.add_memory(store)
        pool_builder.add_compute(Copy(1, nbytes))

    for local_reader in range(buffers.local_readers):
        builder = launcher.builder[local_reader + 1]
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
        streaming_gather=bool(
            buffers.protocol_flags & PoolSliceFlags.STREAMING_GATHER
        ),
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
    queue_signal_base: int = 0,
    data_signal_base: int | None = None,
    return_signal_base: int | None = None,
    streaming_gather: bool = True,
    activation_stages: int = 1,
) -> PoolSliceBuffers:
    """Collectively allocate one logical pool slice on every PE."""

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
    if not isinstance(streaming_gather, bool):
        raise TypeError("streaming_gather must be a bool")
    activation_stages = _positive_uint(
        "activation_stages", activation_stages, 32
    )
    if activation_stages not in (1, 2):
        raise ValueError("activation_stages must be one or two")
    if activation_stages == 2 and not streaming_gather:
        raise ValueError("two activation stages require streaming gather")
    if num_pes > 32:
        raise ValueError("a pool slice supports at most 32 PEs")
    if my_pe >= num_pes:
        raise ValueError("my_pe is outside the PE range")
    if local_readers >= 132:
        raise ValueError("local_readers must leave one SM for the pool core")
    row_bytes = hidden_size * torch.empty((), dtype=dtype).element_size()
    if row_bytes < 1024 or row_bytes % 16:
        raise ValueError("a pool row must be at least 1024 bytes and 16-byte aligned")
    chunk_rows = POOL_SLICE_MAX_TMA_BYTES // row_bytes
    token_chunks = (token_capacity + chunk_rows - 1) // chunk_rows
    if activation_stages == 2 and token_chunks < 2:
        raise ValueError("two activation stages require at least two write chunks")
    early_ready_rows = (
        chunk_rows if activation_stages == 2 else 0
    )

    queue_signal_base = _uint("queue_signal_base", queue_signal_base, 32)
    if data_signal_base is None:
        data_signal_base = queue_signal_base + num_pes
    data_signal_base = _uint("data_signal_base", data_signal_base, 32)
    if return_signal_base is None:
        return_signal_base = data_signal_base + num_pes
    return_signal_base = _uint("return_signal_base", return_signal_base, 32)
    if max(
        queue_signal_base + num_pes,
        data_signal_base + num_pes,
        return_signal_base + num_pes,
    ) > signals.numel():
        raise ValueError("pool-slice signal ranges exceed the signal tensor")

    num_readers = num_pes * local_readers
    send_offsets = nvshmem.zeros(num_readers + 1, dtype=torch.uint32)
    send_rows = nvshmem.zeros(route_capacity, dtype=torch.uint32)
    send_origin_rows = nvshmem.zeros(route_capacity, dtype=torch.uint32)
    token_pool = nvshmem.zeros((token_capacity, hidden_size), dtype=dtype)
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
    offsets_inbox = nvshmem.zeros(
        (num_pes, local_readers + 1), dtype=torch.uint32
    )
    rows_inbox = nvshmem.zeros(
        (num_pes, route_capacity), dtype=torch.uint32
    )
    receive_routes = nvshmem.zeros(
        (local_readers, num_pes, POOL_SLICE_RECEIVE_BYTES), dtype=torch.uint8
    )
    reader_tails = nvshmem.zeros(local_readers, dtype=torch.uint64)
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
        expert_input=expert_input,
        expert_output=expert_output,
        return_inbox=return_inbox,
        send_batches=send_batches,
        receive_batches=receive_batches,
        offsets_inbox=offsets_inbox,
        rows_inbox=rows_inbox,
        receive_routes=receive_routes,
        reader_tails=reader_tails,
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
        queue_signal_base=queue_signal_base,
        data_signal_base=data_signal_base,
        return_signal_base=return_signal_base,
        protocol_flags=(
            PoolSliceFlags.STREAMING_GATHER
            if streaming_gather
            else PoolSliceFlags.NONE
        ),
        data_stages=activation_stages,
        early_ready_rows=early_ready_rows,
    )
    nvshmem.barrier()
    return buffers


__all__ = [
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
    "PoolSliceStatus",
    "PoolSliceBatchFlags",
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
