"""Receiver-owned sharded HBM pools for expert-parallel communication.

The device ABI lives in ``include/dae/ep_pool_abi.cuh``. Producers provide
router output as expert-grouped compressed rows; the VDCores communication warp
reserves contiguous rows on each target expert and publishes one batch signal
per peer rather than one completion per token.
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

EP_BATCH_BYTES = 48
EP_CONFIG_BYTES = 192
EP_CONTROL_WORDS = 8
EP_MAX_TMA_BYTES = (1 << 16) - 1
EP_PROFILE_START = 2
EP_PROFILE_DISPATCH_READY = 3
EP_PROFILE_DONE = 4

_BATCH_STRUCT = struct.Struct("<2Q8I")
_CONFIG_STRUCT = struct.Struct("<14Q20I")
assert _BATCH_STRUCT.size == EP_BATCH_BYTES
assert _CONFIG_STRUCT.size == EP_CONFIG_BYTES


class ExpertPoolStatus(IntEnum):
    OK = 0
    BAD_CONFIG = 1
    ROUTE_RANGE = 2
    CAPACITY = 3
    SEQUENCE = 4
    BATCH = 5
    SIGNAL_RANGE = 6


class ExpertPoolBatchFlags(IntFlag):
    NONE = 0
    ERROR = 1 << 0


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
    source = torch.tensor(list(payload), dtype=torch.uint8)
    destination.view(-1).copy_(source, non_blocking=False)


@dataclass(frozen=True)
class ExpertPoolBatch:
    sequence: int
    base_row: int
    source_base: int
    row_count: int
    source_pe: int
    local_expert: int
    flags: ExpertPoolBatchFlags | int = ExpertPoolBatchFlags.NONE

    def pack(self) -> bytes:
        return _BATCH_STRUCT.pack(
            _positive_uint("sequence", self.sequence, 64),
            _uint("base_row", self.base_row, 64),
            _uint("source_base", self.source_base, 32),
            _uint("row_count", self.row_count, 32),
            _uint("source_pe", self.source_pe, 32),
            _uint("local_expert", self.local_expert, 32),
            _uint("flags", int(self.flags), 32),
            0,
            0,
            0,
        )

    @classmethod
    def unpack(cls, payload: bytes | bytearray | memoryview) -> "ExpertPoolBatch":
        if len(payload) != EP_BATCH_BYTES:
            raise ValueError(f"batch payload must be {EP_BATCH_BYTES} bytes")
        values = _BATCH_STRUCT.unpack(payload)
        return cls(
            sequence=values[0],
            base_row=values[1],
            source_base=values[2],
            row_count=values[3],
            source_pe=values[4],
            local_expert=values[5],
            flags=ExpertPoolBatchFlags(values[6]),
        )


@dataclass(frozen=True)
class ExpertPoolConfig:
    source_address: int
    packed_source_address: int
    expert_input_address: int
    expert_output_address: int
    return_inbox_address: int
    returned_address: int
    send_offsets_address: int
    send_rows_address: int
    send_origin_rows_address: int
    send_batches_address: int
    receive_batches_address: int
    expert_tails_address: int
    sequence_address: int
    control_address: int
    row_bytes: int
    source_stride: int
    expert_row_stride: int
    return_stride: int
    expert_stride: int
    active_rows: int
    route_capacity: int
    expert_capacity_rows: int
    num_experts: int
    experts_per_pe: int
    num_pes: int
    my_pe: int
    dispatch_signal_base: int
    return_signal_base: int
    reset_signal_base: int
    signal_count: int
    return_capacity_rows: int
    flags: int = 0
    source_capacity_rows: int = 1

    def pack(self) -> bytes:
        pointers = (
            self.source_address,
            self.packed_source_address,
            self.expert_input_address,
            self.expert_output_address,
            self.return_inbox_address,
            self.returned_address,
            self.send_offsets_address,
            self.send_rows_address,
            self.send_origin_rows_address,
            self.send_batches_address,
            self.receive_batches_address,
            self.expert_tails_address,
            self.sequence_address,
            self.control_address,
        )
        pointer_values = tuple(
            _positive_uint(f"pointer[{index}]", value, 64)
            for index, value in enumerate(pointers)
        )
        values = (
            _positive_uint("row_bytes", self.row_bytes, 32),
            _positive_uint("source_stride", self.source_stride, 32),
            _positive_uint("expert_row_stride", self.expert_row_stride, 32),
            _positive_uint("return_stride", self.return_stride, 32),
            _positive_uint("expert_stride", self.expert_stride, 32),
            _uint("active_rows", self.active_rows, 32),
            _positive_uint("route_capacity", self.route_capacity, 32),
            _positive_uint("expert_capacity_rows", self.expert_capacity_rows, 32),
            _positive_uint("num_experts", self.num_experts, 32),
            _positive_uint("experts_per_pe", self.experts_per_pe, 32),
            _positive_uint("num_pes", self.num_pes, 32),
            _uint("my_pe", self.my_pe, 32),
            _uint("dispatch_signal_base", self.dispatch_signal_base, 32),
            _uint("return_signal_base", self.return_signal_base, 32),
            _uint("reset_signal_base", self.reset_signal_base, 32),
            _positive_uint("signal_count", self.signal_count, 32),
            _uint("flags", self.flags, 32),
            _positive_uint("source_capacity_rows", self.source_capacity_rows, 32),
            _positive_uint("return_capacity_rows", self.return_capacity_rows, 32),
            0,
        )
        if self.num_experts != self.num_pes * self.experts_per_pe:
            raise ValueError("num_experts must equal num_pes * experts_per_pe")
        if self.my_pe >= self.num_pes:
            raise ValueError("my_pe is outside the PE range")
        if self.row_bytes < 1024 or self.row_bytes % 16:
            raise ValueError("row_bytes must be at least 1024 and a multiple of 16")
        if self.num_pes > 32:
            raise ValueError("the communication warp supports at most 32 PEs")
        if self.num_experts > 132:
            raise ValueError("one-block-per-expert requires at most 132 experts")
        if self.source_stride < self.row_bytes or self.source_stride % 16:
            raise ValueError("source_stride must cover one row")
        if self.expert_row_stride != self.row_bytes:
            raise ValueError("expert rows must be contiguous")
        if self.return_stride < self.row_bytes or self.return_stride % 16:
            raise ValueError("return_stride must cover one row")
        if self.expert_stride < self.expert_capacity_rows * self.expert_row_stride:
            raise ValueError("expert_stride does not cover expert capacity")
        if self.active_rows > self.route_capacity:
            raise ValueError("active_rows exceeds route_capacity")
        if (
            self.dispatch_signal_base + self.num_experts * self.num_pes
            > self.signal_count
        ):
            raise ValueError("dispatch signal range exceeds signal_count")
        if self.return_signal_base + self.num_experts > self.signal_count:
            raise ValueError("return signal range exceeds signal_count")
        if self.reset_signal_base + self.num_pes > self.signal_count:
            raise ValueError("reset signal range exceeds signal_count")
        return _CONFIG_STRUCT.pack(*pointer_values, *values)


def group_routes_by_expert(
    expert_ids: Sequence[int] | torch.Tensor,
    *,
    num_experts: int,
    source_rows: Sequence[int] | torch.Tensor | None = None,
    origin_rows: Sequence[int] | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return prefix offsets and stable expert-grouped row metadata on CPU."""

    num_experts = _positive_uint("num_experts", num_experts, 32)
    experts = torch.as_tensor(expert_ids, dtype=torch.int64, device="cpu").view(-1)
    count = experts.numel()
    if count and (experts.min().item() < 0 or experts.max().item() >= num_experts):
        raise ValueError("expert_ids contains an expert outside the configured range")
    if source_rows is None:
        sources = torch.arange(count, dtype=torch.int64)
    else:
        sources = torch.as_tensor(source_rows, dtype=torch.int64, device="cpu").view(-1)
    if origin_rows is None:
        origins = torch.arange(count, dtype=torch.int64)
    else:
        origins = torch.as_tensor(origin_rows, dtype=torch.int64, device="cpu").view(-1)
    if sources.numel() != count or origins.numel() != count:
        raise ValueError("expert_ids, source_rows, and origin_rows must have equal length")
    if count and (sources.min().item() < 0 or origins.min().item() < 0):
        raise ValueError("row ids must be non-negative")
    if count and (
        sources.max().item() >= 2**32 or origins.max().item() >= 2**32
    ):
        raise ValueError("row ids must fit in uint32")

    order = torch.argsort(experts, stable=True)
    counts = torch.bincount(experts, minlength=num_experts)
    offsets = torch.empty(num_experts + 1, dtype=torch.int64)
    offsets[0] = 0
    offsets[1:] = counts.cumsum(0)
    return (
        offsets.to(torch.uint32),
        sources.index_select(0, order).to(torch.uint32),
        origins.index_select(0, order).to(torch.uint32),
    )


@dataclass
class ExpertPoolBuffers:
    signals: torch.Tensor
    send_offsets: torch.Tensor
    send_rows: torch.Tensor
    send_origin_rows: torch.Tensor
    packed_source: torch.Tensor
    expert_input: torch.Tensor
    expert_output: torch.Tensor
    return_inbox: torch.Tensor
    send_batches: torch.Tensor
    receive_batches: torch.Tensor
    expert_tails: torch.Tensor
    sequence: torch.Tensor
    control: torch.Tensor
    config_tensor: torch.Tensor
    num_pes: int
    my_pe: int
    experts_per_pe: int
    token_capacity: int
    route_capacity: int
    expert_capacity_rows: int
    dispatch_signal_base: int
    return_signal_base: int
    reset_signal_base: int
    active_rows: int = 0
    _source: torch.Tensor | None = None
    _returned: torch.Tensor | None = None
    _required_return_rows: int = 0
    _last_sequence: int = 0

    @property
    def num_experts(self) -> int:
        return self.num_pes * self.experts_per_pe

    @property
    def row_bytes(self) -> int:
        return self.expert_input.shape[-1] * self.expert_input.element_size()

    def write_routes(
        self,
        expert_ids: Sequence[int] | torch.Tensor,
        *,
        source_rows: Sequence[int] | torch.Tensor | None = None,
        origin_rows: Sequence[int] | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        offsets, rows, origins = group_routes_by_expert(
            expert_ids,
            num_experts=self.num_experts,
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
        self.send_offsets.copy_(offsets, non_blocking=False)
        self.send_rows.zero_()
        self.send_origin_rows.zero_()
        if rows.numel():
            self.send_rows[: rows.numel()].copy_(rows, non_blocking=False)
            self.send_origin_rows[: origins.numel()].copy_(origins, non_blocking=False)
        self.active_rows = rows.numel()
        return self.send_offsets, self.send_rows[: self.active_rows], self.send_origin_rows[: self.active_rows]

    def config(self, source: torch.Tensor, returned: torch.Tensor) -> ExpertPoolConfig:
        source = _symmetric_tensor(source, "source")
        returned = _symmetric_tensor(returned, "returned")
        if source.ndim != 2 or returned.ndim != 2:
            raise ValueError("source and returned must be rank-2 row tensors")
        if source.dtype != self.expert_input.dtype or returned.dtype != source.dtype:
            raise ValueError("source, expert pool, and returned tensors must share dtype")
        if source.shape[1] != self.expert_input.shape[2] or returned.shape[1] != source.shape[1]:
            raise ValueError("source, expert pool, and returned row widths must match")
        if source.shape[0] < self.token_capacity or returned.shape[0] < self.token_capacity:
            raise ValueError("source and returned must cover token_capacity rows")
        if returned.shape[0] < self._required_return_rows:
            raise ValueError("returned does not cover the largest origin row")
        element_bytes = source.element_size()
        return ExpertPoolConfig(
            source_address=source.data_ptr(),
            packed_source_address=self.packed_source.data_ptr(),
            expert_input_address=self.expert_input.data_ptr(),
            expert_output_address=self.expert_output.data_ptr(),
            return_inbox_address=self.return_inbox.data_ptr(),
            returned_address=returned.data_ptr(),
            send_offsets_address=self.send_offsets.data_ptr(),
            send_rows_address=self.send_rows.data_ptr(),
            send_origin_rows_address=self.send_origin_rows.data_ptr(),
            send_batches_address=self.send_batches.data_ptr(),
            receive_batches_address=self.receive_batches.data_ptr(),
            expert_tails_address=self.expert_tails.data_ptr(),
            sequence_address=self.sequence.data_ptr(),
            control_address=self.control.data_ptr(),
            row_bytes=self.row_bytes,
            source_stride=source.stride(0) * element_bytes,
            expert_row_stride=self.expert_input.stride(1) * element_bytes,
            return_stride=returned.stride(0) * element_bytes,
            expert_stride=self.expert_input.stride(0) * element_bytes,
            active_rows=self.active_rows,
            route_capacity=self.route_capacity,
            expert_capacity_rows=self.expert_capacity_rows,
            num_experts=self.num_experts,
            experts_per_pe=self.experts_per_pe,
            num_pes=self.num_pes,
            my_pe=self.my_pe,
            dispatch_signal_base=self.dispatch_signal_base,
            return_signal_base=self.return_signal_base,
            reset_signal_base=self.reset_signal_base,
            signal_count=self.signals.numel(),
            source_capacity_rows=source.shape[0],
            return_capacity_rows=returned.shape[0],
        )

    def prepare(self, source: torch.Tensor, returned: torch.Tensor) -> torch.Tensor:
        self._source = source
        self._returned = returned
        _copy_packed(self.config_tensor, self.config(source, returned).pack())
        return self.config_tensor

    def set_sequence(self, sequence: int) -> None:
        sequence = _positive_uint("sequence", sequence, 64)
        if sequence <= self._last_sequence:
            raise ValueError("sequence must increase monotonically")
        self.sequence.fill_(sequence)
        self._last_sequence = sequence

    def reset_dispatch(self, sequence: int = 1) -> None:
        """Set the phase sequence; the VDCores reset op clears pool state."""

        self.set_sequence(sequence)

    def control_state(self) -> tuple[ExpertPoolStatus, int, int, int, int]:
        values = self.control.cpu().tolist()
        return (
            ExpertPoolStatus(values[0]),
            values[1],
            values[2],
            values[3],
            values[4],
        )

    def read_receive_batches(self) -> list[list[ExpertPoolBatch]]:
        raw = self.receive_batches.cpu()
        result: list[list[ExpertPoolBatch]] = []
        for local_expert in range(self.experts_per_pe):
            expert_batches = []
            for source_pe in range(self.num_pes):
                payload = bytes(raw[local_expert, source_pe].tolist())
                expert_batches.append(ExpertPoolBatch.unpack(payload))
            result.append(expert_batches)
        return result


@dataclass(frozen=True)
class ExpertPoolProgram:
    """One persistent, mixed-domain VDCores EP program."""

    launcher: "Launcher"
    reset_barrier: int
    dispatch_barriers: tuple[int, ...]
    compute_barriers: tuple[int, ...]
    chunk_rows: int

    def launch(self) -> None:
        self.launcher.launch()

    def timing_ns(self) -> tuple[int, int, int]:
        """Return dispatch-ready, overlapped tail, and end-to-end nanoseconds."""

        events = self.launcher.profile[
            :, EP_PROFILE_START : EP_PROFILE_DONE + 1
        ].cpu().to(torch.int64)
        start = int(events[:, 0].min().item())
        dispatch_ready = int(events[:, 1].max().item())
        done = int(events[:, 2].max().item())
        if start == 0 or dispatch_ready < start or done < dispatch_ready:
            raise RuntimeError("expert-pool profile events are incomplete")
        return dispatch_ready - start, done - dispatch_ready, done - start


def build_expert_pool_copy_program(
    buffers: ExpertPoolBuffers,
    *,
    benchmark_barrier=None,
) -> ExpertPoolProgram:
    """Build a one-launch identity-expert EP program from VDCores operators.

    This correctness/communication harness deliberately uses the existing
    ``Copy`` compute operator. Real expert schedules can replace the owning
    block's memory/compute subsequences while retaining the same communication
    instructions and two barriers.
    """

    from .instructions import (
        CommWaitBarrier,
        CommRecordEvent,
        Copy,
        ExpertPoolDispatch,
        ExpertPoolReset,
        ExpertPoolReturn,
        IssueBarrier,
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
    if buffers.num_experts > properties.multi_processor_count:
        raise ValueError(
            "the resident one-block-per-expert program requires num_experts "
            f"<= GPU SM count ({properties.multi_processor_count})"
        )

    row_bytes = buffers.row_bytes
    chunk_rows = EP_MAX_TMA_BYTES // row_bytes
    if chunk_rows == 0:
        raise ValueError(
            f"row_bytes={row_bytes} exceeds the 16-bit VDCores TMA size field"
        )
    chunks = (
        buffers.expert_capacity_rows + chunk_rows - 1
    ) // chunk_rows

    launcher = Launcher(
        num_sms=buffers.num_experts,
        device=device,
        signal_array=buffers.signals,
        benchmark_barrier=benchmark_barrier,
    )
    reset_barrier = launcher.new_bar(1)
    dispatch_barriers = tuple(
        launcher.new_bar(1) for _ in range(buffers.num_experts)
    )
    compute_barriers = tuple(
        launcher.new_bar(1) for _ in range(buffers.num_experts)
    )

    config_tensor = buffers.config_tensor
    hidden_size = buffers.expert_input.shape[-1]
    for global_expert, builder in enumerate(launcher.builder):
        builder.add_communication(CommRecordEvent(EP_PROFILE_START))
        if global_expert == 0:
            builder.add_communication(
                ExpertPoolReset(config_tensor, reset_barrier)
            )
        else:
            builder.add_communication(CommWaitBarrier(reset_barrier))
        builder.add_communication(
            ExpertPoolDispatch(
                config_tensor,
                global_expert,
                dispatch_barriers[global_expert],
            )
        )
        builder.add_communication(CommRecordEvent(EP_PROFILE_DISPATCH_READY))
        builder.add_communication(
            ExpertPoolReturn(
                config_tensor,
                global_expert,
                compute_barriers[global_expert],
            )
        )
        builder.add_communication(CommRecordEvent(EP_PROFILE_DONE))
        builder.add_communication(TerminateComm())

        owner_pe = global_expert // buffers.experts_per_pe
        if owner_pe == buffers.my_pe:
            local_expert = global_expert % buffers.experts_per_pe
            builder.add_memory(IssueBarrier(dispatch_barriers[global_expert]))
            input_flat = buffers.expert_input[local_expert].view(-1)
            output_flat = buffers.expert_output[local_expert].view(-1)
            for chunk in range(chunks):
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
                if chunk + 1 == chunks:
                    store.bar(compute_barriers[global_expert])
                builder.add_memory(store)
                builder.add_compute(Copy(1, nbytes))

        builder.add_memory(TerminateM())
        builder.add_compute(TerminateC())

    if 2 * chunks + 2 > launcher.max_insts:
        raise ValueError("expert capacity requires too many memory instructions")
    if chunks + 1 > launcher.max_insts:
        raise ValueError("expert capacity requires too many compute instructions")
    return ExpertPoolProgram(
        launcher=launcher,
        reset_barrier=reset_barrier,
        dispatch_barriers=dispatch_barriers,
        compute_barriers=compute_barriers,
        chunk_rows=chunk_rows,
    )


def allocate_expert_pool(
    signals: torch.Tensor,
    *,
    num_pes: int,
    my_pe: int,
    experts_per_pe: int,
    token_capacity: int,
    route_capacity: int | None = None,
    expert_capacity_rows: int,
    hidden_size: int,
    dtype: torch.dtype = torch.bfloat16,
    dispatch_signal_base: int = 0,
    return_signal_base: int | None = None,
    reset_signal_base: int | None = None,
) -> ExpertPoolBuffers:
    """Collectively allocate one receiver-owned pool shard on every PE."""

    from . import nvshmem

    signals = _symmetric_tensor(signals, "signals")
    if signals.dtype != torch.uint64 or signals.ndim != 1:
        raise ValueError("signals must be a contiguous rank-1 uint64 tensor")
    num_pes = _positive_uint("num_pes", num_pes, 32)
    my_pe = _uint("my_pe", my_pe, 32)
    experts_per_pe = _positive_uint("experts_per_pe", experts_per_pe, 32)
    if num_pes > 32:
        raise ValueError("the communication warp supports at most 32 PEs")
    token_capacity = _positive_uint("token_capacity", token_capacity, 32)
    if route_capacity is None:
        route_capacity = token_capacity
    route_capacity = _positive_uint("route_capacity", route_capacity, 32)
    expert_capacity_rows = _positive_uint(
        "expert_capacity_rows", expert_capacity_rows, 32
    )
    hidden_size = _positive_uint("hidden_size", hidden_size, 32)
    row_bytes = hidden_size * torch.empty((), dtype=dtype).element_size()
    if row_bytes < 1024 or row_bytes % 16:
        raise ValueError("an expert-pool row must be at least 1024 bytes and 16-byte aligned")
    num_experts = num_pes * experts_per_pe
    if num_experts > 132:
        raise ValueError("one-block-per-expert requires at most 132 experts")
    dispatch_signal_base = _uint("dispatch_signal_base", dispatch_signal_base, 32)
    if return_signal_base is None:
        return_signal_base = dispatch_signal_base + num_experts * num_pes
    return_signal_base = _uint("return_signal_base", return_signal_base, 32)
    if reset_signal_base is None:
        reset_signal_base = return_signal_base + num_experts
    reset_signal_base = _uint("reset_signal_base", reset_signal_base, 32)
    if my_pe >= num_pes:
        raise ValueError("my_pe is outside the PE range")
    if max(
        dispatch_signal_base + num_experts * num_pes,
        return_signal_base + num_experts,
        reset_signal_base + num_pes,
    ) > signals.numel():
        raise ValueError("expert pool signal ranges exceed the signal tensor")

    send_offsets = nvshmem.zeros(num_experts + 1, dtype=torch.uint32)
    send_rows = nvshmem.zeros(route_capacity, dtype=torch.uint32)
    send_origin_rows = nvshmem.zeros(route_capacity, dtype=torch.uint32)
    packed_source = nvshmem.zeros(
        (route_capacity, hidden_size), dtype=dtype
    )
    expert_input = nvshmem.zeros(
        (experts_per_pe, expert_capacity_rows, hidden_size), dtype=dtype
    )
    expert_output = nvshmem.zeros(
        (experts_per_pe, expert_capacity_rows, hidden_size), dtype=dtype
    )
    return_inbox = nvshmem.zeros(
        (route_capacity, hidden_size), dtype=dtype
    )
    send_batches = nvshmem.zeros(
        (num_experts, EP_BATCH_BYTES), dtype=torch.uint8
    )
    receive_batches = nvshmem.zeros(
        (experts_per_pe, num_pes, EP_BATCH_BYTES), dtype=torch.uint8
    )
    expert_tails = nvshmem.zeros(experts_per_pe, dtype=torch.uint64)
    sequence = nvshmem.zeros(1, dtype=torch.uint64)
    control = nvshmem.zeros(EP_CONTROL_WORDS, dtype=torch.uint64)
    config_tensor = torch.empty(
        EP_CONFIG_BYTES, dtype=torch.uint8, device=signals.device
    )

    buffers = ExpertPoolBuffers(
        signals=signals,
        send_offsets=send_offsets,
        send_rows=send_rows,
        send_origin_rows=send_origin_rows,
        packed_source=packed_source,
        expert_input=expert_input,
        expert_output=expert_output,
        return_inbox=return_inbox,
        send_batches=send_batches,
        receive_batches=receive_batches,
        expert_tails=expert_tails,
        sequence=sequence,
        control=control,
        config_tensor=config_tensor,
        num_pes=num_pes,
        my_pe=my_pe,
        experts_per_pe=experts_per_pe,
        token_capacity=token_capacity,
        route_capacity=route_capacity,
        expert_capacity_rows=expert_capacity_rows,
        dispatch_signal_base=dispatch_signal_base,
        return_signal_base=return_signal_base,
        reset_signal_base=reset_signal_base,
    )
    nvshmem.barrier()
    return buffers


__all__ = [
    "EP_BATCH_BYTES",
    "EP_CONFIG_BYTES",
    "EP_PROFILE_START",
    "EP_PROFILE_DISPATCH_READY",
    "EP_PROFILE_DONE",
    "ExpertPoolStatus",
    "ExpertPoolBatchFlags",
    "ExpertPoolBatch",
    "ExpertPoolConfig",
    "ExpertPoolBuffers",
    "ExpertPoolProgram",
    "group_routes_by_expert",
    "allocate_expert_pool",
    "build_expert_pool_copy_program",
]
