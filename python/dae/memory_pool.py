"""Dependency-aware HBM mailbox protocol for VDCores memory-pool operators.

The runtime ABI lives in ``include/dae/memory_pool.cuh``.  This module owns the
matching 128-byte packers, collective symmetric-buffer allocation, and a small
host reference model used before multi-PE CUDA verification.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from numbers import Integral
from typing import Iterable, Sequence

import torch


REQUEST_BYTES = 128
CONFIG_BYTES = 128
NO_DEPENDENCY = (1 << 32) - 1

_REQUEST_STRUCT = struct.Struct("<8Q12I2Q")
_CONFIG_STRUCT = struct.Struct("<9Q14I")
assert _REQUEST_STRUCT.size == REQUEST_BYTES
assert _CONFIG_STRUCT.size == CONFIG_BYTES


class MemoryPoolOpcode(IntEnum):
    WRITE = 1
    READ = 2
    SCATTER = 3
    GATHER = 4


class MemoryPoolFlags(IntFlag):
    NONE = 0
    REDUCE_SUM_F32 = 1 << 0


class MemoryPoolStatus(IntEnum):
    OK = 0
    BAD_CONFIG = 1
    BAD_OPCODE = 2
    POOL_RANGE = 3
    DEPENDENCY_RANGE = 4
    SIGNAL_RANGE = 5
    ROUTE_RANGE = 6
    SCRATCH_RANGE = 7
    REDUCE_FORMAT = 8
    SEQUENCE = 9


class MemoryPoolDependencyDeadlock(RuntimeError):
    pass


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


def _tensor_address(value: torch.Tensor | int, name: str) -> int:
    if isinstance(value, torch.Tensor):
        if not value.is_cuda:
            raise ValueError(f"{name} tensor must be CUDA")
        if not value.is_contiguous():
            raise ValueError(f"{name} tensor must be contiguous")
        from . import nvshmem

        if not nvshmem.is_symmetric_tensor(value):
            raise ValueError(f"{name} tensor must be in the NVSHMEM symmetric heap")
        value = value.data_ptr()
    return _uint(name, value, 64)


@dataclass(frozen=True)
class MemoryPoolRequest:
    sequence: int
    opcode: MemoryPoolOpcode | int
    source_address: int = 0
    destination_address: int = 0
    route_address: int = 0
    pool_offset: int = 0
    bytes: int = 0
    wait_value: int = 0
    signal_delta: int = 0
    flags: MemoryPoolFlags | int = MemoryPoolFlags.NONE
    source_pe: int = 0
    target_pe: int = 0
    completion_pe: int = 0
    completion_signal: int = 0
    wait_slot: int = NO_DEPENDENCY
    signal_slot: int = NO_DEPENDENCY
    row_bytes: int = 0
    row_count: int = 0
    source_stride: int = 0
    destination_stride: int = 0
    user_tag: int = 0
    reserved: int = 0

    def pack(self) -> bytes:
        sequence = _positive_uint("sequence", self.sequence, 64)
        try:
            opcode = int(MemoryPoolOpcode(self.opcode))
        except ValueError as error:
            raise ValueError(f"unsupported memory-pool opcode {self.opcode}") from error

        values_u64 = (
            sequence,
            _uint("source_address", self.source_address, 64),
            _uint("destination_address", self.destination_address, 64),
            _uint("route_address", self.route_address, 64),
            _uint("pool_offset", self.pool_offset, 64),
            _uint("bytes", self.bytes, 64),
            _uint("wait_value", self.wait_value, 64),
            _uint("signal_delta", self.signal_delta, 64),
        )
        values_u32 = (
            opcode,
            _uint("flags", int(self.flags), 32),
            _uint("source_pe", self.source_pe, 32),
            _uint("target_pe", self.target_pe, 32),
            _uint("completion_pe", self.completion_pe, 32),
            _uint("completion_signal", self.completion_signal, 32),
            _uint("wait_slot", self.wait_slot, 32),
            _uint("signal_slot", self.signal_slot, 32),
            _uint("row_bytes", self.row_bytes, 32),
            _uint("row_count", self.row_count, 32),
            _uint("source_stride", self.source_stride, 32),
            _uint("destination_stride", self.destination_stride, 32),
        )
        return _REQUEST_STRUCT.pack(
            *values_u64,
            *values_u32,
            _uint("user_tag", self.user_tag, 64),
            _uint("reserved", self.reserved, 64),
        )

    @classmethod
    def unpack(cls, payload: bytes | bytearray | memoryview) -> "MemoryPoolRequest":
        if len(payload) != REQUEST_BYTES:
            raise ValueError(f"request payload must be {REQUEST_BYTES} bytes")
        values = _REQUEST_STRUCT.unpack(payload)
        return cls(
            sequence=values[0],
            opcode=MemoryPoolOpcode(values[8]),
            source_address=values[1],
            destination_address=values[2],
            route_address=values[3],
            pool_offset=values[4],
            bytes=values[5],
            wait_value=values[6],
            signal_delta=values[7],
            flags=MemoryPoolFlags(values[9]),
            source_pe=values[10],
            target_pe=values[11],
            completion_pe=values[12],
            completion_signal=values[13],
            wait_slot=values[14],
            signal_slot=values[15],
            row_bytes=values[16],
            row_count=values[17],
            source_stride=values[18],
            destination_stride=values[19],
            user_tag=values[20],
            reserved=values[21],
        )

    @classmethod
    def write(
        cls,
        *,
        sequence: int,
        source: torch.Tensor | int,
        source_pe: int,
        pool_offset: int,
        nbytes: int,
        completion_pe: int,
        completion_signal: int,
        signal_slot: int = NO_DEPENDENCY,
        signal_delta: int = 0,
        reduce_sum_f32: bool = False,
        user_tag: int = 0,
    ) -> "MemoryPoolRequest":
        return cls(
            sequence=sequence,
            opcode=MemoryPoolOpcode.WRITE,
            source_address=_tensor_address(source, "source"),
            source_pe=source_pe,
            pool_offset=pool_offset,
            bytes=nbytes,
            completion_pe=completion_pe,
            completion_signal=completion_signal,
            signal_slot=signal_slot,
            signal_delta=signal_delta,
            flags=(
                MemoryPoolFlags.REDUCE_SUM_F32
                if reduce_sum_f32
                else MemoryPoolFlags.NONE
            ),
            user_tag=user_tag,
        )

    @classmethod
    def read(
        cls,
        *,
        sequence: int,
        destination: torch.Tensor | int,
        target_pe: int,
        pool_offset: int,
        nbytes: int,
        completion_signal: int,
        wait_slot: int = NO_DEPENDENCY,
        wait_value: int = 0,
        user_tag: int = 0,
    ) -> "MemoryPoolRequest":
        return cls(
            sequence=sequence,
            opcode=MemoryPoolOpcode.READ,
            destination_address=_tensor_address(destination, "destination"),
            target_pe=target_pe,
            pool_offset=pool_offset,
            bytes=nbytes,
            completion_pe=target_pe,
            completion_signal=completion_signal,
            wait_slot=wait_slot,
            wait_value=wait_value,
            user_tag=user_tag,
        )

    @classmethod
    def scatter(
        cls,
        *,
        sequence: int,
        source: torch.Tensor | int,
        routes: torch.Tensor | int,
        source_pe: int,
        pool_offset: int,
        row_count: int,
        row_bytes: int,
        completion_pe: int,
        completion_signal: int,
        source_stride: int = 0,
        pool_stride: int = 0,
        signal_slot: int = NO_DEPENDENCY,
        signal_delta: int = 0,
        user_tag: int = 0,
    ) -> "MemoryPoolRequest":
        return cls(
            sequence=sequence,
            opcode=MemoryPoolOpcode.SCATTER,
            source_address=_tensor_address(source, "source"),
            route_address=_tensor_address(routes, "routes"),
            source_pe=source_pe,
            completion_pe=completion_pe,
            completion_signal=completion_signal,
            pool_offset=pool_offset,
            row_count=row_count,
            row_bytes=row_bytes,
            source_stride=source_stride,
            destination_stride=pool_stride,
            signal_slot=signal_slot,
            signal_delta=signal_delta,
            user_tag=user_tag,
        )

    @classmethod
    def gather(
        cls,
        *,
        sequence: int,
        destination: torch.Tensor | int,
        routes: torch.Tensor | int,
        target_pe: int,
        pool_offset: int,
        row_count: int,
        row_bytes: int,
        completion_signal: int,
        pool_stride: int = 0,
        destination_stride: int = 0,
        wait_slot: int = NO_DEPENDENCY,
        wait_value: int = 0,
        user_tag: int = 0,
    ) -> "MemoryPoolRequest":
        return cls(
            sequence=sequence,
            opcode=MemoryPoolOpcode.GATHER,
            destination_address=_tensor_address(destination, "destination"),
            route_address=_tensor_address(routes, "routes"),
            target_pe=target_pe,
            completion_pe=target_pe,
            completion_signal=completion_signal,
            pool_offset=pool_offset,
            row_count=row_count,
            row_bytes=row_bytes,
            source_stride=pool_stride,
            destination_stride=destination_stride,
            wait_slot=wait_slot,
            wait_value=wait_value,
            user_tag=user_tag,
        )


@dataclass(frozen=True)
class MemoryPoolConfig:
    mailboxes_address: int
    pool_data_address: int
    data_scratch_address: int
    route_scratch_address: int
    dependencies_address: int
    consumed_sequences_address: int
    control_address: int
    pool_bytes: int
    data_scratch_bytes: int
    mailbox_count: int
    dependency_count: int
    submit_signal_base: int
    signal_count: int
    route_capacity: int
    flags: int = 0

    def pack(self) -> bytes:
        values_u64 = (
            _uint("mailboxes_address", self.mailboxes_address, 64),
            _uint("pool_data_address", self.pool_data_address, 64),
            _uint("data_scratch_address", self.data_scratch_address, 64),
            _uint("route_scratch_address", self.route_scratch_address, 64),
            _uint("dependencies_address", self.dependencies_address, 64),
            _uint("consumed_sequences_address", self.consumed_sequences_address, 64),
            _uint("control_address", self.control_address, 64),
            _uint("pool_bytes", self.pool_bytes, 64),
            _uint("data_scratch_bytes", self.data_scratch_bytes, 64),
        )
        values_u32 = (
            _positive_uint("mailbox_count", self.mailbox_count, 32),
            _positive_uint("dependency_count", self.dependency_count, 32),
            _uint("submit_signal_base", self.submit_signal_base, 32),
            _positive_uint("signal_count", self.signal_count, 32),
            _positive_uint("route_capacity", self.route_capacity, 32),
            _uint("flags", self.flags, 32),
        )
        if self.submit_signal_base + self.mailbox_count > self.signal_count:
            raise ValueError("submit signal range exceeds signal_count")
        return _CONFIG_STRUCT.pack(*values_u64, *values_u32, *([0] * 8))


def _copy_packed(destination: torch.Tensor, payload: bytes) -> None:
    if destination.dtype != torch.uint8 or destination.numel() != len(payload):
        raise ValueError("packed destination has the wrong dtype or size")
    source = torch.tensor(list(payload), dtype=torch.uint8)
    destination.view(-1).copy_(source, non_blocking=False)


@dataclass
class MemoryPoolBuffers:
    signals: torch.Tensor
    mailboxes: torch.Tensor
    pool_data: torch.Tensor
    data_scratch: torch.Tensor
    routes: torch.Tensor
    route_scratch: torch.Tensor
    dependencies: torch.Tensor
    consumed_sequences: torch.Tensor
    control: torch.Tensor
    config_tensor: torch.Tensor
    submit_signal_base: int
    completion_signal_base: int

    @property
    def mailbox_count(self) -> int:
        return self.mailboxes.shape[0]

    @property
    def route_capacity(self) -> int:
        return self.routes.shape[1]

    def _mailbox_index(self, mailbox: int) -> int:
        mailbox = _uint("mailbox", mailbox, 32)
        if mailbox >= self.mailbox_count:
            raise ValueError(f"mailbox {mailbox} is outside [0, {self.mailbox_count})")
        return mailbox

    def request_tensor(self, mailbox: int) -> torch.Tensor:
        return self.mailboxes[self._mailbox_index(mailbox)]

    def request_address(self, mailbox: int) -> int:
        return self.request_tensor(mailbox).data_ptr()

    def route_tensor(self, mailbox: int) -> torch.Tensor:
        return self.routes[self._mailbox_index(mailbox)]

    def route_address(self, mailbox: int) -> int:
        return self.route_tensor(mailbox).data_ptr()

    def submit_signal(self, mailbox: int) -> int:
        return self.submit_signal_base + self._mailbox_index(mailbox)

    def completion_signal(self, mailbox: int) -> int:
        signal = self.completion_signal_base + self._mailbox_index(mailbox)
        if signal >= self.signals.numel():
            raise ValueError("completion signal range exceeds signal tensor")
        return signal

    def write_routes(self, mailbox: int, routes: Sequence[int] | torch.Tensor) -> torch.Tensor:
        destination = self.route_tensor(mailbox)
        source = torch.as_tensor(routes, dtype=torch.uint32, device="cpu").view(-1)
        if source.numel() > destination.numel():
            raise ValueError("route table exceeds configured route_capacity")
        destination.zero_()
        if source.numel():
            destination[: source.numel()].copy_(source, non_blocking=False)
        return destination

    def write_request(self, mailbox: int, request: MemoryPoolRequest) -> torch.Tensor:
        if request.completion_signal >= self.signals.numel():
            raise ValueError("request completion signal exceeds signal tensor")
        destination = self.request_tensor(mailbox)
        _copy_packed(destination, request.pack())
        return destination

    def read_request(self, mailbox: int) -> MemoryPoolRequest:
        payload = bytes(self.request_tensor(mailbox).cpu().tolist())
        return MemoryPoolRequest.unpack(payload)

    def config(self) -> MemoryPoolConfig:
        return MemoryPoolConfig(
            mailboxes_address=self.mailboxes.data_ptr(),
            pool_data_address=self.pool_data.data_ptr(),
            data_scratch_address=self.data_scratch.data_ptr(),
            route_scratch_address=self.route_scratch.data_ptr(),
            dependencies_address=self.dependencies.data_ptr(),
            consumed_sequences_address=self.consumed_sequences.data_ptr(),
            control_address=self.control.data_ptr(),
            pool_bytes=self.pool_data.numel(),
            data_scratch_bytes=self.data_scratch.numel(),
            mailbox_count=self.mailbox_count,
            dependency_count=self.dependencies.numel(),
            submit_signal_base=self.submit_signal_base,
            signal_count=self.signals.numel(),
            route_capacity=self.route_capacity,
        )

    def refresh_config(self) -> torch.Tensor:
        _copy_packed(self.config_tensor, self.config().pack())
        return self.config_tensor

    def control_state(self) -> tuple[MemoryPoolStatus, int, int, int]:
        values = self.control.cpu().tolist()
        return MemoryPoolStatus(values[0]), values[1], values[2], values[3]


def allocate_memory_pool(
    signals: torch.Tensor,
    *,
    mailbox_count: int,
    pool_bytes: int,
    data_scratch_bytes: int,
    route_capacity: int,
    dependency_count: int,
    submit_signal_base: int = 0,
    completion_signal_base: int | None = None,
) -> MemoryPoolBuffers:
    """Collectively allocate the symmetric buffers used by one pool protocol."""

    from . import nvshmem

    if not isinstance(signals, torch.Tensor) or not signals.is_cuda:
        raise ValueError("signals must be a CUDA tensor")
    if signals.dtype != torch.uint64 or signals.ndim != 1 or not signals.is_contiguous():
        raise ValueError("signals must be a contiguous rank-1 uint64 tensor")
    if not nvshmem.is_symmetric_tensor(signals):
        raise ValueError("signals must be allocated in the NVSHMEM symmetric heap")

    mailbox_count = _positive_uint("mailbox_count", mailbox_count, 32)
    pool_bytes = _positive_uint("pool_bytes", pool_bytes, 64)
    data_scratch_bytes = _positive_uint("data_scratch_bytes", data_scratch_bytes, 64)
    route_capacity = _positive_uint("route_capacity", route_capacity, 32)
    dependency_count = _positive_uint("dependency_count", dependency_count, 32)
    submit_signal_base = _uint("submit_signal_base", submit_signal_base, 32)
    if completion_signal_base is None:
        completion_signal_base = submit_signal_base + mailbox_count
    completion_signal_base = _uint(
        "completion_signal_base", completion_signal_base, 32
    )
    required_signals = max(
        submit_signal_base + mailbox_count,
        completion_signal_base + mailbox_count,
    )
    if required_signals > signals.numel():
        raise ValueError(
            f"memory pool requires {required_signals} signals, got {signals.numel()}"
        )

    mailboxes = nvshmem.zeros(
        (mailbox_count, REQUEST_BYTES), dtype=torch.uint8, device=signals.device
    )
    pool_data = nvshmem.zeros(pool_bytes, dtype=torch.uint8, device=signals.device)
    data_scratch = nvshmem.zeros(
        data_scratch_bytes, dtype=torch.uint8, device=signals.device
    )
    routes = nvshmem.zeros(
        (mailbox_count, route_capacity), dtype=torch.uint32, device=signals.device
    )
    route_scratch = nvshmem.zeros(
        route_capacity, dtype=torch.uint32, device=signals.device
    )
    dependencies = nvshmem.zeros(
        dependency_count, dtype=torch.uint64, device=signals.device
    )
    consumed_sequences = nvshmem.zeros(
        mailbox_count, dtype=torch.uint64, device=signals.device
    )
    control = nvshmem.zeros(4, dtype=torch.uint64, device=signals.device)
    config_tensor = torch.empty(CONFIG_BYTES, dtype=torch.uint8, device=signals.device)

    buffers = MemoryPoolBuffers(
        signals=signals,
        mailboxes=mailboxes,
        pool_data=pool_data,
        data_scratch=data_scratch,
        routes=routes,
        route_scratch=route_scratch,
        dependencies=dependencies,
        consumed_sequences=consumed_sequences,
        control=control,
        config_tensor=config_tensor,
        submit_signal_base=submit_signal_base,
        completion_signal_base=completion_signal_base,
    )
    buffers.refresh_config()
    nvshmem.barrier()
    return buffers


def make_phase_schedule(
    buffers: MemoryPoolBuffers,
    active_mailboxes: Sequence[int],
    *,
    current_pe: int,
    pool_pe: int,
    expected_requests: int,
    pool_sm: int = 0,
    producer_barriers: dict[int, int] | None = None,
):
    """Build an SM-specific memory stream for one submit/wait pool phase.

    The pool core owns ``pool_sm``. Local producer mailboxes are mapped to the
    following SMs so pool execution never shares an alloc warp with a submit.
    """

    from .instructions import (
        IssueBarrier,
        MemoryPoolRun,
        MemoryPoolSubmit,
        MemoryPoolWait,
    )

    current_pe = _uint("current_pe", current_pe, 16)
    pool_pe = _uint("pool_pe", pool_pe, 16)
    expected_requests = _uint("expected_requests", expected_requests, 32)
    pool_sm = _uint("pool_sm", pool_sm, 16)
    mailboxes = tuple(buffers._mailbox_index(index) for index in active_mailboxes)
    if len(set(mailboxes)) != len(mailboxes):
        raise ValueError("active_mailboxes must be unique")
    producer_barriers = dict(producer_barriers or {})
    unknown_barriers = set(producer_barriers) - set(mailboxes)
    if unknown_barriers:
        raise ValueError(
            f"producer barriers name inactive mailboxes: {sorted(unknown_barriers)}"
        )
    producer_sms = tuple(pool_sm + 1 + index for index in range(len(mailboxes)))

    def schedule(sm: int):
        instructions = []
        if current_pe == pool_pe and sm == pool_sm:
            instructions.append(
                MemoryPoolRun(buffers.refresh_config(), expected_requests)
            )
        for producer_sm, mailbox in zip(producer_sms, mailboxes):
            if sm == producer_sm:
                request = buffers.request_tensor(mailbox)
                if mailbox in producer_barriers:
                    instructions.append(IssueBarrier(producer_barriers[mailbox]))
                instructions.extend(
                    (
                        MemoryPoolSubmit(
                            request,
                            pool_pe=pool_pe,
                            submit_signal=buffers.submit_signal(mailbox),
                        ),
                        MemoryPoolWait(request),
                    )
                )
        return instructions

    schedule.num_sms = pool_sm + 1 + len(mailboxes)
    return schedule


def resolve_dependency_order(
    requests: Iterable[MemoryPoolRequest],
    *,
    initial_dependencies: Sequence[int] | None = None,
) -> tuple[list[int], list[int]]:
    """Reference the pool's ready-request selection and dependency updates."""

    request_list = list(requests)
    max_slot = -1
    for request in request_list:
        for slot in (request.wait_slot, request.signal_slot):
            if slot != NO_DEPENDENCY:
                max_slot = max(max_slot, slot)
    dependencies = list(initial_dependencies or [])
    if len(dependencies) <= max_slot:
        dependencies.extend([0] * (max_slot + 1 - len(dependencies)))

    pending = list(range(len(request_list)))
    order: list[int] = []
    while pending:
        ready_position = None
        for position, index in enumerate(pending):
            request = request_list[index]
            if (
                request.wait_slot == NO_DEPENDENCY
                or dependencies[request.wait_slot] >= request.wait_value
            ):
                ready_position = position
                break
        if ready_position is None:
            tags = [request_list[index].user_tag for index in pending]
            raise MemoryPoolDependencyDeadlock(
                f"no pending request is ready; pending user tags={tags}"
            )

        index = pending.pop(ready_position)
        request = request_list[index]
        order.append(index)
        if request.signal_slot != NO_DEPENDENCY:
            dependencies[request.signal_slot] += request.signal_delta
    return order, dependencies


def reference_scatter_rows(
    source: torch.Tensor,
    routes: Sequence[int] | torch.Tensor,
    pool: torch.Tensor,
) -> torch.Tensor:
    routes_tensor = torch.as_tensor(routes, dtype=torch.long, device=source.device)
    if source.ndim != 2 or pool.ndim != 2:
        raise ValueError("source and pool must be rank-2 row tensors")
    if routes_tensor.numel() != source.shape[0]:
        raise ValueError("one route is required for every source row")
    if source.shape[1] != pool.shape[1]:
        raise ValueError("source and pool row widths must match")
    pool.index_copy_(0, routes_tensor, source)
    return pool


def reference_gather_rows(
    pool: torch.Tensor,
    routes: Sequence[int] | torch.Tensor,
) -> torch.Tensor:
    routes_tensor = torch.as_tensor(routes, dtype=torch.long, device=pool.device)
    if pool.ndim != 2:
        raise ValueError("pool must be a rank-2 row tensor")
    return pool.index_select(0, routes_tensor)


__all__ = [
    "REQUEST_BYTES",
    "CONFIG_BYTES",
    "NO_DEPENDENCY",
    "MemoryPoolOpcode",
    "MemoryPoolFlags",
    "MemoryPoolStatus",
    "MemoryPoolDependencyDeadlock",
    "MemoryPoolRequest",
    "MemoryPoolConfig",
    "MemoryPoolBuffers",
    "allocate_memory_pool",
    "make_phase_schedule",
    "resolve_dependency_order",
    "reference_scatter_rows",
    "reference_gather_rows",
]
