"""Benchmark a Grace-hosted, non-contiguous HBM SGL data path.

This is an isolated transport experiment.  It does not launch a VDCores
operator or replace PoolInst metadata handling.  Each PE registers one
NVSHMEM HBM arena, builds an RC QP to its peer, and compares:

* ``sgl``: several non-contiguous source rows per RDMA WRITE WR;
* ``row-wr``: one RDMA WRITE WR per source row.

Both paths submit one WR chain per batch with ``ibv_post_send`` and finish it
with the same ordered inline readiness write. Multiple batches may remain in
flight on one credit-tracked RC QP before the host polls their ordered
completions. The receiver queries native GPUDirect ordering and uses a host
flush only on platforms that require it.

Build and run on two Vista GH compute nodes::

    make -C benchmarks/host_sgl
    NVSHMEM_DISABLE_NCCL=1 ibrun -n 2 \
      python benchmarks/host_sgl_benchmark.py
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
from pathlib import Path
import statistics
import time
from typing import Callable, TypeVar

import torch
from cuda.bindings import driver as cuda
from mpi4py import MPI

import dae.nvshmem as nvshmem

from host_sgl_probe import (
    ExportedRange,
    VerbsRegistration,
    _cuda_value,
    _export_tensor,
)


_ERROR_BYTES = 512
_T = TypeVar("_T")


class HostSglEndpoint(ctypes.Structure):
    _fields_ = (
        ("qp_num", ctypes.c_uint32),
        ("psn", ctypes.c_uint32),
        ("lid", ctypes.c_uint16),
        ("port_num", ctypes.c_uint8),
        ("gid_index", ctypes.c_uint8),
        ("active_mtu", ctypes.c_uint8),
        ("link_layer", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 2),
        ("gid", ctypes.c_uint8 * 16),
    )


class HostSglRequest(ctypes.Structure):
    _fields_ = (
        ("local_lkey", ctypes.c_uint32),
        ("remote_rkey", ctypes.c_uint32),
        ("source_base", ctypes.c_uint64),
        ("source_stride", ctypes.c_uint64),
        ("row_bytes", ctypes.c_uint32),
        ("row_count", ctypes.c_uint32),
        ("row_indices", ctypes.POINTER(ctypes.c_uint32)),
        ("remote_data", ctypes.c_uint64),
        ("remote_signal", ctypes.c_uint64),
        ("sequence", ctypes.c_uint64),
    )


if ctypes.sizeof(HostSglEndpoint) != 32:
    raise RuntimeError("HostSglEndpoint ctypes ABI must be 32 bytes")
if ctypes.sizeof(HostSglRequest) != 64:
    raise RuntimeError("HostSglRequest ctypes ABI must be 64 bytes")


def _pointer_value(pointer: object) -> int:
    value = getattr(pointer, "value", pointer)
    if value is None:
        return 0
    return int(value)


class HostSglQueue:
    def __init__(
        self,
        library_path: Path,
        *,
        context: object,
        pd: object,
        port: int,
        gid_index: int,
        requested_send_wr: int,
        requested_send_sge: int,
    ) -> None:
        if not library_path.is_file():
            raise FileNotFoundError(
                f"{library_path} is missing; run make -C benchmarks/host_sgl"
            )
        self._library = ctypes.CDLL(str(library_path), use_errno=True)
        self._configure_abi()
        if self._library.host_sgl_abi_version() != 4:
            raise RuntimeError("unsupported host SGL helper ABI")
        error = ctypes.create_string_buffer(_ERROR_BYTES)
        self._handle = self._library.host_sgl_create_qp(
            ctypes.c_void_p(_pointer_value(context)),
            ctypes.c_void_p(_pointer_value(pd)),
            int(port),
            int(gid_index),
            int(requested_send_wr),
            int(requested_send_sge),
            error,
            len(error),
        )
        if not self._handle:
            raise RuntimeError(f"host_sgl_create_qp: {error.value.decode()}")

    def _configure_abi(self) -> None:
        library = self._library
        error_pointer = ctypes.POINTER(ctypes.c_char)
        library.host_sgl_abi_version.argtypes = []
        library.host_sgl_abi_version.restype = ctypes.c_uint32
        library.host_sgl_create_qp.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint8,
            ctypes.c_uint8,
            ctypes.c_uint32,
            ctypes.c_uint32,
            error_pointer,
            ctypes.c_size_t,
        )
        library.host_sgl_create_qp.restype = ctypes.c_void_p
        library.host_sgl_get_endpoint.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(HostSglEndpoint),
            error_pointer,
            ctypes.c_size_t,
        )
        library.host_sgl_get_endpoint.restype = ctypes.c_int
        library.host_sgl_connect.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(HostSglEndpoint),
            error_pointer,
            ctypes.c_size_t,
        )
        library.host_sgl_connect.restype = ctypes.c_int
        library.host_sgl_max_send_wr.argtypes = (ctypes.c_void_p,)
        library.host_sgl_max_send_wr.restype = ctypes.c_uint32
        library.host_sgl_max_sge.argtypes = (ctypes.c_void_p,)
        library.host_sgl_max_sge.restype = ctypes.c_uint32
        library.host_sgl_outstanding_batches.argtypes = (ctypes.c_void_p,)
        library.host_sgl_outstanding_batches.restype = ctypes.c_uint32
        library.host_sgl_outstanding_wrs.argtypes = (ctypes.c_void_p,)
        library.host_sgl_outstanding_wrs.restype = ctypes.c_uint32
        library.host_sgl_post_indexed_batch.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(HostSglRequest),
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint32),
            error_pointer,
            ctypes.c_size_t,
        )
        library.host_sgl_post_indexed_batch.restype = ctypes.c_int
        library.host_sgl_try_poll.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            error_pointer,
            ctypes.c_size_t,
        )
        library.host_sgl_try_poll.restype = ctypes.c_int
        library.host_sgl_poll.argtypes = (
            ctypes.c_void_p,
            ctypes.c_uint64,
            error_pointer,
            ctypes.c_size_t,
        )
        library.host_sgl_poll.restype = ctypes.c_int
        library.host_sgl_create_ring.argtypes = (
            error_pointer,
            ctypes.c_size_t,
        )
        library.host_sgl_create_ring.restype = ctypes.c_void_p
        library.host_sgl_ring_memory.argtypes = (ctypes.c_void_p,)
        library.host_sgl_ring_memory.restype = ctypes.c_void_p
        library.host_sgl_consume_ring.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            error_pointer,
            ctypes.c_size_t,
        )
        library.host_sgl_consume_ring.restype = ctypes.c_int
        library.host_sgl_publish_ring_cuda.argtypes = (
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
            error_pointer,
            ctypes.c_size_t,
        )
        library.host_sgl_publish_ring_cuda.restype = ctypes.c_int
        library.host_sgl_publish_ring_resident_cuda.argtypes = (
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
            error_pointer,
            ctypes.c_size_t,
        )
        library.host_sgl_publish_ring_resident_cuda.restype = ctypes.c_int
        library.host_sgl_destroy_ring.argtypes = (ctypes.c_void_p,)
        library.host_sgl_destroy_ring.restype = None
        library.host_sgl_destroy_qp.argtypes = (ctypes.c_void_p,)
        library.host_sgl_destroy_qp.restype = None

    @staticmethod
    def _check(result: int, operation: str, error: ctypes.Array) -> None:
        if result != 0:
            detail = error.value.decode() or f"error code {result}"
            raise RuntimeError(f"{operation}: {detail}")

    @property
    def max_send_wr(self) -> int:
        return int(self._library.host_sgl_max_send_wr(self._handle))

    @property
    def max_sge(self) -> int:
        return int(self._library.host_sgl_max_sge(self._handle))

    @property
    def outstanding_batches(self) -> int:
        return int(self._library.host_sgl_outstanding_batches(self._handle))

    @property
    def outstanding_wrs(self) -> int:
        return int(self._library.host_sgl_outstanding_wrs(self._handle))

    def endpoint(self) -> HostSglEndpoint:
        endpoint = HostSglEndpoint()
        error = ctypes.create_string_buffer(_ERROR_BYTES)
        self._check(
            self._library.host_sgl_get_endpoint(
                self._handle, ctypes.byref(endpoint), error, len(error)
            ),
            "host_sgl_get_endpoint",
            error,
        )
        return endpoint

    def connect(self, remote: HostSglEndpoint) -> None:
        error = ctypes.create_string_buffer(_ERROR_BYTES)
        self._check(
            self._library.host_sgl_connect(
                self._handle, ctypes.byref(remote), error, len(error)
            ),
            "host_sgl_connect",
            error,
        )

    def post_indexed_batch(
        self,
        requests: ctypes.Array,
        *,
        request_count: int,
        row_wr_mode: bool,
    ) -> int:
        posted_data_wrs = ctypes.c_uint32()
        error = ctypes.create_string_buffer(_ERROR_BYTES)
        self._check(
            self._library.host_sgl_post_indexed_batch(
                self._handle,
                requests,
                int(request_count),
                int(row_wr_mode),
                ctypes.byref(posted_data_wrs),
                error,
                len(error),
            ),
            "host_sgl_post_indexed_batch",
            error,
        )
        return int(posted_data_wrs.value)

    def poll(self, sequence: int) -> None:
        error = ctypes.create_string_buffer(_ERROR_BYTES)
        self._check(
            self._library.host_sgl_poll(
                self._handle, int(sequence), error, len(error)
            ),
            "host_sgl_poll",
            error,
        )

    def try_poll(self) -> int | None:
        completed = ctypes.c_uint64()
        error = ctypes.create_string_buffer(_ERROR_BYTES)
        result = self._library.host_sgl_try_poll(
            self._handle, ctypes.byref(completed), error, len(error)
        )
        if result < 0:
            detail = error.value.decode() or f"error code {result}"
            raise RuntimeError(f"host_sgl_try_poll: {detail}")
        return int(completed.value) if result else None

    def create_coherent_ring(self) -> "HostSglCoherentRing":
        error = ctypes.create_string_buffer(_ERROR_BYTES)
        handle = self._library.host_sgl_create_ring(error, len(error))
        if not handle:
            raise RuntimeError(
                f"host_sgl_create_ring: {error.value.decode()}"
            )
        memory = self._library.host_sgl_ring_memory(handle)
        if not memory:
            self._library.host_sgl_destroy_ring(handle)
            raise RuntimeError("host_sgl_ring_memory returned null")
        return HostSglCoherentRing(
            self._library, handle=handle, memory=memory
        )

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._library.host_sgl_destroy_qp(self._handle)
            self._handle = None


class HostSglCoherentRing:
    def __init__(self, library: ctypes.CDLL, *, handle: int, memory: int) -> None:
        self._library = library
        self._handle = handle
        self._memory = memory

    def publish(
        self,
        row_indices: torch.Tensor,
        *,
        first_generation: int,
        local_lkey: int,
        remote_rkey: int,
        source_base: int,
        source_stride: int,
        row_bytes: int,
        remote_data_base: int,
        remote_data_stride: int,
        remote_signal_base: int,
        remote_signal_stride: int,
        stream: torch.cuda.Stream,
    ) -> None:
        if row_indices.dtype != torch.uint32 or row_indices.ndim != 2:
            raise ValueError("ring row indices must be a 2D uint32 tensor")
        if not row_indices.is_cuda or not row_indices.is_contiguous():
            raise ValueError("ring row indices must be contiguous CUDA memory")
        message_count, row_count = row_indices.shape
        error = ctypes.create_string_buffer(_ERROR_BYTES)
        result = self._library.host_sgl_publish_ring_cuda(
            self._memory,
            int(first_generation),
            int(message_count),
            ctypes.cast(
                ctypes.c_void_p(row_indices.data_ptr()),
                ctypes.POINTER(ctypes.c_uint32),
            ),
            int(row_count),
            int(local_lkey),
            int(remote_rkey),
            int(source_base),
            int(source_stride),
            int(row_bytes),
            int(remote_data_base),
            int(remote_data_stride),
            int(remote_signal_base),
            int(remote_signal_stride),
            ctypes.c_void_p(int(stream.cuda_stream)),
            error,
            len(error),
        )
        if result != 0:
            detail = error.value.decode() or f"error code {result}"
            raise RuntimeError(f"host_sgl_publish_ring_cuda: {detail}")

    def consume(
        self,
        queue: HostSglQueue,
        *,
        first_generation: int,
        request_count: int,
    ) -> int:
        data_wrs = ctypes.c_uint32()
        error = ctypes.create_string_buffer(_ERROR_BYTES)
        result = self._library.host_sgl_consume_ring(
            queue._handle,
            self._handle,
            int(first_generation),
            int(request_count),
            ctypes.byref(data_wrs),
            error,
            len(error),
        )
        if result != 0:
            detail = error.value.decode() or f"error code {result}"
            raise RuntimeError(f"host_sgl_consume_ring: {detail}")
        return int(data_wrs.value)

    def publish_resident(
        self,
        row_indices: torch.Tensor,
        *,
        first_generation: int,
        round_count: int,
        local_lkey: int,
        remote_rkey: int,
        source_base: int,
        source_stride: int,
        row_bytes: int,
        remote_data_base: int,
        remote_data_stride: int,
        remote_signal_base: int,
        remote_signal_stride: int,
        stream: torch.cuda.Stream,
    ) -> None:
        if row_indices.dtype != torch.uint32 or row_indices.ndim != 2:
            raise ValueError("ring row indices must be a 2D uint32 tensor")
        if not row_indices.is_cuda or not row_indices.is_contiguous():
            raise ValueError("ring row indices must be contiguous CUDA memory")
        if round_count <= 0:
            raise ValueError("resident ring round_count must be positive")
        message_count, row_count = row_indices.shape
        error = ctypes.create_string_buffer(_ERROR_BYTES)
        result = self._library.host_sgl_publish_ring_resident_cuda(
            self._memory,
            int(first_generation),
            int(round_count),
            int(message_count),
            ctypes.cast(
                ctypes.c_void_p(row_indices.data_ptr()),
                ctypes.POINTER(ctypes.c_uint32),
            ),
            int(row_count),
            int(local_lkey),
            int(remote_rkey),
            int(source_base),
            int(source_stride),
            int(row_bytes),
            int(remote_data_base),
            int(remote_data_stride),
            int(remote_signal_base),
            int(remote_signal_stride),
            ctypes.c_void_p(int(stream.cuda_stream)),
            error,
            len(error),
        )
        if result != 0:
            detail = error.value.decode() or f"error code {result}"
            raise RuntimeError(
                f"host_sgl_publish_ring_resident_cuda: {detail}"
            )

    def close(self) -> None:
        if self._handle is not None:
            self._library.host_sgl_destroy_ring(self._handle)
            self._handle = None
            self._memory = None


def _collective_call(comm: MPI.Comm, stage: str, function: Callable[[], _T]) -> _T:
    value = None
    local_error = None
    try:
        value = function()
    except Exception as error:  # keep every MPI rank in subsequent cleanup
        local_error = f"{type(error).__name__}: {error}"
    errors = comm.allgather(local_error)
    if any(error is not None for error in errors):
        raise RuntimeError(f"{stage} failed across PEs: {errors}")
    return value  # type: ignore[return-value]


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _ordered_row_indices(rows: int) -> list[int]:
    # The even/odd order proves that SGL concatenation follows the compact
    # route list, not source-address order.  The gap in source_stride makes
    # every neighboring SGE physically non-contiguous as well.
    return [*range(0, rows, 2), *range(1, rows, 2)]


def _gpudirect_write_ordering() -> int:
    device = _cuda_value(cuda.cuCtxGetDevice(), "cuCtxGetDevice")
    return int(
        _cuda_value(
            cuda.cuDeviceGetAttribute(
                cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING,
                device,
            ),
            "CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING",
        )
    )


def _make_inbound_rdma_visible(native_ordering: int) -> None:
    owner_ordering = int(
        cuda.CUGPUDirectRDMAWritesOrdering.CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER
    )
    if native_ordering >= owner_ordering:
        return
    _cuda_value(
        cuda.cuFlushGPUDirectRDMAWrites(
            cuda.CUflushGPUDirectRDMAWritesTarget.CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX,
            cuda.CUflushGPUDirectRDMAWritesScope.CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER,
        ),
        "cuFlushGPUDirectRDMAWrites",
    )


def _register_storage(
    storage: torch.Tensor, args: argparse.Namespace
) -> tuple[ExportedRange | None, VerbsRegistration, str | None]:
    exported = None
    export_error = None
    if args.registration != "legacy":
        try:
            exported = _export_tensor(storage, pcie_bar1=args.pcie_bar1)
        except Exception as error:
            export_error = f"{type(error).__name__}: {error}"
            if args.registration == "dmabuf":
                raise
    try:
        registration = VerbsRegistration(
            address=storage.data_ptr(),
            length=storage.numel() * storage.element_size(),
            exported=exported,
            device_name=args.ib_device,
            registration_mode=args.registration,
        )
    except Exception:
        if exported is not None:
            os.close(exported.fd)
        raise
    return exported, registration, export_error


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument(
        "--row-bytes",
        type=int,
        default=7168 * 2,
        help="activation bytes per routed token (default: 7168 BF16 values)",
    )
    parser.add_argument(
        "--gap-bytes",
        type=int,
        default=1024,
        help="unused bytes between source rows; ensures non-contiguous SGEs",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--batch-depth",
        type=int,
        default=1,
        help="independent data messages submitted by one ibv_post_send",
    )
    parser.add_argument(
        "--pipeline-batches",
        type=int,
        default=1,
        help="ibv_post_send batches kept in flight before polling completions",
    )
    parser.add_argument(
        "--mode", choices=("both", "sgl", "row-wr"), default="both"
    )
    parser.add_argument(
        "--submission",
        choices=("both", "direct", "ring", "resident"),
        default="both",
        help=(
            "CPU-direct descriptors, launch-published coherent ring, "
            "resident-CTA coherent ring, or direct plus launch ring"
        ),
    )
    parser.add_argument("--port", type=int, default=1)
    parser.add_argument("--gid-index", type=int, default=0)
    parser.add_argument("--requested-sge", type=int, default=64)
    parser.add_argument("--symmetric-size", default="128M")
    parser.add_argument("--ib-device")
    parser.add_argument(
        "--registration", choices=("auto", "dmabuf", "legacy"), default="auto"
    )
    parser.add_argument("--pcie-bar1", action="store_true")
    parser.add_argument(
        "--library",
        type=Path,
        default=Path(__file__).with_name("host_sgl") / "libhost_sgl_verbs.so",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rows <= 1:
        raise ValueError("--rows must be greater than one")
    if args.row_bytes <= 0 or args.gap_bytes <= 0:
        raise ValueError("--row-bytes and --gap-bytes must be positive")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("--warmup must be nonnegative and --iterations positive")
    if args.batch_depth <= 0 or args.pipeline_batches <= 0:
        raise ValueError("--batch-depth and --pipeline-batches must be positive")
    if args.requested_sge <= 0:
        raise ValueError("--requested-sge must be positive")

    comm = MPI.COMM_WORLD
    runtime = nvshmem.init(symmetric_size=args.symmetric_size)
    exported = None
    registration = None
    queue = None
    ring = None
    try:
        if runtime.num_pes != 2 or comm.Get_size() != 2:
            raise RuntimeError("the isolated host SGL benchmark currently requires 2 PEs")
        if runtime.rank != comm.Get_rank():
            raise RuntimeError("NVSHMEM and MPI rank order differ")
        peer = 1 - runtime.rank
        native_rdma_ordering = _gpudirect_write_ordering()

        source_stride = args.row_bytes + args.gap_bytes
        source_bytes = args.rows * source_stride
        destination_offset = _align_up(source_bytes, 4096)
        message_bytes = args.rows * args.row_bytes
        messages_per_round = args.batch_depth * args.pipeline_batches
        if args.submission in {"both", "ring", "resident"}:
            if args.rows > 32 or messages_per_round > 32:
                raise ValueError(
                    "the inference-specialized ring supports <=32 rows/message "
                    "and <=32 messages/epoch"
                )
        destination_bytes = messages_per_round * message_bytes
        signal_offset = _align_up(destination_offset + destination_bytes, 8)
        arena_bytes = _align_up(signal_offset + messages_per_round * 8, 4096)
        storage = nvshmem.zeros(arena_bytes, dtype=torch.uint8)

        source = torch.as_strided(
            storage,
            size=(args.rows, args.row_bytes),
            stride=(source_stride, 1),
        )
        columns = torch.arange(
            args.row_bytes, dtype=torch.int64, device=storage.device
        ).unsqueeze(0)
        local_rows = torch.arange(
            args.rows, dtype=torch.int64, device=storage.device
        ).unsqueeze(1)
        source.copy_(
            ((runtime.rank * 37 + local_rows * 17 + columns) % 251).to(
                torch.uint8
            )
        )
        destination = storage[
            destination_offset : destination_offset + destination_bytes
        ].view(
            args.pipeline_batches,
            args.batch_depth,
            args.rows,
            args.row_bytes,
        )
        signal = storage[
            signal_offset : signal_offset + messages_per_round * 8
        ].view(torch.uint64).view(args.pipeline_batches, args.batch_depth)
        base_row_indices = _ordered_row_indices(args.rows)
        route_lists = [
            base_row_indices[rotation % args.rows :]
            + base_row_indices[: rotation % args.rows]
            for rotation in range(args.batch_depth)
        ]
        ring_route_indices = None
        if args.submission in {"both", "ring", "resident"}:
            ring_route_indices = torch.tensor(
                [
                    route_lists[message % args.batch_depth]
                    for message in range(messages_per_round)
                ],
                dtype=torch.uint32,
                device=storage.device,
            )
        row_index_arrays = [
            (ctypes.c_uint32 * args.rows)(*row_indices)
            for row_indices in route_lists
        ]
        expected_batch = torch.stack(
            [
                (
                    (
                        peer * 37
                        + torch.tensor(
                            row_indices, dtype=torch.int64, device=storage.device
                        ).unsqueeze(1)
                        * 17
                        + columns
                    )
                    % 251
                ).to(torch.uint8)
                for row_indices in route_lists
            ]
        )
        expected = expected_batch.unsqueeze(0).expand(
            args.pipeline_batches, -1, -1, -1
        )
        torch.cuda.synchronize(runtime.device)

        exported, registration, export_error = _collective_call(
            comm, "HBM registration", lambda: _register_storage(storage, args)
        )
        mr = registration.mr.contents
        requested_send_wr = max(
            messages_per_round * (args.rows + 1), 64
        )
        queue = _collective_call(
            comm,
            "QP creation",
            lambda: HostSglQueue(
                args.library.resolve(),
                context=registration.context,
                pd=registration.pd,
                port=args.port,
                gid_index=args.gid_index,
                requested_send_wr=requested_send_wr,
                requested_send_sge=args.requested_sge,
            ),
        )
        if args.submission in {"both", "ring", "resident"}:
            ring = _collective_call(
                comm, "coherent ring creation", queue.create_coherent_ring
            )
        endpoint = _collective_call(comm, "endpoint query", queue.endpoint)
        endpoint_bytes = ctypes.string_at(
            ctypes.byref(endpoint), ctypes.sizeof(endpoint)
        )
        endpoints = comm.allgather(endpoint_bytes)
        remote_endpoint = HostSglEndpoint.from_buffer_copy(endpoints[peer])
        _collective_call(comm, "QP connection", lambda: queue.connect(remote_endpoint))

        memory_descriptors = comm.allgather((storage.data_ptr(), int(mr.rkey)))
        remote_base, remote_rkey = memory_descriptors[peer]
        registrations = comm.gather(
            (
                runtime.rank,
                registration.device_name,
                registration.registration_mode,
                export_error,
                queue.max_sge,
                queue.max_send_wr,
                native_rdma_ordering,
            ),
            root=0,
        )
        if runtime.rank == 0:
            print(
                "host-sgl setup "
                f"rows={args.rows} row_bytes={args.row_bytes} "
                f"batch_depth={args.batch_depth} source_stride={source_stride} "
                f"pipeline_batches={args.pipeline_batches} "
                f"bytes_per_message={message_bytes} bytes_per_round={destination_bytes} "
                f"native_rdma_ordering={native_rdma_ordering} "
                f"registrations={registrations}"
            )

        def run_once(mode: str, sequence: int) -> tuple[float, int]:
            request_arrays = []
            sequences = []
            completion_sequences = []
            for pipeline_batch in range(args.pipeline_batches):
                request_array = (HostSglRequest * args.batch_depth)()
                for request_index in range(args.batch_depth):
                    message_index = (
                        pipeline_batch * args.batch_depth + request_index
                    )
                    request_sequence = sequence + message_index
                    sequences.append(request_sequence)
                    request_array[request_index] = HostSglRequest(
                        local_lkey=int(mr.lkey),
                        remote_rkey=remote_rkey,
                        source_base=storage.data_ptr(),
                        source_stride=source_stride,
                        row_bytes=args.row_bytes,
                        row_count=args.rows,
                        row_indices=ctypes.cast(
                            row_index_arrays[request_index],
                            ctypes.POINTER(ctypes.c_uint32),
                        ),
                        remote_data=(
                            remote_base
                            + destination_offset
                            + message_index * message_bytes
                        ),
                        remote_signal=(
                            remote_base + signal_offset + message_index * 8
                        ),
                        sequence=request_sequence,
                    )
                request_arrays.append(request_array)
                completion_sequences.append(
                    sequence + (pipeline_batch + 1) * args.batch_depth - 1
                )
            destination.zero_()
            signal.zero_()
            torch.cuda.synchronize(runtime.device)
            comm.Barrier()

            elapsed_us = 0.0
            posted_data_wrs = 0
            local_error = None
            try:
                start_ns = time.perf_counter_ns()
                for request_array in request_arrays:
                    posted_data_wrs += queue.post_indexed_batch(
                        request_array,
                        request_count=args.batch_depth,
                        row_wr_mode=mode == "row-wr",
                    )
                if queue.outstanding_batches != args.pipeline_batches:
                    raise RuntimeError(
                        "host SGL helper did not retain every posted batch"
                    )
                for completion_sequence in completion_sequences:
                    queue.poll(completion_sequence)
                if queue.outstanding_batches != 0 or queue.outstanding_wrs != 0:
                    raise RuntimeError("host SGL SQ credits did not retire")
                elapsed_us = (time.perf_counter_ns() - start_ns) / 1000.0
            except Exception as error:
                local_error = f"{type(error).__name__}: {error}"
            outcomes = comm.allgather(
                (local_error, elapsed_us, posted_data_wrs)
            )
            errors = [outcome[0] for outcome in outcomes]
            if any(error is not None for error in errors):
                raise RuntimeError(f"{mode} transfer failed across PEs: {errors}")

            _make_inbound_rdma_visible(native_rdma_ordering)
            received_sequences = signal.cpu().reshape(-1).tolist()
            correct_data = bool(torch.equal(destination, expected))
            verification_error = None
            if received_sequences != sequences:
                verification_error = (
                    f"readiness={received_sequences}, expected={sequences}"
                )
            elif not correct_data:
                verification_error = "destination is not the indexed-row concatenation"
            verification_errors = comm.allgather(verification_error)
            if any(error is not None for error in verification_errors):
                raise RuntimeError(
                    f"{mode} verification failed across PEs: {verification_errors}"
                )
            return (
                max(outcome[1] for outcome in outcomes),
                max(outcome[2] for outcome in outcomes),
            )

        def run_ring_once(first_generation: int) -> tuple[float, int]:
            if ring is None or ring_route_indices is None:
                raise RuntimeError("coherent ring was not initialized")
            destination.zero_()
            signal.zero_()
            torch.cuda.synchronize(runtime.device)
            comm.Barrier()

            elapsed_us = 0.0
            posted_data_wrs = 0
            local_error = None
            try:
                start_ns = time.perf_counter_ns()
                ring.publish(
                    ring_route_indices,
                    first_generation=first_generation,
                    local_lkey=int(mr.lkey),
                    remote_rkey=remote_rkey,
                    source_base=storage.data_ptr(),
                    source_stride=source_stride,
                    row_bytes=args.row_bytes,
                    remote_data_base=remote_base + destination_offset,
                    remote_data_stride=message_bytes,
                    remote_signal_base=remote_base + signal_offset,
                    remote_signal_stride=8,
                    stream=torch.cuda.current_stream(storage.device),
                )
                posted_data_wrs = ring.consume(
                    queue,
                    first_generation=first_generation,
                    request_count=messages_per_round,
                )
                elapsed_us = (time.perf_counter_ns() - start_ns) / 1000.0
            except Exception as error:
                local_error = f"{type(error).__name__}: {error}"
            outcomes = comm.allgather(
                (local_error, elapsed_us, posted_data_wrs)
            )
            errors = [outcome[0] for outcome in outcomes]
            if any(error is not None for error in errors):
                raise RuntimeError(
                    f"ring SGL transfer failed across PEs: {errors}"
                )

            _make_inbound_rdma_visible(native_rdma_ordering)
            received_generations = signal.cpu().reshape(-1).tolist()
            expected_generations = list(
                range(
                    first_generation,
                    first_generation + messages_per_round,
                )
            )
            verification_error = None
            if received_generations != expected_generations:
                verification_error = (
                    f"readiness={received_generations}, "
                    f"expected={expected_generations}"
                )
            elif not torch.equal(destination, expected):
                verification_error = (
                    "destination is not the indexed-row concatenation"
                )
            verification_errors = comm.allgather(verification_error)
            if any(error is not None for error in verification_errors):
                raise RuntimeError(
                    "ring SGL verification failed across PEs: "
                    f"{verification_errors}"
                )
            return (
                max(outcome[1] for outcome in outcomes),
                max(outcome[2] for outcome in outcomes),
            )

        modes = ("sgl", "row-wr") if args.mode == "both" else (args.mode,)
        summaries: dict[tuple[str, str], tuple[float, int]] = {}
        if args.submission in {"both", "direct"}:
            for mode_index, mode in enumerate(modes, start=1):
                for iteration in range(args.warmup):
                    sequence = (mode_index << 48) | (
                        iteration * messages_per_round + 1
                    )
                    run_once(mode, sequence)
                samples = []
                data_wrs = 0
                for iteration in range(args.iterations):
                    sequence = (mode_index << 48) | (
                        (args.warmup + iteration) * messages_per_round + 1
                    )
                    elapsed_us, data_wrs = run_once(mode, sequence)
                    samples.append(elapsed_us)
                median_us = statistics.median(samples)
                p95_us = _percentile(samples, 0.95)
                summaries[("direct", mode)] = (median_us, data_wrs)
                if runtime.rank == 0:
                    gib_per_second = destination_bytes / (1 << 30) / (
                        median_us / 1_000_000
                    )
                    print(
                        "host-sgl result submission=direct "
                        f"mode={mode} round_p50_us={median_us:.3f} "
                        f"round_p95_us={p95_us:.3f} "
                        f"amortized_per_message_us="
                        f"{median_us / messages_per_round:.3f} "
                        f"data_wrs={data_wrs} "
                        f"post_calls={args.pipeline_batches} "
                        f"readiness_wrs={messages_per_round} "
                        f"local_completions={args.pipeline_batches} "
                        f"aggregate_gib_per_second_per_direction="
                        f"{gib_per_second:.3f}"
                    )

        if args.submission in {"both", "ring"} and "sgl" in modes:
            for iteration in range(args.warmup):
                generation = iteration * messages_per_round + 1
                run_ring_once(generation)
            samples = []
            data_wrs = 0
            for iteration in range(args.iterations):
                generation = (
                    args.warmup + iteration
                ) * messages_per_round + 1
                elapsed_us, data_wrs = run_ring_once(generation)
                samples.append(elapsed_us)
            median_us = statistics.median(samples)
            p95_us = _percentile(samples, 0.95)
            summaries[("ring", "sgl")] = (median_us, data_wrs)
            if runtime.rank == 0:
                gib_per_second = destination_bytes / (1 << 30) / (
                    median_us / 1_000_000
                )
                print(
                    "host-sgl result submission=ring mode=sgl "
                    f"round_p50_us={median_us:.3f} "
                    f"round_p95_us={p95_us:.3f} "
                    f"amortized_per_message_us="
                    f"{median_us / messages_per_round:.3f} "
                    f"data_wrs={data_wrs} batch_limit=16 "
                    f"readiness_wrs={messages_per_round} "
                    f"aggregate_gib_per_second_per_direction="
                    f"{gib_per_second:.3f}"
                )

        if args.submission == "resident" and "sgl" in modes:
            if ring is None or ring_route_indices is None:
                raise RuntimeError("coherent ring was not initialized")
            total_rounds = args.warmup + args.iterations
            destination.zero_()
            signal.zero_()
            torch.cuda.synchronize(runtime.device)
            comm.Barrier()
            ring.publish_resident(
                ring_route_indices,
                first_generation=1,
                round_count=total_rounds,
                local_lkey=int(mr.lkey),
                remote_rkey=remote_rkey,
                source_base=storage.data_ptr(),
                source_stride=source_stride,
                row_bytes=args.row_bytes,
                remote_data_base=remote_base + destination_offset,
                remote_data_stride=message_bytes,
                remote_signal_base=remote_base + signal_offset,
                remote_signal_stride=8,
                stream=torch.cuda.current_stream(storage.device),
            )
            samples = []
            data_wrs = 0
            for iteration in range(total_rounds):
                generation = iteration * messages_per_round + 1
                start_ns = time.perf_counter_ns()
                data_wrs = ring.consume(
                    queue,
                    first_generation=generation,
                    request_count=messages_per_round,
                )
                elapsed_us = (time.perf_counter_ns() - start_ns) / 1000.0
                elapsed_us = max(comm.allgather(elapsed_us))
                if iteration >= args.warmup:
                    samples.append(elapsed_us)
            torch.cuda.synchronize(runtime.device)
            comm.Barrier()
            _make_inbound_rdma_visible(native_rdma_ordering)
            final_generation = (
                total_rounds - 1
            ) * messages_per_round + 1
            expected_generations = list(
                range(
                    final_generation,
                    final_generation + messages_per_round,
                )
            )
            verification_error = None
            if signal.cpu().reshape(-1).tolist() != expected_generations:
                verification_error = "resident readiness generation mismatch"
            elif not torch.equal(destination, expected):
                verification_error = "resident destination mismatch"
            verification_errors = comm.allgather(verification_error)
            if any(error is not None for error in verification_errors):
                raise RuntimeError(
                    "resident ring verification failed across PEs: "
                    f"{verification_errors}"
                )
            median_us = statistics.median(samples)
            p95_us = _percentile(samples, 0.95)
            summaries[("resident", "sgl")] = (median_us, data_wrs)
            if runtime.rank == 0:
                gib_per_second = destination_bytes / (1 << 30) / (
                    median_us / 1_000_000
                )
                print(
                    "host-sgl result submission=resident mode=sgl "
                    f"round_p50_us={median_us:.3f} "
                    f"round_p95_us={p95_us:.3f} "
                    f"amortized_per_message_us="
                    f"{median_us / messages_per_round:.3f} "
                    f"data_wrs={data_wrs} resident_warps=8 "
                    f"readiness_wrs={messages_per_round} "
                    f"aggregate_gib_per_second_per_direction="
                    f"{gib_per_second:.3f}"
                )

        if runtime.rank == 0 and {
            ("direct", "sgl"),
            ("direct", "row-wr"),
        }.issubset(summaries):
            sgl_us, sgl_wrs = summaries[("direct", "sgl")]
            row_us, row_wrs = summaries[("direct", "row-wr")]
            print(
                "host-sgl comparison "
                f"latency_speedup={row_us / sgl_us:.3f}x "
                f"data_wr_reduction={row_wrs / sgl_wrs:.3f}x"
            )
        if runtime.rank == 0 and {
            ("direct", "sgl"),
            ("ring", "sgl"),
        }.issubset(summaries):
            direct_us, _ = summaries[("direct", "sgl")]
            ring_us, _ = summaries[("ring", "sgl")]
            print(
                "host-sgl coherent-ring comparison "
                f"ring_over_direct={ring_us / direct_us:.3f}x "
                f"handoff_overhead_us={ring_us - direct_us:.3f}"
            )
    finally:
        if ring is not None:
            ring.close()
        if queue is not None:
            queue.close()
        if registration is not None:
            registration.close()
        if exported is not None:
            os.close(exported.fd)
        nvshmem.finalize()


if __name__ == "__main__":
    main()
