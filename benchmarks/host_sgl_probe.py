"""Probe host-verbs registration of a CUDA/NVSHMEM HBM allocation.

This is an external capability probe, not a VDCores operator or timing path.
It answers the first question for a Grace-hosted SGL transport: can CUDA
register the actual symmetric HBM range through CUDA DMA-BUF or the legacy
peer-memory path, and can the local mlx5 device use the resulting ibverbs MR?

Run only on a Vista GH compute allocation, for example::

    NVSHMEM_DISABLE_NCCL=1 ibrun -n 2 \
      python benchmarks/host_sgl_probe.py
"""

from __future__ import annotations

import argparse
import ctypes
import os
from dataclasses import dataclass

import torch
from cuda.bindings import driver as cuda
from mpi4py import MPI

import dae.nvshmem as nvshmem


IBV_ACCESS_LOCAL_WRITE = 1 << 0
IBV_ACCESS_REMOTE_WRITE = 1 << 1
IBV_ACCESS_REMOTE_READ = 1 << 2


class _IbvMr(ctypes.Structure):
    _fields_ = (
        ("context", ctypes.c_void_p),
        ("pd", ctypes.c_void_p),
        ("addr", ctypes.c_void_p),
        ("length", ctypes.c_size_t),
        ("handle", ctypes.c_uint32),
        ("lkey", ctypes.c_uint32),
        ("rkey", ctypes.c_uint32),
    )


@dataclass(frozen=True)
class ExportedRange:
    base: int
    size: int
    fd: int
    allocation_range_queried: bool


def _cuda_value(result: tuple, operation: str):
    status, *values = result
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{operation} failed: {status!r}")
    if len(values) == 1:
        return values[0]
    return tuple(values)


def _export_tensor(tensor: torch.Tensor, *, pcie_bar1: bool) -> ExportedRange:
    pointer = tensor.data_ptr()
    tensor_end = pointer + tensor.numel() * tensor.element_size()
    page_bytes = os.sysconf("SC_PAGE_SIZE")
    export_base = pointer // page_bytes * page_bytes
    export_end = (tensor_end + page_bytes - 1) // page_bytes * page_bytes
    # NVSHMEM may back its symmetric heap with CUDA VMM rather than cuMemAlloc.
    # cuMemGetAddressRange is useful when available, but CUDA explicitly lets
    # cuMemGetHandleForAddressRange export a fully mapped cuMemAddressReserve
    # range as well.  In that case the export call itself is the authority.
    range_result = cuda.cuMemGetAddressRange(cuda.CUdeviceptr(pointer))
    range_status, *range_values = range_result
    allocation_range_queried = range_status == cuda.CUresult.CUDA_SUCCESS
    if allocation_range_queried:
        allocation_base, allocation_bytes = map(int, range_values)
        if (
            export_base < allocation_base
            or export_end > allocation_base + allocation_bytes
        ):
            raise RuntimeError("page-aligned tensor range exceeds its CUDA allocation")
    elif range_status not in {
        cuda.CUresult.CUDA_ERROR_NOT_FOUND,
        cuda.CUresult.CUDA_ERROR_INVALID_VALUE,
    }:
        raise RuntimeError(f"cuMemGetAddressRange failed: {range_status!r}")
    flags = (
        int(cuda.CUmemRangeFlags.CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE)
        if pcie_bar1
        else 0
    )
    fd = _cuda_value(
        cuda.cuMemGetHandleForAddressRange(
            cuda.CUdeviceptr(export_base),
            export_end - export_base,
            cuda.CUmemRangeHandleType.CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
            flags,
        ),
        "cuMemGetHandleForAddressRange",
    )
    return ExportedRange(
        export_base,
        export_end - export_base,
        int(fd),
        allocation_range_queried,
    )


class VerbsRegistration:
    def __init__(
        self,
        *,
        address: int,
        length: int,
        exported: ExportedRange | None,
        device_name: str | None,
        registration_mode: str,
    ):
        self._verbs = ctypes.CDLL("libibverbs.so.1", use_errno=True)
        self._verbs.ibv_get_device_list.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self._verbs.ibv_get_device_list.restype = ctypes.POINTER(ctypes.c_void_p)
        self._verbs.ibv_get_device_name.argtypes = [ctypes.c_void_p]
        self._verbs.ibv_get_device_name.restype = ctypes.c_char_p
        self._verbs.ibv_open_device.argtypes = [ctypes.c_void_p]
        self._verbs.ibv_open_device.restype = ctypes.c_void_p
        self._verbs.ibv_alloc_pd.argtypes = [ctypes.c_void_p]
        self._verbs.ibv_alloc_pd.restype = ctypes.c_void_p
        self._verbs.ibv_reg_dmabuf_mr.argtypes = (
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_int,
        )
        self._verbs.ibv_reg_dmabuf_mr.restype = ctypes.POINTER(_IbvMr)
        self._verbs.ibv_reg_mr.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        )
        self._verbs.ibv_reg_mr.restype = ctypes.POINTER(_IbvMr)
        self._verbs.ibv_dereg_mr.argtypes = [ctypes.POINTER(_IbvMr)]
        self._verbs.ibv_dereg_mr.restype = ctypes.c_int
        self._verbs.ibv_dealloc_pd.argtypes = [ctypes.c_void_p]
        self._verbs.ibv_dealloc_pd.restype = ctypes.c_int
        self._verbs.ibv_close_device.argtypes = [ctypes.c_void_p]
        self._verbs.ibv_close_device.restype = ctypes.c_int
        self._verbs.ibv_free_device_list.argtypes = [
            ctypes.POINTER(ctypes.c_void_p)
        ]
        self._verbs.ibv_free_device_list.restype = None

        count = ctypes.c_int()
        self.devices = self._verbs.ibv_get_device_list(ctypes.byref(count))
        if not self.devices or count.value == 0:
            self._raise_errno("ibv_get_device_list")
        self.device = None
        self.device_name = ""
        self.registration_mode = ""
        self.context = None
        self.pd = None
        self.mr = None
        access = (
            IBV_ACCESS_LOCAL_WRITE
            | IBV_ACCESS_REMOTE_WRITE
            | IBV_ACCESS_REMOTE_READ
        )
        failures: list[str] = []
        for index in range(count.value):
            candidate = self.devices[index]
            candidate_name = self._verbs.ibv_get_device_name(candidate).decode()
            if device_name is not None and candidate_name != device_name:
                continue
            context = self._verbs.ibv_open_device(candidate)
            if not context:
                failures.append(f"{candidate_name}: open errno={ctypes.get_errno()}")
                continue
            pd = self._verbs.ibv_alloc_pd(context)
            if not pd:
                failures.append(f"{candidate_name}: alloc-pd errno={ctypes.get_errno()}")
                self._verbs.ibv_close_device(context)
                continue
            mr = None
            selected_mode = ""
            if registration_mode in {"auto", "dmabuf"} and exported is not None:
                mr = self._verbs.ibv_reg_dmabuf_mr(
                    pd,
                    0,
                    exported.size,
                    exported.base,
                    exported.fd,
                    access,
                )
                if mr:
                    selected_mode = "dmabuf"
                else:
                    failures.append(
                        f"{candidate_name}: reg-dmabuf errno={ctypes.get_errno()}"
                    )
            if not mr and registration_mode in {"auto", "legacy"}:
                mr = self._verbs.ibv_reg_mr(
                    pd,
                    ctypes.c_void_p(address),
                    length,
                    access,
                )
                if mr:
                    selected_mode = "legacy-peer-memory"
                else:
                    failures.append(
                        f"{candidate_name}: reg-mr errno={ctypes.get_errno()}"
                    )
            if mr:
                self.device = candidate
                self.device_name = candidate_name
                self.context = context
                self.pd = pd
                self.mr = mr
                self.registration_mode = selected_mode
                break
            self._verbs.ibv_dealloc_pd(pd)
            self._verbs.ibv_close_device(context)
        if self.mr is None:
            self._verbs.ibv_free_device_list(self.devices)
            self.devices = None
            requested = device_name if device_name is not None else "any device"
            raise RuntimeError(
                f"no HBM DMA-BUF MR on {requested}: {', '.join(failures)}"
            )

    @staticmethod
    def _raise_errno(operation: str) -> None:
        error = ctypes.get_errno()
        raise OSError(error, f"{operation}: {os.strerror(error)}")

    def close(self) -> None:
        if getattr(self, "mr", None):
            if self._verbs.ibv_dereg_mr(self.mr) != 0:
                self._raise_errno("ibv_dereg_mr")
            self.mr = None
        if getattr(self, "pd", None):
            if self._verbs.ibv_dealloc_pd(self.pd) != 0:
                self._raise_errno("ibv_dealloc_pd")
            self.pd = None
        if getattr(self, "context", None):
            if self._verbs.ibv_close_device(self.context) != 0:
                self._raise_errno("ibv_close_device")
            self.context = None
        if getattr(self, "devices", None):
            self._verbs.ibv_free_device_list(self.devices)
            self.devices = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bytes", type=int, default=4 << 20)
    parser.add_argument("--symmetric-size", default="64M")
    parser.add_argument(
        "--ib-device",
        help="verbs device name; default tries local devices in enumeration order",
    )
    parser.add_argument(
        "--registration",
        choices=("auto", "dmabuf", "legacy"),
        default="auto",
        help="GPU HBM MR mechanism; auto tries DMA-BUF then peer-memory",
    )
    parser.add_argument(
        "--pcie-bar1",
        action="store_true",
        help="request the CUDA PCIe BAR1 DMA-BUF mapping instead of native mapping",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bytes <= 0:
        raise ValueError("--bytes must be positive")
    comm = MPI.COMM_WORLD
    runtime = nvshmem.init(symmetric_size=args.symmetric_size)
    exported = None
    registration = None
    try:
        facts = None
        probe_error = None
        try:
            device = _cuda_value(cuda.cuCtxGetDevice(), "cuCtxGetDevice")
            dma_buf = _cuda_value(
                cuda.cuDeviceGetAttribute(
                    cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED,
                    device,
                ),
                "CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED",
            )
            host_dma_buf = _cuda_value(
                cuda.cuDeviceGetAttribute(
                    cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_ALLOC_DMA_BUF_SUPPORTED,
                    device,
                ),
                "CU_DEVICE_ATTRIBUTE_HOST_ALLOC_DMA_BUF_SUPPORTED",
            )
            pageable = _cuda_value(
                cuda.cuDeviceGetAttribute(
                    cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,
                    device,
                ),
                "CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS",
            )
            host_page_tables = _cuda_value(
                cuda.cuDeviceGetAttribute(
                    cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES,
                    device,
                ),
                "CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES",
            )
            host_atomics = _cuda_value(
                cuda.cuDeviceGetAttribute(
                    cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED,
                    device,
                ),
                "CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED",
            )
            storage = nvshmem.zeros(args.bytes, dtype=torch.uint8)
            export_error = None
            if args.registration != "legacy":
                try:
                    exported = _export_tensor(
                        storage, pcie_bar1=args.pcie_bar1
                    )
                except Exception as error:
                    export_error = f"{type(error).__name__}: {error}"
                    if args.registration == "dmabuf":
                        raise
            registration = VerbsRegistration(
                address=storage.data_ptr(),
                length=storage.numel() * storage.element_size(),
                exported=exported,
                device_name=args.ib_device,
                registration_mode=args.registration,
            )
            mr = registration.mr.contents
            facts = (
                int(dma_buf),
                int(host_dma_buf),
                int(pageable),
                int(host_page_tables),
                int(host_atomics),
                registration.device_name,
                registration.registration_mode,
                export_error,
                exported.size if exported is not None else None,
                (
                    exported.allocation_range_queried
                    if exported is not None
                    else None
                ),
                int(mr.lkey),
                int(mr.rkey),
            )
        except Exception as error:  # keep every MPI rank in the probe collectives
            probe_error = f"{type(error).__name__}: {error}"
        gathered = comm.gather(
            (runtime.rank, facts, probe_error), root=0
        )
        failed = comm.allreduce(probe_error is not None, op=MPI.LOR)
        if runtime.rank == 0:
            outcome = "FAIL" if failed else "PASS"
            print(
                f"host-sgl HBM registration {outcome} "
                f"pes={runtime.num_pes} pcie_bar1={args.pcie_bar1} "
                f"facts={gathered}"
            )
        if failed:
            raise RuntimeError("HBM DMA-BUF registration failed on at least one PE")
    finally:
        if registration is not None:
            registration.close()
        if exported is not None:
            os.close(exported.fd)
        nvshmem.finalize()


if __name__ == "__main__":
    main()
