"""MPI-bootstrapped NVSHMEM support for DAE.

All symmetric allocations are collective. Every PE must call allocation APIs
with the same shapes, dtypes, and order. Call :func:`finalize` only after all
CUDA work is complete and no returned tensor will be used again.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from numbers import Integral
from types import ModuleType
from typing import Any, Iterable, Mapping

import torch

from .launcher import Launcher as _DAELauncher


_backend: ModuleType | None = None


@dataclass(frozen=True)
class RuntimeInfo:
    rank: int
    world_size: int
    local_rank: int
    local_size: int
    device: int
    pe: int
    num_pes: int
    mpi_thread_level: int
    owns_mpi: bool
    owns_nvshmem: bool
    nvshmem_name: str
    nvshmem_version: tuple[int, int, int]
    symmetric_size: str
    allocation_count: int

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "RuntimeInfo":
        normalized = dict(values)
        normalized["nvshmem_version"] = tuple(normalized["nvshmem_version"])
        return cls(**normalized)


def _load_backend() -> ModuleType:
    global _backend
    if _backend is not None:
        return _backend
    try:
        _backend = import_module("dae._nvshmem_runtime")
    except ImportError as error:
        raise RuntimeError(
            "The optional DAE NVSHMEM runtime is unavailable. Run "
            "`make nvshmem-pyext` and ensure its MPI/NVSHMEM shared libraries "
            f"are visible. Original import error: {error}"
        ) from error
    return _backend


def available() -> bool:
    try:
        _load_backend()
    except RuntimeError:
        return False
    return True


def _format_symmetric_size(size: str | int | None) -> str:
    if size is None:
        return ""
    if isinstance(size, Integral):
        if int(size) <= 0:
            raise ValueError("symmetric_size must be positive")
        return str(int(size))
    if isinstance(size, str) and size.strip():
        return size.strip()
    raise TypeError("symmetric_size must be a positive byte count, a size string, or None")


def _device_index(device: torch.device | str | int | None) -> int | None:
    if device is None:
        return None
    if isinstance(device, Integral):
        return int(device)
    parsed = torch.device(device)
    if parsed.type != "cuda":
        raise ValueError("NVSHMEM requires a CUDA device")
    return parsed.index


def init(
    *,
    symmetric_size: str | int | None = None,
    device: torch.device | str | int | None = None,
) -> RuntimeInfo:
    """Collectively initialize MPI and NVSHMEM for the current ``ibrun`` rank."""

    requested_device = _device_index(device)
    values = _load_backend().initialize(
        _format_symmetric_size(symmetric_size),
        -1 if requested_device is None else requested_device,
    )
    result = RuntimeInfo.from_mapping(values)
    torch.cuda.set_device(result.device)
    return result


initialize = init


def is_initialized() -> bool:
    return bool(_load_backend().is_initialized())


def info() -> RuntimeInfo:
    return RuntimeInfo.from_mapping(_load_backend().info())


def my_pe() -> int:
    return info().pe


def n_pes() -> int:
    return info().num_pes


def _normalize_shape(shape: int | Iterable[int]) -> tuple[int, ...]:
    if isinstance(shape, Integral):
        values = (int(shape),)
    else:
        try:
            values = tuple(shape)
        except TypeError as error:
            raise TypeError("shape must be an integer or an iterable of integers") from error

    normalized = []
    for dimension in values:
        if not isinstance(dimension, Integral):
            raise TypeError("all tensor dimensions must be integers")
        dimension = int(dimension)
        if dimension < 0:
            raise ValueError("all tensor dimensions must be non-negative")
        normalized.append(dimension)
    return tuple(normalized)


def _normalize_size_args(size: tuple[Any, ...]) -> tuple[int, ...]:
    if len(size) == 1 and not isinstance(size[0], Integral):
        return _normalize_shape(size[0])
    return _normalize_shape(size)


def _validate_allocation_device(device: torch.device | str | int | None) -> None:
    requested = _device_index(device)
    runtime = info() if requested is not None else None
    if runtime is not None and requested != runtime.device:
        raise ValueError(
            f"NVSHMEM is initialized on CUDA device {runtime.device}, not {requested}"
        )


def allocate_tensor(
    shape: int | Iterable[int],
    *,
    dtype: torch.dtype = torch.float32,
    zeroed: bool = False,
    device: torch.device | str | int | None = None,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Collectively allocate a contiguous Torch tensor in the symmetric heap."""

    if not isinstance(dtype, torch.dtype):
        raise TypeError("dtype must be a torch.dtype")
    _validate_allocation_device(device)
    tensor = _load_backend().allocate_tensor(
        _normalize_shape(shape), dtype, bool(zeroed)
    )
    return tensor.requires_grad_(requires_grad)


def empty(
    *size: int | Iterable[int],
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | int | None = None,
    requires_grad: bool = False,
) -> torch.Tensor:
    return allocate_tensor(
        _normalize_size_args(size),
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def zeros(
    *size: int | Iterable[int],
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | int | None = None,
    requires_grad: bool = False,
) -> torch.Tensor:
    return allocate_tensor(
        _normalize_size_args(size),
        dtype=dtype,
        zeroed=True,
        device=device,
        requires_grad=requires_grad,
    )


def init_signal_space(signal_count: int) -> torch.Tensor:
    """Collectively create the process-global symmetric ``uint64`` signal array."""

    if not isinstance(signal_count, Integral) or int(signal_count) <= 0:
        raise ValueError("signal_count must be a positive integer")
    return _load_backend().init_signal_space(int(signal_count))


init_global_signal_space = init_signal_space


def get_signal_space() -> torch.Tensor:
    return _load_backend().get_signal_space()


def is_symmetric_tensor(tensor: torch.Tensor) -> bool:
    if not isinstance(tensor, torch.Tensor):
        return False
    return bool(_load_backend().is_symmetric_tensor(tensor))


def _stream_id(stream: torch.cuda.Stream | int | None) -> int:
    if stream is None:
        return int(torch.cuda.current_stream(info().device).cuda_stream)
    if isinstance(stream, Integral):
        return int(stream)
    if not hasattr(stream, "cuda_stream"):
        raise TypeError("stream must be a torch.cuda.Stream, integer handle, or None")
    return int(stream.cuda_stream)


def signal(
    index: int,
    value: int,
    pe: int,
    *,
    op: str = "set",
    stream: torch.cuda.Stream | int | None = None,
) -> None:
    operations = {
        "set": _load_backend().SIGNAL_SET,
        "add": _load_backend().SIGNAL_ADD,
    }
    try:
        operation = operations[op.lower()]
    except (AttributeError, KeyError) as error:
        raise ValueError("op must be 'set' or 'add'") from error
    _load_backend().signal_on_stream(
        int(index), int(value), operation, int(pe), _stream_id(stream)
    )


def wait_signal(
    index: int,
    value: int,
    *,
    comparison: str = "eq",
    stream: torch.cuda.Stream | int | None = None,
) -> None:
    comparisons = {
        "eq": _load_backend().CMP_EQ,
        "ne": _load_backend().CMP_NE,
        "gt": _load_backend().CMP_GT,
        "ge": _load_backend().CMP_GE,
        "lt": _load_backend().CMP_LT,
        "le": _load_backend().CMP_LE,
    }
    try:
        comparison_code = comparisons[comparison.lower()]
    except (AttributeError, KeyError) as error:
        raise ValueError("comparison must be one of eq/ne/gt/ge/lt/le") from error
    _load_backend().wait_signal_on_stream(
        int(index), comparison_code, int(value), _stream_id(stream)
    )


def quiet(stream: torch.cuda.Stream | int | None = None) -> None:
    _load_backend().quiet_on_stream(_stream_id(stream))


def barrier() -> None:
    _load_backend().barrier_all()


barrier_all = barrier


def finalize() -> None:
    """Collectively free tracked allocations and finalize owned runtimes."""

    _load_backend().finalize()


class NVSHMEMLauncher(_DAELauncher):
    """DAE launcher initialized for one MPI rank / NVSHMEM PE per GPU."""

    def __init__(
        self,
        num_sms: int = 1,
        device: torch.device | str | int | None = None,
        *,
        symmetric_size: str | int | None = None,
        signal_count: int | None = None,
    ):
        self.nvshmem_info = init(
            symmetric_size=symmetric_size,
            device=device,
        )
        selected_device = torch.device("cuda", self.nvshmem_info.device)
        super().__init__(num_sms=num_sms, device=selected_device)
        self.signal_space: torch.Tensor | None = None
        if signal_count is not None:
            self.signal_space = init_signal_space(signal_count)

    @property
    def pe(self) -> int:
        return self.nvshmem_info.pe

    @property
    def num_pes(self) -> int:
        return self.nvshmem_info.num_pes

    def init_signal_space(self, signal_count: int) -> torch.Tensor:
        self.signal_space = init_signal_space(signal_count)
        return self.signal_space

    def allocate_tensor(self, shape, **kwargs) -> torch.Tensor:
        return allocate_tensor(shape, **kwargs)

    def empty(self, *size, **kwargs) -> torch.Tensor:
        return empty(*size, **kwargs)

    def zeros(self, *size, **kwargs) -> torch.Tensor:
        return zeros(*size, **kwargs)

    def barrier(self) -> None:
        barrier()

    def signal(self, index: int, value: int, pe: int, **kwargs) -> None:
        signal(index, value, pe, **kwargs)

    def wait_signal(self, index: int, value: int, **kwargs) -> None:
        wait_signal(index, value, **kwargs)

    def finalize(self) -> None:
        finalize()


Launcher = NVSHMEMLauncher
alloc_tensor = allocate_tensor


__all__ = [
    "RuntimeInfo",
    "NVSHMEMLauncher",
    "Launcher",
    "available",
    "init",
    "initialize",
    "is_initialized",
    "info",
    "my_pe",
    "n_pes",
    "allocate_tensor",
    "alloc_tensor",
    "empty",
    "zeros",
    "init_signal_space",
    "init_global_signal_space",
    "get_signal_space",
    "is_symmetric_tensor",
    "signal",
    "wait_signal",
    "quiet",
    "barrier",
    "barrier_all",
    "finalize",
]
