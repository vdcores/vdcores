"""Optional NVSHMEM support for DAE.

NVSHMEM4Py owns MPI/NVSHMEM initialization and host operations. The compiled
``dae._nvshmem_runtime`` module only turns collective NVSHMEM allocations into
Torch tensors. Every PE must allocate and release those tensors in the same
order.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import import_module, util as importlib_util
from numbers import Integral
from types import ModuleType
from typing import Any, Iterable

import torch


_allocator_backend: ModuleType | None = None
_runtime_backend: ModuleType | None = None
_host_backend: ModuleType | None = None
_bindings_backend: ModuleType | None = None
_mpi_backend: ModuleType | None = None


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
    owns_nvshmem: bool
    nvshmem_name: str
    nvshmem_version: tuple[int, int, int]
    symmetric_size: str
    transport: str


@dataclass
class _RuntimeState:
    runtime_info: RuntimeInfo
    local_comm: Any
    host: ModuleType
    bindings: ModuleType
    signal_space: torch.Tensor | None = None


_state: _RuntimeState | None = None


def _missing_dependency(error: ImportError) -> RuntimeError:
    return RuntimeError(
        "DAE NVSHMEM support requires the optional allocator build and the "
        "official NVSHMEM4Py packages. Run `make nvshmem-pyext` after "
        "installing nvshmem4py-cu13==0.1.3 and an OpenMPI-compatible mpi4py. "
        f"Original import error: {error}"
    )


def _load_allocator() -> ModuleType:
    global _allocator_backend
    if _allocator_backend is None:
        try:
            _allocator_backend = import_module("dae._nvshmem_runtime")
        except ImportError as error:
            raise _missing_dependency(error) from error
    if not bool(getattr(_allocator_backend, "NVSHMEM_ENABLED", False)):
        raise RuntimeError("dae._nvshmem_runtime was not built with NVSHMEM enabled")
    return _allocator_backend


def _load_runtime() -> ModuleType:
    global _runtime_backend
    if _runtime_backend is None:
        _runtime_backend = import_module("dae.runtime")
    if not bool(getattr(_runtime_backend.config, "nvshmem_enabled", False)):
        raise RuntimeError("dae.runtime was not built with NVSHMEM enabled")
    return _runtime_backend


def _load_host() -> ModuleType:
    global _host_backend
    if _host_backend is None:
        try:
            _host_backend = import_module("nvshmem.core")
        except ImportError as error:
            raise _missing_dependency(error) from error
    return _host_backend


def _load_bindings() -> ModuleType:
    global _bindings_backend
    if _bindings_backend is None:
        try:
            _bindings_backend = import_module("nvshmem.bindings")
        except ImportError as error:
            raise _missing_dependency(error) from error
    return _bindings_backend


def _load_mpi() -> ModuleType:
    global _mpi_backend
    if _mpi_backend is None:
        try:
            _mpi_backend = import_module("mpi4py.MPI")
        except ImportError as error:
            raise _missing_dependency(error) from error
    return _mpi_backend


def available() -> bool:
    try:
        required_modules = ("nvshmem.core", "mpi4py.MPI", "cuda.core.experimental")
        if any(importlib_util.find_spec(name) is None for name in required_modules):
            return False
        _load_allocator()
        _load_runtime()
    except (ImportError, RuntimeError):
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


def _configure_environment(symmetric_size: str, transport: str) -> None:
    if transport not in {"auto", "nvlink", "ibgda"}:
        raise ValueError("transport must be one of: auto, nvlink, ibgda")
    defaults = {
        "NVSHMEM_BOOTSTRAP": "MPI",
        "NVSHMEM_SYMMETRIC_SIZE": "512M",
    }
    if transport == "nvlink":
        # NVSHMEM uses its direct P2P path for same-node peers. Disabling the
        # remote transport and IBGDA makes an accidental NIC datapath fail
        # during initialization instead of contaminating NVLink measurements.
        defaults.update({
            "NVSHMEM_REMOTE_TRANSPORT": "none",
            "NVSHMEM_IB_ENABLE_IBGDA": "0",
        })
    elif transport == "ibgda":
        defaults.update({
            "NVSHMEM_REMOTE_TRANSPORT": "ibrc",
            "NVSHMEM_IB_ENABLE_IBGDA": "1",
            "NVSHMEM_IBGDA_NIC_HANDLER": "gpu",
        })
    for name, value in defaults.items():
        os.environ.setdefault(name, value)
    if symmetric_size:
        os.environ["NVSHMEM_SYMMETRIC_SIZE"] = symmetric_size


def _version_tuple(version: str) -> tuple[int, int, int]:
    values = [int(part) for part in version.split(".")[:3]]
    values.extend([0] * (3 - len(values)))
    return values[0], values[1], values[2]


def _host_is_initialized(
    host: ModuleType, bindings: ModuleType | None = None
) -> bool:
    status_fn = getattr(host, "init_status", None)
    if status_fn is None:
        status_fn = getattr(bindings or _load_bindings(), "init_status")
    status = int(status_fn())
    return 2 <= status <= 4


def init(
    *,
    symmetric_size: str | int | None = None,
    device: torch.device | str | int | None = None,
    transport: str = "auto",
) -> RuntimeInfo:
    """Collectively initialize NVSHMEM4Py and the DAE CUDA module."""

    global _state
    requested_size = _format_symmetric_size(symmetric_size)
    requested_device = _device_index(device)

    if _state is not None:
        current = info()
        if requested_device is not None and requested_device != current.device:
            raise ValueError(
                f"NVSHMEM is already initialized on CUDA device {current.device}, "
                f"not {requested_device}"
            )
        return current

    _configure_environment(requested_size, transport)
    _load_allocator()
    dae_runtime = _load_runtime()
    host = _load_host()
    bindings = _load_bindings()
    mpi = _load_mpi()

    if mpi.Is_finalized():
        raise RuntimeError("MPI was already finalized and cannot be reinitialized")

    world = mpi.COMM_WORLD
    rank = world.Get_rank()
    world_size = world.Get_size()
    local_comm = world.Split_type(mpi.COMM_TYPE_SHARED, key=rank)
    local_rank = local_comm.Get_rank()
    local_size = local_comm.Get_size()

    owns_nvshmem = False
    module_initialized = False
    try:
        device_count = torch.cuda.device_count()
        if device_count <= 0:
            raise RuntimeError("No CUDA devices are visible to this MPI rank")

        if requested_device is None:
            selected_device = 0 if device_count == 1 else local_rank
        else:
            selected_device = requested_device
        if not 0 <= selected_device < device_count:
            raise ValueError(
                f"CUDA device {selected_device} is outside [0, {device_count}) for "
                f"local MPI rank {local_rank}/{local_size}"
            )

        torch.cuda.set_device(selected_device)
        cuda_device_type = getattr(
            import_module("cuda.core.experimental"), "Device"
        )
        cuda_device = cuda_device_type(selected_device)
        cuda_device.set_current()

        if not _host_is_initialized(host, bindings):
            host.init(
                device=cuda_device,
                mpi_comm=world,
                initializer_method="mpi",
            )
            owns_nvshmem = True

        pe = int(host.my_pe())
        num_pes = int(host.n_pes())
        if num_pes != world_size:
            raise RuntimeError(
                f"NVSHMEM PE count {num_pes} does not match MPI world size {world_size}"
            )

        version = host.get_version()
        runtime_info = RuntimeInfo(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            local_size=local_size,
            device=selected_device,
            pe=pe,
            num_pes=num_pes,
            mpi_thread_level=int(mpi.Query_thread()),
            owns_nvshmem=owns_nvshmem,
            nvshmem_name="NVSHMEM4Py",
            nvshmem_version=_version_tuple(version.libnvshmem_version),
            symmetric_size=os.environ["NVSHMEM_SYMMETRIC_SIZE"],
            transport=transport,
        )

        status = int(dae_runtime._nvshmem_module_init())
        if status != 0:
            raise RuntimeError(
                f"DAE NVSHMEM CUDA module initialization failed with status {status}"
            )
        module_initialized = True
    except Exception:
        if module_initialized:
            dae_runtime._nvshmem_module_finalize()
        if owns_nvshmem and _host_is_initialized(host, bindings):
            host.finalize()
        local_comm.Free()
        raise

    _state = _RuntimeState(
        runtime_info=runtime_info,
        local_comm=local_comm,
        host=host,
        bindings=bindings,
    )
    return runtime_info


def is_initialized() -> bool:
    return _state is not None and _host_is_initialized(_state.host)


def _require_state() -> _RuntimeState:
    if _state is None or not _host_is_initialized(_state.host):
        raise RuntimeError("NVSHMEM is not initialized; call dae.nvshmem.init() first")
    return _state


def info() -> RuntimeInfo:
    return _require_state().runtime_info


def my_pe() -> int:
    return _require_state().runtime_info.pe


def n_pes() -> int:
    return _require_state().runtime_info.num_pes


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
    runtime = _require_state().runtime_info
    if requested is not None and requested != runtime.device:
        raise ValueError(
            f"NVSHMEM is initialized on CUDA device {runtime.device}, not {requested}"
        )
    torch.cuda.set_device(runtime.device)


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
    tensor = _load_allocator().allocate_tensor(
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
    """Collectively allocate the uint64 signal array as a symmetric tensor."""

    if not isinstance(signal_count, Integral) or int(signal_count) <= 0:
        raise ValueError("signal_count must be a positive integer")
    signal_count = int(signal_count)

    state = _require_state()
    if state.signal_space is not None:
        if state.signal_space.numel() != signal_count:
            raise ValueError(
                "Signal space is already initialized with "
                f"{state.signal_space.numel()} entries, not {signal_count}"
            )
        barrier()
        return state.signal_space

    state.signal_space = zeros(
        signal_count,
        dtype=torch.uint64,
        device=state.runtime_info.device,
    )
    barrier()
    return state.signal_space


def _signal_address(index: int) -> int:
    if not isinstance(index, Integral):
        raise TypeError("signal index must be an integer")
    state = _require_state()
    if state.signal_space is None:
        raise RuntimeError(
            "Signal space is not initialized; call init_signal_space() first"
        )
    signals = state.signal_space
    index = int(index)
    if not 0 <= index < signals.numel():
        raise ValueError(
            f"signal index {index} is outside [0, {signals.numel()})"
        )
    return signals.data_ptr() + index * signals.element_size()


def is_symmetric_tensor(tensor: torch.Tensor) -> bool:
    if not isinstance(tensor, torch.Tensor):
        return False
    try:
        allocator = _load_allocator()
    except RuntimeError:
        return False
    return bool(allocator.is_symmetric_tensor(tensor))


def _stream_id(stream: torch.cuda.Stream | int | None) -> int:
    runtime = _require_state().runtime_info
    if stream is None:
        return int(torch.cuda.current_stream(runtime.device).cuda_stream)
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
    state = _require_state()
    operations = {
        "set": state.host.SignalOp.SIGNAL_SET,
        "add": state.host.SignalOp.SIGNAL_ADD,
    }
    try:
        operation = operations[op.lower()]
    except (AttributeError, KeyError) as error:
        raise ValueError("op must be 'set' or 'add'") from error
    if not 0 <= int(pe) < state.runtime_info.num_pes:
        raise ValueError("target PE is out of range")
    state.bindings.signal_op_on_stream(
        _signal_address(index),
        int(value),
        int(operation),
        int(pe),
        _stream_id(stream),
    )


def wait_signal(
    index: int,
    value: int,
    *,
    comparison: str = "eq",
    stream: torch.cuda.Stream | int | None = None,
) -> None:
    state = _require_state()
    comparisons = {
        "eq": state.host.ComparisonType.CMP_EQ,
        "ne": state.host.ComparisonType.CMP_NE,
        "gt": state.host.ComparisonType.CMP_GT,
        "ge": state.host.ComparisonType.CMP_GE,
        "lt": state.host.ComparisonType.CMP_LT,
        "le": state.host.ComparisonType.CMP_LE,
    }
    try:
        comparison_code = comparisons[comparison.lower()]
    except (AttributeError, KeyError) as error:
        raise ValueError("comparison must be one of eq/ne/gt/ge/lt/le") from error
    state.bindings.signal_wait_until_on_stream(
        _signal_address(index),
        int(comparison_code),
        int(value),
        _stream_id(stream),
    )


def quiet(stream: torch.cuda.Stream | int | None = None) -> None:
    _require_state().bindings.quiet_on_stream(_stream_id(stream))


def barrier() -> None:
    """Block the host until every NVSHMEM PE reaches the barrier."""

    _require_state().bindings.barrier_all()


def benchmark_barrier() -> None:
    """Synchronize local CUDA work and all PEs before a measured launch."""

    runtime = _require_state().runtime_info
    torch.cuda.synchronize(runtime.device)
    barrier()


def finalize() -> None:
    """Collectively release allocations, the CUDA module, and owned host state."""

    global _state
    if _state is None:
        return

    state = _require_state()
    torch.cuda.set_device(state.runtime_info.device)
    torch.cuda.synchronize(state.runtime_info.device)
    state.bindings.barrier_all()
    state.signal_space = None
    _load_allocator().release_allocations()
    state.bindings.barrier_all()
    status = int(_load_runtime()._nvshmem_module_finalize())
    if status != 0:
        raise RuntimeError(
            f"DAE NVSHMEM CUDA module finalization failed with status {status}"
        )
    if state.runtime_info.owns_nvshmem:
        state.host.finalize()
    state.local_comm.Free()
    _state = None


__all__ = [
    "RuntimeInfo",
    "available",
    "init",
    "is_initialized",
    "info",
    "my_pe",
    "n_pes",
    "allocate_tensor",
    "empty",
    "zeros",
    "init_signal_space",
    "is_symmetric_tensor",
    "signal",
    "wait_signal",
    "quiet",
    "barrier",
    "benchmark_barrier",
    "finalize",
]
