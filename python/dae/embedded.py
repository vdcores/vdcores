"""Ownership bridge for the native fixed-graph embedded-loop prototype.

No Torch/CUDA import. Build with ``python tools/build_embedded.py`` first.
Instances are driven by one Python service thread; native CPU workers run in
parallel. Callers must drain external work before releasing their buffer owners.
"""
from __future__ import annotations

import ctypes as C
from pathlib import Path
import sys


class Request(C.Structure):
    _fields_ = [("lane", C.c_uint64), ("frame", C.c_uint64)]


class Trace(C.Structure):
    _fields_ = (
        [(name, C.c_uint64) for name in ("lane", "frame")]
        + [(name, C.c_double) for name in ("previous_action", "feature", "action")]
        + [(name, C.c_int64) for name in (
            "ready_ns", "prep_start_ns", "prep_end_ns", "taken_ns",
            "external_end_ns", "control_start_ns", "end_ns")]
        + [(name, C.c_uint64) for name in ("prep_worker", "control_worker")]
    )

    def as_dict(self) -> dict:
        return {name: getattr(self, name) for name, _ in self._fields_}


def positive(value: int, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")
    return value


class NativeLoop:
    """Fixed sensor -> external feature -> controller graph, repeated per lane.

    ``inputs`` are retained double arrays or contiguous CPU float64 tensors.
    Their ownership transfers to native preprocessing until ``take``; completion
    returns it to native code. Only one external request per lane can be live.
    Do not access a lane's buffer outside its outstanding request.
    """
    def __init__(self, *, work_items: int, frames: int, lanes: int = 1,
                 workers: int = 2, library: str | Path | None = None,
                 inputs: list | None = None):
        positive(work_items, "work_items", 16_777_216)
        positive(frames, "frames", 1_000_000)
        positive(lanes, "lanes", 256)
        positive(workers, "workers", 256)
        if frames * lanes > 1_000_000:
            raise ValueError("at most 1,000,000 frame records per run")
        if work_items * lanes > 33_554_432:
            raise ValueError("input buffers exceed the 256 MiB prototype limit")
        if library is None:
            library = Path(__file__).resolve().parents[2] / "build" / (
                "libembedded_loop.dylib" if sys.platform == "darwin" else "libembedded_loop.so")
        self.lib = C.CDLL(str(library))
        self.lib.el_abi_version.restype = C.c_int
        if self.lib.el_abi_version() != 3:
            raise RuntimeError("native loop ABI mismatch; rebuild the library")
        self.lib.el_error.restype = C.c_char_p
        pointer = C.POINTER(C.c_double)
        signatures = {
            "el_create": ([C.POINTER(pointer), C.c_uint64, C.c_uint64, C.c_uint64, C.c_uint64], C.c_void_p),
            "el_destroy": ([C.c_void_p], None),
            "el_cancel": ([C.c_void_p], None),
            "el_take": ([C.c_void_p, C.POINTER(Request), C.c_int], C.c_int),
            "el_complete": ([C.c_void_p, C.c_uint64, C.c_uint64, C.c_double], C.c_int),
            "el_dispatch_cpu": ([C.c_void_p, C.c_uint64, C.c_uint64], C.c_int),
            "el_cpu_load": ([C.c_void_p], C.c_int),
            "el_cpu_backlog": ([C.c_void_p, C.POINTER(C.c_uint64)], C.c_int),
            "el_wait": ([C.c_void_p, C.c_int], C.c_int),
            "el_traces": ([C.c_void_p, C.POINTER(Trace)], C.c_int),
            "el_worker_counts": ([C.c_void_p, C.POINTER(C.c_uint64)], C.c_int),
            "el_reduce": ([pointer, C.c_uint64], C.c_double),
        }
        for name, (arguments, result) in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = arguments, result
        self.work_items, self.frames, self.lanes, self.workers = work_items, frames, lanes, workers
        self.inputs = tuple(inputs) if inputs is not None else tuple(
            (C.c_double * work_items)() for _ in range(lanes))
        if len(self.inputs) != lanes:
            raise ValueError("one input buffer is required per lane")
        pointers = []
        addresses = set()
        for buffer in self.inputs:
            if isinstance(buffer, C.Array):
                if buffer._type_ is not C.c_double or len(buffer) != work_items:
                    raise ValueError("expected a work_items-length ctypes double array")
                address = C.addressof(buffer)
            else:
                # Validate tensor without importing Torch on the portable path.
                if (str(buffer.dtype) != "torch.float64" or buffer.device.type != "cpu"
                        or not buffer.is_contiguous() or buffer.numel() != work_items):
                    raise ValueError("expected a contiguous CPU float64 tensor of work_items length")
                address = buffer.data_ptr()
            if any(abs(address - old) < work_items * C.sizeof(C.c_double) for old in addresses):
                raise ValueError("lane input buffers must not overlap")
            addresses.add(address)
            pointers.append(C.cast(address, pointer))
        self._pointers = (pointer * lanes)(*pointers)
        self._leases = set()
        self._handle = self.lib.el_create(self._pointers, work_items, frames, lanes, workers)
        if not self._handle:
            raise RuntimeError(self.lib.el_error().decode())

    def _check(self, result: int) -> int:
        if result < 0:
            raise RuntimeError(self.lib.el_error().decode())
        return result

    def _live(self):
        if not self._handle:
            raise RuntimeError("loop is closed")
        return self._handle

    @staticmethod
    def _timeout(value: int) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 60_000:
            raise ValueError("timeout_ms must be an integer in [0, 60000]")
        return value

    def take(self, timeout_ms: int = 0) -> Request | None:
        request = Request()
        result = self._check(self.lib.el_take(self._live(), C.byref(request), self._timeout(timeout_ms)))
        if result:
            self._leases.add((request.lane, request.frame))
        return request if result else None

    def complete(self, request: Request, feature: float) -> None:
        self._check(self.lib.el_complete(self._live(), request.lane, request.frame, feature))
        self._leases.discard((request.lane, request.frame))

    def reference_feature(self, request: Request) -> float:
        self._live()
        if request.lane >= self.lanes:
            raise ValueError("unknown lane")
        if (request.lane, request.frame) not in self._leases:
            raise RuntimeError("no outstanding buffer lease for this request")
        return self.lib.el_reduce(self._pointers[request.lane], self.work_items)

    def dispatch_cpu(self, request: Request) -> None:
        """Return the input lease to a persistent native feature worker."""
        self._check(self.lib.el_dispatch_cpu(self._live(), request.lane, request.frame))
        self._leases.discard((request.lane, request.frame))

    def cpu_load(self) -> int:
        """Queued plus running native feature tasks, sampled under the lock."""
        return self._check(self.lib.el_cpu_load(self._live()))

    def cpu_backlog(self) -> dict[str, int]:
        """Queued plus running work in every native CPU stage."""
        output = (C.c_uint64 * 3)()
        self._check(self.lib.el_cpu_backlog(self._live(), output))
        return dict(zip(("preprocess", "feature", "control"), output))

    def wait(self, timeout_ms: int = 0) -> bool:
        return bool(self._check(self.lib.el_wait(self._live(), self._timeout(timeout_ms))))

    def traces(self) -> list[dict]:
        output = (Trace * (self.frames * self.lanes))()
        self._check(self.lib.el_traces(self._live(), output))
        return [trace.as_dict() for trace in output]

    def worker_counts(self) -> list[int]:
        output = (C.c_uint64 * self.workers)()
        self._check(self.lib.el_worker_counts(self._live(), output))
        return list(output)

    def cancel(self) -> None:
        self.lib.el_cancel(self._live())

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self.lib.el_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        # Join writers before owned buffers are garbage-collected, including
        # forgotten context-manager cleanup. No external GPU work is owned here.
        self.close()
