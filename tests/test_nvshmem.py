"""Focused API smoke coverage for the optional NVSHMEM runtime.

Collective behavior is verified by ``app/python/nvshmem/example.py``
inside a real multi-node allocation; it is intentionally not emulated here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import dae.nvshmem as nvshmem


def test_python_api_surface_is_minimal():
    assert callable(nvshmem.init)
    assert callable(nvshmem.init_signal_space)
    assert callable(nvshmem.allocate_tensor)
    assert callable(nvshmem.benchmark_barrier)
    assert not hasattr(nvshmem, "Launcher")
    assert not hasattr(nvshmem, "get_signal_space")


def test_optional_allocator_extension_import_smoke():
    runtime = pytest.importorskip("dae._nvshmem_runtime")
    dae_runtime = pytest.importorskip("dae.runtime")

    assert runtime.NVSHMEM_ENABLED is True
    assert dae_runtime.config.nvshmem_enabled is True
    assert not hasattr(dae_runtime, "nvshmem_module_init")
    assert not hasattr(dae_runtime, "nvshmem_module_finalize")
    assert callable(dae_runtime._nvshmem_module_init)
    assert callable(dae_runtime._nvshmem_module_finalize)
    assert not hasattr(dae_runtime, "launch_memory_pool_control")
    assert dae_runtime.comm_opcode.COMM_MEMORY_POOL_RUN == 6
    assert dae_runtime.comm_opcode.COMM_POOL_SLICE_EXCHANGE == 8
    assert dae_runtime.config.max_comm_insts == 32
    public = {name for name in dir(runtime) if not name.startswith("_")}
    assert public == {
        "NVSHMEM_ENABLED",
        "allocate_tensor",
        "is_symmetric_tensor",
        "release_allocations",
    }


def test_signal_space_uses_the_symmetric_tensor_factory(monkeypatch):
    state = SimpleNamespace(
        signal_space=None,
        runtime_info=SimpleNamespace(device=3),
    )
    calls = []
    signals = torch.zeros(4, dtype=torch.uint64)

    def fake_zeros(*size, **kwargs):
        calls.append((size, kwargs))
        return signals

    monkeypatch.setattr(nvshmem, "_require_state", lambda: state)
    monkeypatch.setattr(nvshmem, "zeros", fake_zeros)
    monkeypatch.setattr(nvshmem, "barrier", lambda: calls.append(("barrier",)))

    result = nvshmem.init_signal_space(4)

    assert result is signals
    assert state.signal_space is signals
    assert calls == [
        ((4,), {"dtype": torch.uint64, "device": 3}),
        ("barrier",),
    ]
    assert nvshmem._signal_address(2) == signals.data_ptr() + 16

    assert nvshmem.init_signal_space(4) is signals
    assert calls[-1] == ("barrier",)
    with pytest.raises(ValueError, match="already initialized"):
        nvshmem.init_signal_space(3)


def test_available_handles_a_missing_optional_package(monkeypatch):
    def missing(_name):
        raise ModuleNotFoundError

    monkeypatch.setattr(nvshmem.importlib_util, "find_spec", missing)
    assert nvshmem.available() is False
