"""Compile/import smoke coverage for the optional NVSHMEM runtime.

Collective behavior is verified by ``app/python/nvshmem_example.py``
inside a real multi-node allocation; it is intentionally not emulated here.
"""

from __future__ import annotations

import pytest

import dae.nvshmem as nvshmem


def test_python_api_import_is_lazy():
    assert callable(nvshmem.init)
    assert callable(nvshmem.init_signal_space)
    assert callable(nvshmem.allocate_tensor)


def test_optional_allocator_extension_import_smoke():
    runtime = pytest.importorskip("dae._nvshmem_runtime")
    dae_runtime = pytest.importorskip("dae.runtime")

    assert runtime.NVSHMEM_ENABLED is True
    assert dae_runtime.config.nvshmem_enabled is True
    assert {
        "allocate_tensor",
        "allocation_count",
        "get_signal_space",
        "init_signal_space",
        "is_symmetric_tensor",
        "release_allocations",
        "signal_address",
    } <= set(dir(runtime))
