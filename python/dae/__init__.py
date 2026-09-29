"""VDCores Python package.

The CUDA-backed launcher remains available as ``dae.launcher``, but it is
loaded lazily so CPU-only planning and placement tools can run without a built
``dae.runtime`` extension.
"""

from __future__ import annotations

from importlib import import_module

__all__ = ["launcher", "util"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{name}", __name__)
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])
