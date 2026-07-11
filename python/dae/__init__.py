from . import launcher
from . import util


def __getattr__(name):
    if name == "nvshmem":
        from importlib import import_module

        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["launcher", "util", "nvshmem"]
