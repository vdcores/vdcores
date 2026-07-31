"""Compile-time envelopes and per-block runtime VDCores assemblies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import struct


CORE_CONFIG_BYTES = 8
_CORE_CONFIG_STRUCT = struct.Struct("<8B")


class CoreKind(IntEnum):
    COMPUTE_MEMORY = 0
    POOL = 1
    INACTIVE = 2


class KernelVariant(IntEnum):
    AUTO = 0
    COMPUTE_MEMORY = 1
    COMPUTE_MEMORY_ONE_LOAD = 2
    RUNTIME = 3
    POOL = 4
    RUNTIME_COMMUNICATION = 5


_KERNEL_VARIANT_NAMES = {
    "auto": KernelVariant.AUTO,
    "compute_memory": KernelVariant.COMPUTE_MEMORY,
    "default": KernelVariant.COMPUTE_MEMORY,
    "compute_memory_one_load": KernelVariant.COMPUTE_MEMORY_ONE_LOAD,
    "one_load": KernelVariant.COMPUTE_MEMORY_ONE_LOAD,
    "runtime": KernelVariant.RUNTIME,
    "pool_slice": KernelVariant.POOL,
    "pool": KernelVariant.POOL,
    "runtime_communication": KernelVariant.RUNTIME_COMMUNICATION,
    "runtime_comm": KernelVariant.RUNTIME_COMMUNICATION,
}


def parse_kernel_variant(value: KernelVariant | str | int) -> KernelVariant:
    if isinstance(value, str):
        try:
            return _KERNEL_VARIANT_NAMES[value.strip().lower()]
        except KeyError as error:
            names = ", ".join(sorted(_KERNEL_VARIANT_NAMES))
            raise ValueError(
                f"unknown VDCores kernel variant {value!r}; expected one of {names}"
            ) from error
    try:
        return KernelVariant(value)
    except ValueError as error:
        raise ValueError(f"unknown VDCores kernel variant {value!r}") from error


@dataclass(frozen=True)
class CoreConfig:
    """Logical virtual-core roles assigned to one CUDA block.

    Runtime-selectable configurations execute inside the maximum compiled
    envelope. Use a fixed KernelVariant when every block has the same shape to
    physically reduce blockDim and remove unused operator code.
    """

    kind: CoreKind
    compute_warps: int
    load_warps: int
    communication_warps: int
    pool_warps: int = 0
    flags: int = 0

    @classmethod
    def compute_memory(
        cls, *, load_warps: int = 2, communication_warps: int = 0
    ) -> "CoreConfig":
        if load_warps not in (1, 2):
            raise ValueError("compute+memory cores support one or two load warps")
        if communication_warps not in (0, 1):
            raise ValueError(
                "compute+memory cores support zero or one communication warp"
            )
        return cls(
            CoreKind.COMPUTE_MEMORY,
            compute_warps=4,
            load_warps=load_warps,
            communication_warps=communication_warps,
            pool_warps=0,
        )

    @classmethod
    def pool(cls, *, pool_warps: int = 0) -> "CoreConfig":
        if pool_warps < 0:
            raise ValueError("pool_warps must be non-negative")
        return cls(
            CoreKind.POOL,
            compute_warps=0,
            load_warps=0,
            communication_warps=0,
            pool_warps=pool_warps,
        )

    @classmethod
    def inactive(cls) -> "CoreConfig":
        return cls(CoreKind.INACTIVE, 0, 0, 0, 0)

    def validate(self, *, runtime_core_warps: int = 8) -> None:
        fields = (
            self.compute_warps,
            self.load_warps,
            self.communication_warps,
            self.pool_warps,
            self.flags,
        )
        if any(not 0 <= int(value) < 256 for value in fields):
            raise ValueError("core configuration fields must fit in uint8")
        if self.kind == CoreKind.COMPUTE_MEMORY:
            if self.compute_warps != 4:
                raise ValueError("compute operators require one four-warp warpgroup")
            if self.load_warps not in (1, 2):
                raise ValueError("compute+memory cores need one or two load warps")
            if self.communication_warps not in (0, 1):
                raise ValueError(
                    "compute+memory cores support at most one communication stream"
                )
            if self.pool_warps != 0:
                raise ValueError("compute+memory cores cannot contain pool warps")
        elif self.kind == CoreKind.POOL:
            if (
                self.compute_warps != 0
                or self.load_warps != 0
                or self.communication_warps != 0
            ):
                raise ValueError(
                    "PoolInst cores cannot contain compute/memory/CommInst warps"
                )
            if self.pool_warps not in (0, runtime_core_warps):
                raise ValueError(
                    "a runtime PoolInst core must use the complete physical "
                    f"envelope ({runtime_core_warps} warps); zero selects it"
                )
        elif self.kind == CoreKind.INACTIVE:
            if (
                self.compute_warps
                or self.load_warps
                or self.communication_warps
                or self.pool_warps
            ):
                raise ValueError("inactive cores cannot contain active warps")
        else:
            raise ValueError(f"unknown core kind {self.kind!r}")

    def pack(self, *, runtime_core_warps: int = 8) -> bytes:
        self.validate(runtime_core_warps=runtime_core_warps)
        return _CORE_CONFIG_STRUCT.pack(
            int(self.kind),
            self.compute_warps,
            self.load_warps,
            self.communication_warps,
            self.pool_warps,
            self.flags,
            0,
            0,
        )


__all__ = [
    "CORE_CONFIG_BYTES",
    "CoreConfig",
    "CoreKind",
    "KernelVariant",
    "parse_kernel_variant",
]
