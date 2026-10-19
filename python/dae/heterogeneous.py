"""Backend-independent dependency scheduling for heterogeneous operations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .placement import Device, MeasuredCrossoverPolicy, PlacementDecision


class OperationState(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETE = "complete"


@dataclass(frozen=True)
class OperationSpec:
    """A dependency-graph operation and its legal placements."""

    name: str
    work_items: int
    dependencies: tuple[str, ...] = ()
    devices: tuple[Device, ...] = ("cpu", "gpu")
    deadline_ns: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("operation name cannot be empty")
        if self.work_items <= 0:
            raise ValueError("work_items must be positive")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError(f"operation {self.name!r} repeats a dependency")
        if self.name in self.dependencies:
            raise ValueError(f"operation {self.name!r} cannot depend on itself")
        if not self.devices:
            raise ValueError(f"operation {self.name!r} needs at least one device")
        if len(set(self.devices)) != len(self.devices):
            raise ValueError(f"operation {self.name!r} repeats a device")
        if any(device not in ("cpu", "gpu") for device in self.devices):
            raise ValueError(f"operation {self.name!r} has an unknown device")
        if self.deadline_ns is not None and self.deadline_ns <= 0:
            raise ValueError("deadline_ns must be positive")


@dataclass(frozen=True)
class DispatchRecord:
    """One transition from ready to running."""

    operation: str
    device: Device
    reason: str
    placement: PlacementDecision | None = None


class HeterogeneousScheduler:
    """Track a dependency DAG and select legal CPU/GPU placements.

    This class owns dependency and state transitions only. Backends execute the
    returned dispatch records and call ``complete`` when their work finishes.
    """

    def __init__(
        self,
        operations: Iterable[OperationSpec] = (),
        *,
        placement_policy: MeasuredCrossoverPolicy | None = None,
    ) -> None:
        self._operations: dict[str, OperationSpec] = {}
        self._states: dict[str, OperationState] = {}
        self._sealed = False
        self.placement_policy = placement_policy
        for operation in operations:
            self.add(operation)

    def add(self, operation: OperationSpec) -> None:
        if self._sealed:
            raise RuntimeError("cannot add operations after the graph is sealed")
        if operation.name in self._operations:
            raise ValueError(f"duplicate operation {operation.name!r}")
        self._operations[operation.name] = operation
        self._states[operation.name] = OperationState.PENDING

    def seal(self) -> None:
        """Validate dependencies and cycles, then expose initial ready work."""

        unknown = {
            dependency
            for operation in self._operations.values()
            for dependency in operation.dependencies
            if dependency not in self._operations
        }
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown dependencies: {names}")

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visiting:
                raise ValueError(f"dependency cycle includes {name!r}")
            if name in visited:
                return
            visiting.add(name)
            for dependency in self._operations[name].dependencies:
                visit(dependency)
            visiting.remove(name)
            visited.add(name)

        for name in self._operations:
            visit(name)

        self._sealed = True
        self._refresh_ready()

    def _require_sealed(self) -> None:
        if not self._sealed:
            raise RuntimeError("seal the graph before scheduling")

    def _refresh_ready(self) -> None:
        for name, operation in self._operations.items():
            if self._states[name] != OperationState.PENDING:
                continue
            if all(
                self._states[dependency] == OperationState.COMPLETE
                for dependency in operation.dependencies
            ):
                self._states[name] = OperationState.READY

    def ready(self) -> tuple[OperationSpec, ...]:
        self._require_sealed()
        return tuple(
            operation
            for name, operation in self._operations.items()
            if self._states[name] == OperationState.READY
        )

    def state(self, name: str) -> OperationState:
        self._require_sealed()
        try:
            return self._states[name]
        except KeyError as error:
            raise KeyError(f"unknown operation {name!r}") from error

    def dispatch(
        self,
        name: str,
        *,
        cpu_queue_ns: int = 0,
        gpu_queue_ns: int = 0,
        device: Device | None = None,
    ) -> DispatchRecord:
        self._require_sealed()
        if self.state(name) != OperationState.READY:
            raise RuntimeError(f"operation {name!r} is not ready")
        operation = self._operations[name]

        if device is not None:
            if device not in operation.devices:
                raise ValueError(f"illegal device {device!r} for {name!r}")
            record = DispatchRecord(name, device, "explicit executor placement")
        elif len(operation.devices) == 1:
            device = operation.devices[0]
            record = DispatchRecord(
                operation=name,
                device=device,
                reason=f"operation is fixed to {device}",
            )
        else:
            if self.placement_policy is None:
                raise RuntimeError(
                    f"operation {name!r} is flexible but no placement policy is set"
                )
            placement = self.placement_policy.choose(
                operation.work_items,
                cpu_queue_ns=cpu_queue_ns,
                gpu_queue_ns=gpu_queue_ns,
                deadline_ns=operation.deadline_ns,
            )
            if placement.device not in operation.devices:
                raise RuntimeError(
                    f"policy chose illegal device {placement.device!r} for {name!r}"
                )
            record = DispatchRecord(
                operation=name,
                device=placement.device,
                reason=placement.reason,
                placement=placement,
            )

        self._states[name] = OperationState.RUNNING
        return record

    def complete(self, name: str) -> None:
        self._require_sealed()
        if self.state(name) != OperationState.RUNNING:
            raise RuntimeError(f"operation {name!r} is not running")
        self._states[name] = OperationState.COMPLETE
        self._refresh_ready()

    @property
    def is_complete(self) -> bool:
        self._require_sealed()
        return all(state == OperationState.COMPLETE for state in self._states.values())
