"""Measured CPU/GPU placement decisions for heterogeneous VDCores work."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import math
from typing import Iterable, Literal, Mapping


Device = Literal["cpu", "gpu"]


@dataclass(frozen=True)
class PlacementSample:
    """One measured CPU/GPU latency pair for a work size."""

    work_items: int
    cpu_latency_ns: int
    gpu_latency_ns: int

    def __post_init__(self) -> None:
        if self.work_items <= 0:
            raise ValueError("work_items must be positive")
        if self.cpu_latency_ns <= 0 or self.gpu_latency_ns <= 0:
            raise ValueError("measured latencies must be positive")


@dataclass(frozen=True)
class PlacementDecision:
    """A placement choice and the latency estimates behind it."""

    device: Device
    work_items: int
    cpu_estimate_ns: int
    gpu_estimate_ns: int
    cpu_total_ns: int
    gpu_total_ns: int
    reason: str
    clamped_to_measured_range: bool


class MeasuredCrossoverPolicy:
    """Choose CPU or GPU from measured latency samples.

    Latencies are interpolated between measured work sizes in log2(work-items)
    space. Requests outside the measured range use the nearest endpoint instead
    of extrapolating an unsupported trend. Queue-delay estimates are added after
    interpolation, so a busy device can lose even when its isolated kernel is
    faster.
    """

    def __init__(
        self,
        samples: Iterable[PlacementSample],
        *,
        preferred_device: Device = "cpu",
        switch_margin: float = 0.0,
    ) -> None:
        ordered = sorted(samples, key=lambda sample: sample.work_items)
        if not ordered:
            raise ValueError("at least one placement sample is required")
        if len({sample.work_items for sample in ordered}) != len(ordered):
            raise ValueError("work_items values must be unique")
        if preferred_device not in ("cpu", "gpu"):
            raise ValueError("preferred_device must be 'cpu' or 'gpu'")
        if not 0.0 <= switch_margin < 1.0:
            raise ValueError("switch_margin must be in [0, 1)")

        self._samples = tuple(ordered)
        self._sizes = tuple(sample.work_items for sample in ordered)
        self.preferred_device = preferred_device
        self.switch_margin = switch_margin

    @classmethod
    def from_benchmark(
        cls,
        result: Mapping[str, object],
        *,
        metric: str = "end_to_end_ns",
        preferred_device: Device = "cpu",
        switch_margin: float = 0.0,
    ) -> "MeasuredCrossoverPolicy":
        """Build a policy from ``check_attention_placement.py`` JSON output."""

        rows = result.get("rows")
        if not isinstance(rows, list):
            raise ValueError("benchmark result must contain a rows list")

        samples = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("each benchmark row must be an object")
            try:
                cpu_summary = row["cpu"]
                gpu_summary = row["gpu"]
                if not isinstance(cpu_summary, Mapping) or not isinstance(
                    gpu_summary, Mapping
                ):
                    raise TypeError
                samples.append(
                    PlacementSample(
                        work_items=int(row["work_items"]),
                        cpu_latency_ns=int(cpu_summary[metric]["median_ns"]),
                        gpu_latency_ns=int(gpu_summary[metric]["median_ns"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"benchmark row is missing a {metric} median"
                ) from error

        return cls(
            samples,
            preferred_device=preferred_device,
            switch_margin=switch_margin,
        )

    @property
    def measured_range(self) -> tuple[int, int]:
        return self._sizes[0], self._sizes[-1]

    def _estimate_pair(self, work_items: int) -> tuple[int, int, bool]:
        if work_items <= 0:
            raise ValueError("work_items must be positive")

        position = bisect_left(self._sizes, work_items)
        if position == 0:
            sample = self._samples[0]
            return (
                sample.cpu_latency_ns,
                sample.gpu_latency_ns,
                work_items != sample.work_items,
            )
        if position == len(self._samples):
            sample = self._samples[-1]
            return sample.cpu_latency_ns, sample.gpu_latency_ns, True
        if self._sizes[position] == work_items:
            sample = self._samples[position]
            return sample.cpu_latency_ns, sample.gpu_latency_ns, False

        lower = self._samples[position - 1]
        upper = self._samples[position]
        log_lower = math.log2(lower.work_items)
        fraction = (math.log2(work_items) - log_lower) / (
            math.log2(upper.work_items) - log_lower
        )

        def interpolate(low: int, high: int) -> int:
            return round(low + fraction * (high - low))

        return (
            interpolate(lower.cpu_latency_ns, upper.cpu_latency_ns),
            interpolate(lower.gpu_latency_ns, upper.gpu_latency_ns),
            False,
        )

    def choose(
        self,
        work_items: int,
        *,
        cpu_queue_ns: int = 0,
        gpu_queue_ns: int = 0,
        deadline_ns: int | None = None,
    ) -> PlacementDecision:
        """Choose a device using measured compute latency plus queue delay."""

        if cpu_queue_ns < 0 or gpu_queue_ns < 0:
            raise ValueError("queue delays cannot be negative")
        if deadline_ns is not None and deadline_ns <= 0:
            raise ValueError("deadline_ns must be positive")

        cpu_estimate, gpu_estimate, clamped = self._estimate_pair(work_items)
        cpu_total = cpu_estimate + cpu_queue_ns
        gpu_total = gpu_estimate + gpu_queue_ns

        if deadline_ns is not None:
            cpu_meets = cpu_total <= deadline_ns
            gpu_meets = gpu_total <= deadline_ns
            if cpu_meets != gpu_meets:
                device: Device = "cpu" if cpu_meets else "gpu"
                return PlacementDecision(
                    device=device,
                    work_items=work_items,
                    cpu_estimate_ns=cpu_estimate,
                    gpu_estimate_ns=gpu_estimate,
                    cpu_total_ns=cpu_total,
                    gpu_total_ns=gpu_total,
                    reason=f"only {device} meets the deadline",
                    clamped_to_measured_range=clamped,
                )

        preferred_total = cpu_total if self.preferred_device == "cpu" else gpu_total
        alternate: Device = "gpu" if self.preferred_device == "cpu" else "cpu"
        alternate_total = gpu_total if alternate == "gpu" else cpu_total
        required_total = preferred_total * (1.0 - self.switch_margin)

        if alternate_total < required_total:
            device = alternate
            reason = (
                f"{alternate} is faster after queue delay by more than the "
                f"{self.switch_margin:.1%} switch margin"
            )
        else:
            device = self.preferred_device
            reason = (
                f"kept preferred {device}; alternate did not clear the "
                f"{self.switch_margin:.1%} switch margin"
            )

        return PlacementDecision(
            device=device,
            work_items=work_items,
            cpu_estimate_ns=cpu_estimate,
            gpu_estimate_ns=gpu_estimate,
            cpu_total_ns=cpu_total,
            gpu_total_ns=gpu_total,
            reason=reason,
            clamped_to_measured_range=clamped,
        )
