"""Run and audit the bounded feedback loop without a Python dependency scheduler."""
from __future__ import annotations

import math
import time

from .embedded import NativeLoop


def run_loop(backend, *, work_items: int, frames: int, lanes: int,
             workers: int = 2, timeout_s: float = 60, poll_us: float = 50,
             library=None) -> dict:
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be finite and positive")
    if not math.isfinite(poll_us) or poll_us < 0:
        raise ValueError("poll_us must be finite and nonnegative")
    loop = None
    started = time.perf_counter_ns()
    try:
        loop = NativeLoop(work_items=work_items, frames=frames, lanes=lanes,
                          workers=workers, library=library, inputs=backend.inputs)
        deadline = time.monotonic() + timeout_s
        max_pending = 0
        while not loop.wait(0):
            if time.monotonic() >= deadline:
                raise TimeoutError("embedded loop exceeded its run timeout")
            # At most one outstanding external request per lane. Limit draining
            # to lane count so a fast backend cannot starve event polling.
            for _ in range(lanes):
                request = loop.take(0)
                if request is None:
                    break
                backend.submit(loop, request)
                max_pending = max(max_pending, backend.pending)
            backend.poll(loop)
            if not loop.wait(0):
                time.sleep(poll_us / 1e6)
        result = {"traces": loop.traces(), "worker_task_counts": loop.worker_counts(),
                  "max_external_in_flight": max_pending,
                  "host_run_ns_including_pool_startup": time.perf_counter_ns() - started,
                  "device_timings": backend.timings,
                  "placement_decisions": getattr(backend, "decisions", []),
                  "native_abi_version": 3}
        return result
    finally:
        # Join native writers before draining external readers, retaining loop
        # and backend buffer owners until both are quiescent. Timeout is terminal.
        if loop is not None:
            loop.close()
        backend.close()


def verify_feedback(traces: list[dict], *, work_items: int, frames: int,
                    lanes: int, tolerance: float = 1e-10) -> dict:
    """Independent closed-form signal reduction, checked for every frame.

    Avoid using native preprocess/reduce to generate the oracle. Sum the period
    17 base signal analytically; advance reference feedback using reference
    actions, never the implementation's action.
    """
    if len(traces) != frames * lanes:
        raise AssertionError("missing or duplicated frame records")
    by_key = {(t["lane"], t["frame"]): t for t in traces}
    if len(by_key) != len(traces):
        raise AssertionError("duplicate frame identity")
    cycles, remainder = divmod(work_items, 17)
    bases = [(i - 8) / 16 for i in range(17)]
    sum_base = cycles * math.fsum(bases) + math.fsum(bases[:remainder])
    sum_square = cycles * math.fsum(b * b for b in bases) + math.fsum(
        b * b for b in bases[:remainder])
    max_error = 0.0
    for lane in range(lanes):
        action = 0.0
        previous_end = None
        for frame in range(frames):
            t = by_key[(lane, frame)]
            offset = action * 0.25 + (frame % 7) * 0.001 + lane * 0.0001
            expected_feature = (sum_square + 2 * offset * sum_base + work_items * offset * offset) / work_items
            expected_action = min(1.0, max(-1.0, 0.9 * action - 0.5 * expected_feature))
            for actual, expected in ((t["previous_action"], action),
                                     (t["feature"], expected_feature), (t["action"], expected_action)):
                error = abs(actual - expected)
                if not math.isfinite(actual) or error > tolerance:
                    raise AssertionError(f"feedback mismatch at lane={lane}, frame={frame}: {actual} vs {expected}")
                max_error = max(max_error, error)
            times = [t[name] for name in (
                "ready_ns", "prep_start_ns", "prep_end_ns", "taken_ns",
                "external_end_ns", "control_start_ns", "end_ns")]
            if any(a > b for a, b in zip(times, times[1:])):
                raise AssertionError("stage timestamps violate dependency order")
            if previous_end is not None and t["ready_ns"] != previous_end:
                raise AssertionError("next frame was not released by previous control completion")
            previous_end, action = t["end_ns"], expected_action
    return {"passed": True, "checked_frames": len(traces), "max_abs_error": max_error,
            "absolute_tolerance": tolerance, "oracle": "independent_closed_form_feedback"}


def summarize(values: list[int]) -> dict:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no measured frames")
    def percentile(q):
        return ordered[max(0, math.ceil(q * len(ordered)) - 1)]
    return {"count": len(ordered), "min_ns": ordered[0], "median_ns": percentile(0.5),
            "p95_ns": percentile(0.95), "p99_ns": percentile(0.99), "max_ns": ordered[-1]}


def measurements(result: dict, *, warmup: int, lanes: int, deadline_ns: int | None) -> dict:
    traces = [t for t in result["traces"] if t["frame"] >= warmup]
    spans = {
        "cpu_queue_ns": ("ready_ns", "prep_start_ns"),
        "preprocess_ns": ("prep_start_ns", "prep_end_ns"),
        "external_service_queue_ns": ("prep_end_ns", "taken_ns"),
        "external_observed_ns": ("taken_ns", "external_end_ns"),
        "control_queue_ns": ("external_end_ns", "control_start_ns"),
        "control_ns": ("control_start_ns", "end_ns"),
        "end_to_end_ns": ("ready_ns", "end_ns"),
    }
    per_lane = []
    for lane in range(lanes):
        rows = [t for t in traces if t["lane"] == lane]
        latencies = [t["end_ns"] - t["ready_ns"] for t in rows]
        per_lane.append({"lane": lane, "latency": summarize(latencies),
                         "deadline_misses": sum(n > deadline_ns for n in latencies) if deadline_ns else None})
    first = min(t["ready_ns"] for t in traces)
    last = max(t["end_ns"] for t in traces)
    return {
        "stages": {name: summarize([t[end] - t[start] for t in traces])
                   for name, (start, end) in spans.items()},
        "per_lane": per_lane,
        "deadline_ns": deadline_ns,
        "deadline_misses": sum(row["deadline_misses"] for row in per_lane) if deadline_ns else None,
        "throughput_frames_per_second": len(traces) * 1e9 / max(1, last - first),
        "throughput_window_ns": last - first,
        "device_stages": {name: summarize([t[name] for t in result["device_timings"]
                                           if t["frame"] >= warmup])
                          for name in result["device_timings"][0] if name.endswith("_ns")}
                         if result["device_timings"] else {},
    }
