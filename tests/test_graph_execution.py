from __future__ import annotations

from pathlib import Path
import sys
import threading
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from dae.graph_execution import Task, TaskGraph, VDCoresTaskBackend, execute_graph
from dae.heterogeneous import OperationSpec


def task(name, inputs, fn, *, dependencies=None, devices=("cpu",), executor="torch_cuda"):
    return Task(OperationSpec(name, 1, tuple(inputs if dependencies is None else dependencies), devices),
                tuple(inputs), fn, executor)


def diamond():
    return TaskGraph([
        task("prep", ("frame", "previous"), lambda x, p, **kw: x + p, dependencies=()),
        task("left", ("prep",), lambda x, **kw: x * 2),
        task("right", ("prep",), lambda x, **kw: x + 1),
        task("control", ("left", "right"), lambda a, b, **kw: a + b),
    ])


def test_diamond_feedback_and_bounded_contexts():
    graph = diamond()
    result = execute_graph(graph, lambda lane, frame, previous: {"frame": lane + frame, "previous": previous},
                           frames=20, lanes=4, initial=0, poll_us=0)
    assert result["max_active_frames"] <= 4
    assert result["max_pending_tasks"] <= 4 * len(graph.tasks)
    assert len(result["tasks"]) == 320
    tasks = {(t["lane"], t["frame"], t["task"]): t for t in result["tasks"]}
    for lane in range(4):
        prev, end = 0, 0
        for row in sorted((r for r in result["frames"] if r["lane"] == lane), key=lambda r: r["frame"]):
            prev = 3 * (lane + row["frame"] + prev) + 1
            assert row["output"] == prev
            assert row["admitted_ns"] >= end
            end = row["end_ns"]
            for name, node in graph.tasks.items():
                trace = tasks[lane, row["frame"], name]
                assert trace["ready_ns"] <= trace["submitted_ns"] <= trace["started_ns"] <= trace["completed_ns"]
                assert all(tasks[lane, row["frame"], dep]["completed_ns"] <= trace["submitted_ns"]
                           for dep in node.operation.dependencies)


def test_sibling_tasks_really_overlap():
    barrier = threading.Barrier(2, timeout=2)
    def branch(x, **kwargs):
        barrier.wait()
        return x
    graph = TaskGraph([task("a", ("frame",), branch, dependencies=()),
                       task("b", ("frame",), branch, dependencies=()),
                       task("control", ("a", "b"), lambda a, b, **kw: a + b)], external_inputs=("frame",))
    result = execute_graph(graph, lambda *args: {"frame": 2}, frames=1, workers=2)
    assert result["frames"][0]["output"] == 4


def test_control_only_dependency_is_respected():
    graph = TaskGraph([task("a", ("frame",), lambda x, **kw: x, dependencies=()),
                       task("control", ("frame",), lambda x, **kw: x, dependencies=("a",))],
                      external_inputs=("frame",))
    r = execute_graph(graph, lambda *args: {"frame": 1}, frames=1)
    assert r["tasks"][0]["task"] == "a"


def test_graph_rejects_undeclared_inputs_and_cycles():
    with pytest.raises(ValueError, match="inputs"):
        TaskGraph([task("control", ("missing",), lambda **kw: 0, dependencies=())])
    with pytest.raises(ValueError, match="cycle"):
        TaskGraph([task("a", ("control",), lambda x, **kw: x),
                   task("control", ("a",), lambda x, **kw: x)])


def test_failure_drains_running_cpu_reader():
    done = threading.Event()
    def slow(x, **kwargs):
        time.sleep(0.03)
        done.set()
        return x
    def fail(x, **kwargs):
        raise RuntimeError("injected failure")
    graph = TaskGraph([task("a", ("frame",), slow, dependencies=()),
                       task("control", ("frame",), fail, dependencies=())], external_inputs=("frame",))
    with pytest.raises(RuntimeError, match="injected"):
        execute_graph(graph, lambda *args: {"frame": 1}, frames=1)
    assert done.is_set()


def test_timeout_drains_tasks():
    done = threading.Event()
    def slow(x, **kwargs):
        time.sleep(0.03)
        done.set()
        return x
    graph = TaskGraph([task("control", ("frame",), slow, dependencies=())], external_inputs=("frame",))
    with pytest.raises(TimeoutError):
        execute_graph(graph, lambda *args: {"frame": 1}, frames=1, timeout_s=0.01)
    assert done.is_set()


def test_periodic_latency_keeps_scheduled_releases():
    graph = TaskGraph([task("control", ("frame",), lambda x, **kw: (time.sleep(0.002), x)[1], dependencies=())],
                      external_inputs=("frame",))
    r = execute_graph(graph, lambda *args: {"frame": 1}, frames=3, period_ns=100_000)
    rows = sorted(r["frames"], key=lambda x: x["frame"])
    assert rows[1]["release_ns"] - rows[0]["release_ns"] == 100_000
    assert rows[2]["admitted_ns"] > rows[2]["release_ns"] + 1_000_000


def test_serial_baseline_has_no_overlapping_submissions():
    r = execute_graph(diamond(), lambda *args: {"frame": 1, "previous": 0}, frames=3, lanes=2, serial=True)
    assert r["max_pending_tasks"] == 1
    ordered = sorted(r["tasks"], key=lambda x: x["submitted_ns"])
    assert all(a["completed_ns"] <= b["submitted_ns"] for a, b in zip(ordered, ordered[1:]))


def test_vdcores_adapter_uses_actual_launcher_contract_without_torch_fallback():
    class Stream:
        ready = False
        def query(self):
            return self.ready
    class Launcher:
        def __init__(self):
            self.stream = Stream()
            self.launches = self.syncs = 0
        def launch_async(self):
            self.launches += 1
            return self.stream
        def synchronize(self):
            self.syncs += 1
    launcher = Launcher()
    adapter = VDCoresTaskBackend(lambda t, args: launcher, lambda launcher: 42)
    future = adapter.submit(None, ("owned buffer",))
    assert launcher.launches == 1 and not adapter.available()
    with pytest.raises(RuntimeError, match="not ready"):
        future.result()
    with pytest.raises(RuntimeError, match="in flight"):
        adapter.submit(None, ())
    launcher.stream.ready = True
    assert future.result()[0] == 42
    assert launcher.syncs == 1 and adapter.available()
    with pytest.raises(RuntimeError, match="already released"):
        future.result()


def test_vdcores_adapter_drains_partial_launch_failure():
    class Launcher:
        syncs = 0
        def launch_async(self):
            raise RuntimeError("launch failed")
        def synchronize(self):
            self.syncs += 1
    launcher = Launcher()
    backend = VDCoresTaskBackend(lambda *args: launcher, lambda launcher: 0)
    with pytest.raises(RuntimeError, match="launch failed"):
        backend.submit(None, ())
    backend.close()
    assert launcher.syncs == 1


def test_explicit_placement_rejects_illegal_device():
    scheduler = diamond().new_scheduler()
    with pytest.raises(ValueError, match="illegal"):
        scheduler.dispatch("prep", device="gpu")


def test_measured_adaptive_dispatch_and_backend_identity():
    from dae.placement import MeasuredCrossoverPolicy, PlacementSample
    class Backend:
        kind = "torch_cuda"
        identity = "test_only_not_hardware"
        calls = 0
        closed = False
        def available(self):
            return True
        def submit(self, task, args):
            self.calls += 1
            class Future:
                def done(self):
                    return True
                def result(self):
                    started = time.perf_counter_ns()
                    return task.function(*args, device="gpu"), {"started_ns": started, "finished_ns": time.perf_counter_ns()}
            return Future()
        def close(self):
            self.closed = True
    graph = TaskGraph([task("control", ("frame",), lambda x, **kw: x + 1,
                            dependencies=(), devices=("cpu", "gpu"))], external_inputs=("frame",))
    backend = Backend()
    policy = MeasuredCrossoverPolicy([PlacementSample(1, 10_000, 1)])
    r = execute_graph(graph, lambda *args: {"frame": 4}, frames=2, placement="adaptive",
                      policies={"control": policy}, gpu_backend=backend)
    assert backend.calls == 2 and backend.closed
    assert all(t["device"] == "gpu" and t["placement_estimates_ns"] for t in r["tasks"])
    with pytest.raises(ValueError, match="executor"):
        execute_graph(graph, lambda *args: {}, gpu_backend=VDCoresTaskBackend(None, None))
    backend = Backend()
    bad = MeasuredCrossoverPolicy([PlacementSample(2, 1, 2)])
    with pytest.raises(ValueError, match="outside calibration"):
        execute_graph(graph, lambda *args: {"frame": 4}, frames=1, placement="adaptive",
                      policies={"control": bad}, gpu_backend=backend)
    assert backend.closed
