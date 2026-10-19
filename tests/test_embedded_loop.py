from __future__ import annotations

import ctypes as C
import math
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))
from build_embedded import build
from dae.embedded import NativeLoop, Request
from dae.embedded_backends import CpuBackend, CudaEventBackend, NativeCpuBackend, AdaptiveBackend, CudaGraphBackend
from dae.embedded_experiment import measurements, run_loop, summarize, verify_feedback


@pytest.fixture(scope="session")
def library(tmp_path_factory):
    return build(tmp_path_factory.mktemp("native") / "loop.so")


def test_real_feedback_and_worker_reuse(library):
    result = run_loop(CpuBackend(257, 4), work_items=257, frames=100,
                      lanes=4, workers=2, poll_us=0, library=library)
    assert verify_feedback(result["traces"], work_items=257, frames=100, lanes=4)["passed"]
    assert sum(result["worker_task_counts"]) == 800
    assert len(result["worker_task_counts"]) == 2
    assert max(result["worker_task_counts"]) > 2
    assert result["traces"][1]["previous_action"] != 0
    summary = measurements(result, warmup=10, lanes=4, deadline_ns=1)
    assert summary["stages"]["end_to_end_ns"]["count"] == 360
    assert summary["deadline_misses"] == 360
    assert len(summary["per_lane"]) == 4


def test_backpressure_and_independent_out_of_order_completion(library):
    with NativeLoop(work_items=64, frames=2, lanes=2, workers=2, library=library) as loop:
        first, second = loop.take(1000), loop.take(1000)
        assert first is not None and second is not None
        assert first.lane != second.lane
        first_snapshot = list(loop.inputs[first.lane])
        assert loop.take(10) is None
        loop.complete(second, loop.reference_feature(second))
        next_second = loop.take(1000)
        assert next_second.lane == second.lane and next_second.frame == 1
        assert list(loop.inputs[first.lane]) == first_snapshot
        loop.complete(next_second, loop.reference_feature(next_second))
        assert not loop.wait(10)
        loop.complete(first, loop.reference_feature(first))
        next_first = loop.take(1000)
        loop.complete(next_first, loop.reference_feature(next_first))
        assert loop.wait(1000)
        verify_feedback(loop.traces(), work_items=64, frames=2, lanes=2)


def test_stale_duplicate_and_unissued_tokens(library):
    with NativeLoop(work_items=17, frames=2, library=library) as loop:
        with pytest.raises(RuntimeError, match="token"):
            loop.complete(Request(0, 0), 0)
        first = loop.take(1000)
        with pytest.raises(RuntimeError, match="unknown lane"):
            loop.complete(Request(99, 0), 0)
        loop.complete(first, loop.reference_feature(first))
        with pytest.raises(RuntimeError, match="lease"):
            loop.reference_feature(first)
        with pytest.raises(RuntimeError, match="token"):
            loop.complete(first, 0)
        second = loop.take(1000)
        assert second.frame == 1
        with pytest.raises(RuntimeError, match="token"):
            loop.complete(first, 0)
        loop.complete(second, loop.reference_feature(second))
        assert loop.wait(1000)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_result_does_not_release_controller(library, value):
    with NativeLoop(work_items=64, frames=1, library=library) as loop:
        request = loop.take(1000)
        with pytest.raises(RuntimeError, match="nonfinite"):
            loop.complete(request, value)
        assert not loop.wait(1)
        loop.complete(request, loop.reference_feature(request))
        assert loop.wait(1000)


def test_terminal_cancellation_and_idempotent_close(library):
    loop = NativeLoop(work_items=64, frames=100, library=library)
    request = loop.take(1000)
    loop.cancel()
    with pytest.raises(RuntimeError, match="cancelled"):
        loop.complete(request, 0)
    with pytest.raises(RuntimeError, match="cancelled"):
        loop.wait(1)
    loop.close()
    loop.close()
    with pytest.raises(RuntimeError, match="closed"):
        loop.take()


def test_timeout_drains_backend_and_closes_native_writers(library):
    class StalledBackend(CpuBackend):
        closed = False
        @property
        def pending(self):
            return 1
        def submit(self, loop, request):
            pass
        def close(self):
            self.closed = True
    backend = StalledBackend(64, 1)
    with pytest.raises(TimeoutError):
        run_loop(backend, work_items=64, frames=2, lanes=1, timeout_s=0.01, library=library)
    assert backend.closed


def test_backend_failure_propagates_and_drains(library):
    class FailedBackend(CpuBackend):
        closed = False
        def submit(self, loop, request):
            raise RuntimeError("injected submission failure")
        def close(self):
            self.closed = True
    backend = FailedBackend(64, 1)
    with pytest.raises(RuntimeError, match="injected"):
        run_loop(backend, work_items=64, frames=2, lanes=1, library=library)
    assert backend.closed


def test_service_handles_several_delayed_external_completions(library):
    class DeferredBackend(CpuBackend):
        def __init__(self, count, lanes):
            super().__init__(count, lanes)
            self.outstanding = []
        @property
        def pending(self):
            return len(self.outstanding)
        def submit(self, loop, request):
            self.outstanding.append((request, loop.reference_feature(request)))
        def poll(self, loop):
            if self.outstanding:
                request, feature = self.outstanding.pop()
                loop.complete(request, feature)
            return self.pending
    result = run_loop(DeferredBackend(128, 4), work_items=128, frames=20,
                      lanes=4, workers=2, poll_us=100, library=library)
    assert 1 < result["max_external_in_flight"] <= 4
    assert verify_feedback(result["traces"], work_items=128, frames=20, lanes=4)["passed"]


def test_cuda_adapter_never_publishes_before_recorded_completion():
    from types import SimpleNamespace
    class Event:
        ready = False
        def query(self):
            return self.ready
        def elapsed_time(self, _):
            return 0.001
    done = Event()
    backend = object.__new__(CudaEventBackend)
    backend._pending = {0: Request(0, 3)}
    backend.events = [[Event(), Event(), Event(), done]]
    backend.timings = []
    reads, published = [], []
    backend.host_outputs = [SimpleNamespace(item=lambda: reads.append(True) or 0.25)]
    loop = SimpleNamespace(complete=lambda request, value: published.append((request.frame, value)))
    assert backend.poll(loop) == 1
    assert not reads and not published
    done.ready = True
    assert backend.poll(loop) == 0
    assert reads and published == [(3, 0.25)]


def test_cuda_drain_attempts_every_stream_after_error():
    from types import SimpleNamespace
    drained = []
    def fail():
        drained.append(0)
        raise RuntimeError("device fault")
    backend = object.__new__(CudaEventBackend)
    backend.streams = [SimpleNamespace(synchronize=fail),
                       SimpleNamespace(synchronize=lambda: drained.append(1))]
    backend._pending = {0: Request(0, 0)}
    with pytest.raises(RuntimeError, match="drain"):
        backend.close()
    assert drained == [0, 1]


def test_invalid_sizes_and_overlapping_buffers_fail_before_launch(library):
    for kwargs in ({"frames": 0}, {"workers": 0}, {"lanes": -1}, {"work_items": True}):
        arguments = dict(work_items=64, frames=1, library=library)
        arguments.update(kwargs)
        with pytest.raises(ValueError):
            NativeLoop(**arguments)
    buffer = (C.c_double * 64)()
    with pytest.raises(ValueError, match="overlap"):
        NativeLoop(work_items=64, frames=1, lanes=2, inputs=[buffer, buffer], library=library)
    with pytest.raises(ValueError, match="double array"):
        NativeLoop(work_items=64, frames=1, inputs=[(C.c_float * 64)()], library=library)


def test_trace_cannot_be_read_before_completion(library):
    with NativeLoop(work_items=64, frames=1, library=library) as loop:
        with pytest.raises(RuntimeError, match="not complete"):
            loop.traces()
        with pytest.raises(ValueError, match="timeout"):
            loop.take(-1)


def test_oracle_catches_corrupted_action(library):
    result = run_loop(CpuBackend(64, 1), work_items=64, frames=10, lanes=1, library=library)
    result["traces"][4]["action"] += 0.01
    with pytest.raises(AssertionError, match="feedback mismatch"):
        verify_feedback(result["traces"], work_items=64, frames=10, lanes=1)


def test_cli_records_cpu_only_scope_and_preserves_results(library, tmp_path):
    import json
    output = tmp_path / "result.json"
    args = [sys.executable, str(ROOT / "tests/script/run_embedded_loop.py"),
            "--library", str(library), "--iterations", "10", "--warmup", "2",
            "--repeats", "1", "--lanes", "2", "--output", str(output)]
    completed = subprocess.run(args, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    data = json.loads(output.read_text())
    assert data["transport"] == "native_cpu_reference"
    assert [run["mode"] for run in data["runs"]] == ["cpu_reference"]
    assert data["runs"][0]["correctness"]["checked_frames"] == 24
    assert data["native_library_sha256"]
    completed = subprocess.run(args, capture_output=True, text=True)
    assert completed.returncode != 0
    assert "already exists" in completed.stderr


def test_percentiles_use_nearest_rank():
    result = summarize(list(range(1, 101)))
    assert (result["median_ns"], result["p95_ns"], result["p99_ns"]) == (50, 95, 99)


def test_native_feature_workers_participate_in_feedback(library):
    result = run_loop(NativeCpuBackend(257, 4), work_items=257, frames=100,
                      lanes=4, workers=2, poll_us=0, library=library)
    verify_feedback(result["traces"], work_items=257, frames=100, lanes=4)
    assert sum(result["worker_task_counts"]) == 3 * 400


def test_native_cpu_dispatch_enforces_exclusive_token_and_returns_lease(library):
    with NativeLoop(work_items=65536, frames=2, workers=1, library=library) as loop:
        request = loop.take(1000)
        loop.dispatch_cpu(request)
        assert loop.cpu_load() in (0, 1)
        with pytest.raises(RuntimeError, match="token"):
            loop.dispatch_cpu(request)
        with pytest.raises(RuntimeError, match="lease"):
            loop.reference_feature(request)
        with pytest.raises(RuntimeError, match="token"):
            loop.complete(request, 0)
        next_request = loop.take(1000)
        loop.dispatch_cpu(next_request)
        assert loop.wait(1000)
        assert loop.cpu_load() == 0
        assert loop.cpu_backlog() == {"preprocess": 0, "feature": 0, "control": 0}


def test_adaptive_uses_live_backlog_to_change_placement(monkeypatch):
    from types import SimpleNamespace
    from dae.placement import MeasuredCrossoverPolicy, PlacementSample
    placed = []
    monkeypatch.setattr(CudaEventBackend, "submit", lambda self, loop, req: placed.append("gpu"))
    backend = object.__new__(AdaptiveBackend)
    backend.work_items, backend.workers, backend._pending, backend.decisions = 64, 1, {}, []
    backend.policy = MeasuredCrossoverPolicy([PlacementSample(64, 100, 250)])
    backend._isolated = backend.policy.choose(64)
    backend._preprocess_ns, backend._control_ns = 0, 0
    load = [0]
    loop = SimpleNamespace(cpu_backlog=lambda: {"preprocess": 0, "feature": load[0], "control": 0},
                           dispatch_cpu=lambda req: placed.append("cpu"))
    backend.submit(loop, Request(0, 0))
    load[0] = 3
    backend.submit(loop, Request(1, 0))
    backend._pending[2] = Request(2, 0)
    backend.submit(loop, Request(3, 0))
    assert placed == ["cpu", "gpu", "cpu"]
    assert backend.decisions[1]["cpu_features_outstanding"] == 3
    assert backend.decisions[2]["gpu_requests_outstanding"] == 1


def test_adaptive_rejects_other_experiment_calibration(tmp_path):
    import json
    policy = tmp_path / "wrong.json"
    policy.write_text(json.dumps({"experiment": "old_gh200_attention"}))
    with pytest.raises(ValueError, match="embedded-loop"):
        AdaptiveBackend(64, 1, workers=2, policy_path=policy)


def test_cuda_adaptive_execution_from_real_calibration(library, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("real adaptive execution requires CUDA calibration")
    policy = tmp_path / "calibration.json"
    completed = subprocess.run([
        sys.executable, str(ROOT / "tools/calibrate_embedded.py"), "--library", str(library),
        "--work-items", "64,128", "--iterations", "5", "--warmup", "2", "--repeats", "1",
        "--output", str(policy)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    with pytest.raises(ValueError, match="outside"):
        AdaptiveBackend(256, 1, workers=2, policy_path=policy)
    with pytest.raises(ValueError, match="does not match"):
        AdaptiveBackend(64, 1, workers=1, policy_path=policy)
    backend = AdaptiveBackend(64, 3, workers=2, policy_path=policy)
    result = run_loop(backend, work_items=64, frames=10, lanes=3, library=library)
    verify_feedback(result["traces"], work_items=64, frames=10, lanes=3)
    assert len(result["placement_decisions"]) == 30
    output = tmp_path / "paired.json.gz"
    completed = subprocess.run([
        sys.executable, str(ROOT / "tests/script/run_embedded_loop.py"),
        "--backend", "compare", "--policy", str(policy), "--library", str(library),
        "--work-items", "64", "--iterations", "3", "--warmup", "1", "--repeats", "1",
        "--output", str(output)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    import gzip
    import json
    with gzip.open(output, "rt") as file:
        paired = json.load(file)
    assert [run["mode"] for run in paired["runs"]] == ["native_cpu_workers", "events", "adaptive"]


@pytest.mark.parametrize("blocking", [False, True])
def test_cuda_feedback_integration_when_available(library, blocking):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU, including A10/A100; no simulated GPU claim")
    backend = CudaEventBackend(257, 3, blocking=blocking)
    result = run_loop(backend, work_items=257, frames=20, lanes=3, library=library)
    assert verify_feedback(result["traces"], work_items=257, frames=20, lanes=3)["passed"]
    assert len(result["device_timings"]) == 60
    assert result["max_external_in_flight"] <= 3


@pytest.mark.parametrize("blocking", [False, True])
def test_cuda_graph_keeps_feedback_and_buffer_contract(library, blocking):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA Graph capture requires an actual CUDA GPU")
    backend = CudaGraphBackend(257, 3, blocking=blocking)
    result = run_loop(backend, work_items=257, frames=20, lanes=3, library=library)
    verify_feedback(result["traces"], work_items=257, frames=20, lanes=3)
    assert len(result["device_timings"]) == 60
    assert all(t["graph_replay_ns"] > 0 for t in result["device_timings"])
