"""External-stage adapters for the fixed-graph embedded prototype.

The CUDA adapter is portable Torch CUDA, not the Hopper VDCores VM. Completion
is published only after the recorded D2H event finishes. No mapped atomics.
"""
from __future__ import annotations

import ctypes as C
import json
import hashlib
import gzip
from pathlib import Path
import platform

from .placement import MeasuredCrossoverPolicy, PlacementSample


class CpuBackend:
    name = "native_cpu_reference"

    def __init__(self, work_items: int, lanes: int):
        self.inputs = [(C.c_double * work_items)() for _ in range(lanes)]
        self.timings = []

    def submit(self, loop, request):
        loop.complete(request, loop.reference_feature(request))

    def poll(self, loop):
        return 0

    @property
    def pending(self):
        return 0

    def close(self):
        pass


class NativeCpuBackend(CpuBackend):
    name = "native_cpu_workers"

    def submit(self, loop, request):
        loop.dispatch_cpu(request)


class CudaEventBackend:
    name = "torch_cuda_events"

    def __init__(self, work_items: int, lanes: int, *, blocking: bool = False,
                 device: int = 0):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --backend cpu for local correctness checks")
        self.torch, self.blocking, self.device = torch, blocking, device
        self.work_items = work_items
        self.timings, self._pending = [], {}
        with torch.cuda.device(device):
            self.inputs = [torch.empty(work_items, dtype=torch.float64, pin_memory=True)
                           for _ in range(lanes)]
            self.host_outputs = [torch.empty((), dtype=torch.float64, pin_memory=True)
                                 for _ in range(lanes)]
            self.streams = [torch.cuda.Stream(device=device) for _ in range(lanes)]
            self.device_inputs = [torch.empty(work_items, dtype=torch.float64, device=device)
                                  for _ in range(lanes)]
            self.device_outputs = [torch.empty((), dtype=torch.float64, device=device)
                                   for _ in range(lanes)]
            self.events = [[torch.cuda.Event(enable_timing=True) for _ in range(4)]
                           for _ in range(lanes)]
            # Initialize kernels, handles, and allocator state before native work.
            try:
                for lane, stream in enumerate(self.streams):
                    with torch.cuda.stream(stream):
                        self.device_inputs[lane].zero_()
                        self._compute(lane)
                    stream.synchronize()
            except Exception:
                self.close()
                raise

    def _compute(self, lane):
        self.torch.dot(self.device_inputs[lane], self.device_inputs[lane],
                       out=self.device_outputs[lane])
        self.device_outputs[lane].div_(self.work_items)

    def submit(self, loop, request):
        lane = request.lane
        if lane in self._pending:
            raise RuntimeError("attempted to reuse a CUDA slot still in flight")
        # Register before submission so failure cleanup drains partial transfers.
        self._pending[lane] = request
        with self.torch.cuda.device(self.device), self.torch.cuda.stream(self.streams[lane]):
            start, copied, computed, done = self.events[lane]
            start.record()
            self.device_inputs[lane].copy_(self.inputs[lane], non_blocking=True)
            copied.record()
            self._compute(lane)
            computed.record()
            self.host_outputs[lane].copy_(self.device_outputs[lane], non_blocking=True)
            done.record()
        if self.blocking:
            done.synchronize()  # Only this stream's event; never device-wide sync.
            self._complete(loop, lane)

    def _complete(self, loop, lane):
        request = self._pending[lane]
        start, copied, computed, done = self.events[lane]
        # The scalar and the input slot belong to the CPU only after D2H done.
        value = self.host_outputs[lane].item()
        self.timings.append({
            "lane": lane, "frame": request.frame,
            "h2d_ns": round(start.elapsed_time(copied) * 1e6),
            "compute_ns": round(copied.elapsed_time(computed) * 1e6),
            "d2h_ns": round(computed.elapsed_time(done) * 1e6),
        })
        loop.complete(request, value)
        del self._pending[lane]

    def poll(self, loop):
        for lane in list(self._pending):
            if self.events[lane][-1].query():
                self._complete(loop, lane)
        return self.pending

    @property
    def pending(self):
        return len(self._pending)

    def close(self):
        # Drain even when submit failed before recording its event. Keep all
        # pinned/device owners alive through this operation and propagate errors.
        errors = []
        for stream in self.streams:
            try:
                stream.synchronize()
            except Exception as error:
                errors.append(error)
        self._pending.clear()
        if errors:
            raise RuntimeError("CUDA stream drain failed") from errors[0]


class CudaGraphBackend(CudaEventBackend):
    """Capture the fixed-address copy/reduction/copy sequence once per lane.

    The completion event remains outside capture. Report replay as a whole,
    because per-stage timing instrumentation would alter the captured workload.
    """
    name = "torch_cuda_graph"

    def __init__(self, work_items: int, lanes: int, *, blocking: bool = False,
                 device: int = 0):
        super().__init__(work_items, lanes, blocking=blocking, device=device)
        self.graphs = []
        try:
            with self.torch.cuda.device(device):
                for lane, stream in enumerate(self.streams):
                    self.inputs[lane].zero_()
                    graph = self.torch.cuda.CUDAGraph()
                    with self.torch.cuda.graph(graph, stream=stream):
                        self.device_inputs[lane].copy_(self.inputs[lane], non_blocking=True)
                        self._compute(lane)
                        self.host_outputs[lane].copy_(self.device_outputs[lane], non_blocking=True)
                    stream.synchronize()
                    self.graphs.append(graph)
        except Exception:
            self.close()
            raise

    def submit(self, loop, request):
        lane = request.lane
        if lane in self._pending:
            raise RuntimeError("attempted to reuse a graph slot still in flight")
        self._pending[lane] = request
        with self.torch.cuda.device(self.device), self.torch.cuda.stream(self.streams[lane]):
            self.events[lane][0].record()
            self.graphs[lane].replay()
            self.events[lane][-1].record()
        if self.blocking:
            self.events[lane][-1].synchronize()
            self._complete(loop, lane)

    def _complete(self, loop, lane):
        request = self._pending[lane]
        value = self.host_outputs[lane].item()
        self.timings.append({"lane": lane, "frame": request.frame,
                             "graph_replay_ns": round(self.events[lane][0].elapsed_time(
                                 self.events[lane][-1]) * 1e6)})
        loop.complete(request, value)
        del self._pending[lane]


class AdaptiveBackend(CudaEventBackend):
    """Place each ready feature using real calibration and live backlog counts.

    CPU wait estimate accounts for preprocessing, feature, and control work.
    GPU wait estimate conservatively serializes outstanding GPU requests. These
    are queue-delay estimates, not guarantees; streams may actually overlap.
    """
    name = "adaptive_cpu_cuda_events"

    def __init__(self, work_items: int, lanes: int, *, policy_path: str | Path,
                 workers: int, device: int = 0, switch_margin: float = 0.05):
        policy_path = Path(policy_path)
        opener = gzip.open if policy_path.suffix == ".gz" else open
        with opener(policy_path, "rt") as file:
            calibration = json.load(file)
        if (calibration.get("experiment") != "vdcores_embedded_placement_calibration"
                or calibration.get("workload") != "synthetic_float64_energy_feedback"
                or calibration.get("gpu_transport") != "torch_cuda_events"):
            raise ValueError("policy must be an embedded-loop event-transport calibration")
        native_source = Path(__file__).resolve().parents[2] / "src" / "embedded_loop.cpp"
        if calibration.get("native_source_sha256") != hashlib.sha256(native_source.read_bytes()).hexdigest():
            raise ValueError("placement calibration native source does not match; recalibrate")
        self.policy = MeasuredCrossoverPolicy.from_benchmark(
            calibration, metric="external_observed_ns", switch_margin=switch_margin,
            interpolation=calibration.get("interpolation", "log2"))
        self._isolated = self.policy.choose(work_items)
        if self._isolated.clamped_to_measured_range:
            raise ValueError("work size is outside the measured placement range")
        costs = []
        for metric in ("preprocess_ns", "control_ns"):
            samples = [PlacementSample(row["work_items"],
                        max(1, row.get("cpu_stage_costs_ns", {}).get(metric, 1)), 1)
                       for row in calibration["rows"]]
            costs.append(MeasuredCrossoverPolicy(samples, interpolation=calibration.get(
                "interpolation", "log2")).choose(work_items).cpu_estimate_ns)
        self._preprocess_ns, self._control_ns = costs
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("adaptive CPU/GPU execution requires CUDA")
        gpu = torch.cuda.get_device_properties(device)
        expected = calibration.get("hardware", {})
        if (expected.get("gpu_name") != gpu.name or expected.get("machine") != platform.node()
                or expected.get("workers") != workers or expected.get("torch_version") != torch.__version__):
            raise ValueError("placement calibration hardware, worker count, or Torch version does not match")
        super().__init__(work_items, lanes, device=device)
        self.workers, self.decisions = workers, []

    def submit(self, loop, request):
        backlog, gpu_load = loop.cpu_backlog(), self.pending
        cpu_load = backlog["feature"]
        isolated = self._isolated
        cpu_queue = (backlog["preprocess"] * self._preprocess_ns +
                     cpu_load * isolated.cpu_estimate_ns +
                     backlog["control"] * self._control_ns) // self.workers
        decision = self.policy.choose(
            self.work_items,
            cpu_queue_ns=cpu_queue,
            gpu_queue_ns=gpu_load * isolated.gpu_estimate_ns,
        )
        self.decisions.append({"lane": request.lane, "frame": request.frame,
                               "cpu_features_outstanding": cpu_load,
                               "gpu_requests_outstanding": gpu_load, **vars(decision)})
        self.decisions[-1]["cpu_backlog"] = backlog
        if decision.device == "cpu":
            loop.dispatch_cpu(request)
        else:
            super().submit(loop, request)
