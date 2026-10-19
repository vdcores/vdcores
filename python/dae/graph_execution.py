"""Bounded Python-led task graphs; separate from the Hopper GPU VM.

Consumers must treat inputs as immutable. CPU futures and GPU completion events
own all argument/output buffers until completion or terminal drain. No graph
value is published on launch alone. Python deliberately remains the coordinator.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import sys
import threading
import time
from typing import Callable, Mapping

from .heterogeneous import HeterogeneousScheduler, OperationSpec


@dataclass(frozen=True)
class Task:
    operation: OperationSpec
    inputs: tuple[str, ...]
    function: Callable
    gpu_executor: str = "torch_cuda"


class TaskGraph:
    def __init__(self, tasks, *, external_inputs=("frame", "previous"), output="control"):
        tasks = tuple(tasks)
        self.tasks = {t.operation.name: t for t in tasks}
        if len(self.tasks) != len(tasks) or not self.tasks:
            raise ValueError("task names must be unique and graph must be nonempty")
        self.external_inputs = tuple(external_inputs)
        if len(set(external_inputs)) != len(external_inputs):
            raise ValueError("duplicate external input")
        self.output = output
        if output not in self.tasks or set(self.tasks) & set(external_inputs):
            raise ValueError("invalid output or overlapping input/task names")
        for task in tasks:
            allowed = set(task.operation.dependencies) | set(external_inputs)
            if not set(task.inputs) <= allowed:
                raise ValueError("task inputs must be external or declared dependencies")
        self.new_scheduler()  # Validate missing dependencies and cycles up front.

    def new_scheduler(self):
        scheduler = HeterogeneousScheduler(t.operation for t in self.tasks.values())
        scheduler.seal()
        return scheduler


class CpuFuture:
    def __init__(self, future):
        self.future = future

    def done(self):
        return self.future.done()

    def result(self):
        return self.future.result()


class CpuTaskBackend:
    identity = "python_thread_pool_cpu"

    def __init__(self, workers=2, notify=None):
        # OpenMP's thread-count setting is thread-local on some Torch builds.
        # Propagate the caller's budget to each persistent Python worker rather
        # than accidentally creating a full nested compute pool per operation.
        torch = sys.modules.get("torch")
        budget = torch.get_num_threads() if torch is not None else None
        initializer = (lambda: torch.set_num_threads(budget)) if torch is not None else None
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="graph-cpu",
                                       initializer=initializer)
        self.notify = notify

    def available(self):
        return True

    def submit(self, task, arguments):
        def execute():
            started = time.perf_counter_ns()
            value = task.function(*arguments, device="cpu")
            return value, {"started_ns": started, "finished_ns": time.perf_counter_ns()}
        future = self.pool.submit(execute)
        if self.notify is not None:
            future.add_done_callback(lambda _: self.notify.set())
        return CpuFuture(future)

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)


class CudaTaskFuture:
    def __init__(self, backend, slot, arguments, retained, output, done, started, events):
        self.backend, self.slot = backend, slot
        self.arguments, self.retained, self.output = arguments, retained, output
        self.event, self.started, self.events = done, started, events
        self.released = False

    def done(self):
        return self.event.query()

    def result(self):
        if self.released:
            raise RuntimeError("CUDA result already released")
        if not self.done():
            raise RuntimeError("CUDA output is not ready")
        start, copied, computed, done = self.events
        timing = {"started_ns": self.started, "finished_ns": time.perf_counter_ns(),
                  "h2d_ns": round(start.elapsed_time(copied) * 1e6),
                  "compute_ns": round(copied.elapsed_time(computed) * 1e6),
                  "d2h_ns": round(computed.elapsed_time(done) * 1e6)}
        self.released = True
        del self.backend.inflight[self.slot]
        self.backend.free.append(self.slot)
        return self.output, timing


class TorchCudaTaskBackend:
    """Ordinary Torch GPU tasks with explicit copies, NOT VDCores execution.

    This first general adapter allocates per dispatch; those costs remain inside
    measurements. Streams are bounded. Arguments must be CPU tensors; outputs
    are tensors. A future retains pinned/device buffers through its D2H event.
    """
    identity = "torch_cuda_explicit_copy_events_not_vdcores"
    kind = "torch_cuda"

    def __init__(self, slots=4, device=0):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        self.torch, self.device = torch, device
        self.streams = [torch.cuda.Stream(device=device) for _ in range(slots)]
        self.free = list(range(slots))
        self.inflight = {}

    def available(self):
        return bool(self.free)

    def submit(self, task, arguments):
        if not self.free:
            raise RuntimeError("CUDA task capacity exhausted")
        torch = self.torch
        slot = self.free.pop()
        stream = self.streams[slot]
        started = time.perf_counter_ns()
        retained = []
        # Retain partially submitted buffers as well: exceptions must drain
        # before the last Python reference goes away.
        self.inflight[slot] = retained
        with torch.cuda.device(self.device), torch.cuda.stream(stream):
            events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            events[0].record()
            device_args = []
            for value in arguments:
                if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
                    raise TypeError("Torch CUDA task inputs must be CPU tensors")
                pinned = value if value.is_pinned() else value.pin_memory()
                copied = pinned.to(device=f"cuda:{self.device}", non_blocking=True)
                retained.extend((value, pinned, copied))
                device_args.append(copied)
            events[1].record()
            output = task.function(*device_args, device=f"cuda:{self.device}")
            if not isinstance(output, torch.Tensor) or output.device.type != "cuda":
                raise TypeError("Torch CUDA task must return a CUDA tensor")
            retained.append(output)
            events[2].record()
            host = torch.empty_like(output, device="cpu", pin_memory=True)
            retained.append(host)
            host.copy_(output, non_blocking=True)
            events[3].record()
        future = CudaTaskFuture(self, slot, arguments, retained, host, events[-1], started, events)
        return future

    def close(self):
        errors = []
        for stream in self.streams:
            try:
                stream.synchronize()
            except Exception as error:
                errors.append(error)
        self.inflight.clear()
        if errors:
            raise RuntimeError("CUDA task drain failed") from errors[0]


class VDCoresTaskBackend:
    """Exclusive adapter for prepared original Launcher objects, not a fallback.

    prepare(task, arguments) must bind device buffers/barriers and return a
    Launcher. read_output(launcher) runs only after its launch stream finishes;
    it must return independent host output if the launcher reuses its buffers.
    Hardware integration is intentionally separate from Torch adapter testing.
    """
    identity = "original_vdcores_launcher_adapter_hardware_validation_pending"
    kind = "vdcores"

    def __init__(self, prepare, read_output):
        self.prepare, self.read_output = prepare, read_output
        self.active = None

    def available(self):
        return self.active is None

    def submit(self, task, arguments):
        if self.active is not None:
            raise RuntimeError("VDCores launcher already in flight")
        launcher = self.prepare(task, arguments)
        self.active = launcher  # Keep even a partially failed launch for drain.
        started = time.perf_counter_ns()
        stream = launcher.launch_async()
        backend = self

        class Future:
            owners = arguments, launcher
            def done(self):
                return stream.query()
            def result(self):
                if backend.active is not launcher or not self.done():
                    raise RuntimeError("VDCores result is not ready or already released")
                launcher.synchronize()  # Release Launcher's retained launch metadata.
                value = backend.read_output(launcher)
                backend.active = None
                return value, {"started_ns": started, "finished_ns": time.perf_counter_ns()}
        return Future()

    def close(self):
        if self.active is not None:
            try:
                self.active.synchronize()
            finally:
                self.active = None


def execute_graph(graph: TaskGraph, input_factory, *, frames=10, lanes=1,
                  workers=2, placement="cpu", policies: Mapping | None = None,
                  gpu_backend=None, serial=False, initial=None, period_ns=0,
                  timeout_s=60, poll_us=50, cpu_costs: Mapping | None = None):
    """Execute one DAG per feedback frame; retain at most one frame per lane.

    input_factory(lane, frame, previous_output) supplies declared external inputs.
    Periodic runs never drop frames: latency includes lateness/backlog from each
    scheduled release. Fixed GPU placement respects CPU-only operation capability.
    """
    if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in (frames, lanes, workers)):
        raise ValueError("frames, lanes, workers must be positive integers")
    if placement not in ("cpu", "gpu", "adaptive"):
        raise ValueError("unknown placement")
    if not isinstance(period_ns, int) or period_ns < 0:
        raise ValueError("period_ns must be a nonnegative integer")
    if not math.isfinite(timeout_s) or timeout_s <= 0 or not math.isfinite(poll_us) or poll_us < 0:
        raise ValueError("invalid timeout or polling interval")
    policies = policies or {}
    cpu_costs = cpu_costs or {}
    if placement == "adaptive" and any(len(t.operation.devices) > 1 and name not in policies
                                        for name, t in graph.tasks.items()):
        raise ValueError("adaptive placement needs a policy per flexible task")
    if gpu_backend is None and (placement != "cpu" or any(t.operation.devices == ("gpu",)
                                                          for t in graph.tasks.values())):
        raise ValueError("GPU backend required")
    if gpu_backend is not None and any("gpu" in t.operation.devices and
            t.gpu_executor != gpu_backend.kind for t in graph.tasks.values()):
        raise ValueError("task GPU executor does not match the supplied backend")
    wake = threading.Event()
    cpu = CpuTaskBackend(workers, notify=wake)
    started = time.perf_counter_ns()
    process_start = time.process_time_ns()
    expires = time.monotonic() + timeout_s
    contexts, pending, records, task_records = {}, {}, [], []
    next_frames = [0] * lanes
    previous = [initial] * lanes
    max_contexts = max_pending = 0
    fair_cursor = 0
    try:
        while len(records) < frames * lanes:
            wake.clear()  # Clear before checking futures; callbacks cannot be lost.
            if time.monotonic() >= expires:
                raise TimeoutError("graph execution timed out")
            for lane in range(lanes):
                frame = next_frames[lane]
                release = started + frame * period_ns
                if lane in contexts or frame >= frames or time.perf_counter_ns() < release:
                    continue
                admitted = time.perf_counter_ns()
                values = dict(input_factory(lane, frame, previous[lane]))
                if set(values) != set(graph.external_inputs):
                    raise ValueError("input factory does not match graph external inputs")
                contexts[lane] = {"frame": frame, "values": values,
                                  "scheduler": graph.new_scheduler(), "release_ns": release if period_ns else admitted,
                                  "admitted_ns": admitted, "ready": {}}
                max_contexts = max(max_contexts, len(contexts))
            # Publish every completed value before looking for newly ready work.
            for key, item in list(pending.items()):
                future, trace = item
                if not future.done():
                    continue
                value, timing = future.result()
                context = contexts[trace["lane"]]
                context["values"][trace["task"]] = value
                context["scheduler"].complete(trace["task"])
                trace.update(timing, completed_ns=time.perf_counter_ns())
                task_records.append(trace)
                del pending[key]
            for lane, context in list(contexts.items()):
                if context["scheduler"].is_complete:
                    ended = time.perf_counter_ns()
                    previous[lane] = context["values"][graph.output]
                    records.append({"lane": lane, "frame": context["frame"],
                                    "release_ns": context["release_ns"],
                                    "admitted_ns": context["admitted_ns"], "end_ns": ended,
                                    "output": previous[lane]})
                    next_frames[lane] += 1
                    del contexts[lane]
            for lane in sorted(contexts, key=lambda lane: (lane - fair_cursor) % lanes):
                context = contexts[lane]
                for spec in context["scheduler"].ready():
                    context["ready"].setdefault(spec.name, time.perf_counter_ns())
                    if serial and pending:
                        continue
                    task = graph.tasks[spec.name]
                    decision = None
                    cpu_queue = gpu_queue = 0
                    if len(spec.devices) == 1:
                        device = spec.devices[0]
                    elif placement == "adaptive":
                        policy = policies[spec.name]
                        for _, active in pending.values():
                            active_policy = policies.get(active["task"])
                            if active_policy is not None:
                                pair = active_policy.choose(graph.tasks[active["task"]].operation.work_items)
                                if active["device"] == "cpu":
                                    cpu_queue += pair.cpu_estimate_ns // workers
                                else:
                                    gpu_queue += pair.gpu_estimate_ns
                            elif active["device"] == "cpu":
                                cpu_queue += cpu_costs.get(active["task"], 0) // workers
                        decision = policy.choose(spec.work_items, cpu_queue_ns=cpu_queue,
                                                 gpu_queue_ns=gpu_queue, deadline_ns=spec.deadline_ns)
                        if decision.clamped_to_measured_range:
                            raise ValueError("task size is outside calibration range")
                        device = decision.device
                    else:
                        device = placement
                    backend = cpu if device == "cpu" else gpu_backend
                    if not backend.available():
                        continue
                    context["scheduler"].dispatch(spec.name, device=device)
                    trace = {"lane": lane, "frame": context["frame"], "task": spec.name,
                             "device": device, "executor": backend.identity,
                             "ready_ns": context["ready"][spec.name],
                             "submitted_ns": time.perf_counter_ns(),
                             "cpu_queue_estimate_ns": cpu_queue, "gpu_queue_estimate_ns": gpu_queue,
                             "placement_estimates_ns": {"cpu": decision.cpu_total_ns, "gpu": decision.gpu_total_ns}
                                                       if decision else None,
                             "placement_reason": decision.reason if decision else "fixed capability/placement"}
                    arguments = tuple(context["values"][name] for name in task.inputs)
                    pending[lane, spec.name] = backend.submit(task, arguments), trace
                    fair_cursor = (lane + 1) % lanes
                    max_pending = max(max_pending, len(pending))
            if any(trace["device"] == "gpu" for _, trace in pending.values()):
                wake.wait(poll_us / 1e6)
            elif pending:
                # Do not busy-poll CPU futures: repeatedly reacquiring the GIL
                # between tiny Torch calls can starve the actual worker threads.
                wake.wait(.01)
            else:
                next_releases = [started + next_frames[lane] * period_ns
                                 for lane in range(lanes) if next_frames[lane] < frames]
                if next_releases:
                    wake.wait(min(.01, max(0, min(next_releases) - time.perf_counter_ns()) / 1e9))
        return {"frames": records, "tasks": task_records, "max_active_frames": max_contexts,
                "max_pending_tasks": max_pending, "wall_ns": time.perf_counter_ns() - started,
                "process_cpu_ns": time.process_time_ns() - process_start,
                "backend": gpu_backend.identity if gpu_backend is not None else cpu.identity}
    finally:
        # Frame contexts remain alive until CPU writers AND GPU readers drain.
        try:
            cpu.close()
        finally:
            if gpu_backend is not None:
                gpu_backend.close()
