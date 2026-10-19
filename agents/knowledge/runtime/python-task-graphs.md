# Python-led heterogeneous task graphs

`python/dae/graph_execution.py` executes named `Task` / `OperationSpec` DAGs using
the existing `HeterogeneousScheduler`. Operation outputs are named buffers;
inputs must be declared external inputs or dependency outputs. Dependencies may
also express control-only edges. CPU-only/GPU-only/flexible capabilities are
enforced, and flexible tasks can use separate measured placement policies.

The executor keeps one live frame per feedback lane, retains all frame values
until all tasks complete, and releases the next frame with the previous output.
CPU tasks use persistent Python threads. CPU completion wakes the coordinator;
GPU events are polled at a configured interval. Avoid busy-polling CPU futures:
GIL contention between small Torch calls caused major artificial delay. Propagate
the caller's Torch intra-op thread budget to pool threads to avoid nested pools.

Torch GPU futures retain CPU, pinned, device, and result buffers through D2H
completion. Stream slots are bounded. Per-dispatch allocation/pinning costs are
included; this is not an allocation-free or GPU-resident chaining implementation.
Failure/timeout is terminal: join CPU tasks, drain GPU streams, then release
frame contexts. Driver hangs are not preemptible by the host timeout.

`VDCoresTaskBackend` is a distinct adapter for prepared original Launchers.
It calls `launch_async()`, checks the returned stream, then calls `synchronize()`
to release `_inflight_launch` before reading independent host output/reuse.
Only one Launcher operation is outstanding per adapter. The preparation callback
must bind workload-specific buffers/barriers and clean up if preparation fails.
This contract has fake-Launcher unit coverage, NOT Hopper hardware validation.
There is no automatic Torch fallback for a task marked `gpu_executor="vdcores"`.

This graph layer is not MInst/CInst lowering, the coherent counter protocol,
arbitrary GPU VCore-pool sharing, or proof of VDCores performance benefits.
The user explicitly wants A100 work kept as reference/data collection, not a
port or replacement of the original runtime. No existing core CUDA/kernel,
launcher, instruction, or build-target file was changed for these experiments.
