# Native embedded-loop contract

## Scope and files

`src/embedded_loop.cpp` is a standalone C++17 fixed-graph runtime, separate from
the Hopper MInst/CInst interpreter. `python/dae/embedded.py` owns its ABI-v3
handle and buffers via ctypes. `embedded_backends.py` owns CPU reference,
native-worker, CUDA-event, CUDA-graph, and adaptive adapters.

## Lane state and dependencies

Each lane has a fixed input slot, frame number, and previous action:

```text
PREP_QUEUED -> PREP_RUNNING -> EXTERNAL_READY -> EXTERNAL_RUNNING
                                                   |
                          +------------------------+----------------+
                          |                                         |
                  native CPU dispatch                      external completion
                          |                              (CUDA D2H event ready)
                  FEATURE_QUEUED -> FEATURE_RUNNING                  |
                          +------------------------+----------------+
                                                   |
                                          CONTROL_QUEUED
                                                   |
                                          CONTROL_RUNNING
                                                   |
                           next frame PREP_QUEUED or terminal DONE
```

Preprocessing includes the previous action in the synthetic input. Control is
`clamp(0.9 * previous_action - 0.5 * feature, -1, 1)`. The default feature is
the float64 mean square. Only independent lanes overlap; a lane cannot reuse its
buffer until its own control completion. The native pool persists for all frames
in one run. There is a shared native CPU ready queue, not distributed scheduling
of arbitrary GPU virtual-core instructions.

## C ABI and completion tokens

- `el_create`: retain caller-owned double input pointers, initialize lanes, start
  worker threads. Python validates shape/dtype/CPU residency/contiguity and
  rejects overlapping input allocations. Native and Python size caps both apply.
- `el_take`: transfer a ready lane's buffer lease to the service, returning
  `(lane, frame)`. Timeout zero checks state under the mutex without a timed
  condition-variable wait; positive timeout can block.
- `el_complete`: validate token, finite result, and running state, then release
  control. Stale, duplicate, and unissued completions fail.
- `el_dispatch_cpu`: validate the same token and transfer its lease to a native
  feature worker. A service cannot complete/read that token afterward.
- `el_cpu_backlog`: mutex-consistent queued-plus-running counts for preprocessing,
  feature, and control. Counts do not reveal remaining task time.
- `el_wait`: report all lanes complete; zero timeout is a direct state check.
- `el_traces` / `el_worker_counts`: copy completed traces or current task counts.
- `el_cancel` / `el_destroy`: cancellation is terminal; destruction joins native
  threads before the Python owner can release the buffers.

All native state changes and CPU completion publication share one mutex. This
provides CPU happens-before edges for buffer writes and consumers. There is no
concurrent host/GPU atomic access to these buffers in this adapter protocol.

## CUDA memory lifetime

The service may read/copy a lane only while it owns that request. CUDA adapters
retain pinned inputs/outputs, device inputs/outputs, streams, and events. H2D,
reduction, and D2H occur in the lane's stream. A recorded event after D2H must
finish before reading the host result or calling native completion.

Graph capture uses the same fixed buffer addresses; only contents change between
frames. Its completion event is recorded outside capture, and replay timing is
reported as a whole. Per-stage timings are not invented for graph replay.

On error or timeout, `run_loop` joins native workers and drains every owned CUDA
stream while both owners are still retained. A timeout does not preempt a hung
CUDA driver call. Do not reset a counter or reuse a cancelled run's slots to
recover; construct a new run after draining.

## Placement and boundaries

Placement runs when a feature becomes ready. Same-host calibration supplies
remaining-stage cost, and native backlog plus GPU outstanding requests supply
queue estimates. Results log each estimate/choice. CPU/GPU source, workload,
transport, worker count, and host validation prevent silent reuse of incompatible
calibration. Event and graph transports require separate policies.

Python still submits GPU operations and chooses placement. Remaining work is a
native GPU service/placement path, general graph descriptors and lowering,
Hopper instruction integration, coherent transport, and a real embedded workload.
