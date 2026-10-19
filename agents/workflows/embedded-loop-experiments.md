# Embedded feedback-loop experiments

## What this version executes

`src/embedded_loop.cpp` owns a fixed dependency graph and persistent C++ CPU
workers. A lane is an independent feedback loop with one reusable input slot:

```text
native sensor/preprocess -> external mean-square reduction -> native controller
          ^                                                      |
          +---------------- previous action ---------------------+
```

The next frame in a lane starts only after its previous controller completes.
Independent lanes share CPU workers and, on CUDA, one GPU via separate streams.
The workload is deterministic synthetic float64 data, not a neural model or a
real robot. It tests execution mechanics before workload integration.

The Python service submits external work and reports completion; native stage
transitions drive CPU tasks. `NativeCpuBackend` also executes the feature on the
persistent native worker pool. `AdaptiveBackend` chooses CPU/GPU after a feature
becomes ready, using measured remaining-stage costs and live native-stage/GPU
backlog. This is a fixed-graph host runtime prototype beside the original GPU VM.
It does not lower arbitrary Python DAGs into VDCores MInst/CInst or execute the
existing Hopper split-attention kernels on Ampere.

## Hardware and access

Both A10 and A100 can run the new ordinary Torch CUDA adapter. A100 is the
preferred first run for the float64 reduction; A10 is a useful follow-up to
compare a different GPU/host configuration. Record platform differences rather
than attributing them solely to the GPU. Use the existing Lambda CUDA-enabled
Torch environment. Building this native library needs only a C++17 compiler;
`nvcc`, `make pyext`, and the Hopper extension are not required for this path.

Need: reachable instance IP, SSH user (typically `ubuntu` for Lambda), matching
private-key access, repository checkout, C++ compiler, and CUDA-enabled Torch.
Never upload the SSH private key to the GPU instance or record it in results.

CUDA completion is a recorded event after explicit pinned H2D, reduction, and
D2H operations. The CPU reads results only after that event reports completion.
Pinned mapping alone does not establish coherent host/GPU atomic support.
See NVIDIA's [event contract](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EVENT.html)
and [host/device atomic restrictions](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives/atomic.html).
The old GH200 benchmarks must stay separate.

## Build and smoke test

From the repository root:

```bash
python tools/build_embedded.py
python -m pytest -q tests/test_embedded_loop.py
python tests/script/run_embedded_loop.py --backend cuda --mode both \
  --work-items 4096 --iterations 200 --warmup 20 --lanes 1 --workers 2 \
  --repeats 3 --poll-us 0 --deadline-ms 1 \
  --output results/embedded_loop/a100-single-loop.json
```

Use `--backend cpu` for portable native-loop validation. It emits only a
`cpu_reference` mode, never a fabricated CUDA comparison. Choose fresh output
paths; the runner refuses overwrites. Set `--timeout-s` for the whole run.
Use `--backend native-cpu` to execute all three computational stages on the
persistent workers. CPU-reference mode retains its original service-thread
reduction for historical validation. A `.json.gz` output path stores losslessly
compressed JSON; the policy loader and plotting tool also accept this format.
The timeout stops new work and drains submitted work; it does not preempt a
CUDA driver call or a hung GPU. CUDA initialization, pool creation, and warmup
are excluded from reported per-frame summaries; the first frames are warmup.

## Measurements worth collecting

| Question | Controlled comparison | Evidence |
| --- | --- | --- |
| Is transport/host overhead dominant? | 64, 4,096, 65,536 values | native stage times, H2D/compute/D2H event intervals, external service queue |
| Does sharing improve throughput at the expense of response time? | 1, 2, 4, 8 independent lanes | aggregate frame rate and per-lane median/p95/p99 |
| Does event service beat the blocking baseline? | events versus blocking, same operations/workers/streams/buffers | repeated paired configuration results, including single-lane regressions |
| How much delay comes from polling? | 0, 50, 200 us requested poll interval | end-to-end latency, control queue, misses of a chosen deadline |
| Are native CPU workers a bottleneck? | 1, 2, 4 workers | preprocessing queue, per-worker task counts, per-lane tail latency |

Start with this sequential 12-configuration matrix, then expand promising
regions. Do not run multiple sweeps concurrently on the same GPU:

```bash
python tools/run_embedded_sweep.py --backend cuda \
  --output-dir results/embedded_loop/a100-initial-sweep
```

The default is 3 sizes x 2 lane counts x 2 poll intervals, 200 measured frames
and 20 warmup frames per lane, and 3 independent trials per configuration and
mode. Mode order reverses between trials. For stronger p99 evidence, rerun
selected configurations with at least 10,000 frames; 200 frames give a very
coarse tail estimate. Include driver version, clocks/power state if available,
CPU model, software versions, source and library hashes, and background load.
`nvidia_smi` in each result captures basic GPU/driver/power metadata.

The blocking baseline uses the identical native loop and CUDA operations but
waits for each submitted completion before serving another lane. Events mode
may serve multiple lanes before polling completions. Both include the requested
poll sleep, instrumentation, and Python submission costs. `poll-us=0` yields
the service thread rather than promising a zero-duration scheduling interval.

## How to interpret the JSON

- `traces` records every frame, including warmup, so feedback can be audited.
- `correctness` compares every output/action with an independent closed-form
  oracle that advances its own feedback state; it also verifies timestamp order.
- `measurements.stages` reports host-clock spans, including completion-observation
  overhead. `device_stages` reports event intervals separately. Do not subtract
  timestamps across the CPU and GPU clocks.
- Queue delay is directly observed per stage. Deadline misses are relative to
  each frame becoming ready, not a periodic real sensor arrival schedule.
- Throughput is all measured frames divided by their enclosing time window.
  Report per-lane latency as well; faster aggregate throughput can hide starvation.
- `worker_task_counts` should sum to twice the total frames. A lane has at most
  one external request in flight; native input reuse follows control completion.
- Compare independent trial summaries, retain raw per-frame samples, and report
  failures or regressions. Do not combine these results with GH200 atomic latency
  or claim the synthetic float64 operation is representative of model inference.

## Verification and next integration gate

Local tests cover repeated feedback, dependency order, persistent worker reuse,
backpressure, out-of-order independent completion, stale/duplicate tokens,
nonfinite output, timeouts, cancellation, adapter failures, and buffer lifetime.
CUDA integration tests skip explicitly when CUDA is absent. The standalone
`tests/native/embedded_loop_stress.cpp` also supports ASan/UBSan and TSan.

After these GPU results: choose a real sensor/model/controller pipeline, replace
the synthetic external stage with the model (or a VDCores launcher on compatible
Hopper), and preserve its event/buffer completion contract. Arbitrary graph
lowering, native adaptive dispatch without Python, a coherent transport adapter, and direct
Hopper-VM CPU micro-operations remain later milestones.

## Calibrate and run measured adaptive placement

```bash
python tools/calibrate_embedded.py --work-items 64,4096,65536,262144,1048576 \
  --output results/embedded_loop/a100-placement.json
python tests/script/run_embedded_loop.py --backend compare \
  --policy results/embedded_loop/a100-placement.json --work-items 524288 \
  --lanes 4 --workers 2 --poll-us 0 --deadline-ms 1 \
  --output results/embedded_loop/held-out-comparison.json.gz
```

Calibration measures the native feature-worker path and CUDA-event path on one
lane. Each trial's median remaining-stage latency contributes to the policy.
The policy validates workload/transport identity, native source hash, host/GPU,
worker count, Torch version, and measured work-size range. A different transport
or hardware requires a new calibration. `compare` alternates the order of native
CPU, GPU events, and adaptive methods between independent trials.

Default embedded calibration uses linear work-size interpolation because the
reduction and copies scale approximately with bytes. The original general policy
retains log-size interpolation as its default. The first A100 log-size experiment
overpredicted CPU cost at an intermediate size and chose GPU too early; preserve
that failed validation rather than reporting only the refined results.

Live CPU backlog includes preprocessing, feature work, and control, weighted by
their measured stage costs and worker count. GPU backlog counts outstanding
requests with a conservative serialized-service estimate. These are predictions;
active tasks have unknown remaining time and streams can overlap. Per-frame JSON
records the decision, estimates, reason, and observed backlog. The 5% switch
margin and Python decision/logging overhead can also cause regressions near the
crossover. Adaptive placement is tested, not proven optimal.

## Plot and preserve the measurements

To compare ordinary submission with CUDA Graph replay on the same buffers:

```bash
python tests/script/run_embedded_loop.py --backend graph-compare \
  --work-items 4096 --lanes 4 --poll-us 0 --iterations 1000 --warmup 100 \
  --output results/embedded_loop/graph-comparison.json.gz
```

This alternates four modes: ordinary events, ordinary blocking, graph events,
and graph blocking. Capture uses fixed pinned/device addresses; native workers
can change their contents only after the completion event. The graph adapter
reports one whole-replay event interval. Its transport is different from the
ordinary event adapter and must receive its own calibration before adaptive use.
Python still submits replays and queries completion; this is not native GPU
submission or arbitrary graph lowering.

```bash
python tools/compress_embedded_results.py results/embedded_loop
python tools/plot_embedded_results.py
```

The compressor only handles this experiment's generated JSON and verifies exact
decompression before replacing it with `.json.gz` (recoverable with `gunzip`).
The plotting tool writes PNG, editable SVG, captions, and an exhaustive summary
table. It separates local CPU, A100 event transport, and earlier GH200 workloads.
