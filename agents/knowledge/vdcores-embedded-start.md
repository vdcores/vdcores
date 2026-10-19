# VDCores-Embedded Starting Point

This note connects the VDCores-Embedded proposal to the current repository and
records a small first milestone. It is based on the proposal shared on
2026-09-03 and the current `main` branch.

## Proposed System

VDCores-Embedded would represent CPU and GPU work in one dependency graph.
Operations become eligible to run when their inputs are ready. An operation can
be CPU-only, GPU-only, or have both implementations. For operations with both
implementations, the runtime can choose a device using queue depth, input size,
estimated latency, or a deadline.

The first research target should be dependency and handoff cost, not a complete
placement policy. Dynamic placement is only useful if a GPU-to-CPU-to-GPU chain
can advance cheaply enough to beat explicit launches and synchronization.

## Current Runtime

The current runtime is GPU-only:

- Python applications build per-SM compute and memory instruction streams with
  `python/dae/launcher.py` and `python/dae/instructions.py`.
- `src/runtime.cu` launches one persistent block on each selected GPU SM.
- Each block contains compute and memory virtual cores. The memory side moves
  tiles and publishes ready tokens; the compute side consumes those tokens and
  returns completion or writeback tokens.
- Global barriers coordinate work across SMs. Shared-memory slots carry data
  between the memory and compute sides of one SM.

This provides a dependency protocol to extend, but it does not yet provide a
CPU executor, CPU operation encoding, cross-device completion path, capability
metadata, or CPU/GPU placement policy.

## Split-KV Attention Walkthrough

`app/python/attention_split_kv.py` is a grouped-query attention experiment for
a 32,768-token sequence. It has 32 query heads and 8 key/value heads, so four
query heads share each key/value head.

The example divides the sequence into two equal parts. For each split and each
key/value head, one GPU SM performs local attention over 256 blocks of 64
tokens. This uses 16 SMs in total:

```text
2 sequence splits x 8 key/value heads x 1 request = 16 SMs
```

Each worker reads a query tile, streams its assigned key and value tiles, and
writes two results:

1. a locally normalized attention output;
2. a log-sum-exp value that describes the scale of that local result.

All 16 workers release the same global barrier after writing those results.
The first eight SMs then run `ATTN_SPLIT_POST_REDUCE`, one reducer per
key/value head. The reducer uses the log-sum-exp values to rescale and combine
the two partial outputs into the same result that a single full-sequence
softmax would produce.

The path crosses these files:

- `app/python/attention_split_kv.py`: tensor setup, static SM assignment,
  instruction construction, and PyTorch reference functions.
- `python/dae/instructions.py`: Python encodings for split attention and the
  post-reduction operation.
- `include/dae/compute_dispatch.cuh`: opcode dispatch into CUDA task code.
- `include/task/attention.cuh`: local online softmax and split reduction.

The current example assumes the number of 64-token blocks divides evenly by
the split count. `NUM_KV_BLOCK // split_kv` would otherwise leave remainder
blocks unassigned. That does not affect the checked-in configuration, where
512 blocks divide evenly into two splits, but it matters before parameterizing
the experiment.

## First Milestone

1. Reproduce the split-attention result on one Vista GH200 batch node and save
   the job log, environment versions, and normalized mean error.
2. Dump the generated instruction streams for one worker and one reducer, then
   identify the exact barrier and completion-token path between them.
3. Build a minimal heterogeneous chain: GPU producer, CPU operation, GPU
   consumer. Use one fixed placement and one small shared buffer.
4. Measure the GPU-to-CPU and CPU-to-GPU notification latency before adding
   dynamic placement or a robotics workload.

Questions to settle with the project lead before step 3:

- Should CPU virtual cores be native worker threads inside the Torch extension?
- What operation granularity should the first prototype target?
- Which shared-memory mechanism is the intended GH200 baseline?
- Which end-to-end workload will be used after the microbenchmark?

## Fixed Heterogeneous Chain Result

The first fixed GPU-to-CPU-to-GPU chain was completed on a GH200 on
2026-09-22. `tests/script/check_attention_handoff.py` connects VDCores split-KV
attention to a CPU dot product and then to a prequeued GPU dot-product consumer
through a CPU-resident coherent counter. The GPU consumer waits for the CPU
acknowledgement inside CUDA and signals only after its output is ready.

Against a matched blocking synchronize-and-relaunch baseline, the 1,000-run
median fell from 3.389 ms to 2.847 ms, a 0.541 ms or 15.97% reduction. The
prequeued GPU consumer completed 4.192 us after CPU acknowledgement, while the
blocking post-CPU launch and synchronization stage took 694.726 us. Making the
attention result visible to the CPU added about 151 us, so the net gain came
from avoiding the later host launch boundary.

See `results/cpu_gpu_handoff/2026-09-22-attention-chain.md` for the method,
interpretation, environment, and limitations, and the adjacent JSON file for
raw aggregate measurements. This is still fixed placement; workload crossover
measurements should come before adding a dynamic placement policy.

## Measured Placement Follow-On

`tests/script/check_attention_placement.py` is the next experiment. It keeps
the split-KV attention producer and a final prequeued GPU successor fixed, but
runs the middle float64 dot product on either CPU or GPU across configurable
work sizes. This isolates the placement crossover while preserving a real
GPU-producer to flexible-stage to GPU-successor dependency graph.

`python/dae/placement.py` consumes the benchmark JSON as a measured crossover
policy. It interpolates latency within the measured work-size range, adds
caller-supplied CPU/GPU queue delay, supports a deadline, and uses a switch
margin to avoid changing placement for small predicted gains. It deliberately
clamps outside the measured range instead of inventing an extrapolation.

The experiment and policy are implemented but still require a new sequential
GH200/GB200 sweep before the crossover or any selected device can be reported.
They do not yet create CPU virtual-core workers or dynamically schedule a full
VDCores instruction graph.

## Offline Heterogeneous Scheduler

`python/dae/heterogeneous.py` now provides the first backend-independent DAG
scheduler. It validates dependencies and cycles, exposes ready operations,
tracks pending/ready/running/complete transitions, preserves fixed CPU/GPU
placements, and asks the measured crossover policy to place flexible work with
queue-delay and deadline inputs. CUDA execution remains outside the scheduler;
backends consume dispatch records and report completion.

`tests/script/run_adaptive_attention_handoff.py` connects that scheduler to the
attention experiment. Its plan-only mode loads a calibration JSON and chooses
the middle CPU/GPU dot placement without importing CUDA. Execution mode uses
the same decision to run one of the two already-correctness-checked paths on a
coherent machine. Unmeasured work sizes fail closed unless endpoint clamping is
explicitly enabled.

These pieces are verified offline with synthetic values in tests only. No
synthetic crossover or placement result belongs under `results/` or should be
reported as a hardware finding.

## Native fixed feedback loop (2026-10-18)

`src/embedded_loop.cpp` adds a standalone C++17 worker pool, with ctypes ownership
in `python/dae/embedded.py`. Its typed fixed graph repeats native preprocessing,
an external mean-square reduction, and native control; the previous action is
part of the next synthetic sensor input. Per-lane slots and lane/frame completion
tokens serialize feedback and permit independent loops to overlap. Native stage
queues and a mutex/condition-variable completion protocol drive CPU stages.
This is an intentionally fixed-graph prototype, not general MInst/CInst lowering.

`python/dae/embedded_backends.py` provides real CPU-reference execution and a
Torch CUDA adapter with pinned H2D/D2H transfers, per-lane streams, and completion
events. Python owns external submission/query, but does not traverse the graph.
After a recorded D2H event completes, its result releases native control. The
blocking-service comparison uses the same operations and buffers and waits on
each event instead of servicing several outstanding lanes.

The CUDA path targets discrete A10/A100 as well as newer CUDA devices and does
not import the Hopper extension or use host/GPU atomics. Native workers stop
before owners can release memory; stream work drains on completion or failure.
Cancellation is terminal. A CUDA run timeout bounds service progress, not a
potentially hung driver call or GPU operation.

Build with `python tools/build_embedded.py`; no CUDA toolkit is needed for the
native library. Use `tests/script/run_embedded_loop.py` and
`tools/run_embedded_sweep.py` following
`agents/workflows/embedded-loop-experiments.md`. The focused suite passes 38
tests locally and skips two CUDA integration cases. Native ASan/UBSan and TSan
stress runs also passed. Local Apple M4 Pro CPU-only results preserve 3,300
correct feedback frames under `results/embedded_loop/`; no GPU result is implied.

At the initial local milestone, GPU validation and placement integration were
pending. The following A100 milestone completes those two fixed-loop gates.

## A100 execution and placement follow-on (2026-10-18)

The supplied Lambda instance is A100-SXM4-80GB (108 SMs), AMD EPYC 7J13 host,
Torch 2.7.0/CUDA 12.8, driver 580.105.08. All 48 focused tests pass there,
including real event/blocking execution and a real-calibration adaptive test.
Two of these cases also validate captured CUDA Graph replay with changing
feedback inputs and both completion-service modes.
Native ASan/UBSan stress also passes remotely; TSan passes on the local Mac.

Native ABI version 3 adds persistent-worker feature dispatch and a consistent
snapshot of queued/running preprocessing, feature, and controller counts.
`AdaptiveBackend` combines these counts with measured cost and pending GPU
requests, then releases either a native feature task or a CUDA event request.
Calibration is isolated from the earlier coherent attention benchmark and
validates hardware/workload/source identity. `--backend compare` executes both
fixed devices and adaptive in alternating method order. General policy log-size
interpolation is preserved, with an optional linear model for byte-scaled work.

Recorded data includes an initial 12-configuration shared-loop/polling sweep,
exploratory and refined placement trials, order-alternated held-out comparisons,
and longer tail validation. Source/library hashes distinguish intermediate
implementations. Preserve regressions: the first interpolation model selected
GPU too early, and current adaptive dispatch still adds overhead and can lose
to fixed GPU placement near the crossover. Generated plots and a complete
summary table are under `results/embedded_loop/figures/`.

The former GPU-validation and placement-integration gates are now complete for
this fixed event-transport workload. Still open: native GPU submission/selection
without Python, arbitrary graph lowering, coherent/Hopper VM integration, and
a real embedded sensor/model/controller workload. CUDA-stream sharing of fixed
loops is not yet sharing arbitrary GPU VCore instruction graphs.
