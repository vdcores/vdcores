# Embedded loop implementation and review (2026-10-18)

## Target and acceptance criteria

Finish a first executable, bounded feedback-loop prototype on this branch:
synthetic sensor/preprocess (native CPU) -> feature reduction (external backend)
-> controller (native CPU) -> next sensor frame using the previous action.
Multiple independent loops may share one CPU worker pool and GPU service.
The input is a deterministic synthetic signal, not a real robotics dataset or
neural-network model. Completion requires numerical agreement with a sequential
reference across every frame, native worker reuse, bounded buffer ownership,
generation-checked completion, cancellation, and no deadlocks in failure tests.

## Implementation order

1. Add a standalone C++ CPU worker runtime, built without Torch/CUDA. Encode the
   fixed three-stage graph as typed stages and successor transitions. Each lane
   has one reusable input/output slot and monotonically increasing frame token.
   Native workers perform preprocessing/control and release dependent stages.
2. Add a thin Python/ctypes owner. Hold all buffers until workers have stopped
   and external work has drained. Reject stale, duplicate, and nonfinite
   completions. A terminal cancellation prevents further slot reuse.
3. Add external adapters: a real CPU reference reduction for local execution,
   and Torch CUDA streams, pinned H2D/D2H buffers, and recorded completion events
   for discrete GPUs. Python submits external work and reports completion;
   native code, rather than Python graph traversal, advances CPU stages.
4. Add a matched blocking adapter using the same graph, buffers, operations,
   and native workers. Change only whether the GPU service blocks on one event
   or can submit/query several independent loops. Do not promise a speedup.
5. Add an experiment runner with warmup, independent repeats, raw per-frame
   timings/actions, median/p95/p99, deadline misses, per-loop results, throughput,
   correctness, Git/environment metadata, and separate CPU/CUDA result labels.
6. Test repeated feedback, multiple lanes, worker reuse, backpressure, out-of-order
   independent completions, duplicate/stale tokens, cancellation, invalid input,
   and nonfinite output. Add a CUDA integration test that explicitly skips when
   unavailable; prepare A10/A100 run instructions and a measurement matrix.
7. Update README and durable knowledge to distinguish this prototype from the
   original Hopper GPU VM and GH200 coherent-counter experiments.

## Plan review before implementation

- **Hardware:** neither A10 nor A100 supports the checked-in `sm_90a` TMA/WGMMA
  runtime. Use ordinary Torch CUDA operations for the new adapter; keep the old
  attention experiments hardware-gated. Mapped/pinned memory alone does not
  establish host/GPU atomic support. Events are the completion contract here.
- **Scope:** this finishes a fixed-graph loop prototype, not arbitrary native DAG
  lowering, automatic placement, a production robot controller, or Hopper VM
  integration. The native protocol is extensible, but those remain separate work.
- **Feedback versus streaming:** frames in one lane must serialize because the
  next input depends on its action. Only independent lanes may overlap. This
  avoids falsely advertising overlap of causally dependent frames.
- **Baseline fairness:** same native CPU work, same CUDA operators, same lane
  count, and allocations outside measurement. Measure event service versus
  blocking service, and report single-lane overhead as well as shared-loop gain.
- **Lifetime:** cancellation is terminal. Never reuse or free host/device slots
  until native writers stop and all submitted stream work drains. Tokens include
  lane/frame identity. CUDA errors must propagate rather than release control.
- **Measurements:** CPU tests cannot establish GPU speedup or cross-device
  atomic latency. Record full-loop host latency and GPU event intervals
  separately. Pool startup/build/CUDA initialization stay outside frame timing.
  Synthetic injected delays belong only in tests, never hardware result claims.
- **Review outcome:** proceed with this bounded implementation. Collect A10/A100
  event-transport data first; reserve coherent handoff comparisons and real
  VDCores attention integration for a compatible Hopper/coherent machine.

## Implementation review and evidence

- Native transitions and buffer ownership: implemented in `src/embedded_loop.cpp`;
  review added C-ABI size limits and Python buffer-lease checks to reject stale
  reads as well as stale completion. Native destruction joins workers before
  buffer owners are released. External adapters drain submitted work on failure.
- CUDA ordering: completion follows the D2H event, with a unit test showing no
  scalar read/publication before event readiness. Added cleanup for partially
  submitted work and initialization failures. Real CUDA cases remain pending.
- Baseline: native work/graph and CUDA buffers/operators are shared by both modes;
  the service differs only in blocking completion versus outstanding-event polling.
- Validation: 38 tests pass; two real-CUDA cases skip on the local Mac. Separate
  ASan/UBSan and TSan harnesses each pass 4,000 frames and 100 cancellations.
  CPU-reference experiments preserve 3,300 correct local frames over six trials.
- Scope gate: steps 1-7 are implemented for the fixed-graph prototype. GPU adapter
  execution, hardware overlap/latency findings, general graph lowering, placement
  integration, and a real embedded workload are not complete. The experiment
  workflow and README retain these open gates explicitly.

## A100 follow-on plan and review (2026-10-18)

After the CUDA tests passed on the supplied A100, the user requested plots and
maximum feasible checklist progress. Run the initial shared-loop/event sweep,
preserve it, then implement native CPU feature dispatch and measured adaptive
CPU/GPU placement with actual queued/running CPU-feature counts and outstanding
GPU requests. Calibrate the remaining external-stage cost on this same machine;
do not reuse the earlier GH200 attention calibration. Reject incompatible or
out-of-range policy data. Queue delay remains an estimate based on measured
cost and observed outstanding work, not a hardware completion-time guarantee.

Review gates: baseline and adaptive paths must compute identical features and
feedback, native CPU feature tasks must participate in cancellation/token checks,
and measured policy selection must not force a GPU crossover that data does not
support. Preserve original reference runs when changing the CPU executor. Plot
all configurations with repeated-trial spread, per-loop tails, and host/device
stage costs. Finish with held-out adaptive execution and an honest remaining-
work list. General DAG lowering, coherent transport, and real robot/model
integration stay open because they need additional design or compatible inputs.

### Follow-on review outcome

Implemented persistent native feature dispatch, ABI-v3 full CPU-stage backlog,
real same-host calibration, hardware/source policy validation, per-frame adaptive
selection, and an order-alternated three-method comparison. All 46 focused tests
pass on A100; native stress also passes ASan/UBSan there and TSan locally.

Held-out data showed log-size interpolation could overestimate CPU cost between
widely spaced samples. Added optional linear interpolation while preserving the
old policy default, then recalibrated and validated at fresh sizes. CPU backlog
was expanded to include preprocessing/control, because feature-only depth missed
contention with those stages. The revised policy still regresses against fixed
GPU at some intermediate sizes. Keep those results and leave queue-model tuning
and Python dispatch overhead as explicit open tasks.

The final long-tail run is repeated without concurrent development tests. A
previous tail run is retained as exploratory data because remote sanitizer work
was issued during that earlier run window. Plots prefer the clean repetition.

## Submission-overhead follow-on

Host/device stage plots suggest testing CUDA Graph replay of the fixed H2D,
reduction, and D2H sequence. Add a separate graph transport and compare all four
event/blocking and graph-event/graph-blocking variants in alternating order.
Review: capture only warmed, fixed-address buffers; retain the same event-based
ownership contract and drain behavior. Measure graph replay as one GPU interval
rather than inventing per-copy timing inside capture. Do not reuse the existing
event-transport calibration for this new transport. Keep arbitrary native DAG
and native GPU submission as open tasks even if graph replay improves latency.

Review outcome: graph capture and replay pass two additional A100 integration
cases (48 focused tests total). The initial graph comparison exposed zero-time
condition-variable waits in the native poll APIs. Replaced them with direct
mutex-protected checks, rebuilt, passed tests/sanitizers again, and recalibrated
because the native source identity and timing changed. The `a100-final-*` datasets
are the post-fix comparisons; initial datasets remain archived as development
evidence. Do not mix their costs in one adaptive calibration or headline result.
