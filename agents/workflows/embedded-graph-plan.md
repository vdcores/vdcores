# Python graph execution and camera-control evaluation

## Agreed scope (2026-10-18)

Keep Python as the control layer. Generalize the fixed-loop experiment, add a
camera-to-control workload, and evaluate complete application latency before
reconnecting to the original Hopper VM. Work on `anant/cpu-gpu-handoff` in small
commits. Existing native-loop implementation, adapters, measurements, and docs
are preserved in four separate commits before this follow-on.

## Implementation sequence

1. Execute named operations with declared input buffers, dependencies, CPU/GPU
   capabilities, and executor identity. Reuse `HeterogeneousScheduler` for state
   validation. Add bounded concurrent frame contexts, CPU futures, CUDA-event
   futures, failure draining, and explicit placement. Test fan-out/join, control
   dependencies, and multiple feedback lanes without CUDA.
2. Add an explicit adapter for a prepared original VDCores `Launcher`, using its
   actual `launch_async` / `synchronize` contract. Only mark it validated after a
   compatible-hardware run. Do not substitute ordinary Torch work under its name.
3. Add camera replay -> CPU preprocessing -> lane/obstacle perception branches
   -> CPU steering/speed control -> next-frame feedback. Implement simple real
   image-processing algorithms, not an untrained model described as inference.
   Support recorded uint8 RGB frames and a reproducible generated scene sequence.
   The generated inputs are a functional fixture, not real-world validation.
4. Compare serial GPU, dependency-ready GPU, CPU-only, and measured adaptive
   execution with the same operations, transfers, inputs, and output validation.
   Collect independent trials, task/frame traces, CPU usage, throughput, per-lane
   tails, periodic-arrival latency and deadline misses. Offer controlled CPU/GPU
   background load and record it explicitly. Keep calibration separate from test.
5. Run CPU tests, then sequential A100 tests and a bounded evaluation matrix.
   Archive raw data, figures, environment/source identities, and limitations.
   Update the root checklist to defer native submission and prioritize these
   Python-led milestones instead.

## Review before implementation

- A100 is not a valid target for the original sm_90a runtime. The Torch adapter
  establishes application/scheduler evidence, not VDCores GPU-VM speedups.
- Graph values are immutable to consumers; all frame values/leases stay alive
  until every submitted CPU/GPU reader drains. No in-place sibling mutation.
  A lane has at most one frame in flight because control affects the next frame.
- GPU completion is a D2H event, never just successful launch or a returned GPU
  tensor. CPU control cannot inspect results until publication.
- A prepared Launcher must not be relaunched while in flight. Its adapter holds
  exclusive ownership through completion and releases retained launch metadata
  using the existing synchronize method. Adapter contract tests are not hardware
  validation of the original VM.
- Dynamic placement must use this workload's measured CPU/GPU costs, including
  transfers and completion; never reuse energy-reduction or GH200 timings.
- Serial and dependency-ready variants differ only in scheduling. Do not compare
  different algorithms or silently leave GPU inputs resident in one baseline.
- Periodic releases use scheduled timestamps, including backlog/lateness. Closed
  loop throughput is reported separately; no hard real-time or physical-robot
  safety claim follows from generated scenes and finite Linux runs.
- Review outcome: proceed with these bounded Python-led changes; retain original
  VDCores reconnection and recorded real-world evaluation as explicit gates.

## User scope clarification

Use A100 only as reference/data collection to inform future VDCores changes.
Do not port or rewrite the original Hopper runtime for A100. The original CUDA
runtime, kernels, instruction format, launcher, and build target are unchanged;
the general graph and camera workload are additive reference experiments. Stop
expanding the reference executor after the current study. A prepared-Launcher
adapter contract is not completed VDCores workload integration. Finish the
already-running bounded measurement matrix and document what it suggests and
what it cannot establish; defer core architecture changes until reviewed.
