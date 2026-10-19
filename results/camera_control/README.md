# A100 reference study: task graphs and camera-control replay

**This is reference/data collection, not an A100 port of VDCores.** The original
Hopper CUDA runtime, kernels, instruction encoding, launcher, and build target
are unchanged. These ordinary Torch measurements inform future changes; they do
not measure VDCores memory/compute virtual cores or establish VDCores speedups.
Python remains the coordinator intentionally. Do not adopt this reference
executor wholesale as a replacement for the original runtime.

[Figure gallery](figures/README.md) · [Trial summaries](figures/summary.json) ·
[Reproduction workflow](../../agents/workflows/camera-control-experiments.md)

## What was evaluated

Camera replay -> CPU ROI/normalization -> parallel lane/obstacle image processing
on CPU or GPU -> CPU steering/speed output -> previous-command feedback into the
next frame. A reusable Python task graph declares inputs, dependencies, and legal
placements. Each independent loop retains one frame context; CPU futures and
GPU D2H completion events publish results before dependent tasks execute.

This is classical, non-learned perception using generated bright-lane/red-obstacle
scenes. It is not a trained neural model, actual robot, physical closed-loop
simulation, or validated real-world perception system. NPZ recording support is
implemented, but meaningful real recordings and application accuracy criteria
are still needed. The replayed world does not respond to commands; only the next
ROI and controller state do.

## Experimental design and evidence

- Hardware: A100-SXM4-80GB; AMD EPYC 7J13 host, 30 visible CPUs. Ubuntu 22.04,
  Torch 2.7.0/CUDA 12.8, NumPy 1.21.5, NVIDIA driver 580.105.08.
- Nine configurations, four execution methods, three trials: **108 evaluation
  runs**, **117,600 correctly checked frames**, **110,400 measured frames** after
  warmup. Calibration task calls are separate and not counted as feedback frames.
- 128x192 and 480x640 RGB, one/four independent loops. Two application CPU
  workers, one Torch intra-op thread per worker, four GPU stream slots. Same
  formulas, input data, CPU-only preprocessing/control, and explicit GPU transfer
  boundaries across methods; serial/ready-task GPU differ only in scheduling.
- `cpu`: ready-task CPU perception. `serial-gpu`: one outstanding task globally.
  `dag-gpu`: ready independent tasks/loops may overlap. `adaptive`: calibrated
  CPU/GPU choices per perception task with conservative pending-work estimates.
- First eight generated frames calibrate isolated service costs; the other 16
  evaluate feedback. Three calibration trials/device/task, 20 measured samples
  each. Method order rotates/reverses in evaluation; three trials do not fully
  counterbalance four methods.
- Every warmup and measured output passes an independent NumPy feedback oracle;
  timestamp checks enforce dependency order and sequential feedback per loop.
  All **69 focused tests pass on A100**; local CPU runs pass 60 and skip nine
  real-CUDA cases. The original-Launcher adapter has fake-Launcher contract tests,
  not completed VDCores workload binding or Hopper hardware validation.
- Saved compressed raw task/frame traces, source hashes, environment, placement
  decisions, competing-work counts, and exact hash-verified RGB fixtures. All
  five measured source-file hashes match the committed implementation in all
  108 runs. Remote Git metadata retains its earlier base revision and dirty
  state; use the explicit source hashes, not that base revision alone.

No GPU tests or separate benchmarks overlapped these nine final configurations.
The temporary early smoke run was interrupted during development and produced
no result file. The instance was idle after completion; it was not terminated.

## Main observations

Tables report medians of the three per-trial latency summaries, not pooled-frame
medians or confidence intervals. Error bars in the gallery span trial min/max.

### Image size changes the useful placement

| Image / loops | CPU only | Serial GPU | Ready-task GPU | Adaptive |
| --- | ---: | ---: | ---: | ---: |
| 128x192 / 1 | 1.338 ms | 2.202 ms | 1.995 ms | 1.598 ms |
| 128x192 / 4 | 5.332 ms | 8.739 ms | 7.452 ms | 6.595 ms |
| 480x640 / 1 | 4.288 ms | 3.137 ms | 2.966 ms | 3.109 ms |
| 480x640 / 4 | 14.364 ms | 12.098 ms | 7.773 ms | 7.959 ms |

The CPU wins on small images on this host. On large images/four loops, ready-task
GPU execution reduced observed median response **35.7%** relative to serial GPU
service. This supports investigating dependency-ready host/device boundaries;
it is not evidence that VDCores is 35.7% faster, or a prediction for GH200.
The serial baseline is not an optimized fused/GPU-resident application.

### Lower median can hide worse slow frames

The longer 480x640/four-loop run measures 1,000 frames/loop/trial, or 12,000
measured frames per method. A missed deadline means response exceeded 10 ms.

| Method | Median | p99 | Aggregate frames/s | Deadline misses |
| --- | ---: | ---: | ---: | ---: |
| CPU only | 15.061 ms | 20.049 ms | 255.6 | 11,942 / 12,000 |
| Serial GPU | 11.053 ms | 12.164 ms | 351.4 | 7,227 / 12,000 |
| Ready-task GPU | 7.957 ms | 9.541 ms | 478.3 | 32 / 12,000 |
| Adaptive | 7.384 ms | 14.919 ms | 451.4 | 2,013 / 12,000 |

Adaptive improves median response by 7.2% versus ready-task GPU but makes p99
56.4% worse, lowers aggregate throughput, and misses many more deadlines. Its
largest observed response was 105.311 ms versus 11.164 ms for ready-task GPU.
Keep this regression visible: selecting by isolated median service cost is not
enough for predictable application response. Queue estimates currently omit
calibrated fixed CPU-stage costs and conservatively serialize pending GPU work.

Under two competing CPU threads, adaptive similarly reduced median response
from ready-task GPU's 8.137 ms to 7.396 ms, but worsened p99 from 8.912 ms to
14.857 ms. With competing GPU pulses, both paths slowed; adaptive p99 remained
worse (15.526 versus 11.961 ms). These loads are specified competing operations,
not calibrated utilization percentages.

### Arrival-to-output latency matters more than active processing alone

Four loops released every 10 ms request an aggregate 400 frames/s. Ready-task
GPU generally sustained that rate, with a median-of-trials response of 8.413 ms;
88/2,400 measured frames still missed 10 ms. CPU-only could not sustain it:
median response including accumulated backlog was 693.976 ms, despite much
smaller per-frame active processing spans. Serial GPU had a large trial spread:
median responses 463.859, 17.306, and 9.804 ms. Do not hide that variability or
infer a reliable deadline guarantee from the middle value.

At 2 ms per loop (2,000 requested frames/s), every method overloaded and every
measured frame missed 10 ms. Ready-task GPU still processed faster than serial
GPU, but its median response including backlog reached 758.445 ms. A runtime
cannot fix an unsustainable arrival rate through placement alone; admission,
backpressure, or a deliberate frame-drop policy must be defined. This experiment
retains all frames and does not hide overload by dropping them.

## How this should inform future VDCores work

| Reference finding | Design question for the actual runtime |
| --- | --- |
| Small tasks lose on GPU once copies/coordination are included | Which operations should remain CPU-capable, and where is the crossover on the actual coherent host? |
| Independent ready branches outperform serial service in the large case | Can CPU completion release dependent VDCores work without whole-chain host blocking? The GPU VM already has internal dependencies. |
| Adaptive median gains can worsen tails and throughput | What queue telemetry, uncertainty margin, and deadline-aware policy are needed before enabling adaptive placement? |
| Periodic overload dominates end-to-end response | What are the application's arrival, admission, and stale-frame/drop semantics? |

Do not transfer A100 numerical thresholds, copy costs, or speedup percentages to
GH200. The host CPU, interconnect, coherent-memory semantics, and persistent
VDCores GPU execution differ. CUDA Graph gains from the earlier standalone loop
also do not imply the same gain for an already persistent GPU VM. Use these
results to choose hypotheses, then change and validate one actual VDCores path
at a time on compatible hardware. Reuse existing attention/GEMV tasks where
possible rather than adding new VDCores kernels just to match this reference.

Remaining gates: real application recordings/accuracy, stronger fused/resident
and camera CUDA Graph baselines, CPU-stage queue calibration, more trials and
longer observation windows, actual VDCores workload binding, and compatible
Hopper/GH200 measurements. No core-runtime architecture change is justified by
these reference numbers alone.

## Files and reproduction

- `a100-taskgraph-v1/*.json.gz`: nine final configuration files, including all
  trials, warmup, task timing, decisions, outputs, and correctness checks.
- `fixtures/generated-128x192.npz`, `fixtures/generated-480x640.npz`: exact RGB
  arrays, verified against the measured frame hashes before archiving.
- `local-final.json.gz`: current-source CPU validation, 1,320 checked frames.
  Earlier `local-smoke`, `local-worker-budget`, and `local-event-wakeup` archives
  are development snapshots, not final performance baselines. Their hashes and
  code versions differ; they are not combined with A100 measurements.
- `figures/`: seven PNG/SVG figures, captions, and complete trial summaries.

Run `python tools/plot_camera_control.py` to regenerate the gallery and fixtures.
See the workflow linked above to rerun the sequential matrix. Keep real recorded
inputs separate from generated fixtures, and never label Torch runs as execution
of the original VDCores GPU VM.
