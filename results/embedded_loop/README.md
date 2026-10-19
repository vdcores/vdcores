# VDCores embedded loop: implementation and A100 results

The branch now executes a repeating native-CPU -> GPU -> native-CPU feedback
loop on an A100, supports independent loops sharing the GPU, and makes measured
CPU/GPU placement decisions. This is a fixed-graph prototype with synthetic
signals, not yet a real embedded application or arbitrary VDCores graph compiler.

See the [figure gallery](figures/README.md), its [complete summary table](figures/summary.json),
the [native ownership contract](../../agents/knowledge/runtime/embedded-loop-native.md),
and [run instructions](../../agents/workflows/embedded-loop-experiments.md).

## What changed

- Persistent C++ workers execute preprocessing, mean-square feature computation
  when placed on CPU, and controller output. Previous action changes the next
  frame. Buffer ownership and lane/frame tokens prevent stale completion/reuse.
- CUDA event completion releases dependent CPU control only after D2H finishes.
  Cancellation joins native writers and drains owned GPU streams.
- The measured placement policy now executes inside the loop. Live backlog
  includes queued/running preprocessing, feature, and control work plus pending
  GPU requests. Policy files validate hardware, worker count, workload/transport,
  native source, software version, and supported size range.
- Held-out validation exposed premature GPU selection from log-size cost
  interpolation. Added optional linear interpolation and denser calibration.
- CUDA Graph replay captures the same fixed copy/reduction/copy stage. A first
  comparison exposed zero-time condition-variable waits in native polling; those
  now check state directly. Rebuilt, retested, recalibrated, and reran final data.

## Final measurements

Hardware: NVIDIA A100-SXM4-80GB, 108 SMs; AMD EPYC 7J13 host with 30 visible CPUs.
Lambda Stack Ubuntu 22.04, Python 3.10.12, Torch 2.7.0/CUDA 12.8, driver 580.105.08.
The earlier screenshot's 40 GB listing differs from the actual 80 GB instance.

### Captured GPU submission

Three independent trials per configuration and mode, 1,000 measured frames plus
100 warmup frames per loop. Method order reverses between trials. Same CPU work,
GPU operations, transfers, buffers, worker count, and poll setting. Table entries
are medians of per-trial median full-frame latency, after the zero-time-poll fix.

| Input values | Loops | Ordinary events | Graph + events | Reduction |
| --- | ---: | ---: | ---: | ---: |
| 4,096 | 1 | 212.495 us | 131.872 us | 37.9% |
| 4,096 | 4 | 795.178 us | 405.551 us | 49.0% |
| 65,536 | 4 | 756.259 us | 386.510 us | 48.9% |

The gain comes from reducing repeated submission/coordination work, rather than
changing the feature calculation. It does not establish raw atomic or kernel
latency. Captured execution still uses Python for replay submission/completion.
The graph transport needs a separate placement calibration before adaptive use.

### Longer tail validation

The final post-fix run measures 10,000 frames per loop, four loops, and three
trials per mode (120,000 measured frames per mode). All use 65,536 input values.
Entries below are medians of trial summaries; deadline misses sum the measured
frames across trials against a relative 1 ms deadline.

| GPU service | Median | p95 | p99 | Frames/s | 1 ms misses / 120,000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ordinary events | 741.867 us | 766.540 us | 848.620 us | 5,367 | 177 |
| Ordinary blocking | 794.279 us | 812.387 us | 852.960 us | 5,039 | 42 |
| Graph + events | 380.759 us | 413.450 us | 440.721 us | 10,383 | 17 |
| Graph + blocking | 508.831 us | 534.572 us | 549.303 us | 7,834 | 23 |

Graph + events reduced observed median latency 48.7% and p99 48.1% versus
ordinary events, while nearly doubling aggregate frame rate in this experiment.
Ordinary events have a slightly better median than ordinary blocking but more
rare deadline misses, showing why median alone cannot establish predictability.
No hard real-time guarantee follows from these finite runs.

### Measured adaptive placement

Same-host CPU-worker versus ordinary-event GPU calibration measures the remaining
feature stage, including dispatch, copies, and observed completion. At 64 values,
CPU cost is 11.2 us versus GPU 185.9 us; at 1,048,576 values, CPU is 995.0 us
versus GPU 496.3 us. This demonstrates why size-dependent placement matters for
this workload; it does not establish a universal threshold.

Three trials with alternating CPU/GPU/adaptive method order; 300 measured and 30
warmup frames per loop. The two smaller sizes below are held out from the final
calibration grid; the largest size is a calibrated endpoint.

| Values | Loops | Fixed native CPU | Fixed GPU events | Adaptive |
| --- | ---: | ---: | ---: | ---: |
| 131,072 | 1 | 298.798 us | 389.939 us | 320.498 us |
| 131,072 | 4 | 586.525 us | 832.750 us | 533.722 us |
| 524,288 | 1 | 1,115.266 us | 941.601 us | 971.974 us |
| 524,288 | 4 | 2,348.544 us | 1,172.877 us | 1,171.427 us |
| 1,048,576 | 1 | 2,194.612 us | 1,707.731 us | 1,733.152 us |
| 1,048,576 | 4 | 4,488.918 us | 2,358.162 us | 2,360.902 us |

At 131,072 values and four loops, adaptive placement mixed CPU and GPU execution
and reduced observed median latency 9.0% below the better fixed baseline. At
larger sizes it mostly chose GPU and approached the fixed-GPU median. It is not
consistently faster: the single-loop CPU choice adds about 21.7 us of policy/
coordination overhead, and tail latency can also regress. For example, the
four-loop 1,048,576-value adaptive p99 was 3.661 ms versus fixed GPU 2.547 ms in
these short trials. Tail estimates from 300 frames are coarse. Keep these
limitations visible; graph-replay gains and adaptive-placement gains are from
different experiments and must not be added or multiplied into one result.

## Evidence, reproducibility, and limitations

All 48 focused tests pass on A100; local CPU-only execution passes 43 and skips
five real-CUDA cases. Native ASan/UBSan stress passes on Linux, and ASan/UBSan plus
TSan pass on the local Mac. Each native stress run exercises 4,000 frames and
100 cancellation/destruction cycles. Every recorded frame is checked against an
independent feedback oracle and dependency timestamp ordering.

Raw traces and decisions are losslessly archived as JSON gzip. The plotting
tool reads archives directly and exports PNG, editable SVG, captions, and an
exhaustive summary. Metadata records environment, dirty Git state, native source/
library hashes, and (in the final runners) Python source hashes. Exploratory
datasets were collected from intermediate implementations and are retained as
development evidence. Use the `a100-final-*` datasets and
`a100-placement-calibration-final-v2.json.gz` for the final implementation.

The initial shared-loop/polling sweep and first placement experiment explain why
the implementation changed. One earlier tail series may overlap a remote
sanitizer run and is retained as exploratory. Clean repetitions and the final
post-fix tail series are separate. Historical GH200 atomic/attention data is
plotted separately because the hardware, transport, and workload differ.

The workload is synthetic float64 energy reduction with a toy controller. There
is no real camera/model, periodic sensor arrival schedule, hard real-time bound,
power-efficiency conclusion, or multi-GPU claim. Relative deadline misses start
when each frame becomes ready. Measurements include instrumentation and host
submission overhead; GPU event intervals are not exclusive raw kernel time.

## Remaining work, in order

1. Move GPU submission, completion service, and placement into native execution;
   use the Python implementations as reference behavior and retain the recorded
   event/graph baselines. Refine queue estimates and test tail regressions.
2. Add general operation descriptors and dependency lowering, with fan-out/join,
   buffer ownership, and failures beyond this fixed graph.
3. Calibrate graph transport before combining capture with adaptive placement.
4. Connect the native CPU workers to the original VDCores attention/Hopper
   instruction graph on compatible hardware; rerun coherent handoff calibration.
5. Select a real sensor/model/controller workload with the project lead, obtain
   representative inputs or traces, and validate actual sensor-to-action latency.
6. Extend sharing to arbitrary GPU VCore graphs, define scheduling/fairness goals,
   and complete upstream design review and integration coverage.
