# Embedded feedback loop: local native validation, 2026-10-18

Hardware: Apple M4 Pro, macOS, no CUDA device. Native C++ worker runtime and real
CPU reference reduction, 4,096 float64 input elements per lane, 2 workers,
requested polling interval 0 us (service-thread yield). Three independent trials
per configuration, 20 warmup and 200 measured frames per lane per trial.

| Independent loops | Median frame latency | p95 frame latency | Aggregate frames/s |
| --- | ---: | ---: | ---: |
| 1 | 23.375 us | 27.667 us | 42,426 |
| 4 | 42.042 us | 59.459 us | 90,761 |

Each table entry is the median of the three corresponding trial summaries,
including the p95 column (median of per-trial p95 values). Throughput uses each
trial's measured-frame time window. Full raw traces, warmup samples, stage spans,
feedback outputs, worker counts, and source/library hashes are preserved in
[one-loop JSON](2026-10-18-macos-cpu-lanes1.json) and
[four-loop JSON](2026-10-18-macos-cpu-lanes4.json).

All 3,300 frames including warmup matched the independent closed-form feedback
oracle, with maximum absolute error 1.15e-14 or less. Native worker task counts
equaled two CPU stages per frame. These runs validate repeated feedback and
native thread/buffer reuse on this host.

Four loops increased aggregate throughput while increasing per-loop latency.
The largest median stage in the four-loop configuration was the external
service queue (28.458 us), indicating contention for the single Python external
service in this CPU-reference configuration. The extra independent work can
hide some waits, but does not make each feedback loop faster. Medians of separate
stages need not sum to the median total.

Verification: 38 focused Python tests passed, 2 real CUDA integration tests
skipped for lack of CUDA. Standalone address/undefined-behavior and thread
sanitizer runs each passed 4,000 frames, reordered independent completions, and
100 cancellation/destruction cycles without reported errors.

No event-versus-blocking CUDA comparison or GPU speedup is established here.
A10/A100 runs are pending instance access; GH200 coherent-atomic and original
Hopper VDCores attention measurements are separate experiments. The new loop
is a native fixed-graph prototype with a synthetic feature reduction, not a
real robotics workload or a general CPU/GPU instruction compiler.
