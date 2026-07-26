# External Benchmarks

Benchmark-only backends live here rather than in `src/`, `include/`,
`python/dae/`, or the runnable VDCores applications.

`ep_pool_nccl_compare.py` compares the integrated memory-pool EP program with
a dense NCCL ring reference. The VDCores side is timed exclusively from
`dae2`'s internal `g_events` profile space. CUDA events are used only around
the external NCCL reference.

`pool_slice_nccl_compare.py` applies the same dense ring reference to the
pool-owned dynamic-read protocol. Its cost model reports descriptor, route,
payload, RMA, and signal counts so the first optimization decision comes from
protocol structure rather than profiler output.
