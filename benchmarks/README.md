# External Benchmarks

Benchmark-only backends live here rather than in `src/`, `include/`,
`python/dae/`, or the runnable VDCores applications.

`pool_slice_nccl_compare.py` compares the unified pool-slice dynamic-read
protocol with a dense two-ring NCCL reference implemented in
`nccl_ep_reference.py`. The VDCores side is timed exclusively from `dae2`'s
internal `g_events` profile space. CUDA events are used only around the
external NCCL reference.

The harness also reports protocol bytes, batches, and merged signal counts so
transport decisions can be evaluated before low-level profiling.

The checked-in reference is intentionally dense: one expert-major ring
all-reduce for dispatch and one token-major ring all-reduce for return. It is a
stable NCCL comparison boundary, not a claim that production sparse EP must be
implemented as all-reduce. Keep `NVSHMEM_IBGDA_NUM_RC_PER_PE` and mapping
sweeps in the benchmark environment; they are not runtime operators.
