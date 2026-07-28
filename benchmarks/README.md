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

For top-k placement studies, both the PoolInst and DeepEP V1 harnesses accept
`--route-placement`:

- `source-local`: all routes stay on the source PE, the locality upper bound;
- `remote-clustered`: all routes use one forced remote PE;
- `clustered`: each token uses one balanced destination PE;
- `spread`: each token touches every PE round-robin.

These are benchmark-controlled routing patterns, not runtime modes. They make
the cost of pool-slice/expert ownership visible without adding placement logic
to VDCores.

The checked-in reference is intentionally dense: one expert-major ring
all-reduce for dispatch and one token-major ring all-reduce for return. It is a
stable NCCL comparison boundary, not a claim that production sparse EP must be
implemented as all-reduce. Keep `NVSHMEM_IBGDA_NUM_RC_PER_PE` and mapping
sweeps in the benchmark environment; they are not runtime operators.

`deepep_low_latency_reference.py` is the production sparse comparison
boundary. It imports an externally built DeepEP V1 package and measures the
actual low-latency IBGDA dispatch/combine kernels with CUDA events and MPI
rank-maximum reduction. The default shape is the production-style 128-token,
7168-hidden, top-8 case, while `--dispatch-dtype bfloat16` provides a
byte-matched comparison to the current pool protocol. DeepEP remains an
external dependency and is never linked into VDCores.

`deepep_v2_reference.py` is the corresponding current DeepEP V2 cached-decode
boundary. It uses V2's token-deduplicated dispatch and weighted token-major
combine through an external NCCL Gin build. DeepEP V2 requires NCCL 2.30.4 or
newer; keep that library and the DeepEP build outside this repository and
preload the same NCCL runtime used to link the extension. The script is a
benchmark helper only and does not change VDCores profiling or runtime code.

The tested Vista DeepEP V2 build at commit `dd758ca` linked against external
NCCL 2.30.7, but cross-node Gin initialization stopped after topology probing
and timed out before a timed iteration. Do not substitute V1 timings for V2 or
claim a V2 comparison until that bootstrap issue is resolved.

`host_sgl_probe.py` is the registration capability gate for an experimental
Grace-hosted scatter/gather transport. It attempts CUDA DMA-BUF registration
first and legacy GPU peer-memory registration second on an actual NVSHMEM HBM
range.

`host_sgl_benchmark.py` and `host_sgl/host_sgl_verbs.cc` are the isolated
two-PE transport experiment. They build RC QPs, concatenate multiple
non-contiguous HBM rows through true multi-SGE RDMA writes, batch independent
destination messages into one doorbell, and use one ordered readiness write
per message with one local completion per batch. The receiver verifies every
byte and readiness sequence. Its host post-to-CQ timings are external
transport results, never VDCores `g_events` results.

The ABI-v3 path also provides an ordinary-host-memory coherent request ring.
Use `--submission both` to compare direct host submission with the asynchronous
GPU-producer/Grace-consumer path. Ring slots retire only by monotonic
generation; no timeout participates in the protocol.

Build and run only in a compute allocation:

```bash
make -C benchmarks/host_sgl
NVSHMEM_DISABLE_NCCL=1 ibrun -n 2 \
  python benchmarks/host_sgl_benchmark.py \
  --rows 9 --batch-depth 14 --submission both --mode sgl
```
