# External Benchmarks

External baselines and host transport helpers live here. VDCores operator and
protocol code remains under `include/` and `python/dae/`; NCCL, DeepEP, and
verbs progress code is never linked into the main runtime.

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

`uccl_ep_reference.py` adds the UCCL-EP low-latency path through UCCL's
DeepEP-compatible wrapper. It supports BF16 for a byte-matched PoolInst
comparison and FP8 for UCCL's inference-oriented wire format. UCCL's GPU
kernels, registered HBM, CPU proxy threads, and RDMA transport remain in the
external UCCL checkout; only a benchmark adapter lives here.

`triton_distributed_ep_reference.py` adds Triton-distributed's optimized
`EPLowLatencyAllToAllLayer`. That kernel always quantizes BF16 activations to
FP8 online, so its numbers are labeled FP8 and must not be presented as a
byte-matched BF16 comparison. One untimed operator pass checks received expert
counts against globally gathered routes and checks the weighted identity return
against local FP8 quantize/dequantize. Each measured iteration dequantizes that
iteration's dynamically ordered receive buffer, but CUDA events exclude this
identity-expert work from both dispatch and combine timing.

Both adapters share the deterministic placement definitions in
`ep_baseline_common.py`, report the rank-maximum median of CUDA-event samples,
and emit an `ep-baseline-json:` line for result collection. These CUDA events
are external-baseline measurements and never use or alter VDCores `g_events`.
Pinned source/build and Vista launch instructions are in
`agents/workflows/external-ep-baselines.md`; the measured 2/4/8-PE matrix is in
`agents/knowledge/runtime/external-ep-baselines.md`.

Typical one-GPU-per-node launches after installing each dependency externally:

```bash
UCCL_EP_ROOT=/home1/11362/depctg/projects/uccl \
ibrun -n 2 python benchmarks/uccl_ep_reference.py \
  --tokens-per-pe 128 --dispatch-dtype bfloat16

TRITON_DISTRIBUTED_ROOT=/home1/11362/depctg/projects/Triton-distributed \
ibrun -n 2 /path/to/triton-dist-env/bin/python \
  benchmarks/triton_distributed_ep_reference.py --tokens-per-pe 128
```

`host_sgl_probe.py` is the registration capability gate for an experimental
Grace-hosted scatter/gather transport. It attempts CUDA DMA-BUF registration
first and legacy GPU peer-memory registration second on an actual NVSHMEM HBM
range.

`host_sgl_benchmark.py` and `host_sgl/host_sgl_verbs.cc` exercise the external
data plane. The default scalable path uses one mlx5 DCI per PE with one DCT
target and concatenates up to eight non-contiguous HBM rows per RDMA write; RC
remains a comparison fallback. Data is followed by one ordered readiness write
per PoolInst message. Host post-to-CQ timings are external transport results,
never VDCores `g_events` results.

The ABI-v7 coherent ring uses ordinary host memory and monotonic generations;
no timeout participates in the protocol. `pool_slice_host_e2e.py` is the full
weighted-EP validation path. It uses the host only for dispatch/return payload
delivery and ready notification; metadata, dependencies, dynamic gather,
reduction, scatter, and retirement remain the ordinary PoolInst state machine.

Build and run only in a compute allocation:

```bash
make -C benchmarks/host_sgl
NVSHMEM_DISABLE_NCCL=1 ibrun -n 2 \
  python benchmarks/host_sgl_benchmark.py \
  --rows 9 --batch-depth 14 --submission both --mode sgl
```
