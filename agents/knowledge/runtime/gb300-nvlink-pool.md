# GB300 NVLink Pool Port

The GB300/GB200 pool-only build is selected with `make gb300-nvlink-pyext`.
It targets `sm_100a`, selects only `OP_TERMINATEC`, and therefore does not
instantiate Hopper compute kernels. The default local backend has no NVSHMEM
bootstrap or runtime dependency. `DAE_POOL_DATA_PATH=nvshmem` still builds the
original NVSHMEM/IBGDA-capable path for compatibility and compile-time A/Bs.

`DAE_POOL_DATA_PATH=local` preserves `PoolSliceConfig`, metadata envelopes,
ordered queues, and dependency generations from the remote scheduler-worker
protocol. `python/dae/local_pool.py` packs every rank's fields into an arena;
the runtime installs all peer arena bases in device constants. Metadata and
payload use ordinary peer-global loads/stores over NVLink, with system-scope
release/acquire operations for publication. There is no host copy or
NVSHMEM operation in the timed path.

The multimem reduction plane is allocated with CUDA driver VMM in
`src/torch_local_pool_runtime.cpp`: one physical allocation per GPU is bound
to a multicast object and mapped through both unicast and multicast virtual
addresses. Destinations accumulate FP32 expert partials and round once to
BF16 in their physical backing. The source reads all destination partials
with `multimem.red.relaxed.sys.global.add.v4.bf16x2`; system-scope multimem
zero/publish counters order reuse and completion.

## Historical NVSHMEM-backed prototype

On the four-GPU GB200 node, NVSHMEM 3.4.5 diagnostics report all four PEs in
the P2P list and IBGDA disabled. Two-PE dynamic-read correctness passes. Both
the original NVSHMEM data path and the direct path currently stall in the
four-PE pool program, so this is a GB300/NVSHMEM-3.4.5 integration blocker,
not evidence of a direct-copy regression. The system NVSHMEM 3.6.5 package
uses a different device-link distribution and must not be mixed with the
3.4.5 device archive.

Sparse routing is a separate correctness gate: three PEs with top-k=3 and a
single pool CTA complete, while top-k=1 stalls with the same transport and CTA
shape. Debug sparse completion before attributing a multi-PE stall to NVLink
transport or worker scheduling.

## Two-GPU external baselines

On physical GPUs 1 and 2, NCCL 2.29.7's dense two-ring reference (BF16 hidden
4096, one expert/PE, top-k 1) measured 0.789/0.795/0.716 ms total at
8/32/128 tokens per PE. DeepEP v1.2.1 built for `sm_100a` and its low-latency
same-node NVLink path (BF16 hidden 7168, eight experts/PE, top-k 8 clustered)
measured 0.0768/0.1720/0.2254 ms total at 32/128/256 tokens per PE. Both use
rank-maximum medians from 30 samples after 10 warmups. These are different
traffic contracts and must not be presented as a byte-matched head-to-head.

Matched within each contract, current direct-NVLink PoolInst totals are
0.4358/0.3912/0.4822 ms at the NCCL 8/32/128-token shape, or
0.55x/0.49x/0.67x the dense ring latency. At the DeepEP 32/128/256-token
production shape, PoolInst totals are 0.4114/0.3580/0.5183 ms, or
5.36x/2.08x/2.30x DeepEP latency. Pool telemetry places first payload around
0.022--0.035 ms; metadata closure and ordered queue retirement, followed by
the weighted return at larger shapes, are the primary optimization targets.

### Historical matched local-NVLink matrix (2026-07-31)

For 128 tokens/PE, hidden 7168, eight experts/PE, top-k 8 clustered BF16,
10 warmups, and 30 samples, the installed multimem pool measured 0.098416 ms
on two GPUs and 0.114112 ms on three, with exact BF16 output. These clustered
figures are historical transport-development controls and are not valid
global-random-top-8 production results.

DeepEP V1.2.1 measured 0.1389 ms on two GPUs and 0.2628 ms on three. NCCL-EP
LL measured 0.125424 ms on two GPUs. NCCL-EP LL accepts 2, 4, or multiples of
8 ranks, so no three-rank NCCL-EP value exists.

## Rebased scheduler-worker specialization

The GB300 work is rebased on `origin/mempool-ep` commit `cc310f5`. The remote
scheduler-worker protocol, its separate metadata/data planes, immutable
ReduceAdd commands, ordered queue retirement, and dynamic expert-copy workers
remain authoritative. The local specialization changes transport and role
placement without changing those command semantics:

- metadata publishers, source-route expanders, remote SEND workers, and the
  self-pack worker use disjoint CTAs when the assembly is large enough;
- rank zero remains scheduler-only, while workers publish remote metadata
  before self metadata and stream each unique source token once into delivery;
- Copy workers overlap delivery-to-expert placement with later NVLink sends;
- the scheduler posts a source's reduction commands as soon as its END heads
  retire, and STOP has a descriptor-free executor fast path;
- multimem uses four independent `v4.bf16x2` vectors per loop iteration;
  the SM100a kernel has no local-memory spills.

## Corrected NVL72 global-top-8 scan (2026-08-02)

This section supersedes the invalid 2026-08-01 clustered/peer-direct scan.
Those results assumed that all eight experts for a token occupied exactly one
destination GPU. Global random top-8 provides no such guarantee.

All four allocated B300 hosts belong to one rack-scale MNNVL domain. Crossing
four GPUs does not switch the data path to a NIC. CUDA Fabric VMM mappings
carry metadata, activation, signals, and results across the NVLink backplane;
MPI/TCP exchanges opaque handles and bootstrap/control messages only.

### Return correctness

For one source token, independently selected experts can reside on several
destination GPUs. Each destination owns only a partial weighted result.
Having those GPUs issue ordinary stores to the same source row is a data race:
the stores overwrite one another instead of reducing. Therefore
`peer_direct` return is hard-rejected in the Python buffer and build APIs.
Both legacy peer-direct instruction constructors also raise immediately and
are absent from the wildcard export. Opcodes 6 and 7 and their CUDA executor
assemblies are absent from the compiled PoolInst registry, so even a raw old
opcode is rejected as unsupported rather than selecting peer-direct code.
The valid measured backend is `source_gather`: destinations publish distinct
expert-output rows, and the source reads and reduces its eight route
contributions.

`DAE_POOL_LOCAL_DIRECT_SCATTER=1` remains valid only for dispatch. Fixed
source stripes give every route a distinct expert-input slot, so destination
writes do not alias. This optimization does not return final token rows and
does not weaken the source-gather requirement.

### Retained PoolInst execution model

The corrected path still uses the PoolInst protocol and dynamic executor
dispatch. A route handle precomputes stable metadata once. Each invocation
rearms its monotonic sequence, publishers expose route/data readiness, the
scheduler emits immutable reserve/data/end work, and ordinary executor CTAs
claim those instructions dynamically. Dispatch DATA work either copies a
compact source row to delivery and then to expert input or directly scatters
it into its unique expert slot. After expert identity work, source-gather
return plans read the routed result rows, multiply by the 1/8 weights, reduce,
and publish the source-token output. No instruction assumes one destination
per token.

The retained performance changes are:

- reuse of static route metadata and rearming only the invocation sequence;
- source-gather-specific route plans that omit unused reverse maps and fine
  masks;
- cached receive descriptors and validation;
- deterministic source stripes, removing repeated receive-route reads;
- dispatch-only direct activation scatter at four or more GPUs;
- direct-scatter address hoisting: form at most eight disjoint expert-slot
  pointers once per activation row and reuse them across all 896 16-byte
  vectors, eliminating repeated per-vector shuffle/address work;
- CTA-shared top-8 descriptor predecode and a descriptor-proven uniform-1/8
  reduction specialization;
- parallel source-CTA completion scanning and source-owned peer-read
  reduction, with exactly one returned-row writer;
- a 40-logical-executor cap through four GPUs and a 64-executor cap above
  four. The lower cap is optimal at four GPUs but is non-live at eight or more
  because independently ordered source queues and STOPs can be starved;
- 128 PoolInst blocks with tuned group limits 2/8/4/4/2 at
  2/4/8/12/16 GPUs;
- the spill-free ILP4 vector loop; ILP8 regressed and was reverted.

Fused token scatter, four-shard fused scatter, helper-only execution,
split-vector return, route-per-warp shared staging, paired peer loads,
non-returning completion counters, shared address caching, and multiple
lower-CTA policies were profiled and rejected for latency regression or
liveness. Group limit 8 is non-live at eight GPUs. Peer-direct figures are
excluded categorically, regardless of speed.

A reduction-safe destination-partial experiment was also rejected. Each
destination reduced its local experts into a disjoint
`(destination, source token)` inbox plane, after which the source performed
the only final output write. It was correct and reduced the four-GPU return
tail to 10--16 us, but reverse-map expansion and destination reduction raised
total latency to 0.112928--0.116160 ms; eight-way row sharding reached
0.133632 ms. Existing valid multimem and partial-forward controls measured
0.135232 ms and 0.257120 ms respectively. The accepted implementation remains
expert-row source-gather.

### Corrected matched results

The contract is BF16 width 7168, 128 tokens/GPU, eight experts/GPU,
deterministic random global top-8 with seed `20260802`, uniform 1/8 weights,
identity experts, 10 warmups, 30 measured iterations, changing inputs, and
poisoned outputs. Times are medians of rank-maximum device-event samples.

| GPUs | Group limit | Dispatch (ms) | Return tail (ms) | VDCores total (ms) | NCCL-EP (ms) | VDCores advantage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 2 | 0.063632 | 0.019376 | 0.081264 | 0.126944 | 35.98% |
| 4 | 8 | 0.059552 | 0.023136 | 0.082688 | 0.110144 | 24.93% |
| 8 | 4 | 0.070976 | 0.026240 | 0.097136 | 0.154400 | 37.09% |
| 12 | 4 | 0.071328 | 0.027616 | 0.098576 | unsupported | — |
| 16 | 2 | 0.075456 | 0.028704 | 0.103840 | 0.159136 | 34.75% |

All VDCores points passed changing-input/poisoned-output validation. Two GPUs
was bit-exact; the direct-scatter points were BF16-close with maximum absolute
error `0.00390625`, explained by a valid change in BF16 accumulation order.
Latency stays within 1.278x from two to sixteen GPUs, and within 1.256x from
four to sixteen. The requested greater-than-20% advantage is achieved at
every NCCL-supported scale; the smallest margin is 24.93% at four GPUs. Two
additional four-GPU long-run validations measured 0.083808 and 0.085728 ms,
also clearing the threshold. Peer-direct opcodes and CUDA assemblies remain
absent from the accepted runtime.

NCCL-EP provenance is nccl4py 0.3.1, NCCL-EP 0.1.0, NCCL 2.30.7, source
revision `5067397c2676d5aed50042fc39e5c8ee96eb0027`, expert-major BF16, QP8,
and library-auto SM/channels. GIN and IB were disabled and MNNVL was enabled.
NCCL-EP does not support 12 ranks.

### Corrected full baseline scan

Communication totals in milliseconds:

| Implementation | Payload/contract note | 2 GPUs | 4 GPUs | 8 GPUs | 12 GPUs | 16 GPUs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| VDCores source-gather | BF16 global top-8 | 0.081264 | 0.082688 | 0.097136 | 0.098576 | 0.103840 |
| NCCL-EP | BF16 global top-8 | 0.126944 | 0.110144 | 0.154400 | unsupported rank count | 0.159136 |
| DeepEP V1 | BF16 global top-8 | 0.1442 | 0.1485 | unsupported without IBGDA RC QPs | unsupported | unsupported |
| UCCL | BF16 global top-8 | 0.256880 | 0.214112 | cross-host path requires RDMA proxy | unsupported | unsupported |
| Triton-distributed | online FP8; not BF16 byte-matched | 0.309664 | 0.341456 | 0.325376 | 0.347488 | 0.348208 |
| Dense NCCL ring control | BF16 top-k 1; two dense all-reduces | 0.560 | 0.829 | 2.740 | 6.033 | 10.698 |
| DeepEP V2 | requires NCCL GIN | unsupported | unsupported | unsupported | unsupported | unsupported |

DeepEP V1 and UCCL are not extended across hosts by silently enabling a NIC;
that would violate the all-NVLink scan. Triton-distributed used NVSHMEM
P2P/MNNVL with `NVSHMEM_REMOTE_TRANSPORT=none`; its FP8 values are useful
controls but not byte-matched. DeepEP V2's capability check fails when GIN is
correctly disabled. The dense ring is a traffic-heavy surrogate, not sparse
expert parallelism.

The shape is DeepSeek-V3 activation width/top-k, not a full DeepSeek-V3 MoE
layer: experts are identity operations and the benchmark uses
`8 * world_size` routed experts rather than the model's full expert count.

Relevant NVIDIA documentation:

- [MNNVL User Guide](https://docs.nvidia.com/multi-node-nvlink-systems/mnnvl-user-guide/overview.html)
- [Multi-Node Tuning Guide](https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/overview.html)
- [NVSHMEM environment variables](https://docs.nvidia.com/nvshmem/api/gen/env.html)
- [CUDA virtual memory management](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/virtual-memory-management.html)

The reproducible launch and acceptance procedure is in
`agents/workflows/gb300-mnnvl-fabric-scan.md`.
