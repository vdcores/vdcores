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

### Current matched local-NVLink matrix (2026-07-31)

For 128 tokens/PE, hidden 7168, eight experts/PE, top-k 8 clustered BF16,
10 warmups, and 30 samples, the final installed multimem pool measures
0.098416 ms on two GPUs and 0.114112 ms on three, with exact BF16 output.
Repeated longer two-GPU runs span 0.094720--0.098416 ms; a conservative
three-GPU run measured 0.117568 ms.

DeepEP V1.2.1 measures 0.1389 ms on two GPUs
(0.0996 dispatch/0.0383 combine) and 0.2628 ms on three
(0.1716/0.0885). The final matched pool point is 29.1% faster on two GPUs and
56.6% faster on three. The genuine NVIDIA NCCL-EP LL same-node direct path
measures 0.125424 ms on two GPUs (0.0780 dispatch/0.048512 combine), so the
pool is 21.5% faster.
Provenance is nccl4py 0.3.1, NCCL-EP 0.1.0, NCCL 2.30.7, expert-major layout,
QP8, and library-auto SM/channel selection. NCCL-EP LL does not support three
ranks; its accepted world sizes are 2, 4, or multiples of 8. Four-GPU results
remain pending because GPU 0 was busy/unavailable under an unrelated root-owned
workflow. Do not substitute the older dense NCCL ring surrogate for that
missing NCCL-EP point.

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
  the SM100a kernel uses 233 registers and has no local-memory spills;
- the profiled policy selects 37 PoolInst CTAs for two GPUs and 36 for three
  at 128 tokens, with two streaming data groups.

Direct source-to-expert scatter remains available through
`DAE_POOL_LOCAL_DIRECT_SCATTER=1`, but is off by default. It removes the local
delivery copy while expanding NVLink writes by top-k; at the matched top-k=8
shape it measured about 0.130 ms and lost to unique-token staging plus
overlapped Copy workers.
