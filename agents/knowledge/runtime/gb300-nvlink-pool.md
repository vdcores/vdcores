# GB300 NVLink Pool Port

The GB300/GB200 pool-only build is selected with `make gb300-nvlink-pyext`.
It targets `sm_100a`, selects only `OP_TERMINATEC`, and therefore does not
instantiate Hopper compute kernels. `DAE_POOL_DATA_PATH=nvshmem` retains the
original NVSHMEM/IBGDA-capable data path for A/B builds.

`DAE_POOL_DATA_PATH=nvlink` preserves `PoolSliceConfig`, metadata envelopes,
queues, and dependency generations. Payload and metadata bytes use
`nvshmem_ptr` only to resolve the symmetric peer mapping, then issue ordinary
warp-striped peer-global stores. Publication uses system-scope release/acquire
operations. Launch with `transport="nvlink"`; this disables NVSHMEM remote
transports and IBGDA, while NVSHMEM still bootstraps and allocates the
symmetric heap.

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
10 warmups, and 30 samples, the post-rebase multimem pool measures 0.122880 ms
on two GPUs and 0.139792 ms on three. DeepEP V1.2.1 measures 0.1389 ms on two
(0.0996 dispatch/0.0383 combine) and 0.2628 ms on three
(0.1716/0.0885), making the pool 11.5% and 46.8% faster.

The genuine NVIDIA NCCL-EP LL same-node direct path measures 0.125424 ms on
two GPUs (0.0780 dispatch/0.048512 combine), so the pool is 2.0% faster.
Provenance is nccl4py 0.3.1, NCCL-EP 0.1.0, NCCL 2.30.7, expert-major layout,
QP8, and library-auto SM/channel selection. NCCL-EP LL does not support three
ranks; its accepted world sizes are 2, 4, or multiples of 8. Four-GPU results
remain pending because GPU 0 was busy/unavailable under an unrelated root-owned
workflow. Do not substitute the older dense NCCL ring surrogate for that
missing NCCL-EP point.

## Rebased `mempool-ep` checkpoint

The GB300 work is rebased on `origin/mempool-ep` commit `c573d8e`. The remote
fused metadata, distributed queue-head scan, target-major CTA submission, and
dynamic weighted-return grouping remain authoritative. The NVLink specialization
changes only transport primitives: direct peer stores for payload/metadata,
system-scope atomic-add publication for fused metadata deltas, and release-store
generations with acquire loads for data and return completion.

At the matched two-PE 128-token DeepEP shape, the rebased 32-CTA path measures
0.131 ms dispatch, 0.097 ms return, and 0.230 ms total, down 36% from the
pre-rebase 0.358 ms total. DeepEP is 0.172 ms, leaving a 1.34x gap. Metadata
closure is now 0.061 ms; gather completion at 0.119 ms and weighted return are
the next optimization targets. A 16/24/32 CTA sweep retained 32 CTAs.
