# External NCCL EP, UCCL, and Triton EP Baselines

The external adapters are `benchmarks/nccl_ep_reference.py`,
`benchmarks/uccl_ep_reference.py`, and
`benchmarks/triton_distributed_ep_reference.py`. They do not link code into or
write events in the VDCores runtime. Shared deterministic routes and JSON
reporting live in `benchmarks/ep_baseline_common.py`. The older dense NCCL ring
control is a two-all-reduce surrogate, not NCCL EP, and is intentionally absent
from the production-library table below.

## Comparison Contract

- Current NVL72 shape: hidden 7168, 128 tokens/PE, 8 experts/PE, top-k 8,
  deterministic random global routing with seed `20260802`.
- Timing: rank-maximum median, 10 warmup and 30 measured iterations.
- Routing tensors are deterministic and fixed across iterations. Router
  generation and NCCL EP handle creation/update are outside the steady-state
  communication interval, just as PoolInst route construction is outside its
  repeated execution interval.
- Dispatch and weighted combine are real library operations. Identity expert
  work is a no-op or lies outside the CUDA-event intervals and is not timed.
- NVIDIA NCCL EP and UCCL BF16 are byte-matched to PoolInst and DeepEP V1.
  UCCL FP8 and Triton FP8 are not byte-matched to the BF16 paths.
- Any clustered-routing matrix below is a historical platform record, not a
  current production comparison. In particular, clustered peer-direct return
  is invalid for global random top-8 and must not be quoted as VDCores.

## Current NVL72 random-global-top-8 scan (2026-08-02)

All values are communication totals in milliseconds. VDCores uses valid
source-gather return; direct scatter is dispatch-only. Triton-distributed is
online FP8 and the dense NCCL control is top-k 1, so neither is byte-matched
to the BF16 sparse-EP rows.

| Implementation | 2 GPUs | 4 GPUs | 8 GPUs | 12 GPUs | 16 GPUs |
|---|---:|---:|---:|---:|---:|
| VDCores BF16 source-gather | 0.080416 | 0.087744 | 0.097952 | 0.097952 | 0.105072 |
| NCCL-EP BF16 | 0.124432 | 0.118224 | 0.161296 | unsupported | 0.165728 |
| DeepEP V1 BF16 | 0.1716 | 0.1419 | unsupported | unsupported | unsupported |
| UCCL BF16 | 0.236656 | 0.216624 | unsupported | unsupported | unsupported |
| Triton-distributed online FP8 | 0.325264 | 0.350208 | 0.333216 | 0.348208 | 0.344288 |
| Dense NCCL ring, BF16 top-k 1 | 0.304 | 0.353 | 0.549 | 1.009 | 1.643 |

DeepEP V1's cross-host low-latency kernel requires IBGDA RC QPs, UCCL's
cross-host branch launches RDMA proxies, and DeepEP V2 requires NCCL GIN.
Those paths are reported unsupported rather than enabled inside the pure
NVL72 all-NVLink scan. The table is the independent rerun at commit
`809b2351581595222fe35565b145e81f4e14c5ab`; full phase breakdowns,
provenance, route digests, and transport checks are in
`.agentlog/2026-08-02-full-baseline-rerun.md`. The preceding accepted scan is
retained in `.agentlog/2026-08-02-random-global-top8.md`.

## Historical cross-library Vista GH200 matrix

All values are communication total in milliseconds.

| PEs | tokens/PE | PoolInst BF16 | DeepEP V1 BF16 | NCCL EP LL BF16 | UCCL BF16 | UCCL FP8 | Triton FP8 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 32 | 0.126 | 0.156 | 0.132 | 0.225 | 0.266 | 0.430 |
| 2 | 128 | 0.220 | 0.407 | 0.308 | 0.592 | 0.513 | 0.975 |
| 2 | 256 | 0.367 | 0.742 | 0.544 | 1.082 | 0.922 | 1.530 |
| 4 | 32 | 0.150 | 0.192 | 0.148 | 0.261 | 0.290 | 0.442 |
| 4 | 128 | 0.347 | 0.551 | 0.356 | 0.690 | 0.573 | 1.080 |
| 4 | 256 | 0.490 | 1.025 | 0.633 | 1.259 | 1.023 | 1.597 |
| 8 | 32 | 0.195 | 0.272 | 0.168 | 0.289 | 0.273 | 0.541 |
| 8 | 128 | 0.344 | 1.040 | 0.401 | 0.798 | 0.684 | 1.069 |
| 8 | 256 | 0.664 | 2.041 | 0.709 | 1.550 | 1.277 | 1.622 |

The 128-token dispatch/combine breakdown is:

| PEs | NCCL EP LL BF16 | UCCL BF16 | UCCL FP8 | Triton FP8 |
|---:|---:|---:|---:|---:|
| 2 | 0.110 / 0.201 | 0.291 / 0.305 | 0.220 / 0.297 | 0.635 / 0.345 |
| 4 | 0.081 / 0.276 | 0.349 / 0.351 | 0.238 / 0.341 | 0.722 / 0.368 |
| 8 | 0.094 / 0.311 | 0.383 / 0.434 | 0.253 / 0.437 | 0.631 / 0.442 |

In this historical checkpoint, NCCL EP LL is the strongest external baseline
at every tested point. It is
nominally 1% lower latency than the recorded PoolInst control at 4 PEs/32
tokens and 14% lower at 8 PEs/32 tokens. PoolInst remains faster at the other
seven points, with its clearest
advantage at larger token counts: at 256 tokens its latency is 32%, 23%, and 6%
lower at 2, 4, and 8 PEs, respectively. NCCL EP's dispatch is already
destination-deduplicated; its
expert-major combine still returns one weighted row per remote expert route,
which is the dominant phase as token count grows. UCCL FP8 saves dispatch time
at medium/large shapes, but that payload reduction is not a BF16 protocol
comparison. Triton FP8 has substantially higher dispatch cost at these decode
sizes.

## Current matched PoolInst/NCCL-EP control (2026-07-30)

Jobs 876732/876733 reran the retained strict raw-RC PoolInst assembly and the
same pinned NCCL-EP LL build in one allocation at each PE count. These values
supersede only the PoolInst/NCCL-EP columns above; DeepEP, UCCL, and Triton were
not rerun in this final kernel loop. PoolInst uses 64 CTAs and leaves 68 of 132
GH200 SMs available.

| PEs | tokens/PE | PoolInst BF16 | NCCL-EP LL BF16 | PoolInst advantage |
|---:|---:|---:|---:|---:|
| 2 | 32 | 0.102 ms | 0.127 ms | 20% |
| 2 | 128 | 0.146 ms | 0.302 ms | 52% |
| 2 | 256 | 0.258 ms | 0.555 ms | 54% |
| 4 | 32 | 0.106 ms | 0.146 ms | 27% |
| 4 | 128 | 0.154 ms | 0.352 ms | 56% |
| 4 | 256 | 0.274 ms | 0.629 ms | 56% |
| 8 | 32 | 0.113 ms | 0.161 ms | 30% |
| 8 | 128 | 0.163 ms | 0.396 ms | 59% |
| 8 | 256 | 0.265 ms | 0.704 ms | 62% |

The 256-token PoolInst progression is 0.258/0.274/0.265 ms at 2/4/8 PEs,
so it no longer has the former four-to-eight-PE regression. See
`pool-slice-dynamic-read.md` for the retained protocol, resource accounting,
and rejected A/Bs.

## Pins and Runtime Findings

- NVIDIA NCCL EP source commit:
  `5067397c2676d5aed50042fc39e5c8ee96eb0027`; `nccl4py` 0.3.1,
  `libnccl_ep` 0.1.0, and NCCL 2.30.7. The measured path is BF16 LL
  expert-major with 8 QPs/rank and library-auto SM/channel selection.
- NCCL EP dispatch sends one activation per token/destination-rank pair with a
  compact top-k header and performs local expert fanout. Expert-major combine
  sends one result per expert route and applies routing weights in the kernel.
  The alternative rank-major layout moves local expert scatter and weighted
  pre-reduction into the caller, so communication-only rank-major numbers are
  not interchangeable with this expert-ready boundary.
- NCCL EP clustered performance passed at 2/4/8 PEs; spread-routing correctness
  also passed at all three scales.
- UCCL commit: `f071f2e31239cd7d673bf2c9369b5cebe1b98457`.
- Triton-distributed commit: `1512a81189315eca7ba38f884aaf239469a88ba3`;
  Triton submodule: `f53694a72a1e4f464fa245df2c7305ccda7cb2a9`.
- UCCL needs `benchmarks/patches/uccl-gh200-host-atomic-buffer.patch` on
  Vista. It changes only registration of the small atomic buffer; payload HBM
  remains DMA-BUF registered.
- The pinned Triton fork needs
  `benchmarks/patches/triton-distributed-cuda13-ptx.patch` to map CUDA 13.0 to
  PTX 9.0. It does not change generated kernels.
- Triton used NVSHMEM 3.6.5/NVSHMEM4py 0.3.0 with UID bootstrap and IBGDA.
  `NVSHMEM_IBGDA_NIC_HANDLER=gpu` worked at two/four PEs but stalled the first
  dispatch at eight PEs. The default `auto` handler passed and produced the
  recorded eight-PE data.
- Clustered performance and source-local, remote-clustered, and spread
  correctness all passed at eight PEs for the UCCL and Triton adapters.

Reproduction details are in
`agents/workflows/external-ep-baselines.md`; task logs and raw JSON records are
under `.agentlog/`.
