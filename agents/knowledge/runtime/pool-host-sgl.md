# Grace Host-Verbs Pool Transport

This is an experimental data-plane backend for the existing PoolInst queue
protocol. It lives under `benchmarks/` and does not change the production
VDCores runtime. Metadata publication, ordered receiver queue heads, dynamic
read retirement, and expert ownership remain PoolInst work.

## Plane Boundary

The source activation stays in its original GPU HBM row and is written once.
PoolInst still resolves routing and publishes receiver metadata. In the host
variant it also makes a compact data request visible to a Grace worker; the
request names source row indices, one contiguous destination interval, and
the destination's normal data-ready word. No activation bytes enter host
memory and no source staging buffer is required.

The worker translates one data request into verbs work:

1. Each `ibv_sge` names one non-contiguous source HBM row.
2. One RC RDMA WRITE consumes several SGEs. Verbs concatenates their bytes, in
   SGE order, into the contiguous remote delivery interval.
3. More than the queried `max_sge` rows become a short chain of multi-SGE data
   WRs, not one WR per row.
4. An inline 8-byte RDMA WRITE publishes that message's data-ready sequence
   after its data WRs on the same RC QP.

Several ready data requests can share one `ibv_post_send`. Each keeps its own
ordered remote readiness write, while only the last readiness WR is locally
signaled. Its CQ completion reclaims the whole batch. This merges host/NIC
doorbells and local completions without merging PoolInst dependencies.

The destination PoolInst is unchanged: it executes only the ordered metadata
queue head, starts copying as soon as that head's data-ready sequence arrives,
and uses the existing sender-set dependency only to retire the dynamic read.

## Placement And Ordering

- The GPU/Grace request ring uses ordinary aligned `malloc`;
  `numa_alloc_onnode` remains an optional placement refinement. Only compact
  descriptors and row indices cross the coherent CPU/GPU control boundary.
- Payload stays in registered GPU HBM. Vista passed CUDA DMA-BUF export plus
  `ibv_reg_dmabuf_mr` on `mlx5_0`; legacy peer-memory registration remains a
  probe fallback.
- The benchmark GPU producer publishes a task-owned request slot with one
  semantic system-release ready word, and the CPU acquires that word. It does
  not need a general system fence or a shared producer atomic.
- Data and readiness use the same RC QP, so the readiness write cannot pass
  that message's data writes.
- GH200 reports CUDA GPUDirect RDMA write ordering `200` (`ALL_DEVICES`). A GPU
  can therefore consume the remotely written data after observing readiness;
  no receiver host flush or system fence is needed. The benchmark calls
  `cuFlushGPUDirectRDMAWrites` only as a portability fallback when the queried
  native ordering is below `OWNER`.

Vista ConnectX exposes 30 send SGEs. One RC QP per peer already reached about
43 GiB/s per direction in the two-node experiment, so multiple QPs are not
justified yet. Add them only if concurrent metadata/data traffic shows a
single-QP issue bottleneck.

## Two-PE Results

`benchmarks/host_sgl_benchmark.py` sends bidirectionally and reports the
maximum host post-to-CQ time across ranks. Reset, barriers, visibility fallback,
and byte/readiness verification are outside the timed interval. These are
transport microbenchmark numbers, not VDCores internal timings.

For 128 BF16-7168 rows per message (1,835,008 bytes), SGL depth scaling was:

| batch depth | batch p50 (us) | amortized/message (us) | GiB/s/direction |
| ---: | ---: | ---: | ---: |
| 1 | 58.063 | 58.063 | 29.433 |
| 2 | 93.328 | 46.664 | 36.623 |
| 4 | 172.719 | 43.180 | 39.578 |
| 8 | 329.437 | 41.180 | 41.501 |
| 16 | 640.234 | 40.015 | 42.709 |
| 32 | 1273.074 | 39.784 | 42.957 |

Depth 16 is the useful default: depth 32 adds little throughput while doubling
buffer and descriptor residency. At depth 8, SGL used 40 data WRs versus 1,024
one-row WRs and measured 329.4 versus 339.0 us. An SGE-width sweep measured
40.37, 41.34, and 41.50 GiB/s for 1, 4, and 30 SGEs respectively. Batching and
one local completion provide most of the throughput gain; maximum-width SGL
provides the smallest QP footprint.

Smaller messages benefit similarly. Eight-row messages at depth 16 measured
53.9 us per batch (3.37 us amortized) with SGL versus 56.2 us with one-row WRs.
Thirty-two-row messages at depth 8 measured 93.1 versus 96.4 us.

## Experimental Boundary

Build with `make -C benchmarks/host_sgl`. The capability gate is
`benchmarks/host_sgl_probe.py`; the true SGL, batching, ordering, and data
verification harness is `benchmarks/host_sgl_benchmark.py`.

The ABI-v3 benchmark includes the coherent ring, fixed 64-slot capacity,
32-row request bound, 32-message epoch bound, and batches of up to 16 requests.
The GPU and Grace consumer use monotonic generations and in-order queue heads;
there is no deadline or timeout in retirement. In-flight posts own their
descriptor storage through CQ completion.

For one 32-row BF16-7168 message, direct submission measured 26.592 us and the
ring 60.768 us. Fourteen 9-row messages measured 55.536 us direct and 89.344 us
through the best one-CTA-per-message publisher. The standalone coherent
handoff therefore costs about 34 us. Before merging, that fixed cost must be
hidden inside the PoolInst lifetime and compared with GPU IBGDA under identical
routes. Until then, no host-verbs code belongs in `src/`, `include/`, or the
main application.

Two targeted A/Bs show what does not cause that fixed gap. Recycling completed
WR/SGE vectors measured 89.712 us, and publishing four descriptors behind one
system-release head generation measured 89.360 us. Both variants were removed.
The gap is therefore dominated by launching the standalone GPU publisher and
observing its first Grace-coherent store, not by descriptor allocation or the
number of release stores. The next valid host experiment must publish the same
ABI-v3 slots from an already-resident PoolInst execute warp; do not complicate
the ring with merged generations first.

The isolated harness now includes that resident-producer experiment. On
`c642-012`/`c642-031`, one resident eight-warp CTA plus the in-order Grace
consumer measured 18.544 us for one 32-row BF16-7168 message (458,752 bytes).
Four messages behind one epoch measured 49.504 us, or 12.376 us/message and
34.52 GiB/s per direction. Both runs used DMA-BUF registration and native
GPUDirect ordering 200. This removes the standalone launch/handoff penalty,
but remains a benchmark-only data-plane option until a PoolInst instruction
publishes the same ring ABI and is compared end-to-end with GPU IBGDA.
