# External UCCL and Triton EP Baselines

The external adapters are `benchmarks/uccl_ep_reference.py` and
`benchmarks/triton_distributed_ep_reference.py`. They do not link code into or
write events in the VDCores runtime. Shared deterministic routes and JSON
reporting live in `benchmarks/ep_baseline_common.py`.

## Comparison Contract

- Shape: hidden 7168, 8 experts/PE, top-k 8, clustered routes.
- Timing: rank-maximum median, 10 warmup and 30 measured iterations.
- Dispatch and weighted combine are real library operations. Identity expert
  work is between separate CUDA-event intervals and is not timed.
- UCCL BF16 is byte-matched. UCCL FP8 and Triton FP8 are not byte-matched to
  PoolInst or DeepEP V1 BF16.
- PoolInst and DeepEP V1 values below are the previously recorded matched
  controls, not measurements from the UCCL/Triton allocation.

## Vista GH200 Results

All values are communication total in milliseconds.

| PEs | tokens/PE | PoolInst BF16 | DeepEP V1 BF16 | UCCL BF16 | UCCL FP8 | Triton FP8 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 32 | 0.126 | 0.156 | 0.225 | 0.266 | 0.430 |
| 2 | 128 | 0.220 | 0.407 | 0.592 | 0.513 | 0.975 |
| 2 | 256 | 0.367 | 0.742 | 1.082 | 0.922 | 1.530 |
| 4 | 32 | 0.150 | 0.192 | 0.261 | 0.290 | 0.442 |
| 4 | 128 | 0.347 | 0.551 | 0.690 | 0.573 | 1.080 |
| 4 | 256 | 0.490 | 1.025 | 1.259 | 1.023 | 1.597 |
| 8 | 32 | 0.195 | 0.272 | 0.289 | 0.273 | 0.541 |
| 8 | 128 | 0.344 | 1.040 | 0.798 | 0.684 | 1.069 |
| 8 | 256 | 0.664 | 2.041 | 1.550 | 1.277 | 1.622 |

The 128-token dispatch/combine breakdown is:

| PEs | UCCL BF16 | UCCL FP8 | Triton FP8 |
|---:|---:|---:|---:|
| 2 | 0.291 / 0.305 | 0.220 / 0.297 | 0.635 / 0.345 |
| 4 | 0.349 / 0.351 | 0.238 / 0.341 | 0.722 / 0.368 |
| 8 | 0.383 / 0.434 | 0.253 / 0.437 | 0.631 / 0.442 |

PoolInst remains the fastest path at every tested point. UCCL is the stronger
of the two added external baselines. Its BF16 path is slower than DeepEP V1 at
two/four PEs but becomes faster at eight PEs for 128 and 256 tokens. FP8 saves
UCCL dispatch time at medium/large shapes, but that payload reduction is not a
BF16 protocol comparison. Triton FP8 has substantially higher dispatch cost at
these decode sizes.

## Pins and Runtime Findings

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
  correctness all passed at eight PEs for both adapters.

Reproduction details are in
`agents/workflows/external-ep-baselines.md`; task logs and raw JSON records are
under `.agentlog/`.
