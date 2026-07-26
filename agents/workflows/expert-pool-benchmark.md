# Expert-Pool Build And Benchmark

Run GPU steps only inside an allocation with one MPI rank/GPU per node. Use
the NVSHMEM environment from `agents/workflows/nvshmem-tacc.md`.

## Build And Focused Checks

```bash
make nvshmem-pyext
python -m pytest -q \
  tests/test_ep_pool.py tests/test_pool_slice.py tests/test_memory_pool.py \
  tests/test_nvshmem.py
DAE_RUN_NVSHMEM_GPU_TESTS=1 NVSHMEM_DISABLE_NCCL=1 \
  python -m pytest -q tests/test_memory_pool_gpu.py
```

Also compile `make runtime.o`. The ordinary build must remain eight warps with
zero spills and no WGMMA diagnostics. In the optional build, inspect the final
device-linked image—not just pre-link ptxas output—because NVSHMEM device
linking changes register allocation:

```bash
cuobjdump --dump-resource-usage \
  "$(python -c 'import dae.runtime; print(dae.runtime.__file__)')"
```

The nine-warp kernel must be at or below 168 registers/thread on Hopper. The EP
WGMMA GEMV should report zero spills and the build should have no WGMMA
serialization warning.

## Correctness Boundary

Use aligned rows covered by the minimal protocol; there is intentionally no
small-row or byte-tail fallback.

```bash
NVSHMEM_DISABLE_NCCL=1 ibrun python \
  app/python/memory_pool/dependent_rw.py --writes-per-pe 8 --elements 256

NVSHMEM_DISABLE_NCCL=1 ibrun python \
  app/python/memory_pool/ep_pool_top1.py \
  --tokens-per-pe 16 --hidden-size 512 --experts-per-pe 1 --top-k 2

NVSHMEM_DISABLE_NCCL=1 ibrun python \
  app/python/memory_pool/pool_slice_dynamic_read.py \
  --tokens-per-pe 16 --hidden-size 512 --readers-per-pe 1 --top-k 2
```

Both output lines must report every PE and `launches=1` for EP.

## External System-Level Performance Sweep

Start with the protocol cost model, then use profiler traces only to validate a
specific hypothesis. Keep backend comparisons under `benchmarks/`; do not add
NCCL timing or dependencies to the VDCores runtime or application sources.
The pool samples must come from `dae2`'s internal `g_events` timestamps. CUDA
events are permitted only for the external NCCL reference.

```bash
for token_count in 8 32 128; do
  NVSHMEM_DISABLE_NCCL=1 ibrun python \
    benchmarks/ep_pool_nccl_compare.py \
    --mode both --tokens-per-pe "$token_count" --hidden-size 4096 \
    --experts-per-pe 1 --warmup 5 --iterations 20
done
```

Use `benchmarks/pool_slice_nccl_compare.py` for the pool-owned dynamic-read
variant. Run the same sweep at both two and four PEs before changing the
protocol. The first refinement target is the reported one-RMA-per-remote-route
gather; coalesce adjacent source slots before adding another pool warp.

Use `--gather-mode phased` as the serialized A/B control and
`--gather-mode streaming` for the event-driven path. Compare
`--activation-stages 1` with `2` only when the internal
`first_data_published`/`data_published` events show the source write is on the
critical path. Two stages add one doorbell per source/target and are not the
default.

For the common one-reader-per-PE fast path, route boundaries are already in
the 32-byte descriptor, so `offset_metadata_bytes_received_per_pe` should be
zero. A nonzero value is expected when `--experts-per-pe > 1` because internal
reader boundaries still require one offsets RMA per source.

Check these quantities together:

- routed payload bytes per direction;
- nonempty remote batches, RMAs, atomics, and signals;
- local gather/scatter and expert-copy bytes;
- dispatch-ready, overlapped tail, and end-to-end rank-max latency;
- routing balance and expert capacity actually processed.

Sweep `--experts-per-pe` separately. Pool payload stays route-proportional,
while the dense reference's expert-major tensor grows with global experts.

For memory checking, keep the row at least 1 KiB:

```bash
NVSHMEM_DISABLE_NCCL=1 compute-sanitizer --tool memcheck \
  --error-exitcode=99 python app/python/memory_pool/ep_pool_top1.py \
  --tokens-per-pe 2 --hidden-size 512 --experts-per-pe 2 --top-k 2
```

If a trace is needed, capture only a few iterations. Expect one persistent
`dae2` kernel per iteration; communication, memory, and compute are warps in
that kernel, not auxiliary pack/dispatch/scatter kernels or streams.
