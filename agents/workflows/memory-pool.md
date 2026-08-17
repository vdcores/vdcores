# Memory-Pool Workflow

Use this for generic dependent read/write or the batched gathered-read protocol
over the optional NVSHMEM VDCores runtime.

1. Read the contracts:

- `agents/knowledge/runtime/memory-pool-protocol.md`
- `agents/knowledge/runtime/pool-slice-dynamic-read.md`
- `agents/knowledge/runtime/vdcores-communication-core.md`
- `agents/knowledge/nvshmem-runtime.md`

2. On a Vista compute node, build and run focused coverage:

```bash
export NVSHMEM_HOME="$CONDA_PREFIX/lib/python3.13/site-packages/nvidia/nvshmem"
make nvshmem-pyext
python -m pytest -q tests/test_memory_pool.py tests/test_pool_slice.py \
  tests/test_nvshmem.py
```

3. Run generic dependency and batched dynamic-read correctness:

```bash
NVSHMEM_DISABLE_NCCL=1 ibrun -n 2 \
  python app/python/memory_pool/dependent_rw.py \
  --writes-per-pe 8 --elements 256

NVSHMEM_DISABLE_NCCL=1 NVSHMEM_IBGDA_NUM_RC_PER_PE=1 \
NVSHMEM_IBGDA_RC_MAP_BY=cta ibrun -n 2 \
  python app/python/memory_pool/pool_slice_dynamic_read.py \
  --tokens-per-pe 13 --hidden-size 512 --readers-per-pe 2 --top-k 2
```

4. Compare 2, 4, or 8 PEs with the dense two-all-reduce NCCL ring surrogate:

```bash
NVSHMEM_DISABLE_NCCL=1 NVSHMEM_IBGDA_NUM_RC_PER_PE=8 \
NVSHMEM_IBGDA_RC_MAP_BY=cta ibrun -n 8 \
  python benchmarks/pool_slice_nccl_compare.py \
  --mode both --tokens-per-pe 128 --hidden-size 4096 \
  --experts-per-pe 1 --warmup 10 --iterations 50
```

For streaming dispatch, `--data-groups 0` selects a producer group ceiling from
the available PoolInst CTAs and remote targets. Actual groups are derived from
router output, targeting about 256 KiB through four PEs and 512 KiB at eight
or more PEs, with at most 32 activation rows each.
The destination consumes exactly two compiled ordered queue heads per source
and never reconstructs this count. Override the group ceiling only for an
explicit sweep.

For the device data plane, compile the completion scope to match the QP map.
The default CTA-mapped policy posts one generation after every group warp has
posted to that CTA's ordered QP. Set
`DAE_POOL_SLICE_WARP_QP_COMPLETION=1` only with warp-mapped QPs; then each
nonempty payload warp makes its last row run a put-with-signal and the
destination joins that exact set. Do not put a cooperative `quiet` back in
weighted dispatch: in NVSHMEM 3.4 it scans all RC QPs for all peers and turns
each group completion into PE/QP-scaled control work. The fixed PoolInst
assembly may use up to 132 CTAs; the dynamic group ceiling remains 32 and is a
separate protocol bound.

Treat the RC count as a transport shape parameter, not protocol state. The
current matched 2/4/8-PE path starts with CTA-mapped RC8 and request batch 32
at 32--256 tokens. Wider QP counts did not improve the normalized queue or
return intervals after generic launch skew was removed. Always inspect
`metadata_closed`, `payload_done`, and
`first_return_put`, because a process/QP mapping can move all-path rank-max
latency by hundreds of microseconds. For host/device comparisons use
`benchmarks/pool_slice_host_e2e.py --paired-device`; it alternates which path
runs first each iteration and is more diagnostic than two sequential sweeps.

The optimized device-SGL build is an explicit compile-time transport contract:

```bash
DAE_POOL_SLICE_RAW_SGL=1 DAE_POOL_SLICE_RAW_SGL_WIDTH=8 \
DAE_POOL_SLICE_WARP_QP_COMPLETION=0 make nvshmem-pyext
```

Run that binary only with GPU-owned IBGDA, one initialized NIC, CTA-mapped RC
QPs, and symmetric GPU buffers. It deliberately contains no public-NVSHMEM
fallback or device validation matrix. The Python allocator raises the group
ceiling as needed so every noncontiguous raw message has at most 32 rows and
rejects raw capacities above 1024 token slots. The ordinary raw=0 build keeps
the public transport and its generic runtime behavior.

Before enabling the optional Grace host-verbs transport, gate direct HBM
registration on the same allocation:

```bash
NVSHMEM_DISABLE_NCCL=1 ibrun -n 2 \
  python benchmarks/host_sgl_probe.py
```

Then build and run the isolated true-SGL path:

```bash
make -C benchmarks/host_sgl
NVSHMEM_DISABLE_NCCL=1 ibrun -n 2 \
  python benchmarks/host_sgl_benchmark.py \
  --rows 128 --batch-depth 16
```

The host request ring may use ordinary aligned `malloc`; only the CUDA
DMA-BUF-to-verbs registration of symmetric GPU HBM is capability-sensitive.
This experiment may replace data submission only. Metadata, ordered queue
heads, readiness dependencies, and retirement remain PoolInst semantics. Keep
it outside the main source tree until it beats the GPU path with real overlap;
see `agents/knowledge/runtime/pool-host-sgl.md`.

VDCores timings must come from internal `g_events`. NCCL remains under
`benchmarks/` and uses CUDA events only for the external reference. Use
monotonic sequences when reusing signal words and preserve identical symmetric
allocation order on every PE. Streaming metadata is one runtime-sized packet
per source: its 64-byte batch and live ordered-queue prefix are followed by
32-bit `row16/BF16-weight` route words and protected by one payload-coupled
generation. Since every source publishes that packet to every target on every
invocation, advance its transport generation with the monotonic sequence
delta and `NVSHMEM_SIGNAL_ADD`; do not apply that optimization to dynamically
absent data groups without per-slot catch-up state. With at least one PoolInst
CTA per PE, statically assign exactly one target packet to each of the first
`num_pes` CTAs so CTA-mapped QPs issue them in parallel. Smaller generic
assemblies may fall back to rank-zero warps. Multiple publishers for the same
target packet are a correctness and performance bug. Skipped sequence IDs remain
valid without clearing transport generations. Round both the fixed peer stride
and every live metadata transfer length to 16 bytes: the local `uint4` copy
path has no scalar tail, and irregular per-target route counts are the required
correctness test for this invariant.

Keep repeated VDCores launch setup rank-symmetric and O(1). `Launcher` caches
immutable device instruction/TMA/core state and restores its initial counter
image with one same-stream device copy. Do not replace that copy with Python
scalar GPU assignments: after a host/MPI iteration barrier, their rank skew
becomes a max-PE network tail even though local `g_events` exclude host setup.
The PoolInst program also disables the generic persistent-instruction L2
window; normal compute/memory launchers retain it.

For host/device comparisons, sweep `--registration dmabuf` and
`--registration legacy` only within one allocation and retain the paired
device control. Registration/QP state can move NVSHMEM metadata closure by
more than the host data-path delta; `auto` remains the portability default.
