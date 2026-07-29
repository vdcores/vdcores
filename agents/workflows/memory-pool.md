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

4. Compare 2, 4, or 8 PEs with the external NCCL ring reference:

```bash
NVSHMEM_DISABLE_NCCL=1 NVSHMEM_IBGDA_NUM_RC_PER_PE=8 \
NVSHMEM_IBGDA_RC_MAP_BY=cta ibrun -n 8 \
  python benchmarks/pool_slice_nccl_compare.py \
  --mode both --tokens-per-pe 128 --hidden-size 4096 \
  --experts-per-pe 1 --warmup 10 --iterations 50
```

For streaming dispatch, `--data-groups 0` selects a producer group ceiling from
the available PoolInst CTAs and remote targets. Actual groups are derived from
router output, targeting about 512 KiB and at most 32 activation rows each.
The destination consumes exactly two compiled ordered queue heads per source
and never reconstructs this count. Override the group ceiling only for an
explicit sweep.

Treat the RC count as a transport shape parameter, not protocol state. On the
current two-PE GH200 path, use 4 through 128 tokens and 8 at 256. More QPs did
not improve the return tail and made metadata completion less stable.

At four PEs, start with CTA-mapped RC32 for 32 tokens, RC16 for 128, and RC8
for 256. At eight PEs, start with RC16 for 32 and RC24 for 128/256. Always
inspect `metadata_closed`, `payload_done`, and
`first_return_put`, because a process/QP mapping can move all-path rank-max
latency by hundreds of microseconds. For host/device comparisons use
`benchmarks/pool_slice_host_e2e.py --paired-device`; it alternates which path
runs first each iteration and is more diagnostic than two sequential sweeps.

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
generation. Only PoolInst rank zero publishes these packets; duplicate
publication by payload CTAs is a performance bug. Skipped sequence IDs remain
valid without clearing transport generations. Round both the fixed peer stride
and every live metadata transfer length to 16 bytes: the local `uint4` copy
path has no scalar tail, and irregular per-target route counts are the required
correctness test for this invariant.

For host/device comparisons, sweep `--registration dmabuf` and
`--registration legacy` only within one allocation and retain the paired
device control. Registration/QP state can move NVSHMEM metadata closure by
more than the host data-path delta; `auto` remains the portability default.
