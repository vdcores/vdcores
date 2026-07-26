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
NVSHMEM_DISABLE_NCCL=1 NVSHMEM_IBGDA_NUM_RC_PER_PE=1 \
NVSHMEM_IBGDA_RC_MAP_BY=cta ibrun -n 8 \
  python benchmarks/pool_slice_nccl_compare.py \
  --mode both --tokens-per-pe 128 --hidden-size 4096 \
  --experts-per-pe 1 --warmup 10 --iterations 50
```

`--pack-warps 0` is the default measured policy: four pack warps for small
payloads and for four or more PEs, and six pack warps for large two-PE
payloads. Override it only for an explicit sweep.

VDCores timings must come from internal `g_events`. NCCL remains under
`benchmarks/` and uses CUDA events only for the external reference. Use
monotonic sequences when reusing signal words and preserve identical symmetric
allocation order on every PE.
