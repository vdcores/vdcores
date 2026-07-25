# VDCores Memory-Pool Communication

These applications replace collective-style data exchange with dependency-aware
HBM mailbox requests executed by a VDCores pool core.

Build the optional runtime first:

```bash
export NVSHMEM_HOME="$CONDA_PREFIX/lib/python3.13/site-packages/nvidia/nvshmem"
make nvshmem-pyext
python -m pytest -q tests/test_memory_pool.py tests/test_nvshmem.py
DAE_RUN_NVSHMEM_GPU_TESTS=1 python -m pytest -q tests/test_memory_pool_gpu.py
```

The dependent read/write test sends float32 partials to the pool, accumulates
them, and holds every read until all write tickets arrive:

```bash
# One PE, many SMs: local protocol smoke test
python app/python/memory_pool/dependent_rw.py --writes-per-pe 16

# Multi-node Vista allocation
ibrun python app/python/memory_pool/dependent_rw.py --writes-per-pe 8
```

The top-1 EP test performs dispatch scatter, a normal VDCores
load/`Copy`/store handoff plus expert-specific math, return scatter, and gather
back to original token order:

```bash
ibrun python app/python/memory_pool/ep_top1.py
```

Use one MPI rank/NVSHMEM PE per GPU and the environment from
`agents/workflows/nvshmem-tacc.md`. The pool PE defaults to PE 0. All PEs must
use identical allocation order and protocol dimensions.

When a producer and submit share one persistent launch, pass its VDCores
barrier id through `make_phase_schedule(..., producer_barriers={mailbox: bar})`.
The generated memory stream executes `IssueBarrier` immediately before the
submit. Host-prepared buffers in these two applications are already complete
before launch and do not need that additional barrier.
