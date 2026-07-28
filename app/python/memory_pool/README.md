# VDCores Memory-Pool Communication

Pool communication is expressed entirely as VDCores instructions in one
`dae2` launch. There are no helper kernels, side streams, or host-driven
communication phases.

Build and run the focused suite on a compute node:

```bash
export NVSHMEM_HOME="$CONDA_PREFIX/lib/python3.13/site-packages/nvidia/nvshmem"
make nvshmem-pyext
python -m pytest -q tests/test_memory_pool.py tests/test_pool_slice.py \
  tests/test_nvshmem.py
DAE_RUN_NVSHMEM_GPU_TESTS=1 python -m pytest -q tests/test_memory_pool_gpu.py
```

## Generic Dependent Read/Write

`dependent_rw.py` is the small mailbox reference. Producers submit ordinary
write requests that increment a dependency slot. Reads remain pending until
the pool core observes the requested slot value. The pool scans mailboxes
warp-parallel and executes the selected request cooperatively.

```bash
python app/python/memory_pool/dependent_rw.py --writes-per-pe 16
ibrun python app/python/memory_pool/dependent_rw.py --writes-per-pe 8
```

Its program is a sequence of `MemoryPoolSubmit`, `MemoryPoolWait`, and
`MemoryPoolRun` communication instructions plus ordinary terminating streams.

## Batched Dynamic Read

`pool_slice_dynamic_read.py` is the optimized unified protocol. Each PE owns a
logical pool slice. An ordinary VDCores writer stores every source activation
once. Statically assembled PoolInst CTAs execute one macro
`PoolSliceExchange`: they publish route metadata, PUT runtime-sized unique-row
groups directly from source slots, dynamically gather ready queue heads into
contiguous reader storage, release reader blocks, return their output, and
source-scatter it.

```bash
ibrun -n 2 python app/python/memory_pool/pool_slice_dynamic_read.py \
  --tokens-per-pe 32 --hidden-size 4096 --readers-per-pe 1 --top-k 2

ibrun -n 4 python app/python/memory_pool/pool_slice_dynamic_read.py \
  --tokens-per-pe 32 --hidden-size 4096 --readers-per-pe 1 --top-k 2

NVSHMEM_IBGDA_NUM_RC_PER_PE=8 NVSHMEM_IBGDA_RC_MAP_BY=cta \
ibrun -n 8 python app/python/memory_pool/pool_slice_dynamic_read.py \
  --tokens-per-pe 128 --hidden-size 4096 --readers-per-pe 1 --top-k 1
```

Each source publishes one contiguous descriptor-plus-two-queues envelope to
every target, including zero-row targets, plus its compact route map. Those two
metadata ranges advance one merged per-source counter; payload
groups have separate readiness slots and return visibility uses source-indexed
NVSHMEM signals. Ordered `END` instructions retire the shared sender set
without a special epoch-end message. Rows are fixed-width, 16-byte-aligned, at
least 1 KiB, and use fixed-capacity symmetric buffers. The default
`--data-groups 0` derives a group ceiling from PE/PoolInst-CTA placement;
explicit values are intended for tuning sweeps.

## External NCCL Ring Comparison

NCCL code lives only under `benchmarks/`. `pool_slice_nccl_compare.py` times
VDCores from internal `g_events`; CUDA events time only the external dense
two-ring reference.

```bash
NVSHMEM_DISABLE_NCCL=1 NVSHMEM_IBGDA_NUM_RC_PER_PE=8 \
NVSHMEM_IBGDA_RC_MAP_BY=cta ibrun python \
  benchmarks/pool_slice_nccl_compare.py \
  --mode both --tokens-per-pe 32 --hidden-size 4096 \
  --experts-per-pe 1 --warmup 5 --iterations 20
```

Use the Vista/NVSHMEM workflow under `agents/workflows/`. Every rank must use
the same symmetric allocation order and protocol dimensions.
