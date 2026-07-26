# VDCores Memory-Pool Communication

These applications replace collective exchange with first-class VDCores
communication instructions. The optional runtime adds one communication warp
to each persistent `dae2` block; there are no auxiliary pool kernels or CUDA
streams.

Build and run the focused suite on a compute node:

```bash
export NVSHMEM_HOME="$CONDA_PREFIX/lib/python3.13/site-packages/nvidia/nvshmem"
make nvshmem-pyext
python -m pytest -q tests/test_memory_pool.py tests/test_ep_pool.py \
  tests/test_pool_slice.py tests/test_nvshmem.py
DAE_RUN_NVSHMEM_GPU_TESTS=1 python -m pytest -q tests/test_memory_pool_gpu.py
```

## Dependent Read/Write

`dependent_rw.py` is the generic slot-dependency test. Every producer submits
a write that adds one ticket; reads remain in the HBM mailbox until the pool
core observes the full fan-in count.

```bash
python app/python/memory_pool/dependent_rw.py --writes-per-pe 16
ibrun python app/python/memory_pool/dependent_rw.py --writes-per-pe 8
```

The launcher contains ordinary memory/compute termination instructions plus
`MemoryPoolSubmit`, `MemoryPoolWait`, and `MemoryPoolRun` communication
instructions. It is one `dae2` launch.

## Expert Pool

`ep_pool_top1.py` builds one block per global expert. Stable grouped route rows
are metadata; activation rows are contiguous messages. The owner gathers rows
into its receiver-owned pool, existing TMA/`Copy`/store instructions model the
expert, and the communication warp returns and scatters rows to their origin.

```bash
ibrun python app/python/memory_pool/ep_pool_top1.py \
  --tokens-per-pe 32 --hidden-size 4096 --experts-per-pe 1

# Repeated top-k source rows, with a distinct output row per route.
ibrun python app/python/memory_pool/ep_pool_top1.py \
  --tokens-per-pe 16 --hidden-size 512 --experts-per-pe 2 --top-k 2
```

The minimal fast path requires contiguous rows of at least 1 KiB whose byte
width is a multiple of 16. It supports at most 32 PEs and at most one resident
block per global expert (`num_experts <= GPU SM count`, 132 on GH200).

`ep_top1.py` is a compatibility entry point for this same one-launch program.

## Pool-Slice Dynamic Read

`pool_slice_dynamic_read.py` is the pool-owned protocol. There is one logical
slice and one pool block per PE. The source application writes each activation
once plus grouped route metadata. Block 0 publishes one descriptor to each
slice, scans all source queues warp-parallel, resolves the shared sender-set
ticket, pulls named source slots into contiguous reader storage, and owns the
return/scatter. Reader blocks contain only ordinary VDCores memory/compute
instructions behind pool-released barriers.

```bash
# Two PEs, top-k fan-out from one source activation copy.
ibrun -n 2 python app/python/memory_pool/pool_slice_dynamic_read.py \
  --tokens-per-pe 32 --hidden-size 4096 --readers-per-pe 1 --top-k 2

# Four PEs.
ibrun -n 4 python app/python/memory_pool/pool_slice_dynamic_read.py \
  --tokens-per-pe 32 --hidden-size 4096 --readers-per-pe 1 --top-k 2
```

The first buffer set permits one outstanding sequence. Each source posts a
zero-route descriptor as needed, so all dynamic reads retire from the same
monotonic group sequence without a special epoch-end message. Top-k outputs in
this focused harness are route-major; weighted combine is a subsequent pool
reduction.

## NCCL Ring Comparison

The NCCL comparison is deliberately outside the VDCores source/application
tree in `benchmarks/ep_pool_nccl_compare.py`. It forces `NCCL_ALGO=Ring`; its
dense reference performs an expert-major all-reduce for dispatch and a
token-major all-reduce for return, while the pool transfers only routed rows.

VDCores time comes exclusively from `CommRecordEvent` timestamps in the
kernel's internal `g_events` profile space. CUDA events time only the external
NCCL reference. Rank maxima form the distributed critical-path samples.

```bash
NVSHMEM_DISABLE_NCCL=1 ibrun python \
  benchmarks/pool_slice_nccl_compare.py \
  --mode both --tokens-per-pe 32 --hidden-size 4096 \
  --experts-per-pe 1 --warmup 5 --iterations 20
```

Use the environment in `agents/workflows/nvshmem-tacc.md`. All ranks must use
the same symmetric allocation order and protocol dimensions. Route generation
inside a larger VDCores program can release an ordinary barrier and place
`CommWaitBarrier` before `ExpertPoolDispatch`; the checked-in correctness app
prepares route metadata before launch.
