# Memory-Pool EP Workflow

Use this for dependency-aware read/write or expert-parallel communication over
the optional NVSHMEM VDCores runtime.

1. Read the protocol and runtime contracts:

- `agents/knowledge/runtime/memory-pool-ep.md`
- `agents/knowledge/nvshmem-runtime.md`
- `agents/knowledge/runtime/vdcores-operator-semantics.md`

2. Build and run host/API coverage:

```bash
export NVSHMEM_HOME="$CONDA_PREFIX/lib/python3.13/site-packages/nvidia/nvshmem"
make nvshmem-pyext
python -m pytest -q tests/test_memory_pool.py tests/test_nvshmem.py
```

3. Run real singleton GPU kernels:

```bash
DAE_RUN_NVSHMEM_GPU_TESTS=1 \
python -m pytest -q tests/test_memory_pool_gpu.py
```

4. In a two-node Vista allocation with one PE/GPU, run the communication
stages in order:

```bash
ibrun python app/python/memory_pool/dependent_rw.py \
  --writes-per-pe 8 --elements 256
ibrun python app/python/memory_pool/ep_top1.py \
  --tokens-per-pe 8 --hidden-size 32
```

The first command creates 16 total writes on two PEs and checks that every read
waits for ticket 16. The second checks dispatch scatter, a VDCores copy/expert
stage, return scatter, and gather to original token order.

5. For device memcheck, disable NVSHMEM's optional NCCL team backend. The CUDA
13 NCCL cubins in this environment otherwise report `no kernel image` under
instrumentation before the application kernel runs:

```bash
NVSHMEM_DISABLE_NCCL=1 compute-sanitizer --tool memcheck \
  --error-exitcode 99 \
  python app/python/memory_pool/dependent_rw.py \
  --writes-per-pe 4 --elements 32
```

Use monotonic request sequences when reusing mailboxes. If producer data comes
from a VDCores store in the same launch, pass the store's barrier through
`make_phase_schedule(..., producer_barriers={mailbox: bar_id})`; host-prepared
data does not need that extra barrier.
