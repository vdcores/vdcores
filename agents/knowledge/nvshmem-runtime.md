# NVSHMEM Runtime

DAE's NVSHMEM support is an optional control runtime, isolated from the normal
`dae.runtime` extension.

## Entry Points

- `src/torch_nvshmem_runtime.cu`: MPI/NVSHMEM ownership, local-rank GPU mapping, symmetric allocation, signals, barriers, and finalization.
- `python/dae/nvshmem.py`: lazy module API and `NVSHMEMLauncher` subclass.
- `python/dae/nvshmem_launcher.py`: compatibility import exposing the alternative `Launcher`.
- `setup_nvshmem.py` and `make nvshmem-pyext`: optional build using `nvcc` for compilation and TACC `mpicxx` for final linking.
- `experimental/nvshmem/python_binding.py`: `ibrun` smoke test.

## Ownership Model

- `init()` calls `MPI_Init_thread` only when MPI is not already initialized and initializes NVSHMEM with `NVSHMEMX_INIT_WITH_MPI_COMM` only when needed.
- Default GPU selection is node-local MPI rank. Vista's supported launch shape is one MPI rank per GH node/GPU.
- Initialization is transactional: failures clean up communicators and any MPI/NVSHMEM state started by the extension.
- Symmetric tensors are non-owning PyTorch CUDA views over `nvshmem_malloc` or `nvshmem_calloc` memory.
- Tensor destruction never calls `nvshmem_free`, because rank-local Python destruction is not a valid collective free protocol.
- The runtime records allocations and frees them in reverse collective order during explicit `finalize()`.
- All PEs must allocate with identical shape, dtype, and call order. Tensors become invalid immediately after `finalize()`.

## Signal Space

`init_signal_space(count)` allocates one zeroed symmetric `torch.uint64` array per
PE and retains it as process-global runtime state. `signal()` and
`wait_signal()` enqueue NVSHMEM operations on a Torch CUDA stream. The local
tensor can also be passed to existing DAE instructions as an ordinary CUDA
pointer.

The optional extension uses host-side NVSHMEM APIs. Existing DAE kernels can
load/store their local symmetric allocation normally; a future kernel that
directly invokes NVSHMEM device APIs will additionally need NVSHMEM device
relocatable-code linking.

## Verified Integration

On a two-node Vista GH allocation, the Python smoke test completed with one MPI
rank/PE per GH200. NVSHMEM 3.4.5 initialized `ibrc` and IBGDA with GPU NIC
handling on both nodes. Cross-node signals returned the expected values, and a
DAE `TmaLoad1D -> OP_COPY -> TmaStore1D` launch copied data exactly between two
NVSHMEM-backed Torch tensors on both PEs.
