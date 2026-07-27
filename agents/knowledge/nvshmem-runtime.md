# NVSHMEM Runtime

DAE's NVSHMEM support has three deliberately separate pieces:

- `python/dae/nvshmem.py` uses NVIDIA's official `nvshmem.core` and
  `nvshmem.bindings` packages for MPI bootstrap, NVSHMEM lifecycle, PE queries,
  barriers, and stream-ordered signal operations.
- The NVSHMEM build of `dae.runtime` links the NVSHMEM device library and
  provides private hooks to register the CUDA module containing `dae2`.
  `dae.nvshmem` calls those hooks as part of its host lifecycle.
- `src/torch_nvshmem_runtime.cu` is a small DAE-specific extension that only
  allocates symmetric Torch tensors and releases tracked allocations
  collectively. It has no signal-specific allocation or address API.

`setup.py` builds the ordinary runtime by default. Setting
`DAE_ENABLE_NVSHMEM=1` also builds `dae._nvshmem_runtime` and defines the same
macro on both extensions. `make nvshmem-pyext` is the normal entry point;
`setup_nvshmem.py` no longer exists.

## Versions And Ownership

- The Vista-tested combination is NVSHMEM 3.4.5 with
  `nvshmem4py-cu13==0.1.3`; the full validated set is pinned in
  `requirements.txt` and mirrored by the `setup.py` `nvshmem` extra.
- PyTorch 2.10.0+cu130 requires `cuda-bindings==13.0.3`; keep
  `cuda-python==13.0.3` pinned when installing NVSHMEM4Py.
- Build `mpi4py` against TACC OpenMPI rather than mixing MPI implementations.
- Importing `dae.nvshmem` is lazy. `init()` imports MPI/NVSHMEM4Py, maps the
  node-local MPI rank to a CUDA device, and initializes NVSHMEM with
  `MPI.COMM_WORLD`.
- MPI is owned by `mpi4py`; DAE finalization releases its symmetric allocations
  and owned NVSHMEM state, but does not finalize MPI itself.
- NVSHMEM4Py host initialization does not initialize device state in the
  separately linked DAE CUDA module. `dae.nvshmem.init()` therefore initializes
  the module after host initialization, and `dae.nvshmem.finalize()` releases
  allocations and finalizes the module before owned host state.

## Allocation And Signal Rules

- `init_signal_space()` is a Python convenience over
  `zeros(..., dtype=torch.uint64)`, so signals follow the same collective
  allocation rules as every other symmetric tensor. Python computes indexed
  signal addresses directly from the tensor and gives them to the official
  NVSHMEM binding.
- Every PE must call `init_signal_space()`, `empty()`, `zeros()`, and
  `finalize()` in the same order.
- Returned tensors are non-owning Torch views. They become invalid when
  `finalize()` calls `release_allocations()` in reverse allocation order.
- Signal operations use the official low-level Python binding and run on the
  current Torch CUDA stream by default.
- There is no NVSHMEM-specific launcher. Initialize with `dae.nvshmem`, then
  construct the ordinary `dae.launcher.Launcher` with `signal_array=signals`
  and `benchmark_barrier=dae.nvshmem.benchmark_barrier`.
- `Launcher` forwards the signal tensor plus independent ordinary
  communication and PoolInst instruction arrays through `runtime.launch_dae`.
  These are statically assembled roles inside VDCores kernels, not auxiliary
  CUDA streams or helper kernels. A
  separately compiled ninth warp executes `COMM_NVSHMEM_*` and the generic
  `COMM_MEMORY_POOL_*` proof path. `POOL_SLICE_EXCHANGE` selects its own
  eight-warp execute type and never enters that interpreter. Ordinary
  alloc/load/store paths still have no NVSHMEM device calls.
- `dae.nvshmem.benchmark_barrier()` synchronizes the device and all PEs. Base
  `Launcher.bench()` invokes the configured callback before every measured
  iteration, so multi-rank profile timestamps begin only after all ranks are
  ready.

## Verification Boundary

`tests/test_nvshmem.py` is intentionally only an import/API smoke test.
`tests/test_memory_pool.py` covers the request ABI and host reference semantics,
while `tests/test_memory_pool_gpu.py` is an opt-in singleton GPU test. Real
cross-node validation runs `app/python/nvshmem/example.py` and the applications
under `app/python/memory_pool/` with `ibrun` on Vista.
