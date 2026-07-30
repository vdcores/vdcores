# GB300 NVLink Pool Workflow

1. Build the pool-only Blackwell target:
   `make gb300-nvlink-pyext`.
2. Confirm `sm_100a`, `DAE_POOL_DATA_PATH_NVLINK=1`, and one selected compute
   op in build output.
3. Launch one MPI rank per local GPU with `--transport nvlink`. Set
   `NVSHMEM_REMOTE_TRANSPORT=none`, `NVSHMEM_IB_ENABLE_IBGDA=0`, and
   `NVSHMEM_DISABLE_NCCL=1`.
4. Start with the two-PE dynamic-read correctness harness. Then test four PEs
   before collecting timings.
5. For transport A/B, rebuild with
   `make nvshmem-pyext DAE_BUILD_PROFILE=gb300 DAE_POOL_DATA_PATH=nvshmem`.
   Do not compare timings from different builds or allocations.
6. Compare the pool against `benchmarks/pool_slice_nccl_compare.py` only after
   the four-PE correctness gate passes.
