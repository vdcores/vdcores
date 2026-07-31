# GB300 NVLink Pool Workflow

1. Build the pool-only Blackwell target:
   `make gb300-nvlink-pyext`.
2. Confirm `sm_100a`, `DAE_ENABLE_LOCAL_POOL=1`,
   `DAE_POOL_DATA_PATH_NVLINK=1`, and only `OP_TERMINATEC` in build output.
   `cuobjdump --dump-resource-usage build/local_pool/runtime.o` is the fastest
   spill/resource check.
3. Run the NVSHMEM-free local harness in one process. For the matched EP shape:
   `python benchmarks/pool_slice_local_reduce.py --devices 2,3 --backend multimem --tokens 128 --hidden 7168 --readers-per-pe 8 --top-k 8 --routing clustered --warmup 10 --iterations 30 --launch-cpus 90,91`.
4. Require `correctness=exact`. Test striped weighted routing as a separate
   multi-contributor correctness gate before accepting a reduction change.
5. For transport A/B, rebuild with
   `make nvshmem-pyext DAE_BUILD_PROFILE=gb300 DAE_POOL_DATA_PATH=nvshmem`.
   This is a compile/compatibility check on a local-only node; do not execute
   NVSHMEM or claim its timing as the local path.
6. Keep the staged worker as the performance default. Compile
   `DAE_POOL_LOCAL_DIRECT_SCATTER=1` only as an A/B: it multiplies remote
   payload writes by fanout and is slower for clustered top-k=8.
7. Compare rank-maximum medians with the exact same device IDs, CPU pins,
   shape, warmup, iteration count, and expert-ready/output-ready boundaries.
   NCCL-EP LL supports 2, 4, or multiples of 8 ranks, not 3.
