# TACC NVSHMEM Python Workflow

Use this for the optional MPI-bootstrapped DAE runtime on Vista.

1. Request one MPI task per GH node/GPU:

```bash
idev -p gh-dev -N 2 -n 2 -tpn 1 -t 01:00:00
```

2. Activate the CUDA 13/PyTorch environment and locate NVSHMEM:

```bash
export NVSHMEM_HOME="$CONDA_PREFIX/lib/python3.13/site-packages/nvidia/nvshmem"
export NVSHMEM_BOOTSTRAP=MPI
export NVSHMEM_REMOTE_TRANSPORT=ibrc
export NVSHMEM_IB_ENABLE_IBGDA=1
export NVSHMEM_IBGDA_NIC_HANDLER=gpu
export NVSHMEM_SYMMETRIC_SIZE=512M
```

3. Build the ordinary and optional extensions:

```bash
make pyext
make nvshmem-pyext
```

The optional build must show `nvcc` compiling
`src/torch_nvshmem_runtime.cu` and `mpicxx` performing the final link.

4. Verify the extension without initializing MPI:

```bash
python -c 'import dae._nvshmem_runtime as r; print(r.is_initialized())'
readelf -d python/dae/_nvshmem_runtime*.so | grep -E 'libmpi|libnvshmem'
```

5. Run the collective smoke test only inside the allocation:

```bash
ibrun python experimental/nvshmem/python_binding.py
```

Every rank must reach symmetric allocations, signal initialization, barriers,
and finalization in the same order. Never use `mpirun` in place of TACC
`ibrun`.

Expected per-rank output includes the copied first/last values and the signal
received from the preceding PE, followed by `TACC: Shutdown complete`.
