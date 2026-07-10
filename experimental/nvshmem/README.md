# Minimal NVSHMEM IBGDA Example on TACC Vista

This example builds and runs a minimal CUDA/NVSHMEM program on TACC Vista GH nodes. It uses normal NVSHMEM device APIs with IBGDA enabled through environment variables, and launches the program with TACC `ibrun`.

## 1. Get an interactive TACC allocation

Use `idev` to request two GH nodes, with one MPI task per node:

```bash
idev -p gh-dev -N 2 -n 2 -tpn 1 -t 01:00:00
```

Option meanings:

```text
-p gh-dev      Use the GH development partition.
-N 2           Request 2 compute nodes.
-n 2           Request 2 total MPI tasks.
-tpn 1         Run 1 MPI task per node.
-t 01:00:00    Request 1 hour of interactive time.
```

For this example, `-N 2 -n 2 -tpn 1` is important because each Vista GH node has one GPU, so we want one MPI rank / NVSHMEM PE per GPU.

## 3. Install NVSHMEM and its official Python binding

Use the NVSHMEM4Py release matched to NVSHMEM 3.4.5. Keep CUDA Python at
13.0.3 because that is also the exact version required by the CUDA 13 PyTorch
wheel used by this project:

```bash
OMPI_CC=gcc \
MPICC=/opt/apps/nvidia24/openmpi/5.0.5/bin/mpicc \
python -m pip install --no-binary=mpi4py -r requirements.txt
```

Building `mpi4py` through the loaded TACC wrapper keeps it on the same OpenMPI
5.0.5 ABI used by `ibrun`.

## 4. Create NVSHMEM shared-object symlinks if needed

Some pip-installed NVSHMEM packages may include versioned `.so` files but not the unversioned linker names. Check:

```bash
export NVSHMEM_HOME=$CONDA_PREFIX/lib/python3.13/site-packages/nvidia/nvshmem
ls -lh $NVSHMEM_HOME/lib/libnvshmem*
```

If `libnvshmem_host.so` is missing but a versioned file exists, create the symlink:

```bash
cd $NVSHMEM_HOME/lib

ln -s libnvshmem_host.so.* libnvshmem_host.so
```

If `libnvshmem_device.a` is missing, the NVSHMEM installation is incomplete and should be reinstalled:

```bash
python -m pip install --force-reinstall nvidia-nvshmem-cu13
```

## 5. Build

The standalone CUDA bandwidth example still uses its local Makefile:

```bash
make clean
make
```

This should produce:

```bash
./main
```

Build the ordinary DAE runtime and optional symmetric-allocation extension from
the repository root through the unified `setup.py`:

```bash
make nvshmem-smoke
```

This sets `DAE_ENABLE_NVSHMEM=1`, builds `dae.runtime` and
`dae._nvshmem_runtime`, and verifies that the optional module imports without
initializing a collective job. MPI bootstrap and host operations come from the
official `nvshmem.core` / `nvshmem.bindings` packages; the DAE extension only
allocates symmetric Torch tensors and signals.

## 6. Run with `ibrun`

Set the runtime environment:

```bash
export NVSHMEM_BOOTSTRAP=MPI
export NVSHMEM_REMOTE_TRANSPORT=ibrc
export NVSHMEM_IB_ENABLE_IBGDA=1
export NVSHMEM_IBGDA_NIC_HANDLER=gpu
export NVSHMEM_SYMMETRIC_SIZE=512M
```

Launch with TACC `ibrun`:

```bash
ibrun ./main 128 50
```

Run the Python binding smoke test from the repository root:

```bash
ibrun python app/python/nvshmem_example.py
```

The Python test performs a one-SM DAE TMA copy between symmetric Torch tensors
and exchanges one stream-ordered NVSHMEM signal per PE.

When the NVSHMEM launcher is used with `dae_app(..., -b)`, every measured
iteration performs a device synchronization followed by an all-PE barrier
before launching the timed kernel.

Arguments:

```text
128    Transfer chunk size in MiB.
50     Number of iterations.
```

The example performs a ring put:

```text
PE 0 -> PE 1
PE 1 -> PE 0
```

## 7. Expected output

```text
TACC:  Starting up job 813026
TACC:  Setting up parallel environment for OpenMPI mpirun.
TACC:  Starting parallel tasks...
NVSHMEM normal-API IBGDA ring-put: pes=2 chunk=128 MiB iters=50 max_time=142.644 ms per_pe_data=6.25 GiB per_pe_bw=43.82 GiB/s aggregate_bw=87.63 GiB/s PASS
TACC:  Shutdown complete. Exiting.
```

The network bandwidth on TACC is 400gb/s. In this simple experiment, we got ~344gb/s, it's a pretty decent utilization!

## 8. Notes

Use `ibrun`, not `mpirun`, on TACC.

If you accidentally launch too many MPI ranks, the program may print an error like:

```text
local MPI ranks > local GPUs
```

In that case, re-enter `idev` with:

```bash
idev -p gh-dev -N 2 -n 2 -tpn 1 -t 01:00:00
```

## 9. Python API

Use the isolated launcher when a DAE process needs NVSHMEM:

```python
import torch
from dae.nvshmem_launcher import Launcher

dae = Launcher(num_sms=128, symmetric_size="2G")
signals = dae.init_signal_space(dae.num_pes)
weights = dae.empty((4096, 4096), dtype=torch.bfloat16)
output = dae.zeros((8, 4096), dtype=torch.bfloat16)

# Build and launch schedules exactly as with dae.launcher.Launcher.
# `weights` and `output` are normal contiguous CUDA tensors locally.
```

The same functions are available without a launcher:

```python
import dae.nvshmem as nvshmem

runtime = nvshmem.init(symmetric_size="2G")
signals = nvshmem.init_signal_space(1024)
tensor = nvshmem.empty(4096, dtype=torch.float32)
```

Allocation and finalization rules:

- Call `init()` before creating CUDA tensors so local MPI rank to GPU mapping is established first.
- Every PE must call `init_signal_space()`, `empty()`, and `zeros()` in the same order with identical sizes and dtypes.
- Symmetric tensors are non-owning Torch views. The runtime tracks their allocations and frees them collectively, in reverse order, during `finalize()`.
- Do not use a symmetric tensor after `finalize()`. Finalization itself must be called by every PE after all CUDA work completes.
- `signal()` and `wait_signal()` enqueue operations on the current Torch CUDA stream unless another stream is supplied.
- `Launcher.bench()` synchronizes all NVSHMEM PEs before every measured iteration.
- The runtime defaults unset Vista variables to MPI bootstrap, `ibrc`, IBGDA, GPU NIC handling, and a `512M` symmetric heap. Explicit environment variables remain authoritative except for a `symmetric_size=` argument.
