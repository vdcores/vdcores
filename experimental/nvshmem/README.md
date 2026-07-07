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

## 3. Install NVSHMEM if it is not already available

If anything is missing, install the CUDA 13 NVSHMEM package into the conda environment:

```bash
python -m pip install nvidia-nvshmem-cu13
```

## 4. Create NVSHMEM shared-object symlinks if needed

Some pip-installed NVSHMEM packages may include versioned `.so` files but not the unversioned linker names. Check:

```bash
export NVSHMEM_HOME=$(CONDA_PATH)/lib/python3.13/site-packages/nvidia/nvshmem
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

```bash
make clean
make
```

This should produce:

```bash
./main
```

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

