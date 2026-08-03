# NVSHMEM Python Example

This example runs one MPI rank and NVSHMEM PE per GPU. It verifies a one-SM
VDCores copy between symmetric tensors, then exchanges one signal per PE.

## Build

Start from the CUDA 13 / PyTorch environment described in the repository
`setup.sh`, install the NVSHMEM dependencies, and build both extensions:

```bash
OMPI_CC=gcc \
MPICC=/opt/apps/nvidia24/openmpi/5.0.5/bin/mpicc \
python -m pip install --no-binary=mpi4py -r requirements.txt

export NVSHMEM_HOME="$CONDA_PREFIX/lib/python3.13/site-packages/nvidia/nvshmem"
make nvshmem-pyext
python -m pytest -q tests/test_nvshmem.py
```

## Run on TACC Vista

Request one MPI task per GH node/GPU, configure NVSHMEM, and launch from the
repository root:

```bash
idev -p gh-dev -N 2 -n 2 -tpn 1 -t 01:00:00

export NVSHMEM_BOOTSTRAP=MPI
export NVSHMEM_REMOTE_TRANSPORT=ibrc
export NVSHMEM_IB_ENABLE_IBGDA=1
export NVSHMEM_IBGDA_NIC_HANDLER=gpu
export NVSHMEM_SYMMETRIC_SIZE=512M

ibrun python app/python/nvshmem/example.py
```

All ranks must allocate and finalize symmetric tensors in the same order. Use
`ibrun`, not `mpirun`, on TACC. Multi-rank execution requires a Vista
allocation; the local smoke check is
`python -m pytest -q tests/test_nvshmem.py`.
