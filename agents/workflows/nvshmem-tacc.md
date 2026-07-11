# TACC NVSHMEM Python Workflow

Use this for the optional MPI-bootstrapped DAE runtime on Vista.

1. Request one MPI task per GH node/GPU:

```bash
idev -p gh-dev -N 2 -n 2 -tpn 1 -t 01:00:00
```

2. Activate the CUDA 13/PyTorch Conda environment and install the official
   binding versions compatible with NVSHMEM 3.4.5 and PyTorch CUDA 13.0:

```bash
OMPI_CC=gcc \
MPICC=/opt/apps/nvidia24/openmpi/5.0.5/bin/mpicc \
python -m pip install --no-binary=mpi4py -r requirements.txt
```

3. Configure the runtime:

```bash
export NVSHMEM_HOME="$CONDA_PREFIX/lib/python3.13/site-packages/nvidia/nvshmem"
export NVSHMEM_BOOTSTRAP=MPI
export NVSHMEM_REMOTE_TRANSPORT=ibrc
export NVSHMEM_IB_ENABLE_IBGDA=1
export NVSHMEM_IBGDA_NIC_HANDLER=gpu
export NVSHMEM_SYMMETRIC_SIZE=512M
```

4. Build both DAE extensions through the unified setup and run the import
   smoke check:

```bash
make nvshmem-smoke
```

The build must define `DAE_ENABLE_NVSHMEM`, produce both `dae.runtime` and
`dae._nvshmem_runtime`, and link only the allocator extension to
`libnvshmem_host.so`. It must not compile a second MPI/NVSHMEM lifecycle layer
or signal-specific allocator state.

5. Confirm that `mpi4py` uses the loaded TACC OpenMPI:

```bash
python -c 'from mpi4py import MPI; print(MPI.Get_library_version())'
```

6. Run real collective verification only inside the allocation:

```bash
ibrun python app/python/nvshmem/example.py
```

Every rank must reach allocations, signal initialization, barriers, benchmark
iterations, and finalization in the same order. Use `ibrun`, not `mpirun`, on
TACC. The checked-in pytest is compile/import smoke coverage and is not a
substitute for this multi-node run.

Use the ordinary launcher after initializing NVSHMEM:

```python
import torch
import dae.nvshmem as nvshmem
from dae.launcher import Launcher

runtime = nvshmem.init(symmetric_size="512M")
signals = nvshmem.init_signal_space(runtime.num_pes)
dae = Launcher(
    num_sms=1,
    device=torch.device("cuda", runtime.device),
    signal_array=signals,
    benchmark_barrier=nvshmem.benchmark_barrier,
)
```
