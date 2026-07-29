# External EP Baselines

Use this workflow for production-library comparisons without adding third-party
transport code to VDCores. All builds and runs belong on Vista GH compute
nodes, one MPI rank/GPU/node. Keep checkouts and environments outside this
repository.

## Comparison contract

- Match `tokens/PE`, hidden size, experts/PE, top-k, and route placement.
- `clustered` is the default deterministic production-style placement. The
  other placements have the same meaning as the PoolInst and DeepEP harnesses.
- UCCL BF16 is byte-matched to the current PoolInst payload. UCCL FP8 and
  Triton-distributed low-latency FP8 are algorithm-matched but not byte-matched.
- Identity expert work is outside the timed region. Dispatch and weighted
  combine are real library operations.
- External baselines use CUDA events and MPI rank-maximum aggregation.
  VDCores timings continue to come only from internal `g_events`.

## Pinned sources

- UCCL: `/home1/11362/depctg/projects/uccl`, commit
  `f071f2e31239cd7d673bf2c9369b5cebe1b98457`.
- Triton-distributed:
  `/home1/11362/depctg/projects/Triton-distributed`, commit
  `1512a81189315eca7ba38f884aaf239469a88ba3`, with Triton submodule
  `f53694a72a1e4f464fa245df2c7305ccda7cb2a9`.

Record any local compatibility patch with the source pin and keep the external
checkout's `git diff` in the task log. Do not silently compare a modified
algorithm.

## UCCL-EP

Build/install `uccl.ep` from the pinned checkout in an external environment,
then expose the source wrapper with `UCCL_EP_ROOT`. On a one-GPU/node Vista
allocation:

```bash
git -C "$UCCL_EP_ROOT" apply \
  /home1/11362/depctg/vdcores/benchmarks/patches/uccl-gh200-host-atomic-buffer.patch
```

The patch only makes upstream's existing
`UCCL_ATOMICS_USE_HOST_MEMORY=1` selector effective for GH200. Vista mlx5
rejects `ibv_reg_mr` on the otherwise forced CUDA-managed atomic buffer. The
payload remains HBM registered through DMA-BUF; no dispatch/combine code is
changed. Build with `USE_DMABUF=1`, `PER_EXPERT_BATCHING=1`, and enough
`MAX_NUM_GPUS` for the largest sweep.

Set the following Vista runtime choices:

```bash
export UCCL_ATOMICS_USE_HOST_MEMORY=1
export UCCL_SOCKET_IFNAME=ibP2s2
export UCCL_IB_GID_INDEX=-1
export NCCL_SOCKET_IFNAME=ibP2s2
```

Then launch:

```bash
export UCCL_EP_ROOT=/home1/11362/depctg/projects/uccl
ibrun -n 2 python benchmarks/uccl_ep_reference.py \
  --tokens-per-pe 128 --hidden-size 7168 \
  --experts-per-pe 8 --top-k 8 \
  --route-placement clustered --dispatch-dtype bfloat16
```

Repeat with `--dispatch-dtype float8` only when comparing FP8 dispatch paths.
UCCL proxy-thread/NIC tuning is an external benchmark environment choice; log
every non-default setting with the result.

## Triton-distributed

Use an isolated environment because Triton-distributed installs its own Triton
fork and NVSHMEM4py version. Initialize the NVIDIA Triton submodule on the login
node, but build on a GH compute node:

```bash
git -C /home1/11362/depctg/projects/Triton-distributed \
  submodule update --init --depth 1 3rdparty/triton
```

For the tested CUDA 13.0 environment, install `nvidia-nvshmem-cu13==3.6.5`
and `nvshmem4py-cu13==0.3.0` in the isolated environment. The pinned Triton
3.4 fork needs a narrow PTX-version compatibility patch:

```bash
export TRITON_DISTRIBUTED_ROOT=/home1/11362/depctg/projects/Triton-distributed
git -C "$TRITON_DISTRIBUTED_ROOT/3rdparty/triton" apply \
  /home1/11362/depctg/vdcores/benchmarks/patches/triton-distributed-cuda13-ptx.patch
```

The patch maps CUDA 13.x to PTX 9.x; it does not touch a kernel or lowering
pass. Point `LLVM_SYSPATH` at the ARM64 LLVM bundle pinned by the checkout and
build on a compute node:

```bash
export TRITON_OFFLINE_BUILD=1
export TRITON_BUILD_PROTON=0
export USE_TRITON_DISTRIBUTED_AOT=0
MAX_JOBS=4 /path/to/triton-dist-env/bin/python -m pip install \
  -e "$TRITON_DISTRIBUTED_ROOT/python" \
  --no-build-isolation --no-deps --use-pep517 -v
```

The LLVM bundle is about 4.5 GiB and the build tree grows beyond 3 GiB. Put the
bundle in Vista scratch and use a stable symlink when home quota is tight; do
not build or run the compiler on a login node.

Make the external fork win over Torch's bundled Triton and configure the
NVSHMEM UID/IBGDA path before launch:

```bash
export PYTHONPATH="$TRITON_DISTRIBUTED_ROOT/python:$PYTHONPATH"
export NVSHMEM_HOME=/path/to/triton-dist-env/lib/python3.13/site-packages/nvidia/nvshmem
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH"
export NVSHMEM_SYMMETRIC_SIZE=2g
export NVSHMEM_DISABLE_CUDA_VMM=1
export NVSHMEM_BOOTSTRAP=UID
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=ibP2s2
export NVSHMEM_REMOTE_TRANSPORT=ibrc
export NVSHMEM_IB_ENABLE_IBGDA=1
export NVSHMEM_IBGDA_NIC_HANDLER=auto
export NCCL_SOCKET_IFNAME=ibP2s2
```

Keep the NIC handler at `auto` for portable sweeps. Forcing `gpu` worked at two
and four PEs on the tested nodes but left the first dispatch spinning at eight
PEs; `auto` completed the same cached kernel. After installing the pinned
source into the isolated environment:

```bash
export TRITON_DISTRIBUTED_ROOT=/home1/11362/depctg/projects/Triton-distributed
ibrun -n 2 /path/to/triton-dist-env/bin/python \
  benchmarks/triton_distributed_ep_reference.py \
  --tokens-per-pe 128 --hidden-size 7168 \
  --experts-per-pe 8 --top-k 8 --route-placement clustered
```

The low-latency layer requires `hidden-size % 128 == 0` and performs online
FP8 dispatch. The adapter runs one untimed operator pass, verifies per-expert
receive counts and an exact quantized identity return, then times only dispatch
and combine. It converts each current receive layout to BF16 between separate
CUDA-event intervals, so dynamic slot order remains correct and identity expert
work is excluded.

## Result capture

Each adapter prints human-readable timing plus one stable
`ep-baseline-json: {...}` record. Preserve the source commits, allocation,
network/NVSHMEM environment, and exact command alongside that record.
