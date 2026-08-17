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

- NVIDIA NCCL EP: official `NVIDIA/nccl` checkout at
  `/scratch/11362/depctg/vdcores-baselines/nccl-ep-current`, commit
  `5067397c2676d5aed50042fc39e5c8ee96eb0027`; `nccl4py==0.3.1`,
  `libnccl_ep.so` version 0.1.0, and `nvidia-nccl-cu13==2.30.7`.
- UCCL: `/home1/11362/depctg/projects/uccl`, commit
  `f071f2e31239cd7d673bf2c9369b5cebe1b98457`.
- Triton-distributed:
  `/home1/11362/depctg/projects/Triton-distributed`, commit
  `1512a81189315eca7ba38f884aaf239469a88ba3`, with Triton submodule
  `f53694a72a1e4f464fa245df2c7305ccda7cb2a9`.

Record any local compatibility patch with the source pin and keep the external
checkout's `git diff` in the task log. Do not silently compare a modified
algorithm.

## NVIDIA NCCL EP

`benchmarks/nccl_ep_reference.py` is the real NCCL EP LL boundary. The
similarly shaped dense ring control has been renamed
`benchmarks/dense_nccl_ring_reference.py`; never report that control as NCCL
EP.

Use an external environment containing the official CUDA-13 `nccl4py` package
and a matching NCCL runtime. A lean LL-only setup needs `nccl4py==0.3.1`,
`cuda-core~=1.0`, `cuda-pathfinder>=1.5.4`, and
`nvidia-nccl-cu13==2.30.7`; install all four into the same environment so
CUDA Pathfinder does not select an older NCCL from the base environment. The
official `nccl4py[cu13]` extra is also valid. Select that same NCCL library for
both Torch and NCCL4Py before Python starts. On Vista:

```bash
python -m venv --system-site-packages \
  /scratch/11362/depctg/vdcores-baselines/nccl-ep-env
/scratch/11362/depctg/vdcores-baselines/nccl-ep-env/bin/python -m pip install \
  nccl4py==0.3.1 cuda-core==1.1.0 cuda-pathfinder==1.6.0 \
  nvidia-nccl-cu13==2.30.7

export NCCL_EP_ROOT=/scratch/11362/depctg/vdcores-baselines/nccl-ep-current
export NCCL_EP_ENV=/scratch/11362/depctg/vdcores-baselines/nccl-ep-env
export LD_LIBRARY_PATH="$NCCL_EP_ENV/lib/python3.13/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH"
export NCCL_GIN_TYPE=3
export NCCL_SOCKET_IFNAME=ibP2s2
```

Then launch one rank/GPU/node:

```bash
ibrun -n 2 "$NCCL_EP_ENV/bin/python" \
  benchmarks/nccl_ep_reference.py \
  --nccl-ep-root "$NCCL_EP_ROOT" \
  --tokens-per-pe 128 --hidden-size 7168 \
  --experts-per-pe 8 --top-k 8 --route-placement clustered \
  --warmup 10 --iterations 30
```

The adapter uses BF16 LL expert-major layout, real weighted combine, and an
untimed identity expert. `--num-qps-per-rank 0` selects one QP per local expert,
matching the official LL benchmark; `--max-num-sms 0` and `--num-channels 0`
retain library auto-tuning. It reports the real de-duplicated dispatch payload
count (one row per token/distinct destination rank), not route-count payload
bytes. The packaged LL kernels support hidden sizes 2048, 2560, 4096, 5120,
6144, 7168, and 8192. Record all three tuning choices with every result.

The benchmark is a steady-state fixed-route measurement: handle creation and
`Handle.update()` are outside the interval. This matches the repeated PoolInst
benchmark, whose route tensor is also written before its timing loop. A study
of per-step dynamic-router overhead must time `Handle.update()` separately or
include it ahead of each dispatch and label that result distinctly.

Expert-major is NCCL EP's LL benchmark default and is the comparable boundary
when the dispatch result must already be expert-contiguous: dispatch transmits
one token message per destination rank and fans it out locally, while combine
returns one post-expert row per route and applies the weights. NCCL EP also has
a rank-major layout that can return one row per source/destination rank, but it
moves local expert scatter and weighted pre-reduction into the caller. Do not
quote rank-major communication-only timing as an expert-ready end-to-end result
unless that caller work is implemented and accounted for.

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
