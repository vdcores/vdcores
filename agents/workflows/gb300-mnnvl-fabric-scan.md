# GB300 NVL72 MNNVL Pool Scan

Use this workflow for the VDCores PoolInst scan on 2, 4, 8, 12, or 16 B300
GPUs. Canonical development and builds are on `10.0.16.25`; deploy the same
revision at the same absolute path to `.23`, `.31`, and `.24` before a
cross-host run.

## Non-negotiable topology and routing contract

- All 16 allocated GPUs are in one NVL72 MNNVL domain. A host boundary, and in
  particular crossing four GPUs, is not a payload-transport boundary.
- CUDA Fabric VMM mappings over the NVLink backplane carry PoolInst metadata,
  activations, signals, and results. TCP/Ethernet may bootstrap MPI, exchange
  opaque CUDA handles, and run barriers only. Do not enable IB, RoCE, IBGDA,
  GDA-KI, or another NIC payload path within this allocation.
- Routing is deterministic random **global top-8**, seed `20260802`. A token's
  eight experts may occupy several destination GPUs. Never replace this with
  clustered or one-destination routing when reporting a production result.
- Peer-direct return is invalid for this contract: independent destinations
  can race or overwrite the same source-token result. Use
  `--reduction-backend source_gather` (or a genuinely reduction-capable
  backend), never ordinary peer stores.
- Direct activation scatter is dispatch-only. It is valid because every route
  writes a distinct expert-input slot; it provides no correctness argument for
  direct return.

NVIDIA documents an NVL72 as one rack-scale NVLink system in the
[MNNVL User Guide](https://docs.nvidia.com/multi-node-nvlink-systems/mnnvl-user-guide/overview.html)
and states that one NVL72 domain supports up to 72 Blackwell GPUs in the
[Multi-Node Tuning Guide](https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/overview.html).
The MNNVL guide also distinguishes TCP-based IMEX mapping information from
GPU reads and writes that traverse the NVLink backplane.

## Build and deploy

On `.25`:

```bash
cd /home/azhpcuser/jiaxinl/vdcores
timeout --kill-after=5s 600s env \
  PATH=/usr/local/cuda/bin:/home/azhpcuser/miniconda3/bin:/usr/bin:/bin \
  make -B gb300-nvlink-pyext \
  PYTHON=/home/azhpcuser/miniconda3/bin/python \
  DAE_COMPUTE_OPS=OP_TERMINATEC \
  DAE_POOL_LOCAL_DIRECT_SCATTER=1 \
  DAE_POOL_SLICE_WARPS=8
```

Run focused Python validation, including the hard rejection of
`reduction_backend="peer_direct"` and both legacy peer-direct instruction
constructors. Also require opcodes 6 and 7 to be absent from
`runtime.pool_execute_warp_types`; the compiled registry and CUDA executor
assemblies must not retain an alternate peer-direct launch path. Then
synchronize only the task-owned source and built artifacts to the same path on
every selected host. Compare
SHA-256 hashes on all hosts before launch. The accepted corrected artifacts
from 2026-08-02 are:

- `python/dae/pool_slice.py`:
  `462aba7901d0d30f83b4129c276f44682f332af6166658600ca8fce3febbc5d9`
- `python/dae/instructions.py`:
  `abe076a3b94db64cf8e748ecd608ab87aa567b455a2b33e909cd29f11ecea25f`
- `python/dae/mnnvl_pool.py`:
  `4d4a0f38c0a9a1a63f49befa7b67dcfb3cc8f8fc883f7eecff87a06366e5b5f3`
- `include/dae/pool_slice.cuh`:
  `b9e5fee9b53f41e0e57103850aa0c7d34fb2798ea86a688133b21edf70d5002e`
- `include/dae/pool_opcode.cuh.inc`:
  `23521b41ad02907a0f2f30301fa1a384b3eddc62711389990105e1ba487a969f`
- `include/dae/pipeline/poolinst.cuh`:
  `24ae48bbe19bdfc0b5921374cf2ba206690748c3a35bcf2b553a92a9070812fa`
- `build/local_pool/runtime.o`:
  `23ddecc247d4959e6864ef025be9ee91301b76351bb4822f400246355e05e1c3`
- `python/dae/runtime.cpython-314-aarch64-linux-gnu.so`:
  `963150abe9de082923cef6fb0464f0dd775ef492c5ff10460ad701c1009c136c`
- `python/dae/_local_pool_runtime.cpython-314-aarch64-linux-gnu.so`:
  `8f58a2db27636268e1926da2462f97650e627f00f0759fe5694545d56b8f9e44`
- `tests/test_pool_slice.py`:
  `d1ab1fffcfa236933abd53e81e9db315e7a2102619b4ce1c7c4bfff9a2eab0b2`

## Canonical benchmark

The matched shape is BF16 width 7168, 128 tokens/GPU, eight experts/GPU,
global top-8, uniform 1/8 weights, identity experts, 10 warmups, and 30
measured iterations. Route construction is outside the steady-state interval.
Inputs change and output buffers are poisoned on every iteration.

Launch from `.25` with:

```bash
cd /home/azhpcuser/jiaxinl/vdcores
env PYTHONPATH=/home/azhpcuser/jiaxinl/vdcores/python \
  /home/azhpcuser/vdcores-baselines/bin/nvl-scan-run <GPU_COUNT> \
  /home/azhpcuser/miniconda3/bin/python \
  /home/azhpcuser/jiaxinl/vdcores/benchmarks/pool_slice_mnnvl_reduce.py \
  --tokens-per-pe 128 --hidden-size 7168 --experts-per-pe 8 --top-k 8 \
  --route-placement random --route-seed 20260802 \
  --reduction-backend source_gather \
  --pool-blocks 128 --group-limit <GROUP_LIMIT> \
  --warmup 10 --iterations 30 --vary-input-every-iteration
```

Use group limits 2/8/4/4/2 at 2/4/8/12/16 GPUs respectively. The launcher
normalizes each rank's working directory, so keep the benchmark path absolute.
It selects two GPUs on `.25` for the 2-GPU point, all four for the 4-GPU point,
then extends the ordered host list `.25,.23,.31,.24` while keeping every
payload in the same all-NVLink MNNVL domain.

The accepted source-gather kernel caps the logical executor domain at 40 when
`num_pes <= 4` and 64 otherwise. Do not use the four-GPU cap at larger scales:
40 logical executors can starve independent source queues and ordered STOPs at
eight or more GPUs. Group limit 8 is likewise non-live at eight GPUs.

Expected route digests are:

| GPUs | SHA-256 route digest |
| ---: | --- |
| 2 | `f0188449fb96cac28ae7647b70a13f06bde06ab60cdf41ea111e12705e6aa5bb` |
| 4 | `921757034c87b39acc75df2edf3777694e97fcfcde372910e8003154350c8721` |
| 8 | `ca7b04e9e28d5ccfb23a20f4e826efa9db2adb781403ca94283f9fd514e39cb9` |
| 12 | `ba693bf85a7e14eeed4db789600bcb24e985dc95775e68d8c3925e764c2a3c24` |
| 16 | `ce5124a837d5a11e9693ebd6b4852e57422a2ca51ca77a5e20692857e9cf9147` |

Accepted 10-warmup/30-sample medians in milliseconds are:

| GPUs | Dispatch | Return tail | Total |
| ---: | ---: | ---: | ---: |
| 2 | 0.063632 | 0.019376 | 0.081264 |
| 4 | 0.059552 | 0.023136 | 0.082688 |
| 8 | 0.070976 | 0.026240 | 0.097136 |
| 12 | 0.071328 | 0.027616 | 0.098576 |
| 16 | 0.075456 | 0.028704 | 0.103840 |

For the matched NCCL-EP control use NCCL-EP 0.1.0, nccl4py 0.3.1, NCCL
2.30.7, expert-major BF16, QP8, and library-auto SM/channels. Set
`NCCL_GIN_TYPE=0`, disable IB, and keep MNNVL enabled. NCCL-EP supports 2,
4, and multiples of 8 ranks; report 12 as unsupported.

## Acceptance and cleanup

Accept a VDCores point only when:

- the route placement, seed, and digest match this workflow;
- every rank reports PoolSlice OK;
- changing-input/poisoned-output validation passes with
  `max_abs <= 0.00390625` (one BF16 quantum for this reduction order);
- the reported result is the median of rank-maximum internal event times;
- the selected host count, PoolInst CTA/group policy, source-gather return,
  and CUDA-Fabric/MNNVL transport are present in the JSON record.

After each run, verify no owned MPI/GPU process remains and that IMEX is
`active`, GPU fabric is `Completed/Success`, and recovery action is
`None`. Terminate only PIDs proven to belong to the failed launch.
