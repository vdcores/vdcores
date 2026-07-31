# NCCL GIN PoolInst Backend

The weighted PoolInst runtime has a compile-time NCCL GIN/GDAKI backend. It
reuses the unified dispatch `DynamicRead<Copy>` and combine
`DynamicRead<ReduceAdd>` protocol; only remote byte movement and remote signal
publication differ from the NVSHMEM assembly.

## Assembly And Ownership

- `make gin-pyext` builds a separate runtime with
  `DAE_ENABLE_NCCL_GIN=1`. NVSHMEM and GIN cannot coexist in one extension, so
  no transport branch enters PoolInst.
- `python/dae/gin.py` creates the MPI/NCCL communicator, forces GDAKI, creates
  an exclusive full-connect device communicator, and registers one aligned
  HBM arena per PE. Every remotely addressed pool object is a view into that
  window. Pool configuration tensors remain local.
- Host setup installs `ncclDevComm`, `ncclWindow_t`, and arena bounds in device
  constant state. Timed dispatch/combine remains one VDCores PoolInst program;
  the benchmark uses only PoolInst `g_events` timestamps.
- The GIN opcode has its own execute-warp type. Existing NVSHMEM PoolInst and
  default compute/memory assemblies are unchanged.

## Data And Metadata Transport

The portable GIN path uses public `ncclGin` put, put-value, strong VA-add
signal, aggregation, and flush operations. Data and metadata use disjoint
context/QP partitions: three quarters for activation/return payload and one
quarter for route envelopes. RC ordering couples each named readiness update
to its preceding payload on that QP.

The optimized dispatch build adds
`include/dae/pool_gin_gdaki_sgl.cuh`. NCCL still owns communicator setup,
windows, QPs, keys, queue credits, and teardown. The helper only constructs
the activation WQE:

1. Reserve one consecutive SQ range on the CTA's NCCL-owned RC QP.
2. Encode up to eight independently addressed BF16 activation rows as local
   data segments of one RDMA WRITE into a contiguous destination interval.
3. Append an inline progress-generation WRITE after each SGL WQE.
4. Mark the complete range ready and ring one doorbell.

The destination may execute the newly visible prefix after any progress word;
it does not wait for the full group. This preserves metadata/data independence
and progressive gather while eliminating source staging and cutting dispatch
WQE/doorbell overhead. Combine rows are already contiguous; the retained
combine path uses public GIN aggregation because a raw payload-plus-signal WQE
pair delayed progress and regressed 8-PE/256-token latency by about 4%.

No `__threadfence_system` is used. Local dependencies are GPU-scope atomics;
remote dependencies are the exact same-QP payload/signal pairs. The raw helper
relies on NCCL GDAKI internal structs and DOCA GPU verbs headers and therefore
must be version-pinned and rebuilt with the matching NCCL device headers.

## Version Constraint

NCCL4Py 0.3.1 reports a 2.30.4 requirements-structure version while Vista's
loaded NCCL is 2.30.7. GDAKI changed its device-context layout in 2.30.5. If
the old version reaches `create_dev_comm`, device context indexing is wrong
after context zero and PoolInst dereferences a null QP. `GinRuntime` stamps
`requirements._lowpp.version` from `nccl.get_lib_version()` before creation.
Keep this compatibility step until NCCL4Py and libnccl are built from matching
headers; revalidate it whenever either package changes.

## Build And Benchmark

On a compute node with the NCCL4Py environment active:

```bash
make -B gin-pyext \
  PYTHON="$NCCL_EP_ENV/bin/python" \
  NCCL_HOME="$NCCL_HOME" \
  DAE_POOL_SLICE_RAW_SGL=1 \
  DAE_POOL_SLICE_RAW_SGL_WIDTH=8

ibrun -n 8 "$NCCL_EP_ENV/bin/python" \
  benchmarks/pool_slice_gin_benchmark.py \
  --tokens-per-pe 128 --hidden-size 7168 \
  --experts-per-pe 8 --top-k 8 --pool-blocks 64 \
  --gin-contexts 32 --gin-queue-depth 1024
```

The raw pool-only entry compiles with 140 registers, 416 bytes shared memory,
a 192-byte stack frame, and zero spills. The mixed entry uses 176 registers,
15,040 bytes shared memory, a 272-byte stack frame, and zero spills. The build
emits no WGMMA warnings.

## Final Vista Results (2026-07-30)

One PE ran on each GH200 node. The workload was BF16 hidden 7168, eight
experts/PE, top-k 8 clustered routing, source-preloaded input, in-place identity
experts, weighted combine, four warmups, and ten samples. Values are
rank-maximum median internal PoolInst times. Every case was bit-exact and
retired all protocol state.

| PEs | tokens/PE | CTAs | GIN contexts | GIN raw SGL | retained NVSHMEM | gain |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 32 | 32 | 8 | 0.100 ms | 0.106 ms | 5.5% |
| 4 | 128 | 64 | 16 | 0.138 ms | 0.154 ms | 10.1% |
| 4 | 256 | 64 | 16 | 0.238 ms | 0.274 ms | 13.2% |
| 8 | 32 | 64 | 16 | 0.108 ms | 0.113 ms | 4.7% |
| 8 | 128 | 64 | 32 | 0.152 ms | 0.163 ms | 6.6% |
| 8 | 256 | 64 | 16 | 0.247 ms | 0.265 ms | 6.9% |

The retained NVIDIA NCCL-EP LL controls for the same shapes are
0.146/0.352/0.629 ms at four PEs and 0.161/0.396/0.704 ms at eight PEs. Those
are external CUDA-event benchmark totals, while both PoolInst columns use
internal VM timestamps; retain that distinction in comparisons.

## Entry Files

- lifecycle/allocation: `python/dae/gin.py`
- public transport: `include/dae/pool_gin_transport.cuh`
- raw dispatch SGL: `include/dae/pool_gin_gdaki_sgl.cuh`
- executor integration: `include/dae/pipeline/poolinst.cuh`,
  `include/dae/pool_slice.cuh`, `src/runtime.cu`
- standalone benchmark: `benchmarks/pool_slice_gin_benchmark.py`
