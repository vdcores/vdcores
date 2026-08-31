# Configurable VDCores Assemblies

VDCores now separates the logical per-block role from the CUDA kernel's
compile-time physical envelope. CUDA fixes `blockDim`, registers, and static
shared memory for every block in a grid, so runtime configuration may disable
or reassign resident warps but cannot reclaim their launch resources.

## Kernel Assemblies

- default compute+memory: four compute warps, allocator, store, and two load
  warps (`8` total);
- one-load compute+memory: the same VM with one load warp (`7` total);
- runtime mixed: an `8`-warp envelope whose blocks may be compute+memory,
  a selected PoolInst, or inactive;
- fixed pool: only the selected PoolInst executor is instantiated; its
  compile-time execute-warp type owns the CTA width and register budget;
- pool CTA-compute: one selected PoolInst scheduler block shares an eight-warp
  grid with ordinary `COMPUTE_MEMORY` blocks whose first `CInst` is a
  CTA-cooperative operator;
- runtime communication: a `9`-warp specialization containing the default
  compute+memory VM plus one ordinary `CommInst` warp.

The communication specialization is capped with per-kernel `__maxnreg__`
because a nine-warp CTA has a lower Hopper per-thread register residency
limit. Default, mixed pool, and fixed-pool specializations do not inherit that
cap or instantiate the communication decoder. All specializations live in one
runtime object; the old duplicate `runtime_comm.o` translation unit became
unnecessary once the cap was a kernel attribute.

## PoolInst Assembly

`PoolInst` is a distinct 16-byte instruction ABI. Slot zero is a program
header; each entry in `include/dae/pool_opcode.cuh.inc` binds that header's
wire opcode to a concrete executor type with:

- `num_warps`: the complete physical CTA owned by the instruction;
- `max_registers`: the compile-time kernel budget;
- `execute(...)`: the all-warp implementation.

Host dispatch selects and instantiates that executor type. Device code enters
it before allocating ordinary VM state. Following header slots form an
immutable, schedule-generated operation queue; the pool worker decodes one
CTA-uniform operation opcode per claimed work item, outside the row loops. A
fixed pool kernel therefore contains no compute, allocator, load, store, or
ordinary communication interpreter. The runtime mixed kernel may assign other
blocks in the same grid to the default VM, but all blocks retain the common
eight-warp physical envelope.

`PoolSliceExchangeExecuteWarp` is the first registered executor. It currently
owns eight warps. Its per-block stream contains one header, one
`DynamicRead<Copy>` per local expert, and, for weighted combine, one
`DynamicRead<ReduceAdd>` per pool plan. Read-only descriptors are replicated
per pool block and prefetched to shared memory during invocation setup. Their
single GPU-scope claim cursor makes them one logical per-PE instruction queue.
The fixed queue and worker cardinalities match, so weighted combine needs one
claim per CTA and no terminal empty-queue probe.

## Cooperative Pool CTA-Compute Assembly

`DAE_KERNEL_POOL_CTA_COMPUTE` is the specialized heterogeneous assembly used
by the GB300 source-gather worker path. Block zero is the sole
`POOL_SLICE_SOURCE_GATHER_SCHEDULER` PoolInst CTA for both phases: scheduler
warp zero owns dispatch queue advancement, scheduler warp one publishes
combine readiness, and that same CTA performs the final dispatch/scatter
join. The remaining blocks are normal `COMPUTE_MEMORY` cores marked with
`CORE_FLAG_CTA_COMPUTE_OPERATOR`.

Each worker contains one fused ordinary `CInst`, selected from
`OP_POOL_SLICE_METADATA_ROUTE`, `OP_POOL_SLICE_REMOTE_SEND`,
`OP_POOL_SLICE_SELF_DISPATCH`, `OP_POOL_SLICE_EXECUTOR`, or
`OP_POOL_SLICE_DISPATCH_BYPASS`. The opcode is the role; no `pool_rank`
comparison assigns it on device. Its three 16-bit arguments carry the static
task ordinal, ring-executor slot, and source-gather stripe. The common return
is fused into the role operator, avoiding a second decode, sidecar load, and
CTA handoff. A dispatch-bypass CTA enters only its gather stripe.

The flag changes participation, not the physical core shape: all eight warps
enter the cooperative handler so CTA barriers are legal, while the established
four-compute/four-memory ownership remains enforced inside the operator.
Compute warps use registers and shared row tiles only. Memory warps own program
and configuration loads, metadata, route expansion, readiness polling, remote
SEND/self-scatter, ring tickets, source-gather global reads, and the returned
global store. The launcher rejects any other first compute opcode or a worker
without its PoolInst sidecar header, and rejects a second non-termination
compute operator in the CTA-wide program.

For BF16 width 7168 and top-k 8, the assembly opts into 129,024 bytes of
dynamic shared memory per CTA: eight 14,336-byte input rows plus one output
row. The accepted grid has one scheduler plus 128 workers (`129` blocks).
Legacy fixed-PoolInst launches retain their original kernel and do not use
this variant.

## Per-Block Configuration

`DaeCoreConfig` is an eight-byte host/device ABI with three logical kinds:

- `COMPUTE_MEMORY`: four compute warps plus allocator/store and one or two load
  warps; an ordinary communication warp is valid only in the separately
  compiled communication envelope. `CORE_FLAG_CTA_COMPUTE_OPERATOR` selects
  the cooperative eight-warp pool-worker specialization described above;
- `POOL`: the entire envelope runs the compile-time selected PoolInst;
- `INACTIVE`: no virtual core executes.

Use a fixed kernel variant when every block has the same role so unused code
and physical warps disappear. Use runtime configuration only when one launch
must combine ordinary VDCores blocks with pool blocks.

## Synchronization Boundary

PoolInst and ordinary VM blocks interact through named HBM/barrier objects,
not implicit CTA-wide ordering:

- local VDCores producer and reader dependencies reuse ordinary countdown
  barriers: initialize a pending single-producer edge to one, `atomicSub` on
  completion, and poll until zero;
- dependency counters use scoped atomic release/add and acquire/load;
- remote descriptors use NVSHMEM put-with-signal;
- remote NBI payload operations are completed by a matching quiet before the
  message-specific phase signal advances.

No explicit system-wide fence appears in the pool protocol. System-scoped
release/acquire operations are confined to the explicit Grace-coherent host
request-ring handoff; all device-only dependencies remain GPU scoped.

`include/dae/scoped_atomic.cuh` is intentionally a small shared PTX primitive
layer, not another runtime. Relaxed bookkeeping uses native CUDA atomics, while
message publication uses exact `.release.gpu`, `.acquire.gpu`, or
`.acq_rel.gpu` operations. A plain `atomicAdd` cannot replace those publication
edges because CUDA native atomics are relaxed and do not order the payload;
using a general fence or `cuda::atomic` would be broader or heavier than the
named dependency requires.

## Consolidation Verification

On 2026-07-29, a same-allocation split-versus-single-object build on GH200
kept the nine-warp entry at 168 registers, 16 barriers, and 14,612 bytes shared
memory with no entry spills. The weighted PoolInst entry remained at 190
registers, 16 barriers, and 14,628 bytes shared memory. A two-PE 128-token
timing bracket measured 0.252 ms single object, 0.269 ms split control, and
0.237 ms single object, so the consolidation introduced no measurable
PoolInst regression.

## Entry Files

- physical/logical configuration: `include/dae/core_config.cuh`,
  `python/dae/core.py`;
- kernel assembly: `include/dae/dae2.cuh`, `src/runtime.cu`;
- PoolInst ABI/registry: `include/dae/context.cuh`,
  `include/dae/pool_opcode.cuh.inc`;
- current executor: `include/dae/pipeline/poolinst.cuh`;
- launcher selection: `python/dae/launcher.py`.
