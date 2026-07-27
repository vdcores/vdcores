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
- runtime communication: a separately compiled `9`-warp envelope containing
  the default compute+memory VM plus one ordinary `CommInst` warp.

The separate communication object is register-capped because a nine-warp CTA
has a lower Hopper per-thread register residency limit. Default, mixed pool,
and fixed-pool objects do not inherit that cap or the communication decoder.

## PoolInst Assembly

`PoolInst` is a distinct 16-byte instruction ABI. Each entry in
`include/dae/pool_opcode.cuh.inc` binds a wire opcode to a concrete executor
type with:

- `num_warps`: the complete physical CTA owned by the instruction;
- `max_registers`: the compile-time kernel budget;
- `execute(...)`: the all-warp implementation.

Host dispatch selects and instantiates that executor type. Device code enters
it before allocating the ordinary VM state and performs no PoolInst opcode
switch. A fixed pool kernel therefore contains no compute, allocator, load,
store, or ordinary communication interpreter. The runtime mixed kernel may
assign other blocks in the same grid to the default VM, but all blocks retain
the common eight-warp physical envelope.

`PoolSliceExchangeExecuteWarp` is the first registered executor. It currently
owns eight warps: one coordinator, configurable pack workers, and the remaining
dynamic receive/return workers.

## Per-Block Configuration

`DaeCoreConfig` is an eight-byte host/device ABI with three logical kinds:

- `COMPUTE_MEMORY`: four compute warps plus allocator/store and one or two load
  warps; an ordinary communication warp is valid only in the separately
  compiled communication envelope;
- `POOL`: the entire envelope runs the compile-time selected PoolInst;
- `INACTIVE`: no virtual core executes.

Use a fixed kernel variant when every block has the same role so unused code
and physical warps disappear. Use runtime configuration only when one launch
must combine ordinary VDCores blocks with pool blocks.

## Synchronization Boundary

PoolInst and ordinary VM blocks interact through named HBM/barrier objects,
not implicit CTA-wide ordering:

- `pool_signal_release/ready` provides device-scope release/acquire ordering
  for local VDCores producer and reader barriers;
- dependency counters use scoped atomic release/add and acquire/load;
- remote descriptors use NVSHMEM put-with-signal;
- remote NBI payload operations are completed by a matching quiet before the
  message-specific phase signal advances.

No explicit system-wide fence appears in the pool protocol. A system-scoped
atomic is used only when ordinary HBM stores are about to become an RDMA GET
source; all other local dependencies remain GPU scoped.

## Entry Files

- physical/logical configuration: `include/dae/core_config.cuh`,
  `python/dae/core.py`;
- kernel assembly: `include/dae/dae2.cuh`, `src/runtime.cu`,
  `src/runtime_comm.cu`;
- PoolInst ABI/registry: `include/dae/context.cuh`,
  `include/dae/pool_opcode.cuh.inc`;
- current executor: `include/dae/pipeline/poolinst.cuh`;
- launcher selection: `python/dae/launcher.py`.
