# VDCores Communication Core

Communication and pool execution are instruction domains inside `dae2`. They
are not auxiliary kernels, host callbacks, or CUDA streams.

## VM Shape And Isolation

The default NVSHMEM build remains four compute plus four memory warps (`256`
threads). A separately compiled `288`-thread variant adds one ordinary
communication warp. Existing compute dispatch, allocation, store, and
load-warp ids and code paths are unchanged.

`CommInst` is a 16-byte instruction containing four 16-bit fields and one
64-bit address. It has no allocator flags, consumes no shared-memory slots, and
never enters the memory/compute queues. Opcode zero terminates, so an untouched
communication stream is inert. The default and fixed-pool kernels do not
instantiate this interpreter.

The unified pool hot path is a distinct `PoolInst`. Its registry binds
`POOL_SLICE_EXCHANGE` to `PoolSliceExchangeExecuteWarp`; host dispatch
instantiates that type, and the device performs no opcode switch. Its eight
resident warps become one coordinator, configurable pack workers, and
receive/return workers. A fixed pool kernel contains no ordinary VM; a mixed
eight-warp kernel may assign other blocks to the unchanged compute/memory VM.

The single `Launcher.launch_dae` call carries compute, memory, ordinary
communication, and pool instruction arrays plus optional per-block core
configurations. There are no direct pool-kernel bindings. VDCores timing writes
only the existing per-block `g_events` space. NCCL and CUDA-event timing live
strictly under `benchmarks/`.

## Unified Compact Pool Protocol

Each PE owns one logical pool slice and one communication-specialized block.
The source metadata is stable-grouped by destination PE and local reader.

1. An ordinary VDCores writer block copies every active source row exactly
   once into source-owned token slots and releases one barrier per TMA-sized
   write chunk.
2. Router metadata contains a sorted unique source-row list per target and a
   route-to-compact-row index per expert route. The pool builds one 64-byte
   descriptor per target with up to eight target-local reader counts, including
   a valid zero-row batch.
3. Coordinator warp 0 starts metadata publication while worker warps claim
   `(target, compact shard)` tasks. A worker waits only for the source chunk
   containing its next unique row, packs remote rows into private contiguous
   HBM, and issues one NBI PUT per nonempty shard. Self-routed rows stay in the
   source token pool and are resolved directly during gather. There is no
   barrier between metadata work and remote source packing.
4. Each issuing PoolInst CTA quiets its own NBI work and releases a local
   generation. Rank zero joins the generations, publishes the data phase only
   for nonempty destinations, and scans all source signal words lane-parallel.
   Empty descriptors are initially published at the data-complete value.
5. Target workers use the route-to-compact maps to fan rows from their
   source-indexed receive table into deterministic expert input ranges. Rank
   zero advances the group ticket and releases ordinary reader blocks only
   after every gather CTA completes.
6. After reader compute barriers retire, workers issue one contiguous NBI PUT
   for each nonempty `(reader, source)` return range. The default path quiets
   once per direction; the optional path attaches one signal per return batch
   and permits early source scatter.
7. A target that validates a zero-row source acknowledges its return phase
   immediately; nonempty slices publish the merged return phase after quiet.
   Once the source observes all phases, all PoolInst warps scatter token-major
   partial rows to saved origins.

This is a dependent gathered read, not a special EP transport. Route metadata
names which source slots each dynamic reader consumes; the shared sender set
and sequence determine when the read can retire.

## HBM ABI And Merged Signals

All remotely addressed buffers use same-order NVSHMEM symmetric allocations.
The hot path uses:

- source token slots and a two-half delivery buffer: source-private compact
  staging for remote targets, then target-local compact receive rows by remote
  source; self routes read token slots directly;
- a unique source-row list per target and route-to-compact index metadata;
- one descriptor array indexed by source PE;
- one local receive record per `(reader, source)`;
- contiguous reader input/output regions;
- a source return inbox and saved origin-row array;
- one source-indexed signal word per PE.

The signal word carries three monotonic values per sequence: metadata visible,
packed data visible, and returned data visible. Thus one `P`-word range replaces
three independent signal arrays. A target can retire a zero-row source after
metadata validation; nonempty sources also wait for data. All readers with the
same sender set share the group-ready ticket, so there is no epoch-end message.

The baseline permits one outstanding sequence per buffer set. More overlap is
obtained by overlapping work within that sequence, not by adding unbounded
mailboxes or fragmented allocation.

## Warp Roles

- warp 0: descriptor/signal publication, lane-parallel phase polling, reader
  release, compute-barrier polling, and phase timestamps;
- warps 1--7 on payload CTAs: compact target-shard packing, target-side HBM
  fanout, and contiguous return PUTs;
- all eight warps: final source scatter.

`pack_warps=0` selects the measured compact-shard policy: four shards below a
512 KiB source table and all seven worker warps at or above it. An explicit
config value overrides the policy without changing the base VM. With multiple
PoolInst CTAs the optional dedicated-coordinator flag keeps rank zero on
metadata, signal polling, and QP progress while all other CTAs remain payload
executors.

## Ordinary Reducer Composition

Weighted return can move destination reduction out of PoolInst without moving
it out of VDCores. The Launcher assembles additional ordinary compute+memory
blocks beside the PoolInst blocks:

- expert-atomic reducers acquire one expert-ready signal, so balanced or
  skewed experts may begin independently, and atomically contribute to a
  pool-owned token buffer;
- token-sharded reducers acquire the local expert set, own disjoint compact
  token rows, and avoid atomics; the count is configurable through
  `PoolSliceConfig.reducer_count` up to 32 blocks.

Both use raw-address memory instructions for operands and PoolRawAddress for a
named completion release. PoolInst acquires the contiguous reducer completion
range before it posts partial-token NVSHMEM batches. The reducers need no
ordinary communication warp because PoolInst remains the network owner; adding
one would consume registers/warps without changing this dependency graph.

## Ordering Contract

1. A descriptor is published with put-with-signal. The current compact route
   span is a preceding NBI metadata write by the same issuer; the next protocol
   revision merges both into one contiguous signaled message, removing the
   need to rely on QP issue order or add a broad fence.
2. An issuing CTA uses NVSHMEM quiet before releasing its GPU-local completion
   generation. Rank zero's acquire of that generation orders only local
   PoolInst bookkeeping; the NVSHMEM operation supplies remote visibility.
3. Target gathered reads start only after the source data phase is observed.
4. All dispatch RMAs are nonblocking and several PoolInst CTAs may keep
   independent QPs in flight. The current compact direct-put path uses the
   pinned block-cooperative quiet per issuing CTA.
5. Reader completion barriers use GPU-scope release/acquire operations to
   order ordinary VDCores stores before return workers read expert output.
6. All return RMAs are nonblocking; a quiet precedes return-phase publication.
7. Observing all return phases makes every source inbox range consumable before
   scatter.
8. Signal comparisons use `>=` and sequence-derived values. Signal words are
   monotonic and are not cleared between phases.

A valid zero-row descriptor uses the data-complete value, eliminating a
second remote signal. After the target has observed the complete dispatch set,
it may also publish return-complete for that empty pair before expert work.
Invalid descriptors never take this shortcut, so protocol errors cannot hide
behind phase elision.

There is no explicit `__threadfence_system` or `nvshmem_fence` in the protocol.
GPU-scoped atomics order local operators; system-scoped atomics publish only
ordinary HBM ranges that a remote NIC will read; put-with-signal and quiet
order the named NVSHMEM messages. These mechanisms are deliberately not
treated as interchangeable.

## Transport Boundary

The installed Vista NVSHMEM 3.4.5 build exposes IBGDA environment mapping but
not application-created QP handles. The macro keeps target shards independent
and NBI so IBGDA can use its RC/DCI transport. The current production shape
uses 24 PoolInst CTAs/RC16 at two PEs and 32 CTAs with RC16/RC24 at four/eight
PEs. QP count is a shape/assembly parameter, not a protocol constant; sweep it
together with PoolInst CTA count at every 2/4/8-PE point.

NVSHMEM device calls remain behind communication/pool-only function
boundaries. The nine-warp ordinary communication object alone carries the
lower register cap; default and PoolInst assemblies preserve their independent
budgets. Always inspect the linked extension for WGMMA diagnostics, entry
spills, and launch-resource growth.

## Entry Files

- VM/ISA: `include/dae/context.cuh`, `include/dae/core_config.cuh`,
  `include/dae/dae2.cuh`, `include/dae/pipeline/commwarp.cuh`,
  `include/dae/pipeline/poolinst.cuh`
- generic mailbox pool: `include/dae/memory_pool.cuh`,
  `python/dae/memory_pool.py`
- batched dynamic read: `include/dae/pool_slice_abi.cuh`,
  `include/dae/pool_slice.cuh`, `python/dae/pool_slice.py`
- encoding/launch: `python/dae/instructions.py`, `python/dae/launcher.py`
- correctness app: `app/python/memory_pool/pool_slice_dynamic_read.py`
- external comparison: `benchmarks/pool_slice_nccl_compare.py`
