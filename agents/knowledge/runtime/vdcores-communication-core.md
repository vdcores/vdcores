# VDCores Communication Core

Communication and pool execution are instruction domains inside `dae2`. They
are not auxiliary kernels, host callbacks, or CUDA streams.

## VM Shape And Isolation

The default NVSHMEM build remains four compute plus four memory warps (`256`
threads). A `288`-thread specialization adds one ordinary communication warp.
Existing compute dispatch, allocation, store, and load-warp ids and code paths
are unchanged.

`CommInst` is a 16-byte instruction containing four 16-bit fields and one
64-bit address. It has no allocator flags, consumes no shared-memory slots, and
never enters the memory/compute queues. Opcode zero terminates, so an untouched
communication stream is inert. The default and fixed-pool kernels do not
instantiate this interpreter.

The unified pool hot path is a distinct `PoolInst`. Its registry binds generic
and weighted gathered-read opcodes to separate execute-warp types; host
dispatch instantiates one type, and the device performs no opcode or
return-mode switch. Its eight resident warps cooperate on the macro. A fixed
pool kernel contains no ordinary VM; a mixed eight-warp kernel may assign
other blocks to the unchanged compute/memory VM.

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
   route-to-compact-row index per expert route. PoolInst builds one contiguous
   descriptor-plus-queues envelope per target. Queue zero is
   `RESERVE_ROUTES, DATA*, END`; queue one is `DATA*, END`. The compiled
   dispatch consumer interprets `DATA` as `DynamicRead<Copy>`.
3. Coordinator warp 0 starts metadata publication while payload PoolInst CTAs
   issue direct row PUTs from authoritative source slots. There is no source
   activation staging and no barrier between the data and metadata planes.
   Every `DATA` names its exact compact interval and separate readiness
   slot. The route plane and envelope each carry their own arrival update into
   a merged per-source counter. Public payloads remain independent of metadata.
4. Target payload CTAs inspect only queue heads, using one warp ballot for at
   most sixteen heads. `RESERVE_ROUTES` creates all `(reader, source)` spans as one macro.
   For weighted return it also expands that source's reverse map and builds
   source-owned ReduceAdd plans before publishing route-ready.
   A copy head waits for route-ready and its own data-ready slot; later ready
   entries remain behind their queue head. A ready COPY exposes one CTA claim
   per local reader, so two queues/source bound arbitration without limiting
   destination HBM parallelism.
5. Target workers fan ready intervals from their source-indexed receive table
   into expert input. Each nonempty group release-adds an exact per-reader
   counter; metadata supplies the expected count, so the coordinator releases
   each ordinary reader independently. Ordered `END` instructions still
   contribute to the separate terminal mask that retires the invocation.
6. Dispatch Copy and combine ReduceAdd use one compile-time typed executor.
   Each static reduction CTA consumes its already-published plan, waits only
   the named reader-compute barriers, reduces one source-row shard, and
   publishes its coalesced return group with payload-coupled generation
   ordering. The retained schedule executes ReduceAdd after dispatch because
   overlapping both HBM-bound transforms regressed the measured total.
7. Once the source observes the required return-group generations, all
   PoolInst warps scatter and reduce destination partials into token-major
   output. Combine sends no router metadata because both endpoints retained
   the dispatch-derived compact maps.

This is a dependent gathered read, not a special EP transport. Route metadata
names which source slots each dynamic reader consumes; the shared sender set
and sequence determine when the read can retire.

## HBM ABI And Merged Signals

All remotely addressed buffers use same-order NVSHMEM symmetric allocations.
The hot path uses:

- source token slots plus target-local compact receive rows indexed by remote
  source; streaming self routes read token slots directly;
- a unique source-row list per target and route-to-compact index metadata;
- one envelope array plus bounded immutable queue storage indexed by source PE;
- one local receive record per `(reader, source)`;
- contiguous reader input/output regions;
- a source return inbox and saved origin-row array;
- one source-indexed return word and one merged metadata-parts counter per PE.

The receive record is a 32-byte fixed ABI. The public dispatch-ready value is
the same control generation PoolInst CTAs already consume; there is no second
`group_ready` allocation. Generic and weighted executors also allocate only
their required inverse-map and staging planes.

The source-indexed word names returned-data visibility. Streaming metadata
uses its parts counter. Activation
groups use separate per-group readiness slots so metadata and data can arrive
in either order. Each queue has one head/claim pair and an ordered `END`; all
readers sharing the sender set share one terminal mask.

The baseline permits one outstanding sequence per buffer set. More overlap is
obtained by overlapping work within that sequence, not by adding unbounded
mailboxes or fragmented allocation.

## Warp Roles

- warp 0 on rank zero: descriptor/signal publication, lane-parallel metadata
  polling, plan publication, reader release, and phase timestamps;
- all warps on payload CTAs: direct source PUTs, parallel queue-head
  arbitration, target-side HBM gather, and contiguous return PUTs;
- all eight warps: final source scatter.

For streaming dispatch, `group_limit=0` derives a producer group ceiling from
the assembled PoolInst CTA count and number of remote targets; actual groups
target roughly 512 KiB and at most 32 rows. An explicit value overrides only
that producer ceiling. Queue count is the compile-time constant two/source.
Rank zero stays on metadata, signal polling, and QP progress while other
PoolInst CTAs execute payload and queue work. Experimental external reducers
were removed from this ABI; a future alternative must be a separate compiled
PoolInst executor.

## Ordering Contract

1. The route map and contiguous descriptor-plus-queues envelope each use their
   own NBI put-with-signal. Each signal adds the source sequence delta to one
   merged metadata-parts counter. Reaching `sequence * part_count` proves that
   both individually protected byte ranges arrived, in any order, without
   a public-path metadata quiet.
2. Public payload groups use CTA-local NVSHMEM quiet before their named remote
   readiness write. A readiness slot
   never shares bytes with metadata, so early payload completion is safe.
3. Target queue copies start only after both source route-ready and the head
   instruction's data-ready slot are observed. Each reader shard release-adds
   completion; the final acquire-release CAS advances the queue head.
4. All dispatch RMAs are nonblocking and several PoolInst CTAs may keep
   independent QPs in flight. The current compact direct-put path uses the
   pinned block-cooperative quiet per issuing CTA.
5. Reader completion uses ordinary countdown barriers. The VDCores store warp
   waits for its writeback and decrements its barrier; a ReduceAdd plan polls
   only the subset named by its dependency mask before reading expert output.
6. Weighted return groups carry completion on their payload WQE; there is no
   destination-wide quiet or merged return phase in this executor.
7. Observing all required return-group generations makes every source inbox
   range consumable before scatter.
8. Queue claim release/acquire orders consecutive local instructions. The
   per-reader acquire/release counter chain publishes gathered rows before
   reader release; the acq-rel terminal mask independently retires all ordered
   queues.
9. Signal comparisons use `>=` and sequence-derived values. Signal words and
   the metadata-parts counter are monotonic and are not cleared between phases.

An empty pair still consumes its ordered `RESERVE_ROUTES` and `END`; only then
may the target publish return-complete without expert work. Invalid envelopes
cannot bypass queue validation.

There is no explicit `__threadfence_system` or `nvshmem_fence` in the protocol.
GPU-scoped atomics order local operators; put-with-signal and quiet order the
named NVSHMEM messages. These mechanisms are deliberately not treated as
interchangeable.

## Transport Boundary

The weighted PoolInst also has a compile-time NCCL GIN/GDAKI backend. It uses
the same queue and typed DynamicRead executors, one registered NCCL HBM window,
and a raw width-eight multi-SGE dispatch WQE; no runtime transport branch is
added. Data and metadata occupy disjoint NCCL context/QP partitions, and each
readiness update follows its precise payload on the same RC QP. See
`agents/knowledge/runtime/nccl-gin-poolinst.md` for its setup, version pin,
resource shape, and 4/8-PE measurements.

The installed Vista NVSHMEM 3.4.5 build exposes IBGDA environment mapping but
not application-created QP handles. The macro keeps target shards independent
and NBI so IBGDA can use its RC/DCI transport. The current production shape
uses 24 PoolInst CTAs/RC16 at two PEs and 32 CTAs with RC16/RC24 at four/eight
PEs. QP count is a shape/assembly parameter, not a protocol constant; sweep it
together with PoolInst CTA count at every 2/4/8-PE point.

For 64-bit signal operations in this installed IBGDA implementation, an RC
`ADD` uses one atomic WQE while `SET` uses two WQEs for its masked compare/swap;
DCI uses two for either. This makes the one-word metadata-parts `ADD` counter a
better first design than three or four separate `SET` generations, in addition
to reducing destination polling. Re-evaluate this if the NVSHMEM transport or
QP type changes rather than treating the choice as architecture-independent.

NVSHMEM device calls remain behind communication/pool-only function
boundaries. The nine-warp ordinary communication kernel alone carries the
lower per-kernel register cap; default and PoolInst assemblies preserve their
independent budgets in the same runtime object. Always inspect the linked
extension for WGMMA diagnostics, entry spills, and launch-resource growth.

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
