# VDCores Communication Core

Communication is an instruction domain inside the persistent `dae2` kernel.
It is not an auxiliary kernel, host callback, or CUDA stream.

## VM Shape And Isolation

The optional NVSHMEM build has four compute warps, four memory warps, and one
communication warp (288 threads). Existing compute dispatch, allocation,
store, and load-warp ids and code paths are unchanged. A normal block consumes
independent compute, memory, and communication instruction streams.

`CommInst` is a 16-byte instruction containing four 16-bit fields and one
64-bit address. It has no allocator flags, consumes no shared-memory slots, and
never enters the memory/compute queues. Opcode zero terminates, so an untouched
communication stream is inert.

The unified pool hot path uses one isolated exception. If the first
communication instruction is `COMM_POOL_SLICE_EXCHANGE`, every thread in that
block enters the communication macro before the normal warp-role split. The
same nine resident warps temporarily become one coordinator, configurable pack
workers, and receive/return workers. Other blocks and all existing operators
continue through the ordinary VM paths unchanged.

The single `Launcher.launch_dae` call always carries all instruction streams.
There are no direct pool-kernel bindings. VDCores timing writes only the
existing per-block `g_events` profile space. NCCL and CUDA-event timing live
strictly under `benchmarks/`.

## Unified Batched Protocol

Each PE owns one logical pool slice and one communication-specialized block.
The source metadata is stable-grouped by destination PE and local reader.

1. An ordinary VDCores writer block copies every active source row exactly
   once into source-owned token slots and releases one barrier per TMA-sized
   write chunk.
2. The pool block builds one 64-byte descriptor per target. It embeds counts
   for up to eight target-local readers, including a valid zero-row batch.
3. Coordinator warp 0 publishes descriptors while pack warps wait only for the
   source chunk containing their next row and copy routed rows into one
   route-major symmetric delivery buffer. There is deliberately no block
   barrier between descriptor publication and pack/receive work.
4. Receive workers claim metadata-ready sources dynamically. Once that
   source's data phase arrives, a worker issues one contiguous NBI GET for each
   nonempty `(source, local_reader)` range. Multiple source batches remain in
   flight until one dispatch quiet.
5. The coordinator advances the shared group ticket and releases ordinary
   reader blocks only after all source batches complete.
6. After reader compute barriers retire, communication workers issue one
   contiguous NBI PUT for each nonempty return range. One quiet completes all
   return payloads.
7. Every slice publishes its return phase. Once the source observes all return
   phases, all nine warps scatter route-major inbox rows to saved origin rows.

This is a dependent gathered read, not a special EP transport. Route metadata
names which source slots each dynamic reader consumes; the shared sender set
and sequence determine when the read can retire.

## HBM ABI And Merged Signals

All remotely addressed buffers use same-order NVSHMEM symmetric allocations.
The hot path uses:

- source token slots and a route-major delivery buffer;
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
- warps `1..pack_warps`: route-major packing, each row waiting only for its
  writer-chunk barrier;
- remaining warps: dynamic source claims, contiguous dispatch GETs, and
  contiguous return PUTs;
- all nine warps: final source scatter.

`pack_warps=0` selects the measured policy. It uses four pack/four receive
warps for small payloads and for four or more PEs, and six pack/two receive
warps for a large two-PE buffer. An explicit config value overrides the policy
without changing the base VM.

## Ordering Contract

1. A descriptor is published with put-with-signal, so observing its metadata
   value makes the full cache-line record consumable. After lane-parallel
   publication, coordinator lane 0 fences before any later data-phase update;
   in the pinned IBGDA 3.4 implementation this fence walks the active RC/DCI
   QPs when more than one exists, so it also orders descriptors issued by the
   peer lanes before the merged signal word advances.
2. Pack-lane writes are system-fenced before the coordinator publishes the
   data phase.
3. A worker issues only after both descriptor validation and data readiness.
4. All dispatch RMAs are nonblocking; an adaptive quiet completes them before
   the group ticket and reader barriers are released. One remote peer uses
   lane-0 quiet; three or more PEs use the pinned NVSHMEM block-cooperative
   quiet so lanes share QP completion polling.
5. Reader completion barriers order ordinary VDCores stores before return
   workers read expert output.
6. All return RMAs are nonblocking; a quiet precedes return-phase publication.
7. Observing all return phases makes every source inbox range consumable before
   scatter.
8. Signal comparisons use `>=` and sequence-derived values. Signal words are
   monotonic and are not cleared between phases.

Local CUDA barriers order VDCores blocks; NVSHMEM signals and quiet establish
remote visibility. They are deliberately not treated as interchangeable.

## Transport Boundary

The installed Vista NVSHMEM 3.4.5 build exposes IBGDA environment mapping but
not application-created QP handles. The macro keeps source batches independent
and NBI so IBGDA can use its RC/DCI transport. Final 8-PE sweeps favored
`NVSHMEM_IBGDA_NUM_RC_PER_PE=1` and `NVSHMEM_IBGDA_RC_MAP_BY=cta`; two and four
RCs with warp mapping were slower. One communication block with four receive
workers also beat three- and five-pack-warp alternatives. A second pool SM is
therefore not justified by the measured issue rate and would add a new
cross-core completion protocol.

NVSHMEM device calls remain behind communication-only function boundaries.
The optional build preserves the WGMMA no-inline boundary and register cap;
always inspect the final linked extension for WGMMA diagnostics, entry spills,
and launch-resource growth.

## Entry Files

- VM/ISA: `include/dae/context.cuh`, `include/dae/dae2.cuh`,
  `include/dae/pipeline/commwarp.cuh`
- generic mailbox pool: `include/dae/memory_pool.cuh`,
  `python/dae/memory_pool.py`
- batched dynamic read: `include/dae/pool_slice_abi.cuh`,
  `include/dae/pool_slice.cuh`, `python/dae/pool_slice.py`
- encoding/launch: `python/dae/instructions.py`, `python/dae/launcher.py`
- correctness app: `app/python/memory_pool/pool_slice_dynamic_read.py`
- external comparison: `benchmarks/pool_slice_nccl_compare.py`
