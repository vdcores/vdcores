# Pool-Slice Dynamic Read

This is the unified pool-owned gathered-read protocol. Authoritative entry
points are `include/dae/pool_slice_abi.cuh`, `include/dae/pool_slice.cuh`, and
`python/dae/pool_slice.py`.

## Logical Ownership

- The logical pool is split into one slice per PE.
- Source token slots and return origins are homed on the source slice.
- Dynamic readers and their contiguous input/output regions are homed on the
  consuming slice.
- Producers write activation rows and route metadata; the pool PE interprets
  that metadata and owns all inter-PE movement.
- Reader blocks only execute ordinary VDCores memory/compute operations behind
  pool-released barriers.

The optimized implementation has one outstanding monotonic sequence per fixed
buffer set. It does not retain the old expert-specific transport or its
compatibility opcodes.

## Metadata And Dependency

Source route rows are stable-grouped by target PE and target-local reader. For
the unified pool-gather path, the router produces three related arrays:

- `send_token_rows[target]` is the sorted unique source-token list for that
  target pool slice;
- `send_token_counts[target]` is the length of that list;
- `send_rows[route]` maps each expert route to an index in its target's unique
  list. `send_origin_rows[route]` independently preserves the source return
  location.

Thus a token routed to several experts on one PE crosses the network once,
while a token routed to experts on two PEs crosses it once per target PE. One
64-byte `PoolSlicePublishBatch` is sent from every source to every target,
including zero-row targets. It contains the target's route span, compact-row
count, and up to eight reader counts. Only that target's route-to-compact map
is delivered beside it; the old full-table replication is not retained.

`delivery_pool` has two fixed-capacity halves. The first is the receive table,
indexed by source PE and compact row. The second is private source staging,
indexed by target PE and compact row. This deliberately spends contiguous HBM
space to avoid allocation, fragmentation, and per-row network messages.

One signal word per source PE carries three increasing values for each
sequence:

1. metadata descriptor visible;
2. packed delivery rows visible;
3. returned rows visible.

A target accepts all source descriptors as its sender-set completion. Empty
sources retire at metadata; nonempty sources also wait for data. All dynamic
reads on a slice share this sender set and one group-ready ticket, so no
reader-specific end message or epoch-end record exists.

## Macro Operation

The pool program carries one `PoolSliceExchange(config, write_barrier,
dispatch_barrier_base, compute_barrier_base)` PoolInst per statically assembled
PoolInst CTA. Host dispatch selects `PoolSliceExchangeExecuteWarp`; `dae2`
enters it uniformly before the ordinary VM and executes one all-warp macro.
No helper stream or standalone CUDA kernel participates.

1. Ordinary VDCores writer instructions copy source activations once into
   source-owned token slots and release one barrier per TMA-sized row chunk.
2. PoolInst rank zero builds target descriptors and starts metadata delivery.
   Route-map bytes and descriptors move while worker warps wait only for the
   source chunks they actually pack.
3. Warps 1--7 across payload PoolInst CTAs claim `(target, compact shard)`
   tasks. Each remote target packs sorted unique rows into private contiguous
   staging and receives one NBI PUT per nonempty shard. The self target is not
   packed: its gathered reads translate the compact index through
   `send_token_rows` and read the source-owned token slot directly.
4. Every issuing PoolInst CTA performs an NVSHMEM quiet for its own work and
   publishes a GPU-local generation. Rank zero waits for those named
   generations and then publishes the data phase to every nonempty pool slice.
   A valid zero-row descriptor is initially signaled at the data-complete value,
   so it needs no second phase update.
5. Coordinator lanes scan all source phase words in parallel. Once the sender
   set closes, target workers interpret the route-to-compact maps and copy rows
   from `(source, compact row)` into deterministic expert input ranges. A
   bounded self-source gather tranche may overlap the remote phase wait on
   non-coordinator CTAs.
6. Rank zero joins the gather generations and releases ordinary VDCores reader
   barriers. `READER_PIPELINE` instead uses per-reader shard counters so an
   expert may start as soon as its own complete sender set is gathered.
7. Once descriptors are valid, a target immediately acknowledges return
   completion to zero-row sources. The route-major path waits each nonempty
   expert compute barrier and PUTs its
   contiguous `(reader, source)` return batch. `WEIGHTED_RETURN` instead
   reduces local expert rows by route weight to one partial per compact token,
   sends contiguous token batches, and sums at most one partial per pool slice
   into token-major source output.
8. The optional route-major pipelined return attaches a signal to each remote
   reader batch and lets the source scatter a batch as soon as it arrives. It
   removes the merged return phase and local inbox copy, but is selected only
   where the exposed work exceeds its extra signal cost.

At the current 128-token fast path, `pack_warps=0` selects all seven worker
warps when the source token table is at least 512 KiB; smaller tables use four.
With multiple PoolInst CTAs, rank zero can be dedicated to metadata, signal
polling, and QP progress while the other CTAs use all seven worker warps for
payload and gathered-read tasks.

## Scoped Ordering

- Descriptor metadata uses NVSHMEM put-with-signal. The compact route map is
  currently a separate preceding NBI write; this pair is being merged into one
  contiguous metadata message so one signal names the complete dependency and
  no general fence is needed.
- A worker CTA's NBI payloads are completed by NVSHMEM quiet. It then publishes
  a GPU-scope generation to rank zero, which emits the remote data phase. The
  generation coordinates PoolInst CTAs; it does not claim network visibility.
- Self-target phase words use GPU-scope release stores and acquire loads.
  Remote phase words use NVSHMEM signal operations and signal fetches.
- Dispatch and return payloads remain NBI. A direction-level quiet completes
  the named operations before its reader barrier or return signal advances.
- A valid empty descriptor advances directly to the data value. After a target
  has closed the complete incoming data set, it may publish the return value
  immediately for an empty `(source,target)` relationship. This higher value
  also safely dominates earlier values on the shared phase word because the
  target has already published its own dispatch data phase before that wait.
- Ordinary VDCores writer/reader barriers use the separate GPU-scope
  `pool_signal` release/acquire path.

The implementation has no explicit system fence. The scoped atomic wrappers
lower directly to PTX; bookkeeping-only sequence words use native CUDA
atomics without acquiring or publishing payload.

## Fixed Fast-Path Assumptions

- at most 32 PEs and at most eight local readers per slice;
- up to 32 statically assembled PoolInst blocks and, optionally, up to 32
  ordinary external reducer blocks per PE;
- one outstanding sequence per buffer set;
- fixed-capacity, separately allocated symmetric buffers;
- contiguous rows of at least 1 KiB, 16-byte aligned, with no fragmented or
  unaligned tail path;
- source routes remain live and grouped until return scatter completes.

Larger logical reader counts are represented by additional slices, not by a
larger common descriptor.

## Compact Pool-Gather Mode

`POOL_SLICE_DISPATCH_POOL_GATHER` is the unified top-k path. A source packs one
activation for every distinct `(token, target PE)` pair and publishes a compact
route map. Target PoolInst workers resolve that map locally and gather rows into
deterministic `(reader, source)` slots. Thus several local experts selecting one
token share the network copy; fanout costs HBM bandwidth rather than repeated
RDMA messages. The compact list is source-row sorted, so workers wait on writer
chunks monotonically and issue contiguous shard PUTs.

The static runtime may assemble up to 32 PoolInst CTAs. Rank zero owns phase
publication, signal polling, and merged dependency release. Payload work is
sharded over the other ranks and their execute warps. Rank zero must not run
HBM gather during an IBGDA wait: measurements showed that even a small gather
on its CTA delays signal progress/observation. The retained two-PE overlap
starts one self-source route shard on nonzero ranks after all local replication
generations close; remote gather and remaining local shards start after remote
data closure. Both gates are GPU-local release/acquire generations. Remote
visibility remains exclusively in NVSHMEM payload/quiet/signal operations.

The production `WEIGHTED_RETURN` path uses flat cross-source CTA sharding,
FP32 ILP4 destination accumulation, one partial token row per destination
slice, and FP32 ILP4 source scatter. The same 64-bit route word carries the
compact row and BF16 weight. At 128 tokens/PE, hidden 7168 BF16, eight
experts/PE, top-k 8, source-preloaded input, and identity expert output, the
final matched-allocation measurements were:

| PEs | PoolInst CTAs | RC QPs | pool dispatch | pool total | DeepEP V1 total | pool advantage |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 24 | 16 | 0.168 ms | 0.350 ms | 0.408 ms | 14% |
| 4 | 32 | 16 | 0.236 ms | 0.389 ms | 0.551 ms | 29% |
| 8 | 32 | 24 | 0.245 ms | 0.417 ms | 1.040 ms | 60% |

These are rank-maximum medians from 50 measured iterations after 15 warmups.
Fabric load produced substantial run-to-run variance: earlier accepted
samples were 0.305/0.383/0.400 ms at 2/4/8 PEs. Use paired runs from one
allocation for claims rather than mixing isolated best samples.

Pool-slice and expert placement materially change the traffic shape. The
four benchmark placements are: source-local, one forced remote slice,
balanced one-slice clustering, and full top-k spread. A representative 8-PE
run after phase elision and direct-self reads measured:

| placement | pool total | DeepEP V1 total | implication |
|---|---:|---:|---|
| source-local | 0.489 ms | 0.160 ms | pool is limited by empty sender-set closure |
| one remote slice | 0.472 ms | 0.739 ms | compact single-destination transport wins |
| balanced clustered | 0.417 ms | 1.040 ms | production locality target |
| fully spread | 1.209 ms | 1.040 ms | one activation per destination removes compaction gain |

The fully spread case sends about 12.85 MB/PE in each direction versus
1.61 MB/PE for balanced clustering. Placement is therefore part of the
system design: expert ownership should preserve token-to-slice clustering
when model quality/load balance permit it. Source-local exposes a separate
control-plane gap; a root-aggregated sender-set generation could replace
quadratic empty descriptors with active metadata plus O(P) closure signals.

## Reducer Placement Alternatives

Weighted reduction has three compile-time-selected VDCores execution plans;
all remain one `Launcher` program and use no helper kernel or CUDA stream.

- Inline/default: PoolInst worker warps use the reverse `combine_rows` map,
  accumulate BF16 expert rows in FP32 registers, stage BF16 token partials,
  and immediately post batched NVSHMEM returns.
- `EXTERNAL_WEIGHTED_REDUCER`: one ordinary compute+memory VDCores block per
  expert waits that expert's dispatch/compute signal plus a concurrently
  produced zero-buffer signal. It starts independently, uses native
  `atomicAdd(__nv_bfloat162*)` into pool-owned token staging, and releases one
  reducer completion signal. This maximizes expert-level overlap.
- `EXTERNAL_TOKEN_REDUCER`: 1--32 ordinary compute+memory blocks wait the local
  expert set, own disjoint compact-token rows, and perform non-atomic FP32
  accumulation/BF16 stores. `PoolSliceConfig.reducer_count` tells PoolInst how
  many contiguous completion signals to acquire before batched return.

The dependency chain is deliberately scoped: pool gather releases a GPU-scope
reader signal; optional reader compute releases an expert-output signal;
ordinary reducers acquire those signals and release their output signals; and
PoolInst acquires only that reducer range. Cross-PE visibility remains an
NVSHMEM payload/quiet/signal fact. No step uses `__threadfence_system`.

At the production 128x7168/top-8 shape, expert-atomic reduction began about
16 microseconds apart across experts but spent roughly 0.95 ms in contended
BF16 atomics. It measured 1.46 ms at two PEs and 2.09 ms at four PEs. The
token-sharded mode reduced its compute span to about 0.32--0.33 ms with 32
reducer blocks, but competed with PoolInst for HBM/SM resources: its best
measured totals were about 0.80 ms at two PEs (8 PoolInst + 32 reducers) and
1.03 ms at four PEs. These alternatives validate generic VDCores composition,
but the inline reducer remains the performance default.

The optional per-reader pipeline did not move the slowest RMS completion at
the measured 8-PE shape and added coordination. A dense aligned return-chunk
pipeline was also neutral (1.213 ms versus 1.212 ms in the spread case):
balanced chunks completed too closely together to amortize extra signals.
Both remain off. Return reduction stays inline because ordinary expert-atomic
reducers were dominated by contended BF16 atomics and token-sharded reducer
SMs competed with PoolInst for HBM and residency.

## TMA And Expert-Epilogue Placement

VDCores already exposes `OP_ALLOC_WB_TMA_REDUCE_ADD_2D/3D` in
`include/dae/pipeline/stwarp.cuh`. A standalone PoolInst TMA reducer is not a
win for arbitrary route maps: it would first load expert output from HBM into
shared memory, then TMA-reduce it back to HBM, while route weights still need
an irregular gather. The retained inline register reducer avoids that extra
pass.

The useful TMA design point is a fused expert-MLP epilogue. Expert output tiles
are already in shared memory, so an ordinary compute+memory VDCores block can
apply route weights and TMA-reduce a contiguous token tile directly into
pool-owned partial staging. It releases the same reducer signal consumed by
PoolInst, allowing return transport to start from the first completed expert
tile. This requires a router-produced contiguous token-tile schedule; it is
the next implementation target for real expert compute, not for the identity
transport microbenchmark.

## Transport And QPs

The Vista NVSHMEM 3.4.5 installation uses IBGDA for device communication but
does not expose the newer explicit application-QP API. In the default setup it
reported CTA mapping, one shared DCI QP, and two RC QPs. The implementation
therefore expresses concurrency as independent NBI source batches; RC/DCI
count and CTA/warp mapping are explored through NVSHMEM environment settings.

NVSHMEM 3.4's public device `quiet` is thread-scoped. The pool uses the pinned
block-cooperative internal quiet at three or more PEs so QP completion polling
is distributed across lanes; at one remote peer, lane 0 is faster. The
dependency is isolated in `pool_slice_quiet_block`.

The production compact profile uses 24 PoolInst CTAs at two PEs and 32 at four
and eight PEs. Two-PE dispatch over-sharded at 32 CTAs, while 8/16 CTAs were
under-parallelized at larger PE counts. For dense 8-PE spread traffic, RC QP
counts 8/16/24/32/48/64 measured approximately
1.211/1.286/1.208/1.231/1.337/1.374 ms; more QPs did not improve saturated
payload traffic. QP count remains a shape/assembly parameter, not a protocol
constant.

Current DeepEP V2 was built externally with NCCL 2.30.7, but Vista's cross-node
Gin bootstrap timed out before a timing run. This is recorded as an environment
blocker, not a VDCores result. DeepEP V1.2.1 remains the runnable production
baseline until the V2 topology issue is resolved.

## Measured Refinement Boundary

The predecessor issued one GET per routed row. At 128 tokens/PE and 4096 BF16
elements/row on two GH200 PEs it took 0.903--1.098 ms versus 0.474--0.476 ms
for the dense NCCL ring reference.

The unified route-major macro replaces tens of row RMAs with one batch per
nonempty `(source, reader)`. Representative 50-sample results for 4096 BF16
elements/row and one reader/PE are:

| PEs | tokens/PE | pool | NCCL ring | pool/ring |
|---:|---:|---:|---:|---:|
| 2 | 8 | 0.113--0.116 ms | 0.289--0.291 ms | 0.39--0.40x |
| 2 | 32 | 0.144--0.151 ms | 0.530--0.552 ms | 0.26--0.29x |
| 2 | 128 | 0.356--0.376 ms | 0.469--0.486 ms | 0.73--0.80x |
| 4 | 8 | 0.117--0.133 ms | 0.747--0.762 ms | 0.15--0.18x |
| 4 | 32 | 0.173--0.213 ms | 1.067--1.095 ms | 0.16--0.20x |
| 4 | 128 | 0.359--0.408 ms | 1.336--1.378 ms | 0.27--0.30x |
| 8 | 8 | 0.165--0.179 ms | 1.669--1.690 ms | 0.10--0.11x |
| 8 | 32 | 0.183--0.219 ms | 2.367--2.385 ms | 0.08--0.09x |
| 8 | 128 | 0.412--0.464 ms | 4.167--4.193 ms | 0.10--0.11x |

The ranges are repeated pool runs; NCCL columns are the dense two-all-reduce
ring reference, not a production sparse all-to-all. The pool transfers only
routed rows, while the reference materializes dense expert-major dispatch and
token-major return tensors.

The scoped-atomic refactor's 30-sample 2/4-PE points fall inside these ranges.
A longer 4-PE/128-token transport A/B measured about 0.408 ms for one RC QP
per peer with CTA mapping and 0.446 ms for four RC QPs with warp mapping.
Forcing one IBGDA request per submission batch was 0.401 ms, not a material
change, so the retained profile is RC1/CTA with the NVSHMEM default batch size.
At eight PEs the controlled 80-sample result was 0.425 ms with default
batching versus 0.454 ms with one request per batch, confirming that choice.

At eight PEs and 128 tokens, a representative phase split was about 0.112 ms
to packed-data publication, 0.111 ms to all metadata, 0.236 ms to dispatch
payload completion, 0.337 ms to reader completion, 0.375 ms to return-payload
completion, 0.401 ms to all return phases, and 0.428 ms through scatter.

The default `pack_warps=0` policy selects four worker shards below a 512 KiB
source token table and all seven worker warps at or above it. Multi-CTA
parallelism then supplies receive/return concurrency; the value is no longer a
static split between PUT and GET warps.

Rejected variants include rescanning all routes per writer chunk, two-stage
route scans, BF16 accumulation, private return-inbox staging, dense chunk
signals, and per-reader early release. The retained implementation uses one
route traversal, direct self reads, remote-only contiguous staging, named
writer-chunk barriers, sparse empty-phase elision, and one all-warp weighted
scatter after the merged nonempty return dependency closes.

VDCores measurements use only internal `g_events` timestamps. The dense NCCL
reference and CUDA-event timing remain external under `benchmarks/`.
