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

The hot ABI assumes allocator-produced contiguous rows. `PoolSliceConfig` is
192 bytes and carries one row width rather than five equivalent strides;
`PoolSliceReceiveBatch` is 32 bytes because expert row indices are bounded to
`uint32_t`. Device entry does not repeat host configuration validation. The
public sender-set completion value aliases the internal `DispatchReady`
generation instead of occupying a second symmetric signal allocation.

## Metadata And Dependency

Source route rows are stable-grouped by target PE and target-local reader. For
the unified pool-gather path, the router produces three related arrays:

- `send_token_rows[target]` is the sorted unique source-token list for that
  target pool slice. The compiled weighted policy adds a second inverse-map
  plane; the generic policy does not allocate it;
- `send_token_counts[target]` is the length of that list;
- `send_rows[route]` maps each expert route to an index in its target's unique
  list. `send_origin_rows[route]` independently preserves the source return
  location.

Thus a token routed to several experts on one PE crosses the network once,
while a token routed to experts on two PEs crosses it once per target PE. One
runtime-sized metadata packet is sent from every source to every target,
including zero-row targets. Its live prefix contains the 64-byte publish
descriptor, used portions of both fixed queue spans, and that target's packed
route-to-compact words. One put-with-signal protects the complete packet.

Streaming dispatch appends immutable 32-byte instructions behind the envelope.
Queue zero contains `RESERVE_ROUTES`, dynamically many `COPY_ROWS`, then
`END`; queue one contains `COPY_ROWS` and `END`. Exactly two queues are
compiled into the protocol, so the scheduler has no queue-mode branch. A
`COPY_ROWS` instruction carries its exact compact interval and a readiness-slot
id, so the destination neither knows nor reconstructs the producer's group
count. It checks at most two heads per source in parallel and advances each
queue in order. Consuming every ordered `END` bit retires the dynamic read.

Metadata and activation data are distinct planes. The packed metadata packet
uses one NBI put-with-signal while activations use independently signaled data
groups. Activations move directly
from authoritative source token slots into the destination's source-indexed
`delivery_pool`; there is no source activation staging in streaming mode.
Each completed remote data group publishes a separate named readiness slot.
Keeping readiness outside immutable instructions permits data to arrive before
metadata without an overwrite race. The source-indexed signal names return
visibility; group payload visibility is not inferred from it.

## Macro Operation

The pool program carries one gathered-read PoolInst per statically assembled
PoolInst CTA. Host dispatch selects either `PoolSliceExchangeExecuteWarp` or
`PoolSliceWeightedExchangeExecuteWarp`; this compile-time choice removes the
return-mode branch from device code and sizes reverse-map, delivery-staging,
and return-inbox storage for only that policy. `dae2` executes one all-warp
macro. No helper stream or standalone CUDA kernel participates.

1. Ordinary VDCores writer instructions copy source activations once into
   source-owned token slots and release one barrier per TMA-sized row chunk.
2. PoolInst rank zero derives source-side group sizes from the actual unique
   token list, builds envelopes and ordered queues, and starts metadata
   publication. The current policy targets about 512 KiB and at most 32 rows
   per group, capped by the configured producer limit.
3. Other PoolInst CTAs issue direct row PUTs while metadata is in flight. A
   CTA waits only for writer chunks containing its rows. Public NVSHMEM groups
   quiet their own NBI writes before publishing readiness. Metadata and data
   remain independent and may arrive in either order.
4. Coordinator lanes accept source metadata independently and publish a local
   metadata-ready generation. Payload CTAs test all source queue heads in
   parallel, claim one ready head, and execute it. `RESERVE_ROUTES` atomically
   reserves every reader span as one macro and publishes route-ready.
5. `COPY_ROWS` becomes executable only when route-ready and its own data slot
   are visible. Each payload CTA claims one local reader shard and uses all
   eight compiled warps for that expert. Dense remote rows use a 256-thread
   contiguous HBM copy; arbitrary sparse maps remain warp-striped gathered
   row copies. Self rows read the authoritative token pool and acquire only
   their writer chunk. The head advances after all reader shards complete.
6. Per-reader release adds and the final acquire-release CAS publish the
   completed gathered writes before the queue head advances. Each
   `END` contributes to one GPU-scope acq-rel terminal mask; acquiring the full
   mask publishes all expert-input writes. Rank zero then releases ordinary
   VDCores reader barriers. No expected-group counter is part of completion.
7. Once descriptors are valid, a target immediately acknowledges return
   completion to zero-row sources. The route-major path waits each nonempty
   expert compute barrier and PUTs its contiguous `(reader, source)` return
   batch. The compiled weighted executor instead
   reduces local expert rows by route weight to one partial per compact token.
   Fine source-owned reduction shards are coalesced into at most four
   contiguous payload-coupled return groups, matching the default four RC
   QPs. The source consumes its local partial in place, waits only the named
   remote group generations, and sums at most one partial per pool slice into
   token-major output.
With multiple PoolInst CTAs, rank zero remains the metadata/signal coordinator
while other statically assembled PoolInst CTAs alternate outbound groups and
ready destination queue heads. Every PoolInst instruction is still one
VDCores macro operation; no helper kernel or stream is introduced.

## Scoped Ordering

- The runtime-sized descriptor/queue/route packet uses one NBI
  put-with-signal into a monotonic per-source transport generation. Observing
  that generation names exactly the protected metadata range without a
  public-path metadata quiet.
- Public payload groups use CTA-local NVSHMEM quiet followed by an NVSHMEM
  signal to their named readiness slot. No broad fence is used.
- Metadata-ready, route-ready, queue-claim, and reader barriers are GPU-scope
  release/acquire dependencies. COPY reader completions use release reductions
  and one acquire-release CAS; queue head advancement is release-scoped. The
  acq-rel terminal mask chains completed queues before reader release.
- Self-target signal words use GPU-scope release stores and acquire loads.
  Remote signal words use NVSHMEM signal operations and signal fetches.
- Dispatch and return payloads remain NBI. A direction-level quiet completes
  the named operations before its reader barrier or return signal advances.
- An empty relationship still executes its ordered route reservation and
  `END`. When all of that source's queue ends retire, the target may publish
  return completion immediately.
- Ordinary VDCores writer/reader dependencies reuse the normal countdown-barrier
  path. Every pending local edge has one logical producer, starts at one, and
  reaches zero through the same `atomicSub` used by existing store barriers.

The implementation has no explicit system fence. The scoped atomic wrappers
lower directly to PTX; bookkeeping-only sequence words use native CUDA
atomics without acquiring or publishing payload.

## Fixed Fast-Path Assumptions

- at most 32 PEs and at most eight local readers per slice;
- up to 32 statically assembled PoolInst blocks per PE;
- one outstanding sequence per buffer set;
- fixed-capacity, separately allocated symmetric buffers;
- contiguous rows of at least 1 KiB, 16-byte aligned, with no fragmented or
  unaligned tail path;
- source routes remain live and grouped until return scatter completes.

Larger logical reader counts are represented by additional slices, not by a
larger common descriptor.

## Compact Pool-Gather Mode

The unified top-k path transmits
one activation for every distinct `(token, target PE)` pair and publishes a
compact route map. Streaming transport reads the authoritative token slot
directly; no staged source path remains in the runtime.
Target PoolInst workers resolve that map locally and gather rows into
`(reader, source)` spans. Thus several local experts selecting one token share
the network transfer; fanout costs destination HBM bandwidth rather than
repeated RDMA messages. The compact list is source-row sorted.

The static runtime may assemble up to 32 PoolInst CTAs. Rank zero owns metadata
publication, signal polling, and final dependency release. Other ranks issue
outbound data and consume inbound queue heads; each ready COPY exposes one
claim per local reader. Up to two queues per source bound the arbitration
footprint without limiting HBM gather parallelism. Remote visibility remains
exclusively in NVSHMEM payload/quiet/signal operations; queue synchronization
is GPU-local.

The compiled `POOL_SLICE_WEIGHTED_EXCHANGE` path uses source-owned CTA
sharding, FP32 ILP4 destination accumulation, one partial token row per
destination slice, and token-major source scatter. One-contributor scatter is
an aligned copy; two-contributor scatter uses a zero-stack native BF16x2 add;
the arbitrary-fan-in fallback uses four register accumulators instead of a
per-lane 4x4 local-memory array. Local partials are consumed in place rather
than copied through `return_inbox`. The same 64-bit route word carries the
compact row and BF16 weight. At 128 tokens/PE, hidden 7168 BF16, eight
experts/PE, top-k 8, source-preloaded input, and identity expert output, the
final matched-allocation measurements for the staged predecessor were:

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

### Two-PE direct-source result (2026-07-27)

On one PE per GH200 node (`c642-012`/`c642-031`), CTA-mapped IBGDA with four
RC QPs and request batch four measured 0.132/0.216/0.291/0.445 ms at
32/64/128/256 tokens per PE for BF16 hidden 7168, top-k 8 clustered routing,
source-preloaded input, and in-place identity experts. The refreshed DeepEP
V1.2.1 BF16 control at 128 tokens was 0.404 ms, so PoolInst was 28% lower
latency there; the prior matched DeepEP sweep was
0.156/0.243/0.407/0.742 ms. A fully spread 32-token correctness run measured
0.164 ms. Process-to-process metadata/fabric jitter remains visible, so retain
phase medians and paired controls rather than only the best total.

The retained return uses sixteen fine reduction shards per source at two PEs,
coalesced into four contiguous put-with-signal groups by narrow GPU-scope
acquire-release counters. Reducing directly with only four CTAs was rejected
because it added about 30 us. Sixteen uncoalesced return messages were also
slower by about 14--16 us in the return dependency interval.

## Removed Runtime Alternatives

Dedicated-coordinator, phase-word, per-reader return, and external-reducer
flags were explored and then removed from the hot ABI. They either added
coordination without improving the critical path or lost to the inline
PoolInst reducer through atomic contention and SM/HBM competition. Historical
measurements remain in task logs; reintroduce an alternative only as a new
compiled PoolInst executor, not as a device-side mode branch.

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
pool-owned partial staging. A future compiled executor could consume a named
tile-ready dependency and start return transport from the first completed
expert tile. This requires a router-produced contiguous token-tile schedule;
it is not a runtime flag in the current operator.

## Transport And QPs

The Vista NVSHMEM 3.4.5 installation uses IBGDA for device communication but
does not expose the newer explicit application-QP API. In the default setup it
reported CTA mapping, one shared DCI QP, and two RC QPs. The implementation
therefore expresses concurrency as independent NBI source batches; RC/DCI
count and CTA/warp mapping are explored through NVSHMEM environment settings.

NVSHMEM 3.4's public device `quiet` is thread-scoped. The pool uses the pinned
block-cooperative internal quiet so QP completion polling is distributed
across the PoolInst CTA. The dependency is isolated in
`pool_slice_quiet_block`.

The reader-sharded compact profile uses up to 32 PoolInst CTAs. On the matched
two-PE BF16-7168 top-k-8 sweep it measured 0.278/0.471 ms at 128/256 tokens,
versus DeepEP V1 at 0.407/0.742 ms (32%/37% faster). At 32/64 tokens, dispatch
polling and return sharding prefer different CTA counts, so the runtime caps
dispatch arbiters while retaining all compiled CTAs for return. For dense
8-PE spread traffic, RC QP
counts 8/16/24/32/48/64 measured approximately
1.211/1.286/1.208/1.231/1.337/1.374 ms; more QPs did not improve saturated
payload traffic. QP count remains a shape/assembly parameter, not a protocol
constant.

Current DeepEP V2 was built externally with NCCL 2.30.7, but Vista's cross-node
Gin bootstrap timed out before a timing run. This is recorded as an environment
blocker, not a VDCores result. DeepEP V1.2.1 remains the runnable production
baseline until the V2 topology issue is resolved.

### Current DeepEP design boundary (2026-07-27)

DeepEP main now describes V2 as a Gin-backed `ElasticBuffer` runtime with
cached decode handles, asynchronous compute-stream integration, and analytical
SM/QP selection. Its published table uses 8K-token FP8-dispatch/BF16-combine
throughput, so those bandwidth numbers are not substituted for this project's
128-token BF16 latency comparison. The external V2 harness remains the proper
matched boundary once Vista Gin bootstrap works:
<https://github.com/deepseek-ai/DeepEP>.

The official experimental branches identify four directly relevant design
checks:

- Eager RDMA interleaves 4080-byte data regions and 16-byte readiness fields
  in the same RDMA write. On Hopper with sync-memops memory and non-relaxed MR
  ordering, observing a tile signal names that tile without the extra RTT of a
  write-then-atomic acknowledgement. It reports up to 20% latency reduction:
  <https://github.com/deepseek-ai/DeepEP/pull/437>. PoolInst should evaluate
  this as an isolated CX7/Hopper transport after the portable merged-counter
  baseline; its requirements must be checked explicitly.
- The zero-copy/TMA work fuses communication buffers and offloads layout
  movement to TMA while reducing SM occupancy:
  <https://github.com/deepseek-ai/DeepEP/pull/453>. This is the closest external
  analogue to direct source slots plus the proposed explicit-shared/TMA
  context experiment.
- Single-batch overlap begins combine sends from down-GEMM progress rather than
  waiting for the whole expert output:
  <https://github.com/deepseek-ai/DeepEP/pull/483>. The VDCores equivalent is a
  tile/expert release consumed by PoolInst, not a new CUDA stream.
- Layered low-latency dispatch avoids duplicate cross-orbit RDMA by forwarding
  through an NVLink peer: <https://github.com/deepseek-ai/DeepEP/pull/500>.
  Vista has one GH200 GPU per node, so this topology-specific path is not a
  valid local baseline; pool-slice placement and single-network-copy fanout
  are the applicable comparison instead.

## Measured Refinement Boundary

The tables below are historical predecessor boundaries. Do not attribute them
to the ordered direct-source queue path until it has a matched 2/4/8-PE run.

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

For direct streaming, `group_limit=0` derives a producer-side group ceiling from
PoolInst CTAs and remote targets. Actual groups target about 512 KiB and at
most 32 rows. This policy is not visible to the queue consumer.

Rejected variants include rescanning all routes per writer chunk, two-stage
route scans, BF16 accumulation, private return-inbox staging, dense chunk
signals, and per-reader early release. The retained implementation uses one
route traversal, direct self reads, named writer-chunk barriers, sparse
empty-phase elision, and one all-warp weighted scatter after the merged
nonempty return dependency closes.

VDCores measurements use only internal `g_events` timestamps. The dense NCCL
reference and CUDA-event timing remain external under `benchmarks/`.

## Current fast-path constraints (2026-07-27)

- Metadata queues are slot-major and publishers transfer only the used fixed
  prefix. Compact route metadata uses one 32-bit word containing a 16-bit
  compact row and one BF16 weight; it follows the live queue prefix in the
  same put-with-signal packet.
- Metadata and direct-source payloads are posted concurrently. Receiver queue
  heads execute in order and copy a group as soon as both its metadata and
  named payload generation are present. END is an ordered queue entry, not a
  timeout or a separate epoch-end kernel.
- Dispatch worker CTAs are bounded by actual groups and readers, but every
  statically assembled PoolInst CTA retains the uniform return lifecycle.
- Weighted return currently keeps fine 32-CTA sharding because each shard
  combines a remote PUT with a same-PE HBM copy. Coarsening those together is
  counterproductive; future remote batching must preserve independent,
  high-parallelism local copies.
- Two vector shards per source token are retained for weighted scatter. One
  shard halved active warps and nearly doubled the measured scatter tail.
- On two PEs, CTA-mapped RC counts 4/8 are sufficient for low and medium token
  counts, while 256-token traffic can benefit from 16. Counts above the
  useful concurrency amplify metadata variability rather than improving the
  invariant return tail; QP count remains a launch-time shape choice.
- A four-submitter return A/B added one GPU-scope release counter after local
  staging and reduced 16 remote puts to four. Although its helper spill frame
  fell from 232 to 188 bytes, it measured 0.277/0.480 ms at 128/256 tokens
  versus 0.278/0.476 ms controls. Waiting for all staging erased early-put
  overlap, so the counter and batching path were removed.
- Matched production-shape DeepEP V1 comparisons on two PEs are currently:
  32 tokens 0.181 vs 0.156 ms, 64 tokens 0.212 vs 0.243 ms, 128 tokens 0.268
  vs 0.407 ms, and 256 tokens 0.414 vs 0.742 ms. PoolInst uses QP4 through
  128 tokens and QP16 at 256. The remaining 32-token deficit is dominated by
  ordered-queue/gather control and the fixed weighted-scatter tail.
- That one-group specialization was subsequently rejected in a paired run:
  static queue ownership measured 0.270 ms versus 0.184 ms for generic
  warp-ballot arbitration. It serialized eight reader shards into waves on
  each active COPY queue. The generic scheduler and its cross-CTA claim state
  remain the only implementation; there is no small-token opcode or runtime
  compatibility branch.

## Contiguous-run transport update (2026-07-28)

The device sender now gives each warp a contiguous compact-row interval and
merges consecutive source rows, bounded by the named writer-dependency chunk,
into one NVSHMEM PUT. Random routes naturally remain one-row runs. This keeps
eight sender warps active while reducing IBGDA request count for the common
sorted-token case. Two-PE clustered samples at 32/128/256 tokens were
0.141/0.250/0.377 ms (best 128 sample 0.238 ms), versus the existing DeepEP
controls of 0.156/0.407/0.742 ms.

Cross-CTA return profiling showed an approximately 55 us gap from the last
fine reducer completion to the first grouped return PUT. Replacing the
acq-rel counter with per-CTA release generations did not reduce it; ordering
the staged HBM writes was still the cost. Giving each of four groups to one
CTA removed the dependency but grew the return tail to about 0.202 ms by
discarding multi-SM reduction parallelism. Both variants were removed. Future
work should make expert/reducer output transport-ready in its final group tile
rather than adding another synchronization scheme around partial staging.

## Four-PE synchronization boundary (2026-07-28)

For BF16 hidden-7168, eight experts/PE, top-8 clustered routing and 32
PoolInst CTAs, CTA-mapped QP16 measured 0.206/0.273 ms at 32/128 tokens. QP8
was the stable 256-token choice at 0.550 ms. Recorded DeepEP controls are
0.192/0.551/1.025 ms, respectively. QP count is therefore shape-specific:
more connections remove small-message metadata head-of-line delay, but can
make large-message metadata closure less stable.

The approximately 50 us reducer-to-return-PUT transition persisted after
replacing four grouped returns with eight direct reducer-shard PUTs. That A/B
also doubled remote return messages and introduced a 96-byte helper spill
frame, so it was removed. Likewise, batching two or eight local readers under
one queue claim reduced queue-scoped acq-rel operations but lost the HBM copy
parallelism required by dense top-k gather. Keep reader-level CTA claims and
the grouped return counter; neither synchronization is the isolated cause of
the remaining transition delay.

## Compact-metadata and 8-PE boundary (2026-07-28)

Only PoolInst rank zero publishes the one metadata envelope per destination;
payload CTAs enter the direct send/gather loop concurrently. A reverted
CTA-sharding experiment briefly left all CTAs publishing duplicate envelopes.
That raised the 8-PE/128-token result from 0.331--0.365 ms to 0.430 ms and was
removed. `tests/test_pool_slice.py` now checks for the single rank-zero call.

Route metadata was reduced from eight bytes to four bytes without changing
reduction precision: the wire word is `compact_row16 | bf16_weight << 16`, and
the destination expands it into the existing 64-bit combine record. Python
rejects token capacities above 65,536, and peer packet strides are rounded to
16 bytes. At 8 PEs this moved 32/128-token samples from 0.227/0.365 ms to
0.195/0.344 ms; 256 tokens stayed bandwidth-bound at 0.664 ms.

The live packet length must also be rounded to 16 bytes, not only its fixed
peer stride. The same-PE publisher copies `uint4` vectors and otherwise drops
the final one to three 32-bit routes when a target owns an odd/non-vector route
count. A 13-token, top-2, two-reader test exposed this because the common power-
of-two benchmark shapes had hidden it. The padding is inside the reserved peer
packet and is ignored by the route parser; weighted and unweighted irregular
tests now cover the boundary.

Final exact-BF16 clustered samples and the recorded DeepEP V1 controls are:

| PEs | tokens/PE | CTA-mapped RC QPs | PoolInst | DeepEP V1 | advantage |
|---:|---:|---:|---:|---:|---:|
| 2 | 32 | 4 | 0.126 ms | 0.156 ms | 19% |
| 2 | 128 | 4 | 0.220 ms | 0.407 ms | 46% |
| 2 | 256 | 8 | 0.367 ms | 0.742 ms | 51% |
| 4 | 32 | 32 | 0.150 ms | 0.192 ms | 22% |
| 4 | 128 | 16 | 0.347 ms | 0.551 ms | 37% |
| 4 | 256 | 8 | 0.490 ms | 1.025 ms | 52% |
| 8 | 32 | 16 | 0.195 ms | 0.272 ms | 28% |
| 8 | 128 | 24 | 0.344 ms | 1.040 ms | 67% |
| 8 | 256 | 24 | 0.664 ms | 2.041 ms | 67% |

QP choice is shape-specific. For example, 4-PE/32-token QP32 measured
0.150 ms, but the same QP count made 128-token metadata closure unstable and
measured 0.489 ms. These are internal `g_events` medians with source-preloaded
input and in-place identity experts; the DeepEP values are separately recorded
production-baseline runs.

Two return groups per source halved 8-PE return RMAs from 28 to 14 but did not
improve the measured return interval: about 111 us versus 108 us at 128 tokens
and 198 us versus 196 us at 256. Four groups remain compiled. CTA-sharded
metadata, a source-wide return group, reduced PoolInst CTA counts, CPU-affinity
pinning, and one-in-flight host-QP pacing were also rejected after neutral or
negative end-to-end A/Bs.

Two further overlap A/Bs were rejected. A CTA-collective PUT for an already-
ready contiguous dispatch group reduced submissions but removed useful
warp/QP concurrency (8-PE/128-token total 0.356 to 0.371 ms). Token-scoped
return-generation polling attempted to scatter early groups before global
closure, but duplicated dependency polling across scatter warps and left the
return tail unchanged at about 0.15 ms. The existing warp-run sender and one
merged return-closure warp remain simpler and faster. RC32 versus RC24 samples
also tracked allocation-local metadata drift rather than a stable transport
gain, so RC24 remains the eight-PE 128-token starting point.
