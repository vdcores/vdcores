# Pool-Slice Scheduler Models

The pool slice has two viable DynamicRead scheduling models. Both keep all
work inside the compiled PoolInst/VDCores launch and use the same immutable
metadata, payload-ready signals, Copy implementation, and ReduceAdd plans.

## Model 1: One Scheduler, Stateless Executors

One rank-local PoolInst CTA owns every active queue cursor, dependency state,
and retirement. It accepts source metadata and emits executable programs into
a bounded HBM ring. Persistent executor CTAs perform overflow SEND,
`DynamicRead<Copy>`, or `DynamicRead<ReduceAdd>` without scanning or retiring
metadata queues. Fixed source CTAs perform route expansion directly.

The selected implementation uses:

- one scheduler CTA per pool slice for the supported 2--32 PE range;
- shared-memory queue heads and in-flight masks for at most 64 ordered heads;
- a power-of-two HBM ring with monotonic producer tickets and per-slot
  generations;
- a native 64-bit `atomicAdd` consumer ticket, so metadata/data CTAs cannot pin
  ready jobs and any free executor can claim the next dynamic batch;
- one fixed executor CTA per source for direct route expansion; it waits on
  only that source's metadata generation and publishes the queue-zero
  completion bit to the scheduler without an HBM-ring handoff;
- per-source streaming: as soon as one source's route expansion is ready, the
  scheduler appends one dynamically sized DATA run per source queue and local
  reader, without waiting for the other sources;
- executor-side named waits for the payload generation of the first and every
  later DATA packet in a run, so transport and gather overlap without pinning
  executors behind unresolved route work;
- one release-published generation per ring slot and one release completion
  bit per executor job; the scheduler consumes both with GPU acquire loads;
- scheduler-only queue-head advancement, with END heads retired warp-wide and
  up to 32 retirement bits merged into one atomic OR;
- CTA-wide direct route expansion, so O(routes) work cannot block the
  scheduler or consume an HBM-ring handoff;
- the same stateless DynamicRead worker interface for Copy and ReduceAdd, with
  the operation decoded once outside row loops;
- a queue suffix containing every immutable ReduceAdd plan and then one STOP
  per executor; rank zero joins the executor pool after publishing the suffix,
  retaining all assembled CTAs for combine.

SEND groups are dynamically sized but flattened target-major. The common path
uses the predecessor's static CTA stripe: worker `i` submits group task `i`,
and only a group count larger than the direct worker count overflows into the
ring. Metadata-owning CTAs finish their own independent envelope publication
before their SEND task, while otherwise idle CTAs submit later groups. This
naturally paces the first WQE wave without a metadata-submission flag, global
gate, or extra synchronization. Remote consumers still use disjoint metadata
and payload generations, so either plane may arrive first.

In a centralized assembly pool rank zero performs its local metadata copy and
then immediately enters the scheduler; the other metadata CTAs post remote
envelopes concurrently. The direct target-major CTA stripe restores the
predecessor data-plane placement while leaving queue ownership centralized.

The ring is fixed storage but dynamically occupied. No epoch timeout, system
fence, helper kernel, or CUDA stream participates. The scheduler publishes a
source's DATA runs after that source's route generation is ready; the
retained executor waits on each exact per-message generation. The scheduler
hot loop then handles only completion, ordered END retirement, and reader/bar
release. END remains ordered per queue, but all independently ready queue
heads are tested and retired in parallel by the scheduler warp. It terminates
dispatch by appending the pre-generated ReduceAdd programs. Their ordinary
expert barriers are the data dependencies. ReduceAdd jobs precede a STOP
suffix, so every plan is claimed before any executor can terminate. Copy keeps
exclusive HBM priority because the suffix is appended only after dispatch
retirement.

The executor stages its 32-byte ring descriptor, current queue message,
instruction, and batch cursor in shared memory. SEND overflow reuses the same
CTA-uniform scalar fields. Copy and ReduceAdd decode the operation once outside
the row loops and use CTA-wide vectorized SIMT work.
An isolated Hopper TMA-copy experiment was rejected: remote source-address
handling made that form unsafe for the actual pool mapping, and its fallback
state increased local-memory spills. The selected copy path therefore remains
the zero-spill vectorized SIMT implementation.

The benefit is one queue-head scan per pool slice instead of one per worker,
no failed queue-head claims, one owner for retirement, and runtime-sized work
batches. The cost is an HBM job handoff and a single scheduler
issue/completion front end. If that front end becomes the limit, the safe
extension is a small compile-time number of source-sharded schedulers, each
with exclusive ownership of its queues.

### GB300 local-NVLink placement

The local backend specializes the same model for NVLink ordering and GB300
multicast. In a sufficiently large assembly, rank zero is scheduler-only and
the remaining CTAs have disjoint metadata-publish, route-expand, remote-SEND,
and local-pack roles. This avoids serializing route expansion behind SEND and
does not preserve the QP-pacing placement needed by the RDMA backend. Metadata
is published remote-first and self-last, while source payload is packed only
once into each destination delivery segment. Dynamic Copy commands move ready
segments to expert input while later segments are still in flight.

The scheduler preposts immutable per-source ReduceAdd commands immediately
after that source's ordered END heads retire. It does not wait for global
dispatch closure. The STOP suffix still follows every reduction ticket, but
the executor reads only the opcode for STOP, skips the descriptor/queue path,
retires its ring generation, and publishes its final dispatch generation.
These changes preserve the metadata/data separation and command interface
while reducing plan-ready and reduction-start latency.

The local multimem executor writes one BF16 partial per destination into that
destination's physical multicast backing. The source then executes
`multimem.red` over the multicast alias, using four independent 16-byte
vectors per loop iteration. The forwarding ReduceAdd executor remains the
portable command implementation and the NVSHMEM/IBGDA build retains its
original transport path.

In the cooperative ordinary-VDCores source-gather assembly, "rank zero" is
exactly one PoolInst scheduler CTA, not a second combine coordinator. Warp zero
owns dispatch scheduling while warp one concurrently waits the per-source
return generations and publishes `ScatterStart`. After both paths close, all
eight warps in that same CTA join the ordinary workers' `ScatterGeneration`
and logical executors' `DispatchGeneration` arrays and record final
completion. Workers publish only their own generations.

Worker placement is host-compiled into distinct CInst opcodes for metadata /
route, remote SEND, self-dispatch, executor-only, and dispatch-bypass roles.
The instruction arguments carry task, executor slot, and gather rank, so the
device never infers a role from `pool_rank` and the HBM ring still contains
only overflow work plus the ordered STOP suffix. The 4+ GPU direct-scatter
shape reserves enough explicit remote-SEND slots that its steady-state ring
normally contains only STOP tickets.

## Model 2: Stateful DynamicRead Workers

Each worker CTA owns one or more active DynamicReads and directly scans the
source metadata queues. Private reader cursors remove write contention but
replicate queue state and metadata reads. Shared cursors require reader-claim
atomics and a last-reader retirement protocol. Sharding a reader across CTAs
recovers parallelism but multiplies scans and completion joins.

This model removes the scheduler-to-executor handoff and can be attractive
when metadata is already reader-partitioned, the reader count matches the
desired CTA count, and every operation is large and balanced. In the current
protocol all readers consume the same two queues per source, there are only
eight local readers, and measured medium/large-token cases benefit from more
than eight PoolInst CTAs. It therefore has the less favorable scaling shape.

For `P` PEs, `R` readers, `K` CTA shards per reader, and `W` generic workers:

- centralized head scans are `ceil(2P / 32)` warp waves per poll;
- reader-owned scans are `R * K * ceil(2P / 32)` aggregate waves;
- the previous generic queue-first implementation is
  `W * ceil(2P / 32)` aggregate waves.

Model 2 remains a documented fallback design, not a runtime compatibility
mode in the selected hot implementation.

## Scale Boundary

Current ABI bounds are 32 PEs, two queues/source, eight readers, 32 DATA
groups/source, and 132 PoolInst CTAs. Queue-head, expected-completion, and
advance state fit in shared memory; bulk route maps stay in immutable HBM. A
single scheduler is instead limited by job rate. For `W` executors, average
task time `t_e`, a batch of `B` jobs, and scheduler batch cost `t_b`, it
remains non-limiting while approximately `W * t_b < B * t_e`.

Very small jobs, a highly contended consumer ticket, or the artificial maximum
of 8,192 reader jobs/invocation can violate that condition. Retain coarse
activation groups and shard the ring or scheduler only after internal timing
shows executor starvation. The consumer uses native `atomicAdd`; do not
replace it with a heavier general-purpose C++ atomic wrapper.
