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

Source route rows are stable-grouped by target PE and target-local reader. One
64-byte `PoolSlicePublishBatch` is sent from every source to every target,
including zero-row targets. The descriptor contains the source route span and
up to eight reader counts, so the target reconstructs all contiguous ranges
without fetching route lists or offset tables.

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

The pool block's first instruction is `PoolSliceExchange(config,
write_barrier, dispatch_barrier_base, compute_barrier_base)`. `dae2` detects it
uniformly before the normal warp-role split and executes one all-warp macro:

1. Build and publish descriptors while the ordinary writer block copies source
   activations once into token slots.
2. Pack warps traverse route rows once. Before copying a row, they wait only on
   the writer barrier for the TMA-sized chunk containing that source row.
3. Coordinator lanes scan all source signal words in parallel. Receive workers
   dynamically claim metadata-ready sources and wait independently for their
   data phase.
4. Each worker issues one contiguous NBI GET per nonempty
   `(source, local_reader)` range. One quiet after all issuance completes all
   dispatch batches.
5. The coordinator records receiver tails, publishes the group ticket, and
   releases ordinary reader blocks.
6. After reader compute barriers reach zero, workers issue contiguous return
   NBI PUTs by source. One quiet completes the direction.
7. The coordinator publishes and waits for merged return phases. All nine
   warps then scatter route-major inbox rows to source origin rows.

The HBM protocol therefore overlaps descriptor delivery, source writes,
route-major packing, signal scanning, and remote payload issue. There is no
block barrier between descriptor publication and pack/receive work. It keeps
many source batches in flight rather than quieting after each source or reader.

## Fixed Fast-Path Assumptions

- at most 32 PEs and at most eight local readers per slice;
- one pool communication block per PE initially;
- one outstanding sequence per buffer set;
- fixed-capacity, separately allocated symmetric buffers;
- contiguous rows of at least 1 KiB, 16-byte aligned, with no fragmented or
  unaligned tail path;
- source routes remain live and grouped until return scatter completes.

Larger logical reader counts are represented by additional slices, not by a
larger common descriptor.

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

Measured 128-token sweeps favored one RC per PE with CTA mapping. At eight PEs,
one/two/four RC profiles were approximately 0.412/0.467/0.519 ms. The single
pool SM already issued all seven remote source batches before quiet; more QPs
increased control and straggler cost. Pack-warp sweeps at the same shape were
0.517/0.412/0.427 ms for three/four/five pack warps. This does not justify a
second pool SM: it would add cross-core ownership and completion coordination
without addressing the measured bottleneck.

## Measured Refinement Boundary

The predecessor issued one GET per routed row. At 128 tokens/PE and 4096 BF16
elements/row on two GH200 PEs it took 0.903--1.098 ms versus 0.474--0.476 ms
for the dense NCCL ring reference.

The unified route-major macro replaces tens of row RMAs with one batch per
nonempty `(source, reader)`. Representative 50-sample results for 4096 BF16
elements/row and one reader/PE are:

| PEs | tokens/PE | pool | NCCL ring | pool/ring |
|---:|---:|---:|---:|---:|
| 2 | 8 | 0.113 ms | 0.289 ms | 0.39x |
| 2 | 32 | 0.144 ms | 0.552 ms | 0.26x |
| 2 | 128 | 0.356 ms | 0.486 ms | 0.73x |
| 4 | 8 | 0.133 ms | 0.747 ms | 0.18x |
| 4 | 32 | 0.173--0.213 ms | 1.095 ms | 0.16--0.19x |
| 4 | 128 | 0.359--0.377 ms | 1.336 ms | 0.27--0.28x |
| 8 | 8 | 0.165 ms | 1.669 ms | 0.10x |
| 8 | 32 | 0.183 ms | 2.385 ms | 0.08x |
| 8 | 128 | 0.412--0.438 ms | 4.193 ms | 0.10x |

The ranges are repeated pool runs; NCCL columns are the dense two-all-reduce
ring reference, not a production sparse all-to-all. The pool transfers only
routed rows, while the reference materializes dense expert-major dispatch and
token-major return tensors.

At eight PEs and 128 tokens, a representative phase split was about 0.112 ms
to packed-data publication, 0.111 ms to all metadata, 0.236 ms to dispatch
payload completion, 0.337 ms to reader completion, 0.375 ms to return-payload
completion, 0.401 ms to all return phases, and 0.428 ms through scatter.

The default `pack_warps=0` policy selects four pack/four receive warps for
small payloads and for four or more PEs. For a large two-PE buffer it selects
six pack/two receive warps; this reduced 128-token time from about 0.412 to
0.356 ms without hurting small-token receive concurrency.

Rejected variants include rescanning all routes per writer chunk, seven pack
warps at two PEs, two-stage route scans, fused return signaling, and
per-slice early scatter. The last saved only 4--5 microseconds at eight PEs but
added polling contention at two and four PEs. The retained implementation uses
one route traversal, waits only the named writer-chunk barrier, and performs
one all-warp scatter after the merged return dependency closes.

VDCores measurements use only internal `g_events` timestamps. The dense NCCL
reference and CUDA-event timing remain external under `benchmarks/`.
