# Pool-Slice Dynamic Read

This is the pool-owned replacement for the sender-driven expert dispatch
protocol. The authoritative implementation entry points are intended to be
`include/dae/pool_slice_abi.cuh`, `include/dae/pool_slice.cuh`, and
`python/dae/pool_slice.py`.

## Ownership

- A logical pool is horizontally split into one slice per PE initially.
- Token slots are homed on the source PE's slice.
- Dynamic readers are homed on the consuming PE's slice.
- Only a pool communication warp executes writes, metadata delivery, gathered
  reads, dependency resolution, and returns.
- Producers only publish a batch into the target slice's per-sender mailbox;
  consumers only wait on VDCores barriers and consume pool-owned HBM.

The first implementation has one outstanding sequence per buffer set. A
depth-one mailbox is therefore a valid SPSC queue. Sequence values increase
monotonically when the mailbox is reused.

## Slice Completion

Every source publishes exactly one route batch to every target slice,
including a zero-route batch. Queue publication is a transport doorbell, not a
reader completion signal. After a target slice has consumed all source
batches, it advances one local group-ready ticket shared by all dynamic reads
on that slice.

There is no expert-specific or epoch-end message. With multiple outstanding
rounds, a future queue implementation must retain per-sender arrival tickets
before advancing the shared group ticket; an untagged additive count can mix
two rounds.

The source slice publishes readiness in one source-indexed signal word on
every PE. The default publishes once after the pool write completes. An
optional two-stage mode publishes monotonic early/final values in the same
word, so it adds doorbells but no signal storage. All readers on a destination
PE share the source ticket.

## Initial HBM Flow

1. The pool block's ordinary VDCores memory/compute streams copy source rows
   once into the local token-slot pool and release a write barrier.
2. At the same time, the communication warp publishes one small route-batch
   descriptor to each target pool slice. The descriptor includes the target
   slice's route begin and end; with one local reader this avoids a separate
   offsets RMA.
3. The target pool warp event loop scans one mailbox and one readiness word per
   source lane. Descriptor and data arrival are independent; a payload is
   eligible when both are present. The pool fetches the source-owned grouped
   route rows and assigns receiver-local contiguous rows. Multiple local
   readers fetch their internal offset boundaries in one source-batched RMA;
   one local reader uses only the descriptor boundaries.
4. As soon as any source is eligible, the target issues its named token-slot
   gets without waiting for later sources. Remote gets precede the synchronous
   local HBM copy; all are NBI and multiple source batches remain in flight
   until a bounded metadata or final completion quiet.
5. After all reads complete, the pool advances the shared group ticket and
   releases the local reader barriers.
6. After reader compute, the pool uses saved provenance to return contiguous
   per-source/per-reader output batches and performs the source-local scatter.

The target does not fetch return provenance: it remains source-owned and is
validated when the source pool scatters returned route-major rows. Queue
descriptor publication, reciprocal data-ready signals, and reciprocal return
signals do not add redundant local quiet phases. Metadata and row payloads use
nonblocking warp RMAs with one required quiet per consumable batch.

The optional `activation_stages=2` mode puts a VDCores barrier on the first
hardware-sized source-write chunk and another on the final chunk. The source
publishes values `2*sequence-1` and `2*sequence`; targets issue prefix rows on
the early value and remaining rows on the final value. Filtering by source row
means route rows need not be sorted. Reader release still waits for the final
quiet, so expert compute never observes a partial gathered buffer.

## Minimal Assumptions

- at most 32 PEs;
- one pool slice per PE in the baseline;
- one outstanding sequence per buffer set;
- fixed-capacity HBM buffers, no fragmentation;
- rows are contiguous, at least 1 KiB, 16-byte aligned, and a multiple of 16
  bytes;
- route metadata is stable-grouped by reader and remains live until the pool
  sequence completes.

## Performance Accounting

Compared with sender-push dispatch, the pool-pull path removes sender packing
and remote tail atomics. Network payload still scales with routed rows. Its
main risk is smaller RMA granularity, so the refinement order is:

1. issue all row gets nonblocking and quiet once per bounded window;
2. coalesce consecutive source slots;
3. ingest metadata while source writes are still running;
4. add multiple SM slices only if one pool warp is demonstrably saturated.

The checked-in implementation completed steps 1 and 3: all row gets are NBI,
mailbox and readiness polling are lane-parallel and skip accepted slots,
metadata and source writes progress independently, and payload gets start per
source instead of after a global data-ready phase. Remote gets are issued
before local copies. Unused provenance traffic was removed and one-reader
offsets are embedded in the existing descriptor.

Two-stage activation readiness is kept opt-in. It starts a small first-chunk
payload earlier and creates early/final in-flight batches, but the extra
doorbell and second route scan did not improve completion on the measured
32/128-token cases. A half-buffer early experiment was rejected because its
large issuance loop delayed the final-ready doorbell; the retained mode limits
the early prefix to one 16-bit-TMA-sized VDCores chunk. A separate experiment
that fused return payload and signaling also regressed and was reverted.

## Measured Boundary

On four Vista GH200 nodes, 32 tokens/PE, 4096 BF16 elements/row, one
reader/PE, top-1, five warmups, and 20 samples:

| PEs | original pool | event-driven pool | NCCL ring | pool/ring |
|---:|---:|---:|---:|---:|
| 2 | 0.329 ms | 0.328 ms | 0.544 ms | 0.60x |
| 4 | 0.533 ms | 0.438 ms | 1.098 ms | 0.40x |

The final pool cost model reports 128 KiB and 192 KiB of remote routed payload
per direction per PE at 2 and 4 PEs. The dense two-all-reduce ring model is
1.5 MiB and 7.5 MiB per PE respectively. These are protocol comparisons, not
claims about a production fused expert kernel: the checked-in reader compute
is an identity copy and top-k output is route-major before a later combine.

Matched 50-sample phased/event-driven trials at the same 32-token shape were
0.333/0.331 ms (2 PE) and 0.459/0.419 ms (4 PE). At 128 tokens/PE, the early
stage moved first payload issue from 0.070 to 0.058 ms on 2 PE, but
dispatch-ready was 0.588 versus 0.578 ms for one stage, so one stage remains
the default.

The final NVSHMEM-linked `dae2` image is `REG=168, STACK=840`; the entry kernel
has no spills and the build emits no WGMMA diagnostic. An initial noinline
staging helper produced `STACK=1112` and was replaced with an inline helper.

VDCores timing remains in the existing communication-warp profile space.
NCCL comparisons remain outside runtime/application source under
`benchmarks/`.
