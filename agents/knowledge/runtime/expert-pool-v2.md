# Receiver-Owned Expert Pool

This is the optimized EP protocol built on the integrated VDCores
communication core. The authoritative entry points are
`include/dae/ep_pool.cuh`, `include/dae/ep_pool_abi.cuh`, and
`python/dae/ep_pool.py`.

## Fixed Design

- HBM is sharded by expert owner; there is no central data PE.
- Every global expert has one resident VDCores block on every PE.
- Router output is stable-grouped by expert before dispatch.
- Top-1 writes each active activation once. Top-k has one message row and one
  distinct return row per selected expert; gating reduction is later compute.
- Remote traffic is one descriptor and at most one contiguous payload per
  nonempty `(source PE, expert)` batch in each direction.
- Metadata, data transfer, expert work, and return are one `dae2` program.

## HBM ABI

Producer metadata:

- `send_offsets[num_experts + 1]`: prefix range per global expert;
- `send_rows[route_capacity]`: source activation row for each grouped route;
- `send_origin_rows[route_capacity]`: distinct returned row for each route.

Symmetric message and pool buffers:

- `packed_source[route_capacity, hidden]`: remote batch staging only;
- `expert_input/output[experts_per_pe, capacity, hidden]`;
- `return_inbox[route_capacity, hidden]`: remote return staging;
- `send_batches[num_experts]` and
  `receive_batches[experts_per_pe, num_pes]`;
- one 64-bit tail per local expert;
- monotonic sequence, signal array, and eight control words.

`EpPoolBatch` is 48 bytes: sequence, target base row, source grouped-row base,
row count, source PE, local expert, and flags. `EpPoolConfig` is a stable
192-byte pointer/shape/signal ABI. It is local CUDA memory; every remotely
addressed payload/metadata object is a same-order symmetric allocation.

Signals occupy disjoint ranges:

```text
dispatch: num_experts * num_pes  indexed [expert][source]
return:   num_experts            indexed [expert] on each source PE
reset:    num_pes                indexed [source PE]
```

## Dispatch

For expert `e`, every source block executes concurrently:

1. Validate its grouped route range and source/return row ids.
2. Reserve `row_count` contiguous rows in the owner tail. Local owners use a
   CUDA atomic; remote owners use one NVSHMEM fetch-add.
3. Fill the 48-byte descriptor and issue its blocking warp put immediately.
4. While that descriptor progresses remotely, gather source rows:
   - local expert: copy directly into final `expert_input` rows;
   - remote expert: copy once into the disjoint packed-message segment.
5. The vector copy uses four independent 16-byte values per lane iteration to
   expose HBM memory-level parallelism.
6. For a remote owner, fence the descriptor then issue one NBI
   payload-put-with-signal. For a local owner, device-fence the direct gather
   then set the local signal.
7. The owner warp assigns one lane per source PE, polls all dispatch signals,
   validates descriptors, counts rows, and releases its dispatch barrier.

The descriptor is therefore sent in parallel with data preparation, while the
payload readiness signal remains a defined remote completion point.

## Expert And Return

The owner memory stream waits on `dispatch_barrier[e]`. The correctness
harness then runs ordinary `TmaLoad1D -> Copy -> TmaStore1D` chunks; a real EP
schedule replaces only that subsequence with its expert operators. The last
store releases `compute_barrier[e]`.

The owner return warp reads saved descriptors after that barrier:

- remote source: one contiguous output put-with-signal to
  `return_inbox[source_base:]`;
- local source: no inbox copy; the later scatter reads expert output directly;
- after issuing all peer batches, one quiet makes all output sources reusable.

Every source block waits on its expert return signal and scatters grouped rows
to `returned[send_origin_rows[index]]`. Remote rows read the return inbox;
local rows read the expert output range directly. This local fast path removes
two redundant single-warp copies per round trip.

## Why The Fast Path Matters

The first integrated version used a one-warp pack followed by another local
contiguous copy, and staged local returns through the inbox. At 128 tokens/PE,
BF16 hidden 4096 on two GH200 nodes, that took 1.699 ms end to end despite low
network volume. System-level accounting showed extra serialized HBM traffic,
not the network, on the critical path.

Direct local gather/return plus four-way vector-copy ILP reduced that case to
0.397 ms in the final ordered protocol. Dispatch-ready fell from 0.789 to
0.145 ms. This is why local and remote message paths intentionally differ by
one staging copy.

## Cost Model

For each nonempty remote source/expert batch:

- one tail atomic;
- one 48-byte descriptor;
- one activation RMA plus one dispatch signal;
- one return RMA plus one return signal.

Every source also sends one reset signal per remote PE. Local HBM work is one
read/write gather and one read/write return scatter per route, plus actual
expert work. Empty expert batches still publish descriptors/signals so the
owner has a fixed fan-in count.

The external `benchmarks/ep_pool_nccl_compare.py` harness forces
`NCCL_ALGO=Ring`. No NCCL code or timing is part of the VDCores runtime or
application source. Pool time comes only from communication-warp timestamps in
the internal `g_events` profile space; CUDA events are confined to the NCCL
reference. The dense baseline performs an expert-major dispatch all-reduce and
token-major return all-reduce. Ring network traffic is modeled as
`2 * (P - 1) / P * tensor_bytes` per collective per PE. Read latency together
with the printed bytes/RMAs/atomics/signals; do not infer the design from a
profiler trace alone.

Final two-node GH200 medians for BF16 hidden 4096, one expert/PE, five warmups
and 20 measured iterations are:

| Tokens/PE | Pool end-to-end | NCCL ring | Pool/ring |
|---:|---:|---:|---:|
| 8 | 0.162 ms | 0.303 ms | 0.535x |
| 32 | 0.199 ms | 0.540 ms | 0.369x |
| 128 | 0.397 ms | 0.452 ms | 0.879x |

At 32 tokens/PE the pool sends 128 KiB of remote payload per direction and one
remote data RMA, while the two dense ring collectives model 1.5 MiB of network
traffic per PE.

With eight experts/PE at 32 tokens/PE, the pool measures 0.274 ms versus
0.707 ms for ring (0.387x). The pool still moves only routed rows, while the
dense dispatch tensor grows with the 16 global experts.

## Constraints And Failure States

- rows are contiguous, at least 1 KiB, and divisible by 16 bytes;
- at most 32 PEs and 132 global experts;
- expert capacity must cover the maximum routed fan-in;
- source and origin rows must fit their configured capacities;
- sequence values must strictly increase;
- status is first-error-wins: bad config, route range, pool capacity, batch, or
  signal range.

The one-block-per-expert kernel reserves enough shared memory for at most one
block/SM, so all dependency participants must fit within the physical SM count.
