# VDCores Communication Core

Communication is a third instruction domain inside the persistent `dae2`
kernel. It is not an auxiliary kernel, host callback, or CUDA stream.

## VM Shape And Isolation

The ordinary build is unchanged: four compute warps plus four memory warps,
256 threads. `DAE_ENABLE_NVSHMEM` adds one communication warp, for 288 threads.
Existing compute dispatch, allocation, store, and two load-warp roles retain
their original code and relative warp ids.

Each block consumes three independent instruction streams:

| Domain | ABI | Capacity | Consumer |
|---|---:|---:|---|
| compute | `CInst`, 8 B | 512 | four-warp compute group |
| memory | `MInst`, 16 B | 512 | alloc/store/load warps |
| communication | `CommInst`, 16 B | 32 | optional communication warp |

`CommInst` has `opcode`, three 16-bit operands, and one 64-bit HBM address. It
has no allocator flag bits and never consumes a shared-memory slot or enters a
load/store queue. Zero is `COMM_TERMINATE`, so an untouched communication
buffer is inert for existing programs.

The communication ISA is defined in
`include/dae/communication_opcode.cuh.inc`:

- barrier wait and device timestamp;
- corrected NVSHMEM put/wait primitives;
- generic dependency-pool submit/wait/run;
- expert-pool reset/dispatch/return;
- pool-slice publish/gather/return.

`Launcher` always passes all three streams to the single `launch_dae` call.
Communication instructions are rejected unless the optional runtime was
built. There are no direct `launch_ep_pool_*` or control-kernel bindings.

VDCores timing uses only `COMM_RECORD_EVENT` writes to the existing per-block
`g_events` profile space. Backend comparisons and their timing mechanisms live
under `benchmarks/`; NCCL and CUDA-event timing are not runtime operators or
application dependencies.

## Code-Generation Boundary

Relocatable NVSHMEM device calls and inline WGMMA in one function cause ptxas
to insert conservative WGMMA serialization. In the optional build only, the
WGMMA GEMV/GEMM/attention task bodies are no-inline boundaries. The ordinary
build keeps their original force-inline definitions.

Nine Hopper warps place three warps on one SM subpartition. The optional build
therefore uses `-maxrregcount=168`; without it the final NVSHMEM device link
raises the kernel to 254 registers/thread and launch fails. The EP WGMMA GEMV
fits the limit with zero spills. Always inspect the final linked extension,
not only pre-link ptxas output.

## Minimal EP Contract

- one block per global expert, all resident concurrently;
- `num_experts <= physical SM count <= 132`;
- at most 32 PEs, so one lane polls one source;
- contiguous rows, at least 1 KiB, 16-byte aligned, byte width divisible by 16;
- stable expert-grouped route offsets, source rows, and return rows in HBM;
- one outstanding sequence per buffer set, with strictly increasing sequence
  values;
- fixed-capacity, separately allocated message buffers; no fragmentation or
  byte-tail path.

These assumptions match common inference activation rows and keep the hot path
to one vector-copy and one contiguous RMA form.

## Pool-Slice Dynamic Read

The newer protocol uses one pool block per PE instead of one owner block per
global expert. Block 0's communication warp performs all publication, queue
scan, dependency resolution, gathered reads, and return/scatter operations.
Reader blocks have no communication instructions; their memory VMs wait on
pool-released barriers.

Every source pool publishes exactly one descriptor to every target slice,
including an empty descriptor. Queue signal `source_pe` is the sender's
monotonic ticket on that target. Consuming all `P` tickets advances one local
`group_ready` sequence shared by every dynamic reader, so there is no
reader-specific epoch-end message. Source data readiness and return completion
each use another `P`-entry signal range, for a total baseline signal footprint
of `3P` words per PE. Optional early/final source readiness reuses each data
word with monotonic staged values.

The three `P`-word signal ranges must be disjoint. Queue and data-ready slots
are source-indexed; return slots are producing-slice-indexed. Plain reciprocal
signal phases need no extra local quiet because every peer waits on the full
set. Payload RMAs are quieted before their completion dependency is published.

Token rows remain in the source slice and are copied there once. A route table
may name one source row multiple times for top-k fan-out. The target pool pulls
only its stable-grouped metadata span, assigns receiver-owned contiguous rows,
and issues nonblocking gets from the named source slots. See
`pool-slice-dynamic-read.md` for the HBM ABI and optimization order.

## Dependency Structure

Global barrier words connect the three local VDCores domains. Monotonic HBM
signals connect PEs.

```text
communication: reset -> dispatch(e) ------------> return(e) -> terminate
memory(owner):           wait dispatch -> loads -------------> stores
compute(owner):                          expert operators
                                           |
                                  last store releases
                                  compute_barrier[e]
```

Reset is also explicit message traffic. Block 0 clears local tails/control,
sends reset-ready sequence `reset_base + my_pe` to every PE, polls all local
source slots warp-parallel, and releases a local reset barrier. Other expert
blocks wait on that barrier. No NVSHMEM collective is hidden in the protocol.

Dispatch on the owner releases `dispatch_barrier[e]` only after all source
signals and descriptors validate. The owner memory stream waits on that word.
The final expert store releases `compute_barrier[e]`; only the owner return
warp waits on it. Nonowners wait directly for their return signal.

If routing is produced inside the same larger program, the routing operator
can release another ordinary barrier and place `CommWaitBarrier` before
`ExpertPoolDispatch`. The checked-in harness prepares routes before launch.

## Ordering Rules

The protocol deliberately distinguishes local CUDA ordering from remote
NVSHMEM completion:

1. Reset system-fences local symmetric tail/control clears before publishing
   reset-ready to remote PEs.
2. Local direct gathers/scatters use `__syncwarp`, `__threadfence`, then a
   local atomic signal.
3. A remote 48-byte descriptor uses a blocking warp put. It is issued before
   row packing, so its remote progress overlaps the gather.
4. After packing, `nvshmem_fence` orders that blocking descriptor before one
   nonblocking data-put-with-signal.
5. Observing the put-with-signal flag means that payload is delivered; a plain
   signal is never used as completion for an unordered NBI payload.
6. Return issues one put-with-signal per source and performs one quiet after
   all peer batches, making expert output reusable when the VDCores program
   exits.
7. Signal comparisons use `>= sequence`; buffers require monotonically
   increasing sequences and are not cleared between phases.

Generic pool requests use put-with-signal for their 128-byte mailbox record,
warp-parallel mailbox polling, a quiet after request data movement, dependency
ticket update, and then completion signal.

## Entry Files

- VM/ISA: `include/dae/context.cuh`, `include/dae/dae2.cuh`,
  `include/dae/pipeline/commwarp.cuh`
- generic pool: `include/dae/memory_pool.cuh`, `python/dae/memory_pool.py`
- expert pool: `include/dae/ep_pool_abi.cuh`, `include/dae/ep_pool.cuh`,
  `python/dae/ep_pool.py`
- pool-slice dynamic read: `include/dae/pool_slice_abi.cuh`,
  `include/dae/pool_slice.cuh`, `python/dae/pool_slice.py`
- encoding/launch: `python/dae/instructions.py`, `python/dae/launcher.py`
- correctness/benchmark: `app/python/memory_pool/`
