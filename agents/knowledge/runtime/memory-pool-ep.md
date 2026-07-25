# Memory-Pool EP Protocol

This note fixes the first implementation contract for dependency-aware expert
parallel communication. The implementation builds on the optional NVSHMEM
runtime and the alloc-warp operator pattern proposed in GitHub issue `#25`.

## V1 Execution Model

- One NVSHMEM PE is selected as the pool PE for a launch.
- One SM on that PE is dedicated to the pool. Its existing alloc warp executes
  `OP_MEMORY_POOL_RUN`; no fifth warp or change to the ordinary VM shape is
  required for V1.
- Producers execute `OP_MEMORY_POOL_SUBMIT` after any local `IssueBarrier` that
  protects newly written HBM data. Consumers execute `OP_MEMORY_POOL_WAIT`
  before using returned data.
- The pool core is a single scheduler/executor. It scans all producer
  mailboxes, skips requests whose dependency ticket is not ready, and executes
  another ready mailbox. This is the required reordering point.
- One producer owns each mailbox and may have one outstanding request in it.
  The producer must observe completion before reusing that mailbox.

This deliberately favors a small correctness surface. A new pool warp is only
justified after profiling shows that dedicating an alloc warp prevents useful
overlap.

## HBM Layout

All remotely addressed objects are symmetric allocations made in the same
order on every PE.

1. `MemoryPoolRequest[mailbox_count]`: fixed 128-byte metadata records.
2. `pool_data`: byte-addressed payload arena owned logically by the pool PE.
3. `route_tables`: `uint32` row indices, separate from payload data.
4. `data_scratch` and `route_scratch`: pool-local staging areas.
5. `dependencies[dependency_count]`: monotonic 64-bit slot tickets.
6. `consumed_sequences[mailbox_count]`: last sequence consumed per mailbox.
7. `signal_array`: symmetric NVSHMEM submit and completion signals.

A submit uses put-with-signal so the request record is visible at the pool
before its submit signal. Completion signals are monotonic sequence values, so
normal operation does not reset signals and does not have an ABA window.

## Request Contract

Each request carries:

- sequence, opcode, flags, and user tag;
- source, destination, and route-table symmetric addresses;
- source, target, and completion PEs plus completion signal id;
- pool byte offset and byte count;
- `wait_slot` / `wait_value` and `signal_slot` / `signal_delta`;
- row count, row width, and source/destination strides for scatter/gather.

`UINT32_MAX` means no dependency slot. A request is ready when it has no wait
slot or `dependencies[wait_slot] >= wait_value`. Only after the data operation
has completed does the pool add `signal_delta` to `dependencies[signal_slot]`.
For the common fan-in case, 16 writes each add one ticket and dependent reads
wait for value 16.

V1 uses explicit slot tickets rather than address hashing. Slot tickets make
epochs and fan-in counts unambiguous; address-derived keys can be layered on by
the scheduler later without changing transfer semantics.

## Pool Operations

- `WRITE`: NVSHMEM get from `source_pe/source_address` into `pool_data`.
- `READ`: NVSHMEM put from `pool_data` to
  `target_pe/destination_address`.
- `SCATTER`: fetch the route table, then get source rows into routed pool rows.
- `GATHER`: fetch the route table, then put routed pool rows back in request
  row order.
- `REDUCE_SUM_F32` flag on `WRITE`: stage a source contribution and accumulate
  it into the pool arena before releasing the dependency ticket. This provides
  the first tensor-parallel fan-in test without a collective.

Scatter and gather metadata is explicitly separate from payload storage. V1
routes rows with a `uint32` table and requires the addressed spans to fit the
configured pool and scratch capacities. Top-1 EP uses unique routed rows;
weighted top-k combination is a later compute operation, not an implicit
overwrite in `GATHER`.

## VDCores Operators

- `OP_NVSHMEM_PUT` / `OP_NVSHMEM_WAIT`: corrected issue-`#25` one-sided
  primitives with an explicit signal id.
- `OP_MEMORY_POOL_SUBMIT`: copy one 128-byte request to the pool PE and publish
  its sequence on the selected submit signal.
- `OP_MEMORY_POOL_WAIT`: wait for the request's completion signal to reach its
  sequence.
- `OP_MEMORY_POOL_RUN`: scan/reorder/execute a configured number of requests.

All are non-allocating memory/control instructions handled directly by the
alloc warp. They do not consume normal shared-memory slots or enqueue `LdCmd`s.

## Ordering Invariants

1. A producer barrier, when required, reaches zero before submit.
2. Put-with-signal publishes complete request metadata.
3. A pool dependency is advanced only after its transfer or reduction is
   complete.
4. A completion signal is published only after returned data is globally
   visible.
5. A producer does not overwrite its mailbox until completion.
6. Pool termination requires exactly the configured number of completions;
   an unsatisfied dependency is treated as a schedule deadlock and tested with
   the repository timeout wrapper.

## Verification Stages

1. Host-only layout, encoding, dependency-reordering, and scatter/gather
   reference tests.
2. Optional-extension build and focused API tests.
3. Two-PE dependent read/write test: multiple writes fan into a pool slot,
   then a read returns only after the ticket threshold.
4. Two-PE top-1 EP test: scatter routed token rows, execute the local expert,
   and gather rows to original token order.
5. Only after correctness, profile the pool loop with Nsight Systems/Compute
   before changing warp placement or batching.

Multi-PE stages require a Vista allocation and `ibrun`; local pytest is not a
substitute for that boundary.

## Verified Baseline

On 2026-07-25, the implementation was verified on two Vista GH nodes with one
NVSHMEM PE/GH200 per node:

- both `make pyext` and `make nvshmem-pyext` completed;
- host/API suite: `14 passed`, with two opt-in GPU tests skipped by default;
- singleton GPU integration suite: `2 passed`;
- two-PE dependent test: 8 writes/PE produced ticket 16 and both reads returned
  the exact sum value 136;
- two-PE top-1 EP test: both PEs passed scatter, VDCores copy/expert compute,
  return scatter, and original-order gather;
- `compute-sanitizer --tool memcheck` reported zero errors for small dependent
  and EP runs when `NVSHMEM_DISABLE_NCCL=1` was set.

The optional build used 178 registers for `dae2` versus 168 in the ordinary
build and ptxas warned that external NVSHMEM calls may serialize WGMMA. Treat
warp placement, RMA batching, and that optional-build interaction as measured
performance work; they are not correctness blockers.
