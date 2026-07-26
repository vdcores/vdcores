# Generic Dependency Memory Pool

This is the correctness-first mailbox protocol used by
`app/python/memory_pool/dependent_rw.py`. The optimized batched gathered-read
protocol is documented in `pool-slice-dynamic-read.md`.

## Execution Model

Generic pool controls are communication instructions, not memory/allocator
opcodes. In the optional build, the ninth VDCores warp executes
`MemoryPoolSubmit`, `MemoryPoolWait`, and `MemoryPoolRun` inside the same `dae2`
launch as ordinary memory and compute streams.

One logical pool block scans HBM mailboxes and reorders requests according to
dependency tickets. Producer blocks submit and wait. The pool scan is
warp-parallel: each lane polls one mailbox in a 32-entry window, a ballot forms
the ready set, and the warp cooperatively executes one selected request.

## HBM Layout

All remote objects are symmetric, same-order NVSHMEM allocations:

1. `MemoryPoolRequest[mailbox_count]`, 128 bytes per producer mailbox;
2. byte-addressed `pool_data`;
3. separate `uint32` route tables;
4. pool-local data and route scratch;
5. monotonic `dependencies[dependency_count]` slot tickets;
6. `consumed_sequences[mailbox_count]`;
7. disjoint submit/completion signal slots;
8. four control words: status, completed count, mailbox, user tag.

Each producer owns one mailbox and does not reuse it until completion.
Put-with-signal publishes the full request before its submit sequence becomes
visible.

## Request And Dependency Contract

A request contains source/destination/route addresses, participating PEs,
pool span, row shape/strides, completion signal, sequence, user tag, and:

```text
wait_slot / wait_value
signal_slot / signal_delta
```

`UINT32_MAX` means no slot. A request is ready when it has no wait or
`dependencies[wait_slot] >= wait_value`. The pool increments the signal slot
only after its data operation is complete, then publishes completion.

The common fan-in maps directly to a slot barrier: 16 writes each add one;
dependent reads wait for value 16. Slot ids make epochs and fan-in explicit and
avoid ambiguous address-hazard inference.

## Operations

- `WRITE`: get producer bytes into the pool.
- `READ`: put pool bytes to the consumer.
- `WRITE | REDUCE_SUM_F32`: stage and accumulate a contribution before adding
  its dependency ticket.
- `SCATTER`: fetch route metadata, then gather source rows into routed pool
  rows.
- `GATHER`: fetch route metadata, then return routed pool rows in request order.

The selected operation is executed cooperatively by all 32 communication-warp
lanes. Remote NBI movement is quieted before ticket/completion publication.
Malformed spans, dependencies, routes, signals, sequences, or reduction shapes
set an explicit status and terminate the pool program.

## VDCores Schedule

`make_phase_schedule` maps the pool core and producer mailboxes to distinct
blocks. A producer may place `CommWaitBarrier` in its communication stream
before submit when local data is produced in the same launch. `IssueBarrier`
belongs to the independent memory VM and therefore cannot order a
communication submit.

```text
producer communication: [wait local barrier] -> submit -> wait completion
pool communication:      scan -> choose ready -> move data -> ticket -> signal
```

The dependent-read/write app uses only communication work, so it requests zero
dynamic slot storage. It still launches one normal `dae2` program with
terminating memory/compute streams.

## Verified Boundary

On two GH200 nodes, eight writes/PE followed by one dependent read/PE produced
slot value 16 and exact reduced value 136 on both PEs. The opt-in singleton
test also covers a 16-write fan-in; the host suite covers ABI, reordering,
deadlock/reference behavior, and scatter/gather semantics.
