# DeepSeek-V4 profile-optimization loop

Use this loop for the single-GPU, one-launch DeepSeek-V4-Flash target.

## Fixed invariants

- Run the checkpoint-resident path on one allocator-managed B300 on
  `10.0.16.24`; the durable checkpoint stays at
  `/mnt/checkpoints/nvidia/DeepSeek-V4-Flash-NVFP4`.
- Keep one VDCores launch per token. Compute tasks receive allocator-owned
  shared-slot masks only; LDU resolves global, routed, and indexed addresses;
  STU owns global stores.
- Do not introduce `__threadfence`, compute/memory joint barriers, allocator
  `IssueBarrier`, queued-address compatibility state, indirect stores, or
  per-layer instruction unrolling.
- Prefer compact loops and barrier shifts. Preserve reference token 14 for
  input token 791 in the 43-layer position-zero gate.

## Baseline record

For every phase, retain revision, build flags, selected compute operators,
rank/host allocation, warmup/sample count, CUDA samples, internal frontier,
correctness result, and relevant aggregate counters. Use at least five samples
for exploration and 30 samples before making a final TBT comparison.

The starting complete-token baseline is 79.530 ms median with 74.797 ms in 43
layers, 2.555 ms in barrier reloads, and 1.923 ms in the full vocabulary head.
The target is 16 ms.

The current accepted production record is 11.593600 ms median at the native
q_b milestone: one GPU, one persistent launch, three warmups, 30 samples, and
exact token 14. Keep the historical starting baseline for attribution, but
compare new end-to-end milestones against this current record.

## Phase 1: task kernels

1. Rank tasks by end-to-end contribution, then run shape- and math-matched
   VDCores and external baselines on the same GPU.
2. First test layout/traffic changes that can remove whole transactions:
   prepack immutable checkpoint tensors, colocate data and scale for one TMA,
   and preload short fixed/frequent state through raw/fixed address or LDU
   register state.
3. Test Blackwell-native task variants and VDCores-specific queue depth,
   producer/consumer, and slot-lifetime changes.
4. Test adaptive fusion through shared-memory `LoadReg`/`StoreReg` handoff for
   measured adjacent-task traffic. Keep global addresses out of compute.
5. Accept only correctness-preserving task wins that improve the containing
   layer or full token; commit each independent valid milestone.

## Phase 2: overlap

1. Replace unconditional predecessor edges with explicit data dependencies.
2. Build ready sets for Q/KV, eight attention output groups, six routed experts
   plus shared expert, and W1/W3 fan-out.
3. Assign SM partitions jointly from ready-task shapes; do not give every
   independent branch all 152 SMs sequentially.
4. Split independent payload streams across LDU0/LDU1. Gate only dependent
   operands and avoid reserving shared slots while an LDU dependency is not
   ready.
5. Add tile-level release or shared-slot handoff only after the broad DAG is
   correct. Reprofile after each graph milestone.

## Phase 3: progressive slowdown

1. Capture per-layer frontier, barrier reload, allocator, LDU0/LDU1, STU, and
   compute-wait counters for repeated 43-layer samples.
   Add `--profile-layers --profile-all-samples` to report every sample rather
   than only the median sample; each report includes exact command counts and
   counter time normalized to the SM-grid envelope.
2. Compare identical-shape family iterations and correlate the first
   divergence with loop counters, barrier-bank reuse, pointer-table position,
   instruction working set, cache state, and GPU clocks/thermals.
3. Reproduce suspected causes in the smallest looped family benchmark before
   changing the runtime.
4. Separate device runtime growth from host launch/setup and profiling-marker
   overhead. Accept a fix only when both the micro-reproducer and complete
   token improve.

## Milestone gate

Run focused CPU tests, the affected GPU task correctness test, the one-layer
gate, and the 43-layer token gate in that order. Compare medians to the saved
baseline. Commit only a valid improvement and record its measured before/after
result; record rejected experiments in `.agentlog/` and revert only the
experiment's own changes.
