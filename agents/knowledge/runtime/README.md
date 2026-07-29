# Runtime Knowledge Index

Start here for the VDCores runtime and VM model.

## Recommended Load Order

1. [vdcores-vm-model.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/vdcores-vm-model.md)
2. [vdcores-operator-semantics.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/vdcores-operator-semantics.md)
3. [configurable-vdcores.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/configurable-vdcores.md)
4. [vdcores-queues.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/vdcores-queues.md)

Then pull in narrower notes as needed:

- [memory-core-performance-knobs.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/memory-core-performance-knobs.md)
- [gemm-scheduler.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/gemm-scheduler.md)
- [memory-pool-protocol.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/memory-pool-protocol.md)
- [vdcores-communication-core.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/vdcores-communication-core.md)
- [pool-slice-dynamic-read.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/pool-slice-dynamic-read.md)
- [pool-host-sgl.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/pool-host-sgl.md)
- [external-ep-baselines.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/external-ep-baselines.md)

## What Each Note Covers

- [vdcores-vm-model.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/vdcores-vm-model.md):
  per-SM runtime layout, register state, queues, allocator, slot lifetime, and `GROUP` / accumulate behavior
- [vdcores-operator-semantics.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/vdcores-operator-semantics.md):
  instruction-field meanings and operator-by-operator state transitions
- [configurable-vdcores.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/configurable-vdcores.md):
  compile-time kernel envelopes, per-block roles, and PoolInst executor assembly
- [vdcores-queues.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/vdcores-queues.md):
  queue protocol and deadlock-oriented guidance
- [memory-pool-protocol.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/memory-pool-protocol.md):
  generic HBM mailbox layout, dependency tickets, and pool-core execution
- [vdcores-communication-core.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/vdcores-communication-core.md):
  communication ISA, macro-core isolation, warp roles, and ordering rules
- [pool-slice-dynamic-read.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/pool-slice-dynamic-read.md):
  distributed slice ownership, merged signals, batched dynamic reads, and
  measured optimization results
- [pool-host-sgl.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/pool-host-sgl.md):
  optional Grace coherent request ring and direct-HBM host-verbs transport
- [external-ep-baselines.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/external-ep-baselines.md):
  pinned UCCL/Triton EP adapters, comparison contract, and 2/4/8-PE results
