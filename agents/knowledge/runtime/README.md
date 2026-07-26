# Runtime Knowledge Index

Start here for the VDCores runtime and VM model.

## Recommended Load Order

1. [vdcores-vm-model.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/vdcores-vm-model.md)
2. [vdcores-operator-semantics.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/vdcores-operator-semantics.md)
3. [vdcores-queues.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/vdcores-queues.md)

Then pull in narrower notes as needed:

- [memory-core-performance-knobs.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/memory-core-performance-knobs.md)
- [gemm-scheduler.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/gemm-scheduler.md)
- [memory-pool-ep.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/memory-pool-ep.md)
- [expert-pool-v2.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/expert-pool-v2.md)
- [vdcores-communication-core.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/vdcores-communication-core.md)
- [pool-slice-dynamic-read.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/pool-slice-dynamic-read.md)

## What Each Note Covers

- [vdcores-vm-model.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/vdcores-vm-model.md):
  per-SM runtime layout, register state, queues, allocator, slot lifetime, and `GROUP` / accumulate behavior
- [vdcores-operator-semantics.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/vdcores-operator-semantics.md):
  instruction-field meanings and operator-by-operator state transitions
- [vdcores-queues.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/vdcores-queues.md):
  queue protocol and deadlock-oriented guidance
- [memory-pool-ep.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/memory-pool-ep.md):
  HBM mailbox layout, dependency tickets, pool-core execution, and EP routing
- [expert-pool-v2.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/expert-pool-v2.md):
  receiver-owned expert pools, overlapped descriptor/data publication, local
  fast paths, return scatter, and the external NCCL comparison boundary
- [vdcores-communication-core.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/vdcores-communication-core.md):
  third-domain communication ISA, ninth-warp execution, ordering rules,
  code-generation isolation, and the minimal per-expert protocol
- [pool-slice-dynamic-read.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/pool-slice-dynamic-read.md):
  pool-owned sender queues, distributed slice ownership, shared completion
  tickets, and dynamic gathered-read execution
