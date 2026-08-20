# Runtime Knowledge Index

Start here for the VDCores runtime and VM model.

## Recommended Load Order

1. [vdcores-vm-model.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/vdcores-vm-model.md)
2. [vdcores-operator-semantics.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/vdcores-operator-semantics.md)
3. [vdcores-queues.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/vdcores-queues.md)

Then pull in narrower notes as needed:

- [blackwell-port.md](blackwell-port.md)
- [deepseek-v4-flash.md](deepseek-v4-flash.md)
- [memory-core-performance-knobs.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/memory-core-performance-knobs.md)
- [gemm-scheduler.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/gemm-scheduler.md)

## What Each Note Covers

- [vdcores-vm-model.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/vdcores-vm-model.md):
  per-SM runtime layout, register state, queues, allocator, slot lifetime, and `GROUP` / accumulate behavior
- [vdcores-operator-semantics.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/vdcores-operator-semantics.md):
  instruction-field meanings and operator-by-operator state transitions
- [blackwell-port.md](blackwell-port.md):
  SM100/SM103 build targeting, device-derived launch limits, and the Blackwell runtime smoke test
- [deepseek-v4-flash.md](deepseek-v4-flash.md):
  single-GPU model shape, NVFP4/FP8 checkpoint contracts, task coverage, and verified baselines
- [deepseek-v4-nongemm-consolidation.md](deepseek-v4-nongemm-consolidation.md):
  production non-GEMM fusion, MXFP8 conversion ownership, and full-image acceptance gates
- [deepseek-v4-bf16-flashmla-port.md](deepseek-v4-bf16-flashmla-port.md):
  B64 BF16 sparse-attention port contract, general retained internal-ring LDU
  operator, CTA-parallel reducer, and performance acceptance gates
- [vdcores-queues.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/vdcores-queues.md):
  queue protocol and deadlock-oriented guidance
