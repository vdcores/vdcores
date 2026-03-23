# Knowledge Index

Use this directory as the first stop when a task needs stable repo context.

## Topic Map

- Project structure and main entry points:
  - [project-map.md](/home1/11362/depctg/vdcores/agents/knowledge/project-map.md)
- Scheduler and model-path specifics:
  - [llama-scheduling.md](/home1/11362/depctg/vdcores/agents/knowledge/llama-scheduling.md)
  - [qwen3-attention.md](/home1/11362/depctg/vdcores/agents/knowledge/qwen3-attention.md)
- Runtime mechanics:
  - [runtime/vdcores-queues.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/vdcores-queues.md)
  - [runtime/memory-core-performance-knobs.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/memory-core-performance-knobs.md)
- Model/framework behavior:
  - [modeling/pytorch-kv-cache.md](/home1/11362/depctg/vdcores/agents/knowledge/modeling/pytorch-kv-cache.md)
- Performance/debugging lessons:
  - [attention-performance.md](/home1/11362/depctg/vdcores/agents/knowledge/attention-performance.md)
- Utility abstractions:
  - [cord-adapters.md](/home1/11362/depctg/vdcores/agents/knowledge/cord-adapters.md)

## Loading Hints

- If the task touches HF cache bootstrapping or multi-token decode:
  - read [modeling/pytorch-kv-cache.md](/home1/11362/depctg/vdcores/agents/knowledge/modeling/pytorch-kv-cache.md)
  - then read [qwen3-attention.md](/home1/11362/depctg/vdcores/agents/knowledge/qwen3-attention.md) or [llama-scheduling.md](/home1/11362/depctg/vdcores/agents/knowledge/llama-scheduling.md)
- If the task touches attention writeback, new slots, or deadlocks after launch:
  - read [runtime/vdcores-queues.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/vdcores-queues.md)
  - then read [runtime/memory-core-performance-knobs.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/memory-core-performance-knobs.md)
  - then read [qwen3-attention.md](/home1/11362/depctg/vdcores/agents/knowledge/qwen3-attention.md)
- If the task is broad and unfamiliar:
  - start with [project-map.md](/home1/11362/depctg/vdcores/agents/knowledge/project-map.md)
