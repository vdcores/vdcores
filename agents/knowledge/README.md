# Knowledge Index

Use this directory as the first stop when a task needs stable repo context.

## Topic Map

- Project structure and main entry points:
  - [project-map.md](/home1/11362/depctg/vdcores/agents/knowledge/project-map.md)
- Scheduler and model-path specifics:
  - [llama-scheduling.md](/home1/11362/depctg/vdcores/agents/knowledge/llama-scheduling.md)
  - [mistral-small-24b-port.md](/home1/11362/depctg/vdcores/agents/knowledge/mistral-small-24b-port.md)
  - [qwen3-attention.md](/home1/11362/depctg/vdcores/agents/knowledge/qwen3-attention.md)
- Runtime mechanics:
  - [runtime/README.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/README.md)
  - [nvshmem-runtime.md](/home1/11362/depctg/vdcores/agents/knowledge/nvshmem-runtime.md)
  - [runtime/vdcores-vm-model.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/vdcores-vm-model.md)
  - [runtime/vdcores-operator-semantics.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/vdcores-operator-semantics.md)
  - [runtime/vdcores-queues.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/vdcores-queues.md)
  - [runtime/memory-core-performance-knobs.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/memory-core-performance-knobs.md)
  - [runtime/gemm-scheduler.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/gemm-scheduler.md)
  - [runtime/memory-pool-ep.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/memory-pool-ep.md)
- Model/framework behavior:
  - [modeling/pytorch-kv-cache.md](/home1/11362/depctg/vdcores/agents/knowledge/modeling/pytorch-kv-cache.md)
- Performance/debugging lessons:
  - [attention-performance.md](/home1/11362/depctg/vdcores/agents/knowledge/attention-performance.md)
- Scheduling demos:
  - [lora-scheduling-demo.md](/home1/11362/depctg/vdcores/agents/knowledge/lora-scheduling-demo.md)
- Utility abstractions:
  - [cord-adapters.md](/home1/11362/depctg/vdcores/agents/knowledge/cord-adapters.md)

## Loading Hints

- If the task touches HF cache bootstrapping or multi-token decode:
  - read [modeling/pytorch-kv-cache.md](/home1/11362/depctg/vdcores/agents/knowledge/modeling/pytorch-kv-cache.md)
  - then read [qwen3-attention.md](/home1/11362/depctg/vdcores/agents/knowledge/qwen3-attention.md) or [llama-scheduling.md](/home1/11362/depctg/vdcores/agents/knowledge/llama-scheduling.md)
- If the task touches attention writeback, new slots, or deadlocks after launch:
  - read [runtime/vdcores-vm-model.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/vdcores-vm-model.md)
  - then read [runtime/vdcores-operator-semantics.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/vdcores-operator-semantics.md)
  - read [runtime/vdcores-queues.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/vdcores-queues.md)
  - then read [runtime/memory-core-performance-knobs.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/memory-core-performance-knobs.md)
  - then read [qwen3-attention.md](/home1/11362/depctg/vdcores/agents/knowledge/qwen3-attention.md)
- If the task is broad and unfamiliar:
  - start with [project-map.md](/home1/11362/depctg/vdcores/agents/knowledge/project-map.md)
- If the task asks for runtime internals or opcode semantics:
  - start with [runtime/README.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/README.md)
- If the task touches multi-node symmetric memory or MPI bootstrap:
  - read [nvshmem-runtime.md](/home1/11362/depctg/vdcores/agents/knowledge/nvshmem-runtime.md)
  - then follow [../workflows/nvshmem-tacc.md](/home1/11362/depctg/vdcores/agents/workflows/nvshmem-tacc.md)
- If the task touches dependency-aware pool communication or expert routing:
  - read [runtime/memory-pool-ep.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/memory-pool-ep.md)
  - then follow [../workflows/memory-pool-ep.md](/home1/11362/depctg/vdcores/agents/workflows/memory-pool-ep.md)
