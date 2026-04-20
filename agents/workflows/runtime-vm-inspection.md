# Runtime VM Inspection Workflow

Use this when a task asks for VDCores runtime semantics rather than a feature change.

1. Start with the runtime shape:
   - [include/dae/context.cuh](/home1/11362/depctg/vdcores/include/dae/context.cuh)
   - [include/dae/dae2.cuh](/home1/11362/depctg/vdcores/include/dae/dae2.cuh)
2. Read the memory-side mechanics in this order:
   - [include/dae/virtualcore.cuh](/home1/11362/depctg/vdcores/include/dae/virtualcore.cuh)
   - [include/dae/queue.cuh](/home1/11362/depctg/vdcores/include/dae/queue.cuh)
   - [include/dae/allocator.cuh](/home1/11362/depctg/vdcores/include/dae/allocator.cuh)
   - [include/dae/pipeline/allocwarp.cuh](/home1/11362/depctg/vdcores/include/dae/pipeline/allocwarp.cuh)
   - [include/dae/pipeline/ldwarp.cuh](/home1/11362/depctg/vdcores/include/dae/pipeline/ldwarp.cuh)
   - [include/dae/pipeline/stwarp.cuh](/home1/11362/depctg/vdcores/include/dae/pipeline/stwarp.cuh)
3. Read compute dispatch next:
   - [include/dae/compute_dispatch.cuh](/home1/11362/depctg/vdcores/include/dae/compute_dispatch.cuh)
4. For each compute op you need, inspect the matching task header under:
   - [include/task/](/home1/11362/depctg/vdcores/include/task)
5. Cross-check the Python-side field packing:
   - [python/dae/instructions.py](/home1/11362/depctg/vdcores/python/dae/instructions.py)
   - [python/dae/launcher.py](/home1/11362/depctg/vdcores/python/dae/launcher.py)
6. Check the current build-selection layer before claiming an op is runnable:
   - [build/generated/dae/selected_compute_ops.inc](/home1/11362/depctg/vdcores/build/generated/dae/selected_compute_ops.inc)
   - [build/generated/dae/dynamic_compute_handlers.inc](/home1/11362/depctg/vdcores/build/generated/dae/dynamic_compute_handlers.inc)
7. Persist the results in `agents/knowledge/runtime/`:
   - one VM-model note for stable runtime structure
   - one operator-semantics note for opcode-by-opcode behavior
   - update [agents/knowledge/runtime/README.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/README.md) if the load order changes

## Questions To Answer Explicitly

- How many normal slots, special slots, queues, and warps exist?
- What is the exact meaning of each queue payload?
- How are normal slots allocated and returned?
- What do `gpr[0]` and `gpr[1]` do?
- How do `REPEAT`, `JUMP`, `LOOP`, and `GROUP` interact?
- Which operators are source-defined versus actually selected in the current build?
