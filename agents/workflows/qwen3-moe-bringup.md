# Qwen3 MoE Bring-Up Workflow

Use this when porting a new Qwen3 MoE model into `app/python/`.

## 1. Start With Single-Operator Checks

Do not debug the full schedule first. Isolate the new pieces in this order:

- router top-k by itself
- indexed TMA loads with a dummy copy instruction
- router output to `LoadExpertIndex(...)`
- one expert block
- multi-slot expert block
- attention block
- epilogue
- one full layer executed operator-by-operator

This catches three common failures quickly:

- raw-address output bugs
- indexed TMA coordinate/layout mismatches
- barrier waits that are encoded on the wrong instruction kind

## 2. Raw Router Output Rule

If an operator writes through a raw global pointer, follow the special-slot pattern used by `include/task/argmax.cuh`.

For router top-k expert ids:

- write ids via `slot_2_glob_ptr(st_insts, slot)`
- do not model ids as a normal TMA store

## 3. Indexed TMA Rule

Before trusting a new indexed TMA tensor family, verify it with a dummy copy harness.

For Qwen3-30B-A3B, these indexed layouts were validated this way:

- dense `layer`
- expert `layer_expert`
- router weights using the exact router GEMV tile shape

## 4. Barrier Wait Rule

`LoadExpertIndex(...)` is a control op. In this runtime, putting `.bar(...)` on it does not create the wait you usually want.

If a control op depends on an earlier producer:

- insert `IssueBarrier(bar_id)` first
- then emit the control op

This was required for the same-launch router top-k to expert-selection handoff.

## 5. Watch For Unsupported Atoms

Before using a new compute atom in an app schedule, confirm it is actually dispatched in `include/dae/dae2.cuh`.

Important current repo fact:

- `OP_GEMV_M128N8` is defined in Python/opcodes
- but its runtime dispatch path is commented out in `include/dae/dae2.cuh`

So an app schedule must currently avoid `Gemv_M128N8` unless that runtime path is implemented.

## 6. Use Sequential Full-Layer Harnesses

If isolated operators all pass but the main schedule still hangs, build a one-layer harness that runs operator-by-operator with explicit `IssueBarrier(...)` calls after each stage.

This separates:

- operator correctness
- from schedule/barrier choreography bugs

For Qwen3-30B-A3B this was the turning point that proved the kernels were fine and the remaining issue was in the assembled app schedule.

## 7. Correctness Before Performance

Get this path green first:

```bash
python tests/script/run_with_launch_timeout.py \
  --post-launch-timeout 120 \
  --post-launch-idle-timeout 30 \
  -- python app/python/qwen3_30b_a3b/sched.py \
    --local-generated-weights \
    --synthetic-num-layers 1 \
    --correctness
```

Only after that should you switch to benchmark tuning.

## 8. Performance Tuning Levers

After correctness passes, the fastest early wins were:

- fix `TOP_K` to a smaller value such as `2`
- reduce expert activation buffers below `TOP_K`
- keep correctness-only sequencing out of benchmark-only paths when safe

For Qwen3-30B-A3B, `--fixed-top-k 2` is also useful structurally:

- it reduces MoE work
- and it keeps repeated per-layer barrier ids within the `uint16` encoding budget used by memory instructions

Sample microbenchmark command:

```bash
python app/python/qwen3_30b_a3b/sched.py \
  --local-generated-weights \
  --synthetic-num-layers 1 \
  --fixed-top-k 2 \
  --expert-buffers 2 \
  -b 5
```

## 9. Benchmarking Notes

- Do not use full synthetic 48-layer local-generated MoE weights as a timing baseline; tensor construction and memory footprint dominate before scheduling becomes interesting.
- Prefer small synthetic layer counts for scheduling microbenchmarks.
- Compare variants with the same `synthetic-num-layers`, `fixed-top-k`, and `expert-buffers` settings.
