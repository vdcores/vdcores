# VDCores-Embedded Starting Point

This note connects the VDCores-Embedded proposal to the current repository and
records a small first milestone. It is based on the proposal shared on
2026-09-03 and the current `main` branch.

## Proposed System

VDCores-Embedded would represent CPU and GPU work in one dependency graph.
Operations become eligible to run when their inputs are ready. An operation can
be CPU-only, GPU-only, or have both implementations. For operations with both
implementations, the runtime can choose a device using queue depth, input size,
estimated latency, or a deadline.

The first research target should be dependency and handoff cost, not a complete
placement policy. Dynamic placement is only useful if a GPU-to-CPU-to-GPU chain
can advance cheaply enough to beat explicit launches and synchronization.

## Current Runtime

The current runtime is GPU-only:

- Python applications build per-SM compute and memory instruction streams with
  `python/dae/launcher.py` and `python/dae/instructions.py`.
- `src/runtime.cu` launches one persistent block on each selected GPU SM.
- Each block contains compute and memory virtual cores. The memory side moves
  tiles and publishes ready tokens; the compute side consumes those tokens and
  returns completion or writeback tokens.
- Global barriers coordinate work across SMs. Shared-memory slots carry data
  between the memory and compute sides of one SM.

This provides a dependency protocol to extend, but it does not yet provide a
CPU executor, CPU operation encoding, cross-device completion path, capability
metadata, or CPU/GPU placement policy.

## Split-KV Attention Walkthrough

`app/python/attention_split_kv.py` is a grouped-query attention experiment for
a 32,768-token sequence. It has 32 query heads and 8 key/value heads, so four
query heads share each key/value head.

The example divides the sequence into two equal parts. For each split and each
key/value head, one GPU SM performs local attention over 256 blocks of 64
tokens. This uses 16 SMs in total:

```text
2 sequence splits x 8 key/value heads x 1 request = 16 SMs
```

Each worker reads a query tile, streams its assigned key and value tiles, and
writes two results:

1. a locally normalized attention output;
2. a log-sum-exp value that describes the scale of that local result.

All 16 workers release the same global barrier after writing those results.
The first eight SMs then run `ATTN_SPLIT_POST_REDUCE`, one reducer per
key/value head. The reducer uses the log-sum-exp values to rescale and combine
the two partial outputs into the same result that a single full-sequence
softmax would produce.

The path crosses these files:

- `app/python/attention_split_kv.py`: tensor setup, static SM assignment,
  instruction construction, and PyTorch reference functions.
- `python/dae/instructions.py`: Python encodings for split attention and the
  post-reduction operation.
- `include/dae/compute_dispatch.cuh`: opcode dispatch into CUDA task code.
- `include/task/attention.cuh`: local online softmax and split reduction.

The current example assumes the number of 64-token blocks divides evenly by
the split count. `NUM_KV_BLOCK // split_kv` would otherwise leave remainder
blocks unassigned. That does not affect the checked-in configuration, where
512 blocks divide evenly into two splits, but it matters before parameterizing
the experiment.

## First Milestone

1. Reproduce the split-attention result on one Vista GH200 batch node and save
   the job log, environment versions, and normalized mean error.
2. Dump the generated instruction streams for one worker and one reducer, then
   identify the exact barrier and completion-token path between them.
3. Build a minimal heterogeneous chain: GPU producer, CPU operation, GPU
   consumer. Use one fixed placement and one small shared buffer.
4. Measure the GPU-to-CPU and CPU-to-GPU notification latency before adding
   dynamic placement or a robotics workload.

Questions to settle with the project lead before step 3:

- Should CPU virtual cores be native worker threads inside the Torch extension?
- What operation granularity should the first prototype target?
- Which shared-memory mechanism is the intended GH200 baseline?
- Which end-to-end workload will be used after the microbenchmark?
