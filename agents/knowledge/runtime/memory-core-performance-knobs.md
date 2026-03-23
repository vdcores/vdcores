# Memory Core Performance Knobs

These notes summarize the memory-core runtime path that matters for end-to-end decode latency.

## Entry Points

- [include/dae/dae2.cuh](/home1/11362/depctg/vdcores/include/dae/dae2.cuh)
- [include/dae/pipeline/allocwarp.cuh](/home1/11362/depctg/vdcores/include/dae/pipeline/allocwarp.cuh)
- [include/dae/pipeline/ldwarp.cuh](/home1/11362/depctg/vdcores/include/dae/pipeline/ldwarp.cuh)
- [include/dae/pipeline/stwarp.cuh](/home1/11362/depctg/vdcores/include/dae/pipeline/stwarp.cuh)
- [include/dae/allocator.cuh](/home1/11362/depctg/vdcores/include/dae/allocator.cuh)
- [include/dae/queue.cuh](/home1/11362/depctg/vdcores/include/dae/queue.cuh)
- Benchmark harness: [app/python/llama3/sched.py](/home1/11362/depctg/vdcores/app/python/llama3/sched.py)

## Runtime Shape

- `dae2(...)` splits each block into:
  - compute warps in threads `[0, numComputeWarps * 32)`
  - one alloc/control warp
  - one store warp
  - two load warps
- The memory side communicates through three shared queues:
  - `m2c`: load-ready notifications to compute
  - `c2m`: slot free or writeback requests from compute
  - `m2ld[2]`: load commands for the two load ports
- `st_insts[numSlots + numSpecialSlots]` holds the staged memory instruction metadata for active slots.

## Current Hot Spots

- Allocation retry path:
  - `allocwarp_execute(...)` loops on `alloc.allocate(...)` until a slot range is free.
  - The retry loop currently sleeps with `__nanosleep(16)`.
- Issue-barrier wait path:
  - `allocwarp_execute(...)` handles `OP_ISSUE_BARRIER` by polling global barrier state with `__nanosleep(16)`.
  - `ldwarp_execute_singlethread(...)` also polls read barriers before descriptor-backed loads with `__nanosleep(16)`.
- Single-lane queue commit path:
  - lane 0 stages `st_insts[slot]`, writes `m2c`, emits an `LdCmd`, and advances both queues.
  - load/store warps are also single-lane executors today.
- Slot free/writeback path:
  - store warp executes the TMA/global store and only then frees the slot through `c2m.reset(slot_mask)`.
  - any extra wait in `stwarp` lengthens the time from compute completion to allocator reuse.

## Practical Knobs

- Polling backoff:
  - the `__nanosleep(...)` values directly trade off contention against wakeup latency
  - small waits may help if the common case is short-lived slot/barrier availability
- Redundant queue/barrier traffic:
  - `allocwarp` currently performs separate `put/commit/advance` sequences on `m2c` and `m2ld`
  - this path is worth keeping short because it runs per allocated memory instruction
- Shared-memory instruction movement:
  - `st_insts[slot] = inst` copies the full `MInst`
  - reducing repeated copies or dependent operations around this store may lower issue latency
- Load-side wait placement:
  - `ldwarp` advances the queue before the barrier/load handling, so queue progress is decoupled from load completion
  - barrier polling still controls the time from command receipt to actual TMA issue
- Writeback wait placement:
  - `stwarp` waits after each writeback group before freeing slots or releasing global barriers
  - per-store wait behavior is a likely contributor when many short writebacks serialize reuse

## Benchmarked Lesson

- On 2026-03-22, the first runtime tweak that produced a repeatable win on the Llama3 single-token benchmark was reducing the memory-core polling sleeps from `50-64` cycles to `16` cycles:
  - `allocwarp` allocation retry loop
  - `allocwarp` `OP_ISSUE_BARRIER` polling loop
  - `ldwarp` pre-load barrier polling loop
- Measured with `python app/python/llama3/sched.py -N 1 -b 3`:
  - baseline: `4872709.74 ns` average duration, `4887061.33 ns` average execution time
  - final `16`-cycle backoff: `4852520.81 ns` average duration, `4866677.33 ns` average execution time
- Two nearby experiments did not beat the simpler `16`-cycle setting:
  - `8`-cycle polling backoff was slightly slower than `16`
  - explicit power-of-two wrap helpers for queue and instruction indexing were slightly slower than the original code
- Shared-memory-focused follow-up on 2026-03-22 used a longer benchmark window, `python app/python/llama3/sched.py -N 1 -b 10`:
  - current kept version: `4846318.50 ns` average duration, `4860364.80 ns` average execution time
  - reference before the shared-memory experiments: `4844073.19 ns` average duration, `4857955.20 ns` average execution time
- Shared-memory experiments from that round did not beat the kept version:
  - richer `m2ld` queue commands that let LD issue without rereading `st_insts[slot]` increased shared queue traffic enough to regress
  - skipping `st_insts` writes for non-writeback ops was invalid because several compute-side tasks use `slot_2_glob_ptr(...)` on those staged instructions
  - lane-0 fetch plus warp broadcast for `smem_minsts[next_pc]` was effectively flat to slightly slower
  - copying `slot_insts[slot]` into a register-local `MInst` in `stwarp` was slightly slower

## Global Barrier Notes

- The readbar path currently lives in [include/dae/pipeline/ldwarp.cuh](/home1/11362/depctg/vdcores/include/dae/pipeline/ldwarp.cuh):
  - descriptor-backed read barriers poll `bars + inst.bar()` before issuing the load
  - the current code uses a plain volatile poll loop with `__nanosleep(...)`
- `load_l2(...)` in [include/dae/virtualcore.cuh](/home1/11362/depctg/vdcores/include/dae/virtualcore.cuh) provides a cached global read primitive, but in this runtime the barrier path still showed noticeable run-to-run variance even with `-b 10`.
- On 2026-03-22, two global-barrier experiments did not show a stable end-to-end win:
  - allocwarp L2 prefetch of the target barrier line plus `ldwarp` first-check via `load_l2(...)`
  - allocwarp early cached read of the barrier and a compact `LdCmd` flag to let `ldwarp` skip the check when the barrier was already zero
- The skip-on-zero idea is logically attractive for monotonic countdown barriers, but the measured `python app/python/llama3/sched.py -N 1 -b 10` result regressed enough that it was reverted.
- Current measured reverted-code reference after this round:
  - run 1: `4833444.17 ns` average duration, `4847296.00 ns` average execution time
  - run 2: `4843763.95 ns` average duration, `4857731.20 ns` average execution time
  - simple average of those two `-b 10` runs: `4838604.06 ns` duration, `4852513.60 ns` execution time
- The more likely next step is a larger protocol refactor rather than another one-line cache hint:
  - carry barrier-ready state earlier in the memory pipeline with a design that avoids adding extra global reads on the common path
  - or reduce the number of global readbar checks that reach `ldwarp` in the first place

## Instruction Prefetch Notes

- With `dae2LoadInstructions = false` in [include/dae/context.cuh](/home1/11362/depctg/vdcores/include/dae/context.cuh), `allocwarp` fetches memory instructions directly from global memory rather than shared memory.
- On 2026-03-22, a software-pipelined allocwarp instruction prefetch in [include/dae/pipeline/allocwarp.cuh](/home1/11362/depctg/vdcores/include/dae/pipeline/allocwarp.cuh) produced a stable win:
  - seed prefetches for PCs `0` and `1` before entering the loop
  - keep a sequential prefetch two instructions ahead (`pc + 2`)
  - prefetch the loop stream explicitly for `REPEAT` and for resolved `LOOP` targets
- Prefetch helper lives in [include/dae/virtualcore.cuh](/home1/11362/depctg/vdcores/include/dae/virtualcore.cuh) as a `prefetch.global.L1` wrapper.
- Measured with `python app/python/llama3/sched.py -N 1 -b 10`:
  - reverted current-code reference:
    - run 1: `4833444.17 ns` duration, `4847296.00 ns` execution
    - run 2: `4843763.95 ns` duration, `4857731.20 ns` execution
    - two-run average: `4838604.06 ns` duration, `4852513.60 ns` execution
  - instruction-prefetch build:
    - run 1: `4791600.92 ns` duration, `4805532.80 ns` execution
    - run 2: `4817880.41 ns` duration, `4831766.40 ns` execution
    - two-run average: `4804740.67 ns` duration, `4818649.60 ns` execution
- The prefetch version increased `dae2` register usage from `201` to `203` with no spills, and the net end-to-end result was still positive enough to keep.
- A follow-up sweep on 2026-03-22 used longer runs, `python app/python/llama3/sched.py -N 16 -b 10`, to probe prefetch footprint and timing:
  - current kept policy (`distance=2`, `seed=2`, `targetSpan=2`), three-run average:
    - `77975152.39 ns` duration
    - `77989255.47 ns` execution
  - wider branch-target window (`targetSpan=3`) looked promising on one three-run sample:
    - sample 1: `77818710.03 ns` duration, `77832814.93 ns` execution
    - but revalidation came back slower: `78133536.95 ns` duration, `78147636.27 ns` execution
    - combined six-run average was effectively flat to slightly worse than the kept policy: `77976123.49 ns` duration, `77990225.60 ns` execution
  - unconditional early `OP_LOOP` target prefetch regressed clearly:
    - `79759257.51 ns` duration, `79773549.87 ns` execution
  - shorter sequential distance (`pc + 1`) was also slightly slower:
    - `77989615.14 ns` duration, `78003619.20 ns` execution
  - wider target window (`targetSpan=4`) overshot and regressed more clearly:
    - `78338233.00 ns` duration, `78352228.27 ns` execution
- Current takeaway for allocwarp instruction prefetch:
  - the kept policy is still the simpler `pc + 2` sequential hint plus a two-instruction explicit target window for `REPEAT` and resolved `LOOP`
  - under `-N 16`, extra branch-target footprint was not consistently beneficial, and unconditional early `LOOP` prefetch added more pressure than value

## PTX / SASS Notes

- `cuobjdump --dump-sass runtime.o` shows that the memory-core helpers are fully inlined into the single [include/dae/dae2.cuh](/home1/11362/depctg/vdcores/include/dae/dae2.cuh) kernel, so small source-shape changes matter only if they shorten or simplify the inlined hot path.
- On 2026-03-22, two PTX-guiding source rewrites were tested with `python app/python/llama3/sched.py -N 16 -b 10` and neither earned a keep:
  - ldwarp/stwarp 1D fast path:
    - source change: special-case `OP_ALLOC_TMA_LOAD_1D` in [include/dae/pipeline/ldwarp.cuh](/home1/11362/depctg/vdcores/include/dae/pipeline/ldwarp.cuh) and `OP_ALLOC_WB_TMA_STORE_1D` in [include/dae/pipeline/stwarp.cuh](/home1/11362/depctg/vdcores/include/dae/pipeline/stwarp.cuh) ahead of the larger switches, using `__builtin_expect(...)`
    - build stayed at `203` registers and `0` spills
    - three-run average regressed to `78140181.35 ns` duration and `78154289.07 ns` execution
  - allocwarp branch-layout rewrite:
    - source change: mark the allocation path as likely and flatten the non-allocation control dispatch in [include/dae/pipeline/allocwarp.cuh](/home1/11362/depctg/vdcores/include/dae/pipeline/allocwarp.cuh) from a switch into an if/else chain
    - build again stayed at `203` registers and `0` spills
    - three-run average was `77955675.72 ns` duration and `77969752.53 ns` execution versus the current kept-policy baseline of `77975152.39 ns` / `77989255.47 ns`
    - that apparent gain is only about `0.025%`, which is too small to trust given the run-to-run variance already observed in this benchmark
- Current takeaway for PTX-level tuning:
  - avoiding a broader switch is not automatically a win; on this kernel, the extra source-level fast-path branches were at least as expensive as the compiler’s existing lowering
  - a rewrite is only worth keeping here if it produces a clearly repeatable multi-run improvement, not just a tiny change within benchmark noise

## Benchmark Convention

- For a single-token Llama3 end-to-end decode baseline, use:

```bash
python app/python/llama3/sched.py -N 1 -b 1
```

- `-N 1` keeps only the initial token path in [app/python/llama3/sched.py](/home1/11362/depctg/vdcores/app/python/llama3/sched.py).
- Runtime-touching changes should be rebuilt with `make pyext` before rerunning the benchmark.
