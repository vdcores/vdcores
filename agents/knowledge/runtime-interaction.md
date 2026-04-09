# Runtime Interaction Cheat Sheet

This note summarizes the main Python-side contract for driving the VDCores runtime.

## Core Files

- `python/dae/launcher.py`: launcher, SM builders, resource groups, barrier binding, launch
- `python/dae/schedule.py`: schedule classes such as `SchedGemv`, `SchedAttentionDecoding`, `SchedRMSShared`, `SchedCopy`
- `python/dae/instructions.py`: compute atoms (`Gemv_M64N8`, `Gemv_M64N8B2`, etc.) and memory-side instructions (`TmaTensor`, `TmaLoad1D`, `TmaStore1D`, `RegStore`, `RegLoad`)
- `python/dae/tma_utils.py`: TMA builders and cord adapters
- `app/python/llama3/sched.py`: good end-to-end reference for grouped resources and layer looping
- `app/python/llama32_1b/sched.py`: good reference for stage-wise profiling and isolated latency experiments

## Typical Build Order In Python

1. Construct `dae = Launcher(num_sms, device=...)`
2. Define tensors and choose cache policy (`set_persistent`, `set_streaming`) if needed
3. Create resource groups with `dae.get_group()` / `dae.add_group(name, repeat)`
4. Register barriers with `addBarrier(...)`
5. Register reusable TMAs with `addTma(...)`
6. Call `dae.build_groups()`
7. Construct schedules and attach logical bars like `"load"` / `"store"`
8. Call `place(num_sms, base_sm=...)` on schedules after the schedule shape is final
9. Bind late barriers after placement with `dae.bind_late_barrier_counts(...)` or a custom helper
10. Submit schedules with `dae.i(...)`
11. Finish with `dae.s()` and launch through `dae_app(dae)` or `dae.launch()` / `dae.bench(...)`

## Resource Groups

- A `ResourceGroup` is the standard way to create per-layer or repeated TMA/barrier resources.
- `repeat > 1` means one TMA instance per repeat and one barrier generation per repeat, plus one extra barrier generation so `next(...)` / `over(...)` still work after the last repeat.
- `group["name"]` returns the first instance of a resource.
- `group.next("name")` gives the next repeated barrier generation.
- `group.over("name")` jumps over all repeated generations.

## Late-Bound Barrier Counts

- Barrier ids are allocated early, but many counts are unknown until schedules are placed.
- Every schedule can report its release counts through `bar_release_count(...)`.
- `Launcher.collect_barrier_release_counts(...)` scans schedules and sums those counts.
- `Launcher.bind_late_barrier_counts(...)` writes those summed counts back into any late-bound barriers in every resource group.
- If a barrier is not actually used in a debug prefix, bind it to `0` rather than leaving it unresolved.

## `SchedGemv` Mental Model

- `SchedGemv` is the main abstraction for WGMMA GEMV-style compute in app schedules.
- For an atom with `MNK = (TileM, TileN, TileK)` and a placed schedule with matrix `M x N x K`:
  - `M / TileM` determines the number of output tiles
  - placing on more SMs than that introduces split-K folding
  - `fold = num_sms / (M / TileM)` when fold is not specified explicitly
  - `sm_per_fold = num_sms / fold`
  - `k_per_fold = K / fold`
- Each SM gets:
  - one output tile index from `sm % sm_per_fold`
  - one K-fold index from `sm // sm_per_fold`
- If `fold > 1`, the output TMA must be a reduction store because multiple SMs accumulate into the same output tile.

## GEMV Legality Rule

- `SchedGemv.validate()` requires:
  - `k_per_fold % TileK == 0`
  - `k_per_fold % (TileK * Atom.n_batch) == 0`
  - `k_per_fold >= TileK * Atom.n_batch`
- This matters a lot when changing placement or atom family.
- Examples:
  - `Gemv_M64N8`: `TileK * n_batch = 256 * 4 = 1024`
  - `Gemv_M64N8B2`: `256 * 2 = 512`
  - `Gemv_M64N8K64`: `64 * 1 = 64`

## Memory-Side Building Blocks

- `TmaTensor`: reusable multi-dimensional tensor load/store/reduce descriptor
- `TmaLoad1D` / `TmaStore1D`: byte-range vector load/store helpers
- `RegStore` / `RegLoad`: register-backed handoff between adjacent compute stages
- `RawAddress`: direct raw global pointer operand for specialized kernels

## TMA And Cord Adapters

- Build TMAs once and reuse them through `.cord(...)`.
- Use `tma_utils` adapters when the schedule-space coordinates do not match the tensor's native coordinate layout.
- Common adapters:
  - `StaticCordAdapter`
  - `ToLinearCordAdapter`
  - `ToRopeTableCordAdapter`
  - `ToSplitMCordAdapter`
  - `ToAttnKVStoreCordAdapter`
  - `ToAttnVStoreCordAdapter`

## `RepeatM.onSync(...)`

- This is the standard memory-side repetition pattern for WGMMA GEMV schedules.
- It emits repeated TMA loads across K with one optional barrier on the first repeated load.
- In `SchedGemv`, this is how weight/activation tiles are prefetched while the compute atom consumes the current tiles.

## Placement Advice

- For latency work, always calculate fold explicitly in your head before choosing an SM count.
- More SMs are not always better:
  - they may create illegal `k_per_fold`
  - they may force reduction stores
  - they may reduce per-SM pipeline depth
- Different GEMV atoms change the legal placement space because `TileK * n_batch` changes.

## Profiling Advice

- For logical-op timing, use fresh-process prefix subtraction:
  - `--debug-num-layers 1`
  - `--debug-stop-after <stage>`
  - benchmark fresh processes
  - subtract adjacent cumulative prefixes
- Prefer medians over means when a stage occasionally spikes.
- Use `tests/script/run_with_launch_timeout.py` when debugging a new prefix or risky schedule change.

## Practical References

- `app/python/llama3/sched.py`: grouped per-layer resources and looped layer execution
- `app/python/llama32_1b/sched.py`: isolated stage-prefix profiling and current 1B latency experiments
- `agents/knowledge/cord-adapters.md`: adapter-specific notes
- `agents/knowledge/llama-scheduling.md`: 1B-specific lessons and timing findings
