## Qwen3-30B-A3B Port Notes

- Entry path: `app/python/qwen3_30b_a3b/`.
- Model geometry matches the existing Qwen3 fused decode-attention path:
  - hidden size `2048`
  - head dim `128`
  - `32` query heads
  - `4` KV heads
  - grouped-query attention ratio `8`
- MoE geometry for this model:
  - `48` layers
  - `128` experts
  - top-`8` experts per token
  - expert intermediate size `768`
- The port uses dynamic layer/expert-indexed TMA loads instead of per-expert or repeated per-layer descriptor sets.
- Current app runtime keeps Q/K/V scratch as current-layer-local while dense and expert weights are stacked per layer for dynamic indexed loading.
- Single-launch looping must follow the same grouped-resource layout rule as `app/python/llama3/sched.py`:
  - `include/dae/pipeline/allocwarp.cuh` applies one running loop `shift` to every grouped memory instruction via `inst.shifter += shift`
  - that shift is not scoped by Python group name at runtime
  - loop-body grouped TMAs and grouped barrier waits therefore need one consistent repeated footprint
- Router projection must not use `Gemv_M128N8` in the app path today: `OP_GEMV_M128N8` exists in Python/opcodes but is commented out in `include/dae/dae2.cuh`. The working app-level fallback is `Gemv_M64N8` over `M=128`.
- Router top-k expert ids must use the raw-address special-slot pattern from `include/task/argmax.cuh`, not a normal TMA store.
- For same-launch router-to-expert execution, `LoadExpertIndex(...)` does not honor `.bar(...)` as a wait. Use an explicit `IssueBarrier(...)` before `LoadExpertIndex(...)`.
- The full single-token synthetic correctness path in `app/python/qwen3_30b_a3b/sched.py` now passes for `--local-generated-weights --synthetic-num-layers 1 --correctness`.
- For performance exploration, fixed small `TOP_K` matters for both work and schedule encoding:
  - with `TOP_K=8`, the repeated per-layer barrier count for 48 layers can exceed the `uint16` barrier-id encoding budget used in memory instructions
  - `--fixed-top-k 2` keeps the repeated barrier footprint within range for a 48-layer build
- Expert activation buffers are now decoupled from `TOP_K` through `--expert-buffers`. Reusing fewer buffers is safe because expert slots are executed sequentially in the current MoE block.
- Current 1-layer synthetic timing samples on this branch:
  - default `TOP_K=8`: about `0.91 ms` execution time
  - `--fixed-top-k 2 --expert-buffers 1`: about `0.64 ms`
  - `--fixed-top-k 2 --expert-buffers 2`: about `0.63 ms`
- On this environment, CUDA rebuild verification is blocked by the active Conda GCC 14 toolchain with CUDA 12.5 `nvcc`; Python static checks pass, but runtime validation needs a supported host compiler setup.

## 2026-03-23 Real-Model Correctness Status

- The active real-model app path is now `app/python/qwen3_moe_30b/`.
- Real-model `--correctness` reaches all `531` checkpoint shards, so the updated Hugging Face model path itself is no longer the first blocker.
- Keeping the Hugging Face model on CUDA is not viable for this path on a 95 GB GH200:
  - the reference model footprint plus stacked MoE runtime tensors exceed device memory before launch
  - keeping the HF model on CPU and copying only the runtime tensors to CUDA removes that OOM
- During correctness bring-up, `dae.set_streaming(...)` is not usable on the full stacked MoE tensors on this machine:
  - `runtime.set_cache_policy(...)` fails with `cudaStreamSetAttribute failed: invalid argument`
  - skipping that cache hint is acceptable for correctness-oriented runs
- Full 48-layer top-8 correctness no longer fails first on the old launch-time barrier/deadlock path after the loop-group fix.
- The earlier real-model CUDA OOM on `matExpertDownWs` was from running two correctness jobs concurrently; a clean single-process run fits through model load and runtime-context staging on this GH200.
- The current single-process real-model blockers are again schedule-side:
  - plain `--correctness` loads all `531` checkpoint shards, reaches `[launch]`, and then fails during instruction build with `RuntimeError: value cannot be converted to type uint16_t without overflow`
  - `--correctness --expert-buffers 1` loads all `531` checkpoint shards, reaches `[launch]`, and then hits a post-launch idle timeout consistent with a barrier deadlock
- Current best-known state for the real-model path:
  - model loading fixed
  - CUDA memory staging fixed for correctness
  - cache-hint blocker avoided
  - generated multi-layer loop/barrier choreography fixed
  - remaining blockers are the full 48-layer top-8 instruction encoding budget and the launched real-weight deadlock when using fewer expert buffers

## 2026-03-23 Generated-Weight Multi-Layer Status

- The generated-weight single-token path still passes for `--local-generated-weights --synthetic-num-layers 1 --correctness`.
- The first multi-layer synthetic failure was a real scratch-lifetime bug, not only a loop-control bug:
  - shared `attnQ/K/V` scratch let layer 2 accumulate stale values from layer 1
- `attnQ/K/V` scratch is now tracked per layer in `app/python/qwen3_moe_30b/runtime_context.py`, and correctness compares against the final-layer scratch tensors.
- A single giant `SchedCopy` is not a safe way to clear full KV scratch:
  - the generated `OP_COPY` size argument is only `uint16`
  - full-tensor KV clears overflow that encoding before launch
- The working multi-layer schedule now mirrors `app/python/llama3/sched.py`:
  - one repeated `layerg` owns the loop-body grouped TMAs and per-layer barriers
  - `LoopM.toNext(..., resource_group=layerg)` advances that repeated footprint
  - loop-time waits on repeated layer barriers use grouped `IssueBarrier(...).group()`
- Current synthetic correctness status:
  - `--local-generated-weights --synthetic-num-layers 1 --correctness`: pass
  - `--local-generated-weights --synthetic-num-layers 2 --correctness`: pass
  - full 48-layer `--local-generated-weights --correctness`: pass
  - final hidden, final RMS, logits, and final token all pass for the 2-layer run
- The full generated-weight 48-layer pass is strong evidence that the current loop-group/barrier layout is correct and that the remaining real-model issue is now outside the launch-time scheduler choreography.
- Router top-k ids may occasionally differ only by slot order while still matching as the same expert set; this does not change the summed MoE result.
