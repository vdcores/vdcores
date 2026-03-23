Qwen3 decode attention differences from llama3
- File entry points:
  - scheduler: `app/python/qwen3/sched.py`
  - reference capture: `app/python/qwen3/reference.py`
  - decode scheduling helper: `python/dae/schedule.py`
  - kernel/runtime path: `include/task/attention.cuh`, `include/dae/dae2.cuh`

- Qwen3 decode attention fuses more work than the llama3 path:
  - Q and the current-token K both take per-head RMS-affine normalization inside decode attention.
  - Interleaved RoPE is applied inside decode attention for both Q and the current-token K.
  - Historical cached K values stay as-is; only the current-token K row in the last KV block is transformed in-kernel.

- Current-token K writeback uses descriptor-backed TMA store into `attnKs`, not a raw-address path.
  - The working layout is one store descriptor per request over contiguous `attnKs[layer][req]` slices.
  - Head placement is done by a per-head offset in the 1D descriptor space.
  - For BF16 `tensor1d` descriptors, the offset used in `cord(...)` must be expressed in BF16-element units for this path, not raw bytes, or writes land on every other head.

- Qwen3 side inputs are packed offline to reduce decode attention load count.
  - `app/python/qwen3/sched.py` builds `matQwenSideInputs[layer][token_pos] = [q_norm | k_norm | rope]`.
  - Decode attention now issues one grouped TMA load for all three aux inputs and splits that slot in-kernel.
  - The packed row size is `3 * HEAD_DIM` BF16 values, which is `768` bytes for the current 128-dim head path.

- HF prefix caches can be used to bootstrap multi-token decode checks.
  - Qwen3 returns a `DynamicCache` whose layers expose `keys` and `values` tensors shaped like `[batch, num_kv_heads, seq, head_dim]`.
  - `values` can be flattened into `[seq, num_kv_heads * head_dim]` and copied straight into `attnVs`.
  - `keys` need an interleaving conversion before copying into `attnKs`; `permute_rope_activation(...)` matches the cache layout expected by the fused decode path.

- Queue semantics matter for any new writeback slot in attention.
  - Temporary inputs like Q/K norm weights or RoPE rows can be released with `c2m.push(...)`.
  - Any slot that should actually write to global memory must use the writeback queue path, for example `c2m.template push<0, true>(thread_id, slot_id)`.

- Single-token correctness for Qwen3 should focus first on:
  - layer-0/1 `v_proj`
  - layer-0/1 `q_proj_interleaved`
  - layer-0/1 post-fused `q_rope_interleaved`
  - layer-0/1 stored `k_rope_interleaved`
  - final logits and final token
