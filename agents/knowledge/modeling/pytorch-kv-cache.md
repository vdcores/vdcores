# PyTorch KV Cache Notes

These notes summarize the Hugging Face cache behavior confirmed while bootstrapping Qwen3 multi-token decode.

## Entry Points

- Qwen3 scheduler: [app/python/qwen3/sched.py](/home1/11362/depctg/vdcores/app/python/qwen3/sched.py)
- Qwen3 reference helpers: [app/python/qwen3/reference.py](/home1/11362/depctg/vdcores/app/python/qwen3/reference.py)

## Basic HF Behavior

- Use `model(..., use_cache=True)` to ask HF to return a cache object.
- For current Qwen3 in this repo environment, `past_key_values` is a `transformers.cache_utils.DynamicCache`, not a tuple.
- `DynamicCache.layers[i]` is a `DynamicLayer`.
- The useful tensors on each layer are:
  - `layer.keys`
  - `layer.values`

## Confirmed Shapes

- For Qwen3-8B, cached tensors are shaped like:
  - `keys`: `[batch, num_kv_heads, seq, head_dim]`
  - `values`: `[batch, num_kv_heads, seq, head_dim]`
- In the 1-token bootstrap case, layer-0 cache shape was:
  - `[1, 8, 1, 128]`

## Practical Conversion To VDCores Buffers

- VDCores decode buffers in the Qwen3 path use:
  - `attnKs[layer]`: `[REQ, MAX_SEQ_LEN, num_kv_heads * head_dim]`
  - `attnVs[layer]`: `[REQ, MAX_SEQ_LEN, num_kv_heads * head_dim]`
- `values` can be copied almost directly:
  - take `layer.values[0]`
  - permute to `[seq, num_kv_heads, head_dim]`
  - reshape to `[seq, num_kv_heads * head_dim]`
  - copy into `attnVs[layer][req, :seq]`
- `keys` need a layout conversion before copy:
  - take `layer.keys[0]`
  - permute to `[seq, num_kv_heads, head_dim]`
  - reshape to `[seq, num_kv_heads * head_dim]`
  - run `permute_rope_activation(...)`
  - copy into `attnKs[layer][req, :seq]`

## Why K Needs Conversion

- HF cached K layout does not match the interleaved layout expected by the fused Qwen3 decode attention path in this repo.
- Measured on the single-token bootstrap:
  - raw cached K vs. reference interleaved K had a large mean absolute mismatch
  - `permute_rope_activation(...)` reduced that mismatch to near zero
- V did not need that extra conversion.

## Reference-Path Guidance

- Use `use_cache=False` inside reference-capture passes unless the test explicitly needs cache inspection.
- For decode-with-prefill correctness:
  - run HF on the prefix with `use_cache=True` to seed buffers
  - run the full prefix+decode sequence through the reference helper
  - compare the decode token at the later sequence index, not the prefix token

## Position Handling

- When bootstrapping with explicit positions, pass `position_ids` directly through helper inputs.
- For a prefix token at position `0` followed by a decode token at position `1`, the decode attention path should see `last_kv_active_token_len=2` in the first decode attention instruction.
