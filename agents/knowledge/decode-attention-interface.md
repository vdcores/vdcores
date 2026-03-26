# Decode Attention Interface

- Shared decode attention now treats `kv_seq_len` as the runtime source of truth for active KV coverage.
  - Python entry points: `python/dae/instructions.py`, `python/dae/schedule.py`
  - CUDA entry points: `include/dae/compute_dispatch.cuh`, `include/task/attention.cuh`

- Non-split decode attention passes `kv_seq_len` as a whole instruction arg.
  - The kernel reconstructs `num_kv_blocks = ceil(kv_seq_len / KV_BLOCK_SIZE)` and `last_kv_active_token_len` at the start of the kernel.
  - This keeps the internal loop structure unchanged while simplifying Python-side packing.

- Split-KV decode attention also uses a whole-arg `kv_seq_len`, but it is split-local.
  - `kv_seq_len` is the active tokens covered by that split, not the global cache length.
  - `kv_start_idx` still carries the split's global token offset.
  - Remaining split metadata is packed into `arg0` as `split_idx`, `num_active_q`, and runtime flags.

- Compatibility remains for older Python callers.
  - Non-split wrappers still accept `active_kv_len=` and legacy `(num_kv_block, last_kv_active_token_len)` inputs.
  - Experimental scripts with stale `hist_len=` kwargs still instantiate, but the decode path now keys off `kv_seq_len`.

- Practical verification paths for this interface:
  - `python app/python/attention_simple_decoding.py`
  - `python app/python/llama3/sched.py --correctness`
