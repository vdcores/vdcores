# Shared Attention Split Schedule

- `python/dae/schedule.py` now contains `SchedAttentionSplit`, a request-scoped split-attention schedule that uses the usual `.place(num_sms, base_sm)` pattern.
- The class does not choose its own split policy. Callers are expected to decide `split_kv`, `split_q_tile`, and `splits_per_post_load` externally and pass request-local TMA handles plus request-local output/stat tensors.
- `python/dae/tma_utils.py` now contains request-local split-attention TMA helpers:
  - `tma_split_load_q` / `cord_split_load_q`
  - `tma_split_load_k` / `cord_split_load_k`
  - `tma_split_load_v` / `cord_split_load_v`
  - `tma_split_load_o` / `cord_split_load_o`
- The current shared implementation assumes the request-local tensor slice layout used by `app/python/compare_kitten/split_sched.py` and only supports the `HEAD_DIM=128` split attention compute opcode path.
