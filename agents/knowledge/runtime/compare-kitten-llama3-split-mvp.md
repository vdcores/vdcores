# Compare-Kitten Llama3 Split MVP

- `app/python/compare_kitten/attention_simple_split_llama3.py` is a minimal driver for the shared `SchedAttentionSplit`.
- It uses llama3-style grouped-query attention dimensions:
  - `NUM_Q_HEAD = 32`
  - `NUM_KV_HEAD = 8`
  - `HEAD_DIM = 128`
- The script manually builds request-local TMA descriptors with the shared split-attention helpers in `python/dae/tma_utils.py` and then places one `SchedAttentionSplit` per request.
- It also includes a PyTorch grouped-attention reference path for output comparison.
