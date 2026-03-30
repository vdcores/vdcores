# Llama3 Attention Split

- `app/python/llama3/sched.py` now uses `SchedAttentionSplit` instead of `SchedAttentionDecoding` for the attention stage.
- The current integration is manually tuned rather than policy-driven:
  - `ATTN_SPLIT_KV = 2`
  - `ATTN_SPLIT_SMS_PER_REQ = 16`
  - `ATTN_SPLIT_Q_TILE = 4`
  - `ATTN_SPLITS_PER_POST_LOAD = 2`
- The split-attention path reuses the grouped per-layer Q/K/V TMAs through `MappedCordAdapter` and reloads split outputs from a shared scratch tensor.
- `python/dae/schedule.py::SchedAttentionSplit` now schedules post-reduction per KV head using `HEAD_GROUP_SIZE`, which is the shape needed for llama3-style grouped-query attention.
