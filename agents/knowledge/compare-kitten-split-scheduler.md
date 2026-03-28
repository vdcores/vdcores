# Compare-Kitten Split Scheduler

- Entry point: `app/python/compare_kitten/attention_split_schedule.py`
- The legacy scheduled split-attention path assigns attention shards and post-reduce workers separately.
- Only requests with `split_level > 1` should reserve post-reduce workers. Unsplit requests write their final output directly from `sm_attn_task(...)` and should carry `post_groups=[]`.
- `SchedPlan.__post_init__` must therefore special-case `split_level == 1`; otherwise any q-tiling derived from `len(post_groups)` will divide by zero on mixed workloads.
- For split requests, the post worker count must evenly divide `NUM_Q_HEAD`, because `sm_post_task(...)` maps each reserved post group to one contiguous q-head tile.
