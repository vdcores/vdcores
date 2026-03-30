# Attention Split Request Scheduler

- `app/python/compare_kitten/split_sched.py` contains `SchedAttentionSplit`, a request-scoped helper for split-KV attention scheduling.
- The class accepts a `Launcher`, one request id, split level, assigned SM count, sequence lengths, and the backing Q/K/V/O/split-output/stat tensors.
- It infers `num_kv_head`, `head_group_size`, `num_q_head`, `head_dim`, and `kv_seq_len` from the tensor shapes instead of requiring separate shape arguments.
- It builds request-local TMA descriptors for Q, K, V, and split-output reloads by slicing tensors down to a single request before constructing `TmaTensor` objects.
- It allocates a per-request barrier sized to the number of active split tasks for that request and exposes `sm_task(sm)` for local SM scheduling.
- `app/python/compare_kitten/attention_simple_split.py` shows the intended integration pattern: build one `SchedAttentionSplit` per request, then map global SM ids to `(req_id, local_sm)` before calling `sm_task`.
- The helper is intentionally local to `compare_kitten` rather than `python/dae/schedule.py`, so experimental split-attention scheduling can iterate without expanding the shared scheduler surface.
