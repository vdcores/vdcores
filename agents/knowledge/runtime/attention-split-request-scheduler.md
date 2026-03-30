# Attention Split Request Scheduler

- `app/python/compare_kitten/split_sched.py` contains `SchedAttentionSplit`, a request-scoped helper for split-KV attention scheduling.
- The class accepts a `Launcher`, one request id, assigned SM count/base SM, sequence lengths, and the backing Q/K/V/O/split-output/stat tensors.
- It infers `num_kv_head`, `head_group_size`, `num_q_head`, `head_dim`, and `kv_seq_len` from the tensor shapes instead of requiring separate shape arguments.
- It builds request-local TMA descriptors for Q, K, V, and split-output reloads by slicing tensors down to a single request before constructing `TmaTensor` objects.
- It allocates a per-request barrier sized to the number of active split tasks for that request and exposes `sm_task(sm)` for local SM scheduling.
- `app/python/compare_kitten/split_sched.py` also contains `GlobalSchedAttentionSplit`, which assigns SMs across requests with a min-makespan heuristic before instantiating one request-local scheduler per request.
- `app/python/compare_kitten/attention_simple_split.py` now shows the current integration pattern: build one `GlobalSchedAttentionSplit`, inspect per-request assignments, and pass `global_scheduler.schedule` into `dae.i(...)`.
- The helper is intentionally local to `compare_kitten` rather than `python/dae/schedule.py`, so experimental split-attention scheduling can iterate without expanding the shared scheduler surface.
- Runtime tracing now has a separate memory-op append buffer. `Launcher` allocates `mem_trace_count` and a raw `mem_trace` tensor, then passes them through the CUDA launch path.
- The trace record format is defined in `include/dae/context.cuh` as `MemTraceRecord { start, end, address, size, opcode, kind, arg }`, with `maxMemTraceRecords` reserved per SM.
- Load records are appended from `include/dae/pipeline/ldwarp.cuh` after any barrier wait and around the load issue path. Store records are appended from `include/dae/pipeline/stwarp.cuh` around the writeback path and include the trailing bulk wait.
