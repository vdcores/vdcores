# Llama Prefill Scheduling Notes

- `app/python/llama3/prefill_sched.py` is a new 8-token prefill harness for Llama 3.1 8B at a mid-context, seeded from Hugging Face KV cache.
- The reusable attention piece is the local `SchedAttentionTokenList` helper: it maps one token/head pair to one decode-attention instance so prefill can reuse the existing attention opcode without CUDA changes.
- The full prefill path now launches and verifies at `prefix=504, chunk=8, total=512`.
- The key scheduling fixes were:
  - keep the decode-style loop shape with a one-time layer-0 `pre_attn_rms` prologue and a tail `pre_attn_rms` that signals `layerg.next("bar_pre_attn_rms")`;
  - store prefill K/V directly into the final cache window instead of using a temp-buffer flush stage;
  - modulo K-store SM indices by `kw // 64` when `KRope` writes into the cache window;
  - avoid cross-layer accumulation in the 2048-wide MLP high slices by using non-reduction stores on 32 SMs.
- The rope-sensitive checks should use interleaved rope-applied references, not raw `q_proj` / `k_proj` captures.
- Verified end-to-end accuracy on the current harness:
  - `layer0_q_rope_chunk`: about `0.227%`
  - `layer0_k_rope_cache_chunk`: about `0.253%`
  - `layer0_v_cache_chunk`: about `0.353%`
  - `final_hidden_chunk`: about `1.261%`
  - `final_logits_chunk`: about `0.547%`
- Verified benchmark on the same shape:
  - `14.415 ms` mean for the 8-token chunk
  - `1.802 ms/token`
  - about `555 tok/s`
- Current scope is intentionally narrow: `--chunk-size 8` and `prefill_len < 1024`.

## Chunk-64 Follow-up

- `app/python/llama3/prefill_sched.py` now has a 64-token GEMM-based post-attention path in addition to the original `chunk=8` harness.
- The stable chunk-64 MLP recipe today is:
  - keep `Gemm_M64N128K64` for the wide gate/up/down projections;
  - avoid the oversize `loadRMSChunk` GEMV path for the 64-token case;
  - disable grouped/prefetched GEMM load scheduling on the gate and down projections, because those stages were correct in isolation but drifted when embedded in the full layer schedule.
- A useful debugging pattern for this harness is to compare standalone GEMM probes against the scheduled path:
  - layer-0 `gate_proj`, `up_proj`, and `down_proj` GEMMs were essentially exact in isolation;
  - the large remaining errors therefore came from schedule interaction, not from the GEMM kernels themselves.
- Current chunk-64 verification status at `prefix=448, chunk=64, total=512`:
  - one-layer stage checks pass through `down_proj`;
  - the full 32-layer run still misses the current `final_hidden_chunk` threshold at about `7.76%`, though final logits are much closer (`~3.01%` on the full chunk).
- Current clean sequential chunk-64 benchmark on the same shape:
  - `43.832 ms` mean for the 64-token chunk
  - `0.685 ms/token`
  - about `1460 tok/s`
