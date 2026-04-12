# OPT Attention Extension

- `opt_attention/` is a self-contained PyTorch extension for OPT decode attention experiments.
- It intentionally does not include repo headers from `include/`; local CUDA helpers live under `opt_attention/csrc/`.
- The v1 path targets Hugging Face OPT `StaticCache` decode tensors: Q `[B,H,1,128]`, K/V `[B,H,S,128]`, output `[B,1,H,128]`.
- Register the Transformers interface with `opt_attention.register()`, then use `_attn_implementation="vdcores_opt"`.
- Runtime tuning is intentionally small: `OPT_ATTENTION_SPLIT_SIZE` controls split-KV parallelism and defaults to `256`, which is best for the measured OPT-6.7B/30B decode shape `B=64,S=256`.
- The fast path assumes StaticCache-style aligned K/V rows and requires `S` and `OPT_ATTENTION_SPLIT_SIZE` to be multiples of 64. These assumptions are checked on the host so the KV producer has no runtime full-tile/contiguous/alignment branch.
- The Transformers attention interface wrapper falls back to eager attention when `S` or `OPT_ATTENTION_SPLIT_SIZE` is not a multiple of 64; direct C++ `decode` calls enforce the fast-path layout with `TORCH_CHECK`.
- The KV producer uses `cp.async.ca.shared.global` for 16-byte K/V copies and associates completion with the producer `kv_fill` barrier using `cp.async.mbarrier.arrive.shared::cta.b64`; it intentionally does not use `cp.async.commit_group` or `cp.async.wait_group`.
- At the KV tile-load site, comments document the layout as global `K/V[B,H,S,D]` with contiguous `D`, copied into shared `smem.{k,v}[stage][S_tile=64,D=128]` for one `(batch, head)` CTA tile.
- For `B=64,S=256`, real OPT shape benchmarks use `H=32,hidden=4096` for OPT-6.7B and `H=56,hidden=7168` for OPT-30B. The cp.async+mbarrier no-Q-producer kernel is correct vs PyTorch SDPA and measured at about `0.163 ms` for OPT-6.7B and `0.257 ms` for OPT-30B, still slower than SDPA on those high-head-count shapes.
