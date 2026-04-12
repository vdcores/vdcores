# OPT Attention Extension

- `opt_attention/` is a self-contained PyTorch extension for OPT decode attention experiments.
- It intentionally does not include repo headers from `include/`; local CUDA helpers live under `opt_attention/csrc/`.
- The v1 path targets Hugging Face OPT `StaticCache` decode tensors: Q `[B,H,1,128]`, K/V `[B,H,S,128]`, output `[B,1,H,128]`.
- Register the Transformers interface with `opt_attention.register()`, then use `_attn_implementation="vdcores_opt"`.
- Runtime tuning is intentionally small: `OPT_ATTENTION_SPLIT_SIZE` controls split-KV parallelism and defaults to `256`, which is best for the measured OPT-6.7B/30B decode shape `B=64,S=256`.
- For `B=64,S=256`, real OPT shape benchmarks use `H=32,hidden=4096` for OPT-6.7B and `H=56,hidden=7168` for OPT-30B; the extension is correct vs PyTorch SDPA but remains slower than SDPA at those high-head-count shapes.
