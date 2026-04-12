# OPT Attention Extension

- `opt_attention/` is a self-contained PyTorch extension for OPT decode attention experiments.
- It intentionally does not include repo headers from `include/`; local CUDA helpers live under `opt_attention/csrc/`.
- The v1 path targets Hugging Face OPT `StaticCache` decode tensors: Q `[B,H,1,128]`, K/V `[B,H,S,128]`, output `[B,1,H,128]`.
- Register the Transformers interface with `opt_attention.register()`, then use `_attn_implementation="vdcores_opt"`.
- Runtime tuning is intentionally small: `OPT_ATTENTION_SPLIT_SIZE` controls split-KV parallelism and defaults to `256`, which is best for the measured OPT-6.7B/30B decode shape `B=64,S=256`.
- The fast path assumes StaticCache-style aligned K/V rows and requires `S` and `OPT_ATTENTION_SPLIT_SIZE` to be multiples of 64. These assumptions are checked on the host so the KV producer has no runtime full-tile/contiguous/alignment branch.
- The Transformers attention interface wrapper falls back to eager attention when `S` or `OPT_ATTENTION_SPLIT_SIZE` is not a multiple of 64; direct C++ `decode` calls enforce the fast-path layout with `TORCH_CHECK`.
- The default KV producer uses `cp.async.ca.shared.global` for 16-byte K/V copies and associates completion with the producer `kv_fill` barrier using `cp.async.mbarrier.arrive.shared::cta.b64`; it intentionally does not use `cp.async.commit_group` or `cp.async.wait_group`.
- Tensor TMA is the default compiled load path. Build with `OPT_ATTENTION_USE_TMA=0 python setup.py build_ext --inplace --force` to compile the cp.async specialization instead.
- The TMA path builds two 2D `CUtensorMap` descriptors for K and V over a flattened StaticCache view `[D, B*H*S]`, where physical memory is PyTorch `[B,H,S,D]` with contiguous `D`.
- At the KV tile-load site, comments document the layout as global `K/V[B,H,S,D]` with contiguous `D`, copied into shared `smem.{k,v}[stage][S_tile=64,D=128]` for one `(batch, head)` CTA tile. The TMA path uses descriptor parameters `globalDim={D=128,BHS=B*H*S}`, `globalStride[0]=D*sizeof(dtype)`, `boxDim={D=128,S_tile=64}`, `elementStrides={1,1}`, no interleave, and no swizzle.
- For `B=64,S=256`, real OPT shape benchmarks use `H=32,hidden=4096` for OPT-6.7B and `H=56,hidden=7168` for OPT-30B. The cp.async+mbarrier no-Q-producer kernel is correct vs PyTorch SDPA and measured at about `0.163 ms` for OPT-6.7B and `0.257 ms` for OPT-30B, still slower than SDPA on those high-head-count shapes.
- On the same `B=64,S=256` benchmark, the opt-in TMA path measured about `0.152 ms` for OPT-6.7B and `0.233 ms` for OPT-30B after moving the tensor-map proxy fence out of the per-tile loop. SDPA still measured faster at about `0.112 ms` and `0.165 ms`.
