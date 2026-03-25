# Mistral Small 24B Port Notes

## Model Facts

- Model path: `mistralai/Mistral-Small-24B-Instruct-2501`
- Hidden size: `5120`
- Intermediate size: `32768`
- Layers: `40`
- Query heads: `32`
- KV heads: `8`
- Head dim: `128`
- RoPE theta: `100000000.0`
- Vocab size: `131072`

## Port-Specific Scheduling Notes

- The app entrypoint is `app/python/mistral_small_24b/sched.py`.
- Mistral is Llama-like for module structure, but `QW != hidden_size` here:
  - `QW = 4096`
  - `KW = VW = 1024`
  - `attnQs` and `attnO` are shaped on `QW`, not `HIDDEN`
  - `OutProj` uses `MNK=(HIDDEN, N, QW)`
- The working MLP split is:
  - low shared-memory SiLU region: `6144`
  - fused tail: `26624`, scheduled as `8192 + 8192 + 8192 + 2048`
- For single-token decode performance, the tail MLP is sensitive to instruction ordering:
  - stage-major emission across all four tail chunks was much slower
  - chunk-major emission (`gate -> up -> silu -> down` per chunk) is the working faster shape
  - per-chunk tail SiLU barriers are materially better than making every tail down-proj wait on one shared tail barrier
- The default single-token correctness case uses input token `1` at position `0`.

## Performance Snapshot

- Initial full-model single-token benchmark on this port was about `174.7 ms`.
- Reworking the tail dependency chain and instruction ordering reduced the default full-model benchmark to about `41.26 ms`.
- `MISTRAL24B_NO_PREFETCH=down_low,down_tail` measured about `40.80 ms` in one fresh-process run, but that gain was small and not baked in by default.
- `MISTRAL24B_DOWN_ATOM=mma` was a strong regression at about `212.9 ms`.

## Runtime Support Added For This Port

- `select_rms_smem_instruction(5120)` is now supported through `OP_RMS_NORM_F16_K_5120_SMEM`.
- `CC0(...)` now supports non-power-of-two embedding row sizes by switching to `OP_CC0_ROW_BYTES`.
- `src/torch_runtime.cu` now clamps `set_cache_policy(...)` windows to `accessPolicyMaxWindowSize`, which matters for large model weights.

## Build Note

- The repo-root `dae_compute_ops.vdcore.build` file may not include the Mistral operator set.
- For a clean Mistral rebuild, use an explicit operator list such as:

```bash
DAE_COMPUTE_OPS=OP_RMS_NORM_F16_K_5120_SMEM,OP_GEMV_M64N8,OP_GEMV_M64N8_MMA,OP_ROPE_INTERLEAVE_512,OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim,OP_SILU_MUL_SHARED_BF16_K_64_SW128,OP_LOOPC,OP_ARGMAX_PARTIAL_bf16_1152_50688_132,OP_ARGMAX_REDUCE_bf16_1152_132,OP_TERMINATEC,OP_COPY,OP_SILU_MUL_SHARED_BF16_K_4096_INTER make pyext
```
