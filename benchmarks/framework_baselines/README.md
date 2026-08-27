# Fixed-context framework baselines

This directory reproduces the accepted 2026-08-27 single-GPU decode
baselines for vLLM 0.27.1 and SGLang 0.5.12.post1 on the full
`DeepSeek-V4-Flash-NVFP4` checkpoint.

## Measurement contract

- One GPU, batch size one, full 43-layer checkpoint, full 129,280-token LM
  head, BF16 model/head activations, and FP8 KV cache.
- A request has `context - 1` copies of token 791 and generates exactly two
  tokens.  The reported interval is the framework's first-to-second-token
  engine interval, so the measured decode step attends to exactly `context`
  KV tokens.  Prefill, model load, graph capture, kernel tuning, and JIT are
  outside the interval.
- Each context runs in a fresh engine process with three warmups and 21
  measured requests.  The comparison uses the median.
- vLLM may chunk the untimed prefill at contexts above 128.  No prefill chunk
  remains after the first token, so the measured first-to-second interval is
  still decode-only.  SGLang uses an unchunked prefill.
- Context one has no prompt-backed first-to-second-token equivalent in this
  harness and is reported as unavailable for the two frameworks, rather than
  replacing it with a different proxy workload.

The harness is
[`../blackwell_fixed_context_decode.py`](../blackwell_fixed_context_decode.py).
The wrapper starts a separate process for every requested context:

```bash
bash benchmarks/framework_baselines/run_fixed_context_sweep.sh \
  {vllm|sglang} ENV_PYTHON MODEL_DIR 128,256,512,1024
```

## Accepted medians

All values are milliseconds.  VDcores CUDA time is the launch-inclusive
comparison column; its in-device persistent-kernel envelope is retained as a
diagnostic.

| Context | VDcores CUDA | VDcores device | vLLM 0.27.1 | SGLang 0.5.12.post1 |
|---:|---:|---:|---:|---:|
| 1 | 4.743808 | 4.657376 | N/A | N/A |
| 128 | 5.326880 | 5.247904 | 6.874858 | 7.289749 |
| 256 | 5.406688 | 5.319904 | 7.523229 | 7.319158 |
| 512 | 5.475424 | 5.388544 | 6.870377 | 7.324534 |
| 1024 | 5.477216 | 5.400992 | 6.906794 | 7.439609 |

Against the faster framework median in each row, VDcores CUDA time is 22.52%
lower at C128, 26.13% lower at C256, 20.30% lower at C512, and 20.70% lower
at C1024.

Raw framework distributions and retained cluster job IDs:

| Framework | Context | Min | Median | P90 | Max | Job |
|:--|---:|---:|---:|---:|---:|:--|
| vLLM | 128 | 6.841032 | 6.874858 | 6.890026 | 6.926987 | `20260827T152018Z-1808408` |
| vLLM | 256 | 7.506715 | 7.523229 | 7.542493 | 7.548957 | `20260827T153439Z-2025633` |
| vLLM | 512 | 6.841128 | 6.870377 | 6.899594 | 6.906442 | `20260827T153842Z-2089144` |
| vLLM | 1024 | 6.874441 | 6.906794 | 6.937291 | 6.946219 | `20260827T154231Z-2139868` |
| SGLang | 128 | 7.260276 | 7.289749 | 7.320438 | 8.386677 | `20260827T162221Z-2740476` |
| SGLang | 256 | 7.285205 | 7.319158 | 7.344151 | 7.400313 | `20260827T164130Z-3032048` |
| SGLang | 512 | 7.285557 | 7.324534 | 7.363480 | 7.384280 | `20260827T164130Z-3032048` |
| SGLang | 1024 | 7.401977 | 7.439609 | 7.477723 | 7.504156 | `20260827T164130Z-3032048` |

VDcores jobs were `20260827T144133Z-1232596`,
`20260827T144240Z-1248228`, `20260827T144354Z-1261063`,
`20260827T144457Z-1275031`, and `20260827T144601Z-1291231` for contexts
1, 128, 256, 512, and 1024 respectively.  Those runs used ten warmups and 21
samples with token variation allowed and loopback HC fusion enabled.

### VDcores source-snapshot reproduction

Before committing the production source that accompanied the table above, the
same BF16-head/MXFP image was rebuilt from scratch with
`DAE_COMPUTE_OPS_FILE=benchmarks/deepseek_v4_resident_full_checkpoint.ops`,
`num_insts=512`, and `mxfp_direct_tma=1`.  It compiled with 244 registers, ten
barriers, a 256-byte stack, 15,104 bytes of static shared memory, and no
spills.  Each context then ran sequentially on one locked GPU with the same
token, loopback-HC, warmup, sample-count, and token-variation settings as the
accepted scan.

| Context | Reproduced CUDA | Reproduced device | CUDA delta from accepted |
|---:|---:|---:|---:|
| 1 | 4.769984 | 4.682944 | +0.55% |
| 128 | 5.331936 | 5.250176 | +0.09% |
| 256 | 5.408320 | 5.325504 | +0.03% |
| 512 | 5.408128 | 5.321568 | -1.23% |
| 1024 | 5.459808 | 5.366976 | -0.32% |

Reproduction jobs were `20260827T213311Z-3325072`,
`20260827T213416Z-3339244`, `20260827T213519Z-3357499`,
`20260827T213624Z-3375845`, and `20260827T213732Z-3393327`.  These VDcores
contexts above one retain the benchmark's deterministic seeded prefix cache;
they reproduce the production performance workload and are not a real-prompt
prefill correctness claim.

## Reproduction on the configured cluster

Run these commands from the repository root on the jump host.  The cluster
helpers discover the configured worker instead of embedding topology in this
repository.

```bash
cluster_scripts=/home/azhpcuser/.codex/skills/gpu-cluster/scripts
worker="$("${cluster_scripts}/cluster-topology" --master)"
model=/mnt/checkpoints/nvidia/DeepSeek-V4-Flash-NVFP4

"${cluster_scripts}/setup-nfs"
"${cluster_scripts}/verify-nfs"

"${cluster_scripts}/cluster-remote" "${worker}" --cwd "${PWD}" -- \
  bash benchmarks/framework_baselines/setup_vllm_0_27_1.sh
"${cluster_scripts}/cluster-remote" "${worker}" --cwd "${PWD}" -- \
  bash benchmarks/framework_baselines/setup_sglang_0_5_12_post1.sh

"${cluster_scripts}/mpi-run" -n 1 --host "${worker}" \
  --env CUDA_HOME=/usr/local/cuda \
  --env CUDA_PATH=/usr/local/cuda \
  --env HF_HOME=/mnt/checkpoints/huggingface-cache -- \
  bash benchmarks/framework_baselines/run_fixed_context_sweep.sh \
    vllm /mnt/checkpoints/envs/vllm-0.27.1/bin/python \
    "${model}" 128,256,512,1024

"${cluster_scripts}/mpi-run" -n 1 --host "${worker}" \
  --env CUDA_HOME=/usr/local/cuda \
  --env CUDA_PATH=/usr/local/cuda \
  --env HF_HOME=/mnt/checkpoints/huggingface-cache -- \
  bash benchmarks/framework_baselines/run_fixed_context_sweep.sh \
    sglang /mnt/checkpoints/envs/sglang-0.5.12.post1/bin/python \
    "${model}" 128,256,512,1024
```

The SGLang setup builds FlashMLA at revision
`15f13e5030374295491c5ce31b02d7e63a7772c6`; its first execution can also
populate substantial kernel-tuning and JIT caches.  This startup is expected
and is not part of any emitted `FIXED_CONTEXT_RESULT`.

## Accepted software stacks

| Component | vLLM environment | SGLang environment |
|:--|:--|:--|
| Python | 3.12.13 | 3.12.3 (`/usr/bin/python3`) |
| PyTorch / CUDA | 2.13.0+cu130 / 13.0 | 2.11.0+cu130 / 13.0 |
| Framework | vLLM 0.27.1 | SGLang 0.5.12.post1 |
| FlashInfer | 0.6.16.post3 | 0.6.11.post1 |
| FlashMLA | vLLM packaged path | 1.0.0+15f13e5 |
| Framework kernels | humming-kernels 0.1.10 | sglang-kernel 0.4.2.post2 |
| Transformers | 5.15.0 | 5.6.0 |
| TileLang | 0.1.12 | 0.1.8 |
| TokenSpeed MLA | 0.1.8 | 0.1.1 |

The setup scripts pin and validate the performance-critical transitive
packages in addition to the versions shown here.

### vLLM KV-pool stride compatibility

The DeepSeek-V4 FP8 KV token record is 584 bytes.  A 64-token logical page is
37,376 bytes, while the SM100 FlashMLA path requires each physical page start
to be 576-byte aligned; the nearest valid stride is 37,440 bytes.  The
benchmark-only [`../vllm_tma_stride_site/sitecustomize.py`](../vllm_tma_stride_site/sitecustomize.py)
restores that outer stride with `torch.as_strided` on vLLM's existing packed
pool storage.  It introduces no runtime allocation, copy, clear, conversion,
or access to the 64 padding bytes.  Any extra backing capacity is inert and
therefore does not add decode latency.
