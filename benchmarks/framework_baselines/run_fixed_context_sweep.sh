#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 || $# > 4 )); then
  echo "usage: $0 {vllm|sglang} ENV_PYTHON MODEL [CONTEXTS]" >&2
  exit 2
fi

framework=$1
env_python=$2
model=$3
contexts=${4:-128,256,512,1024}

if [[ ${framework} != vllm && ${framework} != sglang ]]; then
  echo "framework must be vllm or sglang" >&2
  exit 2
fi
if [[ ! -x ${env_python} ]]; then
  echo "environment Python is not executable: ${env_python}" >&2
  exit 2
fi
if [[ ! -d ${model} ]]; then
  echo "model directory does not exist: ${model}" >&2
  exit 2
fi
if [[ ! -f benchmarks/blackwell_fixed_context_decode.py ]]; then
  echo "run from the vdcores-dsv4-flash repository root" >&2
  exit 2
fi

export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export HF_HOME=${HF_HOME:-/mnt/checkpoints/huggingface-cache}

IFS=, read -r -a context_values <<< "${contexts}"
for context in "${context_values[@]}"; do
  common_args=(
    benchmarks/blackwell_fixed_context_decode.py
    --framework "${framework}"
    --model "${model}"
    --contexts "${context}"
    --batch 1
    --warmups 3
    --samples 21
  )

  if [[ ${framework} == vllm ]]; then
    # The shim changes only the KV pool view's block stride. It does not copy,
    # clear, or touch padded bytes during setup or decode.
    export PYTHONPATH="${PWD}/benchmarks/vllm_tma_stride_site${PYTHONPATH:+:${PYTHONPATH}}"
    "${env_python}" "${common_args[@]}" \
      --gpu-memory-utilization 0.8 \
      --kv-cache-dtype fp8_ds_mla \
      --num-gpu-blocks-override 512 \
      --max-num-batched-tokens 127
  else
    "${env_python}" "${common_args[@]}" \
      --gpu-memory-utilization 0.98 \
      --sglang-moe-runner-backend flashinfer_mxfp4
  fi
done
