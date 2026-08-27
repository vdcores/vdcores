#!/usr/bin/env bash
set -euo pipefail

# Run this script on a GB200 worker through gpu-cluster's cluster-remote. The
# configured cluster base Python already supplies its CUDA 13 PyTorch build;
# this isolated venv owns vLLM and all framework-specific dependencies.
env_dir=${1:-/mnt/checkpoints/envs/vllm-0.27.1}
python_bootstrap=${PYTHON_BOOTSTRAP:-/home/azhpcuser/miniconda3/bin/python}

"${python_bootstrap}" - <<'PY'
import sys

import torch

if sys.version_info[:3] != (3, 12, 13):
    raise RuntimeError(
        "the accepted vLLM baseline requires Python 3.12.13, found "
        f"{sys.version.split()[0]}"
    )
if torch.__version__ != "2.13.0+cu130" or torch.version.cuda != "13.0":
    raise RuntimeError(
        "the accepted vLLM baseline requires cluster base torch "
        f"2.13.0+cu130/CUDA 13.0, found {torch.__version__}/{torch.version.cuda}"
    )
PY

if [[ ! -x "${env_dir}/bin/python" ]]; then
  "${python_bootstrap}" -m venv --system-site-packages "${env_dir}"
fi

env_python="${env_dir}/bin/python"
"${env_python}" -m pip install --upgrade \
  pip==26.0.1 setuptools==80.10.2 wheel==0.46.3
"${env_python}" -m pip install \
  vllm==0.27.1 \
  flashinfer-python==0.6.16.post3 \
  transformers==5.15.0 \
  compressed-tensors==0.17.0 \
  humming-kernels==0.1.10 \
  quack-kernels==0.6.1 \
  tilelang==0.1.12 \
  tokenspeed-mla==0.1.8 \
  nvidia-cudnn-frontend==1.27.0 \
  nvidia-cutlass-dsl==4.6.0 \
  cuda-tile==1.5.0 \
  apache-tvm-ffi==0.1.11

"${env_python}" - <<'PY'
import importlib.metadata as metadata
import sys

import flashinfer
import torch
import vllm

assert sys.version_info[:3] == (3, 12, 13)
assert torch.__version__ == "2.13.0+cu130"
assert torch.version.cuda == "13.0"
assert vllm.__version__ == "0.27.1"
assert flashinfer.__version__ == "0.6.16.post3"
expected = {
    "transformers": "5.15.0",
    "compressed-tensors": "0.17.0",
    "humming-kernels": "0.1.10",
    "quack-kernels": "0.6.1",
    "tilelang": "0.1.12",
    "tokenspeed-mla": "0.1.8",
    "nvidia-cudnn-frontend": "1.27.0",
    "nvidia-cutlass-dsl": "4.6.0",
    "cuda-tile": "1.5.0",
    "apache-tvm-ffi": "0.1.11",
}
actual = {name: metadata.version(name) for name in expected}
if actual != expected:
    raise RuntimeError(f"vLLM baseline package mismatch: {actual!r}")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"vllm={vllm.__version__}")
print(f"flashinfer={flashinfer.__version__}")
PY
