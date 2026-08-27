#!/usr/bin/env bash
set -euo pipefail

# Run this script on a GB200 worker through gpu-cluster's cluster-remote.
# The environment is worker-local and deliberately separate from the project
# and system Python installations.
env_dir=${1:-/mnt/checkpoints/envs/sglang-0.5.12.post1}
python_bootstrap=${PYTHON_BOOTSTRAP:-/usr/bin/python3}
flashmla_revision=15f13e5030374295491c5ce31b02d7e63a7772c6

"${python_bootstrap}" - <<'PY'
import sys

if sys.version_info[:3] != (3, 12, 3):
    raise RuntimeError(
        "the accepted SGLang baseline requires Python 3.12.3, found "
        f"{sys.version.split()[0]}"
    )
PY

if [[ ! -x "${env_dir}/bin/python" ]]; then
  "${python_bootstrap}" -m venv "${env_dir}"
fi

env_python="${env_dir}/bin/python"
"${env_python}" -m pip install --upgrade \
  pip==26.0.1 setuptools==80.10.2 wheel==0.46.3 packaging==26.0

# These are the exact aarch64 release wheels used by the accepted baseline.
# Install the framework/kernel packages first without dependencies, then let
# SGLang's own metadata supply its remaining runtime requirements.
"${env_python}" -m pip install --no-deps \
  https://files.pythonhosted.org/packages/f7/71/78f9dd95f3d80e415b984b005c9756845484b8d4dbfd2bbf3585647cf83c/sglang-0.5.12.post1-cp312-cp312-manylinux_2_34_aarch64.whl \
  https://files.pythonhosted.org/packages/3b/32/16a7421e6f486c8b4e19b561497b3de88c9e0b8899dcd9a85e1288a220c4/sglang_kernel-0.4.2.post2-cp310-abi3-manylinux2014_aarch64.whl \
  https://files.pythonhosted.org/packages/bf/7d/56d27ca2bcfee8fd28d8eb635cdc453c88e5044c264f091e2f7afc755863/flashinfer_python-0.6.11.post1-py3-none-any.whl \
  https://files.pythonhosted.org/packages/56/0c/80dad211d424c3f25199ccd9bb1913c3e2d7378b5cd3dbcd2f75a635b6dd/flashinfer_cubin-0.6.11.post1-py3-none-any.whl \
  cuda-tile==1.4.0

"${env_python}" - <<'PY'
import importlib.metadata as metadata
import subprocess
import sys

from packaging.requirements import Requirement

blocked = {
    "cuda-tile",
    "flashinfer-cubin",
    "flashinfer-python",
    "sglang",
    "sglang-kernel",
}
requirements = []
for raw_requirement in metadata.requires("sglang") or ():
    requirement = Requirement(raw_requirement)
    normalized_name = requirement.name.lower().replace("_", "-")
    if normalized_name in blocked:
        continue
    if requirement.marker is not None and not requirement.marker.evaluate():
        continue
    requirements.append(raw_requirement)

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", *requirements]
)
PY

# Pin the performance-critical transitive stack observed in the accepted
# environment.  In particular, newer kernels releases changed the
# Transformers repository API consumed by SGLang 0.5.12.
"${env_python}" -m pip install \
  torch==2.11.0 \
  triton==3.6.0 \
  transformers==5.6.0 \
  tokenizers==0.22.2 \
  flash-attn-4==4.0.0b15 \
  tilelang==0.1.8 \
  tokenspeed-mla==0.1.1 \
  tokenspeed-triton==3.8.10.post20260721 \
  sgl-deep-gemm==0.1.0 \
  quack-kernels==0.4.1 \
  cuda-toolkit==13.0.2 \
  nvidia-cutlass-dsl==4.5.1 \
  kernels==0.14.1 \
  nvidia-cudnn-frontend==1.26.0 \
  tabulate==0.10.0

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
workspace_git="${repo_root}/../git-local"
if [[ ! -x ${workspace_git} ]]; then
  echo "workspace-local git wrapper is missing: ${workspace_git}" >&2
  exit 1
fi

if "${env_python}" - <<'PY'
import importlib.metadata as metadata

try:
    installed = metadata.version("flash-mla")
except metadata.PackageNotFoundError:
    installed = None
raise SystemExit(installed != "1.0.0+15f13e5")
PY
then
  flashmla_ready=1
else
  flashmla_ready=0
fi

build_root=
cleanup() {
  case "${build_root}" in
    "${repo_root}"/.vdcores-flashmla.*) rm -rf -- "${build_root}" ;;
    "") ;;
    *) echo "refusing to remove unexpected build path: ${build_root}" >&2 ;;
  esac
}
trap cleanup EXIT

if (( ! flashmla_ready )); then
  build_root=$(mktemp -d "${repo_root}/.vdcores-flashmla.XXXXXX")
  "${workspace_git}" clone --recursive \
    https://github.com/deepseek-ai/FlashMLA.git "${build_root}/FlashMLA"
  "${workspace_git}" -C "${build_root}/FlashMLA" checkout --detach \
    "${flashmla_revision}"
  "${workspace_git}" -C "${build_root}/FlashMLA" submodule update \
    --init --recursive

  CUDA_HOME=/usr/local/cuda \
  CUDA_PATH=/usr/local/cuda \
  CPLUS_INCLUDE_PATH=/usr/local/cuda/include/cccl \
  FLASH_MLA_DISABLE_SM90=1 \
  MAX_JOBS=${MAX_JOBS:-16} \
  NVCC_THREADS=${NVCC_THREADS:-4} \
    "${env_python}" -m pip install --no-build-isolation -v \
    "${build_root}/FlashMLA"
fi

"${env_python}" - <<'PY'
import importlib.metadata as metadata
import sys

import torch
import flashinfer
import sglang
import sgl_kernel
import flash_mla

expected = {
    "sglang": "0.5.12.post1",
    "sglang-kernel": "0.4.2.post2",
    "flashinfer-python": "0.6.11.post1",
    "flash-mla": "1.0.0+15f13e5",
    "torch": "2.11.0",
    "triton": "3.6.0",
    "transformers": "5.6.0",
    "flash-attn-4": "4.0.0b15",
    "tilelang": "0.1.8",
    "tokenspeed-mla": "0.1.1",
    "sgl-deep-gemm": "0.1.0",
}
actual = {name: metadata.version(name) for name in expected}
if actual != expected:
    raise RuntimeError(f"SGLang baseline package mismatch: {actual!r}")
if sys.version_info[:3] != (3, 12, 3):
    raise RuntimeError(f"expected Python 3.12.3, found {sys.version.split()[0]}")
if torch.version.cuda != "13.0":
    raise RuntimeError(f"expected CUDA 13.0, found {torch.version.cuda}")

print("python SGLang environment validated")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"sglang={sglang.__version__}")
print(f"sglang-kernel={metadata.version('sglang-kernel')}")
print(f"flashinfer={flashinfer.__version__}")
print(f"flash-mla={metadata.version('flash-mla')}")
PY
