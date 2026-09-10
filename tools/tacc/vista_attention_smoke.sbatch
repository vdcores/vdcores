#!/bin/bash
#SBATCH --job-name=vdcores-attn
#SBATCH --partition=gh-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the VDCores repository root}"

module reset
module load gcc cuda
# Conda's CUDA activation hooks read optional variables before defining them.
set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate base
set -u

echo "Git commit: $(git rev-parse HEAD)"
echo "Host: $(hostname)"
module list 2>&1
nvcc --version
python -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA runtime:", torch.version.cuda); print("GPU:", torch.cuda.get_device_name(0))'

python -m py_compile app/python/attention_split_kv.py \
    tests/script/check_attention_split_kv.py

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    export DAE_COMPUTE_OPS="OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim_split,OP_ATTN_SPLIT_POST_REDUCE"
    make pyext
fi

python tests/script/run_with_launch_timeout.py \
    --post-launch-timeout 120 \
    --post-launch-idle-timeout 30 \
    -- python tests/script/check_attention_split_kv.py
