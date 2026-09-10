#!/bin/bash
#SBATCH --job-name=vdcores-setup
#SBATCH --partition=gh-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the VDCores repository root}"

module reset
module load gcc cuda

if [[ ! -x "$HOME/miniconda3/bin/conda" ]]; then
    installer="${TMPDIR:-/tmp}/Miniconda3-${SLURM_JOB_ID}-$(uname -m).sh"
    wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh" \
        -O "$installer"
    bash "$installer" -b -p "$HOME/miniconda3"
fi

# Conda's CUDA activation hooks read optional variables before defining them.
set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate base

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda install -y -c conda-forge python cuda-toolkit=13.0.2 cutlass
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install numpy transformers accelerate
set -u

echo "Git commit: $(git rev-parse HEAD)"
echo "Host: $(hostname)"
module list 2>&1
nvcc --version
python -c 'import platform, torch; print("Python:", platform.python_version()); print("PyTorch:", torch.__version__); print("CUDA runtime:", torch.version.cuda); print("GPU:", torch.cuda.get_device_name(0))'
