wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh -O miniconda.sh
bash miniconda.sh -b -f

~/miniconda3/bin/conda init bash
source ~/.bashrc

conda activate

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda install -y -c conda-forge python cuda-toolkit=13.0.2 cutlass
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install numpy transformers accelerate
