FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace/vdcores

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:${PATH}

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Follow setup.sh: use the base conda env with CUDA 13.0.2 and CUTLASS.
RUN conda install -y -c conda-forge python cuda-toolkit=13.0.2 cutlass \
    && conda clean -afy

RUN python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
RUN python -m pip install numpy transformers accelerate

COPY . .

RUN make pyext PYTHON=python

CMD ["/bin/bash"]
