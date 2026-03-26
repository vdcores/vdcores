FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace/vdcores

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    python3 \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# Match the repo's documented CUDA 13.0 / PyTorch build path.
RUN pip install --index-url https://download.pytorch.org/whl/cu130 torch
RUN pip install numpy transformers accelerate

COPY . .

RUN make pyext PYTHON=python3

CMD ["/bin/bash"]
