from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import re

import torch
torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")

extra_link_args = [
    f"-Wl,-rpath,{torch_lib}",
]

this_dir = os.path.dirname(os.path.abspath(__file__))
generated_include_dir = os.path.join(this_dir, "build", "generated")
sources = [os.path.join(this_dir, "src", "torch_runtime.cu")]
runtime_obj = os.path.join(this_dir, "runtime.o")
cuda_arch = os.environ.get("DAE_CUDA_ARCH", "100a").lower()
for prefix in ("compute_", "sm_"):
    if cuda_arch.startswith(prefix):
        cuda_arch = cuda_arch[len(prefix):]
if re.fullmatch(r"[0-9]{2,3}[af]?", cuda_arch) is None:
    raise ValueError(
        f"Invalid DAE_CUDA_ARCH={cuda_arch!r}; expected an architecture such as 90a, 100a, or 103a"
    )
cuda_gencode = f"-gencode=arch=compute_{cuda_arch},code=sm_{cuda_arch}"
cuda_defines = []
if os.environ.get("DAE_GLOBAL_INSTRUCTIONS"):
    cuda_defines.append("-DDAE_LOAD_INSTRUCTIONS=0")
if os.environ.get("DAE_AUX_SLOTS"):
    cuda_defines.extend([
        "-DDAE_NUM_SLOTS=27",
        "-DDAE_NUM_INSTS=192",
        "-DDAE_DYNAMIC_SMEM_KB=219",
        "-DDAE_PACKED_SWAP_ATTENTION_SCRATCH=1",
    ])
if os.environ.get("DAE_NVFP4_UMMA_PIPELINE_STAGES"):
    cuda_defines.append(
        "-DDAE_NVFP4_UMMA_PIPELINE_STAGES="
        + os.environ["DAE_NVFP4_UMMA_PIPELINE_STAGES"]
    )
if os.environ.get("DAE_NVFP4_SCALE_COPY_STAGES"):
    cuda_defines.append(
        "-DDAE_NVFP4_SCALE_COPY_STAGES="
        + os.environ["DAE_NVFP4_SCALE_COPY_STAGES"]
    )
if os.environ.get("DAE_NUM_INSTS"):
    cuda_defines.append("-DDAE_NUM_INSTS=" + os.environ["DAE_NUM_INSTS"])
if os.environ.get("DAE_DYNAMIC_SMEM_KB"):
    cuda_defines.append(
        "-DDAE_DYNAMIC_SMEM_KB=" + os.environ["DAE_DYNAMIC_SMEM_KB"]
    )
include_dirs = [
    os.path.join(this_dir, "include"),
    os.path.join(this_dir, "include", "dae"),
    generated_include_dir,
]

setup(
    name="dae",

    package_dir={"": "python"},
    packages=find_packages("python"),
    ext_modules=[
        CUDAExtension(
            name = "dae.runtime",

            sources=sources,   # your .cu file
            extra_objects=[runtime_obj],    # link your runtime.o
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20", "-DNDEBUG"],
                "nvcc": [
                    cuda_gencode,
                    "-O3",
                    "-std=c++20",
                    "-DNDEBUG",
                    "-Xptxas=-v"
                ] + cuda_defines,
            },
            libraries=["cuda"],             # REQUIRED for cuTensorMap
            extra_link_args=extra_link_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
