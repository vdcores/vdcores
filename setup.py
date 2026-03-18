from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import shutil

import torch
torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")

extra_link_args = [
    f"-Wl,-rpath,{torch_lib}",
]

this_dir = os.path.dirname(os.path.abspath(__file__))
sources = [os.path.join(this_dir, "src", "torch_runtime.cu")]
runtime_obj = os.path.join(this_dir, "runtime.o")
include_dirs = [
    os.path.join(this_dir, "include"),
    os.path.join(this_dir, "include", "dae"),
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
                    '-gencode=arch=compute_90a,code=sm_90a',
                    "-O3",
                    "-std=c++20",
                    "-DNDEBUG",
                    "-Xptxas=-v"
                ],
            },
            libraries=["cuda"],             # REQUIRED for cuTensorMap
            extra_link_args=extra_link_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
