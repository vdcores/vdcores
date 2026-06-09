from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import sys

import torch
torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")

extra_link_args = [
    f"-Wl,-rpath,{torch_lib}",
]

this_dir = os.path.dirname(os.path.abspath(__file__))
generated_include_dir = os.path.join(this_dir, "build", "generated")
sources = [os.path.join(this_dir, "src", "torch_runtime.cu")]
runtime_obj = os.path.join(this_dir, "runtime.o")
include_dirs = [
    os.path.join(this_dir, "include"),
    os.path.join(this_dir, "include", "dae"),
    generated_include_dir,
]
for prefix in (os.environ.get("CONDA_PREFIX"), sys.prefix):
    if prefix:
        include_dir = os.path.join(prefix, "include")
        if os.path.isdir(include_dir) and include_dir not in include_dirs:
            include_dirs.append(include_dir)

conda_prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
cuda_stub_dir = os.path.join(conda_prefix, "targets", "x86_64-linux", "lib", "stubs")
if not os.path.exists(os.path.join(cuda_stub_dir, "libcuda.so")):
    cuda_stub_dir = os.path.join(conda_prefix, "lib", "stubs")

library_dirs = [cuda_stub_dir]
extra_link_args.extend([
    f"-L{cuda_stub_dir}",
    f"-Wl,-rpath-link,{cuda_stub_dir}",
])

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
            library_dirs=library_dirs,
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
