from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import subprocess
import sys

import torch
torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")

extra_link_args = [
    f"-Wl,-rpath,{torch_lib}",
]

this_dir = os.path.dirname(os.path.abspath(__file__))
generated_include_dir = os.path.join(this_dir, "build", "generated")
compute_ops_registry = os.path.join(this_dir, "include", "dae", "compute_ops.inc")
selected_compute_ops = os.path.join(generated_include_dir, "dae", "selected_compute_ops.inc")
compute_ops_generator = os.path.join(this_dir, "tools", "generate_selected_compute_ops.py")
sources = [os.path.join(this_dir, "src", "torch_runtime.cu")]
runtime_obj = os.path.join(this_dir, "runtime.o")
include_dirs = [
    os.path.join(this_dir, "include"),
    os.path.join(this_dir, "include", "dae"),
    generated_include_dir,
]


def prepare_compute_ops():
    os.makedirs(os.path.dirname(selected_compute_ops), exist_ok=True)
    subprocess.check_call(
        [
            sys.executable,
            compute_ops_generator,
            "--registry",
            compute_ops_registry,
            "--output",
            selected_compute_ops,
        ],
        cwd=this_dir,
    )


def build_runtime_object():
    nvcc = os.environ.get("NVCC", "nvcc")
    subprocess.check_call(
        [
            nvcc,
            "-gencode=arch=compute_90a,code=sm_90a",
            "-O3",
            "-Iinclude/dae",
            "-Iinclude",
            f"-I{generated_include_dir}",
            "-std=c++20",
            "-Xptxas=-v",
            "-use_fast_math",
            "-lineinfo",
            "-DNDEBUG",
            "-Xcompiler",
            "-fPIC",
            "-c",
            "-o",
            runtime_obj,
            os.path.join("src", "runtime.cu"),
        ],
        cwd=this_dir,
    )


class DaeBuildExtension(BuildExtension):
    def run(self):
        prepare_compute_ops()
        build_runtime_object()
        super().run()

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
    cmdclass={"build_ext": DaeBuildExtension},
)
