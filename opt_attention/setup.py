from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent


setup(
    name="opt_attention",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="opt_attention._C",
            sources=[
                str(ROOT / "csrc" / "opt_attention_ext.cpp"),
                str(ROOT / "csrc" / "opt_attention_kernel.cu"),
            ],
            include_dirs=[str(ROOT / "csrc")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20", "-DNDEBUG"],
                "nvcc": [
                    "-O3",
                    "-std=c++20",
                    "-DNDEBUG",
                    "-lineinfo",
                    "-Xptxas=-v",
                    "-gencode=arch=compute_90a,code=sm_90a",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
