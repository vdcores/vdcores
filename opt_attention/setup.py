import os
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
USE_TMA = os.environ.get("OPT_ATTENTION_USE_TMA", "1").lower() not in {"0", "false", "off", "no"}
USE_TMA_DEFINE = f"-DOPT_ATTENTION_USE_TMA={1 if USE_TMA else 0}"


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
            libraries=["cuda"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20", "-DNDEBUG", USE_TMA_DEFINE],
                "nvcc": [
                    "-O3",
                    "-std=c++20",
                    "-DNDEBUG",
                    USE_TMA_DEFINE,
                    "-lineinfo",
                    "-Xptxas=-v",
                    "-gencode=arch=compute_90a,code=sm_90a",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
