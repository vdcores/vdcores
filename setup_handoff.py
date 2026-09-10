from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="dae-handoff",
    package_dir={"": "python"},
    packages=find_packages("python"),
    ext_modules=[
        CUDAExtension(
            name="dae.handoff_runtime",
            sources=["src/torch_handoff.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20", "-DNDEBUG"],
                "nvcc": ["-O3", "-std=c++20", "-DNDEBUG", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
