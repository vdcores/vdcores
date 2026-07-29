from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
NVSHMEM_ENABLED = os.environ.get("DAE_ENABLE_NVSHMEM", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def find_nvshmem_home() -> Path:
    candidates = []
    if configured := os.environ.get("NVSHMEM_HOME"):
        candidates.append(Path(configured))
    candidates.append(
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "nvidia"
        / "nvshmem"
    )

    for candidate in candidates:
        if (
            (candidate / "include" / "nvshmem.h").is_file()
            and (candidate / "lib" / "libnvshmem_host.so").exists()
            and (candidate / "lib" / "libnvshmem_device.a").is_file()
        ):
            return candidate.resolve()

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(
        "DAE_ENABLE_NVSHMEM is set, but NVSHMEM headers, libnvshmem_host.so, "
        "or libnvshmem_device.a were not found. "
        f"Set NVSHMEM_HOME; searched: {searched}"
    )


torch_lib = Path(torch.__file__).resolve().parent / "lib"
generated_include_dir = ROOT / "build" / "generated"
runtime_macros = [
    ("DAE_POOL_SLICE_WARPS", os.environ.get("DAE_POOL_SLICE_WARPS", "8")),
]
if NVSHMEM_ENABLED:
    runtime_macros.append(("DAE_ENABLE_NVSHMEM", "1"))
runtime_include_dirs = [
    str(ROOT / "include"),
    str(ROOT / "include" / "dae"),
    str(generated_include_dir),
]
runtime_library_dirs: list[str] = []
runtime_libraries = ["cuda"]
runtime_link_args = [f"-Wl,-rpath,{torch_lib}"]
runtime_objects = [
    os.environ.get("DAE_RUNTIME_OBJECT", str(ROOT / "runtime.o")),
]

nvshmem_home = find_nvshmem_home() if NVSHMEM_ENABLED else None
if nvshmem_home is not None:
    runtime_include_dirs.append(str(nvshmem_home / "include"))
    runtime_library_dirs.append(str(nvshmem_home / "lib"))
    runtime_libraries.extend(["nvshmem_host", "nvshmem_device", "dl", "pthread"])
    runtime_objects.append(
        os.environ.get(
            "DAE_RUNTIME_DLINK_OBJECT",
            str(ROOT / "build" / "nvshmem" / "runtime_dlink.o"),
        )
    )
    runtime_link_args.append(f"-Wl,-rpath,{nvshmem_home / 'lib'}")

extensions = [
    CUDAExtension(
        name="dae.runtime",
        sources=[str(ROOT / "src" / "torch_runtime.cu")],
        extra_objects=runtime_objects,
        include_dirs=runtime_include_dirs,
        library_dirs=runtime_library_dirs,
        define_macros=runtime_macros,
        extra_compile_args={
            "cxx": ["-O3", "-std=c++20", "-DNDEBUG"],
            "nvcc": [
                "-gencode=arch=compute_90a,code=sm_90a",
                "-O3",
                "-std=c++20",
                "-DNDEBUG",
                "-Xptxas=-v",
                *(["-diag-suppress=3012,3013"] if NVSHMEM_ENABLED else []),
            ],
        },
        libraries=runtime_libraries,
        extra_link_args=runtime_link_args,
    )
]

if NVSHMEM_ENABLED:
    assert nvshmem_home is not None
    extensions.append(
        CUDAExtension(
            name="dae._nvshmem_runtime",
            sources=[str(ROOT / "src" / "torch_nvshmem_runtime.cu")],
            include_dirs=[str(nvshmem_home / "include")],
            library_dirs=[str(nvshmem_home / "lib")],
            libraries=["nvshmem_host", "dl", "pthread"],
            define_macros=[("DAE_ENABLE_NVSHMEM", "1")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20", "-DNDEBUG"],
                "nvcc": [
                    "-gencode=arch=compute_90a,code=sm_90a",
                    "-O3",
                    "-std=c++20",
                    "-DNDEBUG",
                ],
            },
            extra_link_args=[f"-Wl,-rpath,{nvshmem_home / 'lib'}"],
        )
    )


setup(
    name="dae",
    package_dir={"": "python"},
    packages=find_packages("python"),
    ext_modules=extensions,
    extras_require={
        "nvshmem": [
            "cuda-bindings==13.0.3",
            "cuda-python==13.0.3",
            "cuda.core==0.4.0",
            "cuda.pathfinder==1.2.3",
            "mpi4py==4.1.1",
            "nvidia-nvshmem-cu13==3.4.5",
            "nvshmem4py-cu13==0.1.3",
        ]
    },
    cmdclass={"build_ext": BuildExtension},
)
