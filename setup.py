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
NCCL_GIN_ENABLED = os.environ.get("DAE_ENABLE_NCCL_GIN", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
LOCAL_POOL_ENABLED = os.environ.get("DAE_ENABLE_LOCAL_POOL", "0").lower() in {
    "1", "true", "yes", "on"
}
if sum((NVSHMEM_ENABLED, NCCL_GIN_ENABLED, LOCAL_POOL_ENABLED)) > 1:
    raise RuntimeError("PoolInst transport backends are compile-time exclusive")
CUDA_ARCH = os.environ.get("DAE_CUDA_ARCH", "90a")
POOL_DATA_PATH = os.environ.get("DAE_POOL_DATA_PATH", "nvshmem")


def cuda_gencode(arch: str) -> str:
    if not arch.replace("a", "").isdigit() or arch.count("a") > 1:
        raise RuntimeError(f"Invalid DAE_CUDA_ARCH={arch!r}; expected e.g. 90a or 100a")
    return f"-gencode=arch=compute_{arch},code=sm_{arch}"


def find_nvshmem_home() -> tuple[Path, Path]:
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
            and (candidate / "lib" / "libnvshmem_device.a").is_file()
        ):
            host_candidates = sorted((candidate / "lib").glob("libnvshmem_host.so*"))
            if host_candidates:
                return candidate.resolve(), host_candidates[0].resolve()

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(
        "DAE_ENABLE_NVSHMEM is set, but NVSHMEM headers, libnvshmem_host.so, "
        "or libnvshmem_device.a were not found. "
        f"Set NVSHMEM_HOME; searched: {searched}"
    )


def find_nccl_home() -> Path:
    candidates = []
    if configured := os.environ.get("NCCL_HOME"):
        candidates.append(Path(configured))
    candidates.append(
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "nvidia"
        / "nccl"
    )
    for candidate in candidates:
        if (
            (candidate / "include" / "nccl_device.h").is_file()
            and (candidate / "lib" / "libnccl.so.2").exists()
        ):
            return candidate.resolve()
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(
        "DAE_ENABLE_NCCL_GIN is set, but NCCL device headers or libnccl.so.2 "
        f"were not found. Set NCCL_HOME; searched: {searched}"
    )


torch_lib = Path(torch.__file__).resolve().parent / "lib"
generated_include_dir = ROOT / "build" / "generated"
runtime_macros = [
    ("DAE_POOL_SLICE_WARPS", os.environ.get("DAE_POOL_SLICE_WARPS", "8")),
    (
        "DAE_POOL_SLICE_WARP_QP_COMPLETION",
        os.environ.get("DAE_POOL_SLICE_WARP_QP_COMPLETION", "0"),
    ),
    ("DAE_POOL_SLICE_RAW_SGL", os.environ.get("DAE_POOL_SLICE_RAW_SGL", "0")),
    (
        "DAE_POOL_SLICE_RAW_SGL_WIDTH",
        os.environ.get("DAE_POOL_SLICE_RAW_SGL_WIDTH", "8"),
    ),
    (
        "DAE_POOL_LOCAL_DIRECT_SCATTER",
        os.environ.get("DAE_POOL_LOCAL_DIRECT_SCATTER", "0"),
    ),
    (
        "DAE_POOL_MULTIMEM_VECTOR_ILP",
        os.environ.get("DAE_POOL_MULTIMEM_VECTOR_ILP", "4"),
    ),
]
if NVSHMEM_ENABLED:
    runtime_macros.append(("DAE_ENABLE_NVSHMEM", "1"))
if NCCL_GIN_ENABLED:
    runtime_macros.append(("DAE_ENABLE_NCCL_GIN", "1"))
if LOCAL_POOL_ENABLED:
    runtime_macros.append(("DAE_ENABLE_LOCAL_POOL", "1"))
if POOL_DATA_PATH in {"nvlink", "local"}:
    runtime_macros.append(("DAE_POOL_DATA_PATH_NVLINK", "1"))
elif POOL_DATA_PATH != "nvshmem":
    raise RuntimeError("DAE_POOL_DATA_PATH must be nvshmem, nvlink, or local")
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

nvshmem_location = find_nvshmem_home() if NVSHMEM_ENABLED else None
nvshmem_home = nvshmem_location[0] if nvshmem_location is not None else None
nvshmem_host_library = nvshmem_location[1] if nvshmem_location is not None else None
if nvshmem_home is not None:
    runtime_include_dirs.append(str(nvshmem_home / "include"))
    runtime_library_dirs.append(str(nvshmem_home / "lib"))
    runtime_libraries.extend(["nvshmem_device", "dl", "pthread"])
    runtime_link_args.append(str(nvshmem_host_library))
    runtime_objects.append(
        os.environ.get(
            "DAE_RUNTIME_DLINK_OBJECT",
            str(ROOT / "build" / "nvshmem" / "runtime_dlink.o"),
        )
    )
    runtime_link_args.append(f"-Wl,-rpath,{nvshmem_home / 'lib'}")

nccl_home = find_nccl_home() if NCCL_GIN_ENABLED else None
if nccl_home is not None:
    runtime_include_dirs.append(str(nccl_home / "include"))
    runtime_library_dirs.append(str(nccl_home / "lib"))
    # GIN device operations are header-inlined. NCCL4Py owns all host API
    # calls and loads the pinned versioned libnccl.so.2; the VDCores extension
    # therefore has no unversioned -lnccl link dependency.
    runtime_link_args.append(f"-Wl,-rpath,{nccl_home / 'lib'}")

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
                cuda_gencode(CUDA_ARCH),
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

if CUDA_ARCH == "100a" and LOCAL_POOL_ENABLED:
    extensions.append(
        CUDAExtension(
            name="dae._local_pool_runtime",
            sources=[str(ROOT / "src" / "torch_local_pool_runtime.cpp")],
            libraries=["cuda"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20", "-DNDEBUG"],
                "nvcc": [cuda_gencode(CUDA_ARCH), "-O3", "-std=c++20"],
            },
        )
    )

if NVSHMEM_ENABLED:
    assert nvshmem_home is not None
    extensions.append(
        CUDAExtension(
            name="dae._nvshmem_runtime",
            sources=[str(ROOT / "src" / "torch_nvshmem_runtime.cu")],
            include_dirs=[str(nvshmem_home / "include")],
            library_dirs=[str(nvshmem_home / "lib")],
            libraries=["dl", "pthread"],
            define_macros=[("DAE_ENABLE_NVSHMEM", "1")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20", "-DNDEBUG"],
                "nvcc": [
                    cuda_gencode(CUDA_ARCH),
                    "-O3",
                    "-std=c++20",
                    "-DNDEBUG",
                ],
            },
            extra_link_args=[
                str(nvshmem_host_library),
                f"-Wl,-rpath,{nvshmem_home / 'lib'}",
            ],
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
