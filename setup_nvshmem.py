"""Build the optional MPI/NVSHMEM Python extension.

This setup entry point is separate from setup.py so installing the normal DAE
runtime never requires MPI or NVSHMEM.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


def find_nvshmem_home() -> Path:
    configured = os.environ.get("NVSHMEM_HOME")
    candidates = []
    if configured:
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
        if (candidate / "include" / "nvshmem.h").is_file() and (
            candidate / "lib" / "libnvshmem_host.so"
        ).exists():
            return candidate.resolve()
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(
        "Could not find NVSHMEM headers and libnvshmem_host.so. "
        f"Set NVSHMEM_HOME; searched: {searched}"
    )


def mpi_wrapper_flags(mpicxx: str, kind: str) -> list[str]:
    try:
        output = subprocess.check_output(
            [mpicxx, f"--showme:{kind}"], text=True, stderr=subprocess.STDOUT
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"{mpicxx} must be an OpenMPI-compatible C++ wrapper supporting --showme:{kind}"
        ) from error
    return shlex.split(output)


nvshmem_home = find_nvshmem_home()
mpicxx = os.environ.get("MPICXX", "mpicxx")
mpicxx = shutil.which(mpicxx) or mpicxx
os.environ["CXX"] = os.environ.get("DAE_NVSHMEM_CXX", "c++")

mpi_compile_flags = mpi_wrapper_flags(mpicxx, "compile")
mpi_include_dirs = [flag[2:] for flag in mpi_compile_flags if flag.startswith("-I")]
mpi_cxx_flags = [flag for flag in mpi_compile_flags if not flag.startswith("-I")]

from torch.utils.cpp_extension import BuildExtension, CUDAExtension  # noqa: E402


class MPIBuildExtension(BuildExtension):
    """Use nvcc for compilation and the TACC MPI wrapper only for host linking."""

    def build_extension(self, extension) -> None:
        # Modern distutils has separate C and C++ shared-object link commands.
        self.compiler.set_executable("compiler_cxx", [mpicxx])
        self.compiler.set_executable("linker_so", [mpicxx, "-shared"])
        self.compiler.set_executable("linker_so_cxx", [mpicxx, "-shared"])
        self.compiler.set_executable("linker_exe_cxx", [mpicxx])
        super().build_extension(extension)


setup(
    name="dae-nvshmem-runtime",
    version="0.0.0",
    package_dir={"": "python"},
    packages=find_packages("python"),
    ext_modules=[
        CUDAExtension(
            name="dae._nvshmem_runtime",
            sources=[str(ROOT / "src" / "torch_nvshmem_runtime.cu")],
            include_dirs=[str(nvshmem_home / "include"), *mpi_include_dirs],
            library_dirs=[str(nvshmem_home / "lib")],
            libraries=["nvshmem_host", "dl", "pthread"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20", "-DNDEBUG", *mpi_cxx_flags],
                "nvcc": [
                    "-O3",
                    "-std=c++20",
                    "-DNDEBUG",
                    "-gencode=arch=compute_90a,code=sm_90a",
                    *mpi_cxx_flags,
                ],
            },
            extra_link_args=[
                f"-Wl,-rpath,{nvshmem_home / 'lib'}",
                "-Wl,--allow-shlib-undefined",
            ],
        )
    ],
    cmdclass={"build_ext": MPIBuildExtension.with_options(use_ninja=False)},
)
