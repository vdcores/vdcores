from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "app" / "tma1d_pure_cuda_bench.cu"
DEFAULT_BINARY = REPO_ROOT / ".agentlog" / "tmp" / "tma1d_pure_cuda_bench"
CUDA_ARCH = "sm_90a"


def should_rebuild(binary: Path) -> bool:
    return not binary.exists() or SOURCE.stat().st_mtime > binary.stat().st_mtime


def build(binary: Path) -> None:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise SystemExit("nvcc was not found in PATH")

    binary.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        nvcc,
        "-O3",
        "-std=c++17",
        "-allow-unsupported-compiler",
        "-ccbin",
        "/usr/bin/g++",
        f"-gencode=arch=compute_90a,code={CUDA_ARCH}",
        "-lineinfo",
        "-o",
        str(binary),
        str(SOURCE),
        "-lcuda",
    ]
    print("[build]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main(argv: list[str]) -> int:
    rebuild = False
    compile_only = False
    forward_args: list[str] = []

    for arg in argv:
        if arg == "--rebuild":
            rebuild = True
        elif arg == "--compile-only":
            compile_only = True
        else:
            forward_args.append(arg)

    binary = DEFAULT_BINARY
    if rebuild or should_rebuild(binary):
        build(binary)

    if compile_only:
        return 0

    cmd = [str(binary), *forward_args]
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
