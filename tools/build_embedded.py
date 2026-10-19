"""Build only the portable native CPU loop, without Torch or a CUDA toolkit."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIBRARY = ROOT / "build" / ("libembedded_loop.dylib" if sys.platform == "darwin" else "libembedded_loop.so")


def build(output: Path = DEFAULT_LIBRARY, *, sanitize: bool = False) -> Path:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    flags = ["-std=c++17", "-O2", "-g", "-pthread", "-Wall", "-Wextra", "-Werror"]
    flags += ["-dynamiclib"] if sys.platform == "darwin" else ["-shared", "-fPIC"]
    if sanitize:
        flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    subprocess.run(
        [*shlex.split(os.environ.get("CXX", "c++")), *flags,
         str(ROOT / "src" / "embedded_loop.cpp"), "-o", str(output)],
        check=True,
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_LIBRARY)
    args = parser.parse_args()
    print(build(args.output))
