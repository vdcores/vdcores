from pathlib import Path

from torch.utils.cpp_extension import load


def load_extension(verbose: bool = False):
    root = Path(__file__).resolve().parents[1]
    return load(
        name="opt_attention_dev",
        sources=[
            str(root / "csrc" / "opt_attention_ext.cpp"),
            str(root / "csrc" / "opt_attention_kernel.cu"),
        ],
        extra_include_paths=[str(root / "csrc")],
        extra_cflags=["-O3", "-std=c++20", "-DNDEBUG"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++20",
            "-DNDEBUG",
            "-lineinfo",
            "-Xptxas=-v",
            "-gencode=arch=compute_90a,code=sm_90a",
        ],
        verbose=verbose,
    )
