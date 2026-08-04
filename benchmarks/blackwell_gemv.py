#!/usr/bin/env python3

import os

import torch

from dae.instructions import Gemv_M64N8, Gemv_M128N8
from dae.launcher import Launcher
from dae.model import GemvLayer


def benchmark_torch(matrix: torch.Tensor, vectors: torch.Tensor, iterations: int) -> float:
    for _ in range(10):
        torch.mm(matrix, vectors.t())
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        torch.mm(matrix, vectors.t())
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1.0e3 / iterations


def main() -> None:
    device = torch.device("cuda")
    tile_m = int(os.environ.get("GEMV_TILE_M", "64"))
    atom = {64: Gemv_M64N8, 128: Gemv_M128N8}.get(tile_m)
    if atom is None:
        raise ValueError("GEMV_TILE_M must be 64 or 128")

    m = int(os.environ.get("GEMV_M", "4096"))
    k = int(os.environ.get("GEMV_K", "4096"))
    sms = int(os.environ.get("GEMV_SMS", "128"))
    iterations = int(os.environ.get("GEMV_ITERS", "50"))
    if min(m, k, sms, iterations) <= 0:
        raise ValueError("GEMV_M, GEMV_K, GEMV_SMS, and GEMV_ITERS must be positive")

    generator = torch.Generator(device=device).manual_seed(0)
    matrix = torch.rand((m, k), generator=generator, dtype=torch.bfloat16, device=device) - 0.5
    vectors = torch.rand((8, k), generator=generator, dtype=torch.bfloat16, device=device) - 0.5
    output = torch.zeros((8, m), dtype=torch.bfloat16, device=device)

    launcher = Launcher(sms, device=device)
    layer = GemvLayer(launcher, atom, "blackwell_bench", (matrix, vectors, output))
    launcher.s(layer.schedule().place(sms))

    launcher.launch()
    torch.cuda.synchronize()
    expected = matrix @ vectors.t()
    avg_diff_percent = (
        (output.t() - expected).abs().float().mean()
        / expected.abs().float().mean()
        * 100.0
    ).item()
    if avg_diff_percent > 1.0:
        raise AssertionError(f"GEMV average error {avg_diff_percent:.6f}% exceeds 1%")
    max_abs_error = (output.t() - expected).abs().max().item()

    output.zero_()
    for _ in range(3):
        launcher.launch()
    torch.cuda.synchronize()
    print(
        "GEMV_CONFIG "
        f"m={m} n=8 k={k} tile_m={tile_m} sms={sms} "
        f"fold={sms // (m // tile_m)} "
        f"avg_diff_percent={avg_diff_percent:.6f} max_abs_error={max_abs_error:.6f}"
    )
    launcher.bench(iterations)
    torch_us = benchmark_torch(matrix, vectors, iterations)
    print(f"TORCH_BF16_MATMUL average_us={torch_us:.4f}")


if __name__ == "__main__":
    main()
