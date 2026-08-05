#!/usr/bin/env python3

import os
from collections.abc import Sequence

import torch

from dae.instructions import Gemv_M64N8, Gemv_M128N8, TmaTensor
from dae.launcher import Launcher
from dae.model import GemvLayer
from dae.schedule import SchedGemv
from dae.tma_utils import (
    Major,
    build_tma_wgmma_mnmajor_m128n8,
    cord_func_m128n8_output,
    pack_weight_tile_major,
)


def benchmark_torch(
    matrices: Sequence[torch.Tensor], vectors: torch.Tensor, iterations: int
) -> float:
    for _ in range(10):
        for matrix in matrices:
            torch.mm(matrix, vectors.t())
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        for matrix in matrices:
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
    split_m = int(os.environ.get("GEMV_SPLIT_M", "1"))
    epochs = int(os.environ.get("GEMV_EPOCHS", "1"))
    tile_packed = os.environ.get("GEMV_TILE_PACKED", "0") == "1"
    iterations = int(os.environ.get("GEMV_ITERS", "50"))
    if min(m, k, sms, split_m, epochs, iterations) <= 0:
        raise ValueError(
            "GEMV_M, GEMV_K, GEMV_SMS, GEMV_SPLIT_M, GEMV_EPOCHS, and "
            "GEMV_ITERS must be positive"
        )
    if m % split_m or (m // split_m) % tile_m:
        raise ValueError("each M split must contain a whole number of M tiles")
    m_tiles_per_split = (m // split_m) // tile_m
    if sms % m_tiles_per_split:
        raise ValueError("GEMV_SMS must be a multiple of the M tiles per split")
    fold = sms // m_tiles_per_split

    generator = torch.Generator(device=device).manual_seed(0)
    matrices = [
        torch.rand(
            (m, k), generator=generator, dtype=torch.bfloat16, device=device
        )
        - 0.5
        for _ in range(epochs)
    ]
    vectors = (
        torch.rand(
            (8, k), generator=generator, dtype=torch.bfloat16, device=device
        )
        - 0.5
    )
    outputs = [
        torch.zeros((8, m), dtype=torch.bfloat16, device=device)
        for _ in range(epochs)
    ]

    launcher = Launcher(sms, device=device)
    schedules = []
    if tile_packed:
        load_vectors = TmaTensor(launcher, vectors).wgmma_load(
            atom.MNK[1], atom.MNK[2] * atom.n_batch, Major.K
        )
        packed_matrices = [
            pack_weight_tile_major(matrix, atom.MNK[0], atom.MNK[2])
            for matrix in matrices
        ]
        for matrix, output in zip(packed_matrices, outputs):
            load_matrix = TmaTensor(launcher, matrix).wgmma_load_tiled(
                atom.MNK[0], atom.MNK[2]
            )
            output_mode = "reduce" if fold > 1 else "store"
            output_tensor = TmaTensor(launcher, output)
            if tile_m == 128:
                store_output = output_tensor._build(
                    output_mode,
                    atom.MNK[0],
                    atom.MNK[1],
                    build_tma_wgmma_mnmajor_m128n8,
                    cord_func_m128n8_output,
                )
            else:
                store_output = output_tensor.wgmma(
                    output_mode, atom.MNK[1], atom.MNK[0], Major.MN
                )
            schedule = SchedGemv(
                atom,
                (m, vectors.shape[0], k),
                (load_matrix, load_vectors, store_output),
            )
            if split_m > 1:
                schedule = schedule.split_M(split_m)
            schedules.append(schedule.place(sms))
    else:
        for epoch, (matrix, output) in enumerate(zip(matrices, outputs)):
            layer = GemvLayer(
                launcher,
                atom,
                f"blackwell_bench_{epoch}",
                (matrix, vectors, output),
            )
            schedule = layer.schedule()
            if split_m > 1:
                schedule = schedule.split_M(split_m)
            schedules.append(schedule.place(sms))
    launcher.s(*schedules)

    launcher.launch()
    torch.cuda.synchronize()
    expected = [matrix @ vectors.t() for matrix in matrices]
    avg_diff_percent = max(
        (
            (output.t() - reference).abs().float().mean()
            / reference.abs().float().mean()
            * 100.0
        ).item()
        for output, reference in zip(outputs, expected)
    )
    if avg_diff_percent > 1.0:
        raise AssertionError(f"GEMV average error {avg_diff_percent:.6f}% exceeds 1%")
    max_abs_error = max(
        (output.t() - reference).abs().max().item()
        for output, reference in zip(outputs, expected)
    )

    for output in outputs:
        output.zero_()
    for _ in range(3):
        launcher.launch()
    torch.cuda.synchronize()
    print(
        "GEMV_CONFIG "
        f"m={m * epochs} epoch_m={m} epochs={epochs} n=8 k={k} "
        f"tile_m={tile_m} tile_packed={int(tile_packed)} sms={sms} split_m={split_m} "
        f"fold={fold} "
        f"avg_diff_percent={avg_diff_percent:.6f} max_abs_error={max_abs_error:.6f}"
    )
    launcher.bench(iterations)
    torch_us = benchmark_torch(matrices, vectors, iterations)
    print(f"TORCH_BF16_MATMUL average_us={torch_us:.4f}")


if __name__ == "__main__":
    main()
