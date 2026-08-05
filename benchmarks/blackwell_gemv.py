#!/usr/bin/env python3

import os
from collections.abc import Sequence
from functools import partial

import torch

from dae.instructions import (
    Gemv_M64N8,
    Gemv_M128N8,
    Gemv_M128N8Direct4,
    Gemv_M128N8Group4B2,
    Gemv_M128N8Group4B3,
    Gemv_M128N8Group4B4,
    Gemv_M128N8Group4B7,
    TmaTensor,
)
from dae.launcher import Launcher
from dae.model import GemvLayer
from dae.schedule import SchedGemv, SchedGemvMGroup, SchedGemvMGroupReduce
from dae.tma_utils import (
    Major,
    build_tma_wgmma_mnmajor_m128n8,
    build_tma_wgmma_mnmajor_m128n8_grouped,
    cord_func_m128n8_output,
    cord_func_m128n8_grouped_output,
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
    grouped_output = os.environ.get("GEMV_GROUPED_OUTPUT", "0") == "1"
    grouped_reduce = int(os.environ.get("GEMV_GROUPED_REDUCE", "0"))
    m = int(os.environ.get("GEMV_M", "4096"))
    k = int(os.environ.get("GEMV_K", "4096"))
    sms = int(os.environ.get("GEMV_SMS", "128"))
    split_m = int(os.environ.get("GEMV_SPLIT_M", "1"))
    epochs = int(os.environ.get("GEMV_EPOCHS", "1"))
    tile_packed = os.environ.get("GEMV_TILE_PACKED", "0") == "1"
    iterations = int(os.environ.get("GEMV_ITERS", "50"))
    if grouped_output and grouped_reduce:
        raise ValueError("direct grouped output and grouped reduction are exclusive")
    if grouped_output:
        atom = Gemv_M128N8Direct4
    elif grouped_reduce:
        atom = {
            (4, 4096, 128): Gemv_M128N8Group4B2,
            (4, 6144, 128): Gemv_M128N8Group4B3,
            (4, 8192, 128): Gemv_M128N8Group4B4,
            (4, 14336, 128): Gemv_M128N8Group4B7,
        }.get((grouped_reduce, k, sms))
    else:
        atom = {64: Gemv_M64N8, 128: Gemv_M128N8}.get(tile_m)
    if atom is None or ((grouped_output or grouped_reduce) and tile_m != 128):
        raise ValueError("unsupported GEMV tile/group/K configuration")
    if min(m, k, sms, split_m, epochs, iterations) <= 0:
        raise ValueError(
            "GEMV_M, GEMV_K, GEMV_SMS, GEMV_SPLIT_M, GEMV_EPOCHS, and "
            "GEMV_ITERS must be positive"
        )
    if m % split_m or (m // split_m) % tile_m:
        raise ValueError("each M split must contain a whole number of M tiles")
    m_tiles_per_split = (m // split_m) // tile_m
    if grouped_output:
        if not tile_packed or split_m != 1:
            raise ValueError(
                "GEMV_GROUPED_OUTPUT requires packed weights and GEMV_SPLIT_M=1"
            )
        if m != sms * tile_m * atom.output_groups:
            raise ValueError(
                "grouped output requires M = SMS * TILE_M * output_groups"
            )
        fold = 1
    elif grouped_reduce:
        if not tile_packed or split_m != 1:
            raise ValueError(
                "GEMV_GROUPED_REDUCE requires packed weights and GEMV_SPLIT_M=1"
            )
        if m % (tile_m * atom.output_groups):
            raise ValueError("M must contain complete grouped M128 outputs")
        m_groups = m // (tile_m * atom.output_groups)
        if sms % m_groups:
            raise ValueError("GEMV_SMS must be divisible by grouped M tiles")
        fold = sms // m_groups
        if k % fold or (k // fold) % (atom.MNK[2] * atom.n_batch):
            raise ValueError("K fold is incompatible with grouped B-load interval")
    else:
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
            if grouped_output:
                schedules.append(
                    SchedGemvMGroup(
                        atom,
                        (m, vectors.shape[0], k),
                        (load_matrix, load_vectors),
                        output,
                    ).place(sms)
                )
                continue

            output_mode = "reduce" if fold > 1 else "store"
            output_tensor = TmaTensor(launcher, output)
            if grouped_reduce:
                store_output = output_tensor._build(
                    output_mode,
                    atom.MNK[0] * atom.output_groups,
                    atom.MNK[1],
                    partial(
                        build_tma_wgmma_mnmajor_m128n8_grouped,
                        output_groups=atom.output_groups,
                    ),
                    partial(
                        cord_func_m128n8_grouped_output,
                        output_groups=atom.output_groups,
                    ),
                )
            elif tile_m == 128:
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
            if grouped_reduce:
                schedules.append(
                    SchedGemvMGroupReduce(
                        atom,
                        (m, vectors.shape[0], k),
                        (load_matrix, load_vectors, store_output),
                        group=False,
                    ).place(sms)
                )
                continue
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
        f"tile_m={tile_m} tile_packed={int(tile_packed)} sms={sms} "
        f"split_m={split_m} fold={fold} grouped_output={int(grouped_output)} "
        f"grouped_reduce={grouped_reduce} "
        f"avg_diff_percent={avg_diff_percent:.6f} max_abs_error={max_abs_error:.6f}"
    )
    launcher.bench(iterations)
    torch_us = benchmark_torch(matrices, vectors, iterations)
    print(f"TORCH_BF16_MATMUL average_us={torch_us:.4f}")


if __name__ == "__main__":
    main()
