#!/usr/bin/env python3

import os
from collections.abc import Sequence
from functools import partial

import torch

from dae.instructions import (
    Gemv_M64N8,
    Gemv_M64N8IssuerOnly,
    Gemv_M64N8_ROPE_128,
    Gemv_M128N8,
    Gemv_M128N8Direct4,
    Gemv_M128N8Group4B2,
    Gemv_M128N8Group4B3,
    Gemv_M128N8Group4B4,
    Gemv_M128N8Group4B7,
    RawAddress,
    RegLoad,
    RegStore,
    ROPE_INTERLEAVE_512,
    TmaTensor,
)
from dae.launcher import Launcher
from dae.model import GemvLayer
from dae.schedule import (
    SchedGemv,
    SchedGemvMGroup,
    SchedGemvMGroupReduce,
    SchedGemvRope,
    SchedRope,
)
from dae.tma_utils import (
    Major,
    ToRopeTableCordAdapter,
    ToSplitMCordAdapter,
    build_tma_wgmma_mnmajor_m128n8,
    build_tma_wgmma_mnmajor_m128n8_grouped,
    cord_func_m128n8_grouped_output,
    cord_func_m128n8_output,
    cord_load_tbl,
    pack_weight_tile_major,
    tma_load_tbl,
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
    issuer_only = os.environ.get("GEMV_ISSUER_ONLY", "0") == "1"
    grouped_output = os.environ.get("GEMV_GROUPED_OUTPUT", "0") == "1"
    grouped_reduce = int(os.environ.get("GEMV_GROUPED_REDUCE", "0"))
    down_schedule = os.environ.get("GEMV_DOWN_SCHEDULE", "none").strip().lower()
    fused_rope = os.environ.get("GEMV_FUSED_ROPE", "0") == "1"
    two_op_rope = os.environ.get("GEMV_TWO_OP_ROPE", "0") == "1"
    m = int(os.environ.get("GEMV_M", "4096"))
    k = int(os.environ.get("GEMV_K", "4096"))
    sms = int(os.environ.get("GEMV_SMS", "128"))
    split_m = int(os.environ.get("GEMV_SPLIT_M", "1"))
    epochs = int(os.environ.get("GEMV_EPOCHS", "1"))
    rope_position = int(os.environ.get("GEMV_ROPE_POSITION", "0"))
    tile_packed = os.environ.get("GEMV_TILE_PACKED", "0") == "1"
    iterations = int(os.environ.get("GEMV_ITERS", "50"))
    if down_schedule not in ("none", "legacy", "balanced"):
        raise ValueError("GEMV_DOWN_SCHEDULE must be none, legacy, or balanced")
    if sum(
        (
            bool(grouped_output),
            bool(grouped_reduce),
            down_schedule != "none",
            fused_rope,
            two_op_rope,
        )
    ) > 1:
        raise ValueError(
            "direct grouped output, grouped reduction, down scheduling, "
            "fused RoPE, and two-op RoPE are exclusive"
        )
    if fused_rope:
        atom = Gemv_M64N8_ROPE_128
    elif grouped_output:
        atom = Gemv_M128N8Direct4
    elif grouped_reduce:
        atom = {
            (4, 4096, 128): Gemv_M128N8Group4B2,
            (4, 6144, 128): Gemv_M128N8Group4B3,
            (4, 8192, 128): Gemv_M128N8Group4B4,
            (4, 14336, 128): Gemv_M128N8Group4B7,
        }.get((grouped_reduce, k, sms))
    else:
        if tile_m == 64 and issuer_only:
            atom = Gemv_M64N8IssuerOnly
        else:
            atom = {64: Gemv_M64N8, 128: Gemv_M128N8}.get(tile_m)
    if atom is None or ((grouped_output or grouped_reduce) and tile_m != 128):
        raise ValueError("unsupported GEMV tile/group/K configuration")
    if fused_rope and (tile_m != 64 or not tile_packed or split_m != 1):
        raise ValueError(
            "GEMV_FUSED_ROPE requires GEMV_TILE_M=64, packed weights, and "
            "GEMV_SPLIT_M=1"
        )
    if two_op_rope and (tile_m != 64 or not tile_packed or split_m != 1):
        raise ValueError(
            "GEMV_TWO_OP_ROPE requires GEMV_TILE_M=64, packed weights, and "
            "GEMV_SPLIT_M=1"
        )
    if (fused_rope or two_op_rope) and m % 128:
        raise ValueError("projection + RoPE requires M to contain whole D128 heads")
    if down_schedule != "none" and (
        (m, k, sms, tile_m, split_m, epochs) != (4096, 14336, 152, 64, 1, 1)
        or not tile_packed
    ):
        raise ValueError(
            "GEMV_DOWN_SCHEDULE requires packed M4096xN8xK14336, "
            "152 SMs, M64, one M split, and one epoch"
        )
    if min(m, k, sms, split_m, epochs, iterations) <= 0 or rope_position < 0:
        raise ValueError(
            "GEMV_M, GEMV_K, GEMV_SMS, GEMV_SPLIT_M, GEMV_EPOCHS, and "
            "GEMV_ITERS must be positive; GEMV_ROPE_POSITION must be nonnegative"
        )
    if m % split_m or (m // split_m) % tile_m:
        raise ValueError("each M split must contain a whole number of M tiles")
    m_tiles_per_split = (m // split_m) // tile_m
    if down_schedule != "none":
        fold = "2+2" if down_schedule == "legacy" else "3+4"
    elif grouped_output:
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
    rope_rows = (
        torch.rand(
            (rope_position + 1, 128),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        - 0.5
    )
    if os.environ.get("GEMV_ROPE_IDENTITY", "0") == "1":
        rope_rows.zero_()
        rope_rows[:, 0::2] = 1
    rope_row = rope_rows[rope_position]
    rope_table = rope_rows[:, None, :].expand(-1, 8, -1).contiguous()

    launcher = Launcher(sms, device=device)
    schedules = []
    rope_table_tma = None
    if two_op_rope:
        rope_table_tma = TmaTensor(launcher, rope_table)._build(
            "load", 64, 8, tma_load_tbl, cord_load_tbl
        )
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
            if down_schedule != "none":
                store_output = TmaTensor(launcher, output).wgmma(
                    "reduce", atom.MNK[1], atom.MNK[0], Major.MN
                )
                if down_schedule == "legacy":
                    partitions = (
                        (((0, 768), vectors.shape[0], 6144), 2, 24, 128),
                        (((768, 3328), vectors.shape[0], 6144), 2, 104, 0),
                        (((0, 768), vectors.shape[0], (6144, 8192)), 2, 24, 104),
                        (((768, 3328), vectors.shape[0], (6144, 8192)), 2, 104, 0),
                    )
                else:
                    partitions = (
                        (((0, 3072), vectors.shape[0], 6144), 3, 144, 0),
                        (((3072, 1024), vectors.shape[0], 6144), 3, 48, 104),
                        (((0, 2432), vectors.shape[0], (6144, 8192)), 4, 152, 0),
                        (((2432, 1664), vectors.shape[0], (6144, 8192)), 4, 104, 0),
                    )
                schedules.extend(
                    SchedGemv(
                        atom,
                        partition_mnk,
                        (load_matrix, load_vectors, store_output),
                        fold=partition_fold,
                    ).place(partition_sms, base_sm=base_sm)
                    for partition_mnk, partition_fold, partition_sms, base_sm
                    in partitions
                )
                continue
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
            if fused_rope:
                schedule = SchedGemvRope(
                    (m, vectors.shape[0], k),
                    (load_matrix, load_vectors, store_output),
                    RawAddress(rope_rows, 32),
                    hist_seq_len=rope_position,
                )
            elif two_op_rope:
                reg_store = RegStore(
                    0, size=atom.MNK[0] * atom.MNK[1] * output.element_size()
                )
                schedules.append(
                    SchedGemv(
                        atom,
                        (m, vectors.shape[0], k),
                        (load_matrix, load_vectors, reg_store),
                    ).place(sms)
                )
                schedules.append(
                    SchedRope(
                        ROPE_INTERLEAVE_512,
                        (
                            ToRopeTableCordAdapter(rope_table_tma, rope_position),
                            RegLoad(0),
                            ToSplitMCordAdapter(
                                store_output, m // atom.MNK[0], atom.MNK[0]
                            ),
                        ),
                    ).place(sms)
                )
                continue
            else:
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
    if fused_rope or two_op_rope:
        cosine = rope_row[0::2].float()[None, :, None]
        sine = rope_row[1::2].float()[None, :, None]
        rotated = []
        for reference in expected:
            heads = reference.view(m // 128, 128, vectors.shape[0]).float()
            even = heads[:, 0::2]
            odd = heads[:, 1::2]
            result = torch.empty_like(heads)
            result[:, 0::2] = even * cosine - odd * sine
            result[:, 1::2] = even * sine + odd * cosine
            rotated.append(result.to(torch.bfloat16).view(m, vectors.shape[0]))
        expected = rotated
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
        f"issuer_only={int(issuer_only)} "
        f"split_m={split_m} fold={fold} grouped_output={int(grouped_output)} "
        f"grouped_reduce={grouped_reduce} down_schedule={down_schedule} "
        f"fused_rope={int(fused_rope)} "
        f"two_op_rope={int(two_op_rope)} rope_position={rope_position} "
        f"avg_diff_percent={avg_diff_percent:.6f} max_abs_error={max_abs_error:.6f}"
    )
    launcher.bench(iterations)
    torch_us = benchmark_torch(matrices, vectors, iterations)
    print(f"TORCH_BF16_MATMUL average_us={torch_us:.4f}")


if __name__ == "__main__":
    main()
