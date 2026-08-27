#!/usr/bin/env python3

"""Isolated BF16 grouped-UMMA + fused-argmax correctness benchmark."""

import os

import torch

from dae.instructions import (
    ARGMAX_REDUCE_GLOBAL_bf16_256,
    Gemv_M128N8Argmax4,
    TmaTensor,
)
from dae.launcher import Launcher
from dae.schedule import SchedArgmaxReduceGlobal, SchedGemvMGroupArgmax
from dae.tma_utils import Major, pack_weight_tile_major


def main() -> None:
    device = torch.device("cuda")
    sms = 128
    epoch_m = sms * Gemv_M128N8Argmax4.MNK[0] * (
        Gemv_M128N8Argmax4.output_groups
    )
    k = int(os.environ.get("GEMV_K", "4096"))
    iterations = int(os.environ.get("GEMV_ITERS", "5"))
    if k % (
        Gemv_M128N8Argmax4.MNK[2] * Gemv_M128N8Argmax4.n_batch
    ):
        raise ValueError("GEMV_K must be a multiple of 512")

    generator = torch.Generator(device=device).manual_seed(0)
    matrices = [
        torch.rand(
            (epoch_m, k),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        - 0.5
        for _ in range(2)
    ]
    vectors = (
        torch.rand(
            (8, k),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        - 0.5
    )
    partials = torch.empty(
        (8, 2 * sms, 16), dtype=torch.uint8, device=device
    )
    output_tokens = torch.full((8,), -1, dtype=torch.int64, device=device)

    launcher = Launcher(sms, device=device)
    load_vectors = TmaTensor(launcher, vectors).wgmma_load(
        Gemv_M128N8Argmax4.MNK[1],
        Gemv_M128N8Argmax4.MNK[2] * Gemv_M128N8Argmax4.n_batch,
        Major.K,
    )
    partial_barrier = launcher.new_bar(2 * sms)
    schedules = []
    packed_matrices = []
    for epoch, matrix in enumerate(matrices):
        packed = pack_weight_tile_major(
            matrix,
            Gemv_M128N8Argmax4.MNK[0],
            Gemv_M128N8Argmax4.MNK[2],
        )
        packed_matrices.append(packed)
        load_matrix = TmaTensor(launcher, packed).wgmma_load_tiled(
            Gemv_M128N8Argmax4.MNK[0],
            Gemv_M128N8Argmax4.MNK[2],
        )
        schedules.append(
            SchedGemvMGroupArgmax(
                Gemv_M128N8Argmax4,
                (epoch_m, 8, k),
                (load_matrix, load_vectors),
                partials,
                vocabulary_base=epoch * epoch_m,
                partial_base=epoch * sms,
            )
            .bar("partial", partial_barrier)
            .place(sms)
        )

    schedules.append(
        SchedArgmaxReduceGlobal(
            num_token=8,
            AtomReduce=ARGMAX_REDUCE_GLOBAL_bf16_256,
            mat_out_partial=partials,
            mat_final_out=output_tokens,
        )
        .bar("partial", partial_barrier)
        .place(8)
    )
    launcher.s(*schedules)

    launcher.launch()
    torch.cuda.synchronize()
    materialized = torch.cat(
        [matrix @ vectors.t() for matrix in matrices], dim=0
    )
    expected_tokens = torch.argmax(materialized, dim=0)
    if not torch.equal(output_tokens, expected_tokens):
        actual_partial_values = (
            partials[..., :2].contiguous().view(torch.bfloat16).reshape(8, 2 * sms)
        )
        actual_partial_indices = (
            partials[..., 8:16].contiguous().view(torch.int64).reshape(8, 2 * sms)
        )
        expected_partial_indices = torch.empty_like(actual_partial_indices)
        expected_partial_values = torch.empty_like(actual_partial_values)
        group_max_indices = torch.empty(
            (8, 2 * sms, Gemv_M128N8Argmax4.output_groups),
            dtype=torch.int64,
            device=device,
        )
        group_max_values = torch.empty(
            (8, 2 * sms, Gemv_M128N8Argmax4.output_groups),
            dtype=torch.bfloat16,
            device=device,
        )
        tile_m = Gemv_M128N8Argmax4.MNK[0]
        for epoch in range(2):
            epoch_logits = materialized[
                epoch * epoch_m : (epoch + 1) * epoch_m
            ]
            task_logits = (
                epoch_logits.view(
                    Gemv_M128N8Argmax4.output_groups, sms, tile_m, 8
                )
                .permute(3, 1, 0, 2)
            )
            per_group_rows = torch.argmax(task_logits, dim=3)
            per_group_values = torch.gather(
                task_logits, 3, per_group_rows.unsqueeze(-1)
            ).squeeze(-1)
            per_group_indices = (
                epoch * epoch_m
                + torch.arange(
                    Gemv_M128N8Argmax4.output_groups, device=device
                )[None, None, :]
                * (sms * tile_m)
                + torch.arange(sms, device=device)[None, :, None] * tile_m
                + per_group_rows
            )
            group_max_indices[:, epoch * sms : (epoch + 1) * sms] = (
                per_group_indices
            )
            group_max_values[:, epoch * sms : (epoch + 1) * sms] = (
                per_group_values
            )
            flat_task_logits = task_logits.reshape(8, sms, -1)
            task_offsets = torch.argmax(flat_task_logits, dim=2)
            task_groups = torch.div(task_offsets, tile_m, rounding_mode="floor")
            task_rows = task_offsets.remainder(tile_m)
            sm_rows = torch.arange(sms, device=device)[None, :] * tile_m
            task_indices = (
                epoch * epoch_m
                + task_groups * (sms * tile_m)
                + sm_rows
                + task_rows
            )
            task_values = torch.gather(
                flat_task_logits, 2, task_offsets.unsqueeze(-1)
            ).squeeze(-1)
            expected_partial_indices[:, epoch * sms : (epoch + 1) * sms] = (
                task_indices
            )
            expected_partial_values[:, epoch * sms : (epoch + 1) * sms] = (
                task_values
            )

        partial_index_matches = actual_partial_indices == expected_partial_indices
        partial_value_matches = actual_partial_values == expected_partial_values
        valid_actual_indices = actual_partial_indices.clamp(0, 2 * epoch_m - 1)
        actual_index_values = torch.gather(
            materialized.t(), 1, valid_actual_indices
        )
        actual_index_value_matches = actual_partial_values == actual_index_values
        actual_is_group_max = (
            (actual_partial_values.unsqueeze(-1) == group_max_values)
            & (actual_partial_indices.unsqueeze(-1) == group_max_indices)
        )
        actual_group = torch.where(
            actual_is_group_max,
            torch.arange(
                Gemv_M128N8Argmax4.output_groups, device=device
            )[None, None, :],
            Gemv_M128N8Argmax4.output_groups,
        ).min(dim=2).values
        actual_group_counts = [
            int((actual_group == group).sum())
            for group in range(Gemv_M128N8Argmax4.output_groups + 1)
        ]
        expected_group = torch.div(
            expected_partial_indices.remainder(epoch_m),
            sms * tile_m,
            rounding_mode="floor",
        )
        expected_group_counts = [
            int((expected_group == group).sum())
            for group in range(Gemv_M128N8Argmax4.output_groups)
        ]
        actual_row_bucket = actual_partial_indices.remainder(tile_m) // 16
        expected_row_bucket = expected_partial_indices.remainder(tile_m) // 16
        actual_row_bucket_counts = [
            int((actual_row_bucket == bucket).sum()) for bucket in range(8)
        ]
        expected_row_bucket_counts = [
            int((expected_row_bucket == bucket).sum()) for bucket in range(8)
        ]
        actual_max_values = actual_partial_values.max(dim=1, keepdim=True).values
        max_index = torch.iinfo(torch.int64).max
        reduced_actual_indices = torch.where(
            actual_partial_values == actual_max_values,
            actual_partial_indices,
            max_index,
        ).min(dim=1).values
        mismatched_records = (~(partial_index_matches & partial_value_matches)).nonzero()
        print(
            "GEMV_ARGMAX_DIAGNOSTIC "
            f"partial_index_matches={int(partial_index_matches.sum())}/2048 "
            f"partial_value_matches={int(partial_value_matches.sum())}/2048 "
            "actual_index_value_matches="
            f"{int(actual_index_value_matches.sum())}/2048 "
            f"actual_group_counts={actual_group_counts} "
            f"expected_group_counts={expected_group_counts} "
            f"actual_row16_counts={actual_row_bucket_counts} "
            f"expected_row16_counts={expected_row_bucket_counts} "
            f"reducer_matches_actual={torch.equal(output_tokens, reduced_actual_indices)} "
            f"actual_reduced={reduced_actual_indices.tolist()}"
        )
        for token, partial in mismatched_records[:16].tolist():
            print(
                "GEMV_ARGMAX_PARTIAL_MISMATCH "
                f"token={token} partial={partial} "
                f"actual_value={float(actual_partial_values[token, partial])} "
                f"actual_index={int(actual_partial_indices[token, partial])} "
                f"expected_value={float(expected_partial_values[token, partial])} "
                f"expected_index={int(expected_partial_indices[token, partial])}"
            )
        raise AssertionError(
            "fused argmax mismatch: "
            f"actual={output_tokens.tolist()} "
            f"expected={expected_tokens.tolist()}"
        )

    print(
        "GEMV_ARGMAX_CORRECT "
        f"m={2 * epoch_m} epoch_m={epoch_m} n=8 k={k} sms={sms} "
        f"tokens={output_tokens.tolist()}"
    )
    launcher.bench(iterations)


if __name__ == "__main__":
    main()
