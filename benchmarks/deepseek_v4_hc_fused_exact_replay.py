#!/usr/bin/env python3
"""Replay one fused mHC post/projection from an in-kernel operand capture."""

from __future__ import annotations

import argparse

import torch

from dae.launcher import Launcher
from dae.schedule import SchedDsv4Fp32Bf16Gemv


def _defined_partial_indices(device: torch.device) -> torch.Tensor:
    indices: list[int] = []
    for split in range(SchedDsv4Fp32Bf16Gemv.FUSED_SPLITS):
        split_base = split * SchedDsv4Fp32Bf16Gemv.FUSED_RECORD_STRIDE
        for group in range(SchedDsv4Fp32Bf16Gemv.FUSED_GROUPS):
            count = 4 if group == 0 else 3
            indices.extend(
                split_base + group * 4 + offset for offset in range(count)
            )
    return torch.tensor(indices, dtype=torch.int64, device=device)


def _expected_partials(
    record: torch.Tensor,
    weights: torch.Tensor,
    coefficients: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    branch = record[0].float()
    residual = record[1:].float()
    post = coefficients[:4]
    comb = coefficients[4:].view(4, 4)
    materialized = (
        post[:, None] * branch[None, :]
        + torch.einsum("ij,id->jd", comb, residual)
    ).to(torch.bfloat16)

    partials = torch.full(
        (
            SchedDsv4Fp32Bf16Gemv.FUSED_SPLITS
            * SchedDsv4Fp32Bf16Gemv.FUSED_RECORD_STRIDE
        ,),
        float("nan"),
        dtype=torch.float32,
        device=record.device,
    )
    for group in range(SchedDsv4Fp32Bf16Gemv.FUSED_GROUPS):
        for split in range(SchedDsv4Fp32Bf16Gemv.FUSED_SPLITS):
            start = split * SchedDsv4Fp32Bf16Gemv.FUSED_TILE_HIDDEN
            tile = materialized[:, start : start + 256].float()
            values = (weights[group, split, 0] * tile[None, :, :]).sum(
                dim=(1, 2)
            )
            output_start = split * 32 + group * 4
            if group == 0:
                partials[output_start] = tile.square().sum()
                partials[output_start + 1 : output_start + 4] = values
            else:
                partials[output_start : output_start + 3] = values
    return materialized, partials


def _report_integrated_capture(saved: dict[str, torch.Tensor]) -> None:
    record_capture = saved["mhc_consumed_record_capture"].contiguous()
    weight_capture = saved["mhc_consumed_weight_capture"].contiguous()
    coefficient_capture = saved[
        "mhc_consumed_coefficient_capture"
    ].contiguous()
    record_arenas = saved["mhc_cross_layer_input_records"]
    weight_reference = saved["mhc_fused_weight_reference"].reshape_as(
        weight_capture
    )
    coefficient_arenas = saved["mhc_output_metadatas"]

    record_matches: list[int] = []
    for arena in record_arenas:
        matches = 0
        for sm in range(SchedDsv4Fp32Bf16Gemv.FUSED_TASK_SMS):
            split = sm % SchedDsv4Fp32Bf16Gemv.FUSED_SPLITS
            start = split * SchedDsv4Fp32Bf16Gemv.FUSED_TILE_HIDDEN
            matches += int(
                torch.equal(
                    record_capture[sm],
                    arena[:, start : start + 256],
                )
            )
        record_matches.append(matches)
    coefficient_matches = [
        sum(
            int(torch.equal(captured, arena))
            for captured in coefficient_capture
        )
        for arena in coefficient_arenas
    ]
    weight_matches = sum(
        int(torch.equal(captured, expected))
        for captured, expected in zip(weight_capture, weight_reference)
    )

    defined = _defined_partial_indices(torch.device("cpu"))
    integrated_partials = saved["mhc_fused_metadata"][: 16 * 32].index_select(
        0, defined
    )
    print(
        "DSV4_HC_FUSED_INTEGRATED_CAPTURE "
        f"record_matches={record_matches} "
        f"weight_matches={weight_matches}/128 "
        f"coefficient_matches={coefficient_matches} "
        f"record_finite={int(torch.isfinite(record_capture).sum())}/"
        f"{record_capture.numel()} "
        f"weight_finite={int(torch.isfinite(weight_capture).sum())}/"
        f"{weight_capture.numel()} "
        f"coefficient_finite={int(torch.isfinite(coefficient_capture).sum())}/"
        f"{coefficient_capture.numel()} "
        f"defined_output_finite="
        f"{int(torch.isfinite(integrated_partials).sum())}/"
        f"{integrated_partials.numel()} "
        f"defined_output_nan={int(torch.isnan(integrated_partials).sum())}/"
        f"{integrated_partials.numel()}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True)
    args = parser.parse_args()

    saved = torch.load(args.capture, map_location="cpu", weights_only=True)
    _report_integrated_capture(saved)
    defined_cpu = _defined_partial_indices(torch.device("cpu"))
    integrated_output = saved["residual"].contiguous()
    integrated_partials = saved["mhc_fused_metadata"][
        : SchedDsv4Fp32Bf16Gemv.FUSED_SPLITS
        * SchedDsv4Fp32Bf16Gemv.FUSED_RECORD_STRIDE
    ].index_select(0, defined_cpu).contiguous()
    source_record_capture = saved["mhc_consumed_record_capture"].contiguous()
    source_weight_capture = saved["mhc_consumed_weight_capture"].contiguous()
    source_coefficient_capture = saved[
        "mhc_consumed_coefficient_capture"
    ].contiguous()

    splits = SchedDsv4Fp32Bf16Gemv.FUSED_SPLITS
    groups = SchedDsv4Fp32Bf16Gemv.FUSED_GROUPS
    record = torch.cat(
        [source_record_capture[split] for split in range(splits)], dim=1
    ).contiguous()
    weights = (
        source_weight_capture.view(groups, splits, 3, 4, 256)
        .unsqueeze(2)
        .contiguous()
    )
    coefficients = source_coefficient_capture[0].contiguous()

    device = torch.device("cuda")
    record = record.to(device)
    weights = weights.to(device)
    coefficients = coefficients.to(device)
    source_record_capture = source_record_capture.to(device)
    source_weight_capture = source_weight_capture.to(device)
    source_coefficient_capture = source_coefficient_capture.to(device)

    output = torch.full(
        (4, 4096), float("nan"), dtype=torch.bfloat16, device=device
    )
    mixes = torch.full((24,), float("nan"), dtype=torch.float32, device=device)
    metadata = saved["mhc_fused_metadata"].to(device).contiguous()
    replay_record_capture = torch.full_like(
        source_record_capture, float("nan")
    )
    replay_weight_capture = torch.full_like(
        source_weight_capture, float("nan")
    )
    replay_coefficient_capture = torch.full_like(
        source_coefficient_capture, float("nan")
    )

    launcher = Launcher(
        SchedDsv4Fp32Bf16Gemv.FUSED_TASK_SMS, device=device
    )
    schedule = SchedDsv4Fp32Bf16Gemv(
        weights,
        output.reshape(-1),
        mixes,
        fused_post_input_record=record,
        fused_post_output=output,
        fused_partial_metadata=metadata,
        packed_coefficients=coefficients,
        launcher=launcher,
        profile_operands=True,
        captured_record=replay_record_capture,
        captured_weight=replay_weight_capture,
        captured_coefficients=replay_coefficient_capture,
    )
    launcher.s(schedule.place(SchedDsv4Fp32Bf16Gemv.FUSED_TASK_SMS))
    launcher.launch()

    expected_output, expected_partials = _expected_partials(
        record, weights, coefficients
    )
    defined = _defined_partial_indices(device)
    actual_partials = metadata[: splits * 32].index_select(0, defined)
    expected_defined = expected_partials.index_select(0, defined)

    record_exact = torch.equal(replay_record_capture, source_record_capture)
    weight_exact = torch.equal(replay_weight_capture, source_weight_capture)
    coefficient_exact = torch.equal(
        replay_coefficient_capture, source_coefficient_capture
    )
    output_finite = bool(torch.isfinite(output).all().item())
    partials_finite = bool(torch.isfinite(actual_partials).all().item())
    output_max_abs = float(
        (output.float() - expected_output.float()).abs().max().item()
    )
    partial_max_abs = float(
        (actual_partials - expected_defined).abs().max().item()
    )
    replay_output_cpu = output.detach().cpu()
    replay_partials_cpu = actual_partials.detach().cpu()
    integrated_output_exact = torch.equal(
        integrated_output, replay_output_cpu
    )
    integrated_partials_exact = torch.equal(
        integrated_partials, replay_partials_cpu
    )
    integrated_output_finite = bool(
        torch.isfinite(integrated_output).all().item()
    )
    integrated_partials_finite = bool(
        torch.isfinite(integrated_partials).all().item()
    )
    status = (
        "PASS"
        if record_exact
        and weight_exact
        and coefficient_exact
        and output_finite
        and partials_finite
        and integrated_output_exact
        and integrated_partials_exact
        and integrated_output_finite
        and integrated_partials_finite
        else "FAIL"
    )
    print(
        "DSV4_HC_FUSED_EXACT_REPLAY "
        f"status={status} record_exact={int(record_exact)} "
        f"weight_exact={int(weight_exact)} "
        f"coeff_exact={int(coefficient_exact)} "
        f"output_finite={int(output_finite)} "
        f"partials_finite={int(partials_finite)} "
        f"integrated_output_exact={int(integrated_output_exact)} "
        f"integrated_partials_exact={int(integrated_partials_exact)} "
        f"integrated_output_finite={int(integrated_output_finite)} "
        f"integrated_partials_finite={int(integrated_partials_finite)} "
        f"output_max_abs={output_max_abs:.9f} "
        f"partial_max_abs={partial_max_abs:.9f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
