#!/usr/bin/env python3
"""One-launch projection-to-packed-mHC/RMS correctness smoke."""

from __future__ import annotations

import torch

from dae.deepseek_v4 import hc_pre_reference
from dae.deepseek_v4_quant import quantize_fp8_block128
from dae.launcher import Launcher
from dae.schedule import SchedDsv4Fp32Bf16Gemv, SchedDsv4HcPreRms
from dae.sequential import SequentialProgram, SequentialStage


def main() -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260819)
    residual = (
        torch.randn(
            (4, 4096),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    projection = (
        torch.randn(
            (24, residual.numel()),
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        * 0.002
    )
    scale = torch.tensor((0.5, 0.75, 1.25), dtype=torch.float32, device=device)
    base = (
        torch.randn(
            (24,), generator=generator, dtype=torch.float32, device=device
        )
        * 0.05
    )
    norm_weight = (
        torch.randn(
            (4096,),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.05
        + 1.0
    )

    packed_metadata = torch.empty((56,), dtype=torch.float32, device=device)
    square_sum = packed_metadata[:1]
    mixes = packed_metadata[1:25]
    metadata_tail = packed_metadata[28:56]
    packed_output = torch.empty((4176,), dtype=torch.uint8, device=device)
    fp8_output = packed_output[:4096].view(torch.float8_e4m3fn)
    output_metadata = packed_output[4096:].view(torch.float32)
    post = output_metadata[:4]
    comb = output_metadata[4:].view(4, 4)
    fp8_scale = torch.empty(
        (32,), dtype=torch.float8_e8m0fnu, device=device
    )

    project = SchedDsv4Fp32Bf16Gemv(
        projection,
        residual.reshape(-1),
        mixes,
        square_sum_output=square_sum,
        metadata_scale=scale,
        metadata_base=base,
        metadata_tail_output=metadata_tail,
    )
    pre_rms = SchedDsv4HcPreRms(
        residual,
        mixes,
        scale,
        base,
        norm_weight,
        None,
        post,
        comb,
        residual_square_sum=square_sum,
        packed_metadata=packed_metadata,
        packed_output=packed_output,
        fp8_output=fp8_output,
        fp8_scale=fp8_scale,
    )
    launcher = Launcher(24, device=device)
    launcher.s(
        SequentialProgram(
            launcher,
            (
                SequentialStage("hc_project", project, 24),
                SequentialStage("hc_pre_rms", pre_rms, 1),
            ),
        )
    )
    launcher.launch()

    expected_mixes = (
        projection * residual.reshape(1, -1).float()
    ).sum(dim=1)
    expected_hidden, expected_post, expected_comb = hc_pre_reference(
        residual,
        expected_mixes,
        scale,
        base,
        sinkhorn_iters=20,
    )
    expected_output = (
        expected_hidden.float()
        * torch.rsqrt(expected_hidden.float().square().mean() + 1.0e-6)
        * norm_weight.float()
    ).to(torch.bfloat16)
    expected_fp8, expected_fp8_scale = quantize_fp8_block128(expected_output)

    torch.testing.assert_close(mixes, expected_mixes, rtol=2.0e-3, atol=2.0e-3)
    torch.testing.assert_close(
        square_sum,
        residual.float().square().sum().reshape(1),
        rtol=2.0e-4,
        atol=2.0e-4,
    )
    torch.testing.assert_close(metadata_tail[:3], scale, rtol=0, atol=0)
    torch.testing.assert_close(metadata_tail[3:27], base, rtol=0, atol=0)
    torch.testing.assert_close(fp8_output, expected_fp8, rtol=0, atol=0)
    torch.testing.assert_close(fp8_scale, expected_fp8_scale, rtol=0, atol=0)
    torch.testing.assert_close(post, expected_post, rtol=2.0e-5, atol=2.0e-5)
    torch.testing.assert_close(comb, expected_comb, rtol=2.0e-5, atol=2.0e-5)
    print("DSV4_HC_PACKED_PIPELINE status=PASS launches=1", flush=True)


if __name__ == "__main__":
    main()
