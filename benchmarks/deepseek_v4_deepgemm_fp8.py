#!/usr/bin/env python3
"""Graph-amortized DeepGEMM baselines for DeepSeek-V4 FP8 projections."""

from __future__ import annotations

import argparse
import statistics

import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    deepgemm_post_process_fp8_weight_block,
    per_token_group_quant_fp8_packed_for_deepgemm,
)
from vllm.third_party.deep_gemm.utils.math import (
    per_block_cast_to_fp8,
    per_token_cast_to_fp8,
)
from vllm.utils.deep_gemm import fp8_gemm_nt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--inner", type=int, default=20)
    args = parser.parse_args()
    if args.m <= 0 or args.k <= 0 or args.m % 128 or args.k % 128:
        parser.error("M and K must be positive multiples of 128")
    if args.batch <= 0:
        parser.error("projection batch must be positive")
    if min(args.warmup, args.samples, args.inner) <= 0:
        parser.error("timing counts must be positive")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260811)
    activation_source = torch.randn(
        (args.batch, 1, args.k),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ) * 0.1
    weight_source = torch.randn(
        (args.batch, args.m, args.k),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ) * 0.05

    activations = []
    activation_scales = []
    activation_refs = []
    activation_scale_refs = []
    weights = []
    weight_scales = []
    weight_refs = []
    weight_scale_refs = []
    outputs = []
    for batch in range(args.batch):
        activation_ref, activation_scale_ref = per_token_cast_to_fp8(
            activation_source[batch], use_ue8m0=True, gran_k=128
        )
        activation, activation_scale = (
            per_token_group_quant_fp8_packed_for_deepgemm(
                activation_source[batch], group_size=128, use_ue8m0=True
            )
        )
        torch.testing.assert_close(
            activation.view(torch.uint8),
            activation_ref.view(torch.uint8),
            rtol=0,
            atol=0,
        )
        weight_ref, weight_scale_ref = per_block_cast_to_fp8(
            weight_source[batch], use_ue8m0=True, gran_k=128
        )
        weight, weight_scale = deepgemm_post_process_fp8_weight_block(
            wq=weight_ref,
            ws=weight_scale_ref,
            quant_block_shape=(128, 128),
            use_e8m0=True,
        )
        activations.append(activation)
        activation_scales.append(activation_scale)
        activation_refs.append(activation_ref)
        activation_scale_refs.append(activation_scale_ref)
        weights.append(weight)
        weight_scales.append(weight_scale)
        weight_refs.append(weight_ref)
        weight_scale_refs.append(weight_scale_ref)
        outputs.append(
            torch.empty((1, args.m), dtype=torch.bfloat16, device=device)
        )

    def run() -> None:
        for batch in range(args.batch):
            fp8_gemm_nt(
                (activations[batch], activation_scales[batch]),
                (weights[batch], weight_scales[batch]),
                outputs[batch],
                is_deep_gemm_e8m0_used=True,
            )

    run()
    torch.cuda.synchronize(device)

    references = []
    for batch in range(args.batch):
        activation_dequant = (
            activation_refs[batch].float()
            * activation_scale_refs[batch].float().repeat_interleave(
                128, dim=1
            )
        )
        weight_dequant = (
            weight_refs[batch].float()
            * weight_scale_refs[batch]
            .float()
            .repeat_interleave(128, dim=0)
            .repeat_interleave(128, dim=1)
        )
        references.append(
            (activation_dequant @ weight_dequant.t()).to(torch.bfloat16)
        )
    output = torch.stack(outputs, dim=0)
    reference = torch.stack(references, dim=0)
    torch.testing.assert_close(output, reference, rtol=3.0e-2, atol=1.0e-1)
    max_abs = (output.float() - reference.float()).abs().max().item()

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(args.inner):
            run()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize(device)

    timings = []
    for _ in range(args.samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1.0e3 / args.inner)

    print(
        "DSV4_DEEPGEMM_FP8_RESULT "
        f"shape={args.m}x{args.batch}x{args.k} inner={args.inner} "
        f"min_us={min(timings):.6f} "
        f"median_us={statistics.median(timings):.6f} "
        f"max_us={max(timings):.6f} max_abs={max_abs:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
