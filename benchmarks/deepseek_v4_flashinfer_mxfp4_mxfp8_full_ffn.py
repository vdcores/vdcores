#!/usr/bin/env python3
"""FlashInfer MXFP4/MXFP8 composed full-FFN cold baseline.

This is the legal seven-group FlashInfer backend chain: grouped FC1, vLLM
SwiGLU, per-expert MXFP8 quantization and packing, grouped FC2, then routed
reduction.  It is not the FlashInfer TRTLLM fused-MoE entry point; that path's
first tactic setup is intentionally outside this bounded benchmark.
"""

from __future__ import annotations

import argparse
import statistics

import torch

from deepseek_v4_cold_timing import cold_graph_timings_us, percentile_us


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=7)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--tile-n", type=int, choices=(64, 128, 192, 256), default=128)
    parser.add_argument("--tile-k", type=int, choices=(128, 256), default=128)
    parser.add_argument("--mma-sm", type=int, choices=(1, 2), default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--cold-l2-scrub-mib", type=int, default=260)
    args = parser.parse_args()
    if args.experts != 7:
        parser.error("the matched shared+routed comparison requires seven experts")
    if args.rows <= 0 or args.rows % 4:
        parser.error("--rows must be a positive multiple of four")

    import flashinfer
    import vllm

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260817)
    e, rows, h, i = args.experts, args.rows, args.hidden, args.intermediate
    total_rows = e * rows
    m_indptr = torch.arange(
        0, total_rows + 1, rows, dtype=torch.int32, device=device
    )

    input_source = (
        torch.randn((rows, h), dtype=torch.bfloat16, device=device, generator=generator)
        * 0.1
    )
    input_q, input_scale_one = flashinfer.mxfp8_quantize(
        input_source, is_sf_swizzled_layout=True
    )
    input_q = input_q.repeat(e, 1)
    input_scale = torch.cat(
        [input_scale_one.reshape(-1, h // 32) for _ in range(e)]
    )

    w1_source = (
        torch.randn(
            (2 * i, h), dtype=torch.bfloat16, device=device, generator=generator
        )
        * 0.05
    )
    w1_one, w1_scale_one = flashinfer.mxfp4_quantize(w1_source)
    w1 = w1_one.unsqueeze(0).repeat(e, 1, 1)
    w1_scale = w1_scale_one.unsqueeze(0).repeat(e, 1, 1)
    del w1_one, w1_scale_one

    w2_source = (
        torch.randn((h, i), dtype=torch.bfloat16, device=device, generator=generator)
        * 0.05
    )
    w2_one, w2_scale_one = flashinfer.mxfp4_quantize(w2_source)
    w2 = w2_one.unsqueeze(0).repeat(e, 1, 1)
    w2_scale = w2_scale_one.unsqueeze(0).repeat(e, 1, 1)
    del w2_one, w2_scale_one

    gate_up = torch.empty((total_rows, 2 * i), dtype=torch.bfloat16, device=device)
    middle = torch.empty((total_rows, i), dtype=torch.bfloat16, device=device)
    down = torch.empty((total_rows, h), dtype=torch.bfloat16, device=device)
    output = torch.empty((rows, h), dtype=torch.float32, device=device)
    route_scales = torch.tensor(
        [1.0, *([1.0 / 6.0] * 6)], dtype=torch.float32, device=device
    ).view(e, 1, 1)
    captured_tensors: dict[str, object] = {}

    def full_ffn() -> None:
        flashinfer.gemm.group_gemm_mxfp8_mxfp4_nt_groupwise(
            input_q,
            w1,
            input_scale,
            w1_scale,
            m_indptr,
            mma_sm=args.mma_sm,
            tile_m=128,
            tile_n=args.tile_n,
            tile_k=args.tile_k,
            swap_ab=True,
            out=gate_up,
        )
        torch.ops._C.silu_and_mul(middle, gate_up)

        quantized_parts = []
        scale_parts = []
        for expert in range(e):
            begin = expert * rows
            end = begin + rows
            quantized, scale = flashinfer.mxfp8_quantize(
                middle[begin:end], is_sf_swizzled_layout=True
            )
            quantized_parts.append(quantized)
            scale_parts.append(scale.reshape(-1, i // 32))
        middle_q = torch.cat(quantized_parts)
        middle_scale = torch.cat(scale_parts)
        flashinfer.gemm.group_gemm_mxfp8_mxfp4_nt_groupwise(
            middle_q,
            w2,
            middle_scale,
            w2_scale,
            m_indptr,
            mma_sm=args.mma_sm,
            tile_m=128,
            tile_n=args.tile_n,
            tile_k=args.tile_k,
            swap_ab=True,
            out=down,
        )
        torch.sum(
            down.view(e, rows, h).float() * route_scales,
            dim=0,
            out=output,
        )
        # Keep graph-capture allocations live for replay.
        captured_tensors["middle_q"] = middle_q
        captured_tensors["middle_scale"] = middle_scale

    full_ffn()
    torch.cuda.synchronize()
    if not torch.isfinite(output).all():
        raise AssertionError("FlashInfer composed full FFN produced non-finite output")
    reference_gate_up = input_source.float() @ w1_source.float().T
    reference_gate, reference_up = reference_gate_up.chunk(2, dim=-1)
    reference_middle = torch.nn.functional.silu(reference_gate) * reference_up
    reference_output = (
        reference_middle @ w2_source.float().T
    ) * float(route_scales.sum())
    mean_relative = (
        (output - reference_output).abs().mean()
        / reference_output.abs().mean().clamp_min(1.0e-8)
    ).item()
    cosine = torch.nn.functional.cosine_similarity(
        output.reshape(-1), reference_output.reshape(-1), dim=0
    ).item()
    if mean_relative > 0.30 or cosine < 0.95:
        raise AssertionError(
            "FlashInfer full FFN failed quantized sanity check: "
            f"mean_relative={mean_relative:.6f}, cosine={cosine:.6f}"
        )
    del w1_source, w2_source, reference_gate_up, reference_middle, reference_output
    reference = output.clone()
    full_ffn()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, reference, rtol=0, atol=0)

    times = cold_graph_timings_us(
        full_ffn,
        stream=torch.cuda.Stream(),
        warmup=args.warmup,
        samples=args.samples,
        l2_scrub_mib=args.cold_l2_scrub_mib,
    )
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
    print(
        "DSV4_FLASHINFER_MXFP4_MXFP8_FULL_FFN_COLD_RESULT "
        f"flashinfer={flashinfer.__version__} vllm={vllm.__version__} "
        f"experts={e} shared=1 routed=6 rows={rows} "
        "backend=FLASHINFER_GROUPED_COMPOSED fused_moe=false "
        "route_packing_timed=false weight_conversion_timed=false "
        "timing=cold_data_one_ffn_graph "
        f"l2_scrub_mib={args.cold_l2_scrub_mib} samples={args.samples} "
        f"min_us={min(times):.6f} "
        f"median_us={statistics.median(times):.6f} "
        f"p90_us={percentile_us(times, 0.90):.6f} "
        f"stddev_us={statistics.pstdev(times):.6f} "
        f"max_us={max(times):.6f} "
        f"mean_relative={mean_relative:.8f} cosine={cosine:.8f} "
        "output_correct=true",
        flush=True,
    )


if __name__ == "__main__":
    main()
