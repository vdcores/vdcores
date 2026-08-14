#!/usr/bin/env python3
"""FlashInfer NVFP4 GEMM baseline for DeepSeek-V4 decode shapes.

This benchmark intentionally lives outside the VDCores runtime.  It provides
the task-level comparison target for a one-token expert projection while using
the same BF16 source tensors and NVFP4 scaling contract as FlashInfer.
"""

from __future__ import annotations

import argparse
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--backend", choices=("cutlass", "cudnn", "auto"), default="cutlass")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--graph-inner", type=int, default=20)
    parser.add_argument(
        "--cuda-profiler-capture",
        action="store_true",
        help="bracket one warmed GEMM with cudaProfilerStart/Stop",
    )
    args = parser.parse_args()

    import flashinfer
    from flashinfer import SfLayout
    from flashinfer.testing import bench_gpu_time

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260810)
    activation_source = torch.randn(
        (1, args.k), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.1
    weight_source = torch.randn(
        (args.m, args.k), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.05

    # FlashInfer takes quantization multipliers.  Its GEMM alpha is the inverse
    # product, equivalent to VDCores' pair of global dequantization scales.
    activation_global_sf = (
        (448.0 * 6.0) / activation_source.float().abs().amax()
    ).reshape(1)
    weight_global_sf = (
        (448.0 * 6.0) / weight_source.float().abs().amax()
    ).reshape(1)
    activation, activation_sf = flashinfer.nvfp4_quantize(
        activation_source,
        activation_global_sf,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
    )
    weight, weight_sf = flashinfer.nvfp4_quantize(
        weight_source,
        weight_global_sf,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
    )
    alpha = (1.0 / (activation_global_sf * weight_global_sf)).float()
    output = torch.empty((1, args.m), dtype=torch.bfloat16, device=device)

    def gemm_tensors(
        activation_arg: torch.Tensor,
        weight_arg: torch.Tensor,
        activation_sf_arg: torch.Tensor,
        weight_sf_arg: torch.Tensor,
        alpha_arg: torch.Tensor,
        output_arg: torch.Tensor,
    ) -> None:
        flashinfer.mm_fp4(
            activation_arg,
            weight_arg,
            activation_sf_arg,
            weight_sf_arg,
            alpha_arg,
            out_dtype=torch.bfloat16,
            out=output_arg,
            backend=args.backend,
        )

    gemm_args = (activation, weight.T, activation_sf, weight_sf.T, alpha, output)

    def gemm() -> None:
        gemm_tensors(*gemm_args)

    # Trigger extension loading and autotuning before collecting samples.
    gemm()
    torch.cuda.synchronize()
    reference = activation_source.float() @ weight_source.float().T
    mean_relative = (
        (output.float() - reference).abs().mean()
        / reference.abs().mean().clamp_min(1.0e-8)
    ).item()
    cosine = torch.nn.functional.cosine_similarity(
        output.float(), reference, dim=-1
    ).item()
    if mean_relative > 0.20 or cosine < 0.98:
        raise AssertionError(
            f"FlashInfer result failed quantized sanity check: "
            f"mean_relative={mean_relative:.6f}, cosine={cosine:.6f}"
        )

    for _ in range(args.warmup):
        gemm()
    torch.cuda.synchronize()
    if args.cuda_profiler_capture:
        torch.cuda.cudart().cudaProfilerStart()
        gemm()
        torch.cuda.cudart().cudaProfilerStop()
        torch.cuda.synchronize()
    hot_event_ms = bench_gpu_time(
        gemm_tensors,
        dry_run_iters=args.warmup,
        repeat_iters=args.samples,
        input_args=gemm_args,
        cold_l2_cache=False,
    )
    cold_event_ms = bench_gpu_time(
        gemm_tensors,
        dry_run_iters=args.warmup,
        repeat_iters=args.samples,
        input_args=gemm_args,
        cold_l2_cache=True,
    )
    hot_graph_ms = bench_gpu_time(
        gemm_tensors,
        dry_run_iters=args.warmup,
        repeat_iters=args.samples,
        input_args=gemm_args,
        use_cuda_graph=True,
        num_iters_within_graph=args.graph_inner,
        cold_l2_cache=False,
    )

    def quantize_activation() -> None:
        flashinfer.nvfp4_quantize(
            activation_source,
            activation_global_sf,
            sfLayout=SfLayout.layout_128x4,
            do_shuffle=False,
        )

    for _ in range(args.warmup):
        quantize_activation()
    torch.cuda.synchronize()
    quant_event_ms = bench_gpu_time(
        quantize_activation,
        dry_run_iters=args.warmup,
        repeat_iters=args.samples,
        cold_l2_cache=False,
    )
    quant_graph_ms = bench_gpu_time(
        quantize_activation,
        dry_run_iters=args.warmup,
        repeat_iters=args.samples,
        use_cuda_graph=True,
        num_iters_within_graph=args.graph_inner,
        cold_l2_cache=False,
    )

    def median_us(values: list[float]) -> float:
        return float(torch.tensor(values, dtype=torch.float64).median()) * 1.0e3

    print(
        "DSV4_FLASHINFER_NVFP4_RESULT "
        f"version={flashinfer.__version__} backend={args.backend} "
        f"shape={args.m}x1x{args.k} "
        f"gemm_hot_event_median_us={median_us(hot_event_ms):.6f} "
        f"gemm_cold_event_median_us={median_us(cold_event_ms):.6f} "
        f"gemm_hot_graph_median_us={median_us(hot_graph_ms):.6f} "
        f"activation_quant_event_median_us={median_us(quant_event_ms):.6f} "
        f"activation_quant_graph_median_us={median_us(quant_graph_ms):.6f} "
        f"mean_relative={mean_relative:.8f} cosine={cosine:.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
