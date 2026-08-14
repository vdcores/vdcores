#!/usr/bin/env python3
"""FlashInfer MXFP4-weight/MXFP8-activation grouped-GEMM baseline.

Quantization and scale-layout conversion are setup-only.  Timings cover only
the Blackwell groupwise mixed-format GEMM used by vLLM's FlashInfer CUTLASS
MXFP4/MXFP8 backend.
"""

from __future__ import annotations

import argparse
import statistics

import torch


def median_us(values: list[float]) -> float:
    return statistics.median(values) * 1.0e3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--mma-sm", type=int, choices=(1, 2), default=1)
    parser.add_argument("--tile-n", type=int, choices=(64, 128, 192, 256), default=128)
    parser.add_argument("--tile-k", type=int, choices=(128, 256), default=128)
    parser.add_argument(
        "--no-swap-ab",
        action="store_true",
        help="disable FlashInfer's default A/B swap",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--graph-inner", type=int, default=20)
    args = parser.parse_args()
    if args.rows <= 0 or args.rows % 4:
        parser.error("--rows must be a positive multiple of four")
    if args.n <= 0 or args.n % 8:
        parser.error("--n must be a positive multiple of eight")
    if args.k <= 0 or args.k % 128:
        parser.error("--k must be a positive multiple of 128")

    import flashinfer
    import vllm
    from flashinfer.testing import bench_gpu_time

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260814)
    activation_source = torch.randn(
        (args.rows, args.k),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ) * 0.1
    weight_source = torch.randn(
        (args.n, args.k),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ) * 0.05

    # Both conversions are checkpoint/token preparation and deliberately sit
    # outside the measured call.  The swizzled activation scale is padded to
    # FlashInfer's native M128 layout.
    activation, activation_scale = flashinfer.mxfp8_quantize(
        activation_source,
        is_sf_swizzled_layout=True,
    )
    weight, weight_scale = flashinfer.mxfp4_quantize(weight_source)
    activation_scale = activation_scale.reshape(-1, args.k // 32)
    weight = weight.unsqueeze(0)
    weight_scale = weight_scale.unsqueeze(0)
    m_indptr = torch.tensor(
        (0, args.rows), dtype=torch.int32, device=device
    )
    output = torch.empty(
        (args.rows, args.n), dtype=torch.bfloat16, device=device
    )

    def gemm_tensors(
        activation_arg: torch.Tensor,
        weight_arg: torch.Tensor,
        activation_scale_arg: torch.Tensor,
        weight_scale_arg: torch.Tensor,
        m_indptr_arg: torch.Tensor,
        output_arg: torch.Tensor,
    ) -> None:
        flashinfer.gemm.group_gemm_mxfp8_mxfp4_nt_groupwise(
            activation_arg,
            weight_arg,
            activation_scale_arg,
            weight_scale_arg,
            m_indptr_arg,
            mma_sm=args.mma_sm,
            tile_m=128,
            tile_n=args.tile_n,
            tile_k=args.tile_k,
            swap_ab=not args.no_swap_ab,
            out=output_arg,
        )

    gemm_args = (
        activation,
        weight,
        activation_scale,
        weight_scale,
        m_indptr,
        output,
    )
    gemm_tensors(*gemm_args)
    torch.cuda.synchronize()
    reference = activation_source.float() @ weight_source.float().T
    mean_relative = (
        (output.float() - reference).abs().mean()
        / reference.abs().mean().clamp_min(1.0e-8)
    ).item()
    cosine = torch.nn.functional.cosine_similarity(
        output.float().reshape(-1), reference.reshape(-1), dim=0
    ).item()
    if mean_relative > 0.20 or cosine < 0.98:
        raise AssertionError(
            "FlashInfer MXFP4/MXFP8 result failed quantized sanity check: "
            f"mean_relative={mean_relative:.6f}, cosine={cosine:.6f}"
        )

    hot_ms = bench_gpu_time(
        gemm_tensors,
        dry_run_iters=args.warmup,
        repeat_iters=args.samples,
        input_args=gemm_args,
        cold_l2_cache=False,
    )
    cold_ms = bench_gpu_time(
        gemm_tensors,
        dry_run_iters=args.warmup,
        repeat_iters=args.samples,
        input_args=gemm_args,
        cold_l2_cache=True,
    )
    graph_ms = bench_gpu_time(
        gemm_tensors,
        dry_run_iters=args.warmup,
        repeat_iters=args.samples,
        input_args=gemm_args,
        use_cuda_graph=True,
        num_iters_within_graph=args.graph_inner,
        cold_l2_cache=False,
    )

    print(
        "DSV4_FLASHINFER_MXFP4_MXFP8_RESULT "
        f"flashinfer={flashinfer.__version__} vllm={vllm.__version__} "
        f"shape={args.rows}x{args.n}x{args.k} "
        f"mma_sm={args.mma_sm} tile_m=128 tile_n={args.tile_n} "
        f"tile_k={args.tile_k} swap_ab={not args.no_swap_ab} "
        f"hot_event_median_us={median_us(hot_ms):.6f} "
        f"cold_event_median_us={median_us(cold_ms):.6f} "
        f"hot_graph_median_us={median_us(graph_ms):.6f} "
        f"mean_relative={mean_relative:.8f} cosine={cosine:.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
