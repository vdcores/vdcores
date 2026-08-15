#!/usr/bin/env python3
"""FlashInfer/vLLM MXFP4 Linear-1 baseline for seven decode experts.

FlashInfer requires every grouped-GEMM indptr entry to be a multiple of four;
the selected comparison uses eight useful rows per shared/routed expert and
therefore needs no synthetic row padding inside a group. The timed GEMM fuses
gate and up in N=4096. Separate timings expose vLLM's SwiGLU and FlashInfer's
MXFP8 quantizer, plus their chained CUDA-graph cost. All input/weight
quantization is setup-only.
"""

from __future__ import annotations

import argparse
import statistics

import torch


def median_us(values: list[float]) -> float:
    return statistics.median(values) * 1.0e3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=7)
    parser.add_argument("--rows-per-expert", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--tile-n", type=int, choices=(64, 128, 192, 256), default=128)
    parser.add_argument("--tile-k", type=int, choices=(128, 256), default=128)
    parser.add_argument("--mma-sm", type=int, choices=(1, 2), default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--graph-inner", type=int, default=20)
    args = parser.parse_args()
    if args.experts <= 0:
        parser.error("--experts must be positive")
    if args.rows_per_expert <= 0 or args.rows_per_expert % 4:
        parser.error("--rows-per-expert must be a positive multiple of four")
    if args.hidden % 128 or args.intermediate % 128:
        parser.error("hidden/intermediate must be M128 aligned")

    import flashinfer
    import vllm
    from flashinfer.testing import bench_gpu_time

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260815)
    groups = args.experts
    rows = groups * args.rows_per_expert
    n = 2 * args.intermediate
    k = args.hidden
    activation_source = torch.randn(
        (rows, k), dtype=torch.bfloat16, device=device, generator=generator
    ) * 0.1
    weight_source = torch.randn(
        (groups, n, k), dtype=torch.bfloat16, device=device, generator=generator
    ) * 0.05

    packed_activations = []
    packed_activation_scales = []
    for expert in range(groups):
        row_start = expert * args.rows_per_expert
        row_stop = row_start + args.rows_per_expert
        packed, scale = flashinfer.mxfp8_quantize(
            activation_source[row_start:row_stop],
            is_sf_swizzled_layout=True,
        )
        packed_activations.append(packed)
        packed_activation_scales.append(scale.reshape(-1, k // 32))
    activation = torch.cat(packed_activations)
    # Grouped GEMM pads the scale-M dimension independently for each expert.
    activation_scale = torch.cat(packed_activation_scales)
    packed_weights = []
    packed_weight_scales = []
    for expert in range(groups):
        packed, scale = flashinfer.mxfp4_quantize(weight_source[expert])
        packed_weights.append(packed)
        packed_weight_scales.append(scale)
    weight = torch.stack(packed_weights)
    weight_scale = torch.stack(packed_weight_scales)
    m_indptr = torch.arange(
        0,
        rows + 1,
        args.rows_per_expert,
        dtype=torch.int32,
        device=device,
    )
    gate_up = torch.empty((rows, n), dtype=torch.bfloat16, device=device)
    middle = torch.empty(
        (rows, args.intermediate), dtype=torch.bfloat16, device=device
    )

    def gemm() -> None:
        flashinfer.gemm.group_gemm_mxfp8_mxfp4_nt_groupwise(
            activation,
            weight,
            activation_scale,
            weight_scale,
            m_indptr,
            mma_sm=args.mma_sm,
            tile_m=128,
            tile_n=args.tile_n,
            tile_k=args.tile_k,
            swap_ab=True,
            out=gate_up,
        )

    def swiglu() -> None:
        torch.ops._C.silu_and_mul(middle, gate_up)

    def quantize() -> tuple[torch.Tensor, torch.Tensor]:
        return flashinfer.mxfp8_quantize(
            middle, is_sf_swizzled_layout=True
        )

    def linear1() -> tuple[torch.Tensor, torch.Tensor]:
        gemm()
        swiglu()
        return quantize()

    gemm()
    swiglu()
    quantized_middle, quantized_scale = quantize()
    torch.cuda.synchronize()
    if not torch.isfinite(gate_up.float()).all():
        raise AssertionError("FlashInfer grouped MXFP4 GEMM produced non-finite output")
    if not torch.isfinite(middle.float()).all():
        raise AssertionError("vLLM SwiGLU produced non-finite output")
    if quantized_middle.shape != middle.shape or quantized_scale.numel() == 0:
        raise AssertionError("FlashInfer MXFP8 quantizer returned an invalid shape")

    common = dict(
        dry_run_iters=args.warmup,
        repeat_iters=args.samples,
        use_cuda_graph=True,
        num_iters_within_graph=args.graph_inner,
        cold_l2_cache=False,
    )
    gemm_ms = bench_gpu_time(gemm, **common)
    swiglu_ms = bench_gpu_time(swiglu, **common)
    quant_ms = bench_gpu_time(quantize, **common)
    linear1_ms = bench_gpu_time(linear1, **common)

    print(
        "DSV4_FLASHINFER_MXFP4_MXFP8_LINEAR1_RESULT "
        f"flashinfer={flashinfer.__version__} vllm={vllm.__version__} "
        f"experts={groups} logical_rows={rows} padded_rows={rows} "
        f"shape_per_expert={args.rows_per_expert}x{n}x{k} "
        f"mma_sm={args.mma_sm} tile_n={args.tile_n} tile_k={args.tile_k} "
        f"gemm_graph_us={median_us(gemm_ms):.6f} "
        f"swiglu_graph_us={median_us(swiglu_ms):.6f} "
        f"quant_graph_us={median_us(quant_ms):.6f} "
        f"linear1_graph_us={median_us(linear1_ms):.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
