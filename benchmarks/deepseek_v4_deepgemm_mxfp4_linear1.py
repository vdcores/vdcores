#!/usr/bin/env python3
"""DeepGEMM/vLLM W4A8 Linear-1 baseline for seven decode experts.

The benchmark follows vLLM's ``DeepGemmFP4Experts`` production data path:

* route/permute FP8 activations into DeepGEMM's contiguous grouped layout;
* run one grouped FP8 x MXFP4 FC1 with gate and up concatenated in N; and
* run vLLM's fused SwiGLU + packed UE8M0 FP8 requantization kernel.

Routing/permute and operand quantization are setup-only. The default uses
eight useful rows per expert to match the VDCores N8 operand. This makes the
reported chained latency a kernel-level Linear-1 comparison with VDCores,
where the routing decision and quantized operands are already resident.
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
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--graph-inner", type=int, default=20)
    args = parser.parse_args()
    if args.experts <= 0:
        parser.error("--experts must be positive")
    if args.rows_per_expert <= 0:
        parser.error("--rows-per-expert must be positive")
    if args.hidden % 128 or args.intermediate % 128:
        parser.error("hidden/intermediate must be M128 aligned")

    import vllm
    from flashinfer.testing import bench_gpu_time
    from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
        compute_aligned_M_and_alignment,
        deepgemm_moe_permute,
    )
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_weight_scale_block,
        silu_mul_quant_fp8_packed_triton,
    )
    from vllm.utils.deep_gemm import (
        get_mk_alignment_for_contiguous_layout,
        m_grouped_fp8_fp4_gemm_nt_contiguous,
        mk_alignment_scope,
    )

    device = torch.device("cuda")
    groups = args.experts
    k = args.hidden
    n = 2 * args.intermediate
    activation_block_k = 128
    weight_block_k = 32

    block_m = get_mk_alignment_for_contiguous_layout()[0]
    padded_rows, alignment = compute_aligned_M_and_alignment(
        M=args.rows_per_expert,
        num_topk=groups,
        local_num_experts=groups,
        alignment=block_m,
        expert_tokens_meta=None,
    )

    # Eight physical decode rows are routed to every selected expert, matching
    # the useful N8 operand consumed by the VDCores UMMA task. E8M0 byte 127
    # represents scale 1.0; setup remains outside the timed frontier.
    activation = torch.ones(
        (args.rows_per_expert, k),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    activation_scale = torch.full(
        (args.rows_per_expert, k // activation_block_k),
        127,
        dtype=torch.uint8,
        device=device,
    )
    topk_ids = (
        torch.arange(groups, dtype=torch.int32, device=device)
        .view(1, -1)
        .expand(args.rows_per_expert, groups)
        .contiguous()
    )
    activation_permuted = torch.empty(
        (padded_rows, k), dtype=torch.float8_e4m3fn, device=device
    )
    (
        activation_permuted,
        activation_scale_permuted,
        expert_ids,
        _inv_perm,
        permute_alignment,
    ) = deepgemm_moe_permute(
        aq=activation,
        aq_scale=activation_scale,
        topk_ids=topk_ids,
        local_num_experts=groups,
        expert_map=None,
        expert_tokens_meta=None,
        aq_out=activation_permuted,
        block_size=activation_block_k,
    )
    if permute_alignment != alignment:
        raise AssertionError(
            f"permute alignment {permute_alignment} != planned alignment {alignment}"
        )

    # Native packed E2M1 bytes and checkpoint-order UE8M0 scales.  vLLM
    # performs only the scale-layout transform for its DeepGEMM FP4 backend;
    # packed weights remain in [expert, N, K/2] order.
    weight = torch.full(
        (groups, n, k // 2), 0x66, dtype=torch.uint8, device=device
    )
    weight_scale_checkpoint = torch.full(
        (groups, n, k // weight_block_k),
        127,
        dtype=torch.uint8,
        device=device,
    )
    weight_scale = deepgemm_post_process_weight_scale_block(
        ws=weight_scale_checkpoint,
        mn=n,
        k=k,
        quant_block_shape=(1, weight_block_k),
        num_groups=groups,
    )

    gate_up = torch.empty((padded_rows, n), dtype=torch.bfloat16, device=device)
    middle = torch.empty(
        (padded_rows, args.intermediate),
        dtype=torch.float8_e4m3fn,
        device=device,
    )

    def gemm() -> None:
        with mk_alignment_scope(alignment):
            m_grouped_fp8_fp4_gemm_nt_contiguous(
                (activation_permuted, activation_scale_permuted),
                (weight.view(torch.int8), weight_scale),
                gate_up,
                expert_ids,
                recipe_a=(1, activation_block_k),
                recipe_b=(1, weight_block_k),
            )

    def swiglu_quant() -> tuple[torch.Tensor, torch.Tensor]:
        return silu_mul_quant_fp8_packed_triton(
            input=gate_up,
            group_size=activation_block_k,
            output_q=middle,
        )

    def linear1() -> tuple[torch.Tensor, torch.Tensor]:
        gemm()
        return swiglu_quant()

    # Trigger all JIT work before timing.
    gemm()
    quantized_middle, quantized_scale = swiglu_quant()
    torch.cuda.synchronize()
    valid_rows = expert_ids >= 0
    expected_valid_rows = groups * args.rows_per_expert
    if int(valid_rows.sum()) != expected_valid_rows:
        raise AssertionError(
            f"expected {expected_valid_rows} valid expert rows, "
            f"got {int(valid_rows.sum())}"
        )
    if not torch.isfinite(gate_up[valid_rows].float()).all():
        raise AssertionError("DeepGEMM grouped FP8xFP4 FC1 produced non-finite output")
    if not torch.isfinite(quantized_middle[valid_rows].float()).all():
        raise AssertionError("vLLM fused SwiGLU+quant produced non-finite output")
    if quantized_scale.numel() == 0:
        raise AssertionError("vLLM fused SwiGLU+quant returned no scales")

    common = dict(
        dry_run_iters=args.warmup,
        repeat_iters=args.samples,
        use_cuda_graph=True,
        num_iters_within_graph=args.graph_inner,
        cold_l2_cache=False,
    )
    gemm_ms = bench_gpu_time(gemm, **common)
    swiglu_quant_ms = bench_gpu_time(swiglu_quant, **common)
    linear1_ms = bench_gpu_time(linear1, **common)

    print(
        "DSV4_DEEPGEMM_MXFP4_LINEAR1_RESULT "
        f"vllm={vllm.__version__} experts={groups} "
        f"logical_rows={groups * args.rows_per_expert} "
        f"padded_rows={padded_rows} alignment={alignment} "
        f"shape_per_expert={args.rows_per_expert}x{n}x{k} "
        f"gemm_graph_us={median_us(gemm_ms):.6f} "
        f"swiglu_quant_graph_us={median_us(swiglu_quant_ms):.6f} "
        f"linear1_graph_us={median_us(linear1_ms):.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
