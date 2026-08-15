#!/usr/bin/env python3
"""Production vLLM/DeepGEMM W4A8 full-FFN comparison.

This follows ``DeepGemmFP4Experts.apply`` through grouped FC1, fused
SwiGLU+MXFP8 quantization, grouped FC2, and unpermute/route reduction.  The
default models one shared expert with weight 1 and six routed experts with
weight 1/6 for each of eight useful rows.  Route packing and input preparation
are outside the timed graph, matching the VDCores full-FFN frontier.
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
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--graph-inner", type=int, default=20)
    args = parser.parse_args()
    if args.experts != 7:
        parser.error("the matched shared+routed comparison requires seven experts")
    if args.rows <= 0:
        parser.error("--rows must be positive")

    import vllm
    from flashinfer.testing import bench_gpu_time
    from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
        compute_aligned_M_and_alignment,
        deepgemm_moe_permute,
        deepgemm_unpermute_and_reduce,
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
    e, rows, h, i = args.experts, args.rows, args.hidden, args.intermediate
    block_k = 128
    weight_block_k = 32
    block_m = get_mk_alignment_for_contiguous_layout()[0]
    padded_rows, alignment = compute_aligned_M_and_alignment(
        M=rows,
        num_topk=e,
        local_num_experts=e,
        alignment=block_m,
        expert_tokens_meta=None,
    )

    activation = torch.ones(
        (rows, h), dtype=torch.float8_e4m3fn, device=device
    )
    activation_scale = torch.full(
        (rows, h // block_k), 127, dtype=torch.uint8, device=device
    )
    topk_ids = (
        torch.arange(e, dtype=torch.int32, device=device)
        .view(1, -1)
        .expand(rows, e)
        .contiguous()
    )
    topk_weights = torch.full(
        (rows, e), 1.0 / 6.0, dtype=torch.float32, device=device
    )
    topk_weights[:, 0] = 1.0
    activation_permuted = torch.empty(
        (padded_rows, h), dtype=torch.float8_e4m3fn, device=device
    )
    (
        activation_permuted,
        activation_scale_permuted,
        expert_ids,
        inv_perm,
        permute_alignment,
    ) = deepgemm_moe_permute(
        aq=activation,
        aq_scale=activation_scale,
        topk_ids=topk_ids,
        local_num_experts=e,
        expert_map=None,
        expert_tokens_meta=None,
        aq_out=activation_permuted,
        block_size=block_k,
    )
    if permute_alignment != alignment:
        raise AssertionError("DeepGEMM permute alignment changed")

    w1 = torch.full(
        (e, 2 * i, h // 2), 0x66, dtype=torch.uint8, device=device
    )
    w2 = torch.full(
        (e, h, i // 2), 0x22, dtype=torch.uint8, device=device
    )
    w1_scale = deepgemm_post_process_weight_scale_block(
        ws=torch.full(
            (e, 2 * i, h // weight_block_k),
            127,
            dtype=torch.uint8,
            device=device,
        ),
        mn=2 * i,
        k=h,
        quant_block_shape=(1, weight_block_k),
        num_groups=e,
    )
    w2_scale = deepgemm_post_process_weight_scale_block(
        ws=torch.full(
            (e, h, i // weight_block_k),
            127,
            dtype=torch.uint8,
            device=device,
        ),
        mn=h,
        k=i,
        quant_block_shape=(1, weight_block_k),
        num_groups=e,
    )
    gate_up = torch.empty(
        (padded_rows, 2 * i), dtype=torch.bfloat16, device=device
    )
    middle = torch.empty(
        (padded_rows, i), dtype=torch.float8_e4m3fn, device=device
    )
    down = torch.empty(
        (padded_rows, h), dtype=torch.bfloat16, device=device
    )
    output = torch.empty((rows, h), dtype=torch.bfloat16, device=device)

    def fc1() -> None:
        with mk_alignment_scope(alignment):
            m_grouped_fp8_fp4_gemm_nt_contiguous(
                (activation_permuted, activation_scale_permuted),
                (w1.view(torch.int8), w1_scale),
                gate_up,
                expert_ids,
                recipe_a=(1, block_k),
                recipe_b=(1, weight_block_k),
            )

    def activate() -> tuple[torch.Tensor, torch.Tensor]:
        return silu_mul_quant_fp8_packed_triton(
            input=gate_up, group_size=block_k, output_q=middle
        )

    quantized_middle, quantized_middle_scale = activate()

    def fc2() -> None:
        with mk_alignment_scope(alignment):
            m_grouped_fp8_fp4_gemm_nt_contiguous(
                (quantized_middle, quantized_middle_scale),
                (w2.view(torch.int8), w2_scale),
                down,
                expert_ids,
                recipe_a=(1, block_k),
                recipe_b=(1, weight_block_k),
            )

    def reduce() -> None:
        deepgemm_unpermute_and_reduce(
            a=down,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            inv_perm=inv_perm,
            expert_map=None,
            output=output,
        )

    def full_ffn() -> torch.Tensor:
        fc1()
        current_middle, current_scale = activate()
        with mk_alignment_scope(alignment):
            m_grouped_fp8_fp4_gemm_nt_contiguous(
                (current_middle, current_scale),
                (w2.view(torch.int8), w2_scale),
                down,
                expert_ids,
                recipe_a=(1, block_k),
                recipe_b=(1, weight_block_k),
            )
        reduce()
        return output

    fc1()
    quantized_middle, quantized_middle_scale = activate()
    fc2()
    reduce()
    torch.cuda.synchronize()
    if not torch.isfinite(output.float()).all():
        raise AssertionError("DeepGEMM full FFN produced non-finite output")

    common = dict(
        dry_run_iters=args.warmup,
        repeat_iters=args.samples,
        use_cuda_graph=True,
        num_iters_within_graph=args.graph_inner,
        cold_l2_cache=False,
    )
    fc1_ms = bench_gpu_time(fc1, **common)
    activation_ms = bench_gpu_time(activate, **common)
    fc2_ms = bench_gpu_time(fc2, **common)
    reduce_ms = bench_gpu_time(reduce, **common)
    full_ms = bench_gpu_time(full_ffn, **common)
    print(
        "DSV4_DEEPGEMM_MXFP4_FULL_FFN_RESULT "
        f"vllm={vllm.__version__} experts={e} rows={rows} "
        f"logical_rows={e * rows} padded_rows={padded_rows} alignment={alignment} "
        f"fc1_us={median_us(fc1_ms):.6f} "
        f"activation_quant_us={median_us(activation_ms):.6f} "
        f"fc2_us={median_us(fc2_ms):.6f} "
        f"reduce_us={median_us(reduce_ms):.6f} "
        f"component_sum_us={median_us(fc1_ms) + median_us(activation_ms) + median_us(fc2_ms) + median_us(reduce_ms):.6f} "
        f"full_ffn_us={median_us(full_ms):.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
