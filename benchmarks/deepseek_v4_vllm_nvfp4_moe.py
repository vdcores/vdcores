#!/usr/bin/env python3
"""vLLM 0.27 NVFP4 routed-MoE kernel baseline for DSV4 decode.

This invokes the same FlashInfer TRTLLM-Gen entry point selected first by
vLLM's Blackwell NVFP4 MoE oracle. Inputs and weights are already quantized;
offline weight shuffling is deliberately outside the timed region.
"""

from __future__ import annotations

import argparse
import statistics

import torch


def graph_timings(run, *, warmup: int, samples: int, inner: int) -> list[float]:
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner):
            run()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    timings = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1.0e3 / inner)
    return timings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=6)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--inner", type=int, default=20)
    parser.add_argument("--swiglu-limit", type=float, default=7.0)
    parser.add_argument(
        "--cuda-profiler-capture",
        action="store_true",
        help="bracket one warmed expert-only call with cudaProfilerStart/Stop",
    )
    args = parser.parse_args()
    if not 0 < args.topk <= args.experts:
        parser.error("topk must be in [1,experts]")
    if args.hidden % 256 or args.intermediate % 128:
        parser.error("hidden must be K256 aligned and intermediate M128 aligned")

    import flashinfer
    import vllm
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.utils import (
        trtllm_moe_pack_topk_ids_weights,
    )
    from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
        prepare_static_weights_for_trtllm_fp4_moe,
    )
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        activation_to_flashinfer_int,
    )

    device = torch.device("cuda")
    e, h, i = args.experts, args.hidden, args.intermediate

    # 0x2 is exact +1 in E2M1. Unit UE4M3 block/global scales make the
    # generated tensors deterministic while avoiding setup-time BF16 GEMMs.
    w13 = torch.full((e, 2 * i, h // 2), 0x22, dtype=torch.uint8, device=device)
    w2 = torch.full((e, h, i // 2), 0x22, dtype=torch.uint8, device=device)
    w13_sf = torch.ones(
        (e, 2 * i, h // 16), dtype=torch.float8_e4m3fn, device=device
    )
    w2_sf = torch.ones(
        (e, h, i // 16), dtype=torch.float8_e4m3fn, device=device
    )
    w13, w13_sf, w2, w2_sf = prepare_static_weights_for_trtllm_fp4_moe(
        w13,
        w2,
        w13_sf.view(torch.uint8),
        w2_sf.view(torch.uint8),
        h,
        i,
        e,
        True,
    )

    hidden = torch.full((1, h // 2), 0x22, dtype=torch.uint8, device=device)
    hidden_sf = torch.ones((1, h // 16), dtype=torch.float8_e4m3fn, device=device)
    topk_ids = torch.arange(args.topk, dtype=torch.int32, device=device)[None]
    topk_weights = torch.full(
        (1, args.topk), 1.0 / args.topk, dtype=torch.float32, device=device
    )
    packed_topk = trtllm_moe_pack_topk_ids_weights(topk_ids, topk_weights)
    output = torch.empty((1, h), dtype=torch.bfloat16, device=device)
    unit_global = torch.ones((e,), dtype=torch.float32, device=device)
    clamp = torch.full(
        (e,), args.swiglu_limit, dtype=torch.float32, device=device
    )
    activation_type = activation_to_flashinfer_int(
        MoEActivation.SWIGLUOAI_UNINTERLEAVE
    )

    def experts(packed_routes: torch.Tensor) -> None:
        flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe(
            topk_ids=packed_routes,
            routing_bias=None,
            hidden_states=hidden,
            hidden_states_scale=hidden_sf,
            gemm1_weights=w13,
            gemm1_weights_scale=w13_sf,
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=clamp,
            gemm2_weights=w2,
            gemm2_weights_scale=w2_sf,
            gemm2_bias=None,
            output1_scale_scalar=unit_global,
            output1_scale_gate_scalar=unit_global,
            output2_scale_scalar=unit_global,
            num_experts=e,
            top_k=args.topk,
            n_group=0,
            topk_group=0,
            intermediate_size=i,
            local_expert_offset=0,
            local_num_experts=e,
            routed_scaling_factor=None,
            routing_method_type=1,
            do_finalize=True,
            activation_type=activation_type,
            per_token_scale=None,
            output=output,
            tune_max_num_tokens=8192,
        )

    def experts_only() -> None:
        experts(packed_topk)

    def vllm_dispatch() -> None:
        routes = trtllm_moe_pack_topk_ids_weights(topk_ids, topk_weights)
        experts(routes)

    experts_only()
    torch.cuda.synchronize()
    if not torch.isfinite(output.float()).all():
        raise AssertionError("vLLM NVFP4 MoE produced non-finite output")
    first = output.clone()
    experts_only()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, first, rtol=0, atol=0)

    if args.cuda_profiler_capture:
        for _ in range(args.warmup):
            experts_only()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        experts_only()
        torch.cuda.cudart().cudaProfilerStop()
        torch.cuda.synchronize()

    expert_times = graph_timings(
        experts_only, warmup=args.warmup, samples=args.samples, inner=args.inner
    )
    dispatch_times = graph_timings(
        vllm_dispatch, warmup=args.warmup, samples=args.samples, inner=args.inner
    )

    print(
        "DSV4_VLLM_NVFP4_MOE_RESULT "
        f"vllm={vllm.__version__} "
        f"vllm_backend=FLASHINFER_TRTLLM flashinfer={flashinfer.__version__} "
        f"torch={torch.__version__} device={torch.cuda.get_device_name()} "
        f"experts={e} topk={args.topk} hidden={h} intermediate={i} "
        f"expert_graph_min_us={min(expert_times):.6f} "
        f"expert_graph_median_us={statistics.median(expert_times):.6f} "
        f"expert_graph_max_us={max(expert_times):.6f} "
        f"dispatch_graph_median_us={statistics.median(dispatch_times):.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
