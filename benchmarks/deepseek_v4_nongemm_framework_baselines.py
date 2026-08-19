#!/usr/bin/env python3
"""Matched framework baselines for selected DeepSeek-V4 non-GEMM tasks."""

from __future__ import annotations

import argparse
import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass
class Case:
    function: Callable[[], object]
    validate: Callable[[], float]
    framework: str
    operation: str
    shape: str


def _max_abs(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected.float()).abs().max().item())


def _time_once(function: Callable[[], object]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1.0e3


def _bench_graph(
    function: Callable[[], object],
    *,
    warmup: int,
    samples: int,
    inner: int,
) -> tuple[float, float, float, float]:
    # First call resolves JIT/cubin loading and validates the callable.  Cold is
    # the next ordinary uncaptured device invocation, not Python compilation.
    function()
    torch.cuda.synchronize()
    cold_us = _time_once(function)
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner):
            function()
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
    return cold_us, min(timings), statistics.median(timings), max(timings)


def build_deepgemm_fp8(
    device: torch.device,
    generator: torch.Generator,
    *,
    k: int,
    with_rms: bool,
) -> Case:
    import vllm
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8_packed_for_deepgemm,
    )

    source = torch.randn(
        (1, k), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.25
    source[0, ::257] *= 8
    if with_rms:
        weight = (
            torch.randn(
                (k,), generator=generator, dtype=torch.bfloat16, device=device
            )
            * 0.05
            + 1.0
        )
        quant_input = torch.empty_like(source)
    else:
        weight = None
        quant_input = source
    captured: dict[str, tuple[torch.Tensor, ...]] = {}

    def function():
        if weight is not None:
            ops.rms_norm(quant_input, source, weight, 1.0e-6)
        captured["output"] = per_token_group_quant_fp8_packed_for_deepgemm(
            quant_input, group_size=128, use_ue8m0=True
        )
        return captured["output"]

    def validate() -> float:
        quantized, scales, *_ = captured["output"]
        if quantized.numel() != k or scales.numel() < k // (128 * 4):
            raise AssertionError("DeepGEMM quantizer returned an invalid shape")
        if not torch.isfinite(quantized.float()).all():
            raise AssertionError("DeepGEMM quantizer returned non-finite data")
        return 0.0

    return Case(
        function,
        validate,
        f"vLLM-{vllm.__version__}/DeepGEMM",
        "rms_norm+packed_fp8_quant128" if with_rms else "packed_fp8_quant128",
        f"rows1_k{k}",
    )


def build_flashinfer_nvfp4(
    device: torch.device,
    generator: torch.Generator,
    *,
    k: int,
) -> Case:
    import flashinfer
    from flashinfer import SfLayout

    source = torch.randn(
        (1, k), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.25
    source[0, ::257] *= 8
    global_sf_value = float(
        ((448.0 * 6.0) / source.float().abs().amax()).item()
    )
    # FlashInfer converts this API argument to a host scalar before launch.
    # Keep it on the host so the production entry point remains graph-safe.
    global_sf = torch.tensor((global_sf_value,), dtype=torch.float32)
    captured: dict[str, tuple[torch.Tensor, ...]] = {}

    def function():
        captured["output"] = flashinfer.nvfp4_quantize(
            source,
            global_sf,
            sfLayout=SfLayout.layout_128x4,
            do_shuffle=False,
            sf_vec_size=16,
            enable_pdl=False,
            backend="cuda",
            per_token_activation=True,
        )
        return captured["output"]

    def validate() -> float:
        quantized, scales, *_ = captured["output"]
        if quantized.numel() != k // 2 or scales.numel() < k // 16:
            raise AssertionError("FlashInfer NVFP4 quantizer returned an invalid shape")
        return 0.0

    return Case(
        function,
        validate,
        f"FlashInfer-{flashinfer.__version__}",
        "nvfp4_quantize",
        f"rows1_k{k}_sf16",
    )


def build_mhc_pre_rms(
    device: torch.device, generator: torch.Generator
) -> Case:
    import vllm
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        mhc_pre_big_fuse_with_norm_tilelang,
    )

    residual = torch.randn(
        (1, 4, 4096), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.125
    mixes = torch.randn(
        (1, 1, 24), generator=generator, dtype=torch.float32, device=device
    ) * 0.1
    square_sum = residual.float().square().sum().reshape(1, 1)
    scale = torch.tensor((0.5, 0.75, 1.25), dtype=torch.float32, device=device)
    base = torch.randn(
        (24,), generator=generator, dtype=torch.float32, device=device
    ) * 0.05
    weight = (
        torch.randn(
            (4096,), generator=generator, dtype=torch.bfloat16, device=device
        )
        * 0.05
        + 1.0
    )
    post = torch.empty((1, 4), dtype=torch.float32, device=device)
    comb = torch.empty((1, 16), dtype=torch.float32, device=device)
    output = torch.empty((1, 4096), dtype=torch.bfloat16, device=device)

    def function():
        return mhc_pre_big_fuse_with_norm_tilelang(
            mixes,
            square_sum,
            scale,
            base,
            residual,
            post,
            comb,
            output,
            weight,
            4096,
            1.0e-6,
            1.0e-6,
            1.0e-6,
            2.0,
            20,
            1.0e-6,
            1,
            4,
            24,
        )

    def validate() -> float:
        if not torch.isfinite(output.float()).all():
            raise AssertionError("vLLM mHC pre/RMS returned non-finite output")
        return 0.0

    return Case(
        function,
        validate,
        f"vLLM-{vllm.__version__}",
        "mhc_pre_big_fuse_with_norm_tilelang",
        "tokens1_hc4_hidden4096_presupplied_projection",
    )


def build_mhc_post(
    device: torch.device, generator: torch.Generator
) -> Case:
    import vllm
    from vllm.model_executor.kernels.mhc.tilelang_kernels import mhc_post_tilelang

    branch = torch.randn(
        (1, 4096), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.125
    residual = torch.randn(
        (1, 4, 4096), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.125
    post = torch.rand(
        (1, 4), generator=generator, dtype=torch.float32, device=device
    )
    comb = torch.rand(
        (1, 4, 4), generator=generator, dtype=torch.float32, device=device
    )
    output = torch.empty_like(residual)

    def function():
        return mhc_post_tilelang(
            comb, residual, post, branch, output, 4, 4096
        )

    def validate() -> float:
        expected = (
            post[:, :, None] * branch[:, None, :].float()
            + torch.einsum("tij,tih->tjh", comb, residual.float())
        ).to(torch.bfloat16)
        torch.testing.assert_close(output, expected, rtol=2.0e-2, atol=1.0e-2)
        return _max_abs(output, expected)

    return Case(
        function,
        validate,
        f"vLLM-{vllm.__version__}",
        "mhc_post_tilelang",
        "tokens1_hc4_hidden4096_bf16",
    )


def build_route(
    device: torch.device,
    generator: torch.Generator,
    *,
    hash_routing: bool,
) -> Case:
    import vllm
    import vllm._custom_ops as ops

    logits = torch.randn(
        (1, 256), generator=generator, dtype=torch.bfloat16, device=device
    )
    bias = torch.randn(
        (256,), generator=generator, dtype=torch.float32, device=device
    ) * 0.1
    fixed = torch.tensor(
        ((9, 71, 5, 255, 130, 44),), dtype=torch.int32, device=device
    )
    topk_weights = torch.empty((1, 6), dtype=torch.float32, device=device)
    topk_indices = torch.empty((1, 6), dtype=torch.int32, device=device)
    token_expert_indices = torch.empty((1, 6), dtype=torch.int32, device=device)
    input_tokens = (
        torch.zeros((1,), dtype=torch.int32, device=device) if hash_routing else None
    )
    hash_table = fixed if hash_routing else None

    def function():
        return ops.topk_hash_softplus_sqrt(
            topk_weights,
            topk_indices,
            token_expert_indices,
            logits,
            True,
            1.5,
            bias,
            input_tokens,
            hash_table,
            None,
        )

    def validate() -> float:
        scores = torch.nn.functional.softplus(logits.float()).sqrt()
        expected_indices = fixed if hash_routing else (scores + bias).topk(6).indices
        expected = scores.gather(1, expected_indices.long())
        expected = expected / expected.sum(dim=1, keepdim=True) * 1.5
        torch.testing.assert_close(topk_indices, expected_indices.to(torch.int32))
        torch.testing.assert_close(topk_weights, expected, rtol=2.0e-5, atol=2.0e-5)
        return _max_abs(topk_weights, expected)

    return Case(
        function,
        validate,
        f"vLLM-{vllm.__version__}",
        "topk_hash_softplus_sqrt",
        f"tokens1_experts256_top6_hash={int(hash_routing)}",
    )


def build_argmax(
    device: torch.device, generator: torch.Generator
) -> Case:
    import vllm

    logits = torch.randn(
        (129_280,), generator=generator, dtype=torch.bfloat16, device=device
    )
    logits[127_777] = 64.0
    captured: dict[str, torch.Tensor] = {}

    def function():
        captured["output"] = torch.argmax(logits)
        return captured["output"]

    def validate() -> float:
        if int(captured["output"].item()) != 127_777:
            raise AssertionError("torch.argmax returned the wrong token")
        return 0.0

    return Case(
        function,
        validate,
        f"vLLM-{vllm.__version__}/torch",
        "argmax",
        "vocab129280_bf16",
    )


def build_flashinfer_attention(
    device: torch.device,
    generator: torch.Generator,
    *,
    rows: int,
) -> Case:
    import flashinfer
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache

    page_size = 16
    pages = math.ceil(rows / page_size)
    q = torch.randn(
        (1, 64, 512), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.125
    kv_flat = torch.zeros(
        (pages * page_size, 512), dtype=torch.bfloat16, device=device
    )
    kv_flat[:rows].copy_(
        torch.randn(
            (rows, 512), generator=generator, dtype=torch.bfloat16, device=device
        ) * 0.125
    )
    kv = kv_flat.view(pages, 1, page_size, 512)
    block_tables = torch.arange(pages, dtype=torch.int32, device=device).view(1, -1)
    seq_lens = torch.tensor((rows,), dtype=torch.int32, device=device)
    sink = torch.linspace(-0.5, 0.5, 64, dtype=torch.float32, device=device)
    output = torch.empty_like(q)
    workspace = torch.empty((128 * 1024 * 1024,), dtype=torch.uint8, device=device)

    def function():
        return trtllm_batch_decode_with_kv_cache(
            q,
            (kv, kv),
            workspace,
            block_tables,
            seq_lens,
            rows,
            bmm1_scale=1.0 / math.sqrt(512.0),
            out=output,
            sinks=sink,
            kv_layout="HND",
            enable_pdl=False,
            backend="trtllm-gen",
        )

    def validate() -> float:
        selected = kv_flat[:rows].float()
        scores = q[0].float() @ selected.t() / math.sqrt(512.0)
        probability = torch.softmax(
            torch.cat((scores, sink[:, None]), dim=1), dim=1
        )[:, :rows]
        expected = (probability @ selected).to(torch.bfloat16)
        torch.testing.assert_close(output[0], expected, rtol=3.0e-2, atol=1.0e-2)
        return _max_abs(output[0], expected)

    return Case(
        function,
        validate,
        f"FlashInfer-{flashinfer.__version__}",
        "trtllm_batch_decode_with_kv_cache",
        f"batch1_qh64_kvh1_d512_rows{rows}_sink",
    )


CASES: dict[str, Callable[[torch.device, torch.Generator], Case]] = {
    "deepgemm_fp8_k1024": lambda d, g: build_deepgemm_fp8(
        d, g, k=1024, with_rms=False
    ),
    "deepgemm_fp8_k2048": lambda d, g: build_deepgemm_fp8(
        d, g, k=2048, with_rms=False
    ),
    "deepgemm_fp8_k4096": lambda d, g: build_deepgemm_fp8(
        d, g, k=4096, with_rms=False
    ),
    "deepgemm_rms_fp8_k1024": lambda d, g: build_deepgemm_fp8(
        d, g, k=1024, with_rms=True
    ),
    "flashinfer_nvfp4_k2048": lambda d, g: build_flashinfer_nvfp4(d, g, k=2048),
    "flashinfer_nvfp4_k4096": lambda d, g: build_flashinfer_nvfp4(d, g, k=4096),
    "vllm_mhc_pre_rms": build_mhc_pre_rms,
    "vllm_mhc_post": build_mhc_post,
    "vllm_route_score": lambda d, g: build_route(d, g, hash_routing=False),
    "vllm_route_hash": lambda d, g: build_route(d, g, hash_routing=True),
    "vllm_argmax": build_argmax,
    "flashinfer_attention_rows128": lambda d, g: build_flashinfer_attention(
        d, g, rows=128
    ),
    "flashinfer_attention_rows129": lambda d, g: build_flashinfer_attention(
        d, g, rows=129
    ),
    "flashinfer_attention_rows160": lambda d, g: build_flashinfer_attention(
        d, g, rows=160
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--inner", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()
    if min(args.warmup, args.samples, args.inner) <= 0:
        parser.error("timing counts must be positive")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    case = CASES[args.case](device, generator)
    cold_us, minimum, median, maximum = _bench_graph(
        case.function,
        warmup=args.warmup,
        samples=args.samples,
        inner=args.inner,
    )
    max_abs = case.validate()
    props = torch.cuda.get_device_properties(device)
    print(
        "DSV4_NONGEMM_BASELINE_RESULT "
        f"case={args.case} framework={case.framework} operation={case.operation} "
        f"shape={case.shape} cold_us={cold_us:.6f} min_us={minimum:.6f} "
        f"median_us={median:.6f} max_us={maximum:.6f} "
        f"max_abs={max_abs:.8f} status=PASS device={props.name!r} "
        f"cc={props.major}.{props.minor}",
        flush=True,
    )


if __name__ == "__main__":
    main()
