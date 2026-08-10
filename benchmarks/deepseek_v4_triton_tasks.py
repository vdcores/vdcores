#!/usr/bin/env python3
"""Matched Triton baselines for DeepSeek-V4-Flash functional task shapes."""

from __future__ import annotations

import argparse
import math
import statistics
from collections.abc import Callable

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fp8_quant128_kernel(input_ptr, output_ptr, scale_bits_ptr, K: tl.constexpr):
    block = tl.program_id(0)
    offsets = block * 128 + tl.arange(0, 128)
    values = tl.load(input_ptr + offsets).to(tl.float32)
    maximum = tl.max(tl.abs(values), axis=0)
    requested = maximum / 448.0
    exponent = tl.where(requested > 0.0, tl.ceil(tl.log2(requested)), -127.0)
    exponent = tl.minimum(tl.maximum(exponent, -127.0), 127.0)
    scale = tl.exp2(exponent)
    quantized = tl.minimum(tl.maximum(values / scale, -448.0), 448.0)
    tl.store(output_ptr + offsets, quantized)
    tl.store(scale_bits_ptr + block, (exponent + 127.0).to(tl.uint8))


@triton.jit
def _nvfp4_e2m1_codes(values, denominator):
    normalized = values / denominator
    magnitude = tl.abs(normalized)
    positive_code = tl.zeros(values.shape, tl.int32)
    positive_code += (magnitude > 0.25).to(tl.int32)
    positive_code += (magnitude > 0.75).to(tl.int32)
    positive_code += (magnitude > 1.25).to(tl.int32)
    positive_code += (magnitude > 1.75).to(tl.int32)
    positive_code += (magnitude > 2.5).to(tl.int32)
    positive_code += (magnitude > 3.5).to(tl.int32)
    positive_code += (magnitude > 5.0).to(tl.int32)
    return tl.where(
        (normalized < 0.0) & (positive_code > 0),
        positive_code + 8,
        positive_code,
    )


@triton.jit
def nvfp4_quant16_kernel(
    input_ptr,
    global_scale_ptr,
    output_ptr,
    scale_bits_ptr,
    K: tl.constexpr,
):
    block = tl.program_id(0)
    offsets = block * 16 + tl.arange(0, 16)
    values = tl.load(input_ptr + offsets).to(tl.float32)
    global_scale = tl.load(global_scale_ptr)
    requested = tl.minimum(
        tl.maximum(tl.max(tl.abs(values), axis=0) / (6.0 * global_scale), 2.0**-9),
        448.0,
    )

    is_subnormal = requested < 2.0**-6
    subnormal_mantissa = tl.ceil(requested * 512.0)
    subnormal_value = subnormal_mantissa / 512.0
    exponent = tl.floor(tl.log2(requested))
    exponent_scale = tl.exp2(exponent)
    mantissa = tl.ceil((requested / exponent_scale - 1.0) * 8.0)
    carry = mantissa >= 8.0
    exponent = exponent + carry.to(tl.float32)
    mantissa = tl.where(carry, 0.0, mantissa)
    normal_value = tl.exp2(exponent) * (1.0 + mantissa / 8.0)
    block_scale = tl.where(is_subnormal, subnormal_value, normal_value)

    subnormal_code = subnormal_mantissa.to(tl.int32)
    normal_code = ((exponent + 7.0) * 8.0 + mantissa).to(tl.int32)
    scale_code = tl.where(is_subnormal, subnormal_code, normal_code)
    tl.store(scale_bits_ptr + block, scale_code.to(tl.uint8))

    pair = tl.arange(0, 8)
    low_values = tl.load(input_ptr + block * 16 + pair * 2).to(tl.float32)
    high_values = tl.load(input_ptr + block * 16 + pair * 2 + 1).to(tl.float32)
    denominator = block_scale * global_scale
    low_codes = _nvfp4_e2m1_codes(low_values, denominator)
    high_codes = _nvfp4_e2m1_codes(high_values, denominator)
    packed = low_codes | (high_codes << 4)
    tl.store(output_ptr + block * 8 + pair, packed.to(tl.uint8))


@triton.jit
def sparse_attention512_kernel(
    q_ptr,
    kv_ptr,
    indices_ptr,
    sink_ptr,
    output_ptr,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    head = tl.program_id(0)
    dimensions = tl.arange(0, 512)
    query = tl.load(q_ptr + head * 512 + dimensions).to(tl.float32)
    running_max = tl.load(sink_ptr + head).to(tl.float32)
    running_sum = 1.0
    accumulator = tl.zeros((512,), tl.float32)
    for start in range(0, TOPK, BLOCK_N):
        selected = start + tl.arange(0, BLOCK_N)
        valid = selected < TOPK
        rows = tl.load(indices_ptr + selected, mask=valid, other=-1)
        valid = valid & (rows >= 0)
        values = tl.load(
            kv_ptr + rows[:, None] * 512 + dimensions[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(values * query[None, :], axis=1) * 0.04419417382415922
        scores = tl.where(valid, scores, -float("inf"))
        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, block_max)
        old_scale = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max)
        running_sum = running_sum * old_scale + tl.sum(probabilities, axis=0)
        accumulator = accumulator * old_scale + tl.sum(
            probabilities[:, None] * values, axis=0
        )
        running_max = new_max
    tl.store(
        output_ptr + head * 512 + dimensions,
        accumulator / running_sum,
    )


@triton.jit
def index_score_kernel(q_ptr, kv_ptr, weights_ptr, output_ptr):
    row = tl.program_id(0)
    heads = tl.arange(0, 64)
    dimensions = tl.arange(0, 128)
    query = tl.load(q_ptr + heads[:, None] * 128 + dimensions[None, :]).to(
        tl.float32
    )
    key = tl.load(kv_ptr + row * 128 + dimensions).to(tl.float32)
    dots = tl.sum(query * key[None, :], axis=1)
    weights = tl.load(weights_ptr + heads).to(tl.float32)
    score = tl.sum(tl.maximum(dots, 0.0) * weights, axis=0)
    tl.store(output_ptr + row, score)


@triton.jit
def rmsnorm_kernel(input_ptr, weight_ptr, output_ptr, WIDTH: tl.constexpr):
    offsets = tl.arange(0, WIDTH)
    values = tl.load(input_ptr + offsets).to(tl.float32)
    weights = tl.load(weight_ptr + offsets).to(tl.float32)
    inverse_rms = tl.rsqrt(tl.sum(values * values, axis=0) / WIDTH + 1.0e-6)
    tl.store(output_ptr + offsets, values * inverse_rms * weights)


@triton.jit
def bounded_swiglu_kernel(gate_ptr, up_ptr, output_ptr, K: tl.constexpr):
    block = tl.program_id(0)
    offsets = block * 256 + tl.arange(0, 256)
    mask = offsets < K
    gate = tl.minimum(tl.load(gate_ptr + offsets, mask=mask), 10.0).to(tl.float32)
    up = tl.minimum(
        tl.maximum(tl.load(up_ptr + offsets, mask=mask), -10.0), 10.0
    ).to(tl.float32)
    output = gate * tl.sigmoid(gate) * up
    tl.store(output_ptr + offsets, output, mask=mask)


def bench_cuda_graph(
    function: Callable[[], object],
    *,
    warmup: int,
    samples: int,
    inner: int,
) -> tuple[float, float, float]:
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
    return min(timings), statistics.median(timings), max(timings)


def emit(name: str, shape: str, function: Callable[[], object], args) -> None:
    minimum, median, maximum = bench_cuda_graph(
        function,
        warmup=args.warmup,
        samples=args.samples,
        inner=args.inner,
    )
    print(
        "DSV4_TRITON_RESULT "
        f"task={name} shape={shape} min_us={minimum:.6f} "
        f"median_us={median:.6f} max_us={maximum:.6f}",
        flush=True,
    )


def run_quantization(device, generator, args) -> None:
    source = torch.randn(
        (4096,), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.125
    source[::257] *= 16

    fp8_output = torch.empty_like(source, dtype=torch.float8_e4m3fn)
    fp8_scale = torch.empty(
        (32,), dtype=torch.float8_e8m0fnu, device=device
    )
    fp8_fn = lambda: fp8_quant128_kernel[(32,)](
        source, fp8_output, fp8_scale.view(torch.uint8), K=4096
    )
    fp8_fn()
    blocks = source.float().reshape(-1, 128)
    exponents = torch.ceil(
        torch.log2(
            (blocks.abs().amax(dim=1) / 448.0).clamp_min(2.0**-127)
        )
    ).clamp(-127, 127)
    expected_scale = torch.exp2(exponents).to(torch.float8_e8m0fnu)
    expected_fp8 = (
        source.float()
        / expected_scale.float().repeat_interleave(128)
    ).clamp(-448, 448).to(torch.float8_e4m3fn)
    torch.testing.assert_close(fp8_output.view(torch.uint8), expected_fp8.view(torch.uint8))
    torch.testing.assert_close(fp8_scale.view(torch.uint8), expected_scale.view(torch.uint8))
    emit("fp8_activation_quant128", "k4096", fp8_fn, args)

    global_scale = (
        source.float().abs().amax() / (6.0 * 448.0)
    ).reshape(1)
    fp4_output = torch.empty((2048,), dtype=torch.uint8, device=device)
    fp4_scale = torch.empty((256,), dtype=torch.float8_e4m3fn, device=device)
    fp4_fn = lambda: nvfp4_quant16_kernel[(256,)](
        source,
        global_scale,
        fp4_output,
        fp4_scale.view(torch.uint8),
        K=4096,
    )
    fp4_fn()
    scale_values = torch.arange(256, dtype=torch.uint8, device=device)
    scale_values = scale_values.view(torch.float8_e4m3fn).float()
    scale_values = torch.unique(
        scale_values[torch.isfinite(scale_values) & (scale_values > 0)],
        sorted=True,
    )
    requested = (
        source.float().reshape(-1, 16).abs().amax(dim=1)
        / (6.0 * global_scale)
    ).clamp(min=scale_values[0], max=448.0)
    expected_scale = scale_values[
        torch.searchsorted(scale_values, requested, right=False)
    ].to(torch.float8_e4m3fn)
    codebook = torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0),
        dtype=torch.float32,
        device=device,
    )
    normalized = source.float().reshape(-1, 16) / (
        expected_scale.float()[:, None] * global_scale
    )
    expected_codes = (
        (normalized[..., None] - codebook).abs().argmin(dim=-1).to(torch.uint8)
    ).reshape(-1)
    expected_packed = (
        expected_codes[0::2] | (expected_codes[1::2] << 4)
    )
    torch.testing.assert_close(fp4_output, expected_packed, rtol=0, atol=0)
    torch.testing.assert_close(
        fp4_scale.view(torch.uint8), expected_scale.view(torch.uint8), rtol=0, atol=0
    )
    emit("nvfp4_activation_quant16", "k4096", fp4_fn, args)


def run_attention(device, generator, args) -> None:
    q = torch.randn(
        (64, 512), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.125
    kv = torch.randn(
        (768, 512), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.125
    indices = torch.randperm(768, generator=generator, device=device)[:512].to(
        torch.int32
    )
    sink = torch.linspace(-0.5, 0.5, 64, dtype=torch.float32, device=device)
    output = torch.empty_like(q)
    function = lambda: sparse_attention512_kernel[(64,)](
        q, kv, indices, sink, output, TOPK=512, BLOCK_N=16, num_warps=8
    )
    function()
    selected = kv[indices.long()].float()
    scores = q.float() @ selected.t() / math.sqrt(512.0)
    probabilities = torch.softmax(
        torch.cat((scores, sink[:, None]), dim=1), dim=1
    )[:, :-1]
    expected = (probabilities @ selected).to(torch.bfloat16)
    torch.testing.assert_close(output, expected, rtol=3.0e-2, atol=1.0e-2)
    emit("sparse_attention", "h64_d512_k512", function, args)


def run_indexer(device, generator, args) -> None:
    rows = args.index_rows
    q = torch.randn(
        (64, 128), generator=generator, dtype=torch.bfloat16, device=device
    )
    kv = torch.randn(
        (rows, 128), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.125
    weights = torch.randn(
        (64,), generator=generator, dtype=torch.float32, device=device
    ) / math.sqrt(128 * 64)
    scores = torch.empty((rows,), dtype=torch.float32, device=device)
    score_fn = lambda: index_score_kernel[(rows,)](
        q, kv, weights, scores, num_warps=4
    )
    score_fn()
    expected = ((q.float() @ kv.float().t()).relu() * weights[:, None]).sum(0)
    torch.testing.assert_close(scores, expected, rtol=2.0e-4, atol=2.0e-4)
    emit("index_score", f"rows{rows}_h64_d128", score_fn, args)

    values = torch.empty((512,), dtype=scores.dtype, device=device)
    indices = torch.empty((512,), dtype=torch.int64, device=device)

    def topk_fn():
        torch.topk(scores, 512, out=(values, indices))

    topk_fn()
    emit("torch_topk", f"rows{rows}_k512", topk_fn, args)


def run_norm_activation(device, generator, args) -> None:
    for width in (512, 1024):
        source = torch.randn(
            (width,), generator=generator, dtype=torch.bfloat16, device=device
        ) * 0.25
        weight = torch.randn(
            (width,), generator=generator, dtype=torch.bfloat16, device=device
        ) * 0.05 + 1.0
        output = torch.empty_like(source)
        function = lambda width=width, source=source, weight=weight, output=output: (
            rmsnorm_kernel[(1,)](source, weight, output, WIDTH=width, num_warps=4)
        )
        function()
        expected = (
            source.float()
            * torch.rsqrt(source.float().square().mean() + 1.0e-6)
            * weight.float()
        ).to(torch.bfloat16)
        torch.testing.assert_close(output, expected, rtol=2.0e-2, atol=2.0e-2)
        emit("rmsnorm", f"k{width}", function, args)

    gate = torch.randn(
        (2048,), generator=generator, dtype=torch.bfloat16, device=device
    ) * 8.0
    up = torch.randn(
        (2048,), generator=generator, dtype=torch.bfloat16, device=device
    ) * 8.0
    output = torch.empty_like(gate)
    function = lambda: bounded_swiglu_kernel[(8,)](
        gate, up, output, K=2048, num_warps=4
    )
    function()
    expected = (
        F.silu(gate.float().clamp(max=10.0))
        * up.float().clamp(-10.0, 10.0)
    ).to(torch.bfloat16)
    torch.testing.assert_close(output, expected, rtol=2.0e-2, atol=6.0e-2)
    emit("bounded_swiglu", "k2048", function, args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=("all", "quantization", "attention", "indexer", "norm-activation"),
        default="all",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--inner", type=int, default=20)
    parser.add_argument("--index-rows", type=int, default=640)
    args = parser.parse_args()
    if min(args.warmup, args.samples, args.inner) <= 0:
        raise ValueError("timing counts must be positive")
    if args.index_rows < 512 or args.index_rows > 0xFFFF:
        raise ValueError("index rows must be in [512,65535]")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260810)
    tasks = {
        "quantization": run_quantization,
        "attention": run_attention,
        "indexer": run_indexer,
        "norm-activation": run_norm_activation,
    }
    selected = tasks.items() if args.task == "all" else ((args.task, tasks[args.task]),)
    for _, function in selected:
        function(device, generator, args)
    print(
        f"DSV4_TRITON_SUMMARY status=PASS triton={triton.__version__} "
        f"torch={torch.__version__} device={torch.cuda.get_device_name(device)!r}",
        flush=True,
    )


if __name__ == "__main__":
    main()
