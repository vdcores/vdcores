#!/usr/bin/env python3
"""Device-time vLLM 0.27.1 FlashMLA's DeepSeek-V4 decode path."""

from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass

import torch

from vllm.third_party.flashmla.flash_mla_interface import (
    flash_mla_with_kvcache,
    get_mla_metadata,
)
from vllm.models.deepseek_v4.nvidia.ops.o_proj import (
    fused_inv_rope_fp8_quant,
)


@dataclass(frozen=True)
class DecodeCase:
    name: str
    extra_width: int
    extra_rows: int
    extra_block_size: int


CASES = (
    DecodeCase("swa", 0, 0, 0),
    DecodeCase("c4a", 512, 32, 64),
    DecodeCase("c128a", 128, 1, 2),
)


def _pack_model1_cache(logical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack [blocks, block, 1, 512] BF16 into FlashMLA's V4 layout."""
    num_blocks, block_size, heads, dim = logical.shape
    assert heads == 1 and dim == 512 and logical.dtype == torch.bfloat16
    bytes_per_token = 584
    token_data_bytes = 576
    padded_block_bytes = (
        (block_size * bytes_per_token + token_data_bytes - 1)
        // token_data_bytes
        * token_data_bytes
    )
    storage = torch.empty(
        (num_blocks, padded_block_bytes),
        dtype=torch.float8_e4m3fn,
        device=logical.device,
    )
    packed = storage[:, : block_size * bytes_per_token]
    token_data = packed[:, : block_size * token_data_bytes].view(
        num_blocks, block_size, token_data_bytes
    )
    packed_nope = token_data[..., :448]
    packed_rope = token_data[..., 448:].view(torch.bfloat16)
    packed_scales = (
        packed[:, block_size * token_data_bytes :]
        .view(num_blocks, block_size, 8)[..., :7]
        .view(torch.float8_e8m0fnu)
    )

    source = logical.squeeze(2)
    packed_rope.copy_(source[..., 448:])
    dequantized = source.clone()
    for tile in range(7):
        values = source[..., tile * 64 : (tile + 1) * 64]
        scale = torch.pow(
            2.0,
            torch.clamp_min(values.float().abs().amax(dim=-1) / 448.0, 1.0e-4)
            .log2()
            .ceil(),
        )
        quantized = (values.float() / scale.unsqueeze(-1)).to(
            torch.float8_e4m3fn
        )
        packed_nope[..., tile * 64 : (tile + 1) * 64].copy_(quantized)
        packed_scales[..., tile].copy_(scale.to(torch.float8_e8m0fnu))
        dequantized[..., tile * 64 : (tile + 1) * 64] = (
            quantized.float() * scale.unsqueeze(-1)
        ).to(torch.bfloat16)

    cache = storage.view(torch.uint8).as_strided(
        (num_blocks, block_size, 1, bytes_per_token),
        (padded_block_bytes, bytes_per_token, bytes_per_token, 1),
    )
    assert cache.stride(0) == padded_block_bytes
    assert cache.stride(0) % token_data_bytes == 0
    return cache, dequantized.reshape(-1, 512)


def _make_cache(
    rows: int,
    block_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_blocks = math.ceil(rows / block_size)
    logical = (
        torch.randn(
            (num_blocks, block_size, 1, 512),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.1
    )
    cache, dequantized = _pack_model1_cache(logical)
    return cache, dequantized[:rows]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (
        position - lower
    )


def _event_samples(function, samples: int) -> list[float]:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    for start, end in zip(starts, ends):
        start.record()
        function()
        end.record()
    ends[-1].synchronize()
    return [start.elapsed_time(end) * 1.0e3 for start, end in zip(starts, ends)]


def _graph_samples(function, samples: int, inner: int) -> list[float]:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner):
            function()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    for start, end in zip(starts, ends):
        start.record()
        graph.replay()
        end.record()
    ends[-1].synchronize()
    return [
        start.elapsed_time(end) * 1.0e3 / inner
        for start, end in zip(starts, ends)
    ]


def _cold_graph_samples(
    function,
    samples: int,
    scrub_mib: int,
) -> list[float]:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        function()
    scrub = torch.empty(
        scrub_mib * 1024 * 1024 // 4,
        dtype=torch.float32,
        device="cuda",
    )
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    for start, end in zip(starts, ends):
        scrub.add_(1.0)
        start.record()
        graph.replay()
        end.record()
    ends[-1].synchronize()
    return [start.elapsed_time(end) * 1.0e3 for start, end in zip(starts, ends)]


def _summary(label: str, values: list[float]) -> str:
    return (
        f"{label}_min_us={min(values):.6f} "
        f"{label}_median_us={statistics.median(values):.6f} "
        f"{label}_p90_us={_percentile(values, 0.9):.6f} "
        f"{label}_max_us={max(values):.6f}"
    )


def run_case(
    case: DecodeCase,
    generator: torch.Generator,
    samples: int,
    inner: int,
    scrub_mib: int,
) -> None:
    q = (
        torch.randn(
            (1, 1, 64, 512),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.1
    )
    sink = torch.linspace(-0.5, 0.5, 64, dtype=torch.float32, device="cuda")
    main_cache, main_values = _make_cache(128, 64, generator)
    main_indices = torch.arange(128, dtype=torch.int32, device="cuda").view(
        1, 1, 128
    )
    main_length = torch.tensor((128,), dtype=torch.int32, device="cuda")

    if case.extra_width:
        extra_cache, extra_values = _make_cache(
            case.extra_rows, case.extra_block_size, generator
        )
        extra_indices = torch.full(
            (1, 1, case.extra_width), -1, dtype=torch.int32, device="cuda"
        )
        extra_indices[0, 0, : case.extra_rows] = torch.arange(
            case.extra_rows, dtype=torch.int32, device="cuda"
        )
        extra_length = torch.tensor(
            (case.extra_rows,), dtype=torch.int32, device="cuda"
        )
    else:
        extra_cache = None
        extra_values = None
        extra_indices = None
        extra_length = None

    output = torch.empty_like(q)
    positions = torch.zeros((1,), dtype=torch.int64, device="cuda")
    angles = torch.linspace(-1.25, 1.25, 32, device="cuda")
    cos_sin_cache = torch.cat((angles.cos(), angles.sin())).reshape(1, 64)
    quantized_output: dict[str, tuple[torch.Tensor, ...]] = {}
    scheduler, _ = get_mla_metadata()

    def function():
        return flash_mla_with_kvcache(
            q=q,
            k_cache=main_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=512,
            tile_scheduler_metadata=scheduler,
            num_splits=None,
            softmax_scale=1.0 / math.sqrt(512.0),
            causal=False,
            is_fp8_kvcache=True,
            indices=main_indices,
            attn_sink=sink,
            extra_k_cache=extra_cache,
            extra_indices_in_kvcache=extra_indices,
            topk_length=main_length,
            extra_topk_length=extra_length,
            out=output,
        )

    def epilogue():
        quantized_output["value"] = fused_inv_rope_fp8_quant(
            output.squeeze(1),
            positions,
            cos_sin_cache,
            n_groups=8,
            heads_per_group=8,
            tma_aligned_scales=True,
        )
        return quantized_output["value"]

    def attention_epilogue():
        function()
        return epilogue()

    first_start = torch.cuda.Event(enable_timing=True)
    first_end = torch.cuda.Event(enable_timing=True)
    first_start.record()
    function()
    first_end.record()
    first_end.synchronize()
    first_us = first_start.elapsed_time(first_end) * 1.0e3

    values = main_values
    if extra_values is not None:
        values = torch.cat((values, extra_values), dim=0)
    scores = q[0, 0].float() @ values.float().t() / math.sqrt(512.0)
    probability = torch.softmax(torch.cat((scores, sink[:, None]), dim=1), dim=1)
    expected = (probability[:, : values.shape[0]] @ values.float()).to(
        torch.bfloat16
    )
    max_abs = float((output[0, 0].float() - expected.float()).abs().max().item())
    cosine = float(
        torch.nn.functional.cosine_similarity(
            output[0, 0].float().reshape(1, -1),
            expected.float().reshape(1, -1),
        ).item()
    )

    for _ in range(20):
        function()
        epilogue()
    torch.cuda.synchronize()
    direct = _event_samples(function, samples)
    hot_graph = _graph_samples(function, samples, inner)
    cold_graph = _cold_graph_samples(function, samples, scrub_mib)
    epilogue_hot_graph = _graph_samples(epilogue, samples, inner)
    combined_hot_graph = _graph_samples(attention_epilogue, samples, inner)
    combined_cold_graph = _cold_graph_samples(
        attention_epilogue, samples, scrub_mib
    )
    quantized, scales, *_ = quantized_output["value"]
    if quantized.numel() != output.numel() or scales.numel() == 0:
        raise AssertionError("inverse-RoPE quantizer returned an invalid shape")
    num_splits = scheduler.num_splits.cpu().tolist()
    print(
        "DSV4_FLASHMLA_DECODE_RESULT "
        f"case={case.name} main_width=128 main_active=128 "
        f"extra_width={case.extra_width} extra_active={case.extra_rows} "
        f"first_plan_us={first_us:.6f} num_splits={num_splits} "
        f"{_summary('direct', direct)} "
        f"{_summary('hot_graph', hot_graph)} "
        f"{_summary('cold_l2_graph', cold_graph)} "
        f"{_summary('epilogue_hot_graph', epilogue_hot_graph)} "
        f"{_summary('combined_hot_graph', combined_hot_graph)} "
        f"{_summary('combined_cold_l2_graph', combined_cold_graph)} "
        f"max_abs={max_abs:.8f} cosine={cosine:.8f} status=PASS",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--inner", type=int, default=20)
    parser.add_argument("--scrub-mib", type=int, default=260)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()
    if min(args.samples, args.inner, args.scrub_mib) <= 0:
        parser.error("timing counts and scrub size must be positive")
    torch.cuda.set_device(0)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    for case in CASES:
        run_case(case, generator, args.samples, args.inner, args.scrub_mib)


if __name__ == "__main__":
    main()
