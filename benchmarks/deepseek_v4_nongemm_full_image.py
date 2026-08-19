#!/usr/bin/env python3
"""Isolated non-GEMM measurements from the complete DeepSeek-V4 image.

Build ``benchmarks/deepseek_v4_resident_compact.ops`` once, then invoke one
case per fresh Python process.  The device envelope comes from the resident
kernel's built-in per-SM start/end timestamps; no profile handler or focused
operator selector is required.
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable
from dataclasses import dataclass

import torch

from dae.deepseek_v4 import (
    apply_partial_rope_128_64,
    apply_partial_rope_512_64,
    gated_pool_reference,
    hadamard_reference,
    hc_head_reference,
    hc_post_reference,
    hc_pre_reference,
    pack_gated_pool_history,
    route_top6_reference,
    sparse_attention_512_reference,
)
from dae.deepseek_v4_quant import quantize_fp8_block128, quantize_nvfp4
from dae.launcher import Launcher
from dae.schedule import (
    SchedArgmaxSmemPartial,
    SchedArgmaxSmemReduce,
    SchedDsv4ContiguousAttention512Block4,
    SchedDsv4Fp8Quant128,
    SchedDsv4Fp8QuantUmmaB,
    SchedDsv4GatedPoolPacked8Shard128,
    SchedDsv4GatedPoolRmsRope,
    SchedDsv4HcHeadRms,
    SchedDsv4HcPost,
    SchedDsv4HcPreRms,
    SchedDsv4Nvfp4QuantUmmaB,
    SchedDsv4PreloadRopeTables,
    SchedDsv4RmsFp8QuantUmmaB,
    SchedDsv4RmsRope512_64,
    SchedDsv4Rope128_64,
    SchedDsv4Rope512_64,
    SchedDsv4RouteTop6,
)


Validator = Callable[[], float]


@dataclass
class Case:
    launcher: Launcher
    validate: Validator
    opcode: str
    shape: str
    includes_preload: bool = False


def _rope_table(device: torch.device, phase: float = 0.0) -> torch.Tensor:
    angles = torch.linspace(
        -1.25 + phase, 1.25 + phase, 32, dtype=torch.float32, device=device
    )
    return torch.stack((angles.cos(), angles.sin()), dim=1).contiguous()


def _rms(
    source: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    values = source.float()
    return values * torch.rsqrt(values.square().mean() + epsilon) * weight.float()


def _max_abs(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected.float()).abs().max().item())


def _launcher(
    device: torch.device,
    num_sms: int,
    *placed_schedules,
) -> Launcher:
    launcher = Launcher(num_sms, device=device)
    launcher.s(*placed_schedules)
    return launcher


def _fixed_launcher(
    device: torch.device,
    tables: tuple[torch.Tensor, ...],
    target,
    target_sms: int,
) -> Launcher:
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    return _launcher(
        device,
        num_sms,
        SchedDsv4PreloadRopeTables(tables).place(num_sms),
        target.place(target_sms),
    )


def _native_fp8_data_reference(source: torch.Tensor) -> torch.Tensor:
    quantized, _ = quantize_fp8_block128(source)
    tiles = source.numel() // 128
    logical = quantized.view(torch.uint8).reshape(tiles, 8, 16)
    expected = torch.empty(
        (tiles, 8, 8, 16), dtype=torch.uint8, device=source.device
    )
    for row in range(8):
        for source_chunk in range(8):
            expected[:, row, source_chunk ^ row].copy_(
                logical[:, source_chunk]
            )
    return expected.reshape(tiles, 1024)


def _native_nvfp4_data_reference(
    source: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    tiles = source.numel() // 256
    packed, _, _ = quantize_nvfp4(
        source.reshape(tiles, 256), global_scale.reshape(())
    )
    logical = packed.reshape(tiles, 8, 16)
    expected = torch.empty(
        (tiles, 8, 8, 16), dtype=torch.uint8, device=source.device
    )
    for row in range(8):
        for source_chunk in range(8):
            expected[:, row, source_chunk ^ row].copy_(
                logical[:, source_chunk]
            )
    return expected.reshape(tiles, 1024)


def build_preload(
    device: torch.device, generator: torch.Generator
) -> Case:
    del generator
    tables = tuple(_rope_table(device, index * 0.03125) for index in range(4))
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    launcher = _launcher(
        device, sms, SchedDsv4PreloadRopeTables(tables).place(sms)
    )
    return Case(
        launcher,
        lambda: 0.0,
        "OP_DSV4_PRELOAD_ROPE_TABLES",
        "tables4_sms152",
    )


def build_hc_pre_rms(
    device: torch.device,
    generator: torch.Generator,
    *,
    zero_output: bool,
) -> Case:
    residual = torch.randn(
        (4, 4096), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.125
    mixes = torch.randn(
        (24,), generator=generator, dtype=torch.float32, device=device
    ) * 0.1
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
    output = torch.empty((4096,), dtype=torch.bfloat16, device=device)
    post = torch.empty((4,), dtype=torch.float32, device=device)
    comb = torch.empty((4, 4), dtype=torch.float32, device=device)
    zero = (
        torch.full((4096,), 7.0, dtype=torch.float32, device=device)
        if zero_output
        else None
    )
    schedule = SchedDsv4HcPreRms(
        residual,
        mixes,
        scale,
        base,
        weight,
        output,
        post,
        comb,
        zero_fp32_output=zero,
    )
    launcher = _launcher(device, 1, schedule.place(1))

    def validate() -> float:
        hidden, expected_post, expected_comb = hc_pre_reference(
            residual, mixes, scale, base
        )
        expected = _rms(hidden, weight, 1.0e-6).to(torch.bfloat16)
        torch.testing.assert_close(output, expected, rtol=2.0e-2, atol=1.0e-2)
        torch.testing.assert_close(post, expected_post, rtol=2.0e-5, atol=2.0e-5)
        torch.testing.assert_close(comb, expected_comb, rtol=2.0e-5, atol=2.0e-5)
        if zero is not None:
            torch.testing.assert_close(zero, torch.zeros_like(zero), rtol=0, atol=0)
        return _max_abs(output, expected)

    return Case(
        launcher,
        validate,
        "OP_DSV4_HC_PRE_RMS",
        f"residual4x4096_zero_fp32={int(zero_output)}",
    )


def build_fp8_quant128(
    device: torch.device, generator: torch.Generator
) -> Case:
    source = torch.randn(
        (4096,), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.25
    source[::257] *= 8
    output = torch.empty_like(source, dtype=torch.float8_e4m3fn)
    scale = torch.empty((32,), dtype=torch.float8_e8m0fnu, device=device)
    launcher = _launcher(
        device,
        32,
        SchedDsv4Fp8Quant128(source, output, scale).place(32),
    )

    def validate() -> float:
        expected, expected_scale = quantize_fp8_block128(source)
        torch.testing.assert_close(
            output.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(
            scale.view(torch.uint8), expected_scale.view(torch.uint8), rtol=0, atol=0
        )
        return 0.0

    return Case(
        launcher,
        validate,
        "OP_DSV4_FP8_QUANT_128",
        "k4096_sms32",
    )


def build_rms_fp8_native(
    device: torch.device, generator: torch.Generator
) -> Case:
    k = 1024
    source = torch.randn(
        (k,), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.25
    weight = (
        torch.randn((k,), generator=generator, dtype=torch.bfloat16, device=device)
        * 0.05
        + 1.0
    )
    output = torch.empty((k // 128, 2048), dtype=torch.uint8, device=device)
    launcher = _launcher(
        device,
        4,
        SchedDsv4RmsFp8QuantUmmaB(
            source, weight, output, 1.0e-6, scale_pack=2
        ).place(4),
    )

    def validate() -> float:
        normalized = _rms(source, weight, 1.0e-6)
        expected = _native_fp8_data_reference(normalized)
        torch.testing.assert_close(output[:, :1024], expected, rtol=0, atol=0)
        return 0.0

    return Case(
        launcher,
        validate,
        "OP_DSV4_RMS_FP8_QUANT_UMMA_B_SM100",
        "k1024_pack2_sms4",
    )


def build_rms_rope(
    device: torch.device,
    generator: torch.Generator,
    *,
    weighted: bool,
) -> Case:
    rows = 1 if weighted else 64
    table = _rope_table(device)
    tables = (table, _rope_table(device, 0.03125), _rope_table(device, 0.0625),
              _rope_table(device, 0.09375))
    source = torch.randn(
        (rows, 512), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.25
    weight = (
        (
            torch.randn(
                (512,), generator=generator, dtype=torch.bfloat16, device=device
            )
            * 0.05
            + 1.0
        )
        if weighted
        else None
    )
    output = torch.empty_like(source)
    target = SchedDsv4RmsRope512_64(
        source,
        table,
        output,
        epsilon=1.0e-6,
        weight=weight,
        fixed_table_id=0,
    )
    launcher = _fixed_launcher(device, tables, target, rows)

    def validate() -> float:
        values = source.float()
        values = values * torch.rsqrt(
            values.square().mean(dim=1, keepdim=True) + 1.0e-6
        )
        if weight is not None:
            values = values * weight.float()
        expected = apply_partial_rope_512_64(values, table).to(torch.bfloat16)
        torch.testing.assert_close(output, expected, rtol=2.0e-2, atol=1.0e-2)
        return _max_abs(output, expected)

    return Case(
        launcher,
        validate,
        "OP_DSV4_RMS_ROPE_512_64",
        f"rows{rows}_d512_weighted={int(weighted)}_sms{rows}",
        includes_preload=True,
    )


def build_attention(
    device: torch.device,
    generator: torch.Generator,
    *,
    rows: int,
) -> Case:
    q = torch.randn(
        (64, 512), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.125
    kv = torch.randn(
        (rows, 512), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.125
    sink = torch.linspace(-0.5, 0.5, 64, dtype=torch.float32, device=device)
    output = torch.empty_like(q)
    launcher = _launcher(
        device,
        64,
        SchedDsv4ContiguousAttention512Block4(
            q, kv, rows, sink, output
        ).place(64),
    )

    def validate() -> float:
        indices = torch.arange(rows, dtype=torch.int32, device=device)
        expected = sparse_attention_512_reference(q, kv, indices, sink)
        torch.testing.assert_close(output, expected, rtol=3.0e-2, atol=1.0e-2)
        return _max_abs(output, expected)

    return Case(
        launcher,
        validate,
        "OP_DSV4_CONTIGUOUS_ATTENTION_512_BLOCK4",
        f"h64_d512_rows{rows}_sms64",
    )


def build_rope(
    device: torch.device,
    generator: torch.Generator,
    *,
    width: int,
    inverse: bool,
) -> Case:
    table = _rope_table(device)
    tables = (table, _rope_table(device, 0.03125), _rope_table(device, 0.0625),
              _rope_table(device, 0.09375))
    source = torch.randn(
        (64, width), generator=generator, dtype=torch.bfloat16, device=device
    )
    output = torch.empty_like(source)
    schedule_cls = SchedDsv4Rope512_64 if width == 512 else SchedDsv4Rope128_64
    reference = apply_partial_rope_512_64 if width == 512 else apply_partial_rope_128_64
    target = schedule_cls(
        source, table, output, inverse=inverse, fixed_table_id=0
    )
    launcher = _fixed_launcher(device, tables, target, 64)

    def validate() -> float:
        expected = reference(source, table, inverse=inverse)
        torch.testing.assert_close(output, expected, rtol=1.0e-2, atol=1.0e-2)
        return _max_abs(output, expected)

    return Case(
        launcher,
        validate,
        "OP_DSV4_ROPE_64",
        f"rows64_d{width}_inverse={int(inverse)}_sms64",
        includes_preload=True,
    )


def build_hc_post(
    device: torch.device,
    generator: torch.Generator,
    *,
    fp32_branch: bool,
) -> Case:
    branch_dtype = torch.float32 if fp32_branch else torch.bfloat16
    branch = torch.randn(
        (4096,), generator=generator, dtype=branch_dtype, device=device
    ) * 0.125
    residual = torch.randn(
        (4, 4096), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.125
    post = torch.rand((4,), generator=generator, dtype=torch.float32, device=device)
    comb = torch.rand(
        (4, 4), generator=generator, dtype=torch.float32, device=device
    )
    output = torch.empty_like(residual)
    launcher = _launcher(
        device,
        32,
        SchedDsv4HcPost(branch, residual, post, comb, output).place(32),
    )

    def validate() -> float:
        expected = hc_post_reference(branch, residual, post, comb).to(torch.bfloat16)
        torch.testing.assert_close(output, expected, rtol=2.0e-2, atol=1.0e-2)
        return _max_abs(output, expected)

    return Case(
        launcher,
        validate,
        "OP_DSV4_HC_POST",
        f"branch_{'fp32' if fp32_branch else 'bf16'}_k4096_sms32",
    )


def build_nvfp4_native(
    device: torch.device,
    generator: torch.Generator,
    *,
    k: int,
) -> Case:
    source = torch.randn(
        (k,), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.25
    source[::257] *= 8
    global_scale = (
        source.float().abs().amax() / (6.0 * 448.0)
    ).reshape(1).contiguous()
    tiles = k // 256
    output = torch.empty((tiles, 3072), dtype=torch.uint8, device=device)
    launcher = _launcher(
        device,
        tiles,
        SchedDsv4Nvfp4QuantUmmaB(source, global_scale, output).place(tiles),
    )

    def validate() -> float:
        expected = _native_nvfp4_data_reference(source, global_scale)
        torch.testing.assert_close(output[:, :1024], expected, rtol=0, atol=0)
        return 0.0

    return Case(
        launcher,
        validate,
        "OP_DSV4_NVFP4_QUANT_UMMA_B_SM100",
        f"k{k}_sms{tiles}",
    )


def build_pool_packed(
    device: torch.device, generator: torch.Generator
) -> Case:
    history_rows = 127
    values = torch.randn(
        (history_rows, 512), generator=generator, dtype=torch.float32, device=device
    ) * 0.125
    scores = torch.randn(
        (history_rows, 512), generator=generator, dtype=torch.float32, device=device
    )
    tail_values = torch.randn(
        (512,), generator=generator, dtype=torch.float32, device=device
    ) * 0.125
    tail_scores = torch.randn(
        (512,), generator=generator, dtype=torch.float32, device=device
    )
    tail_bias = torch.randn(
        (512,), generator=generator, dtype=torch.float32, device=device
    ) * 0.1
    packed = pack_gated_pool_history(values, scores)
    output = torch.empty((512,), dtype=torch.bfloat16, device=device)
    launcher = _launcher(
        device,
        4,
        SchedDsv4GatedPoolPacked8Shard128(
            packed,
            history_rows,
            output,
            tail_values=tail_values,
            tail_scores=tail_scores,
            tail_bias=tail_bias,
        ).place(4),
    )

    def validate() -> float:
        all_values = torch.cat((values, tail_values[None]), dim=0)
        all_scores = torch.cat((scores, (tail_scores + tail_bias)[None]), dim=0)
        expected = gated_pool_reference(all_values, all_scores).to(torch.bfloat16)
        torch.testing.assert_close(output, expected, rtol=2.0e-2, atol=1.0e-2)
        return _max_abs(output, expected)

    return Case(
        launcher,
        validate,
        "OP_DSV4_GATED_POOL_PACKED8_SHARD128",
        "history127_tail1_d512_sms4",
    )


def build_pool_rms_rope(
    device: torch.device,
    generator: torch.Generator,
    *,
    width: int,
) -> Case:
    table = _rope_table(device)
    tables = (table, _rope_table(device, 0.03125), _rope_table(device, 0.0625),
              _rope_table(device, 0.09375))
    history_rows = 7
    values = torch.randn(
        (history_rows, width),
        generator=generator,
        dtype=torch.float32,
        device=device,
    ) * 0.125
    scores = torch.randn(
        (history_rows, width),
        generator=generator,
        dtype=torch.float32,
        device=device,
    )
    tail_values = torch.randn(
        (width,), generator=generator, dtype=torch.float32, device=device
    ) * 0.125
    tail_scores = torch.randn(
        (width,), generator=generator, dtype=torch.float32, device=device
    )
    tail_bias = torch.randn(
        (width,), generator=generator, dtype=torch.float32, device=device
    ) * 0.1
    weight = (
        torch.randn(
            (width,), generator=generator, dtype=torch.bfloat16, device=device
        )
        * 0.05
        + 1.0
    )
    output = torch.empty((width,), dtype=torch.bfloat16, device=device)
    hadamard = width == 128
    target = SchedDsv4GatedPoolRmsRope(
        values,
        scores,
        weight,
        table,
        output,
        epsilon=1.0e-6,
        tail_values=tail_values,
        tail_scores=tail_scores,
        tail_bias=tail_bias,
        hadamard=hadamard,
        fixed_table_id=0,
    )
    launcher = _fixed_launcher(device, tables, target, 1)

    def validate() -> float:
        all_values = torch.cat((values, tail_values[None]), dim=0)
        all_scores = torch.cat((scores, (tail_scores + tail_bias)[None]), dim=0)
        pooled = gated_pool_reference(all_values, all_scores)
        normalized = _rms(pooled, weight, 1.0e-6)
        reference = (
            apply_partial_rope_128_64
            if width == 128
            else apply_partial_rope_512_64
        )
        expected = reference(normalized, table).to(torch.bfloat16)
        if hadamard:
            expected = hadamard_reference(expected)
        torch.testing.assert_close(output, expected, rtol=3.0e-2, atol=2.0e-2)
        return _max_abs(output, expected)

    return Case(
        launcher,
        validate,
        "OP_DSV4_GATED_POOL_RMS_ROPE",
        f"history7_tail1_d{width}_hadamard={int(hadamard)}_sms1",
        includes_preload=True,
    )


def build_argmax_partial(
    device: torch.device, generator: torch.Generator
) -> Case:
    vocab = 129_280
    sms = 152
    logits = torch.randn(
        (vocab,), generator=generator, dtype=torch.bfloat16, device=device
    )
    winner = 127_777
    logits[winner] = 64.0
    partials = torch.empty((sms, 16), dtype=torch.uint8, device=device)
    launcher = _launcher(
        device, sms, SchedArgmaxSmemPartial(logits, partials).place(sms)
    )

    def validate() -> float:
        output = torch.empty((1,), dtype=torch.int64, device=device)
        reduction = _launcher(
            device, 1, SchedArgmaxSmemReduce(partials, output).place(1)
        )
        reduction.launch()
        expected = torch.argmax(logits).to(torch.int64)
        torch.testing.assert_close(output.reshape(()), expected, rtol=0, atol=0)
        return 0.0

    return Case(
        launcher,
        validate,
        "OP_ARGMAX_SMEM_PARTIAL_BF16",
        "vocab129280_sms152",
    )


def build_route(
    device: torch.device,
    generator: torch.Generator,
    *,
    hash_routing: bool,
) -> Case:
    logits = torch.randn(
        (256,), generator=generator, dtype=torch.bfloat16, device=device
    )
    bias = torch.randn(
        (256,), generator=generator, dtype=torch.float32, device=device
    ) * 0.1
    hash_indices = torch.zeros((8,), dtype=torch.int32, device=device)
    hash_indices[:6] = torch.tensor(
        (9, 71, 5, 255, 130, 44), dtype=torch.int32, device=device
    )
    output_indices = torch.empty((8,), dtype=torch.int32, device=device)
    output_weights = torch.empty((8,), dtype=torch.float32, device=device)
    launcher = _launcher(
        device,
        1,
        SchedDsv4RouteTop6(
            logits,
            bias,
            hash_indices,
            output_indices,
            output_weights,
            hash_routing=hash_routing,
        ).place(1),
    )

    def validate() -> float:
        expected_weights, expected_indices = route_top6_reference(
            logits,
            bias,
            hash_indices=hash_indices[:6] if hash_routing else None,
        )
        torch.testing.assert_close(
            output_indices[:6], expected_indices, rtol=0, atol=0
        )
        torch.testing.assert_close(
            output_weights[:6], expected_weights, rtol=2.0e-5, atol=2.0e-5
        )
        return _max_abs(output_weights[:6], expected_weights)

    return Case(
        launcher,
        validate,
        "OP_DSV4_ROUTE_TOP6",
        f"experts256_top6_hash={int(hash_routing)}_sms1",
    )


def build_hc_head_rms(
    device: torch.device, generator: torch.Generator
) -> Case:
    residual = torch.randn(
        (4, 4096), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.125
    mixes = torch.randn(
        (4,), generator=generator, dtype=torch.float32, device=device
    ) * 0.1
    scale = torch.tensor((0.625,), dtype=torch.float32, device=device)
    base = torch.randn(
        (4,), generator=generator, dtype=torch.float32, device=device
    ) * 0.05
    weight = (
        torch.randn(
            (4096,), generator=generator, dtype=torch.bfloat16, device=device
        )
        * 0.05
        + 1.0
    )
    output = torch.empty((4096,), dtype=torch.bfloat16, device=device)
    launcher = _launcher(
        device,
        1,
        SchedDsv4HcHeadRms(
            residual, mixes, scale, base, weight, output
        ).place(1),
    )

    def validate() -> float:
        head = hc_head_reference(residual, mixes, scale, base)
        expected = _rms(head, weight, 1.0e-6).to(torch.bfloat16)
        torch.testing.assert_close(output, expected, rtol=2.0e-2, atol=1.0e-2)
        return _max_abs(output, expected)

    return Case(
        launcher,
        validate,
        "OP_DSV4_HC_HEAD_RMS",
        "residual4x4096_sms1",
    )


def build_argmax_reduce(
    device: torch.device, generator: torch.Generator
) -> Case:
    records = 152
    values = torch.randn(
        (records,), generator=generator, dtype=torch.bfloat16, device=device
    )
    winner = 137
    values[winner] = 64.0
    indices = torch.arange(records, dtype=torch.int64, device=device) * 853
    partials = torch.zeros((records, 16), dtype=torch.uint8, device=device)
    words = partials.view(torch.int64)
    words[:, 0].copy_(values.view(torch.uint16).to(torch.int64))
    words[:, 1].copy_(indices)
    output = torch.empty((1,), dtype=torch.int64, device=device)
    launcher = _launcher(
        device, 1, SchedArgmaxSmemReduce(partials, output).place(1)
    )

    def validate() -> float:
        torch.testing.assert_close(output[0], indices[winner], rtol=0, atol=0)
        return 0.0

    return Case(
        launcher,
        validate,
        "OP_ARGMAX_SMEM_REDUCE_BF16",
        "records152_sms1",
    )


def build_fp8_native(
    device: torch.device,
    generator: torch.Generator,
    *,
    k: int,
) -> Case:
    source = torch.randn(
        (k,), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.25
    source[::257] *= 8
    tiles = k // 128
    sms = tiles // 2
    output = torch.empty((tiles, 2048), dtype=torch.uint8, device=device)
    launcher = _launcher(
        device,
        sms,
        SchedDsv4Fp8QuantUmmaB(source, output, scale_pack=2).place(sms),
    )

    def validate() -> float:
        expected = _native_fp8_data_reference(source)
        torch.testing.assert_close(output[:, :1024], expected, rtol=0, atol=0)
        return 0.0

    return Case(
        launcher,
        validate,
        "OP_DSV4_FP8_QUANT_UMMA_B_SM100__SCALE_PACK_2",
        f"k{k}_pack2_sms{sms}",
    )


CASES: dict[str, Callable[[torch.device, torch.Generator], Case]] = {
    "preload_rope4": build_preload,
    "hc_pre_rms": lambda d, g: build_hc_pre_rms(d, g, zero_output=False),
    "hc_pre_rms_zero": lambda d, g: build_hc_pre_rms(d, g, zero_output=True),
    "fp8_quant128_k4096": build_fp8_quant128,
    "rms_fp8_native_k1024": build_rms_fp8_native,
    "rms_rope_q64": lambda d, g: build_rms_rope(d, g, weighted=False),
    "rms_rope_kv1_weighted": lambda d, g: build_rms_rope(d, g, weighted=True),
    "attention_rows128": lambda d, g: build_attention(d, g, rows=128),
    "attention_rows129": lambda d, g: build_attention(d, g, rows=129),
    "attention_rows160": lambda d, g: build_attention(d, g, rows=160),
    "rope_d512_forward": lambda d, g: build_rope(d, g, width=512, inverse=False),
    "rope_d512_inverse": lambda d, g: build_rope(d, g, width=512, inverse=True),
    "rope_d128_forward": lambda d, g: build_rope(d, g, width=128, inverse=False),
    "hc_post_bf16": lambda d, g: build_hc_post(d, g, fp32_branch=False),
    "hc_post_fp32": lambda d, g: build_hc_post(d, g, fp32_branch=True),
    "nvfp4_native_k2048": lambda d, g: build_nvfp4_native(d, g, k=2048),
    "nvfp4_native_k4096": lambda d, g: build_nvfp4_native(d, g, k=4096),
    "pool_packed_ratio128": build_pool_packed,
    "pool_rms_rope_d512": lambda d, g: build_pool_rms_rope(d, g, width=512),
    "pool_rms_rope_hadamard_d128": lambda d, g: build_pool_rms_rope(d, g, width=128),
    "argmax_partial": build_argmax_partial,
    "route_score": lambda d, g: build_route(d, g, hash_routing=False),
    "route_hash": lambda d, g: build_route(d, g, hash_routing=True),
    "hc_head_rms": build_hc_head_rms,
    "argmax_reduce": build_argmax_reduce,
    "fp8_native_k2048": lambda d, g: build_fp8_native(d, g, k=2048),
    "fp8_native_k4096": lambda d, g: build_fp8_native(d, g, k=4096),
}


def _device_envelope_us(launcher: Launcher) -> float:
    profile = launcher.profile[:, :2].cpu().numpy()
    return float(profile[:, 1].max() - profile[:, 0].min()) / 1.0e3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be non-negative and iterations positive")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    case = CASES[args.case](device, generator)

    case.launcher.launch()
    cold_us = _device_envelope_us(case.launcher)
    for _ in range(args.warmup):
        case.launcher.launch()
    hot_us = []
    for _ in range(args.iterations):
        case.launcher.launch()
        hot_us.append(_device_envelope_us(case.launcher))
    max_abs = case.validate()

    supported = set(getattr(__import__("dae.runtime", fromlist=["runtime"]),
                            "supported_compute_ops", ()))
    if supported and case.opcode not in supported:
        raise AssertionError(f"{case.opcode} is not selected in the loaded image")
    props = torch.cuda.get_device_properties(device)
    print(
        "DSV4_NONGEMM_FULL_IMAGE_RESULT "
        f"case={args.case} opcode={case.opcode} shape={case.shape} "
        f"cold_us={cold_us:.6f} min_us={min(hot_us):.6f} "
        f"median_us={statistics.median(hot_us):.6f} "
        f"max_us={max(hot_us):.6f} max_abs={max_abs:.8f} "
        f"includes_preload={int(case.includes_preload)} status=PASS "
        f"device={props.name!r} cc={props.major}.{props.minor}",
        flush=True,
    )


if __name__ == "__main__":
    main()
