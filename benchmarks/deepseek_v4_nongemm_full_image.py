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
    SchedDsv4Bf16Gemv,
    SchedDsv4ContiguousAttention512Block4,
    SchedDsv4Fp8Quant128,
    SchedDsv4Fp8QuantUmmaB,
    SchedDsv4Fp32Bf16Gemv,
    SchedDsv4GatedPoolPacked8Shard128,
    SchedDsv4GatedPoolRmsRope,
    SchedDsv4HcHeadRms,
    SchedDsv4HcPost,
    SchedDsv4HcPreRms,
    SchedDsv4Mxfp8QuantFfnInput,
    SchedDsv4Nvfp4QuantUmmaB,
    SchedDsv4PreloadRopeTables,
    SchedDsv4RmsFp8QuantUmmaB,
    SchedDsv4RmsRope512_64,
    SchedDsv4Rope128_64,
    SchedDsv4Rope512_64,
    SchedDsv4RouteTop6,
    SchedDsv4RouterBf16Gemv,
)
from dae.sequential import SequentialProgram, SequentialStage


Validator = Callable[[], float]


@dataclass
class Case:
    launcher: Launcher
    validate: Validator
    opcode: str | tuple[str, ...]
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


def _mxfp8_ffn_input_reference(
    source: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return packed K512 data and the 32 active SFB bytes per K128."""
    records = source.numel() // 512
    groups = source.float().reshape(records, 16, 32)
    requested = (groups.abs().amax(dim=-1) / 448.0).clamp_min(2.0**-127)
    exponents = torch.ceil(torch.log2(requested)).clamp(-127, 127)
    scales = torch.exp2(exponents)
    quantized = (
        (groups / scales.unsqueeze(-1))
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .reshape(records, 4, 8, 16)
    )
    packed = torch.empty(
        (records, 4, 8, 8, 16), dtype=torch.uint8, device=source.device
    )
    for row in range(8):
        for source_chunk in range(8):
            packed[:, :, row, source_chunk ^ row].copy_(
                quantized[:, :, source_chunk]
            )

    scale_bytes = (exponents.to(torch.int16) + 127).to(torch.uint8)
    active_scales = scale_bytes.reshape(records, 4, 4)
    active_scales = (
        active_scales.unsqueeze(2).expand(records, 4, 8, 4).contiguous()
    )
    return packed.reshape(records, 4096), active_scales.reshape(records, 4, 32)


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
    output_fp8: bool = False,
) -> Case:
    residual = torch.randn(
        (4, 4096), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.125
    mixes = torch.randn(
        (24,), generator=generator, dtype=torch.float32, device=device
    ) * 0.1
    scale = torch.tensor((0.5, 0.75, 1.25), dtype=torch.float32, device=device)
    residual_square_sum = residual.float().square().sum().reshape(1)
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
    packed_metadata = torch.empty(
        (56,), dtype=torch.float32, device=device
    )
    packed_metadata[:1].copy_(residual_square_sum)
    packed_metadata[1:25].copy_(mixes)
    packed_metadata[28:31].copy_(scale)
    packed_metadata[31:55].copy_(base)
    if output_fp8:
        packed_output = torch.empty((4176,), dtype=torch.uint8, device=device)
        output = None
        fp8_output = packed_output[:4096].view(torch.float8_e4m3fn)
        output_metadata = packed_output[4096:].view(torch.float32)
    else:
        packed_output = torch.empty(
            (4136,), dtype=torch.bfloat16, device=device
        )
        output = packed_output[:4096]
        fp8_output = None
        output_metadata = packed_output[4096:].view(torch.float32)
    post = output_metadata[:4]
    comb = output_metadata[4:].view(4, 4)
    zero = (
        torch.full((4096,), 7.0, dtype=torch.float32, device=device)
        if zero_output
        else None
    )
    fp8_scale = (
        torch.empty((32,), dtype=torch.float8_e8m0fnu, device=device)
        if output_fp8
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
        residual_square_sum=residual_square_sum,
        packed_metadata=packed_metadata,
        packed_output=packed_output,
        zero_fp32_output=zero,
        fp8_output=fp8_output,
        fp8_scale=fp8_scale,
    )
    launcher = _launcher(device, 1, schedule.place(1))

    def validate() -> float:
        hidden, expected_post, expected_comb = hc_pre_reference(
            residual, mixes, scale, base, sinkhorn_iters=20
        )
        expected = _rms(hidden, weight, 1.0e-6).to(torch.bfloat16)
        if output is not None:
            torch.testing.assert_close(
                output, expected, rtol=2.0e-2, atol=1.0e-2
            )
        torch.testing.assert_close(post, expected_post, rtol=2.0e-5, atol=2.0e-5)
        torch.testing.assert_close(comb, expected_comb, rtol=2.0e-5, atol=2.0e-5)
        if zero is not None:
            torch.testing.assert_close(zero, torch.zeros_like(zero), rtol=0, atol=0)
        if fp8_output is not None:
            expected_fp8, expected_fp8_scale = quantize_fp8_block128(expected)
            torch.testing.assert_close(fp8_output, expected_fp8, rtol=0, atol=0)
            torch.testing.assert_close(
                fp8_scale, expected_fp8_scale, rtol=0, atol=0
            )
            return 0.0
        return _max_abs(output, expected)

    return Case(
        launcher,
        validate,
        "OP_DSV4_HC_PRE_RMS",
        f"residual4x4096_zero_fp32={int(zero_output)}_"
        f"output={'fp8' if output_fp8 else 'bf16'}_sinkhorn=20",
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
    sms: int = 32,
) -> Case:
    branch_dtype = torch.float32 if fp32_branch else torch.bfloat16
    packed_input_record = None
    packed_output_record = None
    if fp32_branch:
        branch = torch.empty((4096,), dtype=torch.float32, device=device)
        residual = torch.empty(
            (4, 4096), dtype=torch.bfloat16, device=device
        )
        output = torch.empty_like(residual)
    else:
        packed_input_record = torch.empty(
            (6, 4096), dtype=torch.bfloat16, device=device
        )
        packed_output_record = torch.empty_like(packed_input_record)
        branch = packed_input_record[0]
        residual = packed_input_record[1:5]
        output = packed_output_record[1:5]
    branch.copy_(
        torch.randn(
            (4096,), generator=generator, dtype=branch_dtype, device=device
        ) * 0.125
    )
    residual.copy_(
        torch.randn(
            (4, 4096), generator=generator, dtype=torch.bfloat16, device=device
        ) * 0.125
    )
    coefficients = torch.rand(
        (20,), generator=generator, dtype=torch.float32, device=device
    )
    if packed_input_record is not None:
        coefficient_bits = coefficients.view(torch.bfloat16)
        for start in range(0, 4096, 128):
            packed_input_record[5, start : start + 40].copy_(coefficient_bits)
    post = coefficients[:4]
    comb = coefficients[4:].view(4, 4)
    launcher = Launcher(sms, device=device)
    launcher.s(
        SchedDsv4HcPost(
            branch,
            residual,
            post,
            comb,
            output,
            launcher=launcher,
            packed_coefficients=coefficients,
            packed_input_record=packed_input_record,
            packed_output_record=packed_output_record,
        ).place(sms)
    )

    def validate() -> float:
        expected = hc_post_reference(branch, residual, post, comb).to(torch.bfloat16)
        torch.testing.assert_close(output, expected, rtol=2.0e-2, atol=1.0e-2)
        return _max_abs(output, expected)

    return Case(
        launcher,
        validate,
        "OP_DSV4_HC_POST",
        f"branch_{'fp32' if fp32_branch else 'bf16'}_k4096_sms{sms}",
    )


def build_hc_post_pre_boundary(
    device: torch.device,
    generator: torch.Generator,
) -> Case:
    """Measure post -> metadata projection -> pre/RMS in one resident launch."""
    packed_input_record = torch.empty(
        (6, 4096), dtype=torch.bfloat16, device=device
    )
    branch = packed_input_record[0]
    residual = packed_input_record[1:5]
    branch.copy_(
        torch.randn(
            (4096,), generator=generator, dtype=torch.bfloat16, device=device
        ) * 0.125
    )
    residual.copy_(
        torch.randn(
            (4, 4096), generator=generator, dtype=torch.bfloat16, device=device
        ) * 0.125
    )
    coefficients = torch.rand(
        (20,), generator=generator, dtype=torch.float32, device=device
    )
    coefficient_bits = coefficients.view(torch.bfloat16)
    for start in range(0, 4096, 128):
        packed_input_record[5, start : start + 40].copy_(coefficient_bits)
    post = coefficients[:4]
    comb = coefficients[4:].view(4, 4)
    post_output = torch.empty(
        (4, 4096), dtype=torch.bfloat16, device=device
    )

    projection = torch.randn(
        (24, 4 * 4096),
        generator=generator,
        dtype=torch.float32,
        device=device,
    ) * 0.01
    projection_packed = (
        projection.view(8, 3, 4, 16, 1, 256)
        .permute(0, 3, 4, 1, 2, 5)
        .contiguous()
    )
    metadata = torch.empty((16 * 32 + 28,), dtype=torch.float32, device=device)
    residual_square_sum = metadata[:1]
    mixes = torch.empty((24,), dtype=torch.float32, device=device)
    scale = torch.tensor(
        (0.5, 0.75, 1.25), dtype=torch.float32, device=device
    )
    base = torch.randn(
        (24,), generator=generator, dtype=torch.float32, device=device
    ) * 0.05
    metadata[16 * 32 : 16 * 32 + 3].copy_(scale)
    metadata[16 * 32 + 3 : 16 * 32 + 27].copy_(base)
    norm_weight = (
        torch.randn(
            (4096,), generator=generator, dtype=torch.bfloat16, device=device
        ) * 0.05
        + 1.0
    )
    packed_output = torch.empty(
        (4136,), dtype=torch.bfloat16, device=device
    )
    normalized = packed_output[:4096]
    output_metadata = packed_output[4096:].view(torch.float32)
    next_post = output_metadata[:4]
    next_comb = output_metadata[4:].view(4, 4)

    # Keep the reducer on an otherwise-idle SM so its allocator/LDU can wait
    # on the producer-owned data dependencies without sitting behind a local
    # projection task in the same compute queue.
    launcher = Launcher(8 * 16 + 1, device=device)
    program = SequentialProgram(
        launcher,
        (
            SequentialStage(
                "hc_post_project",
                SchedDsv4Fp32Bf16Gemv(
                    projection_packed,
                    post_output.reshape(-1),
                    mixes,
                    fused_post_input_record=packed_input_record,
                    fused_post_output=post_output,
                    fused_partial_metadata=metadata,
                    launcher=launcher,
                ),
                8 * 16,
                release_group_roles=(
                    ("hc_metadata", "metadata"),
                    ("hc_residual", "residual"),
                ),
            ),
            SequentialStage(
                "hc_pre_rms",
                SchedDsv4HcPreRms(
                    post_output,
                    mixes,
                    scale,
                    base,
                    norm_weight,
                    normalized,
                    next_post,
                    next_comb,
                    residual_square_sum=residual_square_sum,
                    packed_metadata=metadata,
                    packed_output=packed_output,
                    split_metadata_splits=16,
                ),
                1,
                base_sm=8 * 16,
                wait_group_roles=(
                    ("hc_metadata", "metadata"),
                    ("hc_residual", "residual"),
                ),
            ),
        ),
        balance_load_ports=True,
    )
    launcher.s(program)

    def validate() -> float:
        expected_residual_fp32 = post[:, None] * branch.float()[None, :]
        expected_residual_fp32 += torch.einsum(
            "ij,id->jd", comb.float(), residual.float()
        )
        expected_residual = expected_residual_fp32.to(torch.bfloat16)
        expected_mixes = projection @ expected_residual_fp32.reshape(-1)
        coefficient_rstd = torch.rsqrt(
            expected_residual_fp32.square().mean() + 1.0e-6
        )
        normalized_mixes = expected_mixes * coefficient_rstd
        expected_pre = (
            torch.sigmoid(normalized_mixes[:4] * scale[0] + base[:4])
            + 1.0e-6
        )
        expected_post = 2 * torch.sigmoid(
            normalized_mixes[4:8] * scale[1] + base[4:8]
        )
        expected_comb = (
            normalized_mixes[8:].reshape(4, 4) * scale[2]
            + base[8:].reshape(4, 4)
        )
        expected_comb = expected_comb.softmax(dim=-1) + 1.0e-6
        expected_comb = expected_comb / (
            expected_comb.sum(dim=-2, keepdim=True) + 1.0e-6
        )
        for _ in range(19):
            expected_comb = expected_comb / (
                expected_comb.sum(dim=-1, keepdim=True) + 1.0e-6
            )
            expected_comb = expected_comb / (
                expected_comb.sum(dim=-2, keepdim=True) + 1.0e-6
            )
        hidden = (
            expected_pre[:, None] * expected_residual.float()
        ).sum(dim=0).to(torch.bfloat16)
        expected = _rms(hidden, norm_weight, 1.0e-6).to(torch.bfloat16)
        torch.testing.assert_close(
            post_output, expected_residual, rtol=2.0e-2, atol=1.0e-2
        )
        torch.testing.assert_close(
            normalized, expected, rtol=2.0e-2, atol=1.0e-2
        )
        torch.testing.assert_close(
            next_post, expected_post, rtol=2.0e-5, atol=2.0e-5
        )
        torch.testing.assert_close(
            next_comb, expected_comb, rtol=2.0e-5, atol=2.0e-5
        )
        return _max_abs(normalized, expected)

    return Case(
        launcher,
        validate,
        (
            "OP_DSV4_FP32_BF16_GEMV",
            "OP_DSV4_HC_PRE_RMS",
        ),
        "fused_post_project24_splitk16_k256_pre_rms_k4096_reducer_sm128",
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
    pretransformed: bool = False,
) -> Case:
    logits = torch.randn(
        (256,), generator=generator, dtype=torch.float32, device=device
    )
    bias = torch.randn(
        (256,), generator=generator, dtype=torch.float32, device=device
    ) * 0.1
    hash_indices = torch.zeros((8,), dtype=torch.int32, device=device)
    hash_indices[:6] = torch.tensor(
        (9, 71, 5, 255, 130, 44), dtype=torch.int32, device=device
    )
    packed_output = None
    if pretransformed:
        packed_output = torch.empty((64,), dtype=torch.uint8, device=device)
        output_indices = packed_output[:32].view(torch.int32)
        output_weights = packed_output[32:].view(torch.float32)
    else:
        output_indices = torch.empty((8,), dtype=torch.int32, device=device)
        output_weights = torch.empty((8,), dtype=torch.float32, device=device)
    route_input = logits
    route_bias = bias
    if pretransformed:
        original = torch.nn.functional.softplus(logits).sqrt()
        route_input = torch.stack((original, original + bias), dim=1)
        route_bias = None
    launcher = _launcher(
        device,
        1,
        SchedDsv4RouteTop6(
            route_input,
            route_bias,
            hash_indices,
            output_indices,
            output_weights,
            hash_routing=hash_routing,
            pretransformed=pretransformed,
            packed_output=packed_output,
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
        (
            "OP_DSV4_ROUTE_TOP6_PREPARED"
            if pretransformed
            else "OP_DSV4_ROUTE_TOP6"
        ),
        f"experts256_top6_hash={int(hash_routing)}_prepared={int(pretransformed)}_sms1",
    )


def build_router_bf16_gemv(
    device: torch.device,
    generator: torch.Generator,
) -> Case:
    hidden = torch.randn(
        (4096,), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.125
    weight = torch.randn(
        (256, 4096), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.125
    logits = torch.empty((256,), dtype=torch.float32, device=device)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    launcher = _launcher(
        device,
        sms,
        SchedDsv4Bf16Gemv(weight, hidden, logits).place(sms),
    )

    def validate() -> float:
        expected = weight.float() @ hidden.float()
        torch.testing.assert_close(logits, expected, rtol=1.0e-3, atol=1.0e-3)
        return _max_abs(logits, expected)

    return Case(
        launcher,
        validate,
        "OP_DSV4_BF16_GEMV",
        f"router_m256_k4096_fp32_sms{sms}",
    )


def build_router_bf16_gemv_grouped(
    device: torch.device,
    generator: torch.Generator,
    *,
    rows_per_task: int,
) -> Case:
    hidden = torch.randn(
        (4096,), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.125
    weight = torch.randn(
        (256, 4096), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.125
    bias = torch.randn(
        (256,), generator=generator, dtype=torch.float32, device=device
    ) * 0.1
    prepared = torch.empty((256, 2), dtype=torch.float32, device=device)
    tasks = 256 // rows_per_task
    sms = min(torch.cuda.get_device_properties(device).multi_processor_count, tasks)
    launcher = _launcher(
        device,
        sms,
        SchedDsv4RouterBf16Gemv(
            weight,
            hidden,
            bias,
            prepared,
            rows_per_task=rows_per_task,
        ).place(sms),
    )

    def validate() -> float:
        expected = weight.float() @ hidden.float()
        original = torch.nn.functional.softplus(expected).sqrt()
        expected_prepared = torch.stack((original, original + bias), dim=1)
        torch.testing.assert_close(
            prepared, expected_prepared, rtol=1.0e-3, atol=1.0e-3
        )
        return _max_abs(prepared, expected_prepared)

    return Case(
        launcher,
        validate,
        f"OP_DSV4_ROUTER_BF16_GEMV_SM100__ROWS_{rows_per_task}",
        f"router_m256_k4096_routeprep_rows{rows_per_task}_sms{sms}",
    )


def build_router_ffn_ready(
    device: torch.device,
    generator: torch.Generator,
) -> Case:
    """Fork BF16 routing and exact FFN MXFP8 packing in one resident launch."""
    hidden = torch.randn(
        (4096,), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.125
    weight = torch.randn(
        (256, 4096), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.125
    bias = torch.randn(
        (256,), generator=generator, dtype=torch.float32, device=device
    ) * 0.1
    hash_indices = torch.zeros((8,), dtype=torch.int32, device=device)
    prepared = torch.empty((256, 2), dtype=torch.float32, device=device)
    route_output = torch.empty((64,), dtype=torch.uint8, device=device)
    route_indices = route_output[:32].view(torch.int32)
    route_weights = route_output[32:].view(torch.float32)
    activation_records = torch.empty(
        (8, SchedDsv4Mxfp8QuantFfnInput.RECORD_BYTES),
        dtype=torch.uint8,
        device=device,
    )

    resident_sms = torch.cuda.get_device_properties(device).multi_processor_count
    router_rows_per_task = 2
    quant_sms = 8
    router_sms = 128
    route_sm = router_sms
    quant_base_sm = resident_sms - quant_sms
    if router_sms > quant_base_sm:
        raise ValueError("router-to-FFN-ready placement requires at least 137 SMs")

    launcher = Launcher(resident_sms, device=device)
    program = SequentialProgram(
        launcher,
        (
            SequentialStage(
                "ffn_mxfp8_pack",
                SchedDsv4Mxfp8QuantFfnInput(hidden, activation_records),
                quant_sms,
                base_sm=quant_base_sm,
                release_group="mx8_ready",
            ),
            SequentialStage(
                "router_projection",
                SchedDsv4RouterBf16Gemv(
                    weight,
                    hidden,
                    bias,
                    prepared,
                    rows_per_task=router_rows_per_task,
                ),
                router_sms,
                wait_for_previous=False,
                parallel_with_previous=True,
                release_group="router_scores_ready",
            ),
            SequentialStage(
                "router_top6",
                SchedDsv4RouteTop6(
                    prepared,
                    None,
                    hash_indices,
                    route_indices,
                    route_weights,
                    pretransformed=True,
                    packed_output=route_output,
                ),
                1,
                base_sm=route_sm,
                wait_group_roles=(("router_scores_ready", "logits"),),
                release_group="experts_ready",
            ),
        ),
    )
    launcher.s(program)

    def validate() -> float:
        expected_logits = weight.float() @ hidden.float()
        expected_original = torch.nn.functional.softplus(
            expected_logits
        ).sqrt()
        expected_prepared = torch.stack(
            (expected_original, expected_original + bias), dim=1
        )
        torch.testing.assert_close(
            prepared, expected_prepared, rtol=1.0e-3, atol=1.0e-3
        )
        expected_weights, expected_indices = route_top6_reference(
            expected_logits, bias
        )
        torch.testing.assert_close(
            route_indices[:6], expected_indices, rtol=0, atol=0
        )
        torch.testing.assert_close(
            route_weights[:6], expected_weights, rtol=2.0e-5, atol=2.0e-5
        )

        expected_data, expected_scales = _mxfp8_ffn_input_reference(hidden)
        torch.testing.assert_close(
            activation_records[:, :4096], expected_data, rtol=0, atol=0
        )
        active_scale_indices = (
            torch.arange(8, device=device).reshape(-1, 1) * 16
            + torch.arange(4, device=device).reshape(1, -1)
        ).reshape(-1)
        actual_scales = activation_records[:, 4096:].reshape(8, 4, 512)[
            :, :, active_scale_indices
        ]
        torch.testing.assert_close(
            actual_scales, expected_scales, rtol=0, atol=0
        )
        return max(
            _max_abs(prepared, expected_prepared),
            _max_abs(route_weights[:6], expected_weights),
        )

    return Case(
        launcher,
        validate,
        (
            "OP_DSV4_MXFP8_QUANT_FFN_INPUT_SM100",
            "OP_DSV4_ROUTER_BF16_GEMV_SM100__ROWS_2",
            "OP_DSV4_ROUTE_TOP6_PREPARED",
        ),
        "router_m256_k4096_top6_plus_exact_mxfp8_one_launch_sms152",
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


def build_mxfp8_ffn_input(
    device: torch.device,
    generator: torch.Generator,
) -> Case:
    source = torch.randn(
        (4096,), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.25
    source[::257] *= 8
    records = source.numel() // 512
    output = torch.empty((records, 6144), dtype=torch.uint8, device=device)
    launcher = _launcher(
        device,
        records,
        SchedDsv4Mxfp8QuantFfnInput(source, output).place(records),
    )

    def validate() -> float:
        expected_data, expected_scales = _mxfp8_ffn_input_reference(source)
        torch.testing.assert_close(output[:, :4096], expected_data, rtol=0, atol=0)
        active_scale_indices = (
            torch.arange(8, device=device).reshape(-1, 1) * 16
            + torch.arange(4, device=device).reshape(1, -1)
        ).reshape(-1)
        actual_scales = output[:, 4096:].reshape(records, 4, 512)[
            :, :, active_scale_indices
        ]
        torch.testing.assert_close(
            actual_scales, expected_scales, rtol=0, atol=0
        )
        return 0.0

    return Case(
        launcher,
        validate,
        "OP_DSV4_MXFP8_QUANT_FFN_INPUT_SM100",
        "bf16_k4096_to_8x_k512_record_sms8",
    )


CASES: dict[str, Callable[[torch.device, torch.Generator], Case]] = {
    "preload_rope4": build_preload,
    "hc_pre_rms": lambda d, g: build_hc_pre_rms(d, g, zero_output=False),
    "hc_pre_rms_zero": lambda d, g: build_hc_pre_rms(d, g, zero_output=True),
    "hc_pre_rms_fp8": lambda d, g: build_hc_pre_rms(
        d, g, zero_output=False, output_fp8=True
    ),
    "hc_pre_rms_zero_fp8": lambda d, g: build_hc_pre_rms(
        d, g, zero_output=True, output_fp8=True
    ),
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
    **{
        f"hc_post_bf16_sms{sms}": (
            lambda d, g, sms=sms: build_hc_post(
                d, g, fp32_branch=False, sms=sms
            )
        )
        for sms in (1, 2, 4, 8, 16, 64, 128)
    },
    "hc_post_pre_boundary": build_hc_post_pre_boundary,
    "nvfp4_native_k2048": lambda d, g: build_nvfp4_native(d, g, k=2048),
    "nvfp4_native_k4096": lambda d, g: build_nvfp4_native(d, g, k=4096),
    "pool_packed_ratio128": build_pool_packed,
    "pool_rms_rope_d512": lambda d, g: build_pool_rms_rope(d, g, width=512),
    "pool_rms_rope_hadamard_d128": lambda d, g: build_pool_rms_rope(d, g, width=128),
    "argmax_partial": build_argmax_partial,
    "route_score": lambda d, g: build_route(d, g, hash_routing=False),
    "route_score_pretransformed": lambda d, g: build_route(
        d, g, hash_routing=False, pretransformed=True
    ),
    "route_hash": lambda d, g: build_route(d, g, hash_routing=True),
    "router_bf16_gemv": build_router_bf16_gemv,
    "router_bf16_gemv_rows1": lambda d, g: build_router_bf16_gemv_grouped(
        d, g, rows_per_task=1
    ),
    "router_bf16_gemv_rows2": lambda d, g: build_router_bf16_gemv_grouped(
        d, g, rows_per_task=2
    ),
    "router_bf16_gemv_rows4": lambda d, g: build_router_bf16_gemv_grouped(
        d, g, rows_per_task=4
    ),
    "router_ffn_ready": build_router_ffn_ready,
    "hc_head_rms": build_hc_head_rms,
    "argmax_reduce": build_argmax_reduce,
    "fp8_native_k2048": lambda d, g: build_fp8_native(d, g, k=2048),
    "fp8_native_k4096": lambda d, g: build_fp8_native(d, g, k=4096),
    "mxfp8_ffn_input_k4096": build_mxfp8_ffn_input,
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
    parser.add_argument("--print-runtime-profile", action="store_true")
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
    required_opcodes = (
        case.opcode if isinstance(case.opcode, tuple) else (case.opcode,)
    )
    missing = [opcode for opcode in required_opcodes if opcode not in supported]
    if supported and missing:
        raise AssertionError(f"{missing} are not selected in the loaded image")
    opcode_label = ",".join(required_opcodes)
    props = torch.cuda.get_device_properties(device)
    print(
        "DSV4_NONGEMM_FULL_IMAGE_RESULT "
        f"case={args.case} opcode={opcode_label} shape={case.shape} "
        f"cold_us={cold_us:.6f} min_us={min(hot_us):.6f} "
        f"median_us={statistics.median(hot_us):.6f} "
        f"max_us={max(hot_us):.6f} max_abs={max_abs:.8f} "
        f"includes_preload={int(case.includes_preload)} status=PASS "
        f"device={props.name!r} cc={props.major}.{props.minor}",
        flush=True,
    )
    if args.print_runtime_profile:
        names = {
            96: "compute_m2c_wait_ns",
            97: "compute_m2c_wait_calls",
            98: "compute_m2c_contended",
            99: "alloc_slot_stall_ns",
            100: "alloc_slot_stall_events",
            101: "alloc_slot_retries",
            102: "alloc_issue_barrier_ns",
            103: "alloc_issue_barrier_contended",
            104: "alloc_instructions",
            105: "ldu0_queue_wait_ns",
            106: "ldu0_queue_wait_calls",
            107: "ldu0_dependency_wait_ns",
            108: "ldu0_dependency_contended",
            109: "ldu0_commands",
            110: "ldu1_queue_wait_ns",
            111: "ldu1_queue_wait_calls",
            112: "ldu1_dependency_wait_ns",
            113: "ldu1_dependency_contended",
            114: "ldu1_commands",
            115: "store_queue_wait_ns",
            116: "store_queue_wait_calls",
            117: "store_service_ns",
            118: "store_barrier_service_ns",
            119: "store_commands",
            120: "store_barrier_commands",
        }
        profile = case.launcher.profile.cpu()
        for event, name in names.items():
            values = profile[:, event].to(torch.float64)
            suffix = "_us" if name.endswith("_ns") else ""
            scale = 1.0e-3 if suffix else 1.0
            print(
                "DSV4_RUNTIME_PROFILE "
                f"event={name.removesuffix('_ns')}{suffix} "
                f"min={float(values.min()) * scale:.6f} "
                f"median={float(values.median()) * scale:.6f} "
                f"max={float(values.max()) * scale:.6f}",
                flush=True,
            )
        start_ns = float(profile[:, 0].to(torch.float64).min())
        task_ends_us = (profile[:, 1].to(torch.float64) - start_ns) * 1.0e-3
        for quantile in (0.0, 0.5, 0.9, 1.0):
            print(
                "DSV4_TASK_END_PROFILE "
                f"quantile={quantile:.1f} "
                f"time_us={float(torch.quantile(task_ends_us, quantile)):.6f}",
                flush=True,
            )
        if "router_m256_k4096_top6_plus_exact_mxfp8" in case.shape:
            router_ends = task_ends_us[:128]
            quant_ends = task_ends_us[-8:]
            for stage, values in (
                ("router_projection", router_ends),
                ("router_top6", task_ends_us[128:129]),
                ("ffn_mxfp8_pack", quant_ends),
            ):
                print(
                    "DSV4_ROUTER_STAGE_END_PROFILE "
                    f"stage={stage} min_us={float(values.min()):.6f} "
                    f"median_us={float(values.median()):.6f} "
                    f"max_us={float(values.max()):.6f}",
                    flush=True,
                )
        if "router_m256_k4096" in case.shape:
            router_profile = profile[: min(128, case.launcher.num_sms)]
            for stage, begin_event, end_event in (
                ("input_wait", 80, 81),
                ("weight_wait_after_input", 81, 82),
                ("dot", 82, 83),
                ("release_reduce", 83, 84),
                ("metadata_wait", 84, 85),
                ("epilogue", 85, 86),
                ("task", 80, 86),
            ):
                values = (
                    router_profile[:, end_event].to(torch.float64)
                    - router_profile[:, begin_event].to(torch.float64)
                ) * 1.0e-3
                print(
                    "DSV4_ROUTER_INTERNAL_PROFILE "
                    f"stage={stage} min_us={float(values.min()):.6f} "
                    f"median_us={float(values.median()):.6f} "
                    f"max_us={float(values.max()):.6f}",
                    flush=True,
                )
        if "fused_post_project24_splitk" in case.shape:
            producer_splits = 16 if "splitk16" in case.shape else 8
            producer_groups = (
                case.launcher.num_sms - 1
            ) // producer_splits
            for group in range(producer_groups):
                group_ends = task_ends_us[
                    group * producer_splits:(group + 1) * producer_splits
                ]
                print(
                    "DSV4_PRODUCER_GROUP_PROFILE "
                    f"group={group} min_us={float(group_ends.min()):.6f} "
                    f"median_us={float(group_ends.median()):.6f} "
                    f"max_us={float(group_ends.max()):.6f}",
                    flush=True,
                )
        for event, name in (
            (121, "pre_metadata_dependency_ready"),
            (122, "pre_metadata_reduced"),
            (94, "pre_sinkhorn_initial_done"),
            (124, "pre_coefficients_done"),
            (125, "pre_residual_ready"),
            (126, "pre_output_done"),
            (95, "pre_sinkhorn_iteration10_done"),
        ):
            profile_sm = (
                case.launcher.num_sms - 1
                if "reducer_sm" in case.shape
                else 0
            )
            timestamp = float(profile[profile_sm, event].to(torch.float64))
            if timestamp > 0:
                print(
                    "DSV4_STAGE_PROFILE "
                    f"event={name} time_us={(timestamp - start_ns) * 1.0e-3:.6f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
