#!/usr/bin/env python3
"""Single-GPU correctness sweep for DeepSeek-V4-Flash-specific tasks."""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable

import torch

from dae.deepseek_v4 import (
    DeepSeekV4FlashConfig,
    apply_partial_rope_128_64,
    apply_partial_rope_512_64,
    bounded_swiglu,
    gated_pool_reference,
    hadamard_reference,
    hc_head_reference,
    hc_post_reference,
    hc_pre_reference,
    index_score_reference,
    pack_gated_pool_history,
    route_top6_reference,
    sparse_attention_512_reference,
)
from dae.deepseek_v4_quant import quantize_fp8_block128, quantize_nvfp4
from dae.instructions import TmaTensor
from dae.launcher import Launcher
from dae.schedule import (
    SchedDsv4Bf16Gemv,
    SchedDsv4ExpertReduce,
    SchedDsv4Fp8Quant128,
    SchedDsv4Fp32Bf16Gemv,
    SchedDsv4GatedPool,
    SchedDsv4GatedPoolPacked8Shard128,
    SchedDsv4Hadamard,
    SchedDsv4HcHead,
    SchedDsv4HcPost,
    SchedDsv4HcPre,
    SchedDsv4IndexScore,
    SchedDsv4Nvfp4Quant16,
    SchedDsv4Rope128_64,
    SchedDsv4Rope512_64,
    SchedDsv4RouteTop6,
    SchedDsv4SparseAttention512,
    SchedDsv4ContiguousAttention512Block4,
    SchedDsv4ContiguousAttention512UmmaSm100,
    SchedDsv4ContiguousAttention512UmmaTail32Sm100,
    SchedDsv4TopK512,
    SchedRMS,
    SchedSmemSiLUInterleaved,
)
from dae.tma_utils import Major


_BENCH_WARMUP = 0
_BENCH_ITERATIONS = 1
_INDEX_ROWS = 640
_ATTENTION_TOPK = 512
_ATTENTION_IMPLEMENTATION = "both"
_HC_POST_SMS = 32


def launch(schedule, num_sms: int, device: torch.device) -> float:
    launcher = Launcher(num_sms, device=device)
    launcher.s(schedule.place(num_sms))
    for _ in range(_BENCH_WARMUP):
        launcher.launch()
    timings_us = []
    for _ in range(_BENCH_ITERATIONS):
        launcher.launch()
        profile = launcher.profile[:, :2].cpu().numpy()
        timings_us.append(
            float(profile[:, 1].max() - profile[:, 0].min()) / 1.0e3
        )
    return statistics.median(timings_us)


def launch_with_launcher(launcher: Launcher, schedule) -> float:
    launcher.s(schedule.place(launcher.num_sms))
    for _ in range(_BENCH_WARMUP):
        launcher.launch()
    timings_us = []
    for _ in range(_BENCH_ITERATIONS):
        launcher.launch()
        profile = launcher.profile[:, :2].cpu().numpy()
        timings_us.append(
            float(profile[:, 1].max() - profile[:, 0].min()) / 1.0e3
        )
    return statistics.median(timings_us)


def report_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    latency_us: float,
) -> None:
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    max_abs = (actual.float() - expected.float()).abs().max().item()
    print(
        f"DSV4_FUNCTIONAL task={name} status=PASS "
        f"max_abs={max_abs:.8f} latency_us={latency_us:.3f}",
        flush=True,
    )


def run_rope(device: torch.device, generator: torch.Generator) -> None:
    angles = torch.linspace(-1.25, 1.25, 32, dtype=torch.float32, device=device)
    table = torch.stack((angles.cos(), angles.sin()), dim=1)
    for width, schedule_cls, reference in (
        (512, SchedDsv4Rope512_64, apply_partial_rope_512_64),
        (128, SchedDsv4Rope128_64, apply_partial_rope_128_64),
    ):
        source = torch.randn(
            (64, width), generator=generator, dtype=torch.bfloat16, device=device
        )
        for inverse in (False, True):
            output = torch.empty_like(source)
            latency = launch(
                schedule_cls(source, table, output, inverse=inverse), 1, device
            )
            expected = reference(source, table, inverse=inverse)
            report_close(
                f"rope{width}_64_inverse_{int(inverse)}",
                output,
                expected,
                rtol=1.0e-2,
                atol=1.0e-2,
                latency_us=latency,
            )


def run_quantization(device: torch.device, generator: torch.Generator) -> None:
    source = torch.randn(
        (4096,), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.125
    source[::257] *= 16

    fp8_output = torch.empty_like(source, dtype=torch.float8_e4m3fn)
    fp8_scale = torch.empty(
        (source.numel() // 128,), dtype=torch.float8_e8m0fnu, device=device
    )
    num_device_sms = torch.cuda.get_device_properties(device).multi_processor_count
    fp8_sms = min(source.numel() // 128, num_device_sms)
    fp8_latency = launch(
        SchedDsv4Fp8Quant128(source, fp8_output, fp8_scale), fp8_sms, device
    )
    expected_fp8, expected_fp8_scale = quantize_fp8_block128(source)
    torch.testing.assert_close(
        fp8_output.view(torch.uint8),
        expected_fp8.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        fp8_scale.view(torch.uint8),
        expected_fp8_scale.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    print(
        "DSV4_FUNCTIONAL task=fp8_activation_quant128 status=PASS "
        f"max_abs=0.00000000 latency_us={fp8_latency:.3f}",
        flush=True,
    )

    expected_nvfp4, expected_nvfp4_scale, global_scale = quantize_nvfp4(source)
    nvfp4_output = torch.empty(
        (source.numel() // 2,), dtype=torch.uint8, device=device
    )
    nvfp4_scale = torch.empty(
        (source.numel() // 16,), dtype=torch.float8_e4m3fn, device=device
    )
    nvfp4_latency = launch(
        SchedDsv4Nvfp4Quant16(
            source, global_scale, nvfp4_output, nvfp4_scale
        ),
        min(source.numel() // 16, num_device_sms),
        device,
    )
    torch.testing.assert_close(nvfp4_output, expected_nvfp4, rtol=0, atol=0)
    torch.testing.assert_close(
        nvfp4_scale.view(torch.uint8),
        expected_nvfp4_scale.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    print(
        "DSV4_FUNCTIONAL task=nvfp4_activation_quant16 status=PASS "
        f"max_abs=0.00000000 latency_us={nvfp4_latency:.3f}",
        flush=True,
    )


def run_attention(device: torch.device, generator: torch.Generator) -> None:
    config = DeepSeekV4FlashConfig()
    kv_rows = 768
    q = torch.randn(
        (config.num_heads, config.head_dim),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ) * 0.125
    kv = torch.randn(
        (kv_rows, config.head_dim),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ) * 0.125
    indices = torch.randperm(kv_rows, generator=generator, device=device)[
        : _ATTENTION_TOPK
    ].to(torch.int32)
    sink = torch.linspace(
        -0.5, 0.5, config.num_heads, dtype=torch.float32, device=device
    )
    expected = sparse_attention_512_reference(q, kv, indices, sink)
    if _ATTENTION_IMPLEMENTATION in ("both", "scalar"):
        output = torch.empty_like(q)
        latency = launch(
            SchedDsv4SparseAttention512(q, kv, indices, sink, output),
            config.num_heads,
            device,
        )
        report_close(
            f"sparse_attention_scalar_h64_d512_k{_ATTENTION_TOPK}",
            output,
            expected,
            rtol=3.0e-2,
            atol=1.0e-2,
            latency_us=latency,
        )

    if _ATTENTION_IMPLEMENTATION in ("both", "contiguous"):
        contiguous_indices = torch.arange(
            _ATTENTION_TOPK, dtype=torch.int32, device=device
        )
        contiguous_expected = sparse_attention_512_reference(
            q, kv, contiguous_indices, sink
        )
        contiguous_output = torch.empty_like(q)
        contiguous_latency = launch(
            SchedDsv4ContiguousAttention512Block4(
                q, kv, _ATTENTION_TOPK, sink, contiguous_output
            ),
            config.num_heads,
            device,
        )
        report_close(
            f"attention_contiguous_block4_h64_d512_k{_ATTENTION_TOPK}",
            contiguous_output,
            contiguous_expected,
            rtol=3.0e-2,
            atol=1.0e-2,
            latency_us=contiguous_latency,
        )

    if _ATTENTION_IMPLEMENTATION == "umma" and _ATTENTION_TOPK > 160:
        raise ValueError("UMMA attention currently supports at most 160 rows")
    if (_ATTENTION_IMPLEMENTATION in ("both", "umma") and
            _ATTENTION_TOPK <= 160):
        contiguous_indices = torch.arange(
            _ATTENTION_TOPK, dtype=torch.int32, device=device
        )
        contiguous_expected = sparse_attention_512_reference(
            q, kv, contiguous_indices, sink
        )
        umma_output = torch.empty_like(q)
        umma_launcher = Launcher(1, device=device)
        q_tma = TmaTensor(umma_launcher, q).wgmma_load(64, 128, Major.K)
        k_tma = TmaTensor(umma_launcher, kv).wgmma_load(128, 128, Major.K)
        v_tma = TmaTensor(umma_launcher, kv).wgmma_load(128, 128, Major.MN)
        output_tma = TmaTensor(umma_launcher, umma_output).rowmajor_2d(
            "store", 64, 128
        )
        if _ATTENTION_TOPK <= 128:
            umma_schedule = SchedDsv4ContiguousAttention512UmmaSm100(
                q,
                kv,
                _ATTENTION_TOPK,
                sink,
                umma_output,
                q_tma=q_tma,
                k_tma=k_tma,
                v_tma=v_tma,
                output_tma=output_tma,
            )
        else:
            tail_k_tma = TmaTensor(umma_launcher, kv).wgmma_load(
                32, 128, Major.K
            )
            tail_v_tma = TmaTensor(umma_launcher, kv).wgmma_load(
                32, 128, Major.MN
            )
            umma_schedule = SchedDsv4ContiguousAttention512UmmaTail32Sm100(
                q,
                kv,
                _ATTENTION_TOPK,
                sink,
                umma_output,
                q_tma=q_tma,
                prefix_k_tma=k_tma,
                tail_k_tma=tail_k_tma,
                prefix_v_tma=v_tma,
                tail_v_tma=tail_v_tma,
                output_tma=output_tma,
            )
        umma_latency = launch_with_launcher(
            umma_launcher,
            umma_schedule,
        )
        umma_diff = (umma_output.float() - contiguous_expected.float()).abs()
        umma_cos = torch.nn.functional.cosine_similarity(
            umma_output.float().reshape(1, -1),
            contiguous_expected.float().reshape(1, -1),
        ).item()
        per_tile_max = [
            umma_diff[:, start:start + 128].max().item()
            for start in range(0, 512, 128)
        ]
        print(
            "DSV4_FUNCTIONAL_DIAGNOSTIC "
            f"task=attention_contiguous_umma_h64_d512_k{_ATTENTION_TOPK} "
            f"latency_us={umma_latency:.3f} "
            f"max_abs={umma_diff.max().item():.8f} "
            f"mean_abs={umma_diff.mean().item():.8f} "
            f"cosine={umma_cos:.8f} tile_max={per_tile_max}",
            flush=True,
        )
        report_close(
            f"attention_contiguous_umma_h64_d512_k{_ATTENTION_TOPK}",
            umma_output,
            contiguous_expected,
            rtol=3.0e-2,
            atol=1.0e-2,
            latency_us=umma_latency,
        )


def run_router(device: torch.device, generator: torch.Generator) -> None:
    logits = torch.randn(
        (256,), generator=generator, dtype=torch.bfloat16, device=device
    )
    bias = torch.linspace(-0.4, 0.4, 256, dtype=torch.float32, device=device)
    bias[[3, 29, 71]] += torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float32, device=device
    )
    hash_indices = torch.zeros((8,), dtype=torch.int32, device=device)
    hash_indices[:6] = torch.tensor(
        [9, 71, 5, 255, 130, 44], dtype=torch.int32, device=device
    )

    for hash_routing in (False, True):
        output_indices = torch.empty((8,), dtype=torch.int32, device=device)
        output_weights = torch.empty((8,), dtype=torch.float32, device=device)
        latency = launch(
            SchedDsv4RouteTop6(
                logits,
                bias,
                hash_indices,
                output_indices,
                output_weights,
                hash_routing=hash_routing,
            ),
            1,
            device,
        )
        expected_weights, expected_indices = route_top6_reference(
            logits,
            bias,
            hash_indices=hash_indices[:6] if hash_routing else None,
        )
        torch.testing.assert_close(output_indices[:6], expected_indices, rtol=0, atol=0)
        report_close(
            f"route_top6_hash_{int(hash_routing)}",
            output_weights[:6],
            expected_weights,
            rtol=2.0e-5,
            atol=2.0e-5,
            latency_us=latency,
        )


def run_expert_reduce(device: torch.device, generator: torch.Generator) -> None:
    routed = torch.randn(
        (6, 4096), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.125
    weights = torch.rand((6,), generator=generator, dtype=torch.float32, device=device)
    weights = weights / weights.sum() * 1.5
    shared = torch.randn(
        (4096,), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.125
    output = torch.empty_like(shared)
    latency = launch(
        SchedDsv4ExpertReduce(routed, weights, shared, output), 1, device
    )
    expected = (
        shared.float() + (routed.float() * weights[:, None]).sum(dim=0)
    ).to(torch.bfloat16)
    report_close(
        "expert_reduce_top6_shared1",
        output,
        expected,
        rtol=2.0e-2,
        atol=1.0e-2,
        latency_us=latency,
    )


def run_compression_indexer(
    device: torch.device,
    generator: torch.Generator,
) -> None:
    q_source = torch.randn(
        (64, 128), generator=generator, dtype=torch.bfloat16, device=device
    )
    q = torch.empty_like(q_source)
    hadamard_latency = launch(
        SchedDsv4Hadamard(q_source, q), 64, device
    )
    report_close(
        "index_hadamard_h64_d128",
        q,
        hadamard_reference(q_source),
        rtol=2.0e-2,
        atol=2.0e-2,
        latency_us=hadamard_latency,
    )

    wide_source = torch.randn(
        (2, 512), generator=generator, dtype=torch.bfloat16, device=device
    )
    wide_output = torch.empty_like(wide_source)
    wide_latency = launch(
        SchedDsv4Hadamard(wide_source, wide_output), 2, device
    )
    report_close(
        "hadamard_rows2_d512",
        wide_output,
        hadamard_reference(wide_source),
        rtol=2.0e-2,
        atol=2.0e-2,
        latency_us=wide_latency,
    )

    pool_shapes = ((8, 512, "ratio4_overlap"), (128, 512, "ratio128"))
    for pool_rows, width, label in pool_shapes:
        values = torch.randn(
            (pool_rows, width),
            generator=generator,
            dtype=torch.float32,
            device=device,
        ) * 0.125
        scores = torch.randn(
            (pool_rows, width),
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        pooled = torch.empty((width,), dtype=torch.bfloat16, device=device)
        pool_latency = launch(
            SchedDsv4GatedPool(values, scores, pooled), 1, device
        )
        expected_pool = gated_pool_reference(values, scores).to(torch.bfloat16)
        report_close(
            f"compress_pool_{label}",
            pooled,
            expected_pool,
            rtol=2.0e-2,
            atol=1.0e-2,
            latency_us=pool_latency,
        )
        if pool_rows == 8:
            tail_bias = torch.randn(
                (width,),
                generator=generator,
                dtype=torch.float32,
                device=device,
            )
            segmented = torch.empty_like(pooled)
            segmented_latency = launch(
                SchedDsv4GatedPool(
                    values[:-1],
                    scores[:-1],
                    segmented,
                    tail_values=values[-1],
                    tail_scores=scores[-1],
                    tail_bias=tail_bias,
                ),
                1,
                device,
            )
            biased_scores = scores.clone()
            biased_scores[-1].add_(tail_bias)
            expected_segmented = gated_pool_reference(
                values, biased_scores
            ).to(torch.bfloat16)
            report_close(
                "compress_pool_ratio4_segmented_tail_bias",
                segmented,
                expected_segmented,
                rtol=2.0e-2,
                atol=1.0e-2,
                latency_us=segmented_latency,
            )
        if pool_rows == 128:
            tail_bias = torch.randn(
                (width,),
                generator=generator,
                dtype=torch.float32,
                device=device,
            )
            packed_history = pack_gated_pool_history(
                values[:-1], scores[:-1]
            )
            packed_output = torch.empty_like(pooled)
            packed_latency = launch(
                SchedDsv4GatedPoolPacked8Shard128(
                    packed_history,
                    pool_rows - 1,
                    packed_output,
                    tail_values=values[-1],
                    tail_scores=scores[-1],
                    tail_bias=tail_bias,
                ),
                width // 128,
                device,
            )
            biased_scores = scores.clone()
            biased_scores[-1].add_(tail_bias)
            packed_expected = gated_pool_reference(
                values, biased_scores
            ).to(torch.bfloat16)
            report_close(
                "compress_pool_ratio128_packed8_shard128",
                packed_output,
                packed_expected,
                rtol=2.0e-2,
                atol=1.0e-2,
                latency_us=packed_latency,
            )

    rows = _INDEX_ROWS
    kv = torch.randn(
        (rows, 128), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.125
    head_weights = torch.randn(
        (64,), generator=generator, dtype=torch.float32, device=device
    ) / (128 * 64) ** 0.5
    scores = torch.empty((rows,), dtype=torch.float32, device=device)
    num_sms = min(rows, torch.cuda.get_device_properties(device).multi_processor_count)
    score_latency = launch(
        SchedDsv4IndexScore(q, kv, head_weights, scores), num_sms, device
    )
    expected_scores = index_score_reference(q, kv, head_weights)
    report_close(
        "index_score_h64_d128",
        scores,
        expected_scores,
        rtol=2.0e-4,
        atol=2.0e-4,
        latency_us=score_latency,
    )

    indices = torch.empty((512,), dtype=torch.int32, device=device)
    topk_latency = launch(
        SchedDsv4TopK512(scores, indices, index_offset=128), 1, device
    )
    expected_indices = scores.topk(512).indices.to(torch.int32) + 128
    torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)
    print(
        "DSV4_FUNCTIONAL task=index_topk512 status=PASS "
        f"max_abs=0.00000000 latency_us={topk_latency:.3f}",
        flush=True,
    )


def run_hc(device: torch.device, generator: torch.Generator) -> None:
    residual = torch.randn(
        (4, 4096), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.125
    projection = torch.randn(
        (24, 4 * 4096), generator=generator, dtype=torch.float32, device=device
    ) * 0.005
    mixes = torch.empty((24,), dtype=torch.float32, device=device)
    gemv_latency = launch(
        SchedDsv4Fp32Bf16Gemv(projection, residual.reshape(-1), mixes),
        24,
        device,
    )
    expected_mixes = projection @ residual.float().reshape(-1)
    report_close(
        "hc_projection_fp32_bf16",
        mixes,
        expected_mixes,
        rtol=1.0e-4,
        atol=1.0e-4,
        latency_us=gemv_latency,
    )

    scale = torch.tensor([0.75, 1.25, 0.5], dtype=torch.float32, device=device)
    base = torch.randn(
        (24,), generator=generator, dtype=torch.float32, device=device
    ) * 0.1
    hidden = torch.empty((4096,), dtype=torch.bfloat16, device=device)
    post = torch.empty((4,), dtype=torch.float32, device=device)
    comb = torch.empty((4, 4), dtype=torch.float32, device=device)
    pre_latency = launch(
        SchedDsv4HcPre(residual, mixes, scale, base, hidden, post, comb),
        1,
        device,
    )
    expected_hidden, expected_post, expected_comb = hc_pre_reference(
        residual, mixes, scale, base
    )
    report_close(
        "hc_pre_hidden",
        hidden,
        expected_hidden,
        rtol=2.0e-2,
        atol=1.0e-2,
        latency_us=pre_latency,
    )
    torch.testing.assert_close(post, expected_post, rtol=2.0e-5, atol=2.0e-5)
    torch.testing.assert_close(comb, expected_comb, rtol=2.0e-5, atol=2.0e-5)

    branch = torch.randn(
        (4096,), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.125
    output = torch.empty_like(residual)
    post_latency = launch(
        SchedDsv4HcPost(branch, residual, post, comb, output),
        _HC_POST_SMS,
        device,
    )
    expected_output = hc_post_reference(branch, residual, post, comb)
    report_close(
        "hc_post",
        output,
        expected_output,
        rtol=2.0e-2,
        atol=1.0e-2,
        latency_us=post_latency,
    )

    head_mixes = mixes[:4].clone()
    head_scale = torch.tensor([0.625], dtype=torch.float32, device=device)
    head_base = base[:4].clone()
    head_output = torch.empty((4096,), dtype=torch.bfloat16, device=device)
    head_latency = launch(
        SchedDsv4HcHead(
            residual, head_mixes, head_scale, head_base, head_output
        ),
        1,
        device,
    )
    expected_head = hc_head_reference(
        residual, head_mixes, head_scale, head_base
    )
    report_close(
        "hc_head",
        head_output,
        expected_head,
        rtol=2.0e-2,
        atol=1.0e-2,
        latency_us=head_latency,
    )


def run_bf16_linear(device: torch.device, generator: torch.Generator) -> None:
    rows, k = 256, 4096
    weight = torch.randn(
        (rows, k), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.01
    source = torch.randn(
        (k,), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.01
    output = torch.empty((rows,), dtype=torch.bfloat16, device=device)
    latency = launch(
        SchedDsv4Bf16Gemv(weight, source, output),
        min(rows, torch.cuda.get_device_properties(device).multi_processor_count),
        device,
    )
    expected = (weight.float() @ source.float()).to(torch.bfloat16)
    report_close(
        "bf16_checkpoint_gemv",
        output,
        expected,
        rtol=2.0e-2,
        atol=5.0e-2,
        latency_us=latency,
    )

    output_fp32 = torch.empty((rows,), dtype=torch.float32, device=device)
    latency = launch(
        SchedDsv4Bf16Gemv(weight, source, output_fp32),
        min(rows, torch.cuda.get_device_properties(device).multi_processor_count),
        device,
    )
    expected_fp32 = weight.float() @ source.float()
    report_close(
        "bf16_checkpoint_gemv_fp32",
        output_fp32,
        expected_fp32,
        rtol=1.0e-5,
        atol=1.0e-5,
        latency_us=latency,
    )


def run_norm_activation(device: torch.device, generator: torch.Generator) -> None:
    for width in (512, 1024):
        source = torch.randn(
            (1, width), generator=generator, dtype=torch.bfloat16, device=device
        ) * 0.25
        weight = (
            torch.randn(
                (width,), generator=generator, dtype=torch.bfloat16, device=device
            )
            * 0.05
            + 1.0
        )
        output = torch.empty_like(source)
        latency = launch(
            SchedRMS(1, 1.0e-6, source, output, weight, hidden_size=width),
            1,
            device,
        )
        expected = (
            source.float()
            * torch.rsqrt(source.float().square().mean(dim=-1, keepdim=True) + 1.0e-6)
            * weight.float()
        ).to(torch.bfloat16)
        report_close(
            f"rmsnorm_{width}",
            output,
            expected,
            rtol=3.0e-2,
            atol=3.0e-2,
            latency_us=latency,
        )

    gate = torch.randn(
        (1, 2048), generator=generator, dtype=torch.bfloat16, device=device
    ) * 8.0
    up = torch.randn(
        (1, 2048), generator=generator, dtype=torch.bfloat16, device=device
    ) * 8.0
    output = torch.empty_like(gate)
    latency = launch(
        SchedSmemSiLUInterleaved(
            1, gate, up, output, swiglu_limit=10.0
        ),
        1,
        device,
    )
    expected = bounded_swiglu(gate, up, limit=10.0)
    report_close(
        "bounded_swiglu_2048",
        output,
        expected,
        rtol=2.0e-2,
        atol=6.0e-2,
        latency_us=latency,
    )


def main() -> None:
    global _ATTENTION_IMPLEMENTATION, _ATTENTION_TOPK
    global _BENCH_ITERATIONS, _BENCH_WARMUP, _HC_POST_SMS
    global _INDEX_ROWS

    tasks: dict[str, Callable[[torch.device, torch.Generator], None]] = {
        "quantization": run_quantization,
        "rope": run_rope,
        "attention": run_attention,
        "router": run_router,
        "expert-reduce": run_expert_reduce,
        "compression-indexer": run_compression_indexer,
        "hc": run_hc,
        "bf16-linear": run_bf16_linear,
        "norm-activation": run_norm_activation,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("all", *tasks), default="all")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--index-rows", type=int, default=640)
    parser.add_argument("--attention-topk", type=int, default=512)
    parser.add_argument(
        "--attention-implementation",
        choices=("both", "scalar", "contiguous", "umma"),
        default="both",
    )
    parser.add_argument("--hc-post-sms", type=int, default=32)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be non-negative and iterations must be positive")
    _BENCH_WARMUP = args.warmup
    _BENCH_ITERATIONS = args.iterations
    if args.index_rows < 512 or args.index_rows > 0xFFFF:
        parser.error("index rows must be in [512,65535]")
    _INDEX_ROWS = args.index_rows
    if args.attention_topk <= 0 or args.attention_topk > 768:
        parser.error("attention top-k must be in [1,768]")
    _ATTENTION_TOPK = args.attention_topk
    _ATTENTION_IMPLEMENTATION = args.attention_implementation
    if (
        args.hc_post_sms <= 0
        or 4096 % args.hc_post_sms
        or (4096 // args.hc_post_sms) % 8
    ):
        parser.error("hc-post-sms must produce 16-byte-aligned equal shards")
    _HC_POST_SMS = args.hc_post_sms

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("highest")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    selected = tasks.items() if args.task == "all" else ((args.task, tasks[args.task]),)
    for _, function in selected:
        function(device, generator)

    props = torch.cuda.get_device_properties(device)
    print(
        f"DSV4_FUNCTIONAL_SUMMARY status=PASS device={props.name!r} "
        f"cc={props.major}.{props.minor} task={args.task}",
        flush=True,
    )


if __name__ == "__main__":
    main()
