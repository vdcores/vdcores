#!/usr/bin/env python3
"""Focused one-launch correctness probe for clean-room pre-attention fusions."""

from __future__ import annotations

import torch

from dae.deepseek_v4 import (
    apply_partial_rope_512_64,
    pack_gated_pool_history,
)
from dae.instructions import TmaTensor
from dae.launcher import Launcher
from dae.schedule import (
    SchedDsv4Bf16GemvGroup4SplitK,
    SchedDsv4Fp32RmsFp8QuantUmmaB,
    SchedDsv4Fp32RmsRope512_64,
    SchedDsv4Fp32RopeHadamard128,
    SchedDsv4GatedPoolRmsRope,
    SchedDsv4GatedPoolPacked8RmsPartial,
    SchedDsv4GatedPoolPacked8HistoryState,
    SchedDsv4GatedPoolTailRmsPartial,
    SchedDsv4Fp32RmsRopeShard128,
    SchedDsv4HcPre,
    SchedDsv4HcPreRms,
    SchedDsv4RmsFp8QuantUmmaB,
    SchedDsv4RmsRope512_64,
)
from dae.sequential import (
    LoopedSequentialProgram,
    SequentialBlock,
    SequentialProgram,
    SequentialStage,
)
from dae.tma_utils import Major


def rms_rope_reference(
    source: torch.Tensor,
    table: torch.Tensor,
    epsilon: float,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    normalized = source.float() * torch.rsqrt(
        source.float().square().mean(dim=-1, keepdim=True) + epsilon
    )
    if weight is not None:
        normalized *= weight.float()
    return apply_partial_rope_512_64(normalized, table).to(torch.bfloat16)


def partial_rope_reference(
    source: torch.Tensor, table: torch.Tensor
) -> torch.Tensor:
    output = source.float().clone()
    even = output[..., -64::2].clone()
    odd = output[..., -63::2].clone()
    output[..., -64::2] = (
        even * table[:, 0] - odd * table[:, 1]
    )
    output[..., -63::2] = (
        even * table[:, 1] + odd * table[:, 0]
    )
    return output


def hadamard_reference(source: torch.Tensor) -> torch.Tensor:
    output = source.float().clone()
    width = output.shape[-1]
    stride = 1
    while stride < width:
        output = output.reshape(-1, width // (2 * stride), 2, stride)
        lhs = output[:, :, 0].clone()
        rhs = output[:, :, 1].clone()
        output[:, :, 0] = lhs + rhs
        output[:, :, 1] = lhs - rhs
        output = output.reshape(*source.shape)
        stride *= 2
    return output * (width**-0.5)


def main() -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260813)
    epsilon = float(torch.tensor(1.0e-6, dtype=torch.bfloat16).float())
    angles = torch.linspace(-1.0, 1.0, 32, device=device)
    table = torch.stack((angles.cos(), angles.sin()), dim=1)

    q = torch.randn(
        (64, 512), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.25
    q_output = torch.empty_like(q)
    kv = torch.randn(
        (1, 512), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.25
    kv_weight = (
        torch.randn(
            (512,), generator=generator, dtype=torch.bfloat16, device=device
        )
        * 0.05
        + 1.0
    )
    kv_output = torch.empty_like(kv)

    q_rank = torch.randn(
        (1024,), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.25
    q_rank_weight = (
        torch.randn(
            (1024,), generator=generator, dtype=torch.bfloat16, device=device
        )
        * 0.05
        + 1.0
    )
    q_rank_native = torch.empty((8, 2048), dtype=torch.uint8, device=device)
    q_rank_fp32_native = torch.empty_like(q_rank_native)
    q_rank_fp32 = q_rank.float().contiguous()
    q_fp32 = q.float().contiguous()
    q_fp32_output = torch.empty_like(q)

    index_fp32 = torch.randn(
        (64, 128), generator=generator, dtype=torch.float32, device=device
    ) * 0.25
    index_output = torch.empty(
        (64, 128), dtype=torch.bfloat16, device=device
    )

    pool_history_values = torch.randn(
        (3, 512), generator=generator, dtype=torch.float32, device=device
    ) * 0.25
    pool_history_scores = torch.randn(
        (3, 512), generator=generator, dtype=torch.float32, device=device
    ) * 0.1
    pool_tail_values = torch.randn(
        (512,), generator=generator, dtype=torch.float32, device=device
    ) * 0.25
    pool_tail_scores = torch.randn(
        (512,), generator=generator, dtype=torch.float32, device=device
    ) * 0.1
    pool_bias = torch.randn(
        (512,), generator=generator, dtype=torch.float32, device=device
    ) * 0.1
    pool_weight = (
        torch.randn(
            (512,), generator=generator, dtype=torch.bfloat16, device=device
        )
        * 0.05
        + 1.0
    )
    pool_output = torch.empty(
        (512,), dtype=torch.bfloat16, device=device
    )
    index_pool_weight = pool_weight[:128].contiguous()
    index_pool_history_values = pool_history_values[:, :128].contiguous()
    index_pool_history_scores = pool_history_scores[:, :128].contiguous()
    index_pool_tail_values = pool_tail_values[:128].contiguous()
    index_pool_tail_scores = pool_tail_scores[:128].contiguous()
    index_pool_bias = pool_bias[:128].contiguous()
    index_pool_output = torch.empty(
        (128,), dtype=torch.bfloat16, device=device
    )

    grouped_weight = torch.randn(
        (512, 4096), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.01
    grouped_input = torch.randn(
        (4096,), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.01
    grouped_output = torch.zeros((4, 128), dtype=torch.float32, device=device)

    launcher = Launcher(64, device=device)
    grouped_weight_tma = TmaTensor(launcher, grouped_weight).wgmma_load(
        128, 128, Major.K
    )
    grouped_output_tma = TmaTensor(launcher, grouped_output).rowmajor_2d(
        "reduce", 4, 128
    )
    launcher.s(
        SchedDsv4RmsRope512_64(
            q,
            table,
            q_output,
            epsilon=epsilon,
        ).place(64),
        SchedDsv4RmsRope512_64(
            kv,
            table,
            kv_output,
            epsilon=epsilon,
            weight=kv_weight,
        ).place(1),
        SchedDsv4RmsFp8QuantUmmaB(
            q_rank,
            q_rank_weight,
            q_rank_native,
            epsilon,
        ).place(4),
        SchedDsv4Fp32RmsFp8QuantUmmaB(
            q_rank_fp32,
            q_rank_weight,
            q_rank_fp32_native,
            epsilon,
        ).place(4),
        SchedDsv4Fp32RmsRope512_64(
            q_fp32,
            table,
            q_fp32_output,
            epsilon=epsilon,
        ).place(64),
        SchedDsv4Fp32RopeHadamard128(
            index_fp32,
            table,
            index_output,
        ).place(64),
        SchedDsv4GatedPoolRmsRope(
            pool_history_values,
            pool_history_scores,
            pool_weight,
            table,
            pool_output,
            epsilon=epsilon,
            tail_values=pool_tail_values,
            tail_scores=pool_tail_scores,
            tail_bias=pool_bias,
        ).place(1),
        SchedDsv4GatedPoolRmsRope(
            index_pool_history_values,
            index_pool_history_scores,
            index_pool_weight,
            table,
            index_pool_output,
            epsilon=epsilon,
            tail_values=index_pool_tail_values,
            tail_scores=index_pool_tail_scores,
            tail_bias=index_pool_bias,
            hadamard=True,
        ).place(1),
        SchedDsv4Bf16GemvGroup4SplitK(
            grouped_weight,
            grouped_weight_tma,
            grouped_input,
            grouped_output_tma,
            split_k=8,
        ).place(8),
    )
    snapshots = None
    repetitions = 8
    for _ in range(repetitions):
        grouped_output.zero_()
        launcher.launch()
        current = (
            q_output.view(torch.uint16).clone(),
            kv_output.view(torch.uint16).clone(),
            q_rank_native.clone(),
            q_rank_fp32_native.clone(),
            q_fp32_output.view(torch.uint16).clone(),
            index_output.view(torch.uint16).clone(),
            pool_output.view(torch.uint16).clone(),
            # FWHT cancellation may produce either sign of exact zero.
            index_pool_output.clone(),
        )
        if snapshots is None:
            snapshots = current
        else:
            for actual, expected in zip(current, snapshots):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    torch.testing.assert_close(
        q_output,
        rms_rope_reference(q, table, epsilon),
        rtol=2.0e-2,
        atol=1.0e-2,
    )
    torch.testing.assert_close(
        q_fp32_output,
        rms_rope_reference(q, table, epsilon),
        rtol=2.0e-2,
        atol=1.0e-2,
    )
    torch.testing.assert_close(
        kv_output,
        rms_rope_reference(kv, table, epsilon, kv_weight),
        rtol=2.0e-2,
        atol=1.0e-2,
    )

    normalized = q_rank.float() * torch.rsqrt(
        q_rank.float().square().mean() + epsilon
    ) * q_rank_weight.float()
    blocks = normalized.reshape(8, 128)
    requested_scale = torch.clamp(
        blocks.abs().amax(dim=1) / 448.0, min=2.0**-127
    )
    scales = torch.exp2(torch.ceil(torch.log2(requested_scale)).clamp(-127, 127))
    expected_quant = torch.clamp(
        blocks / scales[:, None], -448.0, 448.0
    ).to(torch.float8_e4m3fn)
    torch.testing.assert_close(
        q_rank_native[:, :128],
        expected_quant.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    if not torch.equal(q_rank_fp32_native, q_rank_native):
        print(
            "DSV4_FP32_RMS_FP8_DEBUG "
            f"bf16_head={q_rank_native[0, :16].tolist()} "
            f"fp32_head={q_rank_fp32_native[0, :16].tolist()} "
            f"bf16_nonzero={int(torch.count_nonzero(q_rank_native).item())} "
            f"fp32_nonzero={int(torch.count_nonzero(q_rank_fp32_native).item())}",
            flush=True,
        )
    torch.testing.assert_close(
        q_rank_fp32_native,
        q_rank_native,
        rtol=0,
        atol=0,
    )
    expected_scale_bytes = scales.to(torch.float8_e8m0fnu).view(torch.uint8)
    for group_start in range(0, 8, 2):
        packed_scales = q_rank_native[group_start, 1024:]
        populated = packed_scales[packed_scales != 0]
        assert populated.numel() == 16
        expected = expected_scale_bytes[group_start : group_start + 2]
        expected = expected[:, None].expand(2, 8).reshape(-1)
        torch.testing.assert_close(
            populated.sort().values,
            expected.sort().values,
            rtol=0,
            atol=0,
        )
        assert not bool(q_rank_native[group_start + 1, 1024:].any().item())
    torch.testing.assert_close(
        grouped_output.reshape(-1),
        grouped_weight.float() @ grouped_input.float(),
        rtol=2.0e-3,
        atol=2.0e-3,
    )

    index_reference = hadamard_reference(
        partial_rope_reference(index_fp32, table)
    ).to(torch.bfloat16)
    torch.testing.assert_close(
        index_output,
        index_reference,
        rtol=2.0e-2,
        atol=1.0e-2,
    )

    all_values = torch.cat(
        (pool_history_values, pool_tail_values.unsqueeze(0)), dim=0
    )
    all_scores = torch.cat(
        (
            pool_history_scores,
            (pool_tail_scores + pool_bias).unsqueeze(0),
        ),
        dim=0,
    )
    pooled = (
        torch.softmax(all_scores, dim=0) * all_values
    ).sum(dim=0)
    pooled_normalized = pooled * torch.rsqrt(
        pooled.square().mean() + epsilon
    ) * pool_weight.float()
    pool_reference = partial_rope_reference(
        pooled_normalized, table
    ).to(torch.bfloat16)
    torch.testing.assert_close(
        pool_output,
        pool_reference,
        rtol=2.0e-2,
        atol=1.0e-2,
    )
    index_values = all_values[:, :128]
    index_scores = all_scores[:, :128]
    index_pooled_reference = (
        torch.softmax(index_scores, dim=0) * index_values
    ).sum(dim=0)
    index_pooled_reference = index_pooled_reference * torch.rsqrt(
        index_pooled_reference.square().mean() + epsilon
    ) * index_pool_weight.float()
    index_pool_reference = hadamard_reference(
        partial_rope_reference(index_pooled_reference, table)
    ).to(torch.bfloat16)
    torch.testing.assert_close(
        index_pool_output,
        index_pool_reference,
        rtol=2.0e-2,
        atol=1.0e-2,
    )

    packed_history = pack_gated_pool_history(
        pool_history_values, pool_history_scores
    )
    packed_pooled = torch.empty(
        (4, 128), dtype=torch.float32, device=device
    )
    packed_partials = torch.empty(
        (4,), dtype=torch.float32, device=device
    )
    packed_pool_output = torch.empty_like(pool_output)
    packed_launcher = Launcher(4, device=device)
    packed_program = SequentialProgram(
        packed_launcher,
        (
            SequentialStage(
                "packed_pool_rms_partial",
                SchedDsv4GatedPoolPacked8RmsPartial(
                    packed_history,
                    pool_history_values.shape[0],
                    packed_pooled,
                    packed_partials,
                    tail_values=pool_tail_values,
                    tail_scores=pool_tail_scores,
                    tail_bias=pool_bias,
                ),
                4,
            ),
            SequentialStage(
                "packed_norm_rope",
                SchedDsv4Fp32RmsRopeShard128(
                    packed_pooled,
                    packed_partials,
                    pool_weight,
                    table,
                    packed_pool_output,
                    epsilon=epsilon,
                ),
                4,
            ),
        ),
    )
    packed_launcher.s(packed_program)
    packed_launcher.launch()
    torch.testing.assert_close(
        packed_pool_output,
        pool_reference,
        rtol=2.0e-2,
        atol=1.0e-2,
    )

    split_history_state = torch.empty(
        (4, 3, 128), dtype=torch.float32, device=device
    )
    split_pooled = torch.empty_like(packed_pooled)
    split_partials = torch.empty_like(packed_partials)
    split_pool_output = torch.empty_like(pool_output)
    split_launcher = Launcher(4, device=device)
    split_program = SequentialProgram(
        split_launcher,
        (
            SequentialStage(
                "packed_history_state",
                SchedDsv4GatedPoolPacked8HistoryState(
                    packed_history,
                    pool_history_values.shape[0],
                    split_history_state,
                ),
                4,
            ),
            SequentialStage(
                "projected_tail_rms_partial",
                SchedDsv4GatedPoolTailRmsPartial(
                    split_history_state,
                    split_pooled,
                    split_partials,
                    tail_values=pool_tail_values,
                    tail_scores=pool_tail_scores,
                    tail_bias=pool_bias,
                ),
                4,
            ),
            SequentialStage(
                "split_norm_rope",
                SchedDsv4Fp32RmsRopeShard128(
                    split_pooled,
                    split_partials,
                    pool_weight,
                    table,
                    split_pool_output,
                    epsilon=epsilon,
                ),
                4,
            ),
        ),
    )
    split_launcher.s(split_program)
    split_launcher.launch()
    torch.testing.assert_close(
        split_pool_output,
        pool_reference,
        rtol=2.0e-2,
        atol=1.0e-2,
    )

    hc_residual = torch.randn(
        (4, 4096), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.1
    hc_mixes = torch.randn(
        (24,), generator=generator, dtype=torch.float32, device=device
    ) * 0.1
    hc_scale = torch.tensor(
        [0.5, 0.75, 1.25], dtype=torch.float32, device=device
    )
    hc_base = torch.randn(
        (24,), generator=generator, dtype=torch.float32, device=device
    ) * 0.05
    hc_norm_weight = (
        torch.randn(
            (4096,), generator=generator, dtype=torch.bfloat16, device=device
        )
        * 0.05
        + 1.0
    )
    hc_hidden = torch.empty(
        (4096,), dtype=torch.bfloat16, device=device
    )
    hc_post = torch.empty((4,), dtype=torch.float32, device=device)
    hc_comb = torch.empty((4, 4), dtype=torch.float32, device=device)
    hc_residual_square_sum = hc_residual.float().square().sum().reshape(1)
    hc_packed_metadata = torch.empty(
        (56,), dtype=torch.float32, device=device
    )
    hc_packed_metadata[:1].copy_(hc_residual_square_sum)
    hc_packed_metadata[1:25].copy_(hc_mixes)
    hc_packed_metadata[28:31].copy_(hc_scale)
    hc_packed_metadata[31:55].copy_(hc_base)
    hc_packed_output = torch.empty(
        (4136,), dtype=torch.bfloat16, device=device
    )
    hc_output = hc_packed_output[:4096]
    hc_output_metadata = hc_packed_output[4096:].view(torch.float32)
    hc_fused_post = hc_output_metadata[:4]
    hc_fused_comb = hc_output_metadata[4:].view(4, 4)
    hc_reference_launcher = Launcher(1, device=device)
    hc_reference_launcher.s(
        SchedDsv4HcPre(
            hc_residual,
            hc_mixes,
            hc_scale,
            hc_base,
            hc_hidden,
            hc_post,
            hc_comb,
        ).place(1)
    )
    hc_fused_launcher = Launcher(1, device=device)
    hc_fused_launcher.s(
        SchedDsv4HcPreRms(
            hc_residual,
            hc_mixes,
            hc_scale,
            hc_base,
            hc_norm_weight,
            hc_output,
            hc_fused_post,
            hc_fused_comb,
            residual_square_sum=hc_residual_square_sum,
            packed_metadata=hc_packed_metadata,
            packed_output=hc_packed_output,
        ).place(1)
    )
    hc_reference_launcher.launch()
    hc_fused_launcher.launch()
    hc_reference = (
        hc_hidden.float()
        * torch.rsqrt(hc_hidden.float().square().mean() + epsilon)
        * hc_norm_weight.float()
    ).to(torch.bfloat16)
    torch.testing.assert_close(
        hc_output, hc_reference, rtol=2.0e-2, atol=1.0e-2
    )
    torch.testing.assert_close(
        hc_fused_post, hc_post, rtol=0, atol=0
    )
    torch.testing.assert_close(
        hc_fused_comb, hc_comb, rtol=0, atol=0
    )

    layered_weight = torch.stack(
        (grouped_weight, grouped_weight * 0.5), dim=0
    ).contiguous()
    layered_output = torch.zeros_like(grouped_output)
    layered_launcher = Launcher(8, device=device)
    layered_weight_tma = TmaTensor(
        layered_launcher, layered_weight
    ).wgmma_load(128, 128, Major.K)
    layered_output_tma = TmaTensor(
        layered_launcher, layered_output
    ).rowmajor_2d("reduce", 4, 128)
    layered_schedule = SchedDsv4Bf16GemvGroup4SplitK(
        layered_weight,
        layered_weight_tma,
        grouped_input,
        layered_output_tma,
        split_k=8,
        layer_indexed_weight=True,
    )
    layered_program = LoopedSequentialProgram(
        layered_launcher,
        (
            SequentialBlock(
                "two_layer_grouped_projection",
                (
                    SequentialStage(
                        "grouped_projection",
                        layered_schedule,
                        8,
                    ),
                ),
                repeat=2,
                barrier_banks=2,
            ),
        ),
    )
    layered_launcher.s(layered_program)
    layered_launcher.launch()
    torch.testing.assert_close(
        layered_output.reshape(-1),
        layered_weight.float().sum(dim=0) @ grouped_input.float(),
        rtol=2.0e-3,
        atol=2.0e-3,
    )

    print(
        f"DSV4_PREATTENTION_FUSION status=PASS launches={repetitions} "
        "q_rows=64 kv_rows=1 q_rank_tiles=8 bf16_group4_splitk=8 "
        "fp32_epilogues=4 fused_pool_widths=128,512 "
        "packed_pool_shards=4 split_history_pool_shards=4 "
        "hc_pre_rms=1 layered_weights=2",
        flush=True,
    )


if __name__ == "__main__":
    main()
