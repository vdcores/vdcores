#!/usr/bin/env python3
"""Focused correctness/timing probes for the fused attention epilogue."""

from __future__ import annotations

import argparse
import statistics

import torch

from dae.deepseek_v4 import sparse_attention_512_reference
from dae.deepseek_v4_quant import quantize_fp8_block128
from dae.instructions import TmaTensor
from dae.launcher import Launcher
from dae.schedule import (
    SchedDsv4AttentionSplit32UmmaSm100,
    SchedDsv4AttentionSplitReduceFp8Sm100,
    SchedDsv4InverseRopeFp8QuantUmmaB,
    SchedFp8GemvUmmaSplitK,
)
from dae.sequential import SequentialProgram, SequentialStage
from dae.tma_utils import Major


def inverse_rope_float(source: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    output = source.float().clone()
    pairs = output[:, -64:].reshape(source.shape[0], 32, 2)
    even = pairs[:, :, 0].clone()
    odd = pairs[:, :, 1].clone()
    cosine = table[:, 0]
    sine = table[:, 1]
    pairs[:, :, 0] = even * cosine + odd * sine
    pairs[:, :, 1] = odd * cosine - even * sine
    return output


def validate_pack2_scales(
    packed: torch.Tensor, scale_bits: torch.Tensor
) -> torch.Tensor:
    scale_region = packed[:, :, 1024:]
    expected = torch.zeros_like(scale_region)
    for group_start in range(0, packed.shape[1], 2):
        expected[:, group_start, :8] = scale_bits[:, group_start, None]
        expected[:, group_start, 8:16] = scale_bits[:, group_start + 1, None]
    torch.testing.assert_close(
        scale_region.sort(dim=-1).values,
        expected.sort(dim=-1).values,
        rtol=0,
        atol=0,
    )
    return scale_region


def unpack_native(
    packed: torch.Tensor, scale_bits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    heads = packed.shape[0]
    native_data = packed[:, :, :1024].reshape(heads, 4, 8, 8, 16)
    reconstructed = torch.empty_like(native_data)
    for row in range(8):
        for source_chunk in range(8):
            reconstructed[:, :, row, source_chunk].copy_(
                native_data[:, :, row, source_chunk ^ row]
            )
    bits = reconstructed[:, :, 0].reshape(heads, 512)
    scale_region = validate_pack2_scales(packed, scale_bits)
    dequant = (
        bits.view(torch.float8_e4m3fn).float()
        * scale_bits.view(torch.float8_e8m0fnu)
        .float()
        .repeat_interleave(128, dim=1)
    )
    return dequant, scale_region


def benchmark_split_attention(
    rows: int,
    warmup: int,
    iterations: int,
    device: torch.device,
    *,
    producer_only: bool = False,
    reducer_only: bool = False,
    with_oa: bool = False,
) -> None:
    generator = torch.Generator(device=device).manual_seed(20260813 + rows)
    q = torch.randn(
        (64, 512), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.125
    kv = torch.randn(
        (rows, 512), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.125
    sink = torch.randn(
        (64,), generator=generator, dtype=torch.float32, device=device
    ) * 0.1
    angles = torch.linspace(-1.25, 1.25, 32, device=device)
    table = torch.stack((angles.cos(), angles.sin()), dim=1)
    num_splits = (rows + 31) // 32
    partials = torch.empty(
        (num_splits, 64, 512), dtype=torch.bfloat16, device=device
    )
    metadata = torch.empty(
        (num_splits, 64, 2), dtype=torch.float32, device=device
    )
    packed = torch.empty((64, 4, 2048), dtype=torch.uint8, device=device)

    launcher_sms = 128 if with_oa else (num_splits if producer_only else 64)
    launcher = Launcher(launcher_sms, device=device)
    if reducer_only:
        partials.normal_(generator=generator)
        metadata[:, :, 0].normal_(generator=generator)
        metadata[:, :, 1].uniform_(0.5, 32.0, generator=generator)
        reducer = SchedDsv4AttentionSplitReduceFp8Sm100(
            partials, metadata, sink, table, packed
        )
        launcher.s(reducer.place(64))
        launcher.launch()
        torch.cuda.synchronize(device)
        print(
            "DSV4_SPLIT_ATTN_REDUCER "
            f"rows={rows} splits={num_splits} status=COMPLETE",
            flush=True,
        )
        return
    q_tma = TmaTensor(launcher, q).wgmma_load(64, 128, Major.K)
    k_tma = TmaTensor(launcher, kv).wgmma_load(32, 128, Major.K)
    v_tma = TmaTensor(launcher, kv).wgmma_load(32, 128, Major.MN)
    partial_tma = TmaTensor(
        launcher, partials.reshape(num_splits * 64, 512)
    ).rowmajor_2d("store", 64, 128)
    producer = SchedDsv4AttentionSplit32UmmaSm100(
        q,
        kv,
        rows,
        partials,
        metadata,
        q_tma=q_tma,
        k_tma=k_tma,
        v_tma=v_tma,
        partial_tma=partial_tma,
    )
    if producer_only:
        launcher.s(producer.place(num_splits))
        launcher.launch()
        torch.cuda.synchronize(device)
        for _ in range(warmup):
            launcher.launch()
        timings = []
        for _ in range(iterations):
            launcher.launch()
            profile = launcher.profile[:, :2].cpu().numpy()
            timings.append(
                float(profile[:, 1].max() - profile[:, 0].min()) / 1.0e3
            )
        print(
            "DSV4_SPLIT_ATTN_PRODUCER "
            f"rows={rows} splits={num_splits} status=COMPLETE "
            f"samples={iterations} min_us={min(timings):.3f} "
            f"median_us={statistics.median(timings):.3f} "
            f"max_us={max(timings):.3f}",
            flush=True,
        )
        return

    if with_oa:
        weight_tiles = torch.zeros(
            (8, 32, SchedFp8GemvUmmaSplitK.WEIGHT_TILE_BYTES),
            dtype=torch.uint8,
            device=device,
        )
        # Keep the synthetic payload zero while giving every native UE8M0
        # weight-scale slot the exact encoding for 1.0.
        weight_tiles[:, :, 16384:].fill_(127)
        oa_output = torch.zeros(
            (8, 1024), dtype=torch.bfloat16, device=device
        )
        stages = [
            SequentialStage(
                "attention.producer",
                producer,
                num_splits,
                release_group="attention.partials",
            )
        ]
        grouped_packed = packed.view(8, 32, 2048)
        for group in range(8):
            ready = f"attention.group{group}.native"
            reducer = SchedDsv4AttentionSplitReduceFp8Sm100(
                partials,
                metadata,
                sink,
                table,
                packed,
                head_start=group * 8,
                head_count=8,
            )
            stages.append(
                SequentialStage(
                    f"attention.reducer{group}",
                    reducer,
                    8,
                    base_sm=group * 16,
                    wait_group="attention.partials",
                    release_group=ready,
                )
            )
            output_reduce = TmaTensor(
                launcher, oa_output[group : group + 1]
            ).rowmajor_2d("reduce", 1, 128)
            oa = SchedFp8GemvUmmaSplitK(
                weight_tiles,
                grouped_packed[group],
                output_reduce,
                2,
                scale_pack=2,
            )
            stages.append(
                SequentialStage(
                    f"oa.group{group}",
                    oa,
                    16,
                    base_sm=group * 16,
                    wait_group=ready,
                )
            )
        launcher.s(
            SequentialProgram(
                launcher, stages, balance_load_ports=True
            )
        )
    else:
        partial_bar = launcher.new_bar(num_splits)
        producer = producer.bar("output", partial_bar)
        reducer = SchedDsv4AttentionSplitReduceFp8Sm100(
            partials, metadata, sink, table, packed
        ).bar("partials", partial_bar)
        launcher.s(producer.place(num_splits), reducer.place(64))
    launcher.launch()
    torch.cuda.synchronize(device)
    if with_oa and bool((oa_output != 0).any().item()):
        raise AssertionError("zero-weight O_a schedule produced nonzero output")

    indices = torch.arange(rows, dtype=torch.int32, device=device)
    attention = sparse_attention_512_reference(q, kv, indices, sink)
    reference_float = inverse_rope_float(attention, table)
    reference_q, reference_scale = quantize_fp8_block128(
        reference_float.reshape(-1)
    )
    reference_dequant = (
        reference_q.float().reshape(64, 512)
        * reference_scale.float().reshape(64, 4).repeat_interleave(128, dim=1)
    )
    reference_scale_bits = reference_scale.view(torch.uint8).reshape(64, 4)
    actual_dequant, _ = unpack_native(packed, reference_scale_bits)
    torch.testing.assert_close(
        actual_dequant, reference_dequant, rtol=8.0e-2, atol=8.0e-3
    )
    difference = (actual_dequant - reference_dequant).abs()
    cosine = torch.nn.functional.cosine_similarity(
        actual_dequant.reshape(1, -1), reference_dequant.reshape(1, -1)
    ).item()

    for _ in range(warmup):
        launcher.launch()
    timings = []
    for _ in range(iterations):
        launcher.launch()
        profile = launcher.profile[:, :2].cpu().numpy()
        timings.append(float(profile[:, 1].max() - profile[:, 0].min()) / 1.0e3)
    print(
        f"{'DSV4_SPLIT_ATTN_OA_FUSED' if with_oa else 'DSV4_SPLIT_ATTN_FUSED'} "
        f"rows={rows} splits={num_splits} status=PASS "
        f"max_abs={difference.max().item():.8f} "
        f"mean_abs={difference.mean().item():.8f} cosine={cosine:.8f} "
        f"samples={iterations} min_us={min(timings):.3f} "
        f"median_us={statistics.median(timings):.3f} "
        f"max_us={max(timings):.3f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--rows", type=int, nargs="*", default=[])
    parser.add_argument("--producer-only", action="store_true")
    parser.add_argument("--reducer-only", action="store_true")
    parser.add_argument("--with-oa", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    if not 0 < args.heads <= 64:
        parser.error("heads must be in [1,64]")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("timing counts are invalid")

    device = torch.device("cuda")
    if args.rows:
        for rows in args.rows:
            if not 1 <= rows <= 768:
                parser.error("rows must be in [1,768]")
            benchmark_split_attention(
                rows,
                args.warmup,
                args.iterations,
                device,
                producer_only=args.producer_only,
                reducer_only=args.reducer_only,
                with_oa=args.with_oa,
            )
        return
    generator = torch.Generator(device=device).manual_seed(20260813)
    source = torch.randn(
        (args.heads, 512),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ) * 0.125
    angles = torch.linspace(-1.25, 1.25, 32, device=device)
    table = torch.stack((angles.cos(), angles.sin()), dim=1)
    packed = torch.empty(
        (args.heads, 4, 2048), dtype=torch.uint8, device=device
    )

    launcher = Launcher(args.heads, device=device)
    launcher.s(
        SchedDsv4InverseRopeFp8QuantUmmaB(
            source, table, packed
        ).place(args.heads),
    )
    launcher.launch()
    torch.cuda.synchronize(device)

    reference_float = inverse_rope_float(source, table)
    reference_q, reference_scale = quantize_fp8_block128(
        reference_float.reshape(-1)
    )
    reference_bits = reference_q.view(torch.uint8).reshape(args.heads, 4, 128)
    reference_scale_bits = reference_scale.view(torch.uint8).reshape(
        args.heads, 4
    )

    native_data = packed[:, :, :1024].reshape(args.heads, 4, 8, 8, 16)
    reconstructed = torch.empty_like(native_data)
    for row in range(8):
        for source_chunk in range(8):
            reconstructed[:, :, row, source_chunk].copy_(
                native_data[:, :, row, source_chunk ^ row]
            )
    expected_broadcast = reference_bits.reshape(
        args.heads, 4, 1, 8, 16
    ).expand_as(reconstructed)
    data_mismatches = int((reconstructed != expected_broadcast).sum().item())

    scale_region = packed[:, :, 1024:]
    expected_scale_region = torch.zeros_like(scale_region)
    for group_start in range(0, 4, 2):
        expected_scale_region[:, group_start, :8] = (
            reference_scale_bits[:, group_start, None]
        )
        expected_scale_region[:, group_start, 8:16] = (
            reference_scale_bits[:, group_start + 1, None]
        )
    scale_mismatches = int(
        (
            scale_region.sort(dim=-1).values
            != expected_scale_region.sort(dim=-1).values
        ).sum().item()
    )
    actual_dequant, _ = unpack_native(packed, reference_scale_bits)
    reference_dequant = (
        reference_q.float().reshape(args.heads, 512)
        * reference_scale.float().reshape(args.heads, 4)
        .repeat_interleave(128, dim=1)
    )
    torch.testing.assert_close(
        actual_dequant, reference_dequant, rtol=6.0e-2, atol=4.0e-3
    )
    max_abs = float((actual_dequant - reference_dequant).abs().max().item())
    cosine = float(
        torch.nn.functional.cosine_similarity(
            actual_dequant.reshape(1, -1),
            reference_dequant.reshape(1, -1),
        ).item()
    )

    for _ in range(args.warmup):
        launcher.launch()
    timings = []
    for _ in range(args.iterations):
        launcher.launch()
        profile = launcher.profile[:, :2].cpu().numpy()
        timings.append(float(profile[:, 1].max() - profile[:, 0].min()) / 1.0e3)
    print(
        "DSV4_FUSED_ATTN_EPILOGUE "
        f"heads={args.heads} status=PASS data_mismatches={data_mismatches} "
        f"scale_mismatches={scale_mismatches} max_abs={max_abs:.8f} "
        f"cosine={cosine:.8f} samples={args.iterations} "
        f"min_us={min(timings):.3f} median_us={statistics.median(timings):.3f} "
        f"max_us={max(timings):.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
