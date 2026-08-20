#!/usr/bin/env python3
"""Focused correctness/timing probes for the fused attention epilogue."""

from __future__ import annotations

import argparse
import math
import statistics

import torch

from dae.deepseek_v4 import sparse_attention_512_reference
from dae.deepseek_v4_quant import quantize_fp8_block128
from dae.instructions import TmaTensor
from dae.launcher import Launcher
from dae.schedule import (
    SchedDsv4AttentionSplit32UmmaSm100,
    SchedDsv4AttentionSplit64UmmaSm100,
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
) -> tuple[torch.Tensor, torch.Tensor]:
    scale_region = packed[:, :, 1024:]
    actual_scale_bits = torch.empty_like(scale_bits)
    rows = torch.arange(8, device=packed.device) * 16
    for group_start in range(0, packed.shape[1], 2):
        for scale_in_pair in range(2):
            copies = scale_region[
                :, group_start, rows + scale_in_pair
            ]
            if not bool((copies == copies[:, :1]).all().item()):
                raise AssertionError(
                    "native scale replication differs within an output tile"
                )
            actual_scale_bits[:, group_start + scale_in_pair] = copies[:, 0]
    exponent_delta = (
        actual_scale_bits.to(torch.int16) - scale_bits.to(torch.int16)
    ).abs()
    if int(exponent_delta.max().item()) > 1:
        raise AssertionError(
            "native scale differs from the value reference by more than "
            "one UE8M0 exponent"
        )
    return scale_region, actual_scale_bits


def unpack_native(
    packed: torch.Tensor, scale_bits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    heads = packed.shape[0]
    native_data = packed[:, :, :1024].reshape(heads, 4, 8, 8, 16)
    reconstructed = torch.empty_like(native_data)
    for row in range(8):
        for source_chunk in range(8):
            reconstructed[:, :, row, source_chunk].copy_(
                native_data[:, :, row, source_chunk ^ row]
            )
    bits = reconstructed[:, :, 0].reshape(heads, 512)
    scale_region, actual_scale_bits = validate_pack2_scales(
        packed, scale_bits
    )
    dequant = (
        bits.view(torch.float8_e4m3fn).float()
        * actual_scale_bits.view(torch.float8_e8m0fnu)
        .float()
        .repeat_interleave(128, dim=1)
    )
    return dequant, scale_region, actual_scale_bits


def benchmark_split_attention(
    rows: int,
    warmup: int,
    iterations: int,
    device: torch.device,
    *,
    correctness_repeats: int = 0,
    producer_only: bool = False,
    reducer_only: bool = False,
    with_oa: bool = False,
    split64: bool = False,
    detail_profile: bool = False,
) -> None:
    generator = torch.Generator(device=device).manual_seed(20260813 + rows)
    split_size = 64 if split64 else 32
    num_splits = (rows + split_size - 1) // split_size
    q = torch.randn(
        (64, 512), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.125
    # Model the preallocated physical KV cache used by inference.  The logical
    # row count remains ``rows`` and masks the padded tail in the producer.
    kv = torch.randn(
        (num_splits * split_size, 512),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ) * 0.125
    sink = torch.randn(
        (64,), generator=generator, dtype=torch.float32, device=device
    ) * 0.1
    angles = torch.linspace(-1.25, 1.25, 32, device=device)
    table = torch.stack((angles.cos(), angles.sin()), dim=1)
    partials = torch.empty(
        (num_splits, 64, 512), dtype=torch.bfloat16, device=device
    )
    metadata = torch.empty(
        (num_splits, 64, 2), dtype=torch.float32, device=device
    )
    # Native-record padding is intentionally undefined.  Clear it on the
    # benchmark side so exact comparisons cover only bytes the task owns.
    packed = torch.zeros((64, 4, 2048), dtype=torch.uint8, device=device)

    launcher_sms = num_splits if producer_only else 128
    launcher = Launcher(launcher_sms, device=device)
    if reducer_only:
        partials.normal_(generator=generator)
        metadata[:, :, 0].normal_(generator=generator)
        metadata[:, :, 1].uniform_(0.5, 32.0, generator=generator)
        reducer = SchedDsv4AttentionSplitReduceFp8Sm100(
            partials, metadata, sink, table, packed
        )
        launcher.s(reducer.place(128))
        launcher.launch()
        cold_profile = launcher.profile[:, :2].cpu().numpy()
        cold_us = float(
            cold_profile[:, 1].max() - cold_profile[:, 0].min()
        ) / 1.0e3
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
            "DSV4_SPLIT_ATTN_REDUCER "
            f"rows={rows} splits={num_splits} status=COMPLETE "
            f"cold_us={cold_us:.3f} samples={iterations} "
            f"min_us={min(timings):.3f} "
            f"median_us={statistics.median(timings):.3f} "
            f"max_us={max(timings):.3f}",
            flush=True,
        )
        return
    if split64:
        q_tma = (
            TmaTensor(launcher, q)
            .wgmma_load(64, 512, Major.K)
            .encode_64k()
        )
        kv_tma = TmaTensor(launcher, kv).wgmma_load(64, 512, Major.K)
        kv_v_tma = TmaTensor(launcher, kv).wgmma_load(64, 128, Major.MN)
        partial_tma = TmaTensor(
            launcher, partials.reshape(num_splits * 64, 512)
        ).wgmma("store", 64, 128, Major.K)
        producer = SchedDsv4AttentionSplit64UmmaSm100(
            q,
            kv,
            rows,
            partials,
            metadata,
            q_tma=q_tma,
            kv_tma=kv_tma,
            kv_v_tma=kv_v_tma,
            partial_tma=partial_tma,
        )
    else:
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
        producer = producer.place(num_splits)
        launcher.s(producer)
        launcher.launch()
        torch.cuda.synchronize(device)
        expected_partials = []
        expected_metadata = []
        split_rows = (
            producer.split_rows
            if split64
            else [
                (
                    split * split_size,
                    min(split_size, rows - split * split_size),
                )
                for split in range(num_splits)
            ]
        )
        for row_start, active_tokens in split_rows:
            block = kv[row_start : row_start + active_tokens].float()
            scores = q.float() @ block.t() / math.sqrt(512.0)
            block_max = scores.max(dim=1).values * math.log2(math.e)
            block_mass = torch.exp2(
                scores * math.log2(math.e) - block_max[:, None]
            ).sum(dim=1)
            expected_partials.append(torch.softmax(scores, dim=1) @ block)
            expected_metadata.append(torch.stack((block_max, block_mass), dim=1))
        expected_partials = torch.stack(expected_partials)
        expected_metadata = torch.stack(expected_metadata)
        partial_max_abs = float(
            (partials.float() - expected_partials).abs().max().item()
        )
        partial_cosine = float(
            torch.nn.functional.cosine_similarity(
                partials.float().reshape(1, -1),
                expected_partials.reshape(1, -1),
            ).item()
        )
        metadata_max_abs = float(
            (metadata - expected_metadata).abs().max().item()
        )
        for _ in range(warmup):
            launcher.launch()
        timings = []
        for _ in range(iterations):
            launcher.launch()
            profile = launcher.profile[:, :2].cpu().numpy()
            timings.append(
                float(profile[:, 1].max() - profile[:, 0].min()) / 1.0e3
            )
        if detail_profile:
            detail = launcher.profile.cpu().numpy()[:num_splits]
            if not bool((detail[:, 127] == 0x4454524B50524631).all()):
                raise RuntimeError(
                    "detail profile requires a track_profile=1 runtime"
                )
            event_names = {
                2: "task-enter",
                3: "operands-ready",
                4: "qk-done",
                5: "output-ready",
                6: "softmax-done",
                20: "metadata-ready",
                7: "pv0-done",
                8: "pv0-store",
                9: "pv0-reuse",
                10: "pv1-done",
                11: "pv1-store",
                12: "pv1-reuse",
                13: "pv2-done",
                14: "pv2-store",
                15: "pv2-reuse",
                16: "pv3-done",
                17: "pv3-store",
                18: "pv3-reuse",
                19: "output-published",
                21: "task-done",
            }
            previous = detail[:, 0]
            pieces = []
            for event_id, name in event_names.items():
                current = detail[:, event_id]
                delta_us = statistics.median(
                    ((current - previous) / 1.0e3).tolist()
                )
                offset_us = statistics.median(
                    ((current - detail[:, 0]) / 1.0e3).tolist()
                )
                pieces.append(f"{name}={delta_us:.3f}/{offset_us:.3f}")
                previous = current
            print(
                "DSV4_SPLIT_ATTN_PROFILE delta_us/offset_us "
                + " ".join(pieces),
                flush=True,
            )
        print(
            "DSV4_SPLIT_ATTN_PRODUCER "
            f"rows={rows} splits={num_splits} status=COMPLETE "
            f"partial_max_abs={partial_max_abs:.8f} "
            f"partial_cosine={partial_cosine:.8f} "
            f"metadata_max_abs={metadata_max_abs:.8f} "
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
        partial_bars = (
            launcher.new_bar(num_splits),
            launcher.new_bar(num_splits),
        )
        producer = (
            producer
            .bar("output0", partial_bars[0])
            .bar("output1", partial_bars[1])
        )
        reducer = SchedDsv4AttentionSplitReduceFp8Sm100(
            partials, metadata, sink, table, packed
        )
        reducer = (
            reducer
            .bar("partials0", partial_bars[0])
            .bar("partials1", partial_bars[1])
        )
        launcher.s(producer.place(num_splits), reducer.place(128))
    launcher.launch()
    cold_profile = launcher.profile[:, :2].cpu().numpy()
    cold_us = float(
        cold_profile[:, 1].max() - cold_profile[:, 0].min()
    ) / 1.0e3
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

    def validate_output() -> tuple[torch.Tensor, torch.Tensor]:
        actual, _, actual_scale_bits = unpack_native(
            packed, reference_scale_bits
        )
        torch.testing.assert_close(
            actual, reference_dequant, rtol=8.0e-2, atol=8.0e-3
        )
        return actual, actual_scale_bits

    actual_dequant, actual_scale_bits = validate_output()
    scale_exponent_delta = (
        actual_scale_bits.to(torch.int16)
        - reference_scale_bits.to(torch.int16)
    ).abs()
    scale_exponent_mismatches = int(
        (scale_exponent_delta != 0).sum().item()
    )
    scale_exponent_max_delta = int(scale_exponent_delta.max().item())
    expected_packed = packed.clone()
    for _ in range(correctness_repeats):
        partials.fill_(float("nan"))
        metadata.fill_(float("nan"))
        packed.zero_()
        torch.cuda.synchronize(device)
        launcher.launch()
        torch.cuda.synchronize(device)
        validate_output()
        torch.testing.assert_close(packed, expected_packed, rtol=0, atol=0)
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
    if detail_profile:
        detail = launcher.profile.cpu().numpy()
        if not bool((detail[:, 127] == 0x4454524B50524631).all()):
            raise RuntimeError(
                "detail profile requires a track_profile=1 runtime"
            )
        producer_events = (2, 3, 4, 5, 6, 9, 12, 15, 18, 21)
        producer_offsets = " ".join(
            f"e{event_id}={statistics.median(((detail[:num_splits, event_id] - detail[:num_splits, 0]) / 1.0e3).tolist()):.3f}"
            for event_id in producer_events
        )
        reducer_names = {
            22: "enter",
            23: "dependency-ready",
            24: "weights-ready",
            25: "merge-done",
            26: "output-ready",
            27: "maxima-ready",
            28: "scales-ready",
            29: "published",
        }
        previous = detail[:, 0]
        reducer_pieces = []
        for event_id, name in reducer_names.items():
            current = detail[:, event_id]
            delta = (current - previous) / 1.0e3
            offset = (current - detail[:, 0]) / 1.0e3
            reducer_pieces.append(
                f"{name}={statistics.median(delta.tolist()):.3f}/"
                f"{statistics.median(offset.tolist()):.3f}/"
                f"{max(offset.tolist()):.3f}"
            )
            previous = current
        print(
            "DSV4_SPLIT_ATTN_FULL_PROFILE producer_offset_us "
            + producer_offsets,
            flush=True,
        )
        print(
            "DSV4_SPLIT_ATTN_REDUCER_PROFILE "
            "delta_us/median_offset_us/max_offset_us "
            + " ".join(reducer_pieces),
            flush=True,
        )
        for output_group in range(2):
            group = detail[output_group::2]
            dependency = (group[:, 23] - group[:, 0]) / 1.0e3
            published = (group[:, 29] - group[:, 0]) / 1.0e3
            print(
                "DSV4_SPLIT_ATTN_REDUCER_GROUP_PROFILE "
                f"output_group={output_group} "
                f"dependency_median_us={statistics.median(dependency.tolist()):.3f} "
                f"dependency_max_us={max(dependency.tolist()):.3f} "
                f"published_median_us={statistics.median(published.tolist()):.3f} "
                f"published_max_us={max(published.tolist()):.3f}",
                flush=True,
            )
    print(
        f"{'DSV4_SPLIT_ATTN_OA_FUSED' if with_oa else 'DSV4_SPLIT_ATTN_FUSED'} "
        f"rows={rows} splits={num_splits} status=PASS "
        f"max_abs={difference.max().item():.8f} "
        f"mean_abs={difference.mean().item():.8f} cosine={cosine:.8f} "
        f"scale_exp_mismatches={scale_exponent_mismatches} "
        f"scale_exp_max_delta={scale_exponent_max_delta} "
        f"correctness_repeats={correctness_repeats} "
        f"cold_us={cold_us:.3f} samples={iterations} min_us={min(timings):.3f} "
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
    parser.add_argument("--split64", action="store_true")
    parser.add_argument("--detail-profile", action="store_true")
    parser.add_argument("--correctness-repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    if not 0 < args.heads <= 64:
        parser.error("heads must be in [1,64]")
    if args.correctness_repeats < 0 or args.warmup < 0 or args.iterations <= 0:
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
                correctness_repeats=args.correctness_repeats,
                producer_only=args.producer_only,
                reducer_only=args.reducer_only,
                with_oa=args.with_oa,
                split64=args.split64,
                detail_profile=args.detail_profile,
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
    actual_dequant, _, _ = unpack_native(packed, reference_scale_bits)
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
