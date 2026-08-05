#!/usr/bin/env python3
"""Benchmark isolated non-GEMV VDCores Llama tasks on Blackwell."""

from __future__ import annotations

import argparse
import statistics

import torch
import torch.nn.functional as F

from dae.instructions import (
    ARGMAX_PARTIAL_bf16_1024_65536_128,
    ARGMAX_REDUCE_bf16_1024_128,
    ROPE_INTERLEAVE_512,
    TmaLoad1D,
    TmaStore1D,
    TmaTensor,
)
from dae.launcher import Launcher
from dae.schedule import (
    SchedArgmax,
    SchedRMSShared,
    SchedRope,
    SchedSmemSiLUInterleaved,
)
from dae.tma_utils import (
    Major,
    ToRopeTableCordAdapter,
    ToSplitMCordAdapter,
    tma_load_tbl,
)


BATCH = 8
HIDDEN = 4096
MLP_PREFIX = 6144
HEAD_DIM = 128
VOCAB_PADDED = 131072
EPS = 1.0e-5


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = expected.abs().float().mean().clamp_min(1.0e-8)
    return ((actual - expected).abs().float().mean() / denominator * 100.0).item()


@torch.inference_mode()
def benchmark_launcher(
    launcher: Launcher,
    *,
    op: str,
    shape: str,
    iterations: int,
    warmup: int,
    error_percent: float,
) -> None:
    if error_percent > 1.0:
        raise AssertionError(
            f"{op} {shape} mean-relative error {error_percent:.6f}% exceeds 1%"
        )
    for _ in range(warmup):
        launcher.launch()
    torch.cuda.synchronize()

    timings_us = []
    for _ in range(iterations):
        launcher.launch()
        profile = launcher.profile[:, :2].cpu().numpy()
        timings_us.append((profile[:, 1].max() - profile[:, 0].min()) / 1.0e3)

    print(
        "VDCORES_TASK_RESULT "
        f"op={op} shape={shape} sms={launcher.num_sms} "
        f"min_us={min(timings_us):.6f} median_us={statistics.median(timings_us):.6f} "
        f"max_us={max(timings_us):.6f} error_percent={error_percent:.6f}",
        flush=True,
    )


def benchmark_rms(args: argparse.Namespace, device: torch.device) -> None:
    generator = torch.Generator(device=device).manual_seed(11)
    x = torch.rand(
        (BATCH, HIDDEN), generator=generator, dtype=torch.bfloat16, device=device
    ) - 0.5
    weight = torch.rand(
        (HIDDEN,), generator=generator, dtype=torch.bfloat16, device=device
    ) + 0.5
    out = torch.empty_like(x)

    launcher = Launcher(BATCH, device=device)
    weight_tma = TmaTensor(launcher, weight).tensor1d("load", HIDDEN)
    schedule = SchedRMSShared(
        num_token=BATCH,
        epsilon=EPS,
        tmas=(
            weight_tma.cord(0),
            TmaLoad1D(x, bytes=HIDDEN * 2),
            TmaStore1D(out, bytes=HIDDEN * 2),
        ),
    ).place(BATCH)
    launcher.s(schedule)
    launcher.launch()
    torch.cuda.synchronize()
    expected = (
        x.float()
        * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + EPS)
        * weight.float()
    ).to(torch.bfloat16)
    benchmark_launcher(
        launcher,
        op="rms_norm_smem",
        shape=f"{BATCH}x{HIDDEN}",
        iterations=args.iterations,
        warmup=args.warmup,
        error_percent=relative_error(out, expected),
    )

def benchmark_silu(args: argparse.Namespace, device: torch.device) -> None:
    generator = torch.Generator(device=device).manual_seed(12)
    gate = torch.rand(
        (BATCH, MLP_PREFIX),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ) - 0.5
    up = torch.rand(
        (BATCH, MLP_PREFIX),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ) - 0.5
    out = torch.empty_like(gate)

    launcher = Launcher(BATCH, device=device)
    schedule = SchedSmemSiLUInterleaved(
        num_token=BATCH,
        gate_glob=gate,
        up_glob=up,
        out_glob=out,
    ).place(BATCH)
    launcher.s(schedule)
    launcher.launch()
    torch.cuda.synchronize()
    expected = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
    benchmark_launcher(
        launcher,
        op="silu_mul_smem_prefix",
        shape=f"{BATCH}x{MLP_PREFIX}",
        iterations=args.iterations,
        warmup=args.warmup,
        error_percent=relative_error(out, expected),
    )


def benchmark_silu_sharded(args: argparse.Namespace, device: torch.device) -> None:
    generator = torch.Generator(device=device).manual_seed(12)
    gate = torch.rand(
        (BATCH, MLP_PREFIX), generator=generator, dtype=torch.bfloat16, device=device
    ) - 0.5
    up = torch.rand(
        (BATCH, MLP_PREFIX), generator=generator, dtype=torch.bfloat16, device=device
    ) - 0.5
    out = torch.empty_like(gate)
    num_sms = BATCH * 3

    launcher = Launcher(num_sms, device=device)
    schedule = SchedSmemSiLUInterleaved(
        num_token=BATCH,
        gate_glob=gate,
        up_glob=up,
        out_glob=out,
        shards_per_token=3,
    ).place(num_sms)
    launcher.s(schedule)
    launcher.launch()
    torch.cuda.synchronize()
    expected = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
    benchmark_launcher(
        launcher,
        op="silu_mul_smem_prefix_shard3",
        shape=f"{BATCH}x{MLP_PREFIX}",
        iterations=args.iterations,
        warmup=args.warmup,
        error_percent=relative_error(out, expected),
    )


def build_interleaved_rope_table(
    max_seq_len: int, device: torch.device
) -> torch.Tensor:
    inv_freq = 500000.0 ** (
        -torch.arange(0, HEAD_DIM // 2, dtype=torch.float32, device=device)
        * 2
        / HEAD_DIM
    )
    angles = torch.outer(
        torch.arange(max_seq_len, dtype=torch.float32, device=device), inv_freq
    )
    table = torch.empty(
        (max_seq_len, BATCH, HEAD_DIM), dtype=torch.float32, device=device
    )
    table[:, :, 0::2] = angles.cos().unsqueeze(1)
    table[:, :, 1::2] = angles.sin().unsqueeze(1)
    return table.to(torch.bfloat16)


def benchmark_rope(args: argparse.Namespace, device: torch.device) -> None:
    generator = torch.Generator(device=device).manual_seed(13)
    q = torch.rand(
        (BATCH, HIDDEN), generator=generator, dtype=torch.bfloat16, device=device
    ) - 0.5
    out = torch.empty_like(q)
    position = 127
    table = build_interleaved_rope_table(512, device)
    tile_m = 64
    num_sms = HIDDEN // tile_m

    launcher = Launcher(num_sms, device=device)
    table_tma = TmaTensor(launcher, table)._build(
        "load", tile_m, BATCH, tma_load_tbl, lambda mat, rank: (
            lambda half, batch_seq: [0, 0, half, batch_seq]
        )
    )
    q_tma = TmaTensor(launcher, q).wgmma_load(BATCH, tile_m, Major.MN)
    out_tma = TmaTensor(launcher, out).wgmma_store(BATCH, tile_m, Major.MN)
    schedule = SchedRope(
        ROPE_INTERLEAVE_512,
        tmas=(
            ToRopeTableCordAdapter(table_tma, position),
            ToSplitMCordAdapter(q_tma, num_sms, tile_m),
            ToSplitMCordAdapter(out_tma, num_sms, tile_m),
        ),
    ).place(num_sms)
    launcher.s(schedule)
    launcher.launch()
    torch.cuda.synchronize()

    q_heads = q.view(BATCH, -1, HEAD_DIM).float()
    even, odd = q_heads[..., 0::2], q_heads[..., 1::2]
    cos = table[position, 0, 0::2].float()
    sin = table[position, 0, 1::2].float()
    expected = torch.empty_like(q_heads)
    expected[..., 0::2] = even * cos - odd * sin
    expected[..., 1::2] = even * sin + odd * cos
    rope_error = relative_error(out, expected.view_as(out).to(torch.bfloat16))
    benchmark_launcher(
        launcher,
        op="rope_q_interleaved",
        shape=f"{BATCH}x32x{HEAD_DIM}",
        iterations=args.iterations,
        warmup=args.warmup,
        error_percent=rope_error,
    )


def benchmark_argmax(args: argparse.Namespace, device: torch.device) -> None:
    generator = torch.Generator(device=device).manual_seed(14)
    num_sms = 128
    logits_slice = 65536
    logits = [
        torch.rand(
            (BATCH, logits_slice),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        for _ in range(VOCAB_PADDED // logits_slice)
    ]
    partial_values = torch.empty(
        (BATCH, num_sms), dtype=torch.bfloat16, device=device
    )
    partial_indices = torch.empty(
        (BATCH, num_sms), dtype=torch.long, device=device
    )
    output = torch.empty((BATCH,), dtype=torch.long, device=device)

    launcher = Launcher(num_sms, device=device)
    schedule = SchedArgmax(
        num_token=BATCH,
        logits_slice=logits_slice,
        num_slice=len(logits),
        AtomPartial=ARGMAX_PARTIAL_bf16_1024_65536_128,
        AtomReduce=ARGMAX_REDUCE_bf16_1024_128,
        matLogits=logits,
        matOutVal=partial_values,
        matOutIdx=partial_indices,
        matFinalOut=output,
    ).place(num_sms)
    launcher.s(schedule)
    launcher.launch()
    torch.cuda.synchronize()
    expected = torch.argmax(torch.cat(logits, dim=-1), dim=-1)
    actual_values = torch.gather(torch.cat(logits, dim=-1), 1, output[:, None])
    expected_values = torch.gather(torch.cat(logits, dim=-1), 1, expected[:, None])
    benchmark_launcher(
        launcher,
        op="greedy_argmax",
        shape=f"{BATCH}x{VOCAB_PADDED}",
        iterations=args.iterations,
        warmup=args.warmup,
        error_percent=relative_error(actual_values, expected_values),
    )


def main() -> None:
    global BATCH
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ops", default="rms,silu,rope,argmax", help="comma-separated task names"
    )
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--batches", default="8", help="comma-separated decode batch sizes"
    )
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmup < 0:
        raise ValueError("iterations must be positive and warmup non-negative")

    requested = [name for name in args.ops.split(",") if name]
    runners = {
        "rms": benchmark_rms,
        "silu": benchmark_silu,
        "silu_shard3": benchmark_silu_sharded,
        "rope": benchmark_rope,
        "argmax": benchmark_argmax,
    }
    unknown = sorted(set(requested) - set(runners))
    if unknown:
        raise ValueError(f"unknown ops: {unknown}")
    batches = [int(value) for value in args.batches.split(",") if value]
    if not batches or any(batch <= 0 for batch in batches):
        raise ValueError("--batches must contain positive integers")

    device = torch.device("cuda")
    print(
        "VDCORES_TASK_CONFIG "
        f"device={torch.cuda.get_device_name(device)} capability={torch.cuda.get_device_capability(device)} "
        f"torch={torch.__version__}",
        flush=True,
    )
    for BATCH in batches:
        print(f"VDCORES_TASK_BATCH batch={BATCH}", flush=True)
        for name in requested:
            runners[name](args, device)


if __name__ == "__main__":
    main()
