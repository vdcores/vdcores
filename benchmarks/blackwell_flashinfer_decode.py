#!/usr/bin/env python3
"""Benchmark FlashInfer decode-attention backends on the VDCores Llama shape."""

from __future__ import annotations

import argparse
from math import sqrt

import numpy as np
import torch

import flashinfer
from flashinfer.testing import bench_gpu_time


NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
PAGE_SIZE = 64


def parse_cases(raw: str) -> list[tuple[int, int]]:
    cases = []
    for item in raw.split(","):
        batch, seq_len = item.split(":", 1)
        cases.append((int(batch), int(seq_len)))
    return cases


def reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    batch, seq_len, _, _ = k.shape
    group_size = NUM_Q_HEADS // NUM_KV_HEADS
    q_grouped = q.view(batch, NUM_KV_HEADS, group_size, HEAD_DIM).float()
    scores = torch.einsum("bhgd,bshd->bhgs", q_grouped, k.float()) / sqrt(HEAD_DIM)
    probs = torch.softmax(scores, dim=-1)
    output = torch.einsum("bhgs,bshd->bhgd", probs, v.float())
    return output.reshape(batch, NUM_Q_HEADS, HEAD_DIM).to(q.dtype)


def build_case(batch: int, seq_len: int, device: torch.device):
    pages_per_request = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = batch * pages_per_request
    padded_seq_len = pages_per_request * PAGE_SIZE
    generator = torch.Generator(device=device).manual_seed(batch * 100_000 + seq_len)

    q = torch.rand(
        (batch, NUM_Q_HEADS, HEAD_DIM), generator=generator,
        dtype=torch.bfloat16, device=device,
    ) - 0.5
    k = torch.rand(
        (batch, seq_len, NUM_KV_HEADS, HEAD_DIM), generator=generator,
        dtype=torch.bfloat16, device=device,
    ) - 0.5
    v = torch.rand(
        (batch, seq_len, NUM_KV_HEADS, HEAD_DIM), generator=generator,
        dtype=torch.bfloat16, device=device,
    ) - 0.5

    k_padded = torch.zeros(
        (batch, padded_seq_len, NUM_KV_HEADS, HEAD_DIM),
        dtype=torch.bfloat16, device=device,
    )
    v_padded = torch.zeros_like(k_padded)
    k_padded[:, :seq_len] = k
    v_padded[:, :seq_len] = v
    kv_cache = torch.stack(
        (
            k_padded.view(total_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM),
            v_padded.view(total_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM),
        ),
        dim=1,
    )

    indptr = torch.arange(
        0, total_pages + 1, pages_per_request, dtype=torch.int32, device=device,
    )
    indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    last_page_len = torch.full(
        (batch,), (seq_len - 1) % PAGE_SIZE + 1, dtype=torch.int32, device=device,
    )
    return q, k, v, kv_cache, indptr, indices, last_page_len


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases", default="1:64,1:128,1:512,1:2048,2:128,2:512,4:128,4:512",
    )
    parser.add_argument("--backends", default="auto,fa3,trtllm-gen")
    parser.add_argument("--repeat-ms", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda")
    print(
        f"FLASHINFER_VERSION {flashinfer.__version__} "
        f"device={torch.cuda.get_device_name(device)}"
    )
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    for batch, seq_len in parse_cases(args.cases):
        q, k, v, kv_cache, indptr, indices, last_page_len = build_case(
            batch, seq_len, device,
        )
        expected = reference_attention(q, k, v)
        for backend in args.backends.split(","):
            for use_tensor_cores in (False, True):
                label = f"backend={backend} tensor_cores={int(use_tensor_cores)}"
                try:
                    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                        workspace,
                        kv_layout="NHD",
                        use_tensor_cores=use_tensor_cores,
                        backend=backend,
                    )
                    wrapper.plan(
                        indptr,
                        indices,
                        last_page_len,
                        NUM_Q_HEADS,
                        NUM_KV_HEADS,
                        HEAD_DIM,
                        PAGE_SIZE,
                        pos_encoding_mode="NONE",
                        q_data_type=torch.bfloat16,
                        kv_data_type=torch.bfloat16,
                        o_data_type=torch.bfloat16,
                        sm_scale=1.0 / sqrt(HEAD_DIM),
                    )
                    output = wrapper.run(q, kv_cache)
                    torch.cuda.synchronize()
                    avg_diff_percent = (
                        (expected - output).abs().float().mean()
                        / expected.abs().float().mean()
                        * 100.0
                    ).item()
                    times_ms = bench_gpu_time(
                        lambda: wrapper.run(q, kv_cache),
                        dry_run_time_ms=10,
                        repeat_time_ms=args.repeat_ms,
                        use_cuda_graph=True,
                        num_iters_within_graph=20,
                        cold_l2_cache=False,
                    )
                    print(
                        "FLASHINFER_RESULT "
                        f"batch={batch} seq={seq_len} {label} "
                        f"median_us={np.median(times_ms) * 1000.0:.6f} "
                        f"avg_diff_percent={avg_diff_percent:.6f}"
                    )
                except Exception as exc:
                    print(
                        "FLASHINFER_SKIP "
                        f"batch={batch} seq={seq_len} {label} "
                        f"reason={type(exc).__name__}:{exc}"
                    )


if __name__ == "__main__":
    main()
