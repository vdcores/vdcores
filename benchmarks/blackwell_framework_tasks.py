#!/usr/bin/env python3
"""Benchmark the vLLM/SGLang operators used by BF16 Llama decode on SM100.

Run this file once from each framework environment.  It deliberately imports
the framework's own operator entry points instead of substituting generic
PyTorch implementations, except for operations whose framework implementation
is itself PyTorch (unquantized linear and greedy argmax).
"""

from __future__ import annotations

import argparse
import gc
import math
import statistics
from collections.abc import Callable

import torch
import torch.nn.functional as F


HIDDEN = 4096
INTERMEDIATE = 14336
MLP_PREFIX = 6144
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
VOCAB_PADDED = 131072
MAX_CONTEXT_LEN = 131072
EPS = 1.0e-5


def parse_ints(raw: str) -> list[int]:
    return [int(value) for value in raw.split(",") if value]


def parse_attention_cases(raw: str) -> list[tuple[int, int]]:
    cases = []
    for item in raw.split(","):
        batch, seq_len = item.split(":", 1)
        cases.append((int(batch), int(seq_len)))
    return cases


@torch.inference_mode()
def benchmark_cuda_graph(
    fn: Callable[[], object],
    *,
    samples: int,
    replays: int,
    graph_inner: int,
) -> tuple[float, float, float]:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_inner):
            fn()

    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    timings_us = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        timings_us.append(start.elapsed_time(end) * 1.0e3 / (replays * graph_inner))

    return min(timings_us), statistics.median(timings_us), max(timings_us)


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = expected.abs().float().mean().clamp_min(1.0e-8)
    return ((actual - expected).abs().float().mean() / denominator * 100.0).item()


def emit_result(
    op: str,
    shape: str,
    fn: Callable[[], object],
    args: argparse.Namespace,
    *,
    error_percent: float | None = None,
) -> None:
    if error_percent is not None and error_percent > 1.0:
        raise AssertionError(
            f"{op} {shape} mean-relative error {error_percent:.6f}% exceeds 1%"
        )
    minimum, median, maximum = benchmark_cuda_graph(
        fn,
        samples=args.samples,
        replays=args.replays,
        graph_inner=args.graph_inner,
    )
    suffix = "" if error_percent is None else f" error_percent={error_percent:.6f}"
    print(
        "FRAMEWORK_TASK_RESULT "
        f"framework={args.framework} op={op} shape={shape} "
        f"min_us={minimum:.6f} median_us={median:.6f} max_us={maximum:.6f}"
        f"{suffix}",
        flush=True,
    )


class FrameworkOps:
    def __init__(self, framework: str):
        self.framework = framework
        self.page_size = 16 if framework == "vllm" else 64
        self.attention_max_seq = None if framework == "vllm" else MAX_CONTEXT_LEN

        if framework == "vllm":
            import flashinfer
            import vllm
            import vllm._custom_ops as vllm_ops

            self.framework_version = vllm.__version__
            self.flashinfer_version = flashinfer.__version__
            self.vllm_ops = vllm_ops
            self.workspace_bytes = 413_138_944
        else:
            import flashinfer
            import sglang
            import sgl_kernel
            from sglang.jit_kernel.kvcache import store_cache
            from sglang.jit_kernel.rope import apply_rope_with_cos_sin_cache_inplace
            from sglang.srt.environ import envs

            self.framework_version = sglang.__version__
            self.flashinfer_version = flashinfer.__version__
            self.sgl_kernel = sgl_kernel
            self.sgl_store_cache = store_cache
            self.sgl_rope = apply_rope_with_cos_sin_cache_inplace
            self.sgl_skip_softmax_scale = (
                envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get()
            )
            self.workspace_bytes = 512 * 1024 * 1024

    def rms_norm(
        self, out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor
    ) -> Callable[[], object]:
        if self.framework == "vllm":
            return lambda: self.vllm_ops.rms_norm(out, x, weight, EPS)
        return lambda: self.sgl_kernel.rmsnorm(x, weight, EPS, out=out)

    def fused_add_rms_norm(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
    ) -> Callable[[], object]:
        if self.framework == "vllm":
            return lambda: self.vllm_ops.fused_add_rms_norm(x, residual, weight, EPS)
        return lambda: self.sgl_kernel.fused_add_rmsnorm(x, residual, weight, EPS)

    def silu_and_mul(
        self, out: torch.Tensor, gate_up: torch.Tensor
    ) -> Callable[[], object]:
        if self.framework == "vllm":
            return lambda: torch.ops._C.silu_and_mul(out, gate_up)
        return lambda: self.sgl_kernel.silu_and_mul(gate_up, out=out)

    def rope(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        cos_sin_cache: torch.Tensor,
    ) -> Callable[[], object]:
        if self.framework == "vllm":
            return lambda: self.vllm_ops.rotary_embedding(
                positions, q, k, HEAD_DIM, cos_sin_cache, True
            )
        return lambda: self.sgl_rope(
            q.view(-1, NUM_Q_HEADS, HEAD_DIM),
            k.view(-1, NUM_KV_HEADS, HEAD_DIM),
            cos_sin_cache,
            positions,
            is_neox=True,
        )


def build_cos_sin_cache(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    inv_freq = 1.0 / (
        500000.0
        ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32, device=device) / HEAD_DIM)
    )
    positions = torch.arange(4096, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype)


def reference_neox_rope(
    x: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor
) -> torch.Tensor:
    heads = x.view(x.shape[0], -1, HEAD_DIM).float()
    cache = cos_sin_cache[positions].float().view(x.shape[0], 1, HEAD_DIM)
    first, second = heads[..., : HEAD_DIM // 2], heads[..., HEAD_DIM // 2 :]
    cos, sin = cache[..., : HEAD_DIM // 2], cache[..., HEAD_DIM // 2 :]
    return torch.cat(
        (first * cos - second * sin, second * cos + first * sin), dim=-1
    ).view_as(x).to(torch.bfloat16)


def benchmark_core(ops: FrameworkOps, args: argparse.Namespace) -> None:
    device = torch.device("cuda")
    for batch in args.batches:
        generator = torch.Generator(device=device).manual_seed(1000 + batch)

        x = torch.rand(
            (batch, HIDDEN), generator=generator, dtype=torch.bfloat16, device=device
        ) - 0.5
        weight = torch.rand(
            (HIDDEN,), generator=generator, dtype=torch.bfloat16, device=device
        ) + 0.5
        out = torch.empty_like(x)
        rms_fn = ops.rms_norm(out, x, weight)
        rms_fn()
        rms_expected = (
            x.float()
            * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + EPS)
            * weight.float()
        ).to(torch.bfloat16)
        emit_result(
            "rms_norm",
            f"{batch}x{HIDDEN}",
            rms_fn,
            args,
            error_percent=relative_error(out, rms_expected),
        )

        fused_x = x.clone()
        fused_x_input = fused_x.clone()
        residual = torch.rand(
            (batch, HIDDEN), generator=generator, dtype=torch.bfloat16, device=device
        ) - 0.5
        residual_input = residual.clone()
        fused_fn = ops.fused_add_rms_norm(fused_x, residual, weight)
        fused_fn()
        fused_sum = fused_x_input.float() + residual_input.float()
        fused_expected = (
            fused_sum
            * torch.rsqrt(fused_sum.square().mean(dim=-1, keepdim=True) + EPS)
            * weight.float()
        ).to(torch.bfloat16)
        fused_error = max(
            relative_error(fused_x, fused_expected),
            relative_error(residual, fused_sum.to(torch.bfloat16)),
        )
        emit_result(
            "fused_add_rms_norm",
            f"{batch}x{HIDDEN}",
            fused_fn,
            args,
            error_percent=fused_error,
        )

        silu_tensors = []
        for width in (MLP_PREFIX, INTERMEDIATE):
            gate_up = torch.rand(
                (batch, 2 * width),
                generator=generator,
                dtype=torch.bfloat16,
                device=device,
            ) - 0.5
            silu_out = torch.empty(
                (batch, width), dtype=torch.bfloat16, device=device
            )
            silu_fn = ops.silu_and_mul(silu_out, gate_up)
            silu_fn()
            silu_expected = F.silu(gate_up[:, :width].float()) * gate_up[
                :, width:
            ].float()
            emit_result(
                "silu_and_mul",
                f"{batch}x{2 * width}",
                silu_fn,
                args,
                error_percent=relative_error(
                    silu_out, silu_expected.to(torch.bfloat16)
                ),
            )
            silu_tensors.extend((gate_up, silu_out, silu_expected))

        cache_dtype = torch.bfloat16 if args.framework == "vllm" else torch.float32
        cos_sin_cache = build_cos_sin_cache(cache_dtype, device)
        positions = torch.arange(100, 100 + batch, dtype=torch.long, device=device)
        q = torch.rand(
            (batch, NUM_Q_HEADS * HEAD_DIM),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        ) - 0.5
        k = torch.rand(
            (batch, NUM_KV_HEADS * HEAD_DIM),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        ) - 0.5
        q_expected = reference_neox_rope(q, positions, cos_sin_cache)
        k_expected = reference_neox_rope(k, positions, cos_sin_cache)
        rope_fn = ops.rope(positions, q, k, cos_sin_cache)
        rope_fn()
        rope_error = max(
            relative_error(q, q_expected), relative_error(k, k_expected)
        )
        emit_result(
            "rope_qk",
            f"{batch}xq{NUM_Q_HEADS}xkv{NUM_KV_HEADS}x{HEAD_DIM}",
            rope_fn,
            args,
            error_percent=rope_error,
        )

        cache_rows = 4096
        new_k = torch.rand(
            (batch, NUM_KV_HEADS, HEAD_DIM),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        ) - 0.5
        new_v = torch.rand_like(new_k)
        locations = torch.arange(batch, dtype=torch.long, device=device) * 7 + 3
        if args.framework == "vllm":
            num_pages = math.ceil(cache_rows / ops.page_size)
            physical = torch.empty(
                (num_pages, 2, NUM_KV_HEADS, ops.page_size, HEAD_DIM),
                dtype=torch.bfloat16,
                device=device,
            )
            logical = physical.permute(0, 1, 3, 2, 4)
            k_cache, v_cache = logical[:, 0], logical[:, 1]
            scale = torch.ones(1, dtype=torch.float32, device=device)

            def kv_store_fn():
                return torch.ops._C_cache_ops.reshape_and_cache_flash(
                    new_k,
                    new_v,
                    k_cache,
                    v_cache,
                    locations,
                    "auto",
                    scale,
                    scale,
                )

        else:
            k_cache = torch.empty(
                (cache_rows, NUM_KV_HEADS, HEAD_DIM),
                dtype=torch.bfloat16,
                device=device,
            )
            v_cache = torch.empty_like(k_cache)

            def kv_store_fn():
                return ops.sgl_store_cache(
                    new_k.view(batch, -1),
                    new_v.view(batch, -1),
                    k_cache.view(cache_rows, -1),
                    v_cache.view(cache_rows, -1),
                    locations,
                    row_bytes=NUM_KV_HEADS * HEAD_DIM * 2,
                )

        kv_store_fn()
        torch.cuda.synchronize()
        if args.framework == "vllm":
            cache_pages = torch.div(locations, ops.page_size, rounding_mode="floor")
            cache_offsets = locations % ops.page_size
            stored_k = k_cache[cache_pages, cache_offsets]
            stored_v = v_cache[cache_pages, cache_offsets]
        else:
            stored_k = k_cache[locations]
            stored_v = v_cache[locations]
        kv_error = max(
            relative_error(stored_k, new_k), relative_error(stored_v, new_v)
        )
        emit_result(
            "kv_cache_store",
            f"{batch}x2x{NUM_KV_HEADS}x{HEAD_DIM}",
            kv_store_fn,
            args,
            error_percent=kv_error,
        )

        logits = torch.rand(
            (batch, VOCAB_PADDED),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        emit_result(
            "greedy_argmax",
            f"{batch}x{VOCAB_PADDED}",
            lambda: torch.argmax(logits, dim=-1),
            args,
        )

        del (
            x,
            weight,
            out,
            fused_x,
            fused_x_input,
            residual,
            residual_input,
            fused_expected,
            cos_sin_cache,
            q,
            k,
            q_expected,
            k_expected,
            new_k,
            new_v,
            k_cache,
            v_cache,
            stored_k,
            stored_v,
            logits,
        )
        del silu_tensors
        gc.collect()
        torch.cuda.empty_cache()


def benchmark_linears(args: argparse.Namespace) -> None:
    device = torch.device("cuda")
    shapes = [
        ("kv_component", 1024, HIDDEN),
        ("q_or_o_component", HIDDEN, HIDDEN),
        ("qkv_fused", HIDDEN + 2 * 1024, HIDDEN),
        ("gate_or_up_component", INTERMEDIATE, HIDDEN),
        ("gate_up_fused", 2 * INTERMEDIATE, HIDDEN),
        ("down", HIDDEN, INTERMEDIATE),
        ("lm_head_padded", VOCAB_PADDED, HIDDEN),
    ]
    for name, out_features, in_features in shapes:
        weight = torch.empty(
            (out_features, in_features), dtype=torch.bfloat16, device=device
        )
        weight.fill_(0.01)
        for batch in args.batches:
            x = torch.empty(
                (batch, in_features), dtype=torch.bfloat16, device=device
            )
            x.fill_(0.01)
            emit_result(
                f"linear_{name}",
                f"{out_features}x{batch}x{in_features}",
                lambda x=x, weight=weight: F.linear(x, weight),
                args,
            )
            del x
        del weight
        gc.collect()
        torch.cuda.empty_cache()


def reference_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    batch, seq_len, _, _ = k.shape
    q_grouped = q.view(
        batch, NUM_KV_HEADS, NUM_Q_HEADS // NUM_KV_HEADS, HEAD_DIM
    ).float()
    scores = torch.einsum("bhgd,bshd->bhgs", q_grouped, k.float()) / math.sqrt(
        HEAD_DIM
    )
    probs = torch.softmax(scores, dim=-1)
    output = torch.einsum("bhgs,bshd->bhgd", probs, v.float())
    return output.reshape(batch, NUM_Q_HEADS, HEAD_DIM).to(torch.bfloat16)


def benchmark_attention(ops: FrameworkOps, args: argparse.Namespace) -> None:
    import flashinfer
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache

    device = torch.device("cuda")
    workspace = torch.zeros(ops.workspace_bytes, dtype=torch.uint8, device=device)
    for batch, seq_len in args.attention_cases:
        generator = torch.Generator(device=device).manual_seed(batch * 100000 + seq_len)
        q = torch.rand(
            (batch, NUM_Q_HEADS, HEAD_DIM),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        ) - 0.5
        k = torch.rand(
            (batch, seq_len, NUM_KV_HEADS, HEAD_DIM),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        ) - 0.5
        v = torch.rand_like(k)

        pages_per_request = math.ceil(seq_len / ops.page_size)
        padded_seq_len = pages_per_request * ops.page_size
        k_padded = torch.zeros(
            (batch, padded_seq_len, NUM_KV_HEADS, HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
        )
        v_padded = torch.zeros_like(k_padded)
        k_padded[:, :seq_len] = k
        v_padded[:, :seq_len] = v
        k_cache = (
            k_padded.view(-1, ops.page_size, NUM_KV_HEADS, HEAD_DIM)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        v_cache = (
            v_padded.view(-1, ops.page_size, NUM_KV_HEADS, HEAD_DIM)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        block_tables = torch.arange(
            batch * pages_per_request, dtype=torch.int32, device=device
        ).view(batch, pages_per_request)
        seq_lens = torch.full(
            (batch,), seq_len, dtype=torch.int32, device=device
        )
        attn_out = torch.empty_like(q)
        max_seq_len = seq_len if ops.attention_max_seq is None else ops.attention_max_seq

        common = dict(
            query=q,
            kv_cache=(k_cache, v_cache),
            workspace_buffer=workspace,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            bmm1_scale=1.0 / math.sqrt(HEAD_DIM),
            bmm2_scale=1.0,
            window_left=-1,
            q_len_per_req=1,
        )
        if args.framework == "vllm":
            fn = lambda: trtllm_batch_decode_with_kv_cache(**common, out=attn_out)
        else:
            fn = lambda: trtllm_batch_decode_with_kv_cache(
                **common,
                out_dtype=torch.bfloat16,
                skip_softmax_threshold_scale_factor=ops.sgl_skip_softmax_scale,
            )

        actual = fn()
        torch.cuda.synchronize()
        expected = reference_attention(q, k, v)
        emit_result(
            "decode_attention",
            f"b{batch}_s{seq_len}_p{ops.page_size}",
            fn,
            args,
            error_percent=relative_error(actual, expected),
        )
        del (
            q,
            k,
            v,
            k_padded,
            v_padded,
            k_cache,
            v_cache,
            block_tables,
            seq_lens,
            attn_out,
            actual,
            expected,
        )
        gc.collect()
        torch.cuda.empty_cache()

    print(
        "FRAMEWORK_ATTENTION_BACKEND "
        f"framework={args.framework} api=flashinfer.decode.trtllm_batch_decode_with_kv_cache "
        f"flashinfer={flashinfer.__version__} page_size={ops.page_size} "
        f"max_seq_policy={'actual' if ops.attention_max_seq is None else ops.attention_max_seq}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=("vllm", "sglang"), required=True)
    parser.add_argument(
        "--suite", choices=("all", "core", "linears", "attention"), default="all"
    )
    parser.add_argument("--batches", type=parse_ints, default=parse_ints("1,8"))
    parser.add_argument(
        "--attention-cases",
        type=parse_attention_cases,
        default=parse_attention_cases(
            "1:64,1:128,1:512,1:2048,2:128,2:512,4:128,4:512,8:128,8:512"
        ),
    )
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--graph-inner", type=int, default=10)
    args = parser.parse_args()
    if min(args.samples, args.replays, args.graph_inner) <= 0:
        raise ValueError("samples, replays, and graph-inner must be positive")

    ops = FrameworkOps(args.framework)
    device = torch.device("cuda")
    print(
        "FRAMEWORK_TASK_CONFIG "
        f"framework={args.framework} version={ops.framework_version} "
        f"flashinfer={ops.flashinfer_version} torch={torch.__version__} "
        f"device={torch.cuda.get_device_name(device)} capability={torch.cuda.get_device_capability(device)} "
        f"page_size={ops.page_size}",
        flush=True,
    )

    if args.suite in ("all", "core"):
        benchmark_core(ops, args)
    if args.suite in ("all", "linears"):
        benchmark_linears(args)
    if args.suite in ("all", "attention"):
        benchmark_attention(ops, args)


if __name__ == "__main__":
    main()
