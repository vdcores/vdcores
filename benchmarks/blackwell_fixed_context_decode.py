#!/usr/bin/env python3
"""Measure launch-inclusive framework decode at a fixed KV context length.

Each invocation configures one engine for one context length.  Every request
has ``context - 1`` prompt tokens and produces exactly two output tokens.  The
interval between the framework's first- and second-token engine timestamps is
therefore one decode step whose attention sees ``context`` KV tokens.  Prefill
and the first-token latency are not part of the reported interval.
"""

from __future__ import annotations

import argparse
import os
import statistics


DEFAULT_TOKEN_ID = 791


def parse_ints(raw: str) -> list[int]:
    values = [int(value) for value in raw.split(",") if value]
    if not values or any(value < 2 for value in values):
        raise argparse.ArgumentTypeError("contexts must be comma-separated integers >= 2")
    return values


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round(fraction * (len(ordered) - 1))
    return ordered[index]


def emit_result(
    framework: str,
    version: str,
    context: int,
    batch: int,
    samples_ms: list[float],
) -> None:
    print(
        "FIXED_CONTEXT_RESULT "
        f"framework={framework} version={version} batch={batch} "
        f"context={context} samples={len(samples_ms)} "
        f"min_ms={min(samples_ms):.6f} "
        f"median_ms={statistics.median(samples_ms):.6f} "
        f"p90_ms={percentile(samples_ms, 0.90):.6f} "
        f"max_ms={max(samples_ms):.6f}",
        flush=True,
    )


def run_vllm(args: argparse.Namespace) -> None:
    import vllm
    from vllm import LLM, SamplingParams

    profiler_config = None
    if args.profile_context is not None:
        profiler_config = {"profiler": args.profile_kind}
        if args.profile_kind == "torch":
            profiler_config.update(
                torch_profiler_dir=args.profile_dir,
                torch_profiler_with_stack=False,
                torch_profiler_use_gzip=False,
            )

    context = args.contexts[0]
    engine_max_model_len = context + 1
    # The measured first->second-token interval is decode-only only if every
    # request completes prefill before the first token is emitted.  Size the
    # scheduler's token budget for the complete strict batch; otherwise vLLM
    # can interleave residual chunked prefill with the nominal decode interval
    # once batch * context exceeds its default 16K-token budget.
    engine_max_num_batched_tokens = args.batch * (context - 1)
    print(
        "FIXED_CONTEXT_CONFIG "
        f"framework=vllm batch={args.batch} context={context} "
        f"engine_max_model_len={engine_max_model_len} dtype=bfloat16 "
        f"kv_cache_dtype={args.kv_cache_dtype} "
        f"max_num_batched_tokens={engine_max_num_batched_tokens} "
        "strict_batch=1 unchunked_strict_prefill=1",
        flush=True,
    )
    engine = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        kv_cache_dtype=args.kv_cache_dtype,
        max_model_len=engine_max_model_len,
        max_num_seqs=args.batch,
        max_num_batched_tokens=engine_max_num_batched_tokens,
        enable_prefix_caching=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        skip_tokenizer_init=True,
        trust_remote_code=False,
        disable_log_stats=False,
        profiler_config=profiler_config,
    )
    sampling = SamplingParams(
        temperature=0.0,
        ignore_eos=True,
        max_tokens=2,
        detokenize=False,
    )

    def enqueue_full_batch(prompts):
        # With vLLM's default async scheduling, LLM.generate() can start the
        # engine while it is still enqueueing a Python list of requests.  Pause
        # scheduling at level 0 so every measured decode step is truly the
        # requested batch size, while keeping async scheduling enabled for the
        # execution itself.
        engine.sleep(level=0, mode="wait")
        try:
            engine.enqueue(prompts, sampling, use_tqdm=False)
        except BaseException:
            engine.wake_up(tags=["scheduling"])
            raise

    def generate_full_batch(prompts):
        enqueue_full_batch(prompts)
        engine.wake_up(tags=["scheduling"])
        return engine.wait_for_completion(use_tqdm=False)

    for context in args.contexts:
        # The second generated token consumes the first generated token at
        # position context - 1, so its attention length is exactly context.
        prompts = [
            {"prompt_token_ids": [args.token_id] * (context - 1)}
            for _ in range(args.batch)
        ]
        for _ in range(args.warmups):
            generate_full_batch(prompts)

        samples_ms = []
        for _ in range(args.samples):
            outputs = generate_full_batch(prompts)
            request_ms = []
            for output in outputs:
                metrics = output.metrics
                if metrics is None or metrics.first_token_ts <= 0:
                    raise RuntimeError("vLLM did not return engine token timestamps")
                if metrics.num_generation_tokens != 2:
                    raise RuntimeError(
                        "expected two generated tokens, got "
                        f"{metrics.num_generation_tokens}"
                    )
                request_ms.append(
                    (metrics.last_token_ts - metrics.first_token_ts) * 1.0e3
                )
            # Requests in one decode batch normally share a timestamp.  Taking
            # the maximum retains the full batch step if completion handling
            # introduces a small skew.
            samples_ms.append(max(request_ms))

        emit_result("vllm", vllm.__version__, context, args.batch, samples_ms)

        if context == args.profile_context:
            # Async scheduling can dispatch the next GPU graph before the
            # frontend receives the prior step's output, so starting a profiler
            # between calls to llm_engine.step() is too late.  Bracket request
            # submission instead; the resulting trace has one prefill range and
            # one decode range, whose second-token attention length is exactly
            # ``context``.
            enqueue_full_batch(prompts)
            try:
                engine.start_profile(f"fixed_context_{context}")
                engine.wake_up(tags=["scheduling"])
                engine.wait_for_completion(use_tqdm=False)
                engine.stop_profile()
            except BaseException:
                engine.wake_up(tags=["scheduling"])
                raise
            print(
                "FIXED_CONTEXT_PROFILE "
                f"framework=vllm context={context} kind={args.profile_kind} "
                f"scope=prefill_plus_one_decode directory={args.profile_dir or '-'}",
                flush=True,
            )


def run_sglang(args: argparse.Namespace) -> None:
    import sglang

    context = args.contexts[0]
    engine_context_length = context + 16
    engine_max_prefill_tokens = args.batch * (context - 1)
    print(
        "FIXED_CONTEXT_CONFIG "
        f"framework=sglang batch={args.batch} context={context} "
        f"engine_context_length={engine_context_length} dtype=bfloat16 "
        f"max_prefill_tokens={engine_max_prefill_tokens} "
        "strict_batch=1 unchunked_strict_prefill=1",
        flush=True,
    )
    engine = sglang.Engine(
        model_path=args.model,
        dtype="bfloat16",
        # SGLang keeps a small output/scheduler guard below context_length.
        # This capacity margin does not change the per-request KV length.
        context_length=engine_context_length,
        max_running_requests=args.batch,
        mem_fraction_static=args.gpu_memory_utilization,
        # A chunked prefill can overlap the first request's nominal decode
        # interval with the remaining requests' prefill.  Keep every request
        # whole so the first->second-token metric is a strict decode batch.
        chunked_prefill_size=-1,
        max_prefill_tokens=engine_max_prefill_tokens,
        disable_radix_cache=True,
        cuda_graph_max_bs=args.batch,
        disable_piecewise_cuda_graph=True,
        attention_backend="flashinfer",
        page_size=64,
        skip_tokenizer_init=True,
        enable_metrics=True,
        trust_remote_code=False,
        log_level="error",
    )
    sampling = {
        "temperature": 0.0,
        "ignore_eos": True,
        "max_new_tokens": 2,
    }

    def generate_once(input_ids: list[list[int]]) -> list[dict]:
        final_outputs = {}
        for output in engine.generate(
            input_ids=input_ids,
            sampling_params=sampling,
            stream=True,
        ):
            meta = output["meta_info"]
            if meta.get("completion_tokens") == 2:
                final_outputs[output["index"]] = output
        if len(final_outputs) != args.batch:
            raise RuntimeError(
                f"expected {args.batch} final SGLang outputs, got "
                f"{len(final_outputs)}"
            )
        return [final_outputs[index] for index in range(args.batch)]

    try:
        for context in args.contexts:
            input_ids = [
                [args.token_id] * (context - 1) for _ in range(args.batch)
            ]
            for _ in range(args.warmups):
                generate_once(input_ids)

            samples_ms = []
            for _ in range(args.samples):
                outputs = generate_once(input_ids)
                request_ms = []
                for output in outputs:
                    meta = output["meta_info"]
                    if meta.get("completion_tokens") != 2:
                        raise RuntimeError(
                            "expected two generated tokens, got "
                            f"{meta.get('completion_tokens')}"
                        )
                    decode_throughput = meta.get("decode_throughput")
                    if decode_throughput is None or decode_throughput <= 0:
                        raise RuntimeError(
                            "SGLang did not return a positive decode throughput: "
                            f"meta_info={meta!r}"
                        )
                    # With two output tokens, decode_throughput is exactly the
                    # reciprocal of the one-token decode interval.
                    request_ms.append(1.0e3 / decode_throughput)
                samples_ms.append(max(request_ms))

            emit_result(
                "sglang", sglang.__version__, context, args.batch, samples_ms
            )
    finally:
        engine.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=("vllm", "sglang"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--contexts", type=parse_ints, default=parse_ints("128"))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--token-id", type=int, default=DEFAULT_TOKEN_ID)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--profile-context",
        type=int,
        help="vLLM context to capture as a one-step torch-profiler trace",
    )
    parser.add_argument(
        "--profile-dir",
        help="absolute worker-local output directory for --profile-context",
    )
    parser.add_argument(
        "--profile-kind",
        choices=("torch", "cuda"),
        default="torch",
        help="profiler used for the selected vLLM decode step",
    )
    args = parser.parse_args()

    if args.batch <= 0 or args.warmups < 0 or args.samples <= 0:
        parser.error("batch and samples must be positive; warmups must be non-negative")
    if len(args.contexts) != 1:
        parser.error(
            "fixed-context results require exactly one --contexts value per process "
            "so engine capacity is identical to the measured row"
        )
    if args.profile_context is not None:
        if args.framework != "vllm":
            parser.error("--profile-context currently supports only vLLM")
        if args.profile_context not in args.contexts:
            parser.error("--profile-context must be present in --contexts")
        if args.profile_kind == "torch" and (
            not args.profile_dir or not os.path.isabs(args.profile_dir)
        ):
            parser.error(
                "--profile-dir must be an absolute worker-local path for torch profiling"
            )
        if args.profile_kind == "cuda" and args.profile_dir:
            parser.error("--profile-dir is not used with --profile-kind cuda")
    elif args.profile_dir:
        parser.error("--profile-dir requires --profile-context")

    if args.framework == "vllm":
        run_vllm(args)
    else:
        run_sglang(args)


if __name__ == "__main__":
    main()
