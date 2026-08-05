#!/usr/bin/env python3
"""Measure launch-inclusive framework decode at a fixed KV context length.

Each request has ``context - 1`` prompt tokens and produces exactly two output
tokens.  The interval between the framework's first- and second-token engine
timestamps is therefore one decode step whose attention sees ``context`` KV
tokens.  Prefill and the first-token latency are not part of the reported
interval.
"""

from __future__ import annotations

import argparse
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

    engine = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        max_model_len=max(args.contexts) + 1,
        max_num_seqs=args.batch,
        enable_prefix_caching=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        skip_tokenizer_init=True,
        trust_remote_code=False,
        disable_log_stats=False,
    )
    sampling = SamplingParams(
        temperature=0.0,
        ignore_eos=True,
        max_tokens=2,
        detokenize=False,
    )

    for context in args.contexts:
        # The second generated token consumes the first generated token at
        # position context - 1, so its attention length is exactly context.
        prompts = [
            {"prompt_token_ids": [args.token_id] * (context - 1)}
            for _ in range(args.batch)
        ]
        for _ in range(args.warmups):
            engine.generate(prompts, sampling, use_tqdm=False)

        samples_ms = []
        for _ in range(args.samples):
            outputs = engine.generate(prompts, sampling, use_tqdm=False)
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


def run_sglang(args: argparse.Namespace) -> None:
    import sglang

    engine = sglang.Engine(
        model_path=args.model,
        dtype="bfloat16",
        # SGLang keeps a small output/scheduler guard below context_length.
        # This capacity margin does not change the per-request KV length.
        context_length=max(args.contexts) + 16,
        max_running_requests=args.batch,
        mem_fraction_static=args.gpu_memory_utilization,
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
    parser.add_argument("--contexts", type=parse_ints, default=parse_ints("64,128,512"))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--token-id", type=int, default=DEFAULT_TOKEN_ID)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    if args.batch <= 0 or args.warmups < 0 or args.samples <= 0:
        parser.error("batch and samples must be positive; warmups must be non-negative")

    if args.framework == "vllm":
        run_vllm(args)
    else:
        run_sglang(args)


if __name__ == "__main__":
    main()
