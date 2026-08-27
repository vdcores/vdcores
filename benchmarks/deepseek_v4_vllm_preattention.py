#!/usr/bin/env python3
"""Measure the optimized vLLM DeepSeek-V4 pre-attention critical path.

The timed interval starts with the already-normalized 4096-wide attention
input and ends at ``forward_mqa`` entry, after q_b/Q-RoPE, current-token KV
insertion, and any compressor/indexer work have joined. Sparse attention,
output projection, and FFN work are outside the interval.
"""

from __future__ import annotations

import argparse
import os
import statistics

# Keep the engine and the instrumented module objects in this process so the
# CUDA events captured into vLLM's breakable graph remain directly readable.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch

from vllm import LLM, SamplingParams
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.nvidia.flashmla import (
    DeepseekV4FlashMLAAttention,
)
from vllm.models.deepseek_v4.nvidia.model import DeepseekV4DecoderLayer
from vllm.models.deepseek_v4.nvidia.model import DeepseekV4ForCausalLM
from vllm.v1.sample.sampler import Sampler


def install_event_probes() -> tuple[
    dict[str, DeepseekV4Attention],
    tuple[torch.cuda.Event, torch.cuda.Event, torch.cuda.Event],
]:
    registry: dict[str, DeepseekV4Attention] = {}
    step_start = torch.cuda.Event(enable_timing=True, external=True)
    logits_end = torch.cuda.Event(enable_timing=True, external=True)
    sample_end = torch.cuda.Event(enable_timing=True, external=True)

    original_attention_init = DeepseekV4Attention.__init__

    def attention_init(self, *args, **kwargs):
        original_attention_init(self, *args, **kwargs)
        # External events remain explicit record nodes when this code is
        # captured into vLLM's breakable CUDA graphs, so replay refreshes their
        # timestamps.  Ordinary events become internal dependency nodes and
        # retain capture-time timestamps instead.
        self._preattention_start = torch.cuda.Event(
            enable_timing=True, external=True
        )
        self._preattention_end = torch.cuda.Event(
            enable_timing=True, external=True
        )
        registry[self.prefix] = self

    DeepseekV4Attention.__init__ = attention_init

    original_decoder_init = DeepseekV4DecoderLayer.__init__

    def decoder_init(self, *args, **kwargs):
        original_decoder_init(self, *args, **kwargs)
        self.attn._mhc_rms_start = torch.cuda.Event(
            enable_timing=True, external=True
        )

    DeepseekV4DecoderLayer.__init__ = decoder_init

    original_decoder_forward = DeepseekV4DecoderLayer.forward

    def decoder_forward(self, *args, **kwargs):
        self.attn._mhc_rms_start.record()
        return original_decoder_forward(self, *args, **kwargs)

    DeepseekV4DecoderLayer.forward = decoder_forward

    original_attention_forward = DeepseekV4Attention.forward

    def attention_forward(self, *args, **kwargs):
        # This record is captured into the first breakable-graph segment and
        # therefore replays immediately after fused mHC-pre + attention RMS.
        self._preattention_start.record()
        return original_attention_forward(self, *args, **kwargs)

    DeepseekV4Attention.forward = attention_forward

    original_forward_mqa = DeepseekV4FlashMLAAttention.forward_mqa

    def forward_mqa(self, *args, **kwargs):
        # execute_in_parallel has joined q/kv/compressor/indexer streams before
        # reaching this call. Record before the sparse-attention kernel fires.
        self._preattention_end.record()
        return original_forward_mqa(self, *args, **kwargs)

    DeepseekV4FlashMLAAttention.forward_mqa = forward_mqa

    original_causal_forward = DeepseekV4ForCausalLM.forward

    def causal_forward(self, *args, **kwargs):
        # The final replay in each request is the single-token decode at the
        # requested context.  External events survive graph capture/replay.
        step_start.record()
        return original_causal_forward(self, *args, **kwargs)

    DeepseekV4ForCausalLM.forward = causal_forward

    original_compute_logits = DeepseekV4ForCausalLM.compute_logits

    def compute_logits(self, *args, **kwargs):
        output = original_compute_logits(self, *args, **kwargs)
        logits_end.record()
        return output

    DeepseekV4ForCausalLM.compute_logits = compute_logits

    original_sampler_forward = Sampler.forward

    def sampler_forward(self, *args, **kwargs):
        output = original_sampler_forward(self, *args, **kwargs)
        sample_end.record()
        return output

    Sampler.forward = sampler_forward
    return registry, (step_start, logits_end, sample_end)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument(
        "--engine-max-model-len",
        type=int,
        default=256,
        help=(
            "padded KV allocation length; SM100 FlashMLA requires a "
            "TMA-aligned batch stride while the logical decode context "
            "remains --context"
        ),
    )
    parser.add_argument(
        "--target-layers",
        default="3,4",
        help="comma-separated native checkpoint layer IDs to report",
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--token-id", type=int, default=791)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument(
        "--enable-flashinfer-autotune",
        action="store_true",
        help="enable production FlashInfer kernel autotuning for end-to-end runs",
    )
    args = parser.parse_args()
    if args.context < 2:
        parser.error("context must be >=2")
    if args.engine_max_model_len < args.context + 1:
        parser.error("engine max model length must cover prefill plus decode")
    if args.warmups < 0 or args.samples <= 0:
        parser.error("warmups must be non-negative and samples positive")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        parser.error("gpu memory utilization must be in (0, 1]")
    try:
        target_layers = {int(value) for value in args.target_layers.split(",")}
    except ValueError as error:
        parser.error(f"invalid --target-layers: {error}")
    if not target_layers or min(target_layers) < 0:
        parser.error("target layers must be non-negative")

    registry, step_events = install_event_probes()
    engine = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        kv_cache_dtype="fp8",
        max_model_len=args.engine_max_model_len,
        max_num_seqs=1,
        max_num_batched_tokens=args.context - 1,
        enable_prefix_caching=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        skip_tokenizer_init=True,
        trust_remote_code=False,
        kernel_config={
            "enable_flashinfer_autotune": args.enable_flashinfer_autotune
        },
    )
    sampling = SamplingParams(
        temperature=0.0,
        ignore_eos=True,
        max_tokens=2,
        detokenize=False,
    )
    prompt = {"prompt_token_ids": [args.token_id] * (args.context - 1)}

    def run_once() -> None:
        engine.generate([prompt], sampling, use_tqdm=False)
        torch.cuda.synchronize()

    for _ in range(args.warmups):
        run_once()

    selected = {
        prefix: module
        for prefix, module in registry.items()
        if module.compress_ratio in (4, 128)
        and any(f".layers.{layer_id}." in prefix for layer_id in target_layers)
    }
    if not selected:
        raise RuntimeError("instrumentation found no HCA/CSA attention layers")
    samples: dict[str, list[float]] = {prefix: [] for prefix in selected}
    mhc_samples: dict[str, list[float]] = {prefix: [] for prefix in selected}
    model_logits_samples: list[float] = []
    model_sample_samples: list[float] = []
    for _ in range(args.samples):
        run_once()
        model_logits_samples.append(
            step_events[0].elapsed_time(step_events[1]) * 1.0e3
        )
        model_sample_samples.append(
            step_events[0].elapsed_time(step_events[2]) * 1.0e3
        )
        for prefix, module in selected.items():
            samples[prefix].append(
                module._preattention_start.elapsed_time(
                    module._preattention_end
                )
                * 1.0e3
            )
            mhc_samples[prefix].append(
                module._mhc_rms_start.elapsed_time(
                    module._preattention_start
                )
                * 1.0e3
            )

    for prefix, module in sorted(selected.items()):
        kind = "csa" if module.compress_ratio == 4 else "hca"
        values = samples[prefix]
        mhc_values = mhc_samples[prefix]
        print(
            "DSV4_VLLM_PREATTENTION "
            f"prefix={prefix} kind={kind} context={args.context} "
            f"samples={len(values)} min_us={min(values):.3f} "
            f"median_us={statistics.median(values):.3f} "
            f"max_us={max(values):.3f} "
            f"mhc_rms_median_us={statistics.median(mhc_values):.3f}",
            flush=True,
        )
    print(
        "DSV4_VLLM_DECODE_E2E "
        f"context={args.context} samples={len(model_sample_samples)} "
        f"model_logits_min_us={min(model_logits_samples):.3f} "
        f"model_logits_median_us={statistics.median(model_logits_samples):.3f} "
        f"model_logits_max_us={max(model_logits_samples):.3f} "
        f"model_sample_min_us={min(model_sample_samples):.3f} "
        f"model_sample_median_us={statistics.median(model_sample_samples):.3f} "
        f"model_sample_max_us={max(model_sample_samples):.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
