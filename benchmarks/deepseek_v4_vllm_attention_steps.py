#!/usr/bin/env python3
"""Measure matched DeepSeek-V4 attention/output-projection GPU steps in vLLM.

The final request replay is a batch-one decode at the requested context.  The
probes separate FlashMLA attention, inverse-RoPE plus native FP8 quantization,
the grouped O_a einsum, O_b, and the complete attention-to-output chain.
"""

from __future__ import annotations

import argparse
import os
import statistics

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch

from vllm import LLM, SamplingParams
from vllm.models.deepseek_v4.nvidia import flashmla as flashmla_module
from vllm.models.deepseek_v4.nvidia.flashmla import (
    DeepseekV4FlashMLAAttention,
)
from vllm.models.deepseek_v4.nvidia.ops import o_proj as o_proj_module


_EVENT_NAMES = (
    "attention_start",
    "attention_end",
    "oproj_start",
    "invquant_end",
    "oa_end",
    "oproj_end",
)


def _event() -> torch.cuda.Event:
    # External events remain timestamp records in breakable CUDA graphs.
    return torch.cuda.Event(enable_timing=True, external=True)


def install_event_probes() -> dict[str, DeepseekV4FlashMLAAttention]:
    registry: dict[str, DeepseekV4FlashMLAAttention] = {}
    active: list[DeepseekV4FlashMLAAttention | None] = [None]

    original_init = DeepseekV4FlashMLAAttention.__init__

    def attention_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        for name in _EVENT_NAMES:
            setattr(self, f"_step_{name}", _event())
        registry[self.prefix] = self

    DeepseekV4FlashMLAAttention.__init__ = attention_init

    original_forward_mqa = DeepseekV4FlashMLAAttention.forward_mqa

    def forward_mqa(self, *args, **kwargs):
        self._step_attention_start.record()
        result = original_forward_mqa(self, *args, **kwargs)
        self._step_attention_end.record()
        return result

    DeepseekV4FlashMLAAttention.forward_mqa = forward_mqa

    original_o_proj = DeepseekV4FlashMLAAttention._o_proj

    def attention_o_proj(self, *args, **kwargs):
        active[0] = self
        self._step_oproj_start.record()
        try:
            result = original_o_proj(self, *args, **kwargs)
            self._step_oproj_end.record()
            return result
        finally:
            active[0] = None

    DeepseekV4FlashMLAAttention._o_proj = attention_o_proj

    original_invquant = o_proj_module.fused_inv_rope_fp8_quant

    def fused_invquant(*args, **kwargs):
        result = original_invquant(*args, **kwargs)
        module = active[0]
        if module is not None:
            module._step_invquant_end.record()
        return result

    o_proj_module.fused_inv_rope_fp8_quant = fused_invquant

    original_einsum = o_proj_module.fp8_einsum

    def fp8_einsum(*args, **kwargs):
        result = original_einsum(*args, **kwargs)
        module = active[0]
        if module is not None:
            module._step_oa_end.record()
        return result

    o_proj_module.fp8_einsum = fp8_einsum

    # The class method resolves this symbol in the flashmla module.
    flashmla_module.deep_gemm_fp8_o_proj = o_proj_module.deep_gemm_fp8_o_proj
    return registry


def _span_us(module, begin: str, end: str) -> float:
    start = getattr(module, f"_step_{begin}")
    stop = getattr(module, f"_step_{end}")
    return start.elapsed_time(stop) * 1.0e3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--target-layers", default="3,4")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--token-id", type=int, default=791)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    args = parser.parse_args()
    if args.context < 2:
        parser.error("context must be >=2")
    if args.warmups < 0 or args.samples <= 0:
        parser.error("warmups must be non-negative and samples positive")
    target_layers = {int(value) for value in args.target_layers.split(",")}

    registry = install_event_probes()
    engine = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        kv_cache_dtype="fp8",
        max_model_len=args.context + 1,
        max_num_seqs=1,
        max_num_batched_tokens=args.context - 1,
        enable_prefix_caching=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        skip_tokenizer_init=True,
        trust_remote_code=False,
        kernel_config={"enable_flashinfer_autotune": False},
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
        if any(f".layers.{layer_id}." in prefix for layer_id in target_layers)
    }
    if not selected:
        raise RuntimeError("instrumentation found no target attention layers")

    spans = {
        "attention": ("attention_start", "attention_end"),
        "invrope_quant": ("oproj_start", "invquant_end"),
        "oa": ("invquant_end", "oa_end"),
        "invrope_quant_oa": ("oproj_start", "oa_end"),
        "ob": ("oa_end", "oproj_end"),
        "oproj": ("oproj_start", "oproj_end"),
        "attention_oproj": ("attention_start", "oproj_end"),
    }
    values = {
        prefix: {name: [] for name in spans} for prefix in selected
    }
    for _ in range(args.samples):
        run_once()
        for prefix, module in selected.items():
            for name, (begin, end) in spans.items():
                values[prefix][name].append(_span_us(module, begin, end))

    for prefix, module in sorted(selected.items()):
        kind = (
            "swa" if module.compress_ratio <= 1
            else "csa" if module.compress_ratio == 4
            else "hca"
        )
        for name in spans:
            samples = values[prefix][name]
            print(
                "DSV4_VLLM_ATTENTION_STEP "
                f"prefix={prefix} kind={kind} context={args.context} "
                f"step={name} samples={len(samples)} "
                f"min_us={min(samples):.3f} "
                f"median_us={statistics.median(samples):.3f} "
                f"max_us={max(samples):.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
