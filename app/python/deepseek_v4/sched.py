#!/usr/bin/env python3
"""Offline-PyTorch prefill followed by live VDCores DeepSeek-V4 decode."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.deepseek_v4_resident_one_launch import (
    ResidentOneLaunchDecode,
    build_argument_parser,
)
from dae.deepseek_v4 import DeepSeekV4FlashConfig
from dae.deepseek_v4_live import DeepSeekV4LiveDecodeState
from dae.deepseek_v4_quant import dequantize_nvfp4


DEFAULT_PROMPT = "Write a hello world program in Python."


def _production_args(
    checkpoint: str,
    mxfp_ffn_root: str | None,
    *,
    context_length: int,
    token_id: int,
) -> argparse.Namespace:
    argv = [
        "--checkpoint",
        checkpoint,
        "--layers",
        "43",
        "--context-length",
        str(context_length),
        "--token-id",
        str(token_id),
        "--vocab-size",
        str(DeepSeekV4FlashConfig().vocab_size),
        "--bf16-head",
        "--loopback-hc-fusion",
        "--allow-token-variation",
    ]
    if mxfp_ffn_root is not None:
        argv.extend(("--mxfp-ffn-root", mxfp_ffn_root))
    return build_argument_parser().parse_args(argv)


def _load_source_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _torch_sparse_attention(q, kv, attn_sink, topk_indices, softmax_scale):
    """Small offline-prefill reference for the checkpoint's sparse attention."""

    batch = q.shape[0]
    safe_indices = topk_indices.clamp_min(0).to(torch.long)
    batch_indices = torch.arange(
        batch, device=q.device
    ).reshape(batch, 1, 1)
    selected = kv[batch_indices, safe_indices]
    scores = torch.einsum(
        "bshd,bskd->bshk", q.float(), selected.float()
    ) * softmax_scale
    scores.masked_fill_(topk_indices.unsqueeze(2) < 0, float("-inf"))
    sink = attn_sink.float().reshape(1, 1, -1)
    maximum = torch.maximum(scores.amax(dim=-1), sink)
    probabilities = torch.exp(scores - maximum.unsqueeze(-1))
    denominator = probabilities.sum(dim=-1) + torch.exp(sink - maximum)
    output = torch.einsum(
        "bshk,bskd->bshd", probabilities, selected.float()
    ) / denominator.unsqueeze(-1)
    return output.to(q.dtype)


def _install_offline_nvfp4_linear(reference) -> None:
    """Teach the offline reference to consume the released NVFP4 experts.

    The checkpoint's routed expert data uses ModelOpt's uint8 nibble storage,
    E4M3 block scales, and one FP32 tensor scale.  The reference model bundled
    with the checkpoint recognizes native torch Float4 instead.  Prefill is
    deliberately outside the timed path, so dequantize each selected expert
    explicitly and use ordinary PyTorch linear algebra.  This preserves the
    checkpoint values and avoids introducing another CUDA runtime into the
    live demo.
    """

    original_linear = reference.linear

    def offline_linear(x, weight, bias=None):
        if weight.dtype == torch.uint8:
            scale = getattr(weight, "_vdcores_nvfp4_scale", None)
            scale2 = getattr(weight, "_vdcores_nvfp4_scale2", None)
            if scale is None or scale2 is None:
                raise RuntimeError(
                    "NVFP4 weight is missing its checkpoint scales"
                )
            weight = dequantize_nvfp4(weight, scale, scale2).to(x.dtype)
            return torch.nn.functional.linear(x, weight, bias)
        if weight.dtype in (torch.bfloat16, torch.float16, torch.float32):
            # assign=True intentionally preserves checkpoint storage.  A few
            # compressor projections are declared FP32 by the reference but
            # released as BF16; match the FP32 activation at the operation.
            if weight.dtype != x.dtype:
                weight = weight.to(x.dtype)
            return torch.nn.functional.linear(x, weight, bias)
        return original_linear(x, weight, bias)

    reference.linear = offline_linear


def _tokenize_prompt(
    checkpoint: Path,
    prompt: str,
    thinking_mode: str,
):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True
    )
    encoding = _load_source_module(
        "_vdcores_deepseek_v4_encoding",
        checkpoint / "encoding" / "encoding_dsv4.py",
    )
    formatted = encoding.encode_messages(
        [{"role": "user", "content": prompt}],
        thinking_mode=thinking_mode,
    )
    token_ids = tokenizer.encode(formatted)
    if not token_ids:
        raise RuntimeError("the formatted prompt produced no tokens")
    return tokenizer, token_ids


@torch.inference_mode()
def _offline_prefill(
    checkpoint: Path,
    converted_checkpoint: Path,
    token_ids: list[int],
    state: DeepSeekV4LiveDecodeState,
    device: torch.device,
) -> None:
    """Run the official PyTorch reference on all but the final prompt token."""

    prefix = token_ids[:-1]
    if not prefix:
        print("[prefill] empty prefix; live state starts at context one", flush=True)
        return
    model_file = converted_checkpoint / "model0-mp1.safetensors"
    if not model_file.is_file():
        raise FileNotFoundError(
            "offline prefill needs the official MP1 checkpoint at "
            f"{model_file}; run inference/convert.py once as documented in "
            "the VDCores README"
        )

    inference_dir = checkpoint / "inference"
    config_path = inference_dir / "config.json"
    old_dtype = torch.get_default_dtype()
    sys.path.insert(0, str(inference_dir))
    started = time.perf_counter()
    try:
        try:
            reference = _load_source_module(
                "_vdcores_deepseek_v4_reference", inference_dir / "model.py"
            )
            # TileLang 0.1.8's sparse-attention lowering does not support the
            # CUDA-13/Python-3.12 toolchain used by the VDCores environment.
            # Prefill is offline and its selected context is small, so use the
            # exact PyTorch formulation while retaining the checkpoint's
            # quantized GEMMs and all model math.
            reference.sparse_attn = _torch_sparse_attention
            _install_offline_nvfp4_linear(reference)
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "missing official PyTorch prefill dependency; install "
                f"{inference_dir / 'requirements.txt'}"
            ) from error
        from safetensors.torch import load_file

        with config_path.open() as handle:
            model_args = reference.ModelArgs(**json.load(handle))
        model_args.max_batch_size = 1
        model_args.max_seq_len = state.max_seq_len
        model_args.n_mtp_layers = 0
        # The source config leaves routed-expert dtype unset, while the MP1
        # loader image retains half-width packed NVFP4 storage.  Declare the
        # packed logical shape here; the explicit uint8 adapter above performs
        # its numeric interpretation during offline prefill.
        model_args.expert_dtype = "fp4"
        torch.set_default_dtype(torch.bfloat16)
        # Construct parameter metadata on CPU, then let safetensors allocate
        # the packed quantized tensors directly on CUDA.  PyTorch 2.13's
        # load_state_dict copy path does not implement Float4 copy_, whereas
        # assign=True preserves the checkpoint storage without a conversion or
        # a second device-sized allocation.
        with torch.device("cpu"):
            model = reference.Transformer(model_args)
        # Prefill stops at the final transformer hidden state.  The full
        # vocabulary head is a decode concern and need not stay resident.
        model.head = torch.nn.Identity()
        model.requires_grad_(False)
        print(
            "[prefill] loading official MP1 PyTorch weights "
            f"prefix_tokens={len(prefix)}",
            flush=True,
        )
        state_dict = load_file(str(model_file), device=str(device))
        incompatible = model.load_state_dict(
            state_dict, strict=False, assign=True
        )
        invalid_missing = [
            name
            for name in incompatible.missing_keys
            if not (
                ".ffn.experts." in name
                and name.endswith((".w1.scale", ".w2.scale", ".w3.scale"))
            )
        ]
        invalid_unexpected = [
            name
            for name in incompatible.unexpected_keys
            if not (
                name == "head.weight"
                or name.startswith("mtp.")
                or (
                    ".ffn.experts." in name
                    and name.endswith(
                        (".input_scale", ".weight_scale", ".weight_scale_2")
                    )
                )
            )
        ]
        if invalid_missing or invalid_unexpected:
            raise RuntimeError(
                "offline checkpoint does not match the prefill model: "
                f"missing={invalid_missing[:8]} "
                f"unexpected={invalid_unexpected[:8]}"
            )
        # Nonpersistent KV/frequency buffers are absent from safetensors and
        # remain on the construction device.  Move them directly, without
        # traversing/re-copying the already assigned packed parameters.
        for module_name, module in model.named_modules():
            for name, buffer in tuple(module._buffers.items()):
                if buffer is not None and buffer.device != device:
                    module._buffers[name] = buffer.to(device)
            scale = getattr(module, "scale", None)
            weight = getattr(module, "weight", None)
            if weight is not None and weight.dtype == torch.uint8:
                key_prefix = f"{module_name}." if module_name else ""
                weight._vdcores_nvfp4_scale = state_dict[
                    f"{key_prefix}weight_scale"
                ]
                weight._vdcores_nvfp4_scale2 = state_dict[
                    f"{key_prefix}weight_scale_2"
                ]
            elif scale is not None and weight is not None:
                weight.scale = scale
        del state_dict
        torch.cuda.empty_cache()
        with torch.device(device):
            model.eval()
            input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
            hidden = model.embed(input_ids)
            hidden = hidden.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
            for layer_id, layer in enumerate(model.layers):
                hidden = layer(hidden, 0, input_ids)
                if layer_id == 0 or (layer_id + 1) % 8 == 0:
                    print(
                        f"[prefill] layer={layer_id + 1}/{len(model.layers)}",
                        flush=True,
                    )
            state.import_pytorch_prefill(model, len(prefix))
            torch.cuda.synchronize(device)
        del hidden, input_ids, model
    finally:
        torch.set_default_dtype(old_dtype)
        if sys.path and sys.path[0] == str(inference_dir):
            sys.path.pop(0)
        gc.collect()
        torch.cuda.empty_cache()
    print(
        f"[prefill] complete elapsed_s={time.perf_counter() - started:.3f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prefill a DeepSeek-V4 prompt with the official PyTorch model, "
            "then greedily decode one prepared VDCores launch per token."
        )
    )
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mxfp-ffn-root")
    parser.add_argument(
        "--prefill-checkpoint",
        help=(
            "directory containing model0-mp1.safetensors; defaults to "
            "CHECKPOINT/vdcores-pytorch-mp1"
        ),
    )
    parser.add_argument(
        "--thinking-mode", choices=("chat", "thinking"), default="chat"
    )
    parser.add_argument("-N", "--max-new-tokens", type=int, default=4)
    parser.add_argument(
        "--input-token-id",
        type=int,
        help=(
            "skip textual/PyTorch prefill and start from one token; useful "
            "for a fast context-one smoke run"
        ),
    )
    args = parser.parse_args()

    if not 1 <= args.max_new_tokens <= 64:
        parser.error("max-new-tokens must be in [1,64]")
    cfg = DeepSeekV4FlashConfig()
    checkpoint = Path(args.checkpoint).resolve()
    device = torch.device("cuda")

    if args.input_token_id is None:
        tokenizer, prompt_tokens = _tokenize_prompt(
            checkpoint, args.prompt, args.thinking_mode
        )
        print(
            f"[prompt] tokens={len(prompt_tokens)} text={args.prompt!r}",
            flush=True,
        )
    else:
        if not 0 <= args.input_token_id < cfg.vocab_size:
            parser.error("input-token-id is outside the vocabulary")
        tokenizer = None
        prompt_tokens = [args.input_token_id]
        print(f"[prompt] token_id={args.input_token_id}", flush=True)

    max_seq_len = len(prompt_tokens) + args.max_new_tokens
    state = DeepSeekV4LiveDecodeState(
        max_seq_len, device=device, config=cfg
    )
    if args.input_token_id is None:
        converted = Path(
            args.prefill_checkpoint
            or checkpoint / "vdcores-pytorch-mp1"
        ).resolve()
        _offline_prefill(
            checkpoint, converted, prompt_tokens, state, device
        )

    flows: list[ResidentOneLaunchDecode] = []
    weight_source = None
    prepare_started = time.perf_counter()
    print(
        "[prepare] loading one resident VDCores checkpoint and preparing "
        f"{args.max_new_tokens} position-specialized images",
        flush=True,
    )
    # Schedule construction may bind TMA descriptors to persistent state but
    # must never publish into it.  Keep a small setup-only guard so preparing
    # future positions cannot silently alter the first decode token.
    state_before_prepare = tuple(
        tensor.clone() for tensor in state.persistent_tensors()
    )
    for step in range(args.max_new_tokens):
        flow_args = _production_args(
            str(checkpoint),
            args.mxfp_ffn_root,
            context_length=len(prompt_tokens) + step,
            token_id=prompt_tokens[-1],
        )
        flow = ResidentOneLaunchDecode(
            flow_args,
            device,
            weight_source=weight_source,
            live_state=state,
        )
        if weight_source is None:
            weight_source = flow
        flows.append(flow)
    torch.cuda.synchronize(device)
    if any(
        not torch.equal(
            before.view(torch.uint8), after.view(torch.uint8)
        )
        for before, after in zip(
            state_before_prepare, state.persistent_tensors(), strict=True
        )
    ):
        raise RuntimeError(
            "preparing future VDCores images modified live decode state"
        )
    del state_before_prepare
    print(
        f"[prepare] complete elapsed_s={time.perf_counter() - prepare_started:.3f}",
        flush=True,
    )

    token = prompt_tokens[-1]
    generated: list[int] = []
    cuda_ms: list[float] = []
    device_ms: list[float] = []
    wall_ms: list[float] = []
    eos_token_id = None if tokenizer is None else tokenizer.eos_token_id
    for step, flow in enumerate(flows):
        wall_started = time.perf_counter_ns()
        flow.set_input_token(token)
        output, elapsed_ms, _ = flow.run_once()
        elapsed_wall_ms = (time.perf_counter_ns() - wall_started) / 1.0e6
        frontier_ms = flow.device_frontier_ms()
        generated.append(output)
        cuda_ms.append(elapsed_ms)
        device_ms.append(frontier_ms)
        wall_ms.append(elapsed_wall_ms)
        print(
            "[decode] "
            f"step={step} position={len(prompt_tokens) - 1 + step} "
            f"input_token={token} output_token={output} "
            f"cuda_ms={elapsed_ms:.6f} device_ms={frontier_ms:.6f} "
            f"wall_ms={elapsed_wall_ms:.6f}",
            flush=True,
        )
        if tokenizer is not None:
            print(
                f"[stream] {tokenizer.decode(generated, skip_special_tokens=True)!r}",
                flush=True,
            )
        token = output
        if eos_token_id is not None and output == eos_token_id:
            break

    print(f"[output] token_ids={generated}", flush=True)
    if tokenizer is not None:
        completion = tokenizer.decode(generated, skip_special_tokens=True)
        full_text = tokenizer.decode(
            prompt_tokens + generated, skip_special_tokens=True
        )
        print(f"[output] completion={completion!r}", flush=True)
        print(f"[output] full_text={full_text!r}", flush=True)
    median_wall = statistics.median(wall_ms)
    print(
        "[perf] "
        f"tokens={len(generated)} "
        f"median_cuda_ms={statistics.median(cuda_ms):.6f} "
        f"median_device_ms={statistics.median(device_ms):.6f} "
        f"median_wall_ms={median_wall:.6f} "
        f"tokens_per_s={1000.0 / median_wall:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
