#!/usr/bin/env python3
"""Slow, direct PyTorch C1 hidden-state oracle for DeepSeek-V4-Flash."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open

from dae.deepseek_v4 import (
    apply_partial_rope_128_64,
    apply_partial_rope_512_64,
    bounded_swiglu,
    hc_head_reference,
    hc_post_reference,
    hc_pre_reference,
    route_top6_reference,
    sparse_attention_512_reference,
)
from dae.deepseek_v4_quant import (
    dequantize_fp8_block128,
    quantize_fp8_block128,
)
from dae.deepseek_v4_mxfp_checkpoint import default_mxfp_ffn_directory

from deepseek_v4_checkpoint_decode import CheckpointDecode
from deepseek_v4_resident_ffn_stage_check import (
    _dequantize_mxfp4,
    _quantize_middle_reference,
)


LINEAR1_SLICES = 16
DOWN_SLICES = 32


class TorchCheckpointDecode(CheckpointDecode):
    """Reuse checkpoint orchestration while evaluating every op in PyTorch."""

    def _run(self, schedule, sms: int) -> None:
        raise AssertionError(
            f"pure PyTorch reference reached VDcores schedule {type(schedule).__name__}"
        )

    def _rms(
        self,
        source: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = source.float() * torch.rsqrt(
            source.float().square().mean(dim=-1, keepdim=True)
            + self.config.rms_epsilon
        )
        if weight is not None:
            output *= weight.float()
        return output.to(torch.bfloat16)

    def _quant_fp8(self, source: torch.Tensor):
        return quantize_fp8_block128(source.reshape(-1))

    def _fp8_loaded(
        self,
        linear,
        activation: torch.Tensor,
        activation_scale: torch.Tensor,
        *,
        row_slice: slice | None = None,
    ) -> torch.Tensor:
        weight = linear.weight
        weight_scale = linear.scale
        if row_slice is not None:
            start = 0 if row_slice.start is None else row_slice.start
            stop = weight.shape[0] if row_slice.stop is None else row_slice.stop
            weight = weight[row_slice]
            weight_scale = weight_scale[start // 128 : stop // 128]
        return (
            dequantize_fp8_block128(weight, weight_scale)
            @ dequantize_fp8_block128(activation, activation_scale)
        ).to(torch.bfloat16)

    def _bf16_weight(
        self,
        name: str,
        source: torch.Tensor,
        *,
        row_slice: slice | None = None,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        weight = (
            self._load([name])[name]
            if row_slice is None
            else self.checkpoint.load_tensor_slice(
                name, row_slice, device=str(self.device)
            )
        )
        return (weight.float() @ source.reshape(-1).float()).to(output_dtype)

    def _hc_pre(self, layer_id: int, branch: str, residual: torch.Tensor):
        prefix = f"layers.{layer_id}"
        names = [
            f"{prefix}.hc_{branch}_fn",
            f"{prefix}.hc_{branch}_scale",
            f"{prefix}.hc_{branch}_base",
            f"{prefix}.{branch}_norm.weight",
        ]
        tensors = self._load(names)
        mixes = tensors[names[0]].float() @ residual.reshape(-1).float()
        hidden, post, comb = hc_pre_reference(
            residual, mixes, tensors[names[1]], tensors[names[2]]
        )
        return self._rms(hidden, tensors[names[3]]), post, comb

    def _hc_post(self, branch, residual, post, comb):
        return hc_post_reference(branch, residual, post, comb)

    def _rope512(self, source, *, inverse=False, table=None):
        return apply_partial_rope_512_64(
            source.reshape(-1, 512),
            self.main_rope_table if table is None else table,
            inverse=inverse,
        ).reshape_as(source)

    def _rope128(self, source, *, table=None):
        return apply_partial_rope_128_64(
            source.reshape(-1, 128),
            self.compress_rope_table if table is None else table,
        ).reshape_as(source)

    def _attention_side_paths(
        self,
        layer_id,
        normalized,
        q_rank_activation,
        q_rank_scale,
        position,
    ):
        if position != 0:
            raise ValueError("the pure PyTorch oracle currently covers C1 only")
        # No ratio-4/8 compressor group is complete at the first token, so
        # compressor and indexer outputs cannot participate in C1 attention.
        return None, None

    def _attention(
        self,
        layer_id: int,
        normalized: torch.Tensor,
        position: int,
    ) -> torch.Tensor:
        prefix = f"layers.{layer_id}.attn"
        rope_table = (
            self.main_rope_table
            if self.config.compress_ratios[layer_id] == 0
            else self.compress_rope_table
        )
        hidden_activation, hidden_scale = self._quant_fp8(normalized)
        q_a = self.checkpoint.load_fp8_linear(
            f"{prefix}.wq_a", device=str(self.device)
        )
        q_rank = self._fp8_loaded(q_a, hidden_activation, hidden_scale)
        q_norm = self._load([f"{prefix}.q_norm.weight"])[
            f"{prefix}.q_norm.weight"
        ]
        q_rank = self._rms(q_rank, q_norm)
        q_rank_activation, q_rank_scale = self._quant_fp8(q_rank)
        q_b = self.checkpoint.load_fp8_linear(
            f"{prefix}.wq_b", device=str(self.device)
        )
        q = self._fp8_loaded(
            q_b, q_rank_activation, q_rank_scale
        ).reshape(64, 512)
        q = self._rope512(self._rms(q), table=rope_table)

        kv_linear = self.checkpoint.load_fp8_linear(
            f"{prefix}.wkv", device=str(self.device)
        )
        kv = self._fp8_loaded(kv_linear, hidden_activation, hidden_scale)
        kv_norm = self._load([f"{prefix}.kv_norm.weight"])[
            f"{prefix}.kv_norm.weight"
        ]
        kv = self._rope512(
            self._rms(kv, kv_norm).reshape(1, 512), table=rope_table
        )
        sink = self._load([f"{prefix}.attn_sink"])[f"{prefix}.attn_sink"]
        cache = self.window_kv[layer_id]
        cache[position % self.config.sliding_window].copy_(kv.reshape(-1))
        indices = torch.arange(
            position + 1, dtype=torch.int32, device=self.device
        )
        attended = sparse_attention_512_reference(q, cache, indices, sink)
        attended = self._rope512(attended, inverse=True, table=rope_table)

        grouped = attended.reshape(self.config.o_groups, -1)
        wo_a = self.checkpoint.load_fp8_linear(
            f"{prefix}.wo_a", device=str(self.device)
        )
        o_rank = torch.empty(
            (self.config.o_groups, self.config.o_lora_rank),
            dtype=torch.bfloat16,
            device=self.device,
        )
        for group in range(self.config.o_groups):
            activation, scale = self._quant_fp8(grouped[group])
            start = group * self.config.o_lora_rank
            o_rank[group].copy_(
                self._fp8_loaded(
                    wo_a,
                    activation,
                    scale,
                    row_slice=slice(start, start + self.config.o_lora_rank),
                )
            )
        return self._fp8(f"{prefix}.wo_b", o_rank.reshape(-1))

    def _route(self, layer_id, normalized, token_id):
        prefix = f"layers.{layer_id}.ffn.gate"
        logits = self._bf16_weight(
            f"{prefix}.weight", normalized, output_dtype=torch.float32
        )
        if layer_id < self.config.num_hash_layers:
            hash_indices = self.checkpoint.load_tensor_slice(
                f"{prefix}.tid2eid", token_id, device=str(self.device)
            )
            bias = torch.zeros_like(logits)
        else:
            hash_indices = None
            bias = self._load([f"{prefix}.bias"])[f"{prefix}.bias"]
        weights, indices = route_top6_reference(
            logits,
            bias,
            hash_indices=hash_indices,
            route_scale=self.config.route_scale,
        )
        return indices, weights

    def _mxfp_expert(
        self,
        image,
        stream_expert: int,
        activation: torch.Tensor,
    ) -> torch.Tensor:
        linear1_begin = stream_expert * LINEAR1_SLICES
        linear1_end = linear1_begin + LINEAR1_SLICES
        linear1_weights = image.get_slice("linear1_weights")[
            linear1_begin:linear1_end
        ]
        linear1_scales = image.get_slice("linear1_scales")[
            linear1_begin:linear1_end
        ]
        gate_weight = _dequantize_mxfp4(
            linear1_weights[:, :8], linear1_scales[:, :8]
        ).to(self.device)
        up_weight = _dequantize_mxfp4(
            linear1_weights[:, 8:], linear1_scales[:, 8:]
        ).to(self.device)
        middle = bounded_swiglu(
            gate_weight @ activation,
            up_weight @ activation,
            self.config.swiglu_limit,
        )
        _, _, middle = _quantize_middle_reference(middle)
        del gate_weight, up_weight, linear1_weights, linear1_scales

        down_begin = stream_expert * DOWN_SLICES
        down_end = down_begin + DOWN_SLICES
        down_weight = _dequantize_mxfp4(
            image.get_slice("down_weights")[down_begin:down_end],
            image.get_slice("down_scales")[down_begin:down_end],
        ).to(self.device)
        output = down_weight @ middle
        del down_weight, middle
        return output

    def _ffn(self, layer_id, normalized, token_id):
        indices, weights = self._route(layer_id, normalized, token_id)
        stream_experts = [
            0,
            *[int(expert_id) + 1 for expert_id in indices.cpu().tolist()],
        ]
        route_scales = torch.cat(
            (
                torch.ones(1, dtype=torch.float32, device=self.device),
                weights.float(),
            )
        )
        _, _, activation = _quantize_middle_reference(normalized)
        image_path = (
            Path(self.args.mxfp_ffn_root) / f"layer-{layer_id:03d}.safetensors"
        )
        contributions = []
        with safe_open(str(image_path), framework="pt", device="cpu") as image:
            for stream_expert, route_scale in zip(
                stream_experts, route_scales
            ):
                contribution = self._mxfp_expert(
                    image, stream_expert, activation
                )
                # The production down task applies the routed scale before
                # publishing each BF16 contribution to its split-K reducer.
                contributions.append(
                    (contribution * route_scale).to(torch.bfloat16).float()
                )
        if self.args.trace_stages:
            print(
                "DSV4_TORCH_MXFP_FFN "
                f"layer={layer_id} stream_experts={stream_experts} "
                f"route_weights={weights.float().cpu().tolist()}",
                flush=True,
            )
        return torch.stack(contributions).sum(dim=0).to(torch.bfloat16)

    def _head(self, residual: torch.Tensor):
        names = ["hc_head_fn", "hc_head_scale", "hc_head_base", "norm.weight"]
        tensors = self._load(names)
        mixes = tensors["hc_head_fn"].float() @ residual.reshape(-1).float()
        hidden = hc_head_reference(
            residual, mixes, tensors["hc_head_scale"], tensors["hc_head_base"]
        )
        hidden = self._rms(hidden, tensors["norm.weight"])
        logits = self._bf16_weight(
            "head.weight", hidden, row_slice=slice(0, self.args.vocab_size)
        )
        return int(torch.argmax(logits).item()), logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--mxfp-ffn-root",
        help=(
            "offline MXFP4 checkpoint image; defaults to the production "
            "vdcores-mxfp4-ffn-v1 directory"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--token-id", type=int, default=791)
    parser.add_argument("--expected-token-id", type=int, default=14)
    parser.add_argument("--layers", type=int, default=43)
    parser.add_argument("--vocab-size", type=int, default=129280)
    parser.add_argument(
        "--layer-sequence",
        help=(
            "diagnostically replay this comma-separated layer sequence from "
            "an existing tagged reference input without running the head"
        ),
    )
    parser.add_argument(
        "--input-reference",
        help="tagged MXFP4/FP8 hidden reference supplying the first layer input",
    )
    args = parser.parse_args()
    if args.mxfp_ffn_root is None:
        args.mxfp_ffn_root = str(default_mxfp_ffn_directory(args.checkpoint))
    args.start_pos = 0
    args.decode_tokens = 1
    args.hidden_output = args.output
    args.sms = 152
    args.trace_stages = True
    args.resident = False
    args.resident_reserve_gib = 0.0
    flow = TorchCheckpointDecode(args, torch.device("cuda"))
    if args.layer_sequence is not None:
        if args.input_reference is None:
            parser.error("--layer-sequence requires --input-reference")
        try:
            sequence = tuple(
                int(value) for value in args.layer_sequence.split(",")
            )
        except ValueError as error:
            parser.error(f"layer-sequence must contain integers: {error}")
        if not sequence or any(
            not 0 <= layer_id < flow.config.num_layers
            for layer_id in sequence
        ):
            parser.error("layer-sequence contains an invalid layer")
        reference = torch.load(
            args.input_reference, map_location="cpu", weights_only=True
        )
        residual = reference["pre_layer"][sequence[0]].to(flow.device)
        for sequence_index, layer_id in enumerate(sequence):
            residual = flow._layer(layer_id, residual, args.token_id, 0)
            print(
                "DSV4_TORCH_MXFP_SEQUENCE "
                f"index={sequence_index} layer={layer_id} "
                f"norm={float(residual.float().norm()):.9f}",
                flush=True,
            )
        replay = dict(reference)
        replay["post_layer"] = reference["post_layer"].clone()
        replay["post_layer"][sequence[-1]].copy_(residual.cpu())
        replay.update(
            {
                "ffn_weight_format": "mxfp4",
                "ffn_activation_format": "mxfp8_e4m3_group32",
                "layer_sequence": torch.tensor(sequence, dtype=torch.int64),
            }
        )
        torch.save(replay, args.output)
        print(
            "DSV4_TORCH_MXFP_SEQUENCE status=PASS "
            f"sequence={sequence} path={args.output}",
            flush=True,
        )
        return
    tokens, _, elapsed = flow.run()
    reference = torch.load(args.output, map_location="cpu", weights_only=True)
    reference.update(
        {
            "ffn_weight_format": "mxfp4",
            "ffn_activation_format": "mxfp8_e4m3_group32",
        }
    )
    torch.save(reference, args.output)
    if tokens != [args.expected_token_id]:
        raise AssertionError(
            "pure PyTorch checkpoint oracle did not reproduce the expected "
            f"token: expected={[args.expected_token_id]} actual={tokens}"
        )
    print(
        "DSV4_TORCH_HIDDEN_REFERENCE status=PASS "
        f"layers={args.layers} input_token={args.token_id} "
        f"output_token={tokens[0]} elapsed_s={elapsed:.3f} path={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
