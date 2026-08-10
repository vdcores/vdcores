#!/usr/bin/env python3
"""Stream a real DeepSeek-V4 checkpoint through VDCores on one GPU.

The first breadth gate intentionally supports decode position zero.  At that
position RoPE is exactly identity and no historical compressed/index cache is
consumed, while every transformer layer, both routing modes, both quantized
linear formats, mHC, sparse attention, and the real vocabulary head execute.
Weights are loaded lazily from worker-local storage and released after use, so
the complete model runs on one GPU without first duplicating the 157-GiB
checkpoint in host or device memory.  This is a functional flow, not TBT data.
"""

from __future__ import annotations

import argparse
import gc
import time

import torch

from dae.deepseek_v4 import DeepSeekV4FlashConfig
from dae.deepseek_v4_checkpoint import (
    DeepSeekV4Checkpoint,
    Fp8LinearCheckpointTensors,
    Nvfp4LinearCheckpointTensors,
)
from dae.deepseek_v4_flow import build_layer_decode_plan
from dae.launcher import Launcher
from dae.schedule import (
    SchedDsv4Bf16Gemv,
    SchedDsv4ExpertReduce,
    SchedDsv4Fp32Bf16Gemv,
    SchedDsv4Fp8Quant128,
    SchedDsv4Hadamard,
    SchedDsv4HcHead,
    SchedDsv4HcPost,
    SchedDsv4HcPre,
    SchedDsv4Nvfp4Quant16,
    SchedDsv4Rope128_64,
    SchedDsv4Rope512_64,
    SchedDsv4RouteTop6,
    SchedDsv4SparseAttention512,
    SchedFp8Block128Gemv,
    SchedNvfp4Gemv,
    SchedRMS,
    SchedSmemSiLUInterleaved,
)


class CheckpointDecode:
    def __init__(self, args: argparse.Namespace, device: torch.device):
        self.args = args
        self.device = device
        self.config = DeepSeekV4FlashConfig()
        self.checkpoint = DeepSeekV4Checkpoint(args.checkpoint, self.config)
        self.sms = min(
            args.sms,
            torch.cuda.get_device_properties(device).multi_processor_count,
        )
        self.rope_table = torch.zeros((32, 2), dtype=torch.float32, device=device)
        self.rope_table[:, 0] = 1.0

    def _run(self, schedule, sms: int) -> None:
        launcher = Launcher(sms, device=self.device)
        launcher.s(schedule.place(sms))
        launcher.launch()

    def _load(self, names: list[str]) -> dict[str, torch.Tensor]:
        return self.checkpoint.load_tensors(names, device=str(self.device))

    def _rms(
        self,
        source: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rows = source.reshape(-1, source.shape[-1])
        output = torch.empty_like(rows)
        self._run(
            SchedRMS(
                rows.shape[0],
                self.config.rms_epsilon,
                rows,
                output,
                weight,
                hidden_size=rows.shape[1],
            ),
            rows.shape[0],
        )
        return output.reshape_as(source)

    def _quant_fp8(self, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        source = source.reshape(-1)
        quantized = torch.empty_like(source, dtype=torch.float8_e4m3fn)
        scale = torch.empty(
            (source.numel() // 128,),
            dtype=torch.float8_e8m0fnu,
            device=self.device,
        )
        quant_sms = min(scale.numel(), self.sms)
        self._run(SchedDsv4Fp8Quant128(source, quantized, scale), quant_sms)
        return quantized, scale

    def _fp8_loaded(
        self,
        linear: Fp8LinearCheckpointTensors,
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
            if start % 128 or stop % 128:
                raise ValueError("FP8 checkpoint row slices must be 128 aligned")
            weight = weight[row_slice]
            weight_scale = weight_scale[start // 128 : stop // 128]
        output = torch.empty(
            (weight.shape[0],), dtype=torch.bfloat16, device=self.device
        )
        self._run(
            SchedFp8Block128Gemv(
                weight,
                weight_scale,
                activation,
                activation_scale,
                output,
            ),
            min(output.numel(), self.sms),
        )
        return output

    def _fp8(self, prefix: str, source: torch.Tensor) -> torch.Tensor:
        activation, activation_scale = self._quant_fp8(source)
        linear = self.checkpoint.load_fp8_linear(prefix, device=str(self.device))
        return self._fp8_loaded(linear, activation, activation_scale)

    def _quant_nvfp4(
        self,
        source: torch.Tensor,
        input_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source = source.reshape(-1)
        packed = torch.empty(
            (source.numel() // 2,), dtype=torch.uint8, device=self.device
        )
        scale = torch.empty(
            (source.numel() // 16,),
            dtype=torch.float8_e4m3fn,
            device=self.device,
        )
        self._run(
            SchedDsv4Nvfp4Quant16(
                source,
                input_scale.reshape(1),
                packed,
                scale,
            ),
            min(scale.numel(), self.sms),
        )
        return packed, scale

    def _nvfp4_loaded(
        self,
        linear: Nvfp4LinearCheckpointTensors,
        activation: torch.Tensor,
        activation_scale: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.empty(
            (linear.weight.shape[0],), dtype=torch.bfloat16, device=self.device
        )
        self._run(
            SchedNvfp4Gemv(
                linear.weight,
                linear.weight_scale,
                activation,
                activation_scale,
                linear.alpha,
                output,
            ),
            min(output.numel(), self.sms),
        )
        return output

    def _bf16_weight(
        self,
        name: str,
        source: torch.Tensor,
        *,
        row_slice: slice | None = None,
    ) -> torch.Tensor:
        if row_slice is None:
            weight = self._load([name])[name]
        else:
            weight = self.checkpoint.load_tensor_slice(
                name, row_slice, device=str(self.device)
            )
        output = torch.empty(
            (weight.shape[0],), dtype=torch.bfloat16, device=self.device
        )
        self._run(
            SchedDsv4Bf16Gemv(weight, source.reshape(-1), output),
            min(output.numel(), self.sms),
        )
        return output

    def _hc_pre(
        self,
        layer_id: int,
        branch: str,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix = f"layers.{layer_id}"
        names = [
            f"{prefix}.hc_{branch}_fn",
            f"{prefix}.hc_{branch}_scale",
            f"{prefix}.hc_{branch}_base",
            f"{prefix}.{branch}_norm.weight",
        ]
        tensors = self._load(names)
        mixes = torch.empty((24,), dtype=torch.float32, device=self.device)
        self._run(
            SchedDsv4Fp32Bf16Gemv(
                tensors[names[0]], residual.reshape(-1), mixes
            ),
            24,
        )
        hidden = torch.empty(
            (self.config.hidden_size,), dtype=torch.bfloat16, device=self.device
        )
        post = torch.empty((4,), dtype=torch.float32, device=self.device)
        comb = torch.empty((4, 4), dtype=torch.float32, device=self.device)
        self._run(
            SchedDsv4HcPre(
                residual,
                mixes,
                tensors[names[1]],
                tensors[names[2]],
                hidden,
                post,
                comb,
            ),
            1,
        )
        return self._rms(hidden, tensors[names[3]]), post, comb

    def _hc_post(
        self,
        branch: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.empty_like(residual)
        self._run(SchedDsv4HcPost(branch, residual, post, comb, output), 1)
        return output

    def _rope512(self, source: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
        rows = source.reshape(-1, 512)
        output = torch.empty_like(rows)
        self._run(
            SchedDsv4Rope512_64(rows, self.rope_table, output, inverse=inverse),
            1,
        )
        return output.reshape_as(source)

    def _attention_side_paths(
        self,
        layer_id: int,
        normalized: torch.Tensor,
        q_rank_activation: torch.Tensor,
        q_rank_scale: torch.Tensor,
    ) -> None:
        plan = build_layer_decode_plan(layer_id, 0, self.config)
        if plan.compress_ratio:
            prefix = f"layers.{layer_id}.attn.compressor"
            self._bf16_weight(f"{prefix}.wkv.weight", normalized)
            self._bf16_weight(f"{prefix}.wgate.weight", normalized)
        if plan.attention_kind != "csa":
            return
        prefix = f"layers.{layer_id}.attn.indexer"
        linear = self.checkpoint.load_fp8_linear(
            f"{prefix}.wq_b", device=str(self.device)
        )
        index_q = self._fp8_loaded(linear, q_rank_activation, q_rank_scale)
        index_q = self._rope128(index_q.reshape(64, 128))
        transformed = torch.empty_like(index_q)
        self._run(SchedDsv4Hadamard(index_q, transformed), 64)
        self._bf16_weight(f"{prefix}.weights_proj.weight", normalized)
        compressor = f"{prefix}.compressor"
        self._bf16_weight(f"{compressor}.wkv.weight", normalized)
        self._bf16_weight(f"{compressor}.wgate.weight", normalized)

    def _rope128(self, source: torch.Tensor) -> torch.Tensor:
        rows = source.reshape(-1, 128)
        output = torch.empty_like(rows)
        self._run(SchedDsv4Rope128_64(rows, self.rope_table, output), 1)
        return output.reshape_as(source)

    def _attention(self, layer_id: int, normalized: torch.Tensor) -> torch.Tensor:
        prefix = f"layers.{layer_id}.attn"
        hidden_activation, hidden_scale = self._quant_fp8(normalized)
        q_a = self.checkpoint.load_fp8_linear(
            f"{prefix}.wq_a", device=str(self.device)
        )
        q_rank = self._fp8_loaded(q_a, hidden_activation, hidden_scale)
        q_norm_weight = self._load([f"{prefix}.q_norm.weight"])[
            f"{prefix}.q_norm.weight"
        ]
        q_rank = self._rms(q_rank, q_norm_weight)
        q_rank_activation, q_rank_scale = self._quant_fp8(q_rank)
        q_b = self.checkpoint.load_fp8_linear(
            f"{prefix}.wq_b", device=str(self.device)
        )
        q = self._fp8_loaded(q_b, q_rank_activation, q_rank_scale).reshape(64, 512)
        q = self._rms(q)
        q = self._rope512(q)

        kv_linear = self.checkpoint.load_fp8_linear(
            f"{prefix}.wkv", device=str(self.device)
        )
        kv = self._fp8_loaded(kv_linear, hidden_activation, hidden_scale)
        kv_norm_weight = self._load([f"{prefix}.kv_norm.weight"])[
            f"{prefix}.kv_norm.weight"
        ]
        kv = self._rms(kv, kv_norm_weight)
        kv = self._rope512(kv.reshape(1, 512))
        self._attention_side_paths(
            layer_id, normalized, q_rank_activation, q_rank_scale
        )

        sink = self._load([f"{prefix}.attn_sink"])[f"{prefix}.attn_sink"]
        indices = torch.zeros((1,), dtype=torch.int32, device=self.device)
        attended = torch.empty_like(q)
        self._run(
            SchedDsv4SparseAttention512(q, kv, indices, sink, attended),
            64,
        )
        attended = self._rope512(attended, inverse=True)

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

    def _route(
        self,
        layer_id: int,
        normalized: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix = f"layers.{layer_id}.ffn.gate"
        logits = self._bf16_weight(f"{prefix}.weight", normalized)
        hash_routing = layer_id < self.config.num_hash_layers
        if hash_routing:
            hash_indices = self.checkpoint.load_tensor_slice(
                f"{prefix}.tid2eid",
                self.args.token_id,
                device=str(self.device),
            ).to(torch.int32)
            bias = torch.zeros(
                (self.config.num_experts,), dtype=torch.float32, device=self.device
            )
        else:
            hash_indices = torch.zeros(
                (self.config.experts_per_token,),
                dtype=torch.int32,
                device=self.device,
            )
            bias = self._load([f"{prefix}.bias"])[f"{prefix}.bias"]
        indices = torch.empty(
            (self.config.experts_per_token,),
            dtype=torch.int32,
            device=self.device,
        )
        weights = torch.empty(
            (self.config.experts_per_token,),
            dtype=torch.float32,
            device=self.device,
        )
        self._run(
            SchedDsv4RouteTop6(
                logits,
                bias,
                hash_indices,
                indices,
                weights,
                hash_routing=hash_routing,
                route_scale=self.config.route_scale,
            ),
            1,
        )
        return indices, weights

    def _expert(
        self,
        layer_id: int,
        expert_id: int,
        normalized: torch.Tensor,
    ) -> torch.Tensor:
        prefix = f"layers.{layer_id}.ffn.experts.{expert_id}"
        w1 = self.checkpoint.load_nvfp4_linear(
            f"{prefix}.w1", device=str(self.device)
        )
        w3 = self.checkpoint.load_nvfp4_linear(
            f"{prefix}.w3", device=str(self.device)
        )
        if not bool(torch.equal(w1.input_scale, w3.input_scale)):
            raise ValueError(f"{prefix} w1/w3 input scales differ")
        activation, scale = self._quant_nvfp4(normalized, w1.input_scale)
        gate = self._nvfp4_loaded(w1, activation, scale)
        up = self._nvfp4_loaded(w3, activation, scale)
        middle = torch.empty_like(gate)
        self._run(
            SchedSmemSiLUInterleaved(
                1,
                gate.reshape(1, -1),
                up.reshape(1, -1),
                middle.reshape(1, -1),
                swiglu_limit=self.config.swiglu_limit,
            ),
            1,
        )
        w2 = self.checkpoint.load_nvfp4_linear(
            f"{prefix}.w2", device=str(self.device)
        )
        activation, scale = self._quant_nvfp4(middle, w2.input_scale)
        return self._nvfp4_loaded(w2, activation, scale)

    def _shared_expert(
        self,
        layer_id: int,
        normalized: torch.Tensor,
    ) -> torch.Tensor:
        prefix = f"layers.{layer_id}.ffn.shared_experts"
        activation, scale = self._quant_fp8(normalized)
        w1 = self.checkpoint.load_fp8_linear(
            f"{prefix}.w1", device=str(self.device)
        )
        w3 = self.checkpoint.load_fp8_linear(
            f"{prefix}.w3", device=str(self.device)
        )
        gate = self._fp8_loaded(w1, activation, scale)
        up = self._fp8_loaded(w3, activation, scale)
        middle = torch.empty_like(gate)
        self._run(
            SchedSmemSiLUInterleaved(
                1,
                gate.reshape(1, -1),
                up.reshape(1, -1),
                middle.reshape(1, -1),
                swiglu_limit=self.config.swiglu_limit,
            ),
            1,
        )
        return self._fp8(f"{prefix}.w2", middle)

    def _ffn(self, layer_id: int, normalized: torch.Tensor) -> torch.Tensor:
        indices, weights = self._route(layer_id, normalized)
        expert_ids = [int(value) for value in indices.cpu().tolist()]
        if any(not 0 <= value < self.config.num_experts for value in expert_ids):
            raise AssertionError(f"layer {layer_id} emitted invalid experts {expert_ids}")
        routed = torch.empty(
            (self.config.experts_per_token, self.config.hidden_size),
            dtype=torch.bfloat16,
            device=self.device,
        )
        for rank, expert_id in enumerate(expert_ids):
            routed[rank].copy_(self._expert(layer_id, expert_id, normalized))
        shared = self._shared_expert(layer_id, normalized)
        output = torch.empty_like(shared)
        self._run(SchedDsv4ExpertReduce(routed, weights, shared, output), 1)
        if self.args.trace_stages:
            print(
                f"DSV4_CHECKPOINT_ROUTE layer={layer_id} experts={expert_ids}",
                flush=True,
            )
        return output

    def _layer(self, layer_id: int, residual: torch.Tensor) -> torch.Tensor:
        normalized, post, comb = self._hc_pre(layer_id, "attn", residual)
        branch = self._attention(layer_id, normalized)
        residual = self._hc_post(branch, residual, post, comb)
        normalized, post, comb = self._hc_pre(layer_id, "ffn", residual)
        branch = self._ffn(layer_id, normalized)
        residual = self._hc_post(branch, residual, post, comb)
        if not bool(torch.isfinite(residual).all().item()):
            raise AssertionError(f"layer {layer_id} produced non-finite residual state")
        return residual

    def _head(self, residual: torch.Tensor) -> tuple[int, torch.Tensor]:
        names = ["hc_head_fn", "hc_head_scale", "hc_head_base", "norm.weight"]
        tensors = self._load(names)
        mixes = torch.empty((4,), dtype=torch.float32, device=self.device)
        self._run(
            SchedDsv4Fp32Bf16Gemv(
                tensors["hc_head_fn"], residual.reshape(-1), mixes
            ),
            4,
        )
        hidden = torch.empty(
            (self.config.hidden_size,), dtype=torch.bfloat16, device=self.device
        )
        self._run(
            SchedDsv4HcHead(
                residual,
                mixes,
                tensors["hc_head_scale"],
                tensors["hc_head_base"],
                hidden,
            ),
            1,
        )
        hidden = self._rms(hidden, tensors["norm.weight"])
        logits = self._bf16_weight(
            "head.weight",
            hidden,
            row_slice=slice(0, self.args.vocab_size),
        )
        if not bool(torch.isfinite(logits).all().item()):
            raise AssertionError("checkpoint decode produced non-finite logits")
        return int(torch.argmax(logits).item()), logits

    def run(self) -> tuple[int, torch.Tensor, float]:
        embedding = self.checkpoint.load_tensor_slice(
            "embed.weight",
            self.args.token_id,
            device=str(self.device),
        )
        residual = embedding.reshape(1, -1).repeat(self.config.hc_mult, 1)
        started = time.monotonic()
        for layer_id in range(self.args.layers):
            layer_started = time.monotonic()
            residual = self._layer(layer_id, residual)
            print(
                "DSV4_CHECKPOINT_LAYER status=PASS "
                f"layer={layer_id} kind={self.config.attention_kind(layer_id)} "
                f"elapsed_s={time.monotonic() - layer_started:.3f}",
                flush=True,
            )
            gc.collect()
            torch.cuda.empty_cache()
        token, logits = self._head(residual)
        return token, logits, time.monotonic() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--token-id", type=int, default=791)
    parser.add_argument("--start-pos", type=int, default=0)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--sms", type=int, default=152)
    parser.add_argument("--trace-stages", action="store_true")
    args = parser.parse_args()
    config = DeepSeekV4FlashConfig()
    if not 1 <= args.layers <= config.num_layers:
        parser.error("layers must be in [1,43]")
    if not 0 <= args.token_id < config.vocab_size:
        parser.error("token-id is outside the vocabulary")
    if args.start_pos != 0:
        parser.error("the first real-checkpoint breadth gate supports start-pos=0")
    if not 1 <= args.vocab_size <= config.vocab_size:
        parser.error("vocab-size must be in [1,129280]")
    if args.sms <= 0:
        parser.error("sms must be positive")

    flow = CheckpointDecode(args, torch.device("cuda"))
    token, logits, elapsed = flow.run()
    print(
        "DSV4_CHECKPOINT_DECODE status=PASS "
        f"layers={args.layers} start_pos=0 token_id={args.token_id} "
        f"vocab={args.vocab_size} output_token={token} "
        f"logit_min={float(logits.float().min().item()):.6f} "
        f"logit_max={float(logits.float().max().item()):.6f} "
        f"elapsed_s={elapsed:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
