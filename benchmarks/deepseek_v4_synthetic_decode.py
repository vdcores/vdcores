#!/usr/bin/env python3
"""Synthetic checkpoint-shaped single-GPU DeepSeek-V4 decode flow.

The harness reuses one deterministic tensor per weight shape across layers.
It is intended to validate the complete task ordering and dataflow before a
168-GB checkpoint is introduced; it is not a model-quality or TBT benchmark.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import torch

from dae.deepseek_v4 import DeepSeekV4FlashConfig, decode_window_indices
from dae.deepseek_v4_flow import build_decode_plan
from dae.launcher import Launcher
from dae.routing import RoutedAddressTable
from dae.schedule import (
    SchedDsv4ExpertReduce,
    SchedDsv4Fp32Bf16Gemv,
    SchedDsv4GatedPool,
    SchedDsv4Hadamard,
    SchedDsv4HcHead,
    SchedDsv4HcPost,
    SchedDsv4HcPre,
    SchedDsv4IndexScore,
    SchedDsv4Nvfp4Quant16,
    SchedDsv4Rope128_64,
    SchedDsv4Rope512_64,
    SchedDsv4RouteTop6,
    SchedDsv4SparseAttention512,
    SchedDsv4TopK512,
    SchedFp8Block128Gemv,
    SchedRoutedNvfp4Gemv,
    SchedRMS,
    SchedSmemSiLUInterleaved,
    SchedDsv4Fp8Quant128,
)
from dae.sequential import SequentialProgram, SequentialStage


@dataclass
class Stage:
    name: str
    schedule: object
    num_sms: int
    input_role: str | None = None


def build_stage(
    name: str,
    schedule,
    num_sms: int,
    device: torch.device,
    *,
    input_role: str | None = None,
) -> Stage:
    del device
    return Stage(name, schedule, num_sms, input_role)


class WeightFactory:
    def __init__(self, device: torch.device):
        self.device = device
        self._fp8: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._nvfp4: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def fp8(self, m: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = (m, k)
        if key not in self._fp8:
            if m % 128 or k % 128:
                raise ValueError("synthetic FP8 weights require M/K multiples of 128")
            pattern = torch.linspace(
                -0.03125, 0.03125, k, dtype=torch.bfloat16, device=self.device
            ).to(torch.float8_e4m3fn)
            weight = torch.empty(
                (m, k), dtype=torch.float8_e4m3fn, device=self.device
            )
            weight.copy_(pattern)
            scale = torch.ones(
                (m // 128, k // 128),
                dtype=torch.float8_e8m0fnu,
                device=self.device,
            )
            self._fp8[key] = weight, scale
        return self._fp8[key]

    def nvfp4(self, m: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = (m, k)
        if key not in self._nvfp4:
            weight = torch.full(
                (m, k // 2), 0x22, dtype=torch.uint8, device=self.device
            )
            weight[1::2].fill_(0xAA)
            scale = torch.ones(
                (m, k // 16), dtype=torch.float8_e4m3fn, device=self.device
            )
            self._nvfp4[key] = weight, scale
        return self._nvfp4[key]


class Fp8Activation:
    def __init__(self, name: str, source: torch.Tensor, sms: int, device):
        self.quantized = torch.empty_like(source, dtype=torch.float8_e4m3fn)
        self.scale = torch.empty(
            (source.numel() // 128,),
            dtype=torch.float8_e8m0fnu,
            device=device,
        )
        quant_sms = min(source.numel() // 128, sms)
        self.stage = build_stage(
            f"{name}.quant_fp8",
            SchedDsv4Fp8Quant128(source, self.quantized, self.scale),
            quant_sms,
            device,
        )


class Fp8Linear:
    def __init__(
        self,
        name: str,
        factory: WeightFactory,
        activation: Fp8Activation,
        output: torch.Tensor,
        sms: int,
        device: torch.device,
    ):
        weight, weight_scale = factory.fp8(output.numel(), activation.quantized.numel())
        linear_sms = min(output.numel(), sms)
        self.stage = build_stage(
            f"{name}.gemv_fp8",
            SchedFp8Block128Gemv(
                weight,
                weight_scale,
                activation.quantized,
                activation.scale,
                output.reshape(-1),
            ),
            linear_sms,
            device,
        )


class Nvfp4Activation:
    def __init__(self, name: str, source: torch.Tensor, sms: int, device):
        self.global_scale = torch.tensor([0.01], dtype=torch.float32, device=device)
        self.quantized = torch.empty(
            (source.numel() // 2,), dtype=torch.uint8, device=device
        )
        self.scale = torch.empty(
            (source.numel() // 16,), dtype=torch.float8_e4m3fn, device=device
        )
        quant_sms = min(source.numel() // 16, sms)
        self.stage = build_stage(
            f"{name}.quant_nvfp4",
            SchedDsv4Nvfp4Quant16(
                source,
                self.global_scale,
                self.quantized,
                self.scale,
            ),
            quant_sms,
            device,
        )


class RoutedNvfp4Linear:
    def __init__(
        self,
        name: str,
        table: RoutedAddressTable,
        route_rank: int,
        weight_fields: list[int],
        weight_scale_fields: list[int],
        alpha_field: int,
        rows: int,
        k: int,
        activation: Nvfp4Activation,
        output: torch.Tensor,
        sms: int,
        device: torch.device,
    ):
        linear_sms = min(rows // 8, sms)
        self.stage = build_stage(
            f"{name}.gemv_nvfp4_routed",
            SchedRoutedNvfp4Gemv(
                table.state,
                route_rank,
                weight_fields,
                weight_scale_fields,
                alpha_field,
                rows,
                k,
                activation.quantized,
                activation.scale,
                output.reshape(-1),
            ),
            linear_sms,
            device,
            input_role="route",
        )


class SyntheticDecode:
    def __init__(self, args: argparse.Namespace, device: torch.device):
        self.args = args
        self.device = device
        self.config = DeepSeekV4FlashConfig()
        plan_end = args.first_layer + args.layers
        self.plans = build_decode_plan(args.start_pos, self.config)[
            args.first_layer:plan_end
        ]
        self.sms = min(
            args.sms,
            torch.cuda.get_device_properties(device).multi_processor_count,
        )
        self.factory = WeightFactory(device)
        self.trace = args.trace_stages
        self._allocate_state()
        self._build_attention_path()
        self._build_ffn_path()
        self._build_head()
        self._build_program()

    def _stage(self, name: str, schedule, sms: int = 1) -> Stage:
        return build_stage(name, schedule, sms, self.device)

    def _allocate_state(self) -> None:
        d = self.device
        cfg = self.config
        generator = torch.Generator(device=d).manual_seed(self.args.seed)
        self.initial_residual = torch.randn(
            (4, cfg.hidden_size),
            generator=generator,
            dtype=torch.bfloat16,
            device=d,
        ) * 0.01
        self.residual = self.initial_residual.clone()
        self.next_residual = torch.empty_like(self.residual)
        self.hidden = torch.empty((cfg.hidden_size,), dtype=torch.bfloat16, device=d)
        self.norm_hidden = torch.empty_like(self.hidden)
        self.branch = torch.empty_like(self.hidden)
        self.mixes = torch.empty((24,), dtype=torch.float32, device=d)
        self.post = torch.empty((4,), dtype=torch.float32, device=d)
        self.comb = torch.empty((4, 4), dtype=torch.float32, device=d)
        self.hc_weight = torch.zeros(
            (24, 4 * cfg.hidden_size), dtype=torch.float32, device=d
        )
        self.hc_scale = torch.ones((3,), dtype=torch.float32, device=d)
        self.hc_base = torch.zeros((24,), dtype=torch.float32, device=d)
        self.hidden_norm_weight = torch.ones(
            (cfg.hidden_size,), dtype=torch.bfloat16, device=d
        )

        angles = torch.linspace(-0.75, 0.75, 32, dtype=torch.float32, device=d)
        self.rope_table = torch.stack((angles.cos(), angles.sin()), dim=1)
        self.sink = torch.linspace(-0.25, 0.25, 64, dtype=torch.float32, device=d)

        max_compressed = max(
            (plan.compressed_rows for plan in self.plans), default=0
        )
        cache_rows = cfg.sliding_window + max(1, max_compressed)
        self.cache_seed = torch.randn(
            (cache_rows, cfg.head_dim),
            generator=generator,
            dtype=torch.bfloat16,
            device=d,
        ) * 0.01
        self.kv_cache = self.cache_seed.clone()
        index_rows = max(
            1,
            max(
                (plan.compressed_rows for plan in self.plans if plan.attention_kind == "csa"),
                default=0,
            ),
        )
        self.index_cache_seed = torch.randn(
            (index_rows, cfg.index_head_dim),
            generator=generator,
            dtype=torch.bfloat16,
            device=d,
        ) * 0.01
        self.index_cache = self.index_cache_seed.clone()

    def _build_hc_pair(
        self,
        prefix: str,
        residual: torch.Tensor,
        output_residual: torch.Tensor,
    ):
        project = self._stage(
            f"{prefix}.hc_project",
            SchedDsv4Fp32Bf16Gemv(
                self.hc_weight, residual.reshape(-1), self.mixes
            ),
            24,
        )
        pre = self._stage(
            f"{prefix}.hc_pre",
            SchedDsv4HcPre(
                residual,
                self.mixes,
                self.hc_scale,
                self.hc_base,
                self.hidden,
                self.post,
                self.comb,
            ),
        )
        norm = self._stage(
            f"{prefix}.rms4096",
            SchedRMS(
                1,
                self.config.rms_epsilon,
                self.hidden.reshape(1, -1),
                self.norm_hidden.reshape(1, -1),
                self.hidden_norm_weight,
            ),
        )
        post = self._stage(
            f"{prefix}.hc_post",
            SchedDsv4HcPost(
                self.branch,
                residual,
                self.post,
                self.comb,
                output_residual,
            ),
        )
        return project, pre, norm, post

    def _build_attention_path(self) -> None:
        cfg, d = self.config, self.device
        (
            self.hc_attn_project,
            self.hc_attn_pre,
            self.attn_norm,
            self.hc_attn_post,
        ) = self._build_hc_pair("attn", self.residual, self.next_residual)

        self.hidden_fp8 = Fp8Activation("attn.hidden", self.norm_hidden, self.sms, d)
        self.q_rank = torch.empty((cfg.q_lora_rank,), dtype=torch.bfloat16, device=d)
        self.q_a = Fp8Linear(
            "attn.q_a", self.factory, self.hidden_fp8, self.q_rank, self.sms, d
        )
        self.q_rank_norm = torch.empty_like(self.q_rank)
        self.q_rank_weight = torch.ones_like(self.q_rank)
        self.q_norm = self._stage(
            "attn.q_norm",
            SchedRMS(
                1,
                cfg.rms_epsilon,
                self.q_rank.reshape(1, -1),
                self.q_rank_norm.reshape(1, -1),
                self.q_rank_weight,
            ),
        )
        self.q_rank_fp8 = Fp8Activation("attn.q_rank", self.q_rank_norm, self.sms, d)
        self.q = torch.empty((cfg.num_heads, cfg.head_dim), dtype=torch.bfloat16, device=d)
        self.q_b = Fp8Linear(
            "attn.q_b", self.factory, self.q_rank_fp8, self.q, self.sms, d
        )
        self.q_normalized = torch.empty_like(self.q)
        self.q_head_weight = torch.ones((cfg.head_dim,), dtype=torch.bfloat16, device=d)
        self.q_head_norm = self._stage(
            "attn.q_head_norm",
            SchedRMS(
                cfg.num_heads,
                cfg.rms_epsilon,
                self.q,
                self.q_normalized,
                self.q_head_weight,
            ),
            cfg.num_heads,
        )
        self.q_rope = torch.empty_like(self.q)
        self.q_rope_stage = self._stage(
            "attn.q_rope",
            SchedDsv4Rope512_64(self.q_normalized, self.rope_table, self.q_rope),
            cfg.num_heads,
        )

        self.kv = torch.empty((cfg.head_dim,), dtype=torch.bfloat16, device=d)
        self.kv_proj = Fp8Linear(
            "attn.kv", self.factory, self.hidden_fp8, self.kv, self.sms, d
        )
        self.kv_normalized = torch.empty_like(self.kv)
        self.kv_weight = torch.ones_like(self.kv)
        self.kv_norm = self._stage(
            "attn.kv_norm",
            SchedRMS(
                1,
                cfg.rms_epsilon,
                self.kv.reshape(1, -1),
                self.kv_normalized.reshape(1, -1),
                self.kv_weight,
            ),
        )
        self.kv_rope = self.kv_cache[
            self.args.start_pos % cfg.sliding_window
        ].reshape(1, cfg.head_dim)
        self.kv_rope_stage = self._stage(
            "attn.kv_rope",
            SchedDsv4Rope512_64(
                self.kv_normalized.reshape(1, -1), self.rope_table, self.kv_rope
            ),
        )

        self._build_compressors()
        self.attention_output = torch.empty_like(self.q)
        self.attention_stages: dict[str, Stage] = {}
        for kind in ("swa", "csa", "hca"):
            matching = [plan for plan in self.plans if plan.attention_kind == kind]
            if not matching:
                continue
            plan = matching[0]
            window = decode_window_indices(self.args.start_pos).to(d)
            valid_window = window[window >= 0]
            if kind == "csa":
                self.csa_indices = torch.empty(
                    (plan.attention_candidates,), dtype=torch.int32, device=d
                )
                self.csa_indices[: valid_window.numel()].copy_(valid_window)
                indices = self.csa_indices
            else:
                compressed = torch.arange(
                    cfg.sliding_window,
                    cfg.sliding_window + plan.compressed_selected,
                    dtype=torch.int32,
                    device=d,
                )
                indices = torch.cat((valid_window, compressed))
            self.attention_stages[kind] = self._stage(
                f"attn.sparse_{kind}",
                SchedDsv4SparseAttention512(
                    self.q_rope,
                    self.kv_cache,
                    indices,
                    self.sink,
                    self.attention_output,
                ),
                cfg.num_heads,
            )
        self._build_indexer()

        self.attention_inverse = torch.empty_like(self.attention_output)
        self.attention_inverse_stage = self._stage(
            "attn.inverse_rope",
            SchedDsv4Rope512_64(
                self.attention_output,
                self.rope_table,
                self.attention_inverse,
                inverse=True,
            ),
            cfg.num_heads,
        )
        grouped = self.attention_inverse.reshape(cfg.o_groups, -1)
        self.o_rank = torch.empty(
            (cfg.o_groups, cfg.o_lora_rank), dtype=torch.bfloat16, device=d
        )
        self.o_group_activations = []
        self.o_group_linears = []
        for group in range(cfg.o_groups):
            activation = Fp8Activation(
                f"attn.o_a.g{group}", grouped[group], self.sms, d
            )
            linear = Fp8Linear(
                f"attn.o_a.g{group}",
                self.factory,
                activation,
                self.o_rank[group],
                self.sms,
                d,
            )
            self.o_group_activations.append(activation)
            self.o_group_linears.append(linear)
        self.o_rank_fp8 = Fp8Activation(
            "attn.o_rank", self.o_rank.reshape(-1), self.sms, d
        )
        self.o_b = Fp8Linear(
            "attn.o_b", self.factory, self.o_rank_fp8, self.branch, self.sms, d
        )

    def _build_compressors(self) -> None:
        cfg, d = self.config, self.device
        self.compress = {}
        for ratio, rows in ((4, 8), (128, 128)):
            width = cfg.head_dim
            coff = 2 if ratio == 4 else 1
            projected = coff * width
            value_state = torch.zeros((rows, width), dtype=torch.float32, device=d)
            score_state = torch.zeros_like(value_state)
            values = value_state[-coff:].reshape(projected)
            scores = score_state[-coff:].reshape(projected)
            weight = torch.zeros(
                (projected, cfg.hidden_size), dtype=torch.float32, device=d
            )
            value_proj = self._stage(
                f"compress.r{ratio}.value_proj",
                SchedDsv4Fp32Bf16Gemv(weight, self.norm_hidden, values),
                min(projected, self.sms),
            )
            score_proj = self._stage(
                f"compress.r{ratio}.score_proj",
                SchedDsv4Fp32Bf16Gemv(weight, self.norm_hidden, scores),
                min(projected, self.sms),
            )
            pooled = torch.empty((width,), dtype=torch.bfloat16, device=d)
            pool = self._stage(
                f"compress.r{ratio}.pool",
                SchedDsv4GatedPool(value_state, score_state, pooled),
            )
            normalized = torch.empty_like(pooled)
            norm = self._stage(
                f"compress.r{ratio}.norm",
                SchedRMS(
                    1,
                    cfg.rms_epsilon,
                    pooled.reshape(1, -1),
                    normalized.reshape(1, -1),
                    torch.ones_like(pooled),
                ),
            )
            compressed_rows = max(
                (
                    plan.compressed_rows
                    for plan in self.plans
                    if plan.compress_ratio == ratio
                ),
                default=0,
            )
            rotated = self.kv_cache[
                cfg.sliding_window + max(1, compressed_rows) - 1
            ].reshape(1, width)
            rope = self._stage(
                f"compress.r{ratio}.rope",
                SchedDsv4Rope512_64(
                    normalized.reshape(1, -1), self.rope_table, rotated
                ),
            )
            self.compress[ratio] = {
                "stages": (value_proj, score_proj, pool, norm, rope),
            }

    def _build_indexer(self) -> None:
        cfg, d = self.config, self.device
        csa_plans = [p for p in self.plans if p.attention_kind == "csa"]
        if not csa_plans:
            self.indexer = None
            return
        plan = csa_plans[0]
        self.index_q = torch.empty(
            (cfg.index_heads, cfg.index_head_dim), dtype=torch.bfloat16, device=d
        )
        index_q_linear = Fp8Linear(
            "index.q_b", self.factory, self.q_rank_fp8, self.index_q, self.sms, d
        )
        self.index_q_rope = torch.empty_like(self.index_q)
        q_rope = self._stage(
            "index.q_rope",
            SchedDsv4Rope128_64(
                self.index_q, self.rope_table, self.index_q_rope
            ),
            cfg.index_heads,
        )
        self.index_q_hadamard = torch.empty_like(self.index_q)
        q_hadamard = self._stage(
            "index.q_hadamard",
            SchedDsv4Hadamard(self.index_q_rope, self.index_q_hadamard),
            cfg.index_heads,
        )
        self.index_head_weights = torch.empty(
            (cfg.index_heads,), dtype=torch.float32, device=d
        )
        head_weight_matrix = torch.zeros(
            (cfg.index_heads, cfg.hidden_size), dtype=torch.float32, device=d
        )
        head_weight_matrix[:, 0] = torch.linspace(
            -0.01, 0.01, cfg.index_heads, dtype=torch.float32, device=d
        )
        weights = self._stage(
            "index.weights",
            SchedDsv4Fp32Bf16Gemv(
                head_weight_matrix, self.norm_hidden, self.index_head_weights
            ),
            cfg.index_heads,
        )
        self.index_scores = torch.empty(
            (plan.compressed_rows,), dtype=torch.float32, device=d
        )
        selection = ()
        if plan.compressed_rows:
            score = self._stage(
                "index.score",
                SchedDsv4IndexScore(
                    self.index_q_hadamard,
                    self.index_cache[: plan.compressed_rows],
                    self.index_head_weights,
                    self.index_scores,
                ),
                min(plan.compressed_rows, self.sms),
            )
            selected = self.csa_indices[-plan.compressed_selected :]
            topk = self._stage(
                "index.topk",
                SchedDsv4TopK512(
                    self.index_scores,
                    selected,
                    index_offset=cfg.sliding_window,
                ),
            )
            selection = (score, topk)

        projected = 2 * cfg.index_head_dim
        value_state = torch.zeros((8, 128), dtype=torch.float32, device=d)
        score_state = torch.zeros_like(value_state)
        values = value_state[-2:].reshape(projected)
        scores = score_state[-2:].reshape(projected)
        projection = torch.zeros(
            (projected, cfg.hidden_size), dtype=torch.float32, device=d
        )
        value_proj = self._stage(
            "index.compress_value_proj",
            SchedDsv4Fp32Bf16Gemv(projection, self.norm_hidden, values),
            min(projected, self.sms),
        )
        score_proj = self._stage(
            "index.compress_score_proj",
            SchedDsv4Fp32Bf16Gemv(projection, self.norm_hidden, scores),
            min(projected, self.sms),
        )
        pooled = torch.empty((128,), dtype=torch.bfloat16, device=d)
        pool = self._stage(
            "index.compress_pool",
            SchedDsv4GatedPool(value_state, score_state, pooled),
        )
        normalized = torch.empty_like(pooled)
        norm = self._stage(
            "index.compress_norm",
            SchedRMS(
                1,
                cfg.rms_epsilon,
                pooled.reshape(1, -1),
                normalized.reshape(1, -1),
                torch.ones_like(pooled),
            ),
        )
        rotated = torch.empty((1, 128), dtype=torch.bfloat16, device=d)
        rope = self._stage(
            "index.compress_rope",
            SchedDsv4Rope128_64(
                normalized.reshape(1, -1), self.rope_table, rotated
            ),
        )
        hadamard = self.index_cache[
            max(1, plan.compressed_rows) - 1
        ].reshape(1, cfg.index_head_dim)
        rotate = self._stage(
            "index.compress_hadamard",
            SchedDsv4Hadamard(rotated, hadamard),
        )
        self.indexer = {
            "main": (index_q_linear, q_rope, q_hadamard, weights),
            "selection": selection,
            "compress": (value_proj, score_proj, pool, norm, rope, rotate),
        }

    def _build_ffn_path(self) -> None:
        cfg, d = self.config, self.device
        (
            self.hc_ffn_project,
            self.hc_ffn_pre,
            self.ffn_norm,
            self.hc_ffn_post,
        ) = self._build_hc_pair("ffn", self.next_residual, self.residual)
        self.ffn_hidden_fp8 = Fp8Activation("ffn.hidden", self.norm_hidden, self.sms, d)
        self.router_logits = torch.empty((256,), dtype=torch.bfloat16, device=d)
        self.router_projection = Fp8Linear(
            "ffn.router", self.factory, self.ffn_hidden_fp8, self.router_logits,
            self.sms, d
        )
        self.router_bias = torch.linspace(-0.5, 0.5, 256, dtype=torch.float32, device=d)
        self.hash_indices = torch.zeros((8,), dtype=torch.int32, device=d)
        self.hash_indices[:6] = torch.tensor(
            [3, 17, 29, 71, 130, 255], dtype=torch.int32, device=d
        )

        self.ffn_hidden_nvfp4 = Nvfp4Activation(
            "ffn.hidden", self.norm_hidden, self.sms, d
        )
        routed_columns = {}

        def add_routed_linear(prefix: str, rows: int, k: int):
            weight, weight_scale = self.factory.nvfp4(rows, k)
            alpha = torch.zeros((4,), dtype=torch.float32, device=d)
            alpha[0] = 1.0e-4
            active_sms = min(rows // 8, self.sms)
            groups_per_sm, extra = divmod(rows // 8, active_sms)
            weight_names = []
            scale_names = []
            for sm in range(active_sms):
                group_start = sm * groups_per_sm + min(sm, extra)
                group_count = groups_per_sm + (1 if sm < extra else 0)
                row_start = group_start * 8
                row_stop = row_start + group_count * 8
                weight_name = f"{prefix}.weight.sm{sm}"
                scale_name = f"{prefix}.weight_scale.sm{sm}"
                routed_columns[weight_name] = [weight[row_start:row_stop]] * 256
                routed_columns[scale_name] = [
                    weight_scale[row_start:row_stop]
                ] * 256
                weight_names.append(weight_name)
                scale_names.append(scale_name)
            alpha_name = f"{prefix}.alpha"
            routed_columns[alpha_name] = [alpha] * 256
            return weight_names, scale_names, alpha_name

        gate_names = add_routed_linear("gate", 2048, 4096)
        up_names = add_routed_linear("up", 2048, 4096)
        down_names = add_routed_linear("down", 4096, 2048)
        self.routed_table = RoutedAddressTable(routed_columns)
        self.expert_indices = self.routed_table.route_indices_storage
        self.expert_weights = torch.empty((8,), dtype=torch.float32, device=d)
        self.router_hash = self._stage(
            "ffn.route_hash",
            SchedDsv4RouteTop6(
                self.router_logits,
                self.router_bias,
                self.hash_indices,
                self.expert_indices,
                self.expert_weights,
                hash_routing=True,
            ),
        )
        self.router_score = self._stage(
            "ffn.route_score",
            SchedDsv4RouteTop6(
                self.router_logits,
                self.router_bias,
                self.hash_indices,
                self.expert_indices,
                self.expert_weights,
                hash_routing=False,
            ),
        )

        def field_ids(names):
            weight_names, scale_names, alpha_name = names
            return (
                [self.routed_table.field(name) for name in weight_names],
                [self.routed_table.field(name) for name in scale_names],
                self.routed_table.field(alpha_name),
            )

        gate_fields = field_ids(gate_names)
        up_fields = field_ids(up_names)
        down_fields = field_ids(down_names)
        self.routed_gate = torch.empty((6, 2048), dtype=torch.bfloat16, device=d)
        self.routed_up = torch.empty_like(self.routed_gate)
        self.routed_middle = torch.empty_like(self.routed_gate)
        self.routed_output = torch.empty((6, 4096), dtype=torch.bfloat16, device=d)
        self.routed_gate_linears = []
        self.routed_up_linears = []
        self.routed_swiglu = []
        self.routed_middle_quant = []
        self.routed_down_linears = []
        for rank in range(6):
            self.routed_gate_linears.append(
                RoutedNvfp4Linear(
                    f"ffn.expert{rank}.gate",
                    self.routed_table,
                    rank,
                    *gate_fields,
                    2048,
                    4096,
                    self.ffn_hidden_nvfp4,
                    self.routed_gate[rank],
                    self.sms,
                    d,
                )
            )
            self.routed_up_linears.append(
                RoutedNvfp4Linear(
                    f"ffn.expert{rank}.up",
                    self.routed_table,
                    rank,
                    *up_fields,
                    2048,
                    4096,
                    self.ffn_hidden_nvfp4,
                    self.routed_up[rank],
                    self.sms,
                    d,
                )
            )
            self.routed_swiglu.append(
                self._stage(
                    f"ffn.expert{rank}.swiglu",
                    SchedSmemSiLUInterleaved(
                        1,
                        self.routed_gate[rank : rank + 1],
                        self.routed_up[rank : rank + 1],
                        self.routed_middle[rank : rank + 1],
                        swiglu_limit=cfg.swiglu_limit,
                    ),
                )
            )
            activation = Nvfp4Activation(
                f"ffn.expert{rank}.middle", self.routed_middle[rank], self.sms, d
            )
            self.routed_middle_quant.append(activation)
            self.routed_down_linears.append(
                RoutedNvfp4Linear(
                    f"ffn.expert{rank}.down",
                    self.routed_table,
                    rank,
                    *down_fields,
                    4096,
                    2048,
                    activation,
                    self.routed_output[rank],
                    self.sms,
                    d,
                )
            )

        self.shared_gate = torch.empty((2048,), dtype=torch.bfloat16, device=d)
        self.shared_up = torch.empty_like(self.shared_gate)
        self.shared_middle = torch.empty_like(self.shared_gate)
        self.shared_output = torch.empty((4096,), dtype=torch.bfloat16, device=d)
        self.shared_gate_linear = Fp8Linear(
            "ffn.shared.gate", self.factory, self.ffn_hidden_fp8,
            self.shared_gate, self.sms, d
        )
        self.shared_up_linear = Fp8Linear(
            "ffn.shared.up", self.factory, self.ffn_hidden_fp8,
            self.shared_up, self.sms, d
        )
        self.shared_swiglu = self._stage(
            "ffn.shared.swiglu",
            SchedSmemSiLUInterleaved(
                1,
                self.shared_gate.reshape(1, -1),
                self.shared_up.reshape(1, -1),
                self.shared_middle.reshape(1, -1),
                swiglu_limit=cfg.swiglu_limit,
            ),
        )
        self.shared_middle_fp8 = Fp8Activation(
            "ffn.shared.middle", self.shared_middle, self.sms, d
        )
        self.shared_down_linear = Fp8Linear(
            "ffn.shared.down", self.factory, self.shared_middle_fp8,
            self.shared_output, self.sms, d
        )
        self.expert_reduce = self._stage(
            "ffn.expert_reduce",
            SchedDsv4ExpertReduce(
                self.routed_output,
                self.expert_weights[:6],
                self.shared_output,
                self.branch,
            ),
        )

    def _build_head(self) -> None:
        cfg, d = self.config, self.device
        self.head_mixes = torch.empty((4,), dtype=torch.float32, device=d)
        self.head_project = self._stage(
            "head.hc_project",
            SchedDsv4Fp32Bf16Gemv(
                self.hc_weight[:4], self.residual.reshape(-1), self.head_mixes
            ),
            4,
        )
        self.head_hidden = torch.empty((cfg.hidden_size,), dtype=torch.bfloat16, device=d)
        self.head_hc = self._stage(
            "head.hc_reduce",
            SchedDsv4HcHead(
                self.residual,
                self.head_mixes,
                self.hc_scale[:1],
                self.hc_base[:4],
                self.head_hidden,
            ),
        )
        self.head_norm = torch.empty_like(self.head_hidden)
        self.head_norm_stage = self._stage(
            "head.rms4096",
            SchedRMS(
                1,
                cfg.rms_epsilon,
                self.head_hidden.reshape(1, -1),
                self.head_norm.reshape(1, -1),
                self.hidden_norm_weight,
            ),
        )
        self.logits = torch.empty(
            (self.args.vocab_size,), dtype=torch.float32, device=d
        )
        self.head_weight = torch.zeros(
            (self.args.vocab_size, cfg.hidden_size), dtype=torch.float32, device=d
        )
        self.head_weight[:, 0] = torch.linspace(
            -0.01, 0.01, self.args.vocab_size, dtype=torch.float32, device=d
        )
        self.logits_stage = self._stage(
            "head.logits_fp32",
            SchedDsv4Fp32Bf16Gemv(
                self.head_weight, self.head_norm, self.logits
            ),
            min(self.args.vocab_size, self.sms),
        )

    def reset(self) -> None:
        self.residual.copy_(self.initial_residual)
        self.kv_cache.copy_(self.cache_seed)
        self.index_cache.copy_(self.index_cache_seed)

    @staticmethod
    def _unwrap_stage(stage) -> Stage:
        return stage if isinstance(stage, Stage) else stage.stage

    def _compressor_stages(self, plan) -> list[Stage]:
        ratio = plan.compress_ratio
        data = self.compress[ratio]
        value_proj, score_proj, pool, norm, rope = data["stages"]
        stages = [value_proj, score_proj]
        if plan.should_compress:
            stages.extend((pool, norm, rope))
        return stages

    def _indexer_stages(self, plan) -> list[Stage]:
        if self.indexer is None:
            raise RuntimeError("CSA layer requested without an indexer")
        index_q, q_rope, q_hadamard, weights = self.indexer["main"]
        stages = [self._unwrap_stage(index_q), q_rope, q_hadamard, weights]
        if plan.should_compress:
            value_proj, score_proj, pool, norm, rope, rotate = self.indexer["compress"]
            stages.extend((value_proj, score_proj, pool, norm, rope, rotate))
        stages.extend(self.indexer["selection"])
        return stages

    def _attention_stages(self, plan) -> list[Stage]:
        stages = [
            self.hc_attn_project,
            self.hc_attn_pre,
            self.attn_norm,
            self.hidden_fp8.stage,
            self.q_a.stage,
            self.q_norm,
            self.q_rank_fp8.stage,
            self.q_b.stage,
            self.q_head_norm,
            self.q_rope_stage,
            self.kv_proj.stage,
            self.kv_norm,
            self.kv_rope_stage,
        ]
        if plan.compress_ratio:
            stages.extend(self._compressor_stages(plan))
        if plan.attention_kind == "csa":
            stages.extend(self._indexer_stages(plan))
        stages.extend(
            (
                self.attention_stages[plan.attention_kind],
                self.attention_inverse_stage,
            )
        )
        for activation, linear in zip(
            self.o_group_activations, self.o_group_linears
        ):
            stages.extend((activation.stage, linear.stage))
        stages.extend(
            (
                self.o_rank_fp8.stage,
                self.o_b.stage,
                self.hc_attn_post,
            )
        )
        return stages

    def _ffn_stages(self, plan) -> list[Stage]:
        stages = [
            self.hc_ffn_project,
            self.hc_ffn_pre,
            self.ffn_norm,
            self.ffn_hidden_fp8.stage,
            self.router_projection.stage,
            self.router_hash if plan.hash_routing else self.router_score,
            self.ffn_hidden_nvfp4.stage,
        ]
        for rank in range(6):
            stages.extend(
                (
                    self.routed_gate_linears[rank].stage,
                    self.routed_up_linears[rank].stage,
                    self.routed_swiglu[rank],
                    self.routed_middle_quant[rank].stage,
                    self.routed_down_linears[rank].stage,
                )
            )
        stages.extend(
            (
                self.shared_gate_linear.stage,
                self.shared_up_linear.stage,
                self.shared_swiglu,
                self.shared_middle_fp8.stage,
                self.shared_down_linear.stage,
                self.expert_reduce,
                self.hc_ffn_post,
            )
        )
        return stages

    def _build_program(self) -> None:
        stages = []
        serial_sm = 0

        def append_stage(name: str, stage: Stage) -> None:
            nonlocal serial_sm
            base_sm = 0
            if stage.num_sms == 1:
                base_sm = serial_sm
                serial_sm = (serial_sm + 1) % self.sms
            stages.append(
                SequentialStage(
                    name,
                    stage.schedule,
                    stage.num_sms,
                    base_sm=base_sm,
                    input_role=stage.input_role,
                )
            )

        for plan in self.plans:
            for stage in self._attention_stages(plan) + self._ffn_stages(plan):
                append_stage(f"layer{plan.layer_id}.{stage.name}", stage)
        for stage in (
            self.head_project,
            self.head_hc,
            self.head_norm_stage,
            self.logits_stage,
        ):
            append_stage(stage.name, stage)
        if self.args.max_stages:
            stages = stages[: self.args.max_stages]
        self.partial_program = len(stages) == 0 or stages[-1].name != "head.logits_fp32"
        self.launcher = Launcher(self.sms, device=self.device)
        self.program = SequentialProgram(self.launcher, stages)
        self.launcher.s(self.program)
        if self.trace:
            for stage in stages:
                print(f"DSV4_E2E_STAGE name={stage.name}", flush=True)
        print(
            "DSV4_E2E_PROGRAM "
            f"launches=1 stages={len(stages)} barriers={len(self.program.barriers)} "
            f"compute_insts={self.program.max_compute_instructions} "
            f"memory_insts={self.program.max_memory_instructions}",
            flush=True,
        )

    def run_once(self) -> tuple[int, torch.Tensor]:
        self.reset()
        self.launcher.launch(synchronize=False)
        torch.cuda.synchronize()
        if self.trace and hasattr(self, "csa_indices"):
            print(
                "DSV4_E2E_CSA_INDICES "
                f"min={int(self.csa_indices.min().item())} "
                f"max={int(self.csa_indices.max().item())} "
                f"count={self.csa_indices.numel()}",
                flush=True,
            )
        if not bool(torch.isfinite(self.residual).all().item()):
            raise AssertionError("synthetic decode produced non-finite residual state")
        if self.partial_program:
            return -1, self.residual.clone()
        if not bool(torch.isfinite(self.logits).all().item()):
            raise AssertionError("synthetic decode produced non-finite logits")
        token = int(torch.argmax(self.logits).item())
        return token, self.residual.clone()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--first-layer", type=int, default=0)
    parser.add_argument("--start-pos", type=int, default=127)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--sms", type=int, default=152)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--trace-stages", action="store_true")
    parser.add_argument(
        "--max-stages",
        type=int,
        default=0,
        help="diagnostic: launch only this prefix of the flattened program",
    )
    args = parser.parse_args()
    config = DeepSeekV4FlashConfig()
    if not 1 <= args.layers <= config.num_layers:
        parser.error("layers must be in [1,43]")
    if (
        args.first_layer < 0
        or args.first_layer + args.layers > config.num_layers
    ):
        parser.error("first-layer plus layers must stay inside [0,43)")
    if args.start_pos < 0:
        parser.error("start-pos must be non-negative")
    if args.vocab_size <= 0 or args.vocab_size > config.vocab_size:
        parser.error("vocab-size must be in [1,129280]")
    if min(args.sms, args.iterations) <= 0 or args.warmup < 0:
        parser.error("sms/iterations must be positive and warmup non-negative")
    if args.max_stages < 0:
        parser.error("max-stages must be non-negative")

    device = torch.device("cuda")
    started = time.monotonic()
    flow = SyntheticDecode(args, device)
    torch.cuda.synchronize()
    build_seconds = time.monotonic() - started
    for _ in range(args.warmup):
        flow.run_once()

    timings_ms = []
    reference_token = None
    reference_residual = None
    for _ in range(args.iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        token, residual = flow.run_once()
        end.record()
        end.synchronize()
        timings_ms.append(start.elapsed_time(end))
        if reference_token is None:
            reference_token = token
            reference_residual = residual
        else:
            if token != reference_token:
                raise AssertionError("synthetic decode token is not repeatable")
            torch.testing.assert_close(
                residual, reference_residual, rtol=0, atol=0
            )

    kinds = {kind: 0 for kind in ("swa", "csa", "hca")}
    for plan in flow.plans:
        kinds[plan.attention_kind] += 1
    result_name = (
        "DSV4_E2E_PREFIX" if flow.partial_program else "DSV4_E2E_SYNTHETIC"
    )
    print(
        f"{result_name} status=PASS launches=1 "
        f"first_layer={args.first_layer} layers={args.layers} "
        f"start_pos={args.start_pos} "
        f"swa={kinds['swa']} csa={kinds['csa']} hca={kinds['hca']} "
        f"vocab={args.vocab_size} token={reference_token} "
        f"build_s={build_seconds:.3f} min_ms={min(timings_ms):.6f} "
        f"median_ms={statistics.median(timings_ms):.6f} "
        f"max_ms={max(timings_ms):.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
