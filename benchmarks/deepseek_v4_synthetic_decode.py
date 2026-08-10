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
    SchedNvfp4Gemv,
    SchedRMS,
    SchedSmemSiLUInterleaved,
    SchedDsv4Fp8Quant128,
)


@dataclass
class Stage:
    name: str
    launcher: Launcher
    schedule: object

    def run(self, trace: bool = False) -> None:
        if trace:
            print(f"DSV4_E2E_STAGE name={self.name}", flush=True)
        self.launcher.launch(synchronize=False)


def build_stage(name: str, schedule, num_sms: int, device: torch.device) -> Stage:
    launcher = Launcher(num_sms, device=device)
    launcher.s(schedule.place(num_sms))
    # Keep the schedule alive: RawAddress instructions carry device pointers,
    # and several synthetic weights/views are owned only by the schedule.
    return Stage(name, launcher, schedule)


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

    def run(self, trace: bool) -> None:
        self.stage.run(trace)


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

    def run(self, trace: bool) -> None:
        self.stage.run(trace)


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

    def run(self, trace: bool) -> None:
        self.stage.run(trace)


class Nvfp4Linear:
    def __init__(
        self,
        name: str,
        factory: WeightFactory,
        activation: Nvfp4Activation,
        output: torch.Tensor,
        sms: int,
        device: torch.device,
    ):
        weight, weight_scale = factory.nvfp4(
            output.numel(), activation.quantized.numel() * 2
        )
        alpha = torch.tensor([1.0e-4], dtype=torch.float32, device=device)
        linear_sms = min(output.numel(), sms)
        self.stage = build_stage(
            f"{name}.gemv_nvfp4",
            SchedNvfp4Gemv(
                weight,
                weight_scale,
                activation.quantized,
                activation.scale,
                alpha,
                output.reshape(-1),
            ),
            linear_sms,
            device,
        )

    def run(self, trace: bool) -> None:
        self.stage.run(trace)


class SyntheticDecode:
    def __init__(self, args: argparse.Namespace, device: torch.device):
        self.args = args
        self.device = device
        self.config = DeepSeekV4FlashConfig()
        self.plans = build_decode_plan(args.start_pos, self.config)[: args.layers]
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

    def _build_hc_pair(self, prefix: str):
        project = self._stage(
            f"{prefix}.hc_project",
            SchedDsv4Fp32Bf16Gemv(
                self.hc_weight, self.residual.reshape(-1), self.mixes
            ),
            24,
        )
        pre = self._stage(
            f"{prefix}.hc_pre",
            SchedDsv4HcPre(
                self.residual,
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
                self.residual,
                self.post,
                self.comb,
                self.next_residual,
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
        ) = self._build_hc_pair("attn")

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
        self.kv_rope = torch.empty((1, cfg.head_dim), dtype=torch.bfloat16, device=d)
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
            values = torch.empty((projected,), dtype=torch.float32, device=d)
            scores = torch.empty_like(values)
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
            value_state = torch.zeros((rows, width), dtype=torch.float32, device=d)
            score_state = torch.zeros_like(value_state)
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
            rotated = torch.empty((1, width), dtype=torch.bfloat16, device=d)
            rope = self._stage(
                f"compress.r{ratio}.rope",
                SchedDsv4Rope512_64(
                    normalized.reshape(1, -1), self.rope_table, rotated
                ),
            )
            self.compress[ratio] = {
                "values": values,
                "scores": scores,
                "value_state": value_state,
                "score_state": score_state,
                "pooled": pooled,
                "rotated": rotated,
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

        projected = 2 * cfg.index_head_dim
        values = torch.empty((projected,), dtype=torch.float32, device=d)
        scores = torch.empty_like(values)
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
        value_state = torch.zeros((8, 128), dtype=torch.float32, device=d)
        score_state = torch.zeros_like(value_state)
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
        hadamard = torch.empty_like(rotated)
        rotate = self._stage(
            "index.compress_hadamard",
            SchedDsv4Hadamard(rotated, hadamard),
        )
        self.indexer = {
            "main": (index_q_linear, q_rope, q_hadamard, weights),
            "selection": (score, topk),
            "compress": (value_proj, score_proj, pool, norm, rope, rotate),
            "values": values,
            "scores": scores,
            "value_state": value_state,
            "score_state": score_state,
            "hadamard": hadamard,
        }

    def _build_ffn_path(self) -> None:
        cfg, d = self.config, self.device
        (
            self.hc_ffn_project,
            self.hc_ffn_pre,
            self.ffn_norm,
            self.hc_ffn_post,
        ) = self._build_hc_pair("ffn")
        self.ffn_hidden_fp8 = Fp8Activation("ffn.hidden", self.norm_hidden, self.sms, d)
        self.router_logits = torch.empty((256,), dtype=torch.bfloat16, device=d)
        self.router_projection = Fp8Linear(
            "ffn.router", self.factory, self.ffn_hidden_fp8, self.router_logits,
            self.sms, d
        )
        self.router_bias = torch.linspace(-0.5, 0.5, 256, dtype=torch.float32, device=d)
        self.hash_indices = torch.tensor(
            [3, 17, 29, 71, 130, 255], dtype=torch.int32, device=d
        )
        self.expert_indices = torch.empty((6,), dtype=torch.int32, device=d)
        self.expert_weights = torch.empty((6,), dtype=torch.float32, device=d)
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

        self.ffn_hidden_nvfp4 = Nvfp4Activation(
            "ffn.hidden", self.norm_hidden, self.sms, d
        )
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
                Nvfp4Linear(
                    f"ffn.expert{rank}.gate", self.factory,
                    self.ffn_hidden_nvfp4, self.routed_gate[rank], self.sms, d
                )
            )
            self.routed_up_linears.append(
                Nvfp4Linear(
                    f"ffn.expert{rank}.up", self.factory,
                    self.ffn_hidden_nvfp4, self.routed_up[rank], self.sms, d
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
                Nvfp4Linear(
                    f"ffn.expert{rank}.down", self.factory,
                    activation, self.routed_output[rank], self.sms, d
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
                self.expert_weights,
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

    def _check(self, name: str, *tensors: torch.Tensor) -> None:
        if not self.trace:
            return
        torch.cuda.synchronize()
        for tensor in tensors:
            finite = torch.isfinite(tensor)
            if not bool(finite.all().item()):
                bad = int((~finite).sum().item())
                raise AssertionError(
                    f"stage {name} produced {bad} non-finite values"
                )
        print(f"DSV4_E2E_CHECK name={name} status=finite", flush=True)

    def _run_compressor(self, ratio: int, should_compress: bool) -> None:
        data = self.compress[ratio]
        value_proj, score_proj, pool, norm, rope = data["stages"]
        value_proj.run(self.trace)
        score_proj.run(self.trace)
        projected_rows = 2 if ratio == 4 else 1
        data["value_state"][-projected_rows:].copy_(
            data["values"].reshape(projected_rows, -1)
        )
        data["score_state"][-projected_rows:].copy_(
            data["scores"].reshape(projected_rows, -1)
        )
        if not should_compress:
            return
        pool.run(self.trace)
        norm.run(self.trace)
        rope.run(self.trace)

    def _run_indexer(self, plan) -> None:
        if self.indexer is None:
            raise RuntimeError("CSA layer requested without an indexer")
        index_q, q_rope, q_hadamard, weights = self.indexer["main"]
        index_q.run(self.trace)
        q_rope.run(self.trace)
        q_hadamard.run(self.trace)
        weights.run(self.trace)
        if plan.should_compress:
            value_proj, score_proj, pool, norm, rope, rotate = self.indexer["compress"]
            value_proj.run(self.trace)
            score_proj.run(self.trace)
            self.indexer["value_state"][-2:].copy_(
                self.indexer["values"].reshape(2, -1)
            )
            self.indexer["score_state"][-2:].copy_(
                self.indexer["scores"].reshape(2, -1)
            )
            pool.run(self.trace)
            norm.run(self.trace)
            rope.run(self.trace)
            rotate.run(self.trace)
            self.index_cache[plan.compressed_rows - 1].copy_(
                self.indexer["hadamard"].reshape(-1)
            )
        score, topk = self.indexer["selection"]
        score.run(self.trace)
        topk.run(self.trace)

    def _run_attention(self, plan) -> None:
        self.hc_attn_project.run(self.trace)
        self.hc_attn_pre.run(self.trace)
        self.attn_norm.run(self.trace)
        self._check("attn_hc_norm", self.hidden, self.norm_hidden)
        self.hidden_fp8.run(self.trace)
        self.q_a.run(self.trace)
        self.q_norm.run(self.trace)
        self.q_rank_fp8.run(self.trace)
        self.q_b.run(self.trace)
        self.q_head_norm.run(self.trace)
        self.q_rope_stage.run(self.trace)
        self.kv_proj.run(self.trace)
        self.kv_norm.run(self.trace)
        self.kv_rope_stage.run(self.trace)
        self._check("attn_qkv", self.q_rope, self.kv_rope)
        self.kv_cache[self.args.start_pos % self.config.sliding_window].copy_(
            self.kv_rope.reshape(-1)
        )
        if plan.compress_ratio:
            self._run_compressor(plan.compress_ratio, plan.should_compress)
            if plan.should_compress:
                self.kv_cache[
                    self.config.sliding_window + plan.compressed_rows - 1
                ].copy_(self.compress[plan.compress_ratio]["rotated"].reshape(-1))
        if plan.attention_kind == "csa":
            self._run_indexer(plan)
        self.attention_stages[plan.attention_kind].run(self.trace)
        self.attention_inverse_stage.run(self.trace)
        self._check("attention_output", self.attention_inverse)
        for activation, linear in zip(
            self.o_group_activations, self.o_group_linears
        ):
            activation.run(self.trace)
            linear.run(self.trace)
        self.o_rank_fp8.run(self.trace)
        self.o_b.run(self.trace)
        self._check("attention_projection", self.branch)
        self.hc_attn_post.run(self.trace)
        self._check("attention_hc_post", self.next_residual)
        self.residual.copy_(self.next_residual)

    def _run_ffn(self, plan) -> None:
        self.hc_ffn_project.run(self.trace)
        self.hc_ffn_pre.run(self.trace)
        self.ffn_norm.run(self.trace)
        self._check("ffn_hc_norm", self.hidden, self.norm_hidden)
        self.ffn_hidden_fp8.run(self.trace)
        self.router_projection.run(self.trace)
        (self.router_hash if plan.hash_routing else self.router_score).run(self.trace)
        self.ffn_hidden_nvfp4.run(self.trace)
        for rank in range(6):
            self.routed_gate_linears[rank].run(self.trace)
            self.routed_up_linears[rank].run(self.trace)
            self._check(
                f"expert{rank}_projections",
                self.routed_gate[rank],
                self.routed_up[rank],
            )
            self.routed_swiglu[rank].run(self.trace)
            self._check(f"expert{rank}_swiglu", self.routed_middle[rank])
            self.routed_middle_quant[rank].run(self.trace)
            self.routed_down_linears[rank].run(self.trace)
            self._check(f"expert{rank}_down", self.routed_output[rank])
        self._check("routed_experts", self.routed_output)
        self.shared_gate_linear.run(self.trace)
        self.shared_up_linear.run(self.trace)
        self.shared_swiglu.run(self.trace)
        self.shared_middle_fp8.run(self.trace)
        self.shared_down_linear.run(self.trace)
        self._check("shared_expert", self.shared_output)
        self.expert_reduce.run(self.trace)
        self._check("expert_reduce", self.branch)
        self.hc_ffn_post.run(self.trace)
        self._check("ffn_hc_post", self.next_residual)
        self.residual.copy_(self.next_residual)

    def run_once(self) -> tuple[int, torch.Tensor]:
        self.reset()
        for plan in self.plans:
            self._run_attention(plan)
            self._run_ffn(plan)
        self.head_project.run(self.trace)
        self.head_hc.run(self.trace)
        self.head_norm_stage.run(self.trace)
        self.logits_stage.run(self.trace)
        torch.cuda.synchronize()
        if not bool(torch.isfinite(self.residual).all().item()):
            raise AssertionError("synthetic decode produced non-finite residual state")
        if not bool(torch.isfinite(self.logits).all().item()):
            raise AssertionError("synthetic decode produced non-finite logits")
        token = int(torch.argmax(self.logits).item())
        return token, self.residual.clone()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=43)
    parser.add_argument("--start-pos", type=int, default=127)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--sms", type=int, default=152)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--trace-stages", action="store_true")
    args = parser.parse_args()
    config = DeepSeekV4FlashConfig()
    if not 1 <= args.layers <= config.num_layers:
        parser.error("layers must be in [1,43]")
    if args.start_pos < 127:
        parser.error("the synthetic flow currently requires start-pos >= 127")
    if args.vocab_size <= 0 or args.vocab_size > config.vocab_size:
        parser.error("vocab-size must be in [1,129280]")
    if min(args.sms, args.iterations) <= 0 or args.warmup < 0:
        parser.error("sms/iterations must be positive and warmup non-negative")

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
    print(
        "DSV4_E2E_SYNTHETIC status=PASS "
        f"layers={args.layers} start_pos={args.start_pos} "
        f"swa={kinds['swa']} csa={kinds['csa']} hca={kinds['hca']} "
        f"vocab={args.vocab_size} token={reference_token} "
        f"build_s={build_seconds:.3f} min_ms={min(timings_ms):.6f} "
        f"median_ms={statistics.median(timings_ms):.6f} "
        f"max_ms={max(timings_ms):.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
