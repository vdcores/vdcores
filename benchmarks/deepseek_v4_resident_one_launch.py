#!/usr/bin/env python3
"""Checkpoint-resident DeepSeek-V4 position-0 decode in one VDCores launch.

The full model is represented by four shape families: layers 0-1, layer 2,
odd HCA layers, and even CSA layers.  Repeated families use runtime loop
counters to select resident layer weights; routed expert IDs stay in one fixed
HBM buffer and LDU resolves the selected expert and current layer.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import torch

from dae.deepseek_v4 import DeepSeekV4FlashConfig, deepseek_v4_rope_table
from dae.deepseek_v4_checkpoint import (
    DeepSeekV4Checkpoint,
    DeepSeekV4ResidentCheckpoint,
    expected_inference_tensor_specs,
)
from dae.deepseek_v4_schedule import DeepSeekV4ShapePolicy, ShapeAssignment
from dae.launcher import Launcher
from dae.routing import RoutedAddressTable
from dae.schedule import (
    LayeredSchedule,
    SchedDsv4Bf16Gemv,
    SchedDsv4ExpertReduce,
    SchedDsv4Fp32Bf16Gemv,
    SchedDsv4Fp8Quant128,
    SchedDsv4Hadamard,
    SchedDsv4HcHead,
    SchedDsv4HcPost,
    SchedDsv4HcPre,
    SchedDsv4Rope128_64,
    SchedDsv4Rope512_64,
    SchedDsv4RouteTop6,
    SchedDsv4SparseAttention512,
    SchedFp8Block128Gemv,
    SchedRMS,
    SchedRoutedDsv4Nvfp4Quant16,
    SchedRoutedNvfp4Gemv,
    SchedSmemSiLUInterleaved,
)
from dae.sequential import (
    LoopedSequentialProgram,
    SequentialBlock,
    SequentialProgram,
    SequentialStage,
)


@dataclass(frozen=True)
class LayerFamily:
    name: str
    layer_ids: tuple[int, ...]
    counter_strides: tuple[tuple[int, int], ...] = ()

    @property
    def representative(self) -> int:
        return self.layer_ids[0]


@dataclass(frozen=True)
class Stage:
    name: str
    schedule: object
    num_sms: int
    input_role: str | None = None


class ResidentOneLaunchDecode:
    def __init__(self, args: argparse.Namespace, device: torch.device):
        self.args = args
        self.device = device
        self.config = DeepSeekV4FlashConfig()
        self.sms = min(
            args.sms,
            torch.cuda.get_device_properties(device).multi_processor_count,
        )
        self.policy = DeepSeekV4ShapePolicy(self.sms)
        self.assignments: dict[tuple, ShapeAssignment] = {}
        self.checkpoint = self._load_checkpoint()
        self.families = self._families()
        self._routing_tables: dict[int, RoutedAddressTable] = {}
        self._routing_owners: dict[int, tuple[torch.Tensor, ...]] = {}
        self._hash_rows: dict[int, torch.Tensor] = {}
        self._allocate_state()
        self.family_stages = {
            family.representative: self._build_family(family)
            for family in self.families
        }
        self.head_stages = self._build_head()
        self._build_program()

    def _load_checkpoint(self) -> DeepSeekV4ResidentCheckpoint:
        disk = DeepSeekV4Checkpoint(self.args.checkpoint, self.config)
        names = None
        if self.args.layers != self.config.num_layers:
            prefix = tuple(f"layers.{layer_id}." for layer_id in range(self.args.layers))
            names = tuple(
                name
                for name in expected_inference_tensor_specs(self.config)
                if not name.startswith("layers.") or name.startswith(prefix)
            )
        started = time.monotonic()
        print(
            "DSV4_ONE_LAUNCH_RESIDENT status=START "
            f"checkpoint={self.args.checkpoint} layers={self.args.layers}",
            flush=True,
        )

        def progress(index, count, filename, shard_bytes, loaded_bytes):
            print(
                "DSV4_ONE_LAUNCH_SHARD status=PASS "
                f"shard={index}/{count} filename={filename} "
                f"storage_gib={shard_bytes / (1 << 30):.3f} "
                f"loaded_gib={loaded_bytes / (1 << 30):.3f}",
                flush=True,
            )

        resident = DeepSeekV4ResidentCheckpoint.from_checkpoint(
            disk,
            device=self.device,
            names=names,
            reserve_bytes=int(self.args.resident_reserve_gib * (1 << 30)),
            progress=progress,
        )
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        print(
            "DSV4_ONE_LAUNCH_RESIDENT status=PASS "
            f"tensor_gib={resident.tensor_bytes / (1 << 30):.3f} "
            f"storage_gib={resident.storage_bytes / (1 << 30):.3f} "
            f"free_gib={free_bytes / (1 << 30):.3f} "
            f"total_gib={total_bytes / (1 << 30):.3f} "
            f"elapsed_s={time.monotonic() - started:.3f}",
            flush=True,
        )
        return resident

    def _families(self) -> tuple[LayerFamily, ...]:
        if self.args.layers == 1:
            return (LayerFamily("layer0.swa_hash", (0,)),)
        if self.args.layers == 2:
            return (LayerFamily("layers0-1.swa_hash", (0, 1), ((0, 1),)),)
        if self.args.layers != self.config.num_layers:
            raise ValueError("one-launch resident flow supports 1, 2, or 43 layers")
        return (
            LayerFamily("layers0-1.swa_hash", (0, 1), ((0, 1),)),
            LayerFamily("layer2.csa_hash", (2,)),
            LayerFamily(
                "layers3-41.hca_score",
                tuple(range(3, 43, 2)),
                ((0, 1), (1, 2)),
            ),
            LayerFamily(
                "layers4-42.csa_score",
                tuple(range(4, 43, 2)),
                ((0, 1), (1, 2)),
            ),
        )

    def _tensor(self, name: str) -> torch.Tensor:
        return self.checkpoint.load_tensors(
            (name,), device=self.device
        )[name]

    def _family_tensors(
        self, family: LayerFamily, suffix: str
    ) -> tuple[torch.Tensor, ...]:
        return tuple(
            self._tensor(f"layers.{layer_id}.{suffix}")
            for layer_id in family.layer_ids
        )

    def _remember(self, assignment: ShapeAssignment) -> int:
        key = (
            assignment.task,
            assignment.rows,
            assignment.k,
            assignment.num_sms,
            assignment.row_alignment,
            assignment.tile_rows,
            assignment.tile_k,
        )
        self.assignments[key] = assignment
        return assignment.num_sms

    def _stage(
        self,
        name: str,
        schedule,
        sms: int | ShapeAssignment = 1,
        *,
        input_role: str | None = None,
    ) -> Stage:
        if isinstance(sms, ShapeAssignment):
            sms = self._remember(sms)
        return Stage(name, schedule, int(sms), input_role)

    @staticmethod
    def _groups(*tensor_sets: tuple[torch.Tensor, ...]):
        return tuple((tensors[0], tensors) for tensors in tensor_sets)

    def _layered(
        self,
        schedule,
        family: LayerFamily,
        *tensor_sets: tuple[torch.Tensor, ...],
    ):
        if len(family.layer_ids) == 1 or not tensor_sets:
            return schedule
        return LayeredSchedule(
            schedule,
            self._groups(*tensor_sets),
            counter_strides=family.counter_strides,
        )

    def _routed_layered(
        self,
        schedule,
        family: LayerFamily,
        tables: tuple[RoutedAddressTable, ...],
    ):
        return LayeredSchedule(
            schedule,
            ((tables[0].state, tuple(table.state for table in tables)),),
            counter_strides=family.counter_strides,
            route_indices=self.route_indices,
        )

    def _allocate_state(self) -> None:
        cfg, d = self.config, self.device
        embedding = self._tensor("embed.weight")[self.args.token_id]
        self.initial_residual = embedding.reshape(1, -1).repeat(cfg.hc_mult, 1)
        self.residual = torch.empty_like(self.initial_residual)
        self.next_residual = torch.empty_like(self.residual)
        self.hidden = torch.empty((cfg.hidden_size,), dtype=torch.bfloat16, device=d)
        self.norm_hidden = torch.empty_like(self.hidden)
        self.branch = torch.empty_like(self.hidden)
        self.mixes = torch.empty((24,), dtype=torch.float32, device=d)
        self.post = torch.empty((4,), dtype=torch.float32, device=d)
        self.comb = torch.empty((4, 4), dtype=torch.float32, device=d)

        self.main_rope = deepseek_v4_rope_table(0, config=cfg, device=d)
        self.compress_rope = deepseek_v4_rope_table(
            0, compressed=True, config=cfg, device=d
        )
        self.kv_row = torch.empty((1, cfg.head_dim), dtype=torch.bfloat16, device=d)
        self.attention_indices = torch.zeros((1,), dtype=torch.int32, device=d)

        self.hidden_fp8 = torch.empty(
            (cfg.hidden_size,), dtype=torch.float8_e4m3fn, device=d
        )
        self.hidden_fp8_scale = torch.empty(
            (cfg.hidden_size // 128,), dtype=torch.float8_e8m0fnu, device=d
        )
        self.q_rank = torch.empty((cfg.q_lora_rank,), dtype=torch.bfloat16, device=d)
        self.q_rank_norm = torch.empty_like(self.q_rank)
        self.q_rank_fp8 = torch.empty_like(self.q_rank, dtype=torch.float8_e4m3fn)
        self.q_rank_fp8_scale = torch.empty(
            (cfg.q_lora_rank // 128,), dtype=torch.float8_e8m0fnu, device=d
        )
        self.q = torch.empty(
            (cfg.num_heads, cfg.head_dim), dtype=torch.bfloat16, device=d
        )
        self.q_norm = torch.empty_like(self.q)
        self.q_rope = torch.empty_like(self.q)
        self.kv = torch.empty((cfg.head_dim,), dtype=torch.bfloat16, device=d)
        self.kv_norm = torch.empty_like(self.kv)
        self.attention_output = torch.empty_like(self.q)
        self.attention_inverse = torch.empty_like(self.q)
        group_width = cfg.num_heads * cfg.head_dim // cfg.o_groups
        self.o_group_fp8 = torch.empty(
            (cfg.o_groups, group_width), dtype=torch.float8_e4m3fn, device=d
        )
        self.o_group_scale = torch.empty(
            (cfg.o_groups, group_width // 128),
            dtype=torch.float8_e8m0fnu,
            device=d,
        )
        self.o_rank = torch.empty(
            (cfg.o_groups, cfg.o_lora_rank), dtype=torch.bfloat16, device=d
        )
        self.o_rank_fp8 = torch.empty_like(
            self.o_rank.reshape(-1), dtype=torch.float8_e4m3fn
        )
        self.o_rank_scale = torch.empty(
            (self.o_rank.numel() // 128,), dtype=torch.float8_e8m0fnu, device=d
        )

        self.compress_values = torch.empty((1024,), dtype=torch.float32, device=d)
        self.compress_scores = torch.empty_like(self.compress_values)
        self.index_q = torch.empty(
            (cfg.index_heads, cfg.index_head_dim), dtype=torch.bfloat16, device=d
        )
        self.index_q_rope = torch.empty_like(self.index_q)
        self.index_q_hadamard = torch.empty_like(self.index_q)
        self.index_head_weights = torch.empty(
            (cfg.index_heads,), dtype=torch.float32, device=d
        )
        self.index_compress_values = torch.empty(
            (2 * cfg.index_head_dim,), dtype=torch.float32, device=d
        )
        self.index_compress_scores = torch.empty_like(self.index_compress_values)

        self.router_logits = torch.empty(
            (cfg.num_experts,), dtype=torch.bfloat16, device=d
        )
        self.route_indices = torch.empty((8,), dtype=torch.int32, device=d)
        self.route_weights = torch.empty((8,), dtype=torch.float32, device=d)
        self.zero_bias = torch.zeros(
            (cfg.num_experts,), dtype=torch.float32, device=d
        )
        self.zero_hash = torch.zeros((8,), dtype=torch.int32, device=d)

        self.routed_input = torch.empty(
            (cfg.experts_per_token, cfg.hidden_size // 2),
            dtype=torch.uint8,
            device=d,
        )
        self.routed_input_scale = torch.empty(
            (cfg.experts_per_token, cfg.hidden_size // 16),
            dtype=torch.float8_e4m3fn,
            device=d,
        )
        self.routed_gate = torch.empty(
            (cfg.experts_per_token, cfg.expert_intermediate_size),
            dtype=torch.bfloat16,
            device=d,
        )
        self.routed_up = torch.empty_like(self.routed_gate)
        self.routed_middle = torch.empty_like(self.routed_gate)
        self.routed_middle_packed = torch.empty(
            (cfg.experts_per_token, cfg.expert_intermediate_size // 2),
            dtype=torch.uint8,
            device=d,
        )
        self.routed_middle_scale = torch.empty(
            (cfg.experts_per_token, cfg.expert_intermediate_size // 16),
            dtype=torch.float8_e4m3fn,
            device=d,
        )
        self.routed_output = torch.empty(
            (cfg.experts_per_token, cfg.hidden_size),
            dtype=torch.bfloat16,
            device=d,
        )
        self.shared_gate = torch.empty(
            (cfg.expert_intermediate_size,), dtype=torch.bfloat16, device=d
        )
        self.shared_up = torch.empty_like(self.shared_gate)
        self.shared_middle = torch.empty_like(self.shared_gate)
        self.shared_middle_fp8 = torch.empty_like(
            self.shared_middle, dtype=torch.float8_e4m3fn
        )
        self.shared_middle_scale = torch.empty(
            (cfg.expert_intermediate_size // 128,),
            dtype=torch.float8_e8m0fnu,
            device=d,
        )
        self.shared_output = torch.empty(
            (cfg.hidden_size,), dtype=torch.bfloat16, device=d
        )

    def _fp8_quant_stage(
        self,
        name: str,
        source: torch.Tensor,
        output: torch.Tensor,
        scale: torch.Tensor,
    ) -> Stage:
        assignment = self.policy.quantize(source.numel(), 128)
        return self._stage(
            name,
            SchedDsv4Fp8Quant128(source.reshape(-1), output.reshape(-1), scale),
            assignment,
        )

    def _fp8_linear_stage(
        self,
        name: str,
        family: LayerFamily,
        suffix: str,
        activation: torch.Tensor,
        activation_scale: torch.Tensor,
        output: torch.Tensor,
        *,
        row_slice: slice | None = None,
    ) -> Stage:
        linears = tuple(
            self.checkpoint.load_fp8_linear(
                f"layers.{layer_id}.{suffix}", device=self.device
            )
            for layer_id in family.layer_ids
        )
        if row_slice is None:
            weights = tuple(linear.weight for linear in linears)
            scales = tuple(linear.scale for linear in linears)
        else:
            start = 0 if row_slice.start is None else row_slice.start
            stop = linears[0].weight.shape[0] if row_slice.stop is None else row_slice.stop
            if start % 128 or stop % 128:
                raise ValueError("FP8 family slices must be 128-row aligned")
            weights = tuple(linear.weight[row_slice] for linear in linears)
            scales = tuple(
                linear.scale[start // 128 : stop // 128] for linear in linears
            )
        schedule = SchedFp8Block128Gemv(
            weights[0],
            scales[0],
            activation.reshape(-1),
            activation_scale.reshape(-1),
            output.reshape(-1),
        )
        schedule = self._layered(schedule, family, weights, scales)
        assignment = self.policy.fp8_gemv(output.numel(), activation.numel())
        return self._stage(name, schedule, assignment)

    def _bf16_linear_stage(
        self,
        name: str,
        family: LayerFamily,
        suffix: str,
        source: torch.Tensor,
        output: torch.Tensor,
    ) -> Stage:
        weights = self._family_tensors(family, suffix)
        schedule = SchedDsv4Bf16Gemv(
            weights[0], source.reshape(-1), output.reshape(-1)
        )
        schedule = self._layered(schedule, family, weights)
        assignment = self.policy.bf16_gemv(output.numel(), source.numel())
        return self._stage(name, schedule, assignment)

    def _rms_stage(
        self,
        name: str,
        source: torch.Tensor,
        output: torch.Tensor,
        *,
        family: LayerFamily | None = None,
        weight_suffix: str | None = None,
    ) -> Stage:
        rows = source.reshape(-1, source.shape[-1])
        out_rows = output.reshape_as(rows)
        weights = None
        weight = None
        if weight_suffix is not None:
            if family is None:
                weight = self._tensor(weight_suffix)
            else:
                weights = self._family_tensors(family, weight_suffix)
                weight = weights[0]
        schedule = SchedRMS(
            rows.shape[0],
            self.config.rms_epsilon,
            rows,
            out_rows,
            weight,
            hidden_size=rows.shape[1],
        )
        if weights is not None:
            schedule = self._layered(schedule, family, weights)
        return self._stage(name, schedule, rows.shape[0])

    def _hc_stages(
        self,
        family: LayerFamily,
        branch_name: str,
        residual: torch.Tensor,
        output_residual: torch.Tensor,
    ) -> tuple[list[Stage], Stage]:
        functions = self._family_tensors(family, f"hc_{branch_name}_fn")
        scales = self._family_tensors(family, f"hc_{branch_name}_scale")
        bases = self._family_tensors(family, f"hc_{branch_name}_base")
        project = SchedDsv4Fp32Bf16Gemv(
            functions[0], residual.reshape(-1), self.mixes
        )
        project = self._layered(project, family, functions)
        project_stage = self._stage(
            f"{branch_name}.hc_project",
            project,
            self.policy.fp32_bf16_gemv(24, residual.numel()),
        )
        pre = SchedDsv4HcPre(
            residual,
            self.mixes,
            scales[0],
            bases[0],
            self.hidden,
            self.post,
            self.comb,
        )
        pre = self._layered(pre, family, scales, bases)
        pre_stage = self._stage(f"{branch_name}.hc_pre", pre)
        norm_stage = self._rms_stage(
            f"{branch_name}.rms4096",
            self.hidden,
            self.norm_hidden,
            family=family,
            weight_suffix=f"{branch_name}_norm.weight",
        )
        post_stage = self._stage(
            f"{branch_name}.hc_post",
            SchedDsv4HcPost(
                self.branch,
                residual,
                self.post,
                self.comb,
                output_residual,
            ),
        )
        return [project_stage, pre_stage, norm_stage], post_stage

    def _build_attention(self, family: LayerFamily) -> list[Stage]:
        cfg = self.config
        layer_id = family.representative
        kind = cfg.attention_kind(layer_id)
        stages, post = self._hc_stages(
            family, "attn", self.residual, self.next_residual
        )
        stages.append(
            self._fp8_quant_stage(
                "attn.hidden.quant_fp8",
                self.norm_hidden,
                self.hidden_fp8,
                self.hidden_fp8_scale,
            )
        )
        stages.append(
            self._fp8_linear_stage(
                "attn.q_a",
                family,
                "attn.wq_a",
                self.hidden_fp8,
                self.hidden_fp8_scale,
                self.q_rank,
            )
        )
        stages.append(
            self._rms_stage(
                "attn.q_norm",
                self.q_rank,
                self.q_rank_norm,
                family=family,
                weight_suffix="attn.q_norm.weight",
            )
        )
        stages.append(
            self._fp8_quant_stage(
                "attn.q_rank.quant_fp8",
                self.q_rank_norm,
                self.q_rank_fp8,
                self.q_rank_fp8_scale,
            )
        )
        stages.append(
            self._fp8_linear_stage(
                "attn.q_b",
                family,
                "attn.wq_b",
                self.q_rank_fp8,
                self.q_rank_fp8_scale,
                self.q,
            )
        )
        stages.append(self._rms_stage("attn.q_head_norm", self.q, self.q_norm))
        rope_table = self.main_rope if kind == "swa" else self.compress_rope
        stages.append(
            self._stage(
                "attn.q_rope",
                SchedDsv4Rope512_64(self.q_norm, rope_table, self.q_rope),
                self.policy.attention(cfg.num_heads, cfg.head_dim),
            )
        )
        stages.append(
            self._fp8_linear_stage(
                "attn.kv",
                family,
                "attn.wkv",
                self.hidden_fp8,
                self.hidden_fp8_scale,
                self.kv,
            )
        )
        stages.append(
            self._rms_stage(
                "attn.kv_norm",
                self.kv,
                self.kv_norm,
                family=family,
                weight_suffix="attn.kv_norm.weight",
            )
        )
        stages.append(
            self._stage(
                "attn.kv_rope",
                SchedDsv4Rope512_64(
                    self.kv_norm.reshape(1, -1), rope_table, self.kv_row
                ),
            )
        )

        if kind in ("csa", "hca"):
            width = cfg.head_dim * (2 if kind == "csa" else 1)
            stages.append(
                self._bf16_linear_stage(
                    "attn.compressor.wkv",
                    family,
                    "attn.compressor.wkv.weight",
                    self.norm_hidden,
                    self.compress_values[:width],
                )
            )
            stages.append(
                self._bf16_linear_stage(
                    "attn.compressor.wgate",
                    family,
                    "attn.compressor.wgate.weight",
                    self.norm_hidden,
                    self.compress_scores[:width],
                )
            )

        if kind == "csa":
            stages.append(
                self._fp8_linear_stage(
                    "index.q_b",
                    family,
                    "attn.indexer.wq_b",
                    self.q_rank_fp8,
                    self.q_rank_fp8_scale,
                    self.index_q,
                )
            )
            stages.append(
                self._stage(
                    "index.q_rope",
                    SchedDsv4Rope128_64(
                        self.index_q, self.compress_rope, self.index_q_rope
                    ),
                    cfg.index_heads,
                )
            )
            stages.append(
                self._stage(
                    "index.q_hadamard",
                    SchedDsv4Hadamard(
                        self.index_q_rope, self.index_q_hadamard
                    ),
                    cfg.index_heads,
                )
            )
            stages.append(
                self._bf16_linear_stage(
                    "index.weights",
                    family,
                    "attn.indexer.weights_proj.weight",
                    self.norm_hidden,
                    self.index_head_weights,
                )
            )
            stages.append(
                self._bf16_linear_stage(
                    "index.compressor.wkv",
                    family,
                    "attn.indexer.compressor.wkv.weight",
                    self.norm_hidden,
                    self.index_compress_values,
                )
            )
            stages.append(
                self._bf16_linear_stage(
                    "index.compressor.wgate",
                    family,
                    "attn.indexer.compressor.wgate.weight",
                    self.norm_hidden,
                    self.index_compress_scores,
                )
            )

        sinks = self._family_tensors(family, "attn.attn_sink")
        sparse = SchedDsv4SparseAttention512(
            self.q_rope,
            self.kv_row,
            self.attention_indices,
            sinks[0],
            self.attention_output,
        )
        sparse = self._layered(sparse, family, sinks)
        stages.append(
            self._stage(
                f"attn.sparse_{kind}",
                sparse,
                self.policy.attention(cfg.num_heads, cfg.head_dim),
            )
        )
        stages.append(
            self._stage(
                "attn.inverse_rope",
                SchedDsv4Rope512_64(
                    self.attention_output,
                    rope_table,
                    self.attention_inverse,
                    inverse=True,
                ),
                cfg.num_heads,
            )
        )
        grouped = self.attention_inverse.reshape(cfg.o_groups, -1)
        for group in range(cfg.o_groups):
            stages.append(
                self._fp8_quant_stage(
                    f"attn.o_a.g{group}.quant_fp8",
                    grouped[group],
                    self.o_group_fp8[group],
                    self.o_group_scale[group],
                )
            )
            start = group * cfg.o_lora_rank
            stages.append(
                self._fp8_linear_stage(
                    f"attn.o_a.g{group}",
                    family,
                    "attn.wo_a",
                    self.o_group_fp8[group],
                    self.o_group_scale[group],
                    self.o_rank[group],
                    row_slice=slice(start, start + cfg.o_lora_rank),
                )
            )
        stages.append(
            self._fp8_quant_stage(
                "attn.o_rank.quant_fp8",
                self.o_rank.reshape(-1),
                self.o_rank_fp8,
                self.o_rank_scale,
            )
        )
        stages.append(
            self._fp8_linear_stage(
                "attn.o_b",
                family,
                "attn.wo_b",
                self.o_rank_fp8,
                self.o_rank_scale,
                self.branch,
            )
        )
        stages.append(post)
        return stages

    @staticmethod
    def _row_pointer(tensor: torch.Tensor, row_start: int) -> int:
        return (
            tensor.data_ptr()
            + row_start * tensor.stride(0) * tensor.element_size()
        )

    def _routing_table(self, layer_id: int) -> RoutedAddressTable:
        existing = self._routing_tables.get(layer_id)
        if existing is not None:
            return existing
        cfg = self.config
        assignments = {
            "w1": self.policy.nvfp4_gemv(
                cfg.expert_intermediate_size, cfg.hidden_size
            ),
            "w3": self.policy.nvfp4_gemv(
                cfg.expert_intermediate_size, cfg.hidden_size
            ),
            "w2": self.policy.nvfp4_gemv(
                cfg.hidden_size, cfg.expert_intermediate_size
            ),
        }
        columns: dict[str, list[int]] = {}
        for tag, assignment in assignments.items():
            for sm in range(assignment.num_sms):
                columns[f"{tag}.weight.sm{sm}"] = []
                columns[f"{tag}.weight_scale.sm{sm}"] = []

        linears = {tag: [] for tag in assignments}
        for expert_id in range(cfg.num_experts):
            prefix = f"layers.{layer_id}.ffn.experts.{expert_id}"
            for tag, assignment in assignments.items():
                linear = self.checkpoint.load_nvfp4_linear(
                    f"{prefix}.{tag}", device=self.device
                )
                linears[tag].append(linear)
                for sm in range(assignment.num_sms):
                    row_start, _ = assignment.shard(sm)
                    columns[f"{tag}.weight.sm{sm}"].append(
                        self._row_pointer(linear.weight, row_start)
                    )
                    columns[f"{tag}.weight_scale.sm{sm}"].append(
                        self._row_pointer(linear.weight_scale, row_start)
                    )

        def stack(tag: str, field: str) -> torch.Tensor:
            return torch.stack(
                [getattr(linear, field).reshape(()) for linear in linears[tag]]
            )

        w1_input = stack("w1", "input_scale")
        w3_input = stack("w3", "input_scale")
        if not torch.equal(w1_input, w3_input):
            raise ValueError(f"layer {layer_id} w1/w3 input scales differ")
        w2_input = stack("w2", "input_scale")
        alpha = {
            tag: stack(tag, "weight_scale_2") * stack(tag, "input_scale")
            for tag in assignments
        }

        def padded(values: torch.Tensor) -> torch.Tensor:
            result = torch.zeros(
                (cfg.num_experts, 4), dtype=torch.float32, device=self.device
            )
            result[:, 0].copy_(values)
            return result

        derived = {
            "up.input_scale": padded(w1_input),
            "down.input_scale": padded(w2_input),
            **{f"{tag}.alpha": padded(values) for tag, values in alpha.items()},
        }
        for name, tensor in derived.items():
            columns[name] = [
                self._row_pointer(tensor, expert_id)
                for expert_id in range(cfg.num_experts)
            ]
        owners = tuple(derived.values())
        table = RoutedAddressTable.from_pointer_columns(
            columns, device=self.device, owners=owners
        )
        self._routing_tables[layer_id] = table
        self._routing_owners[layer_id] = owners
        return table

    def _routed_quant_stage(
        self,
        name: str,
        family: LayerFamily,
        tables: tuple[RoutedAddressTable, ...],
        rank: int,
        field_name: str,
        source: torch.Tensor,
        output: torch.Tensor,
        scale: torch.Tensor,
    ) -> Stage:
        representative = tables[0]
        schedule = SchedRoutedDsv4Nvfp4Quant16(
            representative.state,
            rank,
            representative.field(field_name),
            source.reshape(-1),
            output.reshape(-1),
            scale.reshape(-1),
        )
        schedule = self._routed_layered(schedule, family, tables)
        return self._stage(
            name,
            schedule,
            self.policy.quantize(source.numel(), 16),
            input_role="route",
        )

    def _routed_linear_stage(
        self,
        name: str,
        family: LayerFamily,
        tables: tuple[RoutedAddressTable, ...],
        rank: int,
        tag: str,
        rows: int,
        k: int,
        activation: torch.Tensor,
        activation_scale: torch.Tensor,
        output: torch.Tensor,
    ) -> Stage:
        assignment = self.policy.nvfp4_gemv(rows, k)
        table = tables[0]
        weight_fields = [
            table.field(f"{tag}.weight.sm{sm}")
            for sm in range(assignment.num_sms)
        ]
        scale_fields = [
            table.field(f"{tag}.weight_scale.sm{sm}")
            for sm in range(assignment.num_sms)
        ]
        schedule = SchedRoutedNvfp4Gemv(
            table.state,
            rank,
            weight_fields,
            scale_fields,
            table.field(f"{tag}.alpha"),
            rows,
            k,
            activation.reshape(-1),
            activation_scale.reshape(-1),
            output.reshape(-1),
        )
        schedule = self._routed_layered(schedule, family, tables)
        return self._stage(
            name, schedule, assignment, input_role="route"
        )

    def _hash_row(self, layer_id: int) -> torch.Tensor:
        existing = self._hash_rows.get(layer_id)
        if existing is not None:
            return existing
        source = self._tensor(f"layers.{layer_id}.ffn.gate.tid2eid")[
            self.args.token_id
        ]
        row = torch.zeros((8,), dtype=torch.int32, device=self.device)
        row[: self.config.experts_per_token].copy_(source.to(torch.int32))
        self._hash_rows[layer_id] = row
        return row

    def _build_ffn(self, family: LayerFamily) -> list[Stage]:
        cfg = self.config
        stages, post = self._hc_stages(
            family, "ffn", self.next_residual, self.residual
        )
        stages.append(
            self._fp8_quant_stage(
                "ffn.hidden.quant_fp8",
                self.norm_hidden,
                self.hidden_fp8,
                self.hidden_fp8_scale,
            )
        )
        stages.append(
            self._bf16_linear_stage(
                "ffn.router",
                family,
                "ffn.gate.weight",
                self.norm_hidden,
                self.router_logits,
            )
        )
        hash_routing = family.representative < cfg.num_hash_layers
        if hash_routing:
            hash_rows = tuple(self._hash_row(layer) for layer in family.layer_ids)
            biases = (self.zero_bias,) * len(family.layer_ids)
        else:
            hash_rows = (self.zero_hash,) * len(family.layer_ids)
            biases = self._family_tensors(family, "ffn.gate.bias")
        route = SchedDsv4RouteTop6(
            self.router_logits,
            biases[0],
            hash_rows[0],
            self.route_indices,
            self.route_weights,
            hash_routing=hash_routing,
            route_scale=cfg.route_scale,
        )
        route_groups = (hash_rows,) if hash_routing else (biases,)
        route = self._layered(route, family, *route_groups)
        stages.append(self._stage("ffn.route", route))

        tables = tuple(self._routing_table(layer) for layer in family.layer_ids)
        for rank in range(cfg.experts_per_token):
            stages.append(
                self._routed_quant_stage(
                    f"ffn.expert{rank}.input.quant_nvfp4",
                    family,
                    tables,
                    rank,
                    "up.input_scale",
                    self.norm_hidden,
                    self.routed_input[rank],
                    self.routed_input_scale[rank],
                )
            )
            stages.append(
                self._routed_linear_stage(
                    f"ffn.expert{rank}.w1",
                    family,
                    tables,
                    rank,
                    "w1",
                    cfg.expert_intermediate_size,
                    cfg.hidden_size,
                    self.routed_input[rank],
                    self.routed_input_scale[rank],
                    self.routed_gate[rank],
                )
            )
            stages.append(
                self._routed_linear_stage(
                    f"ffn.expert{rank}.w3",
                    family,
                    tables,
                    rank,
                    "w3",
                    cfg.expert_intermediate_size,
                    cfg.hidden_size,
                    self.routed_input[rank],
                    self.routed_input_scale[rank],
                    self.routed_up[rank],
                )
            )
            stages.append(
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
            stages.append(
                self._routed_quant_stage(
                    f"ffn.expert{rank}.middle.quant_nvfp4",
                    family,
                    tables,
                    rank,
                    "down.input_scale",
                    self.routed_middle[rank],
                    self.routed_middle_packed[rank],
                    self.routed_middle_scale[rank],
                )
            )
            stages.append(
                self._routed_linear_stage(
                    f"ffn.expert{rank}.w2",
                    family,
                    tables,
                    rank,
                    "w2",
                    cfg.hidden_size,
                    cfg.expert_intermediate_size,
                    self.routed_middle_packed[rank],
                    self.routed_middle_scale[rank],
                    self.routed_output[rank],
                )
            )

        stages.append(
            self._fp8_linear_stage(
                "ffn.shared.w1",
                family,
                "ffn.shared_experts.w1",
                self.hidden_fp8,
                self.hidden_fp8_scale,
                self.shared_gate,
            )
        )
        stages.append(
            self._fp8_linear_stage(
                "ffn.shared.w3",
                family,
                "ffn.shared_experts.w3",
                self.hidden_fp8,
                self.hidden_fp8_scale,
                self.shared_up,
            )
        )
        stages.append(
            self._stage(
                "ffn.shared.swiglu",
                SchedSmemSiLUInterleaved(
                    1,
                    self.shared_gate.reshape(1, -1),
                    self.shared_up.reshape(1, -1),
                    self.shared_middle.reshape(1, -1),
                    swiglu_limit=cfg.swiglu_limit,
                ),
            )
        )
        stages.append(
            self._fp8_quant_stage(
                "ffn.shared.middle.quant_fp8",
                self.shared_middle,
                self.shared_middle_fp8,
                self.shared_middle_scale,
            )
        )
        stages.append(
            self._fp8_linear_stage(
                "ffn.shared.w2",
                family,
                "ffn.shared_experts.w2",
                self.shared_middle_fp8,
                self.shared_middle_scale,
                self.shared_output,
            )
        )
        stages.append(
            self._stage(
                "ffn.expert_reduce",
                SchedDsv4ExpertReduce(
                    self.routed_output,
                    self.route_weights[: cfg.experts_per_token],
                    self.shared_output,
                    self.branch,
                ),
            )
        )
        stages.append(post)
        return stages

    def _build_family(self, family: LayerFamily) -> list[Stage]:
        return self._build_attention(family) + self._build_ffn(family)

    def _build_head(self) -> list[Stage]:
        cfg = self.config
        head_fn = self._tensor("hc_head_fn")
        head_scale = self._tensor("hc_head_scale")
        head_base = self._tensor("hc_head_base")
        self.head_mixes = torch.empty((4,), dtype=torch.float32, device=self.device)
        self.head_hidden = torch.empty(
            (cfg.hidden_size,), dtype=torch.bfloat16, device=self.device
        )
        self.head_norm = torch.empty_like(self.head_hidden)
        self.logits = torch.empty(
            (self.args.vocab_size,), dtype=torch.bfloat16, device=self.device
        )
        stages = [
            self._stage(
                "head.hc_project",
                SchedDsv4Fp32Bf16Gemv(
                    head_fn, self.residual.reshape(-1), self.head_mixes
                ),
                self.policy.fp32_bf16_gemv(4, self.residual.numel()),
            ),
            self._stage(
                "head.hc",
                SchedDsv4HcHead(
                    self.residual,
                    self.head_mixes,
                    head_scale,
                    head_base,
                    self.head_hidden,
                ),
            ),
            self._rms_stage(
                "head.rms4096",
                self.head_hidden,
                self.head_norm,
                weight_suffix="norm.weight",
            ),
        ]
        head_weight = self._tensor("head.weight")[: self.args.vocab_size]
        stages.append(
            self._stage(
                "head.logits",
                SchedDsv4Bf16Gemv(head_weight, self.head_norm, self.logits),
                self.policy.bf16_gemv(
                    self.args.vocab_size, cfg.hidden_size
                ),
            )
        )
        return stages

    def _build_program(self) -> None:
        serial_sm = 0

        def queued(stage: Stage, prefix: str = "") -> SequentialStage:
            nonlocal serial_sm
            base_sm = 0
            if stage.num_sms == 1:
                base_sm = serial_sm
                serial_sm = (serial_sm + 1) % self.sms
            return SequentialStage(
                f"{prefix}{stage.name}",
                stage.schedule,
                stage.num_sms,
                base_sm=base_sm,
                input_role=stage.input_role,
            )

        self.launcher = Launcher(self.sms, device=self.device)
        if self.args.layers == 1:
            family = self.families[0]
            stages = [
                queued(stage, f"{family.name}.")
                for stage in self.family_stages[family.representative]
            ]
            stages.extend(queued(stage) for stage in self.head_stages)
            self.program = SequentialProgram(self.launcher, stages)
            logical_stages = len(stages)
            queue_stages = logical_stages
        elif self.args.layers == 2:
            family = self.families[0]
            family_stages = [
                queued(stage, f"{family.name}.")
                for stage in self.family_stages[family.representative]
            ]
            head_stages = [queued(stage) for stage in self.head_stages]
            blocks = (
                SequentialBlock(
                    family.name,
                    family_stages,
                    repeat=2,
                    barrier_banks=2,
                ),
                SequentialBlock("head", head_stages, reload_after=False),
            )
            self.program = LoopedSequentialProgram(self.launcher, blocks)
            logical_stages = sum(
                len(block.stages) * block.repeat for block in blocks
            )
            queue_stages = sum(len(block.stages) for block in blocks)
        else:
            swa, layer2, hca, csa = self.families
            swa_stages = [
                queued(stage, f"{swa.name}.")
                for stage in self.family_stages[swa.representative]
            ]
            layer2_stages = [
                queued(stage, f"{layer2.name}.")
                for stage in self.family_stages[layer2.representative]
            ]
            pair_stages = [
                queued(stage, f"{hca.name}.")
                for stage in self.family_stages[hca.representative]
            ] + [
                queued(stage, f"{csa.name}.")
                for stage in self.family_stages[csa.representative]
            ]
            head_stages = [queued(stage) for stage in self.head_stages]
            blocks = (
                SequentialBlock(
                    swa.name, swa_stages, repeat=2, barrier_banks=2
                ),
                SequentialBlock(layer2.name, layer2_stages),
                SequentialBlock(
                    "layers3-42.hca_csa_score",
                    pair_stages,
                    repeat=20,
                    barrier_banks=2,
                ),
                SequentialBlock("head", head_stages, reload_after=False),
            )
            self.program = LoopedSequentialProgram(self.launcher, blocks)
            logical_stages = sum(
                len(block.stages) * block.repeat for block in blocks
            )
            queue_stages = sum(len(block.stages) for block in blocks)
        self.launcher.s(self.program)
        print(
            "DSV4_ONE_LAUNCH_PROGRAM "
            f"model_launches=1 layers={self.args.layers} "
            f"logical_stages={logical_stages} queue_stages={queue_stages} "
            f"barriers={len(self.program.barriers)} "
            f"compute_insts={self.program.max_compute_instructions} "
            f"memory_insts={self.program.max_memory_instructions}",
            flush=True,
        )
        for assignment in sorted(
            self.assignments.values(),
            key=lambda item: (item.task, item.rows, item.k),
        ):
            print(
                "DSV4_SHAPE_ASSIGNMENT "
                f"task={assignment.task} rows={assignment.rows} k={assignment.k} "
                f"sms={assignment.num_sms} alignment={assignment.row_alignment} "
                f"tile_rows={assignment.tile_rows} tile_k={assignment.tile_k}",
                flush=True,
            )

    def run_once(self) -> tuple[int, float, torch.Tensor]:
        self.residual.copy_(self.initial_residual)
        torch.cuda.synchronize(self.device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        self.launcher.launch(synchronize=False)
        end.record()
        end.synchronize()
        logits_cpu = self.logits.cpu()
        logits_fp32 = logits_cpu.float()
        if not bool(torch.isfinite(logits_fp32).all().item()):
            raise AssertionError("one-launch checkpoint logits are not finite")
        token = int(torch.argmax(logits_fp32).item())
        return token, start.elapsed_time(end), logits_fp32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--layers", type=int, choices=(1, 2, 43), default=1)
    parser.add_argument("--token-id", type=int, default=791)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--sms", type=int, default=152)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--expected-token-id", type=int)
    parser.add_argument("--resident-reserve-gib", type=float, default=8.0)
    args = parser.parse_args()
    cfg = DeepSeekV4FlashConfig()
    if not 0 <= args.token_id < cfg.vocab_size:
        parser.error("token-id is outside the vocabulary")
    if not 1 <= args.vocab_size <= cfg.vocab_size:
        parser.error("vocab-size must be in [1,129280]")
    if args.sms <= 0 or args.iterations <= 0 or args.warmup < 0:
        parser.error("sms/iterations must be positive and warmup non-negative")
    if args.resident_reserve_gib < 0:
        parser.error("resident-reserve-gib must be non-negative")

    device = torch.device("cuda")
    build_started = time.monotonic()
    flow = ResidentOneLaunchDecode(args, device)
    torch.cuda.synchronize(device)
    build_seconds = time.monotonic() - build_started
    for _ in range(args.warmup):
        token, _, _ = flow.run_once()
        if args.expected_token_id is not None and token != args.expected_token_id:
            raise AssertionError(
                f"warmup emitted token {token}, expected {args.expected_token_id}"
            )

    timings = []
    reference_token = None
    logits = None
    for _ in range(args.iterations):
        token, elapsed_ms, logits = flow.run_once()
        timings.append(elapsed_ms)
        if reference_token is None:
            reference_token = token
        elif token != reference_token:
            raise AssertionError("one-launch checkpoint token is not repeatable")
    if args.expected_token_id is not None and reference_token != args.expected_token_id:
        raise AssertionError(
            f"checkpoint emitted token {reference_token}, "
            f"expected {args.expected_token_id}"
        )
    assert logits is not None
    print(
        "DSV4_ONE_LAUNCH_DECODE status=PASS model_launches=1 gpu=1 "
        f"layers={args.layers} token_id={args.token_id} "
        f"vocab={args.vocab_size} output_token={reference_token} "
        f"build_s={build_seconds:.3f} min_ms={min(timings):.6f} "
        f"median_ms={statistics.median(timings):.6f} "
        f"max_ms={max(timings):.6f} "
        f"logit_min={float(logits.min().item()):.6f} "
        f"logit_max={float(logits.max().item()):.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
