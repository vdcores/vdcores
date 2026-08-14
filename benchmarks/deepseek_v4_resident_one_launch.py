#!/usr/bin/env python3
"""Checkpoint-resident DeepSeek-V4 decode in one VDCores launch.

The full model is represented by four shape families: layers 0-1, layer 2,
odd HCA layers, and even CSA layers.  Repeated families use runtime loop
counters to select resident layer weights; routed expert IDs stay in one fixed
HBM buffer and LDU resolves the selected expert and current layer.
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from dataclasses import dataclass, replace

import torch

from dae.deepseek_v4 import (
    DeepSeekV4FlashConfig,
    deepseek_v4_rope_table,
    pack_gated_pool_history,
)
from dae.deepseek_v4_checkpoint import (
    DeepSeekV4Checkpoint,
    DeepSeekV4ResidentCheckpoint,
    expected_inference_tensor_specs,
)
from dae.deepseek_v4_flow import build_layer_decode_plan
from dae.deepseek_v4_schedule import DeepSeekV4ShapePolicy, ShapeAssignment
from dae.deepseek_v4_quant import (
    dequantize_fp8_block128,
    quantize_fp8_block128,
)
from dae.launcher import Launcher
from dae.instructions import TmaTensor
from dae.routing import RoutedAddressTable
from dae.runtime import config as runtime_config
from dae.schedule import (
    LayeredSchedule,
    SchedArgmaxSmemPartial,
    SchedArgmaxSmemReduce,
    SchedDsv4AttentionSplit32UmmaSm100,
    SchedDsv4AttentionSplitReduceFp8Sm100,
    SchedDsv4Bf16Gemv,
    SchedDsv4Bf16GemvGroup4SplitK,
    SchedDsv4ExpertReduce,
    SchedDsv4Fp8QuantUmmaB,
    SchedDsv4Fp32RmsFp8QuantUmmaB,
    SchedDsv4RmsFp8QuantUmmaB,
    SchedDsv4Fp32Bf16Gemv,
    SchedDsv4Fp32ToBf16,
    SchedDsv4Fp8Quant128,
    SchedDsv4GatedPool,
    SchedDsv4GatedPoolRmsRope,
    SchedDsv4GatedPoolPacked8Shard128,
    SchedDsv4GatedPoolPacked8HistoryState,
    SchedDsv4GatedPoolTailRmsPartial,
    SchedDsv4Hadamard,
    SchedDsv4HcHead,
    SchedDsv4HcPost,
    SchedDsv4HcPreRms,
    SchedDsv4IndexScore,
    SchedDsv4PreloadRopeTables,
    SchedDsv4Rope128_64,
    SchedDsv4Rope512_64,
    SchedDsv4Fp32RmsRope512_64,
    SchedDsv4Fp32RopeHadamard128,
    SchedDsv4Fp32RmsRopeShard128,
    SchedDsv4RmsRope512_64,
    SchedDsv4RouteTop6,
    SchedDsv4SparseAttention512,
    SchedDsv4ContiguousAttention512Block4,
    SchedDsv4SwiGluShard128,
    SchedDsv4TopK512,
    SchedDsv4ZeroFill,
    SchedFp8Block128Gemv,
    SchedFp8Block128GemvBf16,
    SchedFp8GemvUmmaStream,
    SchedFp8GemvUmmaSplitK,
    SchedRMS,
    SchedRoutedDsv4Nvfp4QuantUmmaB,
    SchedRoutedNvfp4GemvUmmaStream,
    SchedSmemSiLUInterleaved,
)
from dae.sequential import (
    LoopedSequentialProgram,
    SequentialBlock,
    SequentialProgram,
    SequentialStage,
)
from dae.tma_utils import Major


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
    wait_for_previous: bool = True
    base_sm: int | None = None
    wait_group: str | None = None
    release_group: str | None = None
    prefetch_before_wait: bool = False


class ResidentOneLaunchDecode:
    def __init__(self, args: argparse.Namespace, device: torch.device):
        self.args = args
        self.device = device
        self.config = DeepSeekV4FlashConfig()
        self.sms = min(
            args.sms,
            torch.cuda.get_device_properties(device).multi_processor_count,
        )
        all_splitk_components = {
            "q_a", "q_b", "kv", "index_q_b", "o_a", "o_b"
        }
        requested_components = {
            item.strip()
            for item in args.fp8_splitk_components.split(",")
            if item.strip()
        }
        if requested_components == {"all"}:
            requested_components = all_splitk_components
        unknown_components = requested_components - all_splitk_components
        if unknown_components:
            raise ValueError(
                f"unknown split-K projection components: {sorted(unknown_components)}"
            )
        self.splitk_components = (
            frozenset(requested_components)
            if args.fp8_projection_mode == "splitk"
            else frozenset()
        )
        self.direct_splitk_bf16 = bool(self.splitk_components) and (
            args.fp8_splitk_reduction == "bf16"
        )
        self.ffn_fp32_tma = args.ffn_reduction == "fp32-tma"
        self.splitk_accumulators: list[torch.Tensor] = []
        self._active_splitk_workspace: torch.Tensor | None = None
        self._active_splitk_offset = 0
        self.policy = DeepSeekV4ShapePolicy(self.sms)
        self.assignments: dict[tuple, ShapeAssignment] = {}
        self.launcher = Launcher(self.sms, device=self.device)
        self.checkpoint = self._load_checkpoint()
        self.families = self._families()
        self._routing_tables: dict[int, RoutedAddressTable] = {}
        self._routing_owners: dict[int, tuple[torch.Tensor, ...]] = {}
        self._hash_rows: dict[int, torch.Tensor] = {}
        self._fused_bf16_weight_cache: dict[tuple, tuple[torch.Tensor, ...]] = {}
        self._allocate_state()
        rope_tables = [self.main_rope, self.compress_rope]
        rope_tables.extend(
            self.compressed_output_rope[kind]
            for kind in ("csa", "hca")
            if kind in self.compressed_output_rope
        )
        self.resident_rope_tables = tuple(rope_tables)
        self.resident_rope_table_ids = {
            table.data_ptr(): table_id
            for table_id, table in enumerate(self.resident_rope_tables)
        }
        self.family_stages = {
            family.representative: self._build_family(family)
            for family in self.families
        }
        self.head_stages = self._build_head()
        self._build_program()
        print(
            "DSV4_COMPUTE_OPS "
            f"count={len(self.launcher.compute_operator_names())} "
            f"ops={','.join(self.launcher.compute_operator_names())}",
            flush=True,
        )
        prepare_started = time.monotonic()
        self.launcher.prepare_launch()
        torch.cuda.synchronize(self.device)
        print(
            "DSV4_ONE_LAUNCH_PREPARE status=PASS "
            f"elapsed_s={time.monotonic() - prepare_started:.3f}",
            flush=True,
        )

    def _load_checkpoint(self) -> DeepSeekV4ResidentCheckpoint:
        disk = DeepSeekV4Checkpoint(self.args.checkpoint, self.config)
        resident_layer_ids = (
            (self.args.single_layer_id,)
            if self.args.layers == 1
            else tuple(range(self.args.layers))
        )
        names = None
        if self.args.layers != self.config.num_layers:
            prefix = tuple(
                f"layers.{layer_id}." for layer_id in resident_layer_ids
            )
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

        native_fp8_prefixes = []
        for layer_id in resident_layer_ids:
            attention_prefix = f"layers.{layer_id}.attn"
            if self.ffn_fp32_tma:
                native_fp8_prefixes.append(
                    f"layers.{layer_id}.ffn.shared_experts.w2"
                )
            if "q_a" in self.splitk_components:
                native_fp8_prefixes.append(f"{attention_prefix}.wq_a")
            if "kv" in self.splitk_components:
                native_fp8_prefixes.append(f"{attention_prefix}.wkv")
            if (
                "q_b" in self.splitk_components
                or self.args.fp8_qb_mode == "native"
            ):
                native_fp8_prefixes.append(f"{attention_prefix}.wq_b")
            if "o_a" in self.splitk_components:
                native_fp8_prefixes.append(f"{attention_prefix}.wo_a")
            if "o_b" in self.splitk_components:
                native_fp8_prefixes.append(f"{attention_prefix}.wo_b")
            if (
                self.config.attention_kind(layer_id) == "csa"
                and (
                    "index_q_b" in self.splitk_components
                    or self.args.fp8_qb_mode == "native"
                )
            ):
                native_fp8_prefixes.append(
                    f"{attention_prefix}.indexer.wq_b"
                )

        resident = DeepSeekV4ResidentCheckpoint.from_checkpoint(
            disk,
            device=self.device,
            names=names,
            reserve_bytes=int(self.args.resident_reserve_gib * (1 << 30)),
            native_nvfp4=True,
            native_fp8_prefixes=tuple(native_fp8_prefixes),
            native_fp8_scale_pack=self.args.fp8_umma_scale_pack,
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
            layer_id = self.args.single_layer_id
            kind = self.config.attention_kind(layer_id)
            routing = "hash" if layer_id < self.config.num_hash_layers else "score"
            return (LayerFamily(f"layer{layer_id}.{kind}_{routing}", (layer_id,)),)
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
        wait_for_previous: bool = True,
        base_sm: int | None = None,
        wait_group: str | None = None,
        release_group: str | None = None,
        prefetch_before_wait: bool = False,
    ) -> Stage:
        if isinstance(sms, ShapeAssignment):
            sms = self._remember(sms)
        return Stage(
            name,
            schedule,
            int(sms),
            input_role,
            wait_for_previous,
            base_sm,
            wait_group,
            release_group,
            prefetch_before_wait,
        )

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
        direct_projection_views = {}
        if self.direct_splitk_bf16:
            projection_rows = (
                cfg.hidden_size
                + cfg.q_lora_rank
                + cfg.head_dim
                + cfg.num_heads * cfg.head_dim
                + cfg.index_heads * cfg.index_head_dim
                + cfg.o_groups * cfg.o_lora_rank
            )
            self.splitk_output_arena = torch.empty(
                (projection_rows,), dtype=torch.bfloat16, device=d
            )
            offset = 0

            def direct_view(name, shape):
                nonlocal offset
                elements = math.prod(shape)
                view = self.splitk_output_arena[offset : offset + elements].view(
                    shape
                )
                direct_projection_views[name] = view
                offset += elements

            direct_view("branch", (cfg.hidden_size,))
            direct_view("q_rank", (cfg.q_lora_rank,))
            direct_view("kv", (cfg.head_dim,))
            direct_view("q", (cfg.num_heads, cfg.head_dim))
            direct_view("index_q", (cfg.index_heads, cfg.index_head_dim))
            direct_view("o_rank", (cfg.o_groups, cfg.o_lora_rank))
            if offset != projection_rows:
                raise AssertionError("split-K output arena was not carved exactly")
            self.branch = direct_projection_views["branch"]
        else:
            self.splitk_output_arena = None
            self.branch = torch.empty_like(self.hidden)
        self.mixes = torch.empty((24,), dtype=torch.float32, device=d)
        self.post = torch.empty((4,), dtype=torch.float32, device=d)
        self.comb = torch.empty((4, 4), dtype=torch.float32, device=d)

        self.decode_position = self.args.context_length - 1
        self.main_rope = deepseek_v4_rope_table(
            self.decode_position, config=cfg, device=d
        )
        self.compress_rope = deepseek_v4_rope_table(
            self.decode_position, compressed=True, config=cfg, device=d
        )

        def seeded(shape, *, dtype, seed, scale=0.125):
            generator = torch.Generator(device=d).manual_seed(seed)
            values = torch.randn(
                shape,
                dtype=torch.float32,
                device=d,
                generator=generator,
            ).mul_(scale)
            return values if dtype == torch.float32 else values.to(dtype)

        representatives = {"swa": 0, "csa": 2, "hca": 3}
        self.attention_plans = {
            kind: build_layer_decode_plan(layer_id, self.decode_position, cfg)
            for kind, layer_id in representatives.items()
        }
        self.attention_cache = {}
        self.attention_indices_by_kind = {}
        self.current_kv_rows = {}
        self.current_compressed_rows = {}
        for kind_index, kind in enumerate(("swa", "csa", "hca")):
            plan = self.attention_plans[kind]
            cache_rows = cfg.sliding_window + plan.compressed_rows
            cache = seeded(
                (cache_rows, cfg.head_dim),
                dtype=torch.bfloat16,
                seed=202608110 + kind_index,
            )
            valid_window = min(cfg.sliding_window, self.args.context_length)
            indices = torch.empty(
                (plan.attention_candidates,), dtype=torch.int32, device=d
            )
            indices[:valid_window].copy_(
                torch.arange(valid_window, dtype=torch.int32, device=d)
            )
            if plan.compressed_selected:
                indices[valid_window:].copy_(
                    torch.arange(
                        cfg.sliding_window,
                        cfg.sliding_window + plan.compressed_selected,
                        dtype=torch.int32,
                        device=d,
                    )
                )
            self.attention_cache[kind] = cache
            self.attention_indices_by_kind[kind] = indices
            window_row = self.decode_position % cfg.sliding_window
            self.current_kv_rows[kind] = cache[window_row : window_row + 1]
            if plan.should_compress:
                compressed_row = cfg.sliding_window + plan.compressed_rows - 1
                self.current_compressed_rows[kind] = cache[
                    compressed_row : compressed_row + 1
                ]

        self.hidden_fp8 = torch.empty(
            (cfg.hidden_size,), dtype=torch.float8_e4m3fn, device=d
        )
        self.hidden_fp8_scale = torch.empty(
            (cfg.hidden_size // 128,), dtype=torch.float8_e8m0fnu, device=d
        )
        if {"q_a", "kv"} & self.splitk_components:
            self.hidden_native_fp8 = torch.empty(
                (cfg.hidden_size // 128, 2048), dtype=torch.uint8, device=d
            )
        self.q_rank = direct_projection_views.get("q_rank")
        if self.q_rank is None:
            self.q_rank = torch.empty(
                (cfg.q_lora_rank,), dtype=torch.bfloat16, device=d
            )
        self.q_rank_norm = torch.empty_like(self.q_rank)
        self.q_rank_native_fp8 = torch.empty(
            (cfg.q_lora_rank // 128, 2048), dtype=torch.uint8, device=d
        )
        self.q_rank_fp8 = torch.empty_like(
            self.q_rank, dtype=torch.float8_e4m3fn
        )
        self.q_rank_fp8_scale = torch.empty(
            (cfg.q_lora_rank // 128,),
            dtype=torch.float8_e8m0fnu,
            device=d,
        )
        self.q = direct_projection_views.get("q")
        if self.q is None:
            self.q = torch.empty(
                (cfg.num_heads, cfg.head_dim), dtype=torch.bfloat16, device=d
            )
        self.q_norm = torch.empty_like(self.q)
        self.q_rope = torch.empty_like(self.q)
        self.kv = direct_projection_views.get("kv")
        if self.kv is None:
            self.kv = torch.empty(
                (cfg.head_dim,), dtype=torch.bfloat16, device=d
            )
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
        self.o_rank = direct_projection_views.get("o_rank")
        if self.o_rank is None:
            self.o_rank = torch.empty(
                (cfg.o_groups, cfg.o_lora_rank),
                dtype=torch.bfloat16,
                device=d,
            )
        self.o_rank_fp8 = torch.empty_like(
            self.o_rank.reshape(-1), dtype=torch.float8_e4m3fn
        )
        self.o_rank_scale = torch.empty(
            (self.o_rank.numel() // 128,), dtype=torch.float8_e8m0fnu, device=d
        )
        if {"o_a", "o_b"} & self.splitk_components:
            self.o_group_native_fp8 = torch.empty(
                (cfg.o_groups, group_width // 128, 2048),
                dtype=torch.uint8,
                device=d,
            )
            self.o_rank_native_fp8 = torch.empty(
                (self.o_rank.numel() // 128, 2048), dtype=torch.uint8, device=d
            )
        if "o_a" in self.splitk_components:
            max_attention_splits = max(
                (indices.numel() + 31) // 32
                for indices in self.attention_indices_by_kind.values()
            )
            self.attention_partial_workspace = torch.empty(
                (max_attention_splits, cfg.num_heads, cfg.head_dim),
                dtype=torch.bfloat16,
                device=d,
            )
            self.attention_metadata_workspace = torch.empty(
                (max_attention_splits, cfg.num_heads, 2),
                dtype=torch.float32,
                device=d,
            )

        self.compress_values = torch.empty((1024,), dtype=torch.float32, device=d)
        self.compress_scores = torch.empty_like(self.compress_values)
        self.compress_fused_projection = torch.empty(
            (2048,), dtype=torch.float32, device=d
        )
        self.attention_pool_history_values = {}
        self.attention_pool_history_scores = {}
        self.attention_pool_history_packed = {}
        self.attention_pool_history_state = {}
        self.attention_pooled = {}
        self.attention_pooled_fp32 = {}
        self.attention_pooled_rms_partials = {}
        self.attention_pooled_norm = {}
        self.compressed_output_rope = {}
        for kind_index, kind in enumerate(("csa", "hca")):
            plan = self.attention_plans[kind]
            if not plan.should_compress:
                continue
            pool_rows = (
                plan.compress_ratio
                if plan.compress_ratio != 4 or plan.compressed_rows == 1
                else 2 * plan.compress_ratio
            )
            history_values = seeded(
                (pool_rows - 1, cfg.head_dim),
                dtype=torch.float32,
                seed=202608120 + kind_index,
            )
            history_scores = seeded(
                (pool_rows - 1, cfg.head_dim),
                dtype=torch.float32,
                seed=202608130 + kind_index,
                scale=0.25,
            )
            self.attention_pool_history_values[kind] = history_values
            self.attention_pool_history_scores[kind] = history_scores
            self.attention_pool_history_packed[kind] = pack_gated_pool_history(
                history_values, history_scores
            )
            self.attention_pool_history_state[kind] = torch.empty(
                (cfg.head_dim // 128, 3, 128),
                dtype=torch.float32,
                device=d,
            )
            self.attention_pooled[kind] = torch.empty(
                (cfg.head_dim,), dtype=torch.bfloat16, device=d
            )
            self.attention_pooled_fp32[kind] = torch.empty(
                (cfg.head_dim // 128, 128),
                dtype=torch.float32,
                device=d,
            )
            self.attention_pooled_rms_partials[kind] = torch.empty(
                (cfg.head_dim // 128,),
                dtype=torch.float32,
                device=d,
            )
            self.attention_pooled_norm[kind] = torch.empty_like(
                self.attention_pooled[kind]
            )
            compressed_position = self.decode_position - plan.compress_ratio + 1
            self.compressed_output_rope[kind] = deepseek_v4_rope_table(
                compressed_position,
                compressed=True,
                config=cfg,
                device=d,
            )
        self.index_q = direct_projection_views.get("index_q")
        if self.index_q is None:
            self.index_q = torch.empty(
                (cfg.index_heads, cfg.index_head_dim),
                dtype=torch.bfloat16,
                device=d,
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
        self.index_compress_fused_projection = torch.empty(
            (4, cfg.index_head_dim), dtype=torch.float32, device=d
        )
        csa_plan = self.attention_plans["csa"]
        self.index_cache = seeded(
            (csa_plan.compressed_rows, cfg.index_head_dim),
            dtype=torch.bfloat16,
            seed=202608140,
        )
        self.index_scores = torch.empty(
            (csa_plan.compressed_rows,), dtype=torch.float32, device=d
        )
        if csa_plan.should_compress:
            index_pool_rows = (
                csa_plan.compress_ratio
                if csa_plan.compressed_rows == 1
                else 2 * csa_plan.compress_ratio
            )
            self.index_pool_history_values = seeded(
                (index_pool_rows - 1, cfg.index_head_dim),
                dtype=torch.float32,
                seed=202608150,
            )
            self.index_pool_history_scores = seeded(
                (index_pool_rows - 1, cfg.index_head_dim),
                dtype=torch.float32,
                seed=202608160,
                scale=0.25,
            )
            self.index_pooled = torch.empty(
                (cfg.index_head_dim,), dtype=torch.bfloat16, device=d
            )
            self.index_pooled_norm = torch.empty_like(self.index_pooled)
            self.index_pooled_rope = torch.empty_like(self.index_pooled).reshape(1, -1)

        self.router_logits = torch.empty(
            (cfg.num_experts,), dtype=torch.bfloat16, device=d
        )
        self.route_indices = torch.empty((8,), dtype=torch.int32, device=d)
        self.route_weights = torch.empty((8,), dtype=torch.float32, device=d)
        self.zero_bias = torch.zeros(
            (cfg.num_experts,), dtype=torch.float32, device=d
        )
        self.zero_fill_gate = torch.zeros(
            (1,), dtype=torch.uint32, device=d
        )
        self.zero_hash = torch.zeros((8,), dtype=torch.int32, device=d)

        self.routed_input = torch.empty(
            (cfg.experts_per_token, cfg.hidden_size // 256, 3072),
            dtype=torch.uint8,
            device=d,
        )
        self.routed_middle = torch.empty(
            (cfg.experts_per_token, cfg.expert_intermediate_size),
            dtype=torch.bfloat16,
            device=d,
        )
        self.routed_middle_packed = torch.empty(
            (
                cfg.experts_per_token,
                cfg.expert_intermediate_size // 256,
                3072,
            ),
            dtype=torch.uint8,
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
        if self.ffn_fp32_tma:
            self.shared_middle_native_fp8 = torch.empty(
                (cfg.expert_intermediate_size // 128, 2048),
                dtype=torch.uint8,
                device=d,
            )
            self.ffn_accumulator = torch.empty(
                (1, cfg.hidden_size), dtype=torch.float32, device=d
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
        *,
        wait_for_previous: bool = True,
        placement: tuple[int, int] | None = None,
        wait_group: str | None = None,
        release_group: str | None = None,
    ) -> Stage:
        assignment = self.policy.quantize(source.numel(), 128)
        base_sm = None
        if placement is not None:
            base_sm, num_sms = placement
            assignment = replace(assignment, num_sms=num_sms)
        return self._stage(
            name,
            SchedDsv4Fp8Quant128(source.reshape(-1), output.reshape(-1), scale),
            assignment,
            wait_for_previous=wait_for_previous,
            base_sm=base_sm,
            wait_group=wait_group,
            release_group=release_group,
        )

    def _native_fp8_quant_stage(
        self,
        name: str,
        source: torch.Tensor,
        output: torch.Tensor,
        *,
        wait_for_previous: bool = True,
        placement: tuple[int, int] | None = None,
        wait_group: str | None = None,
        release_group: str | None = None,
    ) -> Stage:
        assignment = self.policy.quantize(source.numel(), 128)
        scale_pack = self.args.fp8_umma_scale_pack
        scale_groups = source.numel() // (128 * scale_pack)
        assignment = replace(
            assignment, num_sms=min(assignment.num_sms, scale_groups)
        )
        base_sm = None
        if placement is not None:
            base_sm, num_sms = placement
            if not 0 < num_sms <= scale_groups:
                raise ValueError(
                    "native FP8 quant placement must fit its scale groups"
                )
            assignment = replace(assignment, num_sms=num_sms)
        return self._stage(
            name,
            SchedDsv4Fp8QuantUmmaB(
                source.reshape(-1), output, scale_pack
            ),
            assignment,
            wait_for_previous=wait_for_previous,
            base_sm=base_sm,
            wait_group=wait_group,
            release_group=release_group,
        )

    def _rms_native_fp8_quant_stage(
        self,
        name: str,
        family: LayerFamily,
        source: torch.Tensor,
        output: torch.Tensor,
        *,
        weight_suffix: str,
        placement: tuple[int, int] | None = None,
        wait_group: str | None = None,
        release_group: str | None = None,
    ) -> Stage:
        weights = self._family_tensors(family, weight_suffix)
        scale_pack = self.args.fp8_umma_scale_pack
        scale_groups = source.numel() // (128 * scale_pack)
        schedule = SchedDsv4RmsFp8QuantUmmaB(
            source.reshape(-1),
            weights[0],
            output,
            self.config.rms_epsilon,
            scale_pack,
        )
        schedule = self._layered(schedule, family, weights)
        assignment = self.policy.quantize(source.numel(), 128)
        assignment = replace(
            assignment, num_sms=min(assignment.num_sms, scale_groups)
        )
        base_sm = None
        if placement is not None:
            base_sm, num_sms = placement
            if not 0 < num_sms <= scale_groups:
                raise ValueError(
                    "fused RMS/native-FP8 placement must fit its scale groups"
                )
            assignment = replace(assignment, num_sms=num_sms)
        return self._stage(
            name,
            schedule,
            assignment,
            base_sm=base_sm,
            wait_group=wait_group,
            release_group=release_group,
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
        wait_for_previous: bool = True,
        placement: tuple[int, int] | None = None,
        wait_group: str | None = None,
        release_group: str | None = None,
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
        base_sm = None
        if placement is not None:
            base_sm, num_sms = placement
            assignment = replace(assignment, num_sms=num_sms)
        return self._stage(
            name,
            schedule,
            assignment,
            wait_for_previous=wait_for_previous,
            base_sm=base_sm,
            wait_group=wait_group,
            release_group=release_group,
        )

    def _fp8_bf16_linear_stage(
        self,
        name: str,
        family: LayerFamily,
        suffix: str,
        activation: torch.Tensor,
        output: torch.Tensor,
        *,
        row_slice: slice | None = None,
        placement: tuple[int, int] | None = None,
        wait_group: str | None = None,
        release_group: str | None = None,
        prefetch_before_wait: bool = False,
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
            stop = (
                linears[0].weight.shape[0]
                if row_slice.stop is None
                else row_slice.stop
            )
            if start % 128 or stop % 128:
                raise ValueError("fused FP8 family slices must be 128-row aligned")
            weights = tuple(linear.weight[row_slice] for linear in linears)
            scales = tuple(
                linear.scale[start // 128 : stop // 128] for linear in linears
            )
        schedule = SchedFp8Block128GemvBf16(
            weights[0],
            scales[0],
            activation.reshape(-1),
            output.reshape(-1),
        )
        schedule = self._layered(schedule, family, weights, scales)
        assignment = self.policy.fp8_gemv(output.numel(), activation.numel())
        base_sm = None
        if placement is not None:
            base_sm, num_sms = placement
            assignment = replace(assignment, num_sms=num_sms)
        return self._stage(
            name,
            schedule,
            assignment,
            input_role="activation" if prefetch_before_wait else None,
            base_sm=base_sm,
            wait_group=wait_group,
            release_group=release_group,
            prefetch_before_wait=prefetch_before_wait,
        )

    def _native_fp8_linear_stage(
        self,
        name: str,
        family: LayerFamily,
        suffix: str,
        activation: torch.Tensor,
        output: torch.Tensor,
        *,
        wait_for_previous: bool = True,
        placement: tuple[int, int] | None = None,
        wait_group: str | None = None,
        release_group: str | None = None,
    ) -> Stage:
        linears = tuple(
            self.checkpoint.load_native_fp8_linear(
                f"layers.{layer_id}.{suffix}", device=self.device
            )
            for layer_id in family.layer_ids
        )
        weights = tuple(linear.weight_tiles for linear in linears)
        scale_packs = {linear.scale_pack for linear in linears}
        if len(scale_packs) != 1:
            raise ValueError("layered native FP8 weights must share scale packing")
        scale_pack = scale_packs.pop()
        schedule = SchedFp8GemvUmmaStream(
            weights[0],
            activation,
            output.reshape(-1),
            scale_pack,
            self.args.fp8_umma_output_group_size,
        )
        schedule = self._layered(schedule, family, weights)
        assignment = self.policy.fp8_umma_gemv(
            output.numel(), activation.shape[0] * 128
        )
        base_sm = None
        if placement is not None:
            base_sm, num_sms = placement
            assignment = replace(assignment, num_sms=num_sms)
        return self._stage(
            name,
            schedule,
            assignment,
            wait_for_previous=wait_for_previous,
            base_sm=base_sm,
            wait_group=wait_group,
            release_group=release_group,
        )

    def _splitk_fp8_linear_stages(
        self,
        name: str,
        family: LayerFamily,
        suffix: str,
        activation: torch.Tensor,
        output: torch.Tensor,
        *,
        row_slice: slice | None = None,
        base_sm: int = 0,
        split_k: int | None = None,
        num_sms: int | None = None,
        wait_group: str | None = None,
        release_group: str | None = None,
        fp32_finalizer=None,
    ) -> list[Stage]:
        linears = tuple(
            self.checkpoint.load_native_fp8_linear(
                f"layers.{layer_id}.{suffix}", device=self.device
            )
            for layer_id in family.layer_ids
        )
        scale_packs = {linear.scale_pack for linear in linears}
        if len(scale_packs) != 1:
            raise ValueError("layered native FP8 weights must share scale packing")
        scale_pack = scale_packs.pop()
        if row_slice is None:
            weights = tuple(linear.weight_tiles for linear in linears)
        else:
            start = 0 if row_slice.start is None else row_slice.start
            stop = linears[0].weight_tiles.shape[0] * 128 if row_slice.stop is None else row_slice.stop
            if start % 128 or stop % 128:
                raise ValueError("native split-K slices must be M128 aligned")
            weights = tuple(
                linear.weight_tiles[start // 128 : stop // 128]
                for linear in linears
            )
        rows = weights[0].shape[0] * 128
        k = activation.shape[0] * 128
        if output.numel() != rows:
            raise ValueError("native split-K output size must match selected rows")
        policy_split, policy_sms = self.policy.fp8_umma_split_k(rows, k)
        split_k = policy_split if split_k is None else int(split_k)
        if k // 128 % split_k:
            raise ValueError("split-K override must divide K tiles")
        work_tiles = rows // 128 * split_k
        num_sms = policy_sms if num_sms is None else int(num_sms)
        if not 0 < num_sms <= work_tiles:
            raise ValueError("split-K SM override exceeds logical work tiles")
        output_vector = output.reshape(-1)
        if split_k == 1:
            schedule = SchedFp8GemvUmmaStream(
                weights[0],
                activation,
                output_vector,
                scale_pack,
                self.args.fp8_umma_output_group_size,
            )
            schedule = self._layered(schedule, family, weights)
            return [
                self._stage(
                    name,
                    schedule,
                    num_sms,
                    base_sm=base_sm,
                    wait_group=wait_group,
                    release_group=release_group,
                )
            ]
        if self.direct_splitk_bf16:
            if fp32_finalizer is not None:
                raise ValueError(
                    "custom FP32 finalizer requires FP32 split-K reduction"
                )
            accumulator = output_vector.reshape(1, rows)
            partial_release_group = release_group
        else:
            if self._active_splitk_workspace is None:
                raise ValueError(
                    "split-K projection requires an active accumulator workspace"
                )
            start = self._active_splitk_offset
            stop = start + rows
            if stop > self._active_splitk_workspace.numel():
                raise ValueError("split-K accumulator workspace is too small")
            accumulator = self._active_splitk_workspace[start:stop].reshape(
                1, rows
            )
            self._active_splitk_offset = stop
            partial_release_group = f"{family.name}.{name}.reduce.ready"
        output_reduce = TmaTensor(
            self.launcher, accumulator
        ).rowmajor_2d("reduce", 1, 128)
        schedule = SchedFp8GemvUmmaSplitK(
            weights[0],
            activation,
            output_reduce,
            split_k,
            scale_pack,
            self.args.fp8_umma_output_group_size,
        )
        schedule = self._layered(schedule, family, weights)
        gemv = self._stage(
            f"{name}.partial",
            schedule,
            num_sms,
            base_sm=base_sm,
            wait_group=wait_group,
            release_group=partial_release_group,
        )
        if self.direct_splitk_bf16:
            return [gemv]
        if fp32_finalizer is None:
            finalize_schedule = SchedDsv4Fp32ToBf16(
                accumulator.reshape(-1), output_vector
            )
            finalize_sms = min(self.sms, rows // 128)
        else:
            finalize_schedule, finalize_sms = fp32_finalizer(
                accumulator.reshape(-1)
            )
        finalize = self._stage(
            name,
            finalize_schedule,
            finalize_sms,
            base_sm=base_sm,
            wait_group=partial_release_group,
            release_group=release_group,
        )
        return [gemv, finalize]

    def _native_fp8_reduce_stage(
        self,
        name: str,
        family: LayerFamily,
        suffix: str,
        activation: torch.Tensor,
        output_reduce,
        *,
        split_k: int,
        placement: tuple[int, int] | None = None,
        wait_group: str | None = None,
        release_group: str | None = None,
    ) -> Stage:
        """Write native-FP8 UMMA accumulators directly through TMA reduce."""

        linears = tuple(
            self.checkpoint.load_native_fp8_linear(
                f"layers.{layer_id}.{suffix}", device=self.device
            )
            for layer_id in family.layer_ids
        )
        weights = tuple(linear.weight_tiles for linear in linears)
        scale_packs = {linear.scale_pack for linear in linears}
        if len(scale_packs) != 1:
            raise ValueError("layered native FP8 weights must share scale packing")
        scale_pack = scale_packs.pop()
        schedule = SchedFp8GemvUmmaSplitK(
            weights[0],
            activation,
            output_reduce,
            split_k,
            scale_pack,
            self.args.fp8_umma_output_group_size,
        )
        schedule = self._layered(schedule, family, weights)
        rows = weights[0].shape[0] * 128
        if placement is None:
            k = activation.shape[0] * 128
            _, policy_sms = self.policy.fp8_umma_split_k(rows, k)
            base_sm = 0
            num_sms = min(policy_sms, rows // 128 * split_k)
        else:
            base_sm, num_sms = placement
        return self._stage(
            name,
            schedule,
            num_sms,
            base_sm=base_sm,
            wait_group=wait_group,
            release_group=release_group,
        )

    def _bf16_linear_stage(
        self,
        name: str,
        family: LayerFamily,
        suffix: str,
        source: torch.Tensor,
        output: torch.Tensor,
        *,
        wait_for_previous: bool = True,
        placement: tuple[int, int] | None = None,
        wait_group: str | None = None,
        release_group: str | None = None,
    ) -> Stage:
        weights = self._family_tensors(family, suffix)
        schedule = SchedDsv4Bf16Gemv(
            weights[0], source.reshape(-1), output.reshape(-1)
        )
        schedule = self._layered(schedule, family, weights)
        assignment = self.policy.bf16_gemv(output.numel(), source.numel())
        base_sm = None
        if placement is not None:
            base_sm, num_sms = placement
            assignment = replace(assignment, num_sms=num_sms)
        return self._stage(
            name,
            schedule,
            assignment,
            wait_for_previous=wait_for_previous,
            base_sm=base_sm,
            wait_group=wait_group,
            release_group=release_group,
        )

    def _fused_bf16_weights(
        self,
        family: LayerFamily,
        suffixes: tuple[str, ...],
    ) -> torch.Tensor:
        key = (family.representative, family.layer_ids, suffixes)
        cached = self._fused_bf16_weight_cache.get(key)
        if cached is not None:
            return cached
        columns = tuple(
            self._family_tensors(family, suffix) for suffix in suffixes
        )
        per_layer = tuple(
            torch.cat(
                tuple(column[layer_index] for column in columns), dim=0
            ).contiguous()
            for layer_index in range(len(family.layer_ids))
        )
        fused = (
            per_layer[0]
            if len(per_layer) == 1
            else torch.stack(per_layer, dim=0).contiguous()
        )
        self._fused_bf16_weight_cache[key] = fused
        return fused

    def _grouped_bf16_splitk_stage(
        self,
        name: str,
        family: LayerFamily,
        suffixes: tuple[str, ...],
        source: torch.Tensor,
        output: torch.Tensor,
        *,
        split_k: int,
        base_sm: int,
        wait_group: str,
        release_group: str | None = None,
    ) -> Stage:
        weights = self._fused_bf16_weights(family, suffixes)
        rows, k = weights.shape[-2:]
        if output.numel() != rows:
            raise ValueError("grouped BF16 projection output must match fused rows")
        output_matrix = output.reshape(rows // 128, 128)
        weight_tma = TmaTensor(
            self.launcher, weights
        ).wgmma_load(128, 128, Major.K)
        output_reduce = TmaTensor(
            self.launcher, output_matrix
        ).rowmajor_2d("reduce", 4, 128)
        schedule = SchedDsv4Bf16GemvGroup4SplitK(
            weights,
            weight_tma,
            source.reshape(-1),
            output_reduce,
            split_k,
            layer_indexed_weight=weights.ndim == 3,
        )
        work_items = rows // 512 * split_k
        return self._stage(
            name,
            schedule,
            work_items,
            base_sm=base_sm,
            wait_group=wait_group,
            release_group=release_group,
        )

    def _rms_stage(
        self,
        name: str,
        source: torch.Tensor,
        output: torch.Tensor,
        *,
        family: LayerFamily | None = None,
        weight_suffix: str | None = None,
        base_sm: int | None = None,
        wait_group: str | None = None,
        release_group: str | None = None,
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
        return self._stage(
            name,
            schedule,
            rows.shape[0],
            base_sm=base_sm,
            wait_group=wait_group,
            release_group=release_group,
        )

    def _rms_rope_stage(
        self,
        name: str,
        source: torch.Tensor,
        table: torch.Tensor,
        output: torch.Tensor,
        *,
        family: LayerFamily | None = None,
        weight_suffix: str | None = None,
        base_sm: int | None = None,
        wait_group: str | None = None,
        release_group: str | None = None,
    ) -> Stage:
        rows = source.reshape(-1, 512)
        out_rows = output.reshape_as(rows)
        weights = None
        weight = None
        if weight_suffix is not None:
            if family is None:
                weight = self._tensor(weight_suffix)
            else:
                weights = self._family_tensors(family, weight_suffix)
                weight = weights[0]
        schedule = SchedDsv4RmsRope512_64(
            rows,
            table,
            out_rows,
            epsilon=self.config.rms_epsilon,
            weight=weight,
            fixed_table_id=self.resident_rope_table_ids[table.data_ptr()],
        )
        if weights is not None:
            schedule = self._layered(schedule, family, weights)
        return self._stage(
            name,
            schedule,
            rows.shape[0],
            base_sm=base_sm,
            wait_group=wait_group,
            release_group=release_group,
        )

    def _hc_stages(
        self,
        family: LayerFamily,
        branch_name: str,
        residual: torch.Tensor,
        output_residual: torch.Tensor,
        *,
        branch: torch.Tensor | None = None,
        zero_fp32_output: torch.Tensor | None = None,
    ) -> tuple[list[Stage], Stage]:
        branch = self.branch if branch is None else branch
        functions = self._family_tensors(family, f"hc_{branch_name}_fn")
        scales = self._family_tensors(family, f"hc_{branch_name}_scale")
        bases = self._family_tensors(family, f"hc_{branch_name}_base")
        norm_weights = self._family_tensors(
            family, f"{branch_name}_norm.weight"
        )
        project = SchedDsv4Fp32Bf16Gemv(
            functions[0], residual.reshape(-1), self.mixes
        )
        project = self._layered(project, family, functions)
        project_stage = self._stage(
            f"{branch_name}.hc_project",
            project,
            self.policy.fp32_bf16_gemv(24, residual.numel()),
        )
        pre = SchedDsv4HcPreRms(
            residual,
            self.mixes,
            scales[0],
            bases[0],
            norm_weights[0],
            self.norm_hidden,
            self.post,
            self.comb,
            zero_fp32_output=zero_fp32_output,
            rms_epsilon=self.config.rms_epsilon,
        )
        pre = self._layered(
            pre, family, scales, bases, norm_weights
        )
        pre_stage = self._stage(f"{branch_name}.hc_pre_rms4096", pre)
        post_stage = self._stage(
            f"{branch_name}.hc_post",
            SchedDsv4HcPost(
                branch,
                residual,
                self.post,
                self.comb,
                output_residual,
            ),
            self.policy.hc_post(
                self.config.hidden_size, self.config.hc_mult
            ),
        )
        return [project_stage, pre_stage], post_stage

    def _build_attention(self, family: LayerFamily) -> list[Stage]:
        cfg = self.config
        layer_id = family.representative
        kind = cfg.attention_kind(layer_id)
        plan = self.attention_plans[kind]
        rope_table = self.main_rope if kind == "swa" else self.compress_rope
        q_placement = self.policy.weighted_parallel_partition(
            0, (cfg.q_lora_rank, cfg.head_dim)
        )
        kv_placement = self.policy.weighted_parallel_partition(
            1, (cfg.q_lora_rank, cfg.head_dim)
        )
        q_base, q_sms = q_placement
        kv_base, _ = kv_placement
        q_quant_sms = min(
            q_sms,
            self.policy.quantize(cfg.q_lora_rank, 128).num_sms,
        )
        native_q_quant_sms = min(
            q_quant_sms,
            cfg.q_lora_rank // (128 * self.args.fp8_umma_scale_pack),
        )
        qkv_input_ready = f"{family.name}.attn.qkv.input.ready"
        q_a_ready = f"{family.name}.attn.q_a.ready"
        q_norm_ready = f"{family.name}.attn.q_norm.ready"
        kv_ready = f"{family.name}.attn.kv.ready"
        kv_norm_ready = f"{family.name}.attn.kv_norm.ready"
        qkv_prefix_join = f"{family.name}.attn.qkv.prefix.join"
        attention_input_ready = f"{family.name}.attn.input.ready"
        compressor_reset_ready = f"{family.name}.attn.compressor.reset.ready"
        compressor_projection_ready = (
            f"{family.name}.attn.compressor.projection.ready"
        )
        index_compressor_reset_ready = (
            f"{family.name}.index.compressor.reset.ready"
        )
        index_compressor_projection_ready = (
            f"{family.name}.index.compressor.projection.ready"
        )
        index_selection_input_join = (
            f"{family.name}.index.selection.input.join"
        )
        stages, post = self._hc_stages(
            family, "attn", self.residual, self.next_residual
        )
        split_q_a = "q_a" in self.splitk_components
        split_q_b = "q_b" in self.splitk_components
        split_kv = "kv" in self.splitk_components
        split_index_q_b = "index_q_b" in self.splitk_components
        split_o_a = "o_a" in self.splitk_components
        split_o_b = "o_b" in self.splitk_components
        use_grouped_preattention = split_q_a and split_kv and split_q_b
        if use_grouped_preattention:
            stages[-1] = replace(
                stages[-1], release_group=attention_input_ready
            )
        compress_values = self.compress_values
        compress_scores = self.compress_scores
        index_compress_values = self.index_compress_values
        index_compress_scores = self.index_compress_scores
        run_index_selection = kind == "csa" and (
            self.args.index_selection_mode == "force"
            or plan.requires_index_selection
        )
        split_index_active = split_index_q_b and run_index_selection
        workspace_rows = (
            cfg.q_lora_rank * int(split_q_a)
            + cfg.head_dim * int(split_kv)
            + cfg.num_heads * cfg.head_dim * int(split_q_b)
            + cfg.index_heads * cfg.index_head_dim * int(split_index_active)
            + cfg.o_groups * cfg.o_lora_rank * int(split_o_a)
            + cfg.hidden_size * int(split_o_b)
        )
        if workspace_rows:
            if self.direct_splitk_bf16:
                workspace = self.splitk_output_arena
                if workspace is None:
                    raise AssertionError("direct split-K output arena is missing")
                self._active_splitk_workspace = None
                self._active_splitk_offset = 0
            else:
                workspace = torch.empty(
                    (workspace_rows,), dtype=torch.float32, device=self.device
                )
                self.splitk_accumulators.append(workspace)
                self._active_splitk_workspace = workspace
                self._active_splitk_offset = 0
            stages.append(
                self._stage(
                    "attn.projections.reset",
                    SchedDsv4ZeroFill(self.zero_fill_gate, workspace),
                    min(self.sms, workspace_rows // 4),
                    wait_group=(
                        attention_input_ready
                        if use_grouped_preattention
                        else None
                    ),
                    release_group=(
                        qkv_input_ready
                        if use_grouped_preattention
                        else None
                    ),
                )
            )
        else:
            self._active_splitk_workspace = None
            self._active_splitk_offset = 0
        need_native_hidden = split_q_a or split_kv
        need_scalar_hidden = not split_q_a or not split_kv
        if need_native_hidden:
            stages.append(
                self._native_fp8_quant_stage(
                    "attn.hidden.quant_native_fp8",
                    self.norm_hidden,
                    self.hidden_native_fp8,
                    wait_group=(
                        attention_input_ready
                        if use_grouped_preattention
                        else None
                    ),
                    release_group=qkv_input_ready,
                )
            )
        if need_scalar_hidden:
            stages.append(
                self._fp8_quant_stage(
                    "attn.hidden.quant_fp8",
                    self.norm_hidden,
                    self.hidden_fp8,
                    self.hidden_fp8_scale,
                    wait_for_previous=not need_native_hidden,
                    wait_group=(
                        attention_input_ready
                        if use_grouped_preattention
                        else None
                    ),
                    release_group=qkv_input_ready,
                )
            )
        need_native_q_rank = (
            split_q_b
            or split_index_active
            or self.args.fp8_qb_mode == "native"
        )
        need_scalar_q_rank = (
            (not split_q_b or not split_index_q_b)
            and self.args.fp8_qb_mode == "scalar"
        )
        fuse_q_rank_splitk_epilogue = (
            split_q_a
            and not self.direct_splitk_bf16
            and need_native_q_rank
            and not need_scalar_q_rank
        )
        q_rank_fp32_finalizer = None
        if fuse_q_rank_splitk_epilogue:
            q_norm_weights = self._family_tensors(
                family, "attn.q_norm.weight"
            )

            def q_rank_fp32_finalizer(accumulator):
                schedule = SchedDsv4Fp32RmsFp8QuantUmmaB(
                    accumulator,
                    q_norm_weights[0],
                    self.q_rank_native_fp8,
                    self.config.rms_epsilon,
                    self.args.fp8_umma_scale_pack,
                )
                schedule = self._layered(
                    schedule, family, q_norm_weights
                )
                return schedule, native_q_quant_sms

        if split_q_a:
            stages.extend(
                self._splitk_fp8_linear_stages(
                    "attn.q_a",
                    family,
                    "attn.wq_a",
                    self.hidden_native_fp8,
                    self.q_rank,
                    base_sm=q_base,
                    wait_group=qkv_input_ready,
                    release_group=(
                        qkv_prefix_join
                        if fuse_q_rank_splitk_epilogue
                        else q_a_ready
                    ),
                    fp32_finalizer=q_rank_fp32_finalizer,
                )
            )
        else:
            stages.append(
                self._fp8_linear_stage(
                    "attn.q_a",
                    family,
                    "attn.wq_a",
                    self.hidden_fp8,
                    self.hidden_fp8_scale,
                    self.q_rank,
                    placement=q_placement,
                    wait_group=qkv_input_ready,
                    release_group=q_a_ready,
                )
            )
        if fuse_q_rank_splitk_epilogue:
            pass
        elif need_native_q_rank and not need_scalar_q_rank:
            stages.append(
                self._rms_native_fp8_quant_stage(
                    "attn.q_rank.rms_quant_native_fp8",
                    family,
                    self.q_rank,
                    self.q_rank_native_fp8,
                    weight_suffix="attn.q_norm.weight",
                    placement=(q_base, native_q_quant_sms),
                    wait_group=q_a_ready,
                    release_group=qkv_prefix_join,
                )
            )
        else:
            stages.append(
                self._rms_stage(
                    "attn.q_norm",
                    self.q_rank,
                    self.q_rank_norm,
                    family=family,
                    weight_suffix="attn.q_norm.weight",
                    base_sm=q_base,
                    wait_group=q_a_ready,
                    release_group=q_norm_ready,
                )
            )
            if need_native_q_rank:
                stages.append(
                    self._native_fp8_quant_stage(
                        "attn.q_rank.quant_native_fp8",
                        self.q_rank_norm,
                        self.q_rank_native_fp8,
                        placement=(q_base, native_q_quant_sms),
                        wait_group=q_norm_ready,
                        release_group=qkv_prefix_join,
                    )
                )
            if need_scalar_q_rank:
                stages.append(
                    self._fp8_quant_stage(
                        "attn.q_rank.quant_fp8",
                        self.q_rank_norm,
                        self.q_rank_fp8,
                        self.q_rank_fp8_scale,
                        wait_for_previous=not need_native_q_rank,
                        placement=(q_base, q_quant_sms),
                        wait_group=q_norm_ready,
                        release_group=qkv_prefix_join,
                    )
                )
        fuse_kv_splitk_epilogue = (
            split_kv and not self.direct_splitk_bf16
        )
        kv_fp32_finalizer = None
        if fuse_kv_splitk_epilogue:
            kv_norm_weights = self._family_tensors(
                family, "attn.kv_norm.weight"
            )

            def kv_fp32_finalizer(accumulator):
                schedule = SchedDsv4Fp32RmsRope512_64(
                    accumulator.reshape(1, cfg.head_dim),
                    rope_table,
                    self.current_kv_rows[kind],
                    epsilon=cfg.rms_epsilon,
                    weight=kv_norm_weights[0],
                    fixed_table_id=self.resident_rope_table_ids[
                        rope_table.data_ptr()
                    ],
                )
                schedule = self._layered(
                    schedule, family, kv_norm_weights
                )
                return schedule, 1

        if split_kv:
            stages.extend(
                self._splitk_fp8_linear_stages(
                    "attn.kv",
                    family,
                    "attn.wkv",
                    self.hidden_native_fp8,
                    self.kv,
                    base_sm=kv_base,
                    wait_group=qkv_input_ready,
                    release_group=(
                        qkv_prefix_join
                        if fuse_kv_splitk_epilogue
                        else kv_ready
                    ),
                    fp32_finalizer=kv_fp32_finalizer,
                )
            )
        else:
            stages.append(
                self._fp8_linear_stage(
                    "attn.kv",
                    family,
                    "attn.wkv",
                    self.hidden_fp8,
                    self.hidden_fp8_scale,
                    self.kv,
                    placement=kv_placement,
                    wait_group=qkv_input_ready,
                    release_group=kv_ready,
                )
            )
        if not fuse_kv_splitk_epilogue:
            stages.append(
                self._rms_rope_stage(
                    "attn.kv_rms_rope",
                    self.kv,
                    rope_table,
                    self.current_kv_rows[kind],
                    family=family,
                    weight_suffix="attn.kv_norm.weight",
                    base_sm=kv_base,
                    wait_group=kv_ready,
                    release_group=qkv_prefix_join,
                )
            )
        if use_grouped_preattention and kind in ("csa", "hca"):
            width = cfg.head_dim * (2 if kind == "csa" else 1)
            fused_output = self.compress_fused_projection[: 2 * width]
            compress_values = fused_output[:width]
            compress_scores = fused_output[width:]
            _, q_prefix_sms = self.policy.fp8_umma_split_k(
                cfg.q_lora_rank, cfg.hidden_size
            )
            compressor_base = q_base + q_prefix_sms
            compressor_sms = 2 * width // 512 * 8
            stages.append(
                self._stage(
                    "attn.compressor.projection_reset",
                    SchedDsv4ZeroFill(self.zero_fill_gate, fused_output),
                    compressor_sms,
                    base_sm=compressor_base,
                    wait_group=attention_input_ready,
                    release_group=compressor_reset_ready,
                )
            )
            stages.append(
                self._grouped_bf16_splitk_stage(
                    "attn.compressor.wkv_wgate_group4",
                    family,
                    (
                        "attn.compressor.wkv.weight",
                        "attn.compressor.wgate.weight",
                    ),
                    self.norm_hidden,
                    fused_output,
                    split_k=8,
                    base_sm=compressor_base,
                    wait_group=compressor_reset_ready,
                    release_group=compressor_projection_ready,
                )
            )
        if use_grouped_preattention and kind == "csa":
            fused_index_output = self.index_compress_fused_projection.reshape(-1)
            index_compress_values = fused_index_output[: 2 * cfg.index_head_dim]
            index_compress_scores = fused_index_output[2 * cfg.index_head_dim :]
            _, kv_prefix_sms = self.policy.fp8_umma_split_k(
                cfg.head_dim, cfg.hidden_size
            )
            index_compressor_base = kv_base + kv_prefix_sms
            index_compressor_sms = 8
            stages.append(
                self._stage(
                    "index.compressor.projection_reset",
                    SchedDsv4ZeroFill(
                        self.zero_fill_gate, fused_index_output
                    ),
                    index_compressor_sms,
                    base_sm=index_compressor_base,
                    wait_group=attention_input_ready,
                    release_group=index_compressor_reset_ready,
                )
            )
            stages.append(
                self._grouped_bf16_splitk_stage(
                    "index.compressor.wkv_wgate_group4",
                    family,
                    (
                        "attn.indexer.compressor.wkv.weight",
                        "attn.indexer.compressor.wgate.weight",
                    ),
                    self.norm_hidden,
                    fused_index_output,
                    split_k=8,
                    base_sm=index_compressor_base,
                    wait_group=index_compressor_reset_ready,
                    release_group=index_compressor_projection_ready,
                )
            )
        if (
            use_grouped_preattention
            and kind in ("csa", "hca")
            and plan.should_compress
        ):
            ape_tensors = self._family_tensors(
                family, "attn.compressor.ape"
            )
            tail_offset = cfg.head_dim if plan.compress_ratio == 4 else 0
            ape_rows = tuple(
                ape[
                    self.decode_position % plan.compress_ratio,
                    tail_offset : tail_offset + cfg.head_dim,
                ]
                for ape in ape_tensors
            )
            history_values = self.attention_pool_history_values[kind]
            tail_values = compress_values[
                tail_offset : tail_offset + cfg.head_dim
            ]
            tail_scores = compress_scores[
                tail_offset : tail_offset + cfg.head_dim
            ]
            norm_weights = self._family_tensors(
                family, "attn.compressor.norm.weight"
            )
            use_packed_pool = (
                self.args.gated_pool_mode == "packed"
                or (
                    self.args.gated_pool_mode == "auto"
                    and plan.compress_ratio == 128
                )
            )
            if use_packed_pool:
                compressor_pool_partial_ready = (
                    f"{family.name}.attn.compressor.pool.partial.ready"
                )
                history_pool_base = compressor_base + compressor_sms
                history_pool = SchedDsv4GatedPoolPacked8HistoryState(
                    self.attention_pool_history_packed[kind],
                    history_values.shape[0],
                    self.attention_pool_history_state[kind],
                )
                stages.append(
                    self._stage(
                        "attn.compressor.history_pool_state",
                        history_pool,
                        cfg.head_dim // 128,
                        base_sm=history_pool_base,
                        wait_group=attention_input_ready,
                    )
                )
                merge_tail = SchedDsv4GatedPoolTailRmsPartial(
                    self.attention_pool_history_state[kind],
                    self.attention_pooled_fp32[kind],
                    self.attention_pooled_rms_partials[kind],
                    tail_values=tail_values,
                    tail_scores=tail_scores,
                    tail_bias=ape_rows[0],
                )
                merge_tail = self._layered(
                    merge_tail, family, ape_rows
                )
                stages.append(
                    self._stage(
                        "attn.compressor.tail_rms_partial",
                        merge_tail,
                        cfg.head_dim // 128,
                        base_sm=history_pool_base,
                        wait_group=compressor_projection_ready,
                        release_group=compressor_pool_partial_ready,
                    )
                )
                finalize = SchedDsv4Fp32RmsRopeShard128(
                    self.attention_pooled_fp32[kind],
                    self.attention_pooled_rms_partials[kind],
                    norm_weights[0],
                    self.compressed_output_rope[kind],
                    self.current_compressed_rows[kind],
                    epsilon=cfg.rms_epsilon,
                    fixed_table_id=self.resident_rope_table_ids[
                        self.compressed_output_rope[kind].data_ptr()
                    ],
                )
                finalize = self._layered(
                    finalize, family, norm_weights
                )
                stages.append(
                    self._stage(
                        "attn.compressor.norm_rope_shard4",
                        finalize,
                        cfg.head_dim // 128,
                        base_sm=history_pool_base,
                        wait_group=compressor_pool_partial_ready,
                    )
                )
            else:
                pool = SchedDsv4GatedPoolRmsRope(
                    history_values,
                    self.attention_pool_history_scores[kind],
                    norm_weights[0],
                    self.compressed_output_rope[kind],
                    self.current_compressed_rows[kind],
                    epsilon=cfg.rms_epsilon,
                    tail_values=tail_values,
                    tail_scores=tail_scores,
                    tail_bias=ape_rows[0],
                    fixed_table_id=self.resident_rope_table_ids[
                        self.compressed_output_rope[kind].data_ptr()
                    ],
                )
                pool = self._layered(
                    pool, family, ape_rows, norm_weights
                )
                stages.append(
                    self._stage(
                        "attn.compressor.pool_norm_rope",
                        pool,
                        base_sm=compressor_base,
                        wait_group=compressor_projection_ready,
                    )
                )
        if (
            use_grouped_preattention
            and kind == "csa"
            and plan.should_compress
        ):
            index_ape_tensors = self._family_tensors(
                family, "attn.indexer.compressor.ape"
            )
            index_ape_rows = tuple(
                ape[
                    self.decode_position % plan.compress_ratio,
                    cfg.index_head_dim : 2 * cfg.index_head_dim,
                ]
                for ape in index_ape_tensors
            )
            index_norm_weights = self._family_tensors(
                family, "attn.indexer.compressor.norm.weight"
            )
            index_pool = SchedDsv4GatedPoolRmsRope(
                self.index_pool_history_values,
                self.index_pool_history_scores,
                index_norm_weights[0],
                self.compressed_output_rope[kind],
                self.index_cache[-1:],
                epsilon=cfg.rms_epsilon,
                tail_values=index_compress_values[
                    cfg.index_head_dim : 2 * cfg.index_head_dim
                ],
                tail_scores=index_compress_scores[
                    cfg.index_head_dim : 2 * cfg.index_head_dim
                ],
                tail_bias=index_ape_rows[0],
                hadamard=True,
                fixed_table_id=self.resident_rope_table_ids[
                    self.compressed_output_rope[kind].data_ptr()
                ],
            )
            index_pool = self._layered(
                index_pool,
                family,
                index_ape_rows,
                index_norm_weights,
            )
            stages.append(
                self._stage(
                    "index.compressor.pool_norm_rope_hadamard",
                    index_pool,
                    base_sm=index_compressor_base,
                    wait_group=index_compressor_projection_ready,
                    release_group=(
                        index_selection_input_join
                        if run_index_selection
                        else None
                    ),
                )
            )

        # The CSA query/weight/selection branch is independent of the main
        # Q projection once q_a is ready.  Queue it first on a bounded SM band:
        # those SMs execute the index branch while the compressor bands finish,
        # and the remaining SMs can enter q_b immediately.  q_b spans the full
        # grid and therefore provides the attention-ready join for every branch.
        if kind == "csa" and use_grouped_preattention and run_index_selection:
            stages.append(
                self._bf16_linear_stage(
                    "index.weights",
                    family,
                    "attn.indexer.weights_proj.weight",
                    self.norm_hidden,
                    self.index_head_weights,
                    placement=(0, cfg.index_heads),
                    wait_group=attention_input_ready,
                    release_group=index_selection_input_join,
                )
            )
            fuse_index_q_splitk_epilogue = (
                split_index_active and not self.direct_splitk_bf16
            )
            index_q_fp32_finalizer = None
            if fuse_index_q_splitk_epilogue:

                def index_q_fp32_finalizer(accumulator):
                    return (
                        SchedDsv4Fp32RopeHadamard128(
                            accumulator.reshape(
                                cfg.index_heads, cfg.index_head_dim
                            ),
                            self.compress_rope,
                            self.index_q_hadamard,
                            fixed_table_id=self.resident_rope_table_ids[
                                self.compress_rope.data_ptr()
                            ],
                        ),
                        cfg.index_heads,
                    )

            if split_index_active:
                stages.extend(
                    self._splitk_fp8_linear_stages(
                        "index.q_b",
                        family,
                        "attn.indexer.wq_b",
                        self.q_rank_native_fp8,
                        self.index_q,
                        base_sm=0,
                        num_sms=cfg.index_heads,
                        wait_group=qkv_prefix_join,
                        release_group=index_selection_input_join,
                        fp32_finalizer=index_q_fp32_finalizer,
                    )
                )
            elif self.args.fp8_qb_mode == "native":
                stages.append(
                    self._native_fp8_linear_stage(
                        "index.q_b",
                        family,
                        "attn.indexer.wq_b",
                        self.q_rank_native_fp8,
                        self.index_q,
                        placement=(0, cfg.index_heads),
                        wait_group=qkv_prefix_join,
                    )
                )
            else:
                stages.append(
                    self._fp8_linear_stage(
                        "index.q_b",
                        family,
                        "attn.indexer.wq_b",
                        self.q_rank_fp8,
                        self.q_rank_fp8_scale,
                        self.index_q,
                        placement=(0, cfg.index_heads),
                        wait_group=qkv_prefix_join,
                    )
                )
            if not fuse_index_q_splitk_epilogue:
                stages.append(
                    self._stage(
                        "index.q_rope",
                        SchedDsv4Rope128_64(
                            self.index_q,
                            self.compress_rope,
                            self.index_q_rope,
                            fixed_table_id=self.resident_rope_table_ids[
                                self.compress_rope.data_ptr()
                            ],
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
                        release_group=index_selection_input_join,
                    )
                )
            if plan.compressed_rows:
                stages.append(
                    self._stage(
                        "index.score",
                        SchedDsv4IndexScore(
                            self.index_q_hadamard,
                            self.index_cache,
                            self.index_head_weights,
                            self.index_scores,
                        ),
                        min(plan.compressed_rows, self.sms),
                        wait_group=index_selection_input_join,
                    )
                )
                stages.append(
                    self._stage(
                        "index.topk",
                        SchedDsv4TopK512(
                            self.index_scores,
                            self.attention_indices_by_kind[kind][
                                -plan.compressed_selected :
                            ],
                            index_offset=cfg.sliding_window,
                        ),
                    )
                )
        fuse_q_splitk_epilogue = (
            split_q_b and not self.direct_splitk_bf16
        )
        q_fp32_finalizer = None
        if fuse_q_splitk_epilogue:

            def q_fp32_finalizer(accumulator):
                return (
                    SchedDsv4Fp32RmsRope512_64(
                        accumulator.reshape(cfg.num_heads, cfg.head_dim),
                        rope_table,
                        self.q_rope,
                        epsilon=cfg.rms_epsilon,
                        fixed_table_id=self.resident_rope_table_ids[
                            rope_table.data_ptr()
                        ],
                    ),
                    cfg.num_heads,
                )

        if split_q_b:
            stages.extend(
                self._splitk_fp8_linear_stages(
                    "attn.q_b",
                    family,
                    "attn.wq_b",
                    self.q_rank_native_fp8,
                    self.q,
                    wait_group=qkv_prefix_join,
                    fp32_finalizer=q_fp32_finalizer,
                )
            )
        elif self.args.fp8_qb_mode == "native":
            stages.append(
                self._native_fp8_linear_stage(
                    "attn.q_b",
                    family,
                    "attn.wq_b",
                    self.q_rank_native_fp8,
                    self.q,
                    wait_group=qkv_prefix_join,
                )
            )
        else:
            stages.append(
                self._fp8_linear_stage(
                    "attn.q_b",
                    family,
                    "attn.wq_b",
                    self.q_rank_fp8,
                    self.q_rank_fp8_scale,
                    self.q,
                    wait_group=qkv_prefix_join,
                )
            )
        if not fuse_q_splitk_epilogue:
            stages.append(
                self._rms_rope_stage(
                    "attn.q_head_rms_rope",
                    self.q,
                    rope_table,
                    self.q_rope,
                )
            )

        if kind in ("csa", "hca") and not use_grouped_preattention:
            width = cfg.head_dim * (2 if kind == "csa" else 1)
            if not use_grouped_preattention:
                stages.append(
                    self._bf16_linear_stage(
                        "attn.compressor.wkv",
                        family,
                        "attn.compressor.wkv.weight",
                        self.norm_hidden,
                        compress_values[:width],
                    )
                )
                stages.append(
                    self._bf16_linear_stage(
                        "attn.compressor.wgate",
                        family,
                        "attn.compressor.wgate.weight",
                        self.norm_hidden,
                        compress_scores[:width],
                        wait_for_previous=False,
                    )
                )
            if plan.should_compress:
                ape_tensors = self._family_tensors(
                    family, "attn.compressor.ape"
                )
                tail_offset = cfg.head_dim if plan.compress_ratio == 4 else 0
                ape_rows = tuple(
                    ape[
                        self.decode_position % plan.compress_ratio,
                        tail_offset : tail_offset + cfg.head_dim,
                    ]
                    for ape in ape_tensors
                )
                history_values = self.attention_pool_history_values[kind]
                tail_values = compress_values[
                    tail_offset : tail_offset + cfg.head_dim
                ]
                tail_scores = compress_scores[
                    tail_offset : tail_offset + cfg.head_dim
                ]
                use_packed_pool = (
                    self.args.gated_pool_mode == "packed"
                    or (
                        self.args.gated_pool_mode == "auto"
                        and plan.compress_ratio == 128
                    )
                )
                fuse_scalar_pool_epilogue = (
                    use_grouped_preattention and not use_packed_pool
                )
                if fuse_scalar_pool_epilogue:
                    norm_weights = self._family_tensors(
                        family, "attn.compressor.norm.weight"
                    )
                    pool = SchedDsv4GatedPoolRmsRope(
                        history_values,
                        self.attention_pool_history_scores[kind],
                        norm_weights[0],
                        self.compressed_output_rope[kind],
                        self.current_compressed_rows[kind],
                        epsilon=cfg.rms_epsilon,
                        tail_values=tail_values,
                        tail_scores=tail_scores,
                        tail_bias=ape_rows[0],
                        fixed_table_id=self.resident_rope_table_ids[
                            self.compressed_output_rope[kind].data_ptr()
                        ],
                    )
                    pool = self._layered(
                        pool, family, ape_rows, norm_weights
                    )
                    stages.append(
                        self._stage(
                            "attn.compressor.pool_norm_rope", pool
                        )
                    )
                elif use_packed_pool:
                    pool = SchedDsv4GatedPoolPacked8Shard128(
                        self.attention_pool_history_packed[kind],
                        history_values.shape[0],
                        self.attention_pooled[kind],
                        tail_values=tail_values,
                        tail_scores=tail_scores,
                        tail_bias=ape_rows[0],
                    )
                    pool_sms = self.policy.gated_pool(
                        cfg.head_dim,
                        history_values.shape[0] + 1,
                        packed=True,
                    )
                else:
                    pool = SchedDsv4GatedPool(
                        history_values,
                        self.attention_pool_history_scores[kind],
                        self.attention_pooled[kind],
                        tail_values=tail_values,
                        tail_scores=tail_scores,
                        tail_bias=ape_rows[0],
                    )
                    pool_sms = 1
                if not fuse_scalar_pool_epilogue:
                    pool = self._layered(pool, family, ape_rows)
                    stages.append(
                        self._stage(
                            "attn.compressor.pool", pool, pool_sms
                        )
                    )
                    stages.append(
                        self._rms_stage(
                            "attn.compressor.norm",
                            self.attention_pooled[kind],
                            self.attention_pooled_norm[kind],
                            family=family,
                            weight_suffix="attn.compressor.norm.weight",
                        )
                    )
                    stages.append(
                        self._stage(
                            "attn.compressor.rope",
                            SchedDsv4Rope512_64(
                                self.attention_pooled_norm[kind].reshape(1, -1),
                                self.compressed_output_rope[kind],
                                self.current_compressed_rows[kind],
                                fixed_table_id=self.resident_rope_table_ids[
                                    self.compressed_output_rope[kind].data_ptr()
                                ],
                            ),
                        )
                    )

        if kind == "csa" and not use_grouped_preattention:
            fuse_index_q_splitk_epilogue = (
                run_index_selection
                and split_index_active
                and not self.direct_splitk_bf16
            )
            index_q_fp32_finalizer = None
            if fuse_index_q_splitk_epilogue:

                def index_q_fp32_finalizer(accumulator):
                    return (
                        SchedDsv4Fp32RopeHadamard128(
                            accumulator.reshape(
                                cfg.index_heads, cfg.index_head_dim
                            ),
                            self.compress_rope,
                            self.index_q_hadamard,
                            fixed_table_id=self.resident_rope_table_ids[
                                self.compress_rope.data_ptr()
                            ],
                        ),
                        cfg.index_heads,
                    )

            if run_index_selection and split_index_active:
                stages.extend(
                    self._splitk_fp8_linear_stages(
                        "index.q_b",
                        family,
                        "attn.indexer.wq_b",
                        self.q_rank_native_fp8,
                        self.index_q,
                        fp32_finalizer=index_q_fp32_finalizer,
                    )
                )
            elif run_index_selection and self.args.fp8_qb_mode == "native":
                stages.append(
                    self._native_fp8_linear_stage(
                        "index.q_b",
                        family,
                        "attn.indexer.wq_b",
                        self.q_rank_native_fp8,
                        self.index_q,
                    )
                )
            elif run_index_selection:
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
            if run_index_selection and not fuse_index_q_splitk_epilogue:
                stages.append(
                    self._stage(
                        "index.q_rope",
                        SchedDsv4Rope128_64(
                            self.index_q,
                            self.compress_rope,
                            self.index_q_rope,
                            fixed_table_id=self.resident_rope_table_ids[
                                self.compress_rope.data_ptr()
                            ],
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
            if run_index_selection:
                stages.append(
                    self._bf16_linear_stage(
                        "index.weights",
                        family,
                        "attn.indexer.weights_proj.weight",
                        self.norm_hidden,
                        self.index_head_weights,
                    )
                )
            if not use_grouped_preattention:
                stages.append(
                    self._bf16_linear_stage(
                        "index.compressor.wkv",
                        family,
                        "attn.indexer.compressor.wkv.weight",
                        self.norm_hidden,
                        index_compress_values,
                    )
                )
                stages.append(
                    self._bf16_linear_stage(
                        "index.compressor.wgate",
                        family,
                        "attn.indexer.compressor.wgate.weight",
                        self.norm_hidden,
                        index_compress_scores,
                        wait_for_previous=False,
                    )
                )
            if plan.should_compress and not use_grouped_preattention:
                index_ape_tensors = self._family_tensors(
                    family, "attn.indexer.compressor.ape"
                )
                index_ape_rows = tuple(
                    ape[
                        self.decode_position % plan.compress_ratio,
                        cfg.index_head_dim : 2 * cfg.index_head_dim,
                    ]
                    for ape in index_ape_tensors
                )
                index_tail_values = index_compress_values[
                    cfg.index_head_dim : 2 * cfg.index_head_dim
                ]
                index_tail_scores = index_compress_scores[
                    cfg.index_head_dim : 2 * cfg.index_head_dim
                ]
                if use_grouped_preattention:
                    index_norm_weights = self._family_tensors(
                        family,
                        "attn.indexer.compressor.norm.weight",
                    )
                    index_pool = SchedDsv4GatedPoolRmsRope(
                        self.index_pool_history_values,
                        self.index_pool_history_scores,
                        index_norm_weights[0],
                        self.compressed_output_rope[kind],
                        self.index_cache[-1:],
                        epsilon=cfg.rms_epsilon,
                        tail_values=index_tail_values,
                        tail_scores=index_tail_scores,
                        tail_bias=index_ape_rows[0],
                        hadamard=True,
                        fixed_table_id=self.resident_rope_table_ids[
                            self.compressed_output_rope[kind].data_ptr()
                        ],
                    )
                    index_pool = self._layered(
                        index_pool,
                        family,
                        index_ape_rows,
                        index_norm_weights,
                    )
                    stages.append(
                        self._stage(
                            "index.compressor.pool_norm_rope_hadamard",
                            index_pool,
                        )
                    )
                else:
                    index_pool = SchedDsv4GatedPool(
                        self.index_pool_history_values,
                        self.index_pool_history_scores,
                        self.index_pooled,
                        tail_values=index_tail_values,
                        tail_scores=index_tail_scores,
                        tail_bias=index_ape_rows[0],
                    )
                    index_pool = self._layered(
                        index_pool, family, index_ape_rows
                    )
                    stages.append(
                        self._stage("index.compressor.pool", index_pool)
                    )
                    stages.append(
                        self._rms_stage(
                            "index.compressor.norm",
                            self.index_pooled,
                            self.index_pooled_norm,
                            family=family,
                            weight_suffix=(
                                "attn.indexer.compressor.norm.weight"
                            ),
                        )
                    )
                    stages.append(
                        self._stage(
                            "index.compressor.rope",
                            SchedDsv4Rope128_64(
                                self.index_pooled_norm.reshape(1, -1),
                                self.compressed_output_rope[kind],
                                self.index_pooled_rope,
                                fixed_table_id=self.resident_rope_table_ids[
                                    self.compressed_output_rope[kind].data_ptr()
                                ],
                            ),
                        )
                    )
                    stages.append(
                        self._stage(
                            "index.compressor.hadamard",
                            SchedDsv4Hadamard(
                                self.index_pooled_rope,
                                self.index_cache[-1:],
                            ),
                        )
                    )
            if run_index_selection and plan.compressed_rows:
                stages.append(
                    self._stage(
                        "index.score",
                        SchedDsv4IndexScore(
                            self.index_q_hadamard,
                            self.index_cache,
                            self.index_head_weights,
                            self.index_scores,
                        ),
                        min(plan.compressed_rows, self.sms),
                    )
                )
                stages.append(
                    self._stage(
                        "index.topk",
                        SchedDsv4TopK512(
                            self.index_scores,
                            self.attention_indices_by_kind[kind][
                                -plan.compressed_selected :
                            ],
                            index_offset=cfg.sliding_window,
                        ),
                    )
                )

        sinks = self._family_tensors(family, "attn.attn_sink")
        attention_rows = self.attention_indices_by_kind[kind].numel()
        use_split_umma_attention = split_o_a and self.args.attention_mode in (
            "auto",
            "umma-split",
        )
        if self.args.attention_mode == "umma-split" and not split_o_a:
            raise ValueError(
                "UMMA split attention requires native split-K O_a"
            )
        if use_split_umma_attention and (
            plan.compressed_selected != plan.compressed_rows
        ):
            raise ValueError(
                "UMMA split attention currently requires exhaustive cache rows"
            )
        use_contiguous_attention = (
            not use_split_umma_attention
            and (
                self.args.attention_mode == "contiguous"
                or (
                    self.args.attention_mode == "auto"
                    and attention_rows >= 16
                )
            )
        )
        output_join_group = f"{family.name}.attn.output.join"
        if use_split_umma_attention:
            num_splits = (attention_rows + 31) // 32
            partials = self.attention_partial_workspace[:num_splits]
            metadata = self.attention_metadata_workspace[:num_splits]
            q_tma = TmaTensor(
                self.launcher, self.q_rope
            ).wgmma_load(64, 128, Major.K)
            k_tma = TmaTensor(
                self.launcher, self.attention_cache[kind]
            ).wgmma_load(32, 128, Major.K)
            v_tma = TmaTensor(
                self.launcher, self.attention_cache[kind]
            ).wgmma_load(32, 128, Major.MN)
            partial_tma = TmaTensor(
                self.launcher,
                partials.reshape(num_splits * cfg.num_heads, cfg.head_dim),
            ).rowmajor_2d("store", cfg.num_heads, 128)
            partial_ready_group = (
                f"{family.name}.attn.split32.partials.ready"
            )
            producer = SchedDsv4AttentionSplit32UmmaSm100(
                self.q_rope,
                self.attention_cache[kind],
                attention_rows,
                partials,
                metadata,
                q_tma=q_tma,
                k_tma=k_tma,
                v_tma=v_tma,
                partial_tma=partial_tma,
            )
            stages.append(
                self._stage(
                    f"attn.sparse_{kind}.split32_umma",
                    producer,
                    num_splits,
                    release_group=partial_ready_group,
                )
            )
            native_heads = self.o_group_native_fp8.view(
                cfg.num_heads, 4, 2048
            )
            for group in range(cfg.o_groups):
                group_input_ready = (
                    f"{family.name}.attn.o_a.g{group}.input.ready"
                )
                reducer = SchedDsv4AttentionSplitReduceFp8Sm100(
                    partials,
                    metadata,
                    sinks[0],
                    rope_table,
                    native_heads,
                    head_start=group * 8,
                    head_count=8,
                )
                reducer = self._layered(reducer, family, sinks)
                stages.append(
                    self._stage(
                        f"attn.sparse_{kind}.reduce_quant_g{group}",
                        reducer,
                        8,
                        base_sm=group * 16,
                        wait_group=partial_ready_group,
                        release_group=group_input_ready,
                    )
                )
                start = group * cfg.o_lora_rank
                stages.extend(
                    self._splitk_fp8_linear_stages(
                        f"attn.o_a.g{group}",
                        family,
                        "attn.wo_a",
                        self.o_group_native_fp8[group],
                        self.o_rank[group],
                        row_slice=slice(start, start + cfg.o_lora_rank),
                        base_sm=group * 16,
                        split_k=2,
                        num_sms=16,
                        wait_group=group_input_ready,
                        release_group=output_join_group,
                    )
                )
        else:
            if use_contiguous_attention:
                if plan.compressed_selected != plan.compressed_rows:
                    raise ValueError(
                        "contiguous attention requires the complete compressed cache"
                    )
                sparse = SchedDsv4ContiguousAttention512Block4(
                    self.q_rope,
                    self.attention_cache[kind],
                    attention_rows,
                    sinks[0],
                    self.attention_output,
                )
            else:
                sparse = SchedDsv4SparseAttention512(
                    self.q_rope,
                    self.attention_cache[kind],
                    self.attention_indices_by_kind[kind],
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
            output_ready_group = f"{family.name}.attn.output.ready"
            stages.append(
                self._stage(
                    "attn.inverse_rope",
                    SchedDsv4Rope512_64(
                        self.attention_output,
                        rope_table,
                        self.attention_inverse,
                        inverse=True,
                        fixed_table_id=self.resident_rope_table_ids[
                            rope_table.data_ptr()
                        ],
                    ),
                    cfg.num_heads,
                    release_group=output_ready_group,
                )
            )
            grouped = self.attention_inverse.reshape(cfg.o_groups, -1)
            for group in range(cfg.o_groups):
                placement = (
                    (group * 16, 16)
                    if split_o_a
                    else self.policy.parallel_partition(group, cfg.o_groups)
                )
                start = group * cfg.o_lora_rank
                if split_o_a:
                    group_input_ready = (
                        f"{family.name}.attn.o_a.g{group}.input.ready"
                    )
                    stages.append(
                        self._native_fp8_quant_stage(
                            f"attn.o_a.g{group}.quant_native_fp8",
                            grouped[group],
                            self.o_group_native_fp8[group],
                            placement=placement,
                            wait_group=output_ready_group,
                            release_group=group_input_ready,
                        )
                    )
                    stages.extend(
                        self._splitk_fp8_linear_stages(
                            f"attn.o_a.g{group}",
                            family,
                            "attn.wo_a",
                            self.o_group_native_fp8[group],
                            self.o_rank[group],
                            row_slice=slice(start, start + cfg.o_lora_rank),
                            base_sm=placement[0],
                            split_k=2,
                            num_sms=placement[1],
                            wait_group=group_input_ready,
                            release_group=output_join_group,
                        )
                    )
                else:
                    stages.append(
                        self._fp8_bf16_linear_stage(
                            f"attn.o_a.g{group}",
                            family,
                            "attn.wo_a",
                            grouped[group],
                            self.o_rank[group],
                            row_slice=slice(start, start + cfg.o_lora_rank),
                            placement=placement,
                            wait_group=output_ready_group,
                            release_group=output_join_group,
                            prefetch_before_wait=(
                                placement[0] >= cfg.num_heads
                            ),
                        )
                    )
        if split_o_b:
            o_rank_ready = f"{family.name}.attn.o_rank.native.ready"
            stages.append(
                self._native_fp8_quant_stage(
                    "attn.o_rank.quant_native_fp8",
                    self.o_rank.reshape(-1),
                    self.o_rank_native_fp8,
                    wait_group=output_join_group,
                    release_group=o_rank_ready,
                )
            )
            stages.extend(
                self._splitk_fp8_linear_stages(
                    "attn.o_b",
                    family,
                    "attn.wo_b",
                    self.o_rank_native_fp8,
                    self.branch,
                    wait_group=o_rank_ready,
                )
            )
        else:
            stages.append(
                self._fp8_quant_stage(
                    "attn.o_rank.quant_fp8",
                    self.o_rank.reshape(-1),
                    self.o_rank_fp8,
                    self.o_rank_scale,
                    wait_group=output_join_group,
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
        if self._active_splitk_workspace is not None:
            if self._active_splitk_offset != self._active_splitk_workspace.numel():
                raise ValueError(
                    "split-K accumulator workspace was not consumed exactly"
                )
            self._active_splitk_workspace = None
            self._active_splitk_offset = 0
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
        shapes = {
            "w1": (cfg.expert_intermediate_size, cfg.hidden_size),
            "w3": (cfg.expert_intermediate_size, cfg.hidden_size),
            "w2": (cfg.hidden_size, cfg.expert_intermediate_size),
        }
        columns: dict[str, list[int]] = {}
        for tag, (rows, _) in shapes.items():
            for tile in range(rows // 128):
                columns[f"{tag}.m{tile}"] = []

        linears = {tag: [] for tag in shapes}
        for expert_id in range(cfg.num_experts):
            prefix = f"layers.{layer_id}.ffn.experts.{expert_id}"
            for tag, (rows, k) in shapes.items():
                linear = self.checkpoint.load_native_nvfp4_linear(
                    f"{prefix}.{tag}", device=self.device
                )
                linears[tag].append(linear)
                expected_shape = (rows // 128, k // 256, 18432)
                if tuple(linear.weight_tiles.shape) != expected_shape:
                    raise ValueError(
                        f"{linear.prefix} native tiles must be {expected_shape}"
                    )
                for tile in range(rows // 128):
                    columns[f"{tag}.m{tile}"].append(
                        linear.weight_tiles[tile].data_ptr()
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
            for tag in shapes
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

    def _routed_native_quant_stage(
        self,
        name: str,
        family: LayerFamily,
        tables: tuple[RoutedAddressTable, ...],
        rank: int,
        field_name: str,
        source: torch.Tensor,
        output: torch.Tensor,
        *,
        base_sm: int | None = None,
        wait_group: str | None = None,
        release_group: str | None = None,
    ) -> Stage:
        representative = tables[0]
        schedule = SchedRoutedDsv4Nvfp4QuantUmmaB(
            representative.state,
            rank,
            representative.field(field_name),
            source.reshape(-1),
            output,
        )
        schedule = self._routed_layered(schedule, family, tables)
        return self._stage(
            name,
            schedule,
            source.numel() // 256,
            input_role="route",
            base_sm=base_sm,
            wait_group=wait_group,
            release_group=release_group,
        )

    def _routed_native_linear_stage(
        self,
        name: str,
        family: LayerFamily,
        tables: tuple[RoutedAddressTable, ...],
        rank: int,
        tag: str,
        rows: int,
        k: int,
        activation: torch.Tensor,
        output: torch.Tensor | None,
        *,
        wait_for_previous: bool = True,
        activation_mode: str = "load",
        output_mode: str = "store",
        output_register: int = 0,
        output_port: int = 0,
        output_scale: torch.Tensor | None = None,
        pipeline: bool = False,
        activation_tiles_per_load: int | None = None,
        placement: tuple[int, int] | None = None,
        wait_group: str | None = None,
        release_group: str | None = None,
    ) -> Stage:
        if placement is None:
            base_sm, num_sms = 0, rows // 128
        else:
            base_sm, num_sms = placement
        assignment = replace(
            self.policy.nvfp4_gemv(rows, k), num_sms=num_sms
        )
        table = tables[0]
        weight_fields = tuple(
            table.field(f"{tag}.m{tile}") for tile in range(rows // 128)
        )
        schedule = SchedRoutedNvfp4GemvUmmaStream(
            table.state,
            rank,
            weight_fields,
            table.field(f"{tag}.alpha"),
            activation,
            output,
            route_ready=not wait_for_previous,
            activation_mode=activation_mode,
            output_mode=output_mode,
            output_register=output_register,
            output_port=output_port,
            output_scale=output_scale,
            pipeline=pipeline,
            activation_tiles_per_load=activation_tiles_per_load,
        )
        schedule = self._routed_layered(schedule, family, tables)
        return self._stage(
            name,
            schedule,
            assignment,
            input_role="route" if wait_for_previous else None,
            wait_for_previous=wait_for_previous,
            base_sm=base_sm,
            wait_group=wait_group,
            release_group=release_group,
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
        routed_branch_sms = cfg.expert_intermediate_size // 128
        routed_sms = cfg.experts_per_token * routed_branch_sms
        shared_base = routed_sms
        shared_sms = self.sms - routed_sms
        if shared_sms <= 0:
            raise ValueError("FFN placement has no SMs left for the shared expert")
        w2_routed_sms = self.args.ffn_w2_routed_sms
        w2_shared_base = cfg.experts_per_token * w2_routed_sms
        w2_shared_sms = self.sms - w2_shared_base
        if not 0 < w2_routed_sms <= cfg.hidden_size // 128:
            raise ValueError("routed W2 placement must use 1..M/128 SMs")
        if not 0 < w2_shared_sms <= 2 * (cfg.hidden_size // 128):
            raise ValueError("shared split-K W2 placement must use 1..64 SMs")
        output_reduce = None
        ffn_branch = self.branch
        if self.ffn_fp32_tma:
            output_reduce = TmaTensor(
                self.launcher, self.ffn_accumulator
            ).rowmajor_2d("reduce", 1, 128)
            ffn_branch = self.ffn_accumulator.reshape(-1)
        stages, post = self._hc_stages(
            family,
            "ffn",
            self.next_residual,
            self.residual,
            branch=ffn_branch,
            zero_fp32_output=(
                ffn_branch if self.ffn_fp32_tma else None
            ),
        )
        ffn_input_ready = f"{family.name}.ffn.input.ready"
        stages[-1] = replace(stages[-1], release_group=ffn_input_ready)
        shared_ready = f"{family.name}.ffn.shared.ready"
        stages.append(
            self._fp8_quant_stage(
                "ffn.hidden.quant_fp8",
                self.norm_hidden,
                self.hidden_fp8,
                self.hidden_fp8_scale,
                wait_group=ffn_input_ready,
                release_group=shared_ready,
            )
        )
        stages.append(
            replace(
                self._bf16_linear_stage(
                    "ffn.router",
                    family,
                    "ffn.gate.weight",
                    self.norm_hidden,
                    self.router_logits,
                ),
                wait_group=ffn_input_ready,
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
        experts_ready = f"{family.name}.ffn.experts.ready"
        expert_join = f"{family.name}.ffn.experts.join"
        deferred_w2: list[Stage] = []
        stages.append(
            self._stage(
                "ffn.route",
                route,
                release_group=experts_ready,
            )
        )

        tables = tuple(self._routing_table(layer) for layer in family.layer_ids)
        for rank in range(cfg.experts_per_token):
            placement = (rank * routed_branch_sms, routed_branch_sms)
            base_sm, branch_sms = placement
            input_ready = f"{family.name}.ffn.expert{rank}.input.ready"
            middle_ready = f"{family.name}.ffn.expert{rank}.middle.ready"
            down_ready = f"{family.name}.ffn.expert{rank}.down.ready"
            stages.append(
                self._routed_native_quant_stage(
                    f"ffn.expert{rank}.input.quant_nvfp4",
                    family,
                    tables,
                    rank,
                    "up.input_scale",
                    self.norm_hidden,
                    self.routed_input[rank],
                    base_sm=base_sm,
                    wait_group=experts_ready,
                    release_group=input_ready,
                )
            )
            stages.append(
                self._routed_native_linear_stage(
                    f"ffn.expert{rank}.w1",
                    family,
                    tables,
                    rank,
                    "w1",
                    cfg.expert_intermediate_size,
                    cfg.hidden_size,
                    self.routed_input[rank],
                    None,
                    activation_mode="retain",
                    output_mode="retain",
                    output_register=1,
                    output_port=0,
                    placement=(base_sm, cfg.expert_intermediate_size // 128),
                    wait_group=input_ready,
                )
            )
            stages.append(
                self._routed_native_linear_stage(
                    f"ffn.expert{rank}.w3",
                    family,
                    tables,
                    rank,
                    "w3",
                    cfg.expert_intermediate_size,
                    cfg.hidden_size,
                    self.routed_input[rank],
                    None,
                    wait_for_previous=False,
                    activation_mode="reuse",
                    output_mode="retain",
                    output_register=1,
                    output_port=1,
                    placement=(base_sm, cfg.expert_intermediate_size // 128),
                )
            )
            stages.append(
                self._stage(
                    f"ffn.expert{rank}.swiglu",
                    SchedDsv4SwiGluShard128(
                        1,
                        0,
                        1,
                        1,
                        self.routed_middle[rank],
                        swiglu_limit=cfg.swiglu_limit,
                    ),
                    cfg.expert_intermediate_size // 128,
                    base_sm=base_sm,
                    wait_for_previous=False,
                    release_group=middle_ready,
                )
            )
            stages.append(
                self._routed_native_quant_stage(
                    f"ffn.expert{rank}.middle.quant_nvfp4",
                    family,
                    tables,
                    rank,
                    "down.input_scale",
                    self.routed_middle[rank],
                    self.routed_middle_packed[rank],
                    base_sm=base_sm,
                    wait_group=middle_ready,
                    release_group=down_ready,
                )
            )
            w2_stage = self._routed_native_linear_stage(
                    f"ffn.expert{rank}.w2",
                    family,
                    tables,
                    rank,
                    "w2",
                    cfg.hidden_size,
                    cfg.expert_intermediate_size,
                    self.routed_middle_packed[rank],
                    (
                        output_reduce
                        if self.ffn_fp32_tma
                        else self.routed_output[rank]
                    ),
                    output_mode=(
                        "reduce" if self.ffn_fp32_tma else "store"
                    ),
                    output_scale=(
                        self.route_weights[rank : rank + 1]
                        if self.ffn_fp32_tma
                        else None
                    ),
                    placement=(rank * w2_routed_sms, w2_routed_sms),
                    wait_group=down_ready,
                    release_group=expert_join,
            )
            if w2_routed_sms == routed_branch_sms:
                stages.append(w2_stage)
            else:
                deferred_w2.append(w2_stage)

        stages.append(
            self._fp8_linear_stage(
                "ffn.shared.w1",
                family,
                "ffn.shared_experts.w1",
                self.hidden_fp8,
                self.hidden_fp8_scale,
                self.shared_gate,
                placement=(shared_base, shared_sms),
                wait_group=shared_ready,
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
                wait_for_previous=False,
                placement=(shared_base, shared_sms),
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
                base_sm=shared_base,
            )
        )
        if self.ffn_fp32_tma:
            shared_down_ready = f"{family.name}.ffn.shared.down.ready"
            shared_quant_sms = cfg.expert_intermediate_size // (
                128 * self.args.fp8_umma_scale_pack
            )
            stages.append(
                self._native_fp8_quant_stage(
                    "ffn.shared.middle.quant_native_fp8",
                    self.shared_middle,
                    self.shared_middle_native_fp8,
                    placement=(shared_base, shared_quant_sms),
                    release_group=shared_down_ready,
                )
            )
            stages.extend(deferred_w2)
            stages.append(
                self._native_fp8_reduce_stage(
                    "ffn.shared.w2",
                    family,
                    "ffn.shared_experts.w2",
                    self.shared_middle_native_fp8,
                    output_reduce,
                    split_k=2,
                    placement=(w2_shared_base, w2_shared_sms),
                    wait_group=shared_down_ready,
                    release_group=expert_join,
                )
            )
            stages.append(replace(post, wait_group=expert_join))
        else:
            stages.append(
                self._fp8_quant_stage(
                    "ffn.shared.middle.quant_fp8",
                    self.shared_middle,
                    self.shared_middle_fp8,
                    self.shared_middle_scale,
                    placement=(
                        shared_base,
                        cfg.expert_intermediate_size // 128,
                    ),
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
                    placement=(shared_base, shared_sms),
                    release_group=expert_join,
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
                    wait_group=expert_join,
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
        self.fp8_head = self.args.vocab_size == cfg.vocab_size
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
        if self.fp8_head:
            print(
                "DSV4_HEAD_PREPROCESS status=START "
                f"rows={self.args.vocab_size} k={cfg.hidden_size} format=fp8_block128",
                flush=True,
            )
            preprocess_started = time.monotonic()
            self.head_weight_fp8, self.head_weight_scale = (
                quantize_fp8_block128(head_weight)
            )
            self.head_input_fp8 = torch.empty(
                (cfg.hidden_size,), dtype=torch.float8_e4m3fn, device=self.device
            )
            self.head_input_scale = torch.empty(
                (cfg.hidden_size // 128,),
                dtype=torch.float8_e8m0fnu,
                device=self.device,
            )
            head_assignment = self.policy.fp8_gemv(
                self.args.vocab_size, cfg.hidden_size
            )
            self.head_argmax_partial = torch.empty(
                (head_assignment.num_sms, 16),
                dtype=torch.uint8,
                device=self.device,
            )
            self.output_token = torch.empty(
                (1,), dtype=torch.int64, device=self.device
            )
            stages.extend(
                (
                    self._fp8_quant_stage(
                        "head.quant_fp8",
                        self.head_norm,
                        self.head_input_fp8,
                        self.head_input_scale,
                    ),
                    self._stage(
                        "head.logits.fp8",
                        SchedFp8Block128Gemv(
                            self.head_weight_fp8,
                            self.head_weight_scale,
                            self.head_input_fp8,
                            self.head_input_scale,
                            self.logits,
                        ),
                        head_assignment,
                    ),
                    self._stage(
                        "head.argmax.partial",
                        SchedArgmaxSmemPartial(
                            self.logits, self.head_argmax_partial
                        ),
                        head_assignment.num_sms,
                    ),
                    self._stage(
                        "head.argmax.reduce",
                        SchedArgmaxSmemReduce(
                            self.head_argmax_partial, self.output_token
                        ),
                        1,
                    ),
                )
            )
            print(
                "DSV4_HEAD_PREPROCESS status=PASS "
                f"weight_gib={self.head_weight_fp8.numel() * self.head_weight_fp8.element_size() / (1 << 30):.3f} "
                f"elapsed_s={time.monotonic() - preprocess_started:.3f}",
                flush=True,
            )
            return stages

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
        self.stage_profile_labels: list[str] = []
        self.step_profile_records: list[tuple[int, str, int, int, int]] = []
        self.step_profile_total = 0

        def profile_stage(name: str) -> bool:
            if not self.args.profile_stages:
                return False
            if self.args.profile_preattention_only:
                return name in {
                    "attn.hc_pre_rms4096",
                    "attn.q_b",
                }
            if name in {
                "attn.hc_pre",
                "attn.hc_pre_rms4096",
                "attn.hidden.quant_fp8",
                "attn.q_b",
                "attn.q_rope",
                "attn.kv_rope",
                "attn.compressor.wgate",
                "attn.compressor.pool",
                "attn.compressor.rope",
                "index.q_hadamard",
                "index.compressor.wgate",
                "index.compressor.pool",
                "index.compressor.hadamard",
                "index.score",
                "index.topk",
                "attn.sparse_swa",
                "attn.sparse_csa",
                "attn.sparse_hca",
                "attn.inverse_rope",
                "attn.o_a.g7",
                "attn.o_b",
                "attn.hc_post",
                "ffn.hc_pre",
                "ffn.route",
                "ffn.shared.w2",
                "ffn.expert_reduce",
                "ffn.hc_post",
            }:
                return True
            return False

        def queued(
            stage: Stage,
            prefix: str = "",
            *,
            profile_after: bool = False,
            profile_step_event: int | None = None,
        ) -> SequentialStage:
            nonlocal serial_sm
            base_sm = 0 if stage.base_sm is None else stage.base_sm
            if stage.base_sm is None and stage.num_sms == 1:
                base_sm = serial_sm
                serial_sm = (serial_sm + 1) % self.sms
            return SequentialStage(
                f"{prefix}{stage.name}",
                stage.schedule,
                stage.num_sms,
                base_sm=base_sm,
                input_role=stage.input_role,
                profile_after=profile_after,
                wait_for_previous=stage.wait_for_previous,
                wait_group=stage.wait_group,
                release_group=stage.release_group,
                profile_step_event=profile_step_event,
                prefetch_before_wait=stage.prefetch_before_wait,
            )

        def queued_family(family: LayerFamily) -> list[SequentialStage]:
            stages = self.family_stages[family.representative]
            self.step_profile_total = len(stages)
            queued_stages = []
            for index, stage in enumerate(stages):
                stage_profile_after = profile_stage(stage.name)
                if stage_profile_after:
                    label = (
                        "ffn.outputs_join"
                        if stage.name == "ffn.shared.w2"
                        else stage.name
                    )
                    self.stage_profile_labels.append(label)
                step_event = None
                if (
                    self.args.profile_steps
                    and self.args.profile_step_start
                    <= index
                    < self.args.profile_step_start + self.args.profile_step_count
                ):
                    step_event = (
                        runtime_config.layer_profile_event_base
                        + index
                        - self.args.profile_step_start
                    )
                queued_stage = queued(
                    stage,
                    f"{family.name}.",
                    profile_after=(
                        self.args.profile_layers and index + 1 == len(stages)
                    ) or stage_profile_after,
                    profile_step_event=step_event,
                )
                queued_stages.append(queued_stage)
                if step_event is not None:
                    self.step_profile_records.append(
                        (
                            index,
                            stage.name,
                            step_event,
                            queued_stage.base_sm,
                            queued_stage.num_sms,
                        )
                    )
            return queued_stages

        self.launcher.i(
            SchedDsv4PreloadRopeTables(self.resident_rope_tables).place(
                self.sms
            )
        )
        if self.args.layers == 1:
            family = self.families[0]
            stages = queued_family(family)
            stages.extend(queued(stage) for stage in self.head_stages)
            self.program = SequentialProgram(
                self.launcher,
                stages,
                profile_special_slot=7 if self.args.profile_stages else 0,
                balance_load_ports=True,
            )
            logical_stages = len(stages)
            queue_stages = logical_stages
        elif self.args.layers == 2:
            family = self.families[0]
            family_stages = queued_family(family)
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
            self.program = LoopedSequentialProgram(
                self.launcher, blocks, balance_load_ports=True
            )
            logical_stages = sum(
                len(block.stages) * block.repeat for block in blocks
            )
            queue_stages = sum(len(block.stages) for block in blocks)
        else:
            swa, layer2, hca, csa = self.families
            swa_stages = queued_family(swa)
            layer2_stages = queued_family(layer2)
            pair_stages = queued_family(hca) + queued_family(csa)
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
            self.program = LoopedSequentialProgram(
                self.launcher, blocks, balance_load_ports=True
            )
            logical_stages = sum(
                len(block.stages) * block.repeat for block in blocks
            )
            queue_stages = sum(len(block.stages) for block in blocks)
        self.launcher.s(self.program)
        print(
            "DSV4_ONE_LAUNCH_PROGRAM "
            f"model_launches=1 layers={self.args.layers} "
            f"instruction_storage="
            f"{'smem' if runtime_config.load_instructions else 'hbm'} "
            f"instruction_capacity={runtime_config.max_insts} "
            f"context={self.args.context_length} "
            f"position={self.decode_position} "
            f"attention={self.args.attention_mode} "
            f"fp8_projection_mode={self.args.fp8_projection_mode} "
            f"fp8_splitk_reduction={self.args.fp8_splitk_reduction} "
            f"ffn_reduction={self.args.ffn_reduction} "
            f"ffn_w2_routed_sms={self.args.ffn_w2_routed_sms} "
            f"fp8_splitk_components={','.join(sorted(self.splitk_components)) or 'none'} "
            f"index_selection={self.args.index_selection_mode} "
            f"gated_pool={self.args.gated_pool_mode} "
            f"prefix_cache={'current_token' if self.args.context_length == 1 else 'deterministic_seeded'} "
            f"logical_stages={logical_stages} queue_stages={queue_stages} "
            f"barriers={len(self.program.barriers)} "
            f"compute_insts={self.program.max_compute_instructions} "
            f"memory_insts={self.program.max_memory_instructions} "
            f"rope_preload_tables={len(self.resident_rope_tables)} "
            f"layer_profile_events={self.program.profile_event_count} "
            f"step_profile_events={len(self.step_profile_records)}",
            flush=True,
        )
        if self.args.profile_layers and self.program.profile_event_count != self.args.layers:
            raise AssertionError(
                "internal layer counter does not cover every requested layer"
            )
        if (
            self.args.profile_stages
            and self.program.profile_event_count != len(self.stage_profile_labels)
        ):
            raise AssertionError(
                "internal stage counter does not cover every requested boundary"
            )
        if self.args.profile_steps and not self.step_profile_records:
            raise AssertionError(
                "step profile window does not overlap this layer's queued steps"
            )
        if self.args.profile_layers:
            for family in self.families:
                stages = self.family_stages[family.representative]
                print(
                    "DSV4_LAYER_PROCESS "
                    f"family={family.name} "
                    f"layers={','.join(str(layer) for layer in family.layer_ids)} "
                    f"attention={self.config.attention_kind(family.representative)} "
                    f"routing={'hash' if family.representative < self.config.num_hash_layers else 'score'} "
                    f"stage_count={len(stages)} "
                    f"sequence={','.join(stage.name for stage in stages)}",
                    flush=True,
                )
        for kind in ("swa", "csa", "hca"):
            plan = self.attention_plans[kind]
            print(
                "DSV4_CONTEXT_PLAN "
                f"kind={kind} context={self.args.context_length} "
                f"position={self.decode_position} "
                f"window={min(self.config.sliding_window, self.args.context_length)} "
                f"compressed_rows={plan.compressed_rows} "
                f"compressed_selected={plan.compressed_selected} "
                f"attention_candidates={plan.attention_candidates} "
                f"compress_now={int(plan.should_compress)}",
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
        if self.args.profile_layers or self.args.profile_stages:
            self.launcher.profile.zero_()
        torch.cuda.synchronize(self.device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        self.launcher.launch(synchronize=False)
        end.record()
        end.synchronize()
        if self.fp8_head:
            return int(self.output_token.item()), start.elapsed_time(end), torch.empty(0)
        logits_cpu = self.logits.cpu()
        logits_fp32 = logits_cpu.float()
        if not bool(torch.isfinite(logits_fp32).all().item()):
            raise AssertionError("one-launch checkpoint logits are not finite")
        token = int(torch.argmax(logits_fp32).item())
        return token, start.elapsed_time(end), logits_fp32

    def validate_fp8_head(self, token: int) -> None:
        if not self.fp8_head:
            return
        reference_logits = torch.mv(
            self._tensor("head.weight")[: self.config.vocab_size],
            self.head_norm,
        )
        reference_token = int(torch.argmax(reference_logits).item())
        if token != reference_token:
            raise AssertionError(
                "FP8 head emitted "
                f"token {token}, reference BF16 GEMV emitted {reference_token}"
            )
        print(
            "DSV4_HEAD_REFERENCE status=PASS "
            f"output_token={token}",
            flush=True,
        )

    def report_projection_diagnostics(self) -> None:
        """Compare resident Q_b output with its raw-checkpoint FP8 oracle."""
        layer_id = self.families[0].representative
        linear = DeepSeekV4Checkpoint(
            self.args.checkpoint, self.config
        ).load_fp8_linear(
            f"layers.{layer_id}.attn.wq_b", device=str(self.device)
        )
        q_norm_weight = self._tensor(
            f"layers.{layer_id}.attn.q_norm.weight"
        )
        q_rank_fp32 = self.q_rank.float()
        rms_rcp = torch.rsqrt(
            q_rank_fp32.square().mean() + self.config.rms_epsilon
        )
        q_rank_norm = (
            q_rank_fp32 * rms_rcp * q_norm_weight.float()
        ).to(torch.bfloat16)
        activation, activation_scale = quantize_fp8_block128(q_rank_norm)
        reference = (
            dequantize_fp8_block128(linear.weight, linear.scale)
            @ dequantize_fp8_block128(activation, activation_scale)
        ).to(torch.bfloat16)
        actual = self.q.reshape(-1)
        delta = actual.float() - reference.float()
        cosine = torch.nn.functional.cosine_similarity(
            actual.float(), reference.float(), dim=0
        )
        print(
            "DSV4_PROJECTION_DIAGNOSTIC "
            f"stage=q_b layer={layer_id} "
            f"actual_norm={actual.float().norm().item():.6f} "
            f"reference_norm={reference.float().norm().item():.6f} "
            f"max_abs={delta.abs().max().item():.6f} "
            f"mean_abs={delta.abs().mean().item():.6f} "
            f"cosine={cosine.item():.8f} "
            f"actual_head={actual[:8].float().cpu().tolist()} "
            f"reference_head={reference[:8].float().cpu().tolist()}",
            flush=True,
        )

    def report_layer_profile(
        self,
        profile: torch.Tensor | None = None,
        *,
        sample_index: int | None = None,
        sample_cuda_ms: float | None = None,
    ) -> None:
        if not self.args.profile_layers:
            return
        if profile is None:
            profile = self.launcher.profile.cpu()
        magic = 0x4454524B50524631
        if any(int(value) != magic for value in profile[:, 127]):
            raise RuntimeError(
                "layer profiling requires a runtime built with track_profile=1"
            )
        start_frontier = max(int(value) for value in profile[:, 0])
        end_frontier = max(int(value) for value in profile[:, 1])
        physical_sm_ids = [
            int(value)
            for value in profile[:, runtime_config.track_profile_event_base + 25]
        ]
        profile_layer_ids = (
            (self.args.single_layer_id,)
            if self.args.layers == 1
            else tuple(range(self.args.layers))
        )
        boundaries = []
        spreads = []
        frontier_vcores = []
        for profile_index, layer_id in enumerate(profile_layer_ids):
            event_id = runtime_config.layer_profile_event_base + profile_index
            values = [int(value) for value in profile[:, event_id]]
            if any(value == 0 for value in values):
                raise RuntimeError(f"layer {layer_id} profile event was not recorded")
            boundaries.append(max(values))
            spreads.append(max(values) - min(values))
            frontier_vcores.append(max(range(len(values)), key=values.__getitem__))

        if self.args.layers == 1:
            reload_after_layers = ()
        elif self.args.layers == 2:
            reload_after_layers = (0, 1)
        else:
            reload_after_layers = (0, 1, 2, *range(4, self.args.layers, 2))
        reload_frontiers = []
        reload_spreads = []
        reload_frontier_vcores = []
        for reload_index, layer_id in enumerate(reload_after_layers):
            event_id = runtime_config.reload_profile_event_base + reload_index
            values = [int(value) for value in profile[:, event_id]]
            if any(value == 0 for value in values):
                raise RuntimeError(
                    f"reload after layer {layer_id} was not recorded"
                )
            reload_frontiers.append(max(values))
            reload_spreads.append(max(values) - min(values))
            reload_frontier_vcores.append(
                max(range(len(values)), key=values.__getitem__)
            )

        previous = start_frontier
        layer_total = 0
        reload_total = 0
        reload_index = 0
        for layer_id, boundary, spread, frontier_vcore in zip(
            profile_layer_ids, boundaries, spreads, frontier_vcores
        ):
            elapsed = boundary - previous
            if elapsed < 0:
                raise RuntimeError("layer profile frontiers are not monotonic")
            layer_total += elapsed
            previous = boundary
            family = next(
                family for family in self.families if layer_id in family.layer_ids
            )
            print(
                "DSV4_LAYER_TIME "
                f"layer={layer_id} family={family.name} "
                f"attention={self.config.attention_kind(layer_id)} "
                f"routing={'hash' if layer_id < self.config.num_hash_layers else 'score'} "
                f"stages={len(self.family_stages[family.representative])} "
                f"elapsed_ms={elapsed / 1.0e6:.6f} "
                f"frontier_spread_us={spread / 1.0e3:.3f} "
                f"frontier_vcore={frontier_vcore} "
                f"frontier_physical_sm={physical_sm_ids[frontier_vcore]}",
                flush=True,
            )
            if layer_id in reload_after_layers:
                reload_frontier = reload_frontiers[reload_index]
                reload_elapsed = reload_frontier - boundary
                if reload_elapsed < 0:
                    raise RuntimeError("reload profile frontier precedes its layer")
                reload_total += reload_elapsed
                print(
                    "DSV4_RELOAD_TIME "
                    f"after_layer={layer_id} "
                    f"barriers={'pair' if layer_id >= 4 else 'family'} "
                    f"elapsed_ms={reload_elapsed / 1.0e6:.6f} "
                    f"frontier_spread_us={reload_spreads[reload_index] / 1.0e3:.3f} "
                    f"frontier_vcore={reload_frontier_vcores[reload_index]} "
                    "frontier_physical_sm="
                    f"{physical_sm_ids[reload_frontier_vcores[reload_index]]}",
                    flush=True,
                )
                previous = reload_frontier
                reload_index += 1
            else:
                previous = boundary
        head_elapsed = end_frontier - previous
        if head_elapsed < 0:
            raise RuntimeError("head profile frontier precedes the final layer")
        internal_span = end_frontier - start_frontier
        grid_envelope = internal_span * profile.shape[0]
        counter_base = runtime_config.track_profile_event_base

        def grid_percent(offset: int) -> float:
            if grid_envelope <= 0:
                return 0.0
            return 100.0 * sum(
                int(value) for value in profile[:, counter_base + offset]
            ) / grid_envelope

        def counter_sum(offset: int) -> int:
            return sum(int(value) for value in profile[:, counter_base + offset])

        sm_clock_ghz = []
        for vcore in range(profile.shape[0]):
            elapsed_ns = int(profile[vcore, 1]) - int(profile[vcore, 0])
            elapsed_cycles = int(profile[vcore, counter_base + 27]) - int(
                profile[vcore, counter_base + 26]
            )
            if elapsed_ns > 0:
                sm_clock_ghz.append(elapsed_cycles / elapsed_ns)

        print(
            "DSV4_TRACK_PROFILE_SAMPLE "
            f"sample_index={sample_index if sample_index is not None else -1} "
            f"sample_cuda_ms={sample_cuda_ms if sample_cuda_ms is not None else -1.0:.6f} "
            f"internal_span_ms={internal_span / 1.0e6:.6f} "
            f"compute_m2c_wait_grid_pct={grid_percent(0):.3f} "
            f"allocator_slot_stall_grid_pct={grid_percent(3):.3f} "
            f"ldu0_queue_wait_grid_pct={grid_percent(9):.3f} "
            f"ldu0_dependency_wait_grid_pct={grid_percent(11):.3f} "
            f"ldu1_queue_wait_grid_pct={grid_percent(14):.3f} "
            f"ldu1_dependency_wait_grid_pct={grid_percent(16):.3f} "
            f"store_queue_wait_grid_pct={grid_percent(19):.3f} "
            f"store_service_grid_pct={grid_percent(21):.3f} "
            f"sm_clock_ghz_min={min(sm_clock_ghz):.3f} "
            f"sm_clock_ghz_median={statistics.median(sm_clock_ghz):.3f} "
            f"sm_clock_ghz_max={max(sm_clock_ghz):.3f} "
            f"allocator_instructions={counter_sum(8)} "
            f"ldu0_commands={counter_sum(13)} "
            f"ldu1_commands={counter_sum(18)} "
            f"store_commands={counter_sum(23)} "
            f"compute_m2c_contended={counter_sum(2)} "
            f"allocator_slot_stall_events={counter_sum(4)} "
            f"ldu0_dependency_contended={counter_sum(12)} "
            f"ldu1_dependency_contended={counter_sum(17)}",
            flush=True,
        )
        print(
            "DSV4_LAYER_PROFILE_SUMMARY "
            f"layers={self.args.layers} layer_total_ms={layer_total / 1.0e6:.6f} "
            f"reload_total_ms={reload_total / 1.0e6:.6f} "
            f"head_ms={head_elapsed / 1.0e6:.6f} "
            f"internal_span_ms={internal_span / 1.0e6:.6f} "
            f"sample_index={sample_index if sample_index is not None else -1} "
            f"sample_cuda_ms={sample_cuda_ms if sample_cuda_ms is not None else -1.0:.6f}",
            flush=True,
        )

    def report_stage_profile(
        self,
        profile: torch.Tensor | None = None,
        *,
        sample_index: int | None = None,
        sample_cuda_ms: float | None = None,
    ) -> None:
        if not self.args.profile_stages:
            return
        if profile is None:
            profile = self.launcher.profile.cpu()
        magic = 0x4454524B50524631
        if any(int(value) != magic for value in profile[:, 127]):
            raise RuntimeError(
                "stage profiling requires a runtime built with track_profile=1"
            )
        previous = max(int(value) for value in profile[:, 0])
        end_frontier = max(int(value) for value in profile[:, 1])
        grouped_total = 0
        for index, label in enumerate(self.stage_profile_labels):
            event_id = runtime_config.layer_profile_event_base + index
            values = [int(value) for value in profile[:, event_id]]
            if any(value == 0 for value in values):
                raise RuntimeError(
                    f"stage profile boundary {label!r} was not recorded"
                )
            boundary = max(values)
            elapsed = boundary - previous
            if elapsed < 0:
                raise RuntimeError("stage profile frontiers are not monotonic")
            grouped_total += elapsed
            print(
                "DSV4_STAGE_GROUP_TIME "
                f"index={index} through={label} "
                f"elapsed_ms={elapsed / 1.0e6:.6f} "
                f"frontier_spread_us={(max(values) - min(values)) / 1.0e3:.3f}",
                flush=True,
            )
            previous = boundary
        head_elapsed = end_frontier - previous
        if head_elapsed < 0:
            raise RuntimeError("head frontier precedes the final stage boundary")
        print(
            "DSV4_STAGE_PROFILE_SUMMARY "
            f"boundaries={len(self.stage_profile_labels)} "
            f"layer_ms={grouped_total / 1.0e6:.6f} "
            f"head_ms={head_elapsed / 1.0e6:.6f} "
            f"internal_span_ms={(end_frontier - max(int(value) for value in profile[:, 0])) / 1.0e6:.6f} "
            f"sample_index={sample_index if sample_index is not None else -1} "
            f"sample_cuda_ms={sample_cuda_ms if sample_cuda_ms is not None else -1.0:.6f}",
            flush=True,
        )

    def report_step_profile(
        self,
        profile: torch.Tensor | None = None,
        *,
        sample_index: int | None = None,
        sample_cuda_ms: float | None = None,
    ) -> None:
        if not self.args.profile_steps:
            return
        if profile is None:
            profile = self.launcher.profile.cpu()
        magic = 0x4454524B50524631
        if any(int(value) != magic for value in profile[:, 127]):
            raise RuntimeError(
                "step profiling requires a runtime built with track_profile=1"
            )

        summed_local_elapsed_ns = 0
        summed_wait_ns = 0
        for (
            step_index,
            name,
            event_id,
            base_sm,
            num_sms,
        ) in self.step_profile_records:
            samples = []
            for sm in range(base_sm, base_sm + num_sms):
                packed = int(profile[sm, event_id])
                if packed == 0:
                    continue
                samples.append(
                    (
                        sm,
                        packed & 0xFFFFFFFF,
                        (packed >> 32) & 0xFFFFFFFF,
                    )
                )
            if not samples:
                raise RuntimeError(
                    f"step profile event {event_id} for {name!r} was not recorded"
                )
            critical_sm, elapsed_ns, wait_ns = max(
                samples, key=lambda sample: sample[1]
            )
            active_samples = [
                (sm, max(0, sample_elapsed_ns - sample_wait_ns))
                for sm, sample_elapsed_ns, sample_wait_ns in samples
            ]
            max_active_sm, max_active_ns = max(
                active_samples,
                key=lambda sample: sample[1],
            )
            active_ns = max(0, elapsed_ns - wait_ns)
            summed_local_elapsed_ns += elapsed_ns
            summed_wait_ns += wait_ns
            elapsed_values = [sample[1] for sample in samples]
            active_values = [sample[1] for sample in active_samples]
            print(
                "DSV4_STEP_TIME "
                f"step={step_index} name={name} "
                f"base_sm={base_sm} assigned_sms={num_sms} "
                f"active_sms={len(samples)} critical_sm={critical_sm} "
                f"elapsed_us={elapsed_ns / 1.0e3:.3f} "
                f"median_elapsed_us={statistics.median(elapsed_values) / 1.0e3:.3f} "
                f"m2c_wait_us={wait_ns / 1.0e3:.3f} "
                f"compute_active_us={active_ns / 1.0e3:.3f} "
                f"max_active_sm={max_active_sm} "
                f"max_compute_active_us={max_active_ns / 1.0e3:.3f} "
                f"median_compute_active_us="
                f"{statistics.median(active_values) / 1.0e3:.3f} "
                f"m2c_wait_pct={100.0 * wait_ns / elapsed_ns if elapsed_ns else 0.0:.3f}",
                flush=True,
            )
        print(
            "DSV4_STEP_PROFILE_SUMMARY "
            f"window_start={self.args.profile_step_start} "
            f"window_steps={len(self.step_profile_records)} "
            f"layer_steps={self.step_profile_total} "
            f"summed_local_elapsed_us={summed_local_elapsed_ns / 1.0e3:.3f} "
            f"summed_local_m2c_wait_us={summed_wait_ns / 1.0e3:.3f} "
            f"sample_index={sample_index if sample_index is not None else -1} "
            f"sample_cuda_ms={sample_cuda_ms if sample_cuda_ms is not None else -1.0:.6f}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--layers", type=int, choices=(1, 2, 43), default=1)
    parser.add_argument(
        "--single-layer-id",
        type=int,
        default=0,
        help="checkpoint layer to use when --layers=1",
    )
    parser.add_argument("--token-id", type=int, default=791)
    parser.add_argument(
        "--context-length",
        type=int,
        default=1,
        help=(
            "timed decode context in [1,128]; contexts above one use a "
            "deterministic resident prefix while the current KV/compressed "
            "rows are produced inside the launch"
        ),
    )
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--sms", type=int, default=152)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--expected-token-id", type=int)
    parser.add_argument("--resident-reserve-gib", type=float, default=8.0)
    parser.add_argument(
        "--fp8-qb-mode",
        choices=("native", "scalar"),
        default="native",
        help="select the q_b kernel for matched end-to-end A/B profiling",
    )
    parser.add_argument(
        "--fp8-projection-mode",
        choices=("legacy", "splitk"),
        default="legacy",
        help=(
            "select legacy attention FP8 projections or native split-K/TMA "
            "reduction for q_a, q_b, kv, index q_b, o_a, and o_b"
        ),
    )
    parser.add_argument(
        "--fp8-splitk-components",
        default="all",
        help=(
            "comma-separated diagnostic subset of q_a,q_b,kv,index_q_b,o_a,o_b; "
            "used only with --fp8-projection-mode=splitk"
        ),
    )
    parser.add_argument(
        "--fp8-splitk-reduction",
        choices=("bf16", "fp32"),
        default="bf16",
        help=(
            "reduce split-K projections directly to BF16 model outputs or "
            "use an exact FP32 accumulator plus BF16 finalizer"
        ),
    )
    parser.add_argument(
        "--ffn-reduction",
        choices=("legacy", "fp32-tma"),
        default="fp32-tma",
        help=(
            "select BF16 expert outputs plus a compute reducer, or direct "
            "FP32 tensor-core outputs into one TMA reduce accumulator"
        ),
    )
    parser.add_argument(
        "--ffn-w2-routed-sms",
        type=int,
        default=16,
        help=(
            "SMs per routed W2 after the fixed 16-SM W1/W3 front half; "
            "values above 16 defer W2 and repartition the full device"
        ),
    )
    parser.add_argument(
        "--fp8-umma-scale-pack",
        type=int,
        choices=(2,),
        default=2,
        help=(
            "use the resident image's compile-time pack-2 native UMMA layout"
        ),
    )
    parser.add_argument(
        "--fp8-umma-output-group-size",
        type=int,
        choices=(2,),
        default=2,
        help=(
            "use the resident image's compile-time two-M128 UMMA epilogue "
            "grouping (split-K tails retain their static one-tile handler)"
        ),
    )
    parser.add_argument(
        "--attention-mode",
        choices=("auto", "umma-split", "contiguous", "scalar"),
        default="auto",
        help="select the sparse-attention compute mechanism for matched A/B profiling",
    )
    parser.add_argument(
        "--index-selection-mode",
        choices=("auto", "force"),
        default="auto",
        help="force CSA score/top-k work for exhaustive-selection A/B profiling",
    )
    parser.add_argument(
        "--gated-pool-mode",
        choices=("auto", "packed", "scalar"),
        default="auto",
        help="select the compressor pooling layout for matched A/B profiling",
    )
    parser.add_argument(
        "--profile-layers",
        action="store_true",
        help="record compact per-layer LDU globaltimer frontiers",
    )
    parser.add_argument(
        "--diagnose-projections",
        action="store_true",
        help="compare resident projection output with a raw-checkpoint oracle",
    )
    parser.add_argument(
        "--profile-stages",
        action="store_true",
        help="record selected one-layer stage-group completion frontiers",
    )
    parser.add_argument(
        "--profile-preattention-only",
        action="store_true",
        help=(
            "with --profile-stages, retain only the fused mHC/RMS and "
            "attention-ready boundaries so auxiliary index probes do not "
            "serialize the measured DAG"
        ),
    )
    parser.add_argument(
        "--profile-steps",
        action="store_true",
        help=(
            "record per-queued-step compute-side duration and M2C wait without "
            "adding a dependency barrier"
        ),
    )
    parser.add_argument(
        "--profile-step-start",
        type=int,
        default=0,
        help="first queued layer step included by --profile-steps",
    )
    parser.add_argument(
        "--profile-step-count",
        type=int,
        default=(
            runtime_config.reload_profile_event_base
            - runtime_config.layer_profile_event_base
        ),
        help="number of queued layer steps included by --profile-steps",
    )
    parser.add_argument(
        "--profile-all-samples",
        action="store_true",
        help="report layer frontiers and aggregate counters for every sample",
    )
    args = parser.parse_args()
    cfg = DeepSeekV4FlashConfig()
    if not 0 <= args.token_id < cfg.vocab_size:
        parser.error("token-id is outside the vocabulary")
    if not 0 <= args.single_layer_id < cfg.num_layers:
        parser.error("single-layer-id is outside the transformer")
    if args.layers != 1 and args.single_layer_id != 0:
        parser.error("single-layer-id is only valid with --layers=1")
    if not 1 <= args.context_length <= cfg.sliding_window:
        parser.error("context-length must be in [1,128]")
    if not 1 <= args.vocab_size <= cfg.vocab_size:
        parser.error("vocab-size must be in [1,129280]")
    if args.sms <= 0 or args.iterations <= 0 or args.warmup < 0:
        parser.error("sms/iterations must be positive and warmup non-negative")
    if args.resident_reserve_gib < 0:
        parser.error("resident-reserve-gib must be non-negative")
    if args.ffn_reduction == "legacy" and args.ffn_w2_routed_sms != 16:
        parser.error("repartitioned W2 requires --ffn-reduction fp32-tma")
    if not 16 <= args.ffn_w2_routed_sms <= 25:
        parser.error("ffn-w2-routed-sms must be in [16,25]")
    profile_modes = sum(
        (args.profile_layers, args.profile_stages, args.profile_steps)
    )
    if profile_modes > 1:
        parser.error(
            "--profile-layers, --profile-stages, and --profile-steps are mutually exclusive"
        )
    if args.profile_preattention_only and not args.profile_stages:
        parser.error("--profile-preattention-only requires --profile-stages")
    if (args.profile_stages or args.profile_steps) and args.layers != 1:
        parser.error("stage/step profiling requires --layers 1")
    step_capacity = (
        runtime_config.reload_profile_event_base
        - runtime_config.layer_profile_event_base
    )
    if (
        args.profile_step_start < 0
        or not 1 <= args.profile_step_count <= step_capacity
    ):
        parser.error(
            f"step profile window requires start >= 0 and count in [1,{step_capacity}]"
        )
    if not args.profile_steps and args.profile_step_start != 0:
        parser.error("--profile-step-start requires --profile-steps")
    if args.profile_all_samples and not profile_modes:
        parser.error("--profile-all-samples requires a profiling mode")

    device = torch.device("cuda")
    build_started = time.monotonic()
    flow = ResidentOneLaunchDecode(args, device)
    torch.cuda.synchronize(device)
    build_seconds = time.monotonic() - build_started
    prime_token, prime_ms, prime_logits = flow.run_once()
    flow.validate_fp8_head(prime_token)
    if args.diagnose_projections:
        flow.report_projection_diagnostics()
    if args.expected_token_id is not None and prime_token != args.expected_token_id:
        raise AssertionError(
            f"prime launch emitted token {prime_token}, "
            f"expected {args.expected_token_id}"
        )
    logit_summary = (
        f"logit_min={float(prime_logits.min().item()):.6f} "
        f"logit_max={float(prime_logits.max().item()):.6f}"
        if prime_logits.numel()
        else "logits=fp8_argmax"
    )
    print(
        "DSV4_ONE_LAUNCH_PRIME status=PASS "
        f"output_token={prime_token} elapsed_ms={prime_ms:.6f}",
        flush=True,
    )
    for _ in range(args.warmup):
        token, _, _ = flow.run_once()
        if args.expected_token_id is not None and token != args.expected_token_id:
            raise AssertionError(
                f"warmup emitted token {token}, expected {args.expected_token_id}"
            )

    timings = []
    profile_samples = []
    reference_token = None
    logits = None
    for iteration in range(args.iterations):
        token, elapsed_ms, logits = flow.run_once()
        timings.append(elapsed_ms)
        if profile_modes:
            profile_samples.append(flow.launcher.profile.cpu().clone())
        print(
            "DSV4_ONE_LAUNCH_SAMPLE "
            f"iteration={iteration} elapsed_ms={elapsed_ms:.6f}",
            flush=True,
        )
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
    if profile_modes:
        median_timing = statistics.median(timings)
        profile_index = min(
            range(len(timings)),
            key=lambda index: abs(timings[index] - median_timing),
        )
        profile_indices = (
            range(len(profile_samples))
            if args.profile_all_samples
            else (profile_index,)
        )
        for sample_index in profile_indices:
            if args.profile_layers:
                reporter = flow.report_layer_profile
            elif args.profile_stages:
                reporter = flow.report_stage_profile
            else:
                reporter = flow.report_step_profile
            reporter(
                profile_samples[sample_index],
                sample_index=sample_index,
                sample_cuda_ms=timings[sample_index],
            )
    print(
        "DSV4_ONE_LAUNCH_DECODE status=PASS model_launches=1 gpu=1 "
        f"layers={args.layers} token_id={args.token_id} "
        f"context={args.context_length} position={args.context_length - 1} "
        f"attention={args.attention_mode} "
        f"ffn_reduction={args.ffn_reduction} "
        f"ffn_w2_routed_sms={args.ffn_w2_routed_sms} "
        f"index_selection={args.index_selection_mode} "
        f"gated_pool={args.gated_pool_mode} "
        f"prefix_cache={'current_token' if args.context_length == 1 else 'deterministic_seeded'} "
        f"vocab={args.vocab_size} output_token={reference_token} "
        f"build_s={build_seconds:.3f} min_ms={min(timings):.6f} "
        f"median_ms={statistics.median(timings):.6f} "
        f"max_ms={max(timings):.6f} "
        f"{logit_summary}",
        flush=True,
    )


if __name__ == "__main__":
    main()
