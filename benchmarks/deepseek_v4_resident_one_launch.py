#!/usr/bin/env python3
"""Checkpoint-resident DeepSeek-V4 decode in one VDCores launch.

The full model is represented by four shape families: layers 0-1, layer 2,
odd HCA layers, and even CSA layers.  Repeated families use runtime loop
counters to select resident layer weights; routed expert IDs stay in one fixed
HBM buffer and LDU resolves the selected expert and current layer.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import struct
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from dae import runtime
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
from dae.deepseek_v4_mxfp_checkpoint import (
    DeepSeekV4MxfpFfnLayer,
    default_mxfp_ffn_directory,
    load_mxfp_ffn_layer,
    mxfp_ffn_layer_path,
)
from dae.deepseek_v4_schedule import DeepSeekV4ShapePolicy, ShapeAssignment
from dae.deepseek_v4_quant import (
    dequantize_fp8_block128,
    quantize_fp8_block128,
)
from dae.launcher import Launcher
from dae.instructions import (
    Gemv_M128N8Direct4,
    TmaLoad1D,
    TmaStore1D,
    TmaTensor,
)
from dae.runtime import config as runtime_config
from dae.schedule import (
    LayeredSchedule,
    SchedDsv4AttentionContext1Fp8Sm100,
    SchedArgmaxSmemPartial,
    SchedArgmaxSmemReduce,
    SchedDsv4AttentionSplit64UmmaSm100,
    SchedDsv4AttentionSplitReduceFp8Sm100,
    SchedDsv4Bf16Gemv,
    SchedDsv4Bf16GemvGroup4SplitK,
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
    SchedDsv4HcHeadRms,
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
    SchedDsv4TopK512,
    SchedDsv4ZeroFill,
    SchedCopy,
    SchedDsv4Mxfp8QuantFfnInput,
    SchedDsv4SplitMxfp8FfnInputRecords,
    SchedFp8Block128Gemv,
    SchedFp8Block128GateUpSwiGlu,
    SchedFp8GemvUmmaCoupled,
    SchedGemvMGroup,
    SchedLayeredDsv4RouterBf16Gemv,
    SchedLayeredMxfp4Mxfp8RoutedResidentFfn,
    SchedMxfp4Mxfp8DownFixedRing,
    SchedMxfp4Mxfp8GateUpSiluFixedRing,
    SchedMxfp4Mxfp8RoutedResidentFfn,
    SchedOverlapAsyncBarrierReload,
    SchedRMS,
)
from dae.sequential import (
    LoopedSequentialProgram,
    SequentialBlock,
    SequentialProgram,
    SequentialStage,
)
from dae.tma_utils import Major, pack_weight_tile_major


# Task-local detail traces occupy events 2--29 when DAE_TRACK_PROFILE is
# enabled.  Keep step-duration windows in the disjoint low-profile tail so a
# task cannot overwrite its enclosing begin/end pair.
STEP_PROFILE_EVENT_BASE = 32
STEP_PROFILE_FRONTIER_BASE = runtime_config.layer_profile_event_base
LOOPBACK_PROFILE_CSA_COMPLETIONS = 20
LOOPBACK_PROFILE_FRONTIER_BASE = (
    runtime_config.layer_profile_event_base
    + LOOPBACK_PROFILE_CSA_COMPLETIONS
)
FP8_COUPLED_STEP_BEGIN_EVENT = runtime_config.reload_profile_event_base - 1
FP8_COUPLED_LAYER_BEGIN_EVENT = runtime_config.detail_profile_event_base + 25
FFN_OUTPUT_PROFILE_EVENT_BASE = 55
STU_RAW_POP_BEGIN_EVENT = 64
STU_RAW_SERVICE_IDENTITY_EVENT = 65
STU_RAW_OUTPUT_TOKEN_EVENT = 66
STU_RAW_PTR_MATCH_EVENT_BASE = 67
STU_RAW_POST_EVENT_BASE = 71
STU_RAW_PTR_EVENT_BASE = 75
STU_RAW_ARRIVAL_EVENT_BASE = 79
STU_HISTORY_EVENT_BASE = 83
STU_HISTORY_COMMANDS = 4


@dataclass(frozen=True)
class LayerFamily:
    name: str
    layer_ids: tuple[int, ...]
    counter_strides: tuple[tuple[int, int], ...] = ()

    @property
    def representative(self) -> int:
        return self.layer_ids[0]


@dataclass(frozen=True)
class MxfpFfnRuntimeLayer:
    image: DeepSeekV4MxfpFfnLayer
    linear1_tma: TmaTensor
    down_tma: TmaTensor
    linear1_metadata: torch.Tensor
    down_metadata: torch.Tensor


@dataclass(frozen=True)
class Stage:
    name: str
    schedule: object
    num_sms: int
    input_role: str | None = None
    wait_for_previous: bool = True
    parallel_with_previous: bool = False
    base_sm: int | None = None
    wait_group: str | None = None
    release_group: str | None = None
    prefetch_before_wait: bool = False
    prefetch_before_resident_reset: bool = False
    wait_group_roles: tuple[tuple[str, str], ...] = ()
    release_group_roles: tuple[tuple[str, str], ...] = ()


class ResidentOneLaunchDecode:
    def __init__(self, args: argparse.Namespace, device: torch.device):
        self.args = args
        self.device = device
        self.config = DeepSeekV4FlashConfig()
        if args.layers == 1:
            self.layer_ids = (args.single_layer_id,)
        elif args.layers == 2:
            second_layer_id = (
                args.two_layer_start_id
                if args.repeat_same_layer
                else args.two_layer_start_id + 1
            )
            self.layer_ids = (args.two_layer_start_id, second_layer_id)
        else:
            self.layer_ids = tuple(range(args.layers))
        self.profile_layer_ids = (
            self.layer_ids * args.two_layer_pair_repeats
            if args.layers == 2
            else self.layer_ids
        )
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
        self.splitk_accumulators: list[torch.Tensor] = []
        self._active_splitk_workspace: torch.Tensor | None = None
        self._active_splitk_offset = 0
        self.policy = DeepSeekV4ShapePolicy(self.sms)
        self.assignments: dict[tuple, ShapeAssignment] = {}
        self.launcher = Launcher(self.sms, device=self.device)
        self.checkpoint = self._load_checkpoint()
        self.families = self._families()
        self._hash_rows: dict[int, torch.Tensor] = {}
        self._fused_bf16_weight_cache: dict[tuple, tuple[torch.Tensor, ...]] = {}
        self._fused_hc_projection_cache: dict[
            tuple, tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]
        ] = {}
        self._allocate_state()
        self._allocate_mxfp_ffn_runtime()
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
        cross_layer_hc_pair = (
            args.layers == 43
            or (
                args.layers == 2
                and args.two_layer_start_id == 3
                and not args.repeat_same_layer
            )
        )
        if cross_layer_hc_pair and not args.disable_cross_layer_hc_fusion:
            previous_family, next_family = (
                self.families
                if args.layers == 2
                else self.families[2:4]
            )
            self._apply_cross_layer_hc_fusion(
                previous_family, next_family
            )
        self.head_stages = self._build_head()
        if args.loopback_hc_fusion:
            self._apply_loopback_hc_fusion(
                self.families[1],
                self.families[2],
                self.families[3],
            )
        self._build_program()
        print(
            "DSV4_COMPUTE_OPS "
            f"count={len(self.launcher.compute_operator_names())} "
            f"ops={','.join(self.launcher.compute_operator_names())}",
            flush=True,
        )
        prepare_started = time.monotonic()
        self.launcher.prepare_launch()
        self._l2_scrub = (
            torch.zeros(
                args.cold_l2_scrub_mib * 1024 * 1024,
                dtype=torch.uint8,
                device=self.device,
            )
            if args.cold_l2_scrub_mib
            else None
        )
        torch.cuda.synchronize(self.device)
        print(
            "DSV4_ONE_LAUNCH_PREPARE status=PASS "
            f"elapsed_s={time.monotonic() - prepare_started:.3f}",
            flush=True,
        )

    def _load_checkpoint(self) -> DeepSeekV4ResidentCheckpoint:
        disk = DeepSeekV4Checkpoint(self.args.checkpoint, self.config)
        # A repeated-layer diagnostic references the same checkpoint tensors
        # twice but should materialize only one resident weight image.
        resident_layer_ids = tuple(dict.fromkeys(self.layer_ids))
        prefix = tuple(
            f"layers.{layer_id}." for layer_id in resident_layer_ids
        )
        names = tuple(
            name
            for name in expected_inference_tensor_specs(self.config)
            if (
                (not name.startswith("layers.") or name.startswith(prefix))
                and ".ffn.experts." not in name
                and ".ffn.shared_experts." not in name
            )
        )
        self.mxfp_ffn_root = (
            Path(self.args.mxfp_ffn_root)
            if self.args.mxfp_ffn_root is not None
            else default_mxfp_ffn_directory(self.args.checkpoint)
        )
        missing = [
            mxfp_ffn_layer_path(self.mxfp_ffn_root, layer_id)
            for layer_id in resident_layer_ids
            if not mxfp_ffn_layer_path(
                self.mxfp_ffn_root, layer_id
            ).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "offline MXFP FFN images are required; missing "
                + ", ".join(str(path) for path in missing[:4])
            )
        mxfp_reserve_bytes = sum(
            mxfp_ffn_layer_path(
                self.mxfp_ffn_root, layer_id
            ).stat().st_size
            for layer_id in resident_layer_ids
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
            reserve_bytes=(
                int(self.args.resident_reserve_gib * (1 << 30))
                + mxfp_reserve_bytes
            ),
            native_nvfp4=False,
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
        self.mxfp_ffn_images = {}
        stacked_mxfp_storage = None
        first_cpu_image = None
        if len(resident_layer_ids) > 1:
            # Install the offline image in four layer-major CUDA arenas.  The
            # token path still sees ordinary per-layer contiguous views, but
            # adjacent layers now share virtual/page-table locality instead
            # of occupying 4 * num_layers independent allocator segments.
            first_cpu_image = load_mxfp_ffn_layer(
                self.mxfp_ffn_root,
                resident_layer_ids[0],
                device="cpu",
            )
            stacked_mxfp_storage = DeepSeekV4MxfpFfnLayer(
                **{
                    name: torch.empty(
                        (len(resident_layer_ids), *getattr(first_cpu_image, name).shape),
                        dtype=torch.uint8,
                        device=self.device,
                    )
                    for name in (
                        "linear1_weights",
                        "linear1_scales",
                        "down_weights",
                        "down_scales",
                    )
                }
            )
            self._mxfp_ffn_stacked_storage = stacked_mxfp_storage
        for index, layer_id in enumerate(resident_layer_ids, 1):
            image_started = time.monotonic()
            if stacked_mxfp_storage is None:
                image = load_mxfp_ffn_layer(
                    self.mxfp_ffn_root,
                    layer_id,
                    device=self.device,
                )
            else:
                cpu_image = (
                    first_cpu_image
                    if index == 1
                    else load_mxfp_ffn_layer(
                        self.mxfp_ffn_root,
                        layer_id,
                        device="cpu",
                    )
                )
                image = DeepSeekV4MxfpFfnLayer(
                    **{
                        name: getattr(stacked_mxfp_storage, name)[index - 1]
                        for name in (
                            "linear1_weights",
                            "linear1_scales",
                            "down_weights",
                            "down_scales",
                        )
                    }
                )
                for name in (
                    "linear1_weights",
                    "linear1_scales",
                    "down_weights",
                    "down_scales",
                ):
                    getattr(image, name).copy_(getattr(cpu_image, name))
                if index == 1:
                    first_cpu_image = None
            self.mxfp_ffn_images[layer_id] = image
            print(
                "DSV4_MXFP_RESIDENT_LAYER status=PASS "
                f"layer={layer_id} index={index}/{len(resident_layer_ids)} "
                f"gib={image.nbytes / (1 << 30):.3f} "
                f"elapsed_s={time.monotonic() - image_started:.3f}",
                flush=True,
            )
        free_bytes, _ = torch.cuda.mem_get_info(self.device)
        print(
            "DSV4_MXFP_RESIDENT status=PASS "
            f"layers={len(self.mxfp_ffn_images)} "
            f"tensor_gib={sum(image.nbytes for image in self.mxfp_ffn_images.values()) / (1 << 30):.3f} "
            f"free_gib={free_bytes / (1 << 30):.3f} conversion=offline",
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
            first_layer_id, second_layer_id = self.layer_ids
            if self.args.repeat_same_layer:
                kind = self.config.attention_kind(first_layer_id)
                routing = (
                    "hash"
                    if first_layer_id < self.config.num_hash_layers
                    else "score"
                )
                return (
                    LayerFamily(
                        f"layer{first_layer_id}-twice.{kind}_{routing}",
                        self.profile_layer_ids,
                        ((0, 1),),
                    ),
                )
            first_kind = self.config.attention_kind(first_layer_id)
            second_kind = self.config.attention_kind(second_layer_id)
            first_routing = (
                "hash" if first_layer_id < self.config.num_hash_layers else "score"
            )
            second_routing = (
                "hash" if second_layer_id < self.config.num_hash_layers else "score"
            )
            if (first_kind, first_routing) == (second_kind, second_routing):
                return (
                    LayerFamily(
                        f"layers{first_layer_id}-{second_layer_id}."
                        f"{first_kind}_{first_routing}",
                        self.profile_layer_ids,
                        ((0, 1),),
                    ),
                )
            counter_strides = (
                ((0, 1), (1, 2))
                if self.args.two_layer_pair_repeats > 1
                else ()
            )
            return (
                LayerFamily(
                    f"layer{first_layer_id}.{first_kind}_{first_routing}",
                    (first_layer_id,) * self.args.two_layer_pair_repeats,
                    counter_strides,
                ),
                LayerFamily(
                    f"layer{second_layer_id}.{second_kind}_{second_routing}",
                    (second_layer_id,) * self.args.two_layer_pair_repeats,
                    counter_strides,
                ),
            )
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
        parallel_with_previous: bool = False,
        base_sm: int | None = None,
        wait_group: str | None = None,
        release_group: str | None = None,
        prefetch_before_wait: bool = False,
        prefetch_before_resident_reset: bool = False,
        wait_group_roles: tuple[tuple[str, str], ...] = (),
        release_group_roles: tuple[tuple[str, str], ...] = (),
    ) -> Stage:
        if isinstance(sms, ShapeAssignment):
            sms = self._remember(sms)
        return Stage(
            name=name,
            schedule=schedule,
            num_sms=int(sms),
            input_role=input_role,
            wait_for_previous=wait_for_previous,
            parallel_with_previous=parallel_with_previous,
            base_sm=base_sm,
            wait_group=wait_group,
            release_group=release_group,
            prefetch_before_wait=prefetch_before_wait,
            prefetch_before_resident_reset=(
                prefetch_before_resident_reset
            ),
            wait_group_roles=wait_group_roles,
            release_group_roles=release_group_roles,
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

    def _allocate_state(self) -> None:
        cfg, d = self.config, self.device
        embedding = self._tensor("embed.weight")[self.args.token_id]
        self.initial_residual = embedding.reshape(1, -1).repeat(cfg.hc_mult, 1)
        self.attention_post_input_record = torch.empty(
            (1 + cfg.hc_mult, cfg.hidden_size),
            dtype=torch.bfloat16,
            device=d,
        )
        self.branch = self.attention_post_input_record[0]
        self.residual = self.attention_post_input_record[1:]
        self.next_residual = torch.empty_like(self.residual)
        self.hidden = torch.empty((cfg.hidden_size,), dtype=torch.bfloat16, device=d)
        self.mhc_packed_output = torch.empty(
            (cfg.hidden_size + 40,), dtype=torch.bfloat16, device=d
        )
        self.norm_hidden = self.mhc_packed_output[:cfg.hidden_size]
        mhc_output_metadata = self.mhc_packed_output[cfg.hidden_size:].view(
            torch.float32
        )
        self.mhc_output_metadata = mhc_output_metadata
        direct_projection_views = {}
        if self.direct_splitk_bf16:
            projection_rows = (
                cfg.q_lora_rank
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

            direct_view("q_rank", (cfg.q_lora_rank,))
            direct_view("kv", (cfg.head_dim,))
            direct_view("q", (cfg.num_heads, cfg.head_dim))
            direct_view("index_q", (cfg.index_heads, cfg.index_head_dim))
            direct_view("o_rank", (cfg.o_groups, cfg.o_lora_rank))
            if offset != projection_rows:
                raise AssertionError("split-K output arena was not carved exactly")
        else:
            self.splitk_output_arena = None
        self.mhc_packed_metadata = torch.empty(
            (56,), dtype=torch.float32, device=d
        )
        self.mhc_residual_square_sum = self.mhc_packed_metadata[:1]
        self.mixes = self.mhc_packed_metadata[1:25]
        self.mhc_metadata_tail = self.mhc_packed_metadata[28:56]
        fused_records = (
            SchedDsv4Fp32Bf16Gemv.FUSED_SPLITS
            * SchedDsv4Fp32Bf16Gemv.FUSED_RECORD_STRIDE
        )
        self.mhc_fused_metadata = torch.empty(
            (fused_records + SchedDsv4Fp32Bf16Gemv.FUSED_TAIL_ITEMS,),
            dtype=torch.float32,
            device=d,
        )
        self.mhc_fused_residual_square_sum = self.mhc_fused_metadata[:1]
        self.mhc_fused_metadata_tail = self.mhc_fused_metadata[fused_records:]
        self.post = mhc_output_metadata[:4]
        self.comb = mhc_output_metadata[4:].view(4, 4)

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
                (indices.numel() + 63) // 64
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

        self.router_prepared = torch.empty(
            (cfg.num_experts, 2), dtype=torch.float32, device=d
        )
        # Ping-pong route storage with the score-layer barrier banks. The LDU
        # route cache is keyed by address, so alternating records invalidates
        # it naturally at each layer boundary without a queue-local reset op.
        self.route_records = torch.empty(
            (2, 128), dtype=torch.uint8, device=d
        )
        self.route_record = self.route_records[0]
        self.route_indices = self.route_record[:32].view(torch.int32)
        self.route_weights = self.route_record[32:64].view(torch.float32)
        self.zero_bias = torch.zeros(
            (cfg.num_experts,), dtype=torch.float32, device=d
        )
        self.zero_fill_gate = torch.zeros(
            (1,), dtype=torch.uint32, device=d
        )
        self.zero_hash = torch.zeros((8,), dtype=torch.int32, device=d)

        self.mxfp_input_records = torch.empty(
            (8, 6144), dtype=torch.uint8, device=d
        )
        self.mxfp_activation_data = torch.empty(
            (8, 4096), dtype=torch.uint8, device=d
        )
        self.mxfp_activation_scales = torch.empty(
            (8, 2048), dtype=torch.uint8, device=d
        )
        self.mxfp_middle_records = torch.empty(
            (7, 16, 1536), dtype=torch.uint8, device=d
        )
        middle_flat = self.mxfp_middle_records.view(112, 1536)
        self.mxfp_middle_data = middle_flat[:, :1024]
        self.mxfp_middle_scales = middle_flat[:, 1024:]
        self.mxfp_ffn_output = torch.empty(
            (8, cfg.hidden_size), dtype=torch.bfloat16, device=d
        )

    def _allocate_mxfp_ffn_runtime(self) -> None:
        if self.sms < 148:
            raise ValueError(
                "the production MXFP FFN overlap requires at least 148 SMs"
            )
        # A heterogeneous two-layer diagnostic has the same adjacent HCA/CSA
        # ownership as the production repeated pair.  Keep their internal
        # MXFP rings disjoint even when the pair-tail clear is synchronous;
        # otherwise CSA reuses HCA's live ring before that tail can restore it.
        heterogeneous_pair = (
            self.args.layers == 2 and self.layer_ids[0] != self.layer_ids[1]
        )
        barrier_sets = (
            2
            if (
                self.args.layers == self.config.num_layers
                or runtime_config.async_barrier_reload_enabled
                or heterogeneous_pair
            )
            else 1
        )
        self.mxfp_internal_barrier_start = self.launcher.num_bars
        self.mxfp_ready_bars = []
        self.mxfp_zero_ready_bars = []
        for _ in range(barrier_sets):
            self.mxfp_ready_bars.append(
                tuple(self.launcher.new_bar(1) for _ in range(112))
            )
            self.mxfp_zero_ready_bars.append(
                tuple(self.launcher.new_bar(1) for _ in range(32))
            )
        self.mxfp_internal_barrier_stop = self.launcher.num_bars
        self.mxfp_output_tma = TmaTensor(
            self.launcher, self.mxfp_ffn_output
        ).m128n8_output("reduce")
        self._mxfp_runtime_layers: dict[
            tuple[int, int], MxfpFfnRuntimeLayer
        ] = {}

    @staticmethod
    def _f32_bits(value: float) -> int:
        return struct.unpack("<I", struct.pack("<f", float(value)))[0]

    def _mxfp_runtime_layer(
        self, layer_id: int, barrier_set: int
    ) -> MxfpFfnRuntimeLayer:
        key = (int(layer_id), int(barrier_set))
        existing = self._mxfp_runtime_layers.get(key)
        if existing is not None:
            return existing
        image = self.mxfp_ffn_images[layer_id]
        ready_bars = self.mxfp_ready_bars[barrier_set]
        zero_ready = self.mxfp_zero_ready_bars[barrier_set]
        linear1_tma = TmaTensor(
            self.launcher, image.linear1_weights
        ).mxfp4_load(512)
        down_tma = TmaTensor(
            self.launcher, image.down_weights
        ).mxfp4_load(256)

        linear1_records = torch.zeros(
            (112, 16), dtype=torch.int64, device="cpu"
        )
        middle_flat = self.mxfp_middle_records.view(112, 1536)
        for task in range(112):
            linear1_records[task, 0] = self.mxfp_activation_data.data_ptr()
            linear1_records[task, 2] = image.linear1_scales[
                task, 0
            ].data_ptr()
            linear1_records[task, 3] = (
                self.mxfp_activation_scales.data_ptr()
            )
            linear1_records[task, 4] = image.linear1_scales[
                task, 8
            ].data_ptr()
            linear1_records[task, 5] = (
                linear1_tma.arg
                | (linear1_tma.arg << 16)
                | (task << 32)
            )
            linear1_records[task, 6] = middle_flat[task].data_ptr()
            linear1_records[task, 7] = 2048
            linear1_records[task, 8] = ready_bars[task]
        linear1_metadata = linear1_records.view(torch.uint8).to(self.device)

        # Three metadata views per output tile: one full K2048 task and two
        # K1024 halves. The scheduler uses all 152 full first-wave records and
        # only the 144 half records needed to replace the 72-task second wave.
        down_records = torch.zeros(
            (7 * 32 * 3, 16), dtype=torch.int64, device="cpu"
        )
        for task in range(7 * 32):
            expert, output_tile = divmod(task, 32)
            route_rank = expert - 1
            for variant, (k_start, extra_flags) in enumerate(
                ((0, 0), (0, 8), (4, 8 | 16))
            ):
                record = 3 * task + variant
                down_records[record, 0] = image.down_scales[
                    task, k_start
                ].data_ptr()
                down_records[record, 1] = self.mxfp_middle_records[
                    expert, 0
                ].data_ptr()
                down_records[record, 3] = (
                    down_tma.arg
                    | (self.mxfp_output_tma.arg << 16)
                    | (task << 32)
                )
                down_records[record, 4] = (
                    ready_bars[expert * 16]
                    | (zero_ready[output_tile] << 32)
                )
                down_records[record, 5] = self._f32_bits(1.0)
                down_records[record, 6] = self.mxfp_ffn_output.data_ptr()
                # Expert zero establishes the BF16 model handoff directly.
                # Routed experts and the optional second K half reduce-add
                # after that one-shot frontier.
                down_flags = 4 | 32 | extra_flags
                down_records[record, 8] = k_start | (down_flags << 32)
                down_records[record, 9] = route_rank
        down_metadata = down_records.view(torch.uint8).to(self.device)

        runtime_layer = MxfpFfnRuntimeLayer(
            image=image,
            linear1_tma=linear1_tma,
            down_tma=down_tma,
            linear1_metadata=linear1_metadata,
            down_metadata=down_metadata,
        )
        self._mxfp_runtime_layers[key] = runtime_layer
        return runtime_layer

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
        profile_store_event = None
        if (
            self.args.profile_fp8_coupled_detail
            and name == "attn.hidden.quant_native_fp8"
        ):
            profile_store_event = (
                runtime_config.detail_profile_event_base + 24
            )
        return self._stage(
            name,
            SchedDsv4Fp8QuantUmmaB(
                source.reshape(-1),
                output,
                scale_pack,
                profile_store_event=profile_store_event,
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

    def _fp8_gate_up_swiglu_stage(
        self,
        name: str,
        family: LayerFamily,
        gate_suffix: str,
        up_suffix: str,
        activation: torch.Tensor,
        activation_scale: torch.Tensor,
        output: torch.Tensor,
        *,
        placement: tuple[int, int] | None = None,
        wait_group: str | None = None,
        release_group: str | None = None,
    ) -> Stage:
        gate_linears = tuple(
            self.checkpoint.load_fp8_linear(
                f"layers.{layer_id}.{gate_suffix}", device=self.device
            )
            for layer_id in family.layer_ids
        )
        up_linears = tuple(
            self.checkpoint.load_fp8_linear(
                f"layers.{layer_id}.{up_suffix}", device=self.device
            )
            for layer_id in family.layer_ids
        )
        schedule = SchedFp8Block128GateUpSwiGlu(
            gate_linears[0].weight,
            gate_linears[0].scale,
            up_linears[0].weight,
            up_linears[0].scale,
            activation.reshape(-1),
            activation_scale.reshape(-1),
            output.reshape(-1),
            swiglu_limit=self.config.swiglu_limit,
        )
        schedule = self._layered(
            schedule,
            family,
            tuple(linear.weight for linear in gate_linears),
            tuple(linear.scale for linear in gate_linears),
            tuple(linear.weight for linear in up_linears),
            tuple(linear.scale for linear in up_linears),
        )
        assignment = self.policy.fp8_gemv(output.numel(), activation.numel())
        base_sm = None
        if placement is not None:
            base_sm, num_sms = placement
            assignment = replace(assignment, num_sms=num_sms)
        return self._stage(
            name,
            schedule,
            assignment,
            base_sm=base_sm,
            wait_group=wait_group,
            release_group=release_group,
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
        if scale_pack != SchedFp8GemvUmmaCoupled.SCALE_PACK:
            raise ValueError("common coupled FP8 projections require scale pack 2")
        rows = weights[0].shape[0] * 128
        schedule = SchedFp8GemvUmmaCoupled(
            weights[0],
            activation,
            output.reshape(-1),
            weight_layers=weights,
        )
        assignment = self.policy.fp8_umma_gemv(
            output.numel(), activation.shape[0] * 128
        )
        task_rows = (
            SchedFp8GemvUmmaCoupled.OUTPUT_TILES
            * SchedFp8GemvUmmaCoupled.TILE_M
        )
        assignment = replace(
            assignment,
            num_sms=min(assignment.num_sms, rows // task_rows),
            row_alignment=task_rows,
            tile_rows=task_rows,
            tile_k=256,
        )
        base_sm = None
        if placement is not None:
            base_sm, num_sms = placement
            assignment = replace(
                assignment, num_sms=min(num_sms, rows // task_rows)
            )
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
        if scale_pack != SchedFp8GemvUmmaCoupled.SCALE_PACK:
            raise ValueError("common coupled FP8 projections require scale pack 2")
        if row_slice is None:
            weights = tuple(linear.weight_tiles for linear in linears)
        else:
            start = 0 if row_slice.start is None else row_slice.start
            stop = linears[0].weight_tiles.shape[0] * 128 if row_slice.stop is None else row_slice.stop
            task_rows = (
                SchedFp8GemvUmmaCoupled.OUTPUT_TILES
                * SchedFp8GemvUmmaCoupled.TILE_M
            )
            if start % task_rows or stop % task_rows:
                raise ValueError(
                    "coupled split-K slices must be task-row aligned"
                )
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
        if k // 256 % split_k:
            raise ValueError("split-K override must divide K256 pairs")
        task_rows = (
            SchedFp8GemvUmmaCoupled.OUTPUT_TILES
            * SchedFp8GemvUmmaCoupled.TILE_M
        )
        work_tiles = rows // task_rows * split_k
        num_sms = policy_sms if num_sms is None else int(num_sms)
        num_sms = min(num_sms, work_tiles)
        if not 0 < num_sms <= work_tiles:
            raise ValueError("split-K SM override exceeds logical work tiles")
        output_vector = output.reshape(-1)
        if split_k == 1:
            schedule = SchedFp8GemvUmmaCoupled(
                weights[0],
                activation,
                output_vector,
                weight_layers=weights,
            )
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
        schedule = SchedFp8GemvUmmaCoupled(
            weights[0],
            activation,
            output_reduce,
            split_k=split_k,
            weight_layers=weights,
        )
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
        balanced_k: bool = False,
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
        if scale_pack != SchedFp8GemvUmmaCoupled.SCALE_PACK:
            raise ValueError("common coupled FP8 projections require scale pack 2")
        schedule = SchedFp8GemvUmmaCoupled(
            weights[0],
            activation,
            output_reduce,
            split_k=1 if balanced_k else split_k,
            balanced_k=balanced_k,
            weight_layers=weights,
        )
        rows = weights[0].shape[0] * 128
        k_pairs = activation.shape[0] // 2
        task_rows = (
            SchedFp8GemvUmmaCoupled.OUTPUT_TILES
            * SchedFp8GemvUmmaCoupled.TILE_M
        )
        work_tiles = rows // task_rows * (
            k_pairs if balanced_k else split_k
        )
        if placement is None:
            k = activation.shape[0] * 128
            _, policy_sms = self.policy.fp8_umma_split_k(rows, k)
            base_sm = 0
            num_sms = min(policy_sms, work_tiles)
        else:
            base_sm, num_sms = placement
            num_sms = min(num_sms, work_tiles)
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

    def _fused_hc_projection_operands(
        self,
        family: LayerFamily,
        branch_name: str = "ffn",
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Offline-pack mHC projection weights and scale/base tails."""
        key = (family.representative, family.layer_ids, branch_name)
        cached = self._fused_hc_projection_cache.get(key)
        if cached is not None:
            return cached
        functions = self._family_tensors(family, f"hc_{branch_name}_fn")
        scales = self._family_tensors(family, f"hc_{branch_name}_scale")
        bases = self._family_tensors(family, f"hc_{branch_name}_base")
        packed_weights = tuple(
            function.view(8, 3, 4, 16, 1, 256)
            .permute(0, 3, 4, 1, 2, 5)
            .contiguous()
            for function in functions
        )
        metadata_tails = []
        for scale, base in zip(scales, bases, strict=True):
            tail = torch.empty(
                (SchedDsv4Fp32Bf16Gemv.FUSED_TAIL_ITEMS,),
                dtype=torch.float32,
                device=self.device,
            )
            tail[:3].copy_(scale)
            tail[3:27].copy_(base)
            # The final padding word is not consumed by the reducer.
            metadata_tails.append(tail)
        result = (packed_weights, tuple(metadata_tails))
        self._fused_hc_projection_cache[key] = result
        return result

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
            profile_store_event=(
                runtime_config.detail_profile_event_base + 46
                if self.args.profile_attention_detail
                and name == "attn.q_head_rms_rope"
                else None
            ),
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
            functions[0],
            residual.reshape(-1),
            self.mixes,
            square_sum_output=self.mhc_residual_square_sum,
            metadata_scale=scales[0],
            metadata_base=bases[0],
            metadata_tail_output=self.mhc_metadata_tail,
        )
        project = self._layered(project, family, functions, scales, bases)
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
            residual_square_sum=self.mhc_residual_square_sum,
            packed_metadata=self.mhc_packed_metadata,
            packed_output=self.mhc_packed_output,
            zero_fp32_output=zero_fp32_output,
        )
        pre = self._layered(pre, family, norm_weights)
        pre_stage = self._stage(f"{branch_name}.hc_pre_rms4096", pre)
        post_stage = self._stage(
            f"{branch_name}.hc_post",
            SchedDsv4HcPost(
                branch,
                residual,
                self.post,
                self.comb,
                output_residual,
                launcher=self.launcher,
                packed_coefficients=self.mhc_output_metadata,
            ),
            self.policy.hc_post(
                self.config.hidden_size, self.config.hc_mult
            ),
        )
        return [project_stage, pre_stage], post_stage

    def _fused_attention_ffn_hc_stages(
        self,
        family: LayerFamily,
    ) -> tuple[Stage, list[Stage]]:
        """Fuse attention post into the following FFN mHC projection."""
        packed_weights, metadata_tails = self._fused_hc_projection_operands(
            family
        )
        norm_weights = self._family_tensors(family, "ffn_norm.weight")
        attention_input_ready = f"{family.name}.attn.input.ready"
        attention_post_input_ready = f"{family.name}.attn.post_input.ready"
        metadata_ready = f"{family.name}.ffn.hc.metadata.ready"
        residual_ready = f"{family.name}.ffn.hc.residual.ready"
        ffn_input_ready = f"{family.name}.ffn.input.ready"
        grouped_preattention = {
            "q_a", "q_b", "kv"
        }.issubset(self.splitk_components)

        tail_copy = SchedCopy(
            (
                TmaLoad1D(metadata_tails[0]),
                TmaStore1D(self.mhc_fused_metadata_tail),
            ),
            size=self.mhc_fused_metadata_tail.numel()
            * self.mhc_fused_metadata_tail.element_size(),
        )
        tail_copy = self._layered(
            tail_copy,
            family,
            metadata_tails,
        )
        tail_stage = self._stage(
            "ffn.hc_metadata_tail",
            tail_copy,
            1,
            base_sm=self.sms - 1,
            wait_for_previous=not grouped_preattention,
            wait_group=(
                attention_input_ready if grouped_preattention else None
            ),
            release_group=metadata_ready,
        )

        fused_project = SchedDsv4Fp32Bf16Gemv(
            packed_weights[0],
            self.next_residual.reshape(-1),
            self.mixes,
            fused_post_input_record=self.attention_post_input_record,
            fused_post_output=self.next_residual,
            fused_partial_metadata=self.mhc_fused_metadata,
            packed_coefficients=self.mhc_output_metadata,
            launcher=self.launcher,
        )
        fused_project = self._layered(
            fused_project,
            family,
            packed_weights,
        )
        fused_stage = self._stage(
            "attn.hc_post_ffn.hc_project",
            fused_project,
            SchedDsv4Fp32Bf16Gemv.FUSED_TASK_SMS,
            base_sm=0,
            wait_for_previous=False,
            wait_group_roles=(
                (attention_input_ready, "coefficients"),
                (attention_post_input_ready, "record"),
            ),
            release_group_roles=(
                (metadata_ready, "metadata"),
                (residual_ready, "residual"),
            ),
        )

        pre = SchedDsv4HcPreRms(
            self.next_residual,
            self.mixes,
            metadata_tails[0][:3],
            metadata_tails[0][3:27],
            norm_weights[0],
            self.norm_hidden,
            self.post,
            self.comb,
            residual_square_sum=self.mhc_fused_residual_square_sum,
            packed_metadata=self.mhc_fused_metadata,
            packed_output=self.mhc_packed_output,
            split_metadata_splits=SchedDsv4Fp32Bf16Gemv.FUSED_SPLITS,
        )
        pre = self._layered(pre, family, norm_weights)
        pre_stage = self._stage(
            "ffn.hc_pre_rms4096",
            pre,
            1,
            base_sm=128,
            wait_for_previous=False,
            wait_group_roles=(
                (metadata_ready, "metadata"),
                (residual_ready, "residual"),
            ),
            release_group=ffn_input_ready,
        )
        return tail_stage, [fused_stage, pre_stage]

    def _build_attention(
        self, family: LayerFamily
    ) -> tuple[list[Stage], Stage]:
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
        attention_post_input_ready = f"{family.name}.attn.post_input.ready"
        q_a_ready = f"{family.name}.attn.q_a.ready"
        q_norm_ready = f"{family.name}.attn.q_norm.ready"
        split_attention_q_ready = (
            f"{family.name}.attn.split_attention.q.ready"
        )
        kv_ready = f"{family.name}.attn.kv.ready"
        kv_norm_ready = f"{family.name}.attn.kv_norm.ready"
        qkv_prefix_join = f"{family.name}.attn.qkv.prefix.join"
        attention_input_ready = f"{family.name}.attn.input.ready"
        compressor_reset_ready = f"{family.name}.attn.compressor.reset.ready"
        compressor_projection_ready = (
            f"{family.name}.attn.compressor.projection.ready"
        )
        compressor_output_ready = (
            f"{family.name}.attn.compressor.output.ready"
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
        split_q_a = "q_a" in self.splitk_components
        split_q_b = "q_b" in self.splitk_components
        split_kv = "kv" in self.splitk_components
        split_index_q_b = "index_q_b" in self.splitk_components
        split_o_a = "o_a" in self.splitk_components
        split_o_b = "o_b" in self.splitk_components
        attention_rows = self.attention_indices_by_kind[kind].numel()
        use_context1_attention = (
            split_o_a
            and attention_rows == 1
            and self.args.attention_mode in ("auto", "umma-split")
        )
        use_split_umma_attention = split_o_a and self.args.attention_mode in (
            "auto",
            "umma-split",
        )
        context1_q_ready = f"{family.name}.attn.context1.q.ready"
        use_grouped_preattention = split_q_a and split_kv and split_q_b
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
        projection_reset_sms = self.args.projection_reset_sms
        if projection_reset_sms == 0:
            # Compressed layers have independent main- and index-compressor
            # clears on SM64+ after the same hidden-ready edge.  Restricting
            # the projection arena clear to SM0--63 lets those clears overlap
            # instead of queueing behind an all-grid reset.  SWA has no such
            # parallel branch and retains the full-grid reset.
            projection_reset_sms = (
                64
                if use_grouped_preattention
                and kind in ("csa", "hca")
                and plan.should_compress
                else self.sms
            )
        projection_reset = None
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
            projection_reset = self._stage(
                "attn.projections.reset",
                SchedDsv4ZeroFill(
                    self.zero_fill_gate
                    if self.args.projection_reset_position == "after-input"
                    else None,
                    workspace,
                    profile_store_event=(
                        runtime_config.detail_profile_event_base + 26
                        if self.args.profile_fp8_coupled_detail
                        else None
                    ),
                ),
                min(
                    projection_reset_sms,
                    workspace_rows // 4,
                ),
                wait_group=(
                    attention_input_ready
                    if use_grouped_preattention
                    and self.args.projection_reset_position == "after-input"
                    else None
                ),
                release_group=(
                    qkv_input_ready
                    if use_grouped_preattention
                    and self.args.projection_reset_position == "after-input"
                    else None
                ),
            )
        else:
            self._active_splitk_workspace = None
            self._active_splitk_offset = 0
        stages, post = self._hc_stages(
            family, "attn", self.residual, self.next_residual
        )
        if use_grouped_preattention:
            stages[-1] = replace(
                stages[-1], release_group=attention_input_ready
            )
        if projection_reset is not None:
            if self.args.projection_reset_position == "layer-first":
                stages.insert(0, projection_reset)
            else:
                stages.append(projection_reset)
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
                    wait_group=(
                        q_a_ready
                        if self.args.qkv_projection_schedule == "q-first"
                        else qkv_input_ready
                    ),
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
        if (
            use_grouped_preattention
            and kind in ("csa", "hca")
            and plan.should_compress
        ):
            width = cfg.head_dim * (2 if kind == "csa" else 1)
            fused_output = self.compress_fused_projection[: 2 * width]
            compress_values = fused_output[:width]
            compress_scores = fused_output[width:]
            _, q_prefix_sms = self.policy.fp8_umma_split_k(
                cfg.q_lora_rank, cfg.hidden_size
            )
            compressor_sms = 2 * width // 512 * 8
            compressor_base = q_base + q_prefix_sms
            _, q_b_sms = self.policy.fp8_umma_split_k(
                cfg.num_heads * cfg.head_dim, cfg.q_lora_rank
            )
            q_b_free_base = q_base + q_b_sms
            compressor_pool_base = q_b_free_base
            compressor_projection_base = (
                32
                if kind == "csa"
                else compressor_base
            )
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
                    split_k=16,
                    base_sm=compressor_projection_base,
                    wait_group=compressor_reset_ready,
                    release_group=compressor_projection_ready,
                )
            )
        if (
            use_grouped_preattention
            and kind == "csa"
            and plan.should_compress
        ):
            fused_index_output = self.index_compress_fused_projection.reshape(-1)
            index_tail_values = fused_index_output[
                cfg.index_head_dim : 2 * cfg.index_head_dim
            ]
            index_tail_scores = fused_index_output[
                3 * cfg.index_head_dim : 4 * cfg.index_head_dim
            ]
            _, kv_prefix_sms = self.policy.fp8_umma_split_k(
                cfg.head_dim, cfg.hidden_size
            )
            index_compressor_base = kv_base + kv_prefix_sms
            index_compressor_sms = 8
            index_compressor_pool_base = index_compressor_base
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
                    split_k=16,
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
                history_pool_base = compressor_pool_base
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
                        release_group=compressor_output_ready,
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
                    prefetch_static_inputs=True,
                )
                pool = self._layered(
                    pool, family, ape_rows, norm_weights
                )
                stages.append(
                    self._stage(
                        "attn.compressor.pool_norm_rope",
                        pool,
                        base_sm=compressor_pool_base,
                        input_role="tail",
                        wait_group=compressor_projection_ready,
                        prefetch_before_wait=True,
                        release_group=compressor_output_ready,
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
                tail_values=index_tail_values,
                tail_scores=index_tail_scores,
                tail_bias=index_ape_rows[0],
                hadamard=True,
                fixed_table_id=self.resident_rope_table_ids[
                    self.compressed_output_rope[kind].data_ptr()
                ],
                prefetch_static_inputs=True,
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
                    base_sm=index_compressor_pool_base,
                    input_role="tail",
                    wait_group=index_compressor_projection_ready,
                    prefetch_before_wait=True,
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
        fuse_context1_q_rms = (
            use_context1_attention
            and split_q_b
            and self.direct_splitk_bf16
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
                    release_group=(
                        context1_q_ready
                        if use_context1_attention
                        and (fuse_q_splitk_epilogue or fuse_context1_q_rms)
                        else (
                            split_attention_q_ready
                            if use_split_umma_attention
                            and fuse_q_splitk_epilogue
                            else None
                        )
                    ),
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
        if not fuse_q_splitk_epilogue and not fuse_context1_q_rms:
            stages.append(
                self._rms_rope_stage(
                    "attn.q_head_rms_rope",
                    self.q,
                    rope_table,
                    self.q_rope,
                    release_group=(
                        context1_q_ready
                        if use_context1_attention
                        else (
                            split_attention_q_ready
                            if use_split_umma_attention
                            else None
                        )
                    ),
                )
            )
        if (
            kind in ("csa", "hca")
            and not use_grouped_preattention
            and plan.should_compress
        ):
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
                    not use_packed_pool
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
                        self._rms_rope_stage(
                            "attn.compressor.norm_rope",
                            self.attention_pooled[kind],
                            self.compressed_output_rope[kind],
                            self.current_compressed_rows[kind],
                            family=family,
                            weight_suffix="attn.compressor.norm.weight",
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
            if not use_grouped_preattention and plan.should_compress:
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
        o_a_split_k = 4 if self.args.context_length == cfg.sliding_window else 2
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
        o_rank_ready = f"{family.name}.attn.o_rank.native.ready"
        if use_context1_attention:
            native_heads = self.o_group_native_fp8.view(
                cfg.num_heads, 4, 2048
            )
            for group in range(cfg.o_groups):
                group_input_ready = (
                    f"{family.name}.attn.o_a.g{group}.input.ready"
                )
                context1 = SchedDsv4AttentionContext1Fp8Sm100(
                    self.q if fuse_context1_q_rms else self.q_rope,
                    self.current_kv_rows[kind],
                    sinks[0],
                    rope_table,
                    native_heads,
                    head_start=group * 8,
                    head_count=8,
                    normalize_q=fuse_context1_q_rms,
                )
                context1 = self._layered(context1, family, sinks)
                stages.append(
                    self._stage(
                        f"attn.sparse_{kind}.context1_quant_g{group}",
                        context1,
                        8,
                        base_sm=group * 16,
                        input_role="input",
                        wait_group=context1_q_ready,
                        prefetch_before_wait=True,
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
                        split_k=o_a_split_k,
                        num_sms=16,
                        wait_group=group_input_ready,
                        release_group=output_join_group,
                    )
                )
        elif use_split_umma_attention:
            num_splits = (attention_rows + 63) // 64
            # Keep split-attention production off the early vcores that carry
            # q_b and then become the first reducer/O_a partition.  The final
            # vcores are otherwise idle here and fit the complete producer
            # split, avoiding a producer -> reducer placement tail.
            producer_base_sm = (
                self.args.sms - num_splits
                if self.args.sms >= 152
                else 0
            )
            if producer_base_sm + num_splits > self.args.sms:
                raise ValueError(
                    "split-attention producer placement exceeds resident grid"
                )
            partials = self.attention_partial_workspace[:num_splits]
            metadata = self.attention_metadata_workspace[:num_splits]
            q_tma = TmaTensor(
                self.launcher, self.q_rope
            ).wgmma_load(64, 512, Major.K).encode_64k()
            kv_tma = TmaTensor(
                self.launcher, self.attention_cache[kind]
            ).wgmma_load(64, 512, Major.K)
            kv_v_tma = TmaTensor(
                self.launcher, self.attention_cache[kind]
            ).wgmma_load(64, 128, Major.MN)
            partial_tma = TmaTensor(
                self.launcher,
                partials.reshape(num_splits * cfg.num_heads, cfg.head_dim),
            ).rowmajor_2d("store", cfg.num_heads, 128)
            partial_ready_groups = (
                f"{family.name}.attn.split64.group0.ready",
                f"{family.name}.attn.split64.group1.ready",
            )
            producer = SchedDsv4AttentionSplit64UmmaSm100(
                self.q_rope,
                self.attention_cache[kind],
                attention_rows,
                partials,
                metadata,
                q_tma=q_tma,
                kv_tma=kv_tma,
                kv_v_tma=kv_v_tma,
                partial_tma=partial_tma,
                gate_kv_last_split_only=(
                    use_grouped_preattention
                    and kind in ("csa", "hca")
                    and plan.should_compress
                ),
            )
            stages.append(
                self._stage(
                    f"attn.sparse_{kind}.split64_umma",
                    producer,
                    num_splits,
                    base_sm=producer_base_sm,
                    input_role="q",
                    prefetch_before_wait=True,
                    wait_group_roles=(
                        (
                            (split_attention_q_ready, "q"),
                            (compressor_output_ready, "kv"),
                        )
                        if use_grouped_preattention
                        and kind in ("csa", "hca")
                        and plan.should_compress
                        else ((split_attention_q_ready, "q"),)
                    ),
                    release_group_roles=(
                        (partial_ready_groups[0], "output0"),
                        (partial_ready_groups[1], "output1"),
                    ),
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
                        16,
                        base_sm=group * 16,
                        wait_group_roles=(
                            (partial_ready_groups[0], "partials0"),
                            (partial_ready_groups[1], "partials1"),
                        ),
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
                        split_k=o_a_split_k,
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
                group_input_ready = (
                    f"{family.name}.attn.o_a.g{group}.input.ready"
                )
                placement = (
                    (group * 16, 16)
                    if split_o_a
                    else self.policy.parallel_partition(group, cfg.o_groups)
                )
                start = group * cfg.o_lora_rank
                if split_o_a:
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
                        self._fp8_quant_stage(
                            f"attn.o_a.g{group}.quant_fp8",
                            grouped[group],
                            self.o_group_fp8[group],
                            self.o_group_scale[group],
                            placement=placement,
                            wait_group=output_ready_group,
                            release_group=group_input_ready,
                        )
                    )
                    stages.append(
                        self._fp8_linear_stage(
                            f"attn.o_a.g{group}",
                            family,
                            "attn.wo_a",
                            self.o_group_fp8[group],
                            self.o_group_scale[group],
                            self.o_rank[group],
                            row_slice=slice(start, start + cfg.o_lora_rank),
                            placement=placement,
                            wait_group=group_input_ready,
                            release_group=output_join_group,
                        )
                    )
        if split_o_b:
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
                    split_k=8,
                    num_sms=128,
                    wait_group=o_rank_ready,
                    release_group=attention_post_input_ready,
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
                    release_group=attention_post_input_ready,
                )
            )
        if self._active_splitk_workspace is not None:
            if self._active_splitk_offset != self._active_splitk_workspace.numel():
                raise ValueError(
                    "split-K accumulator workspace was not consumed exactly"
                )
            self._active_splitk_workspace = None
            self._active_splitk_offset = 0
        return stages, post

    @staticmethod
    def _row_pointer(tensor: torch.Tensor, row_start: int) -> int:
        return (
            tensor.data_ptr()
            + row_start * tensor.stride(0) * tensor.element_size()
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

    def _build_mxfp_ffn(
        self, family: LayerFamily
    ) -> tuple[Stage, list[Stage]]:
        cfg = self.config
        async_reload = (
            runtime_config.async_barrier_reload_enabled
            and family.representative >= cfg.num_hash_layers
        )
        heterogeneous_pair = (
            self.args.layers == 2 and self.layer_ids[0] != self.layer_ids[1]
        )
        barrier_set = (
            (
                family.representative
                - (
                    self.layer_ids[0]
                    if heterogeneous_pair
                    else cfg.num_hash_layers
                )
            )
            & 1
            if async_reload or heterogeneous_pair
            else (
                1
                if (
                    self.args.layers == cfg.num_layers
                    and family.name == "layers4-42.csa_score"
                )
                else 0
            )
        )
        # The two score-layer families own disjoint internal MXFP banks.  The
        # first bank remains dead after HCA, so defer its restore and clear the
        # two contiguous banks together during the second layer's post work.
        # This preserves a full HCA->CSA interval before bank-0 reuse while
        # halving clear-command decode, publication, and stage-join overhead.
        overlap_internal_clear = async_reload and barrier_set == 1
        route_record = self.route_records[barrier_set]
        route_indices = route_record[:32].view(torch.int32)
        route_weights = route_record[32:64].view(torch.float32)
        runtime_layers = tuple(
            self._mxfp_runtime_layer(layer_id, barrier_set)
            for layer_id in family.layer_ids
        )
        representative = runtime_layers[0]
        # The fixed compute task validates seven physical slots.  Its retained
        # LDU plans below still point at the complete 257-stream offline image,
        # so routed slots are selected dynamically from the packed top-k record.
        linear1_physical = representative.image.linear1_weights[:112]
        linear1_scale_physical = representative.image.linear1_scales[:112]
        gate_weight = linear1_physical[:, :8].contiguous()
        up_weight = linear1_physical[:, 8:].contiguous()
        gate_scale = linear1_scale_physical[:, :8].contiguous()
        up_scale = linear1_scale_physical[:, 8:].contiguous()
        down_weight = representative.image.down_weights[: 7 * 32]
        down_scale = representative.image.down_scales[: 7 * 32]
        linear1 = SchedMxfp4Mxfp8GateUpSiluFixedRing(
            gate_weight,
            gate_scale,
            up_weight,
            up_scale,
            self.mxfp_activation_data,
            self.mxfp_activation_scales,
            self.mxfp_middle_data,
            self.mxfp_middle_scales,
            representative.linear1_tma,
            representative.linear1_tma,
            representative.linear1_metadata,
            tile_k=512,
        )
        down = SchedMxfp4Mxfp8DownFixedRing(
            down_weight,
            down_scale,
            self.mxfp_middle_records,
            self.mxfp_ffn_output,
            representative.down_tma,
            representative.down_metadata,
            output_n_major=True,
            fp32_output=False,
        )
        resident = SchedMxfp4Mxfp8RoutedResidentFfn(
            linear1,
            down,
            route_record,
            representative.image.linear1_weights,
            representative.image.linear1_scales,
            representative.image.down_weights,
            representative.image.down_scales,
            profile_output_event=(
                FFN_OUTPUT_PROFILE_EVENT_BASE
                if self.args.profile_steps
                else None
            ),
        )
        resident = SchedLayeredMxfp4Mxfp8RoutedResidentFfn(
            resident,
            tuple(layer.linear1_metadata for layer in runtime_layers),
            tuple(layer.down_metadata for layer in runtime_layers),
            tuple(layer.image.linear1_weights for layer in runtime_layers),
            tuple(layer.image.linear1_scales for layer in runtime_layers),
            tuple(layer.image.down_weights for layer in runtime_layers),
            tuple(layer.image.down_scales for layer in runtime_layers),
            counter_strides=family.counter_strides,
            linear1_tmas=tuple(layer.linear1_tma for layer in runtime_layers),
            down_tmas=tuple(layer.down_tma for layer in runtime_layers),
        )
        tail_stage, stages = self._fused_attention_ffn_hc_stages(family)
        post = self._stage(
            "ffn.hc_post",
            SchedDsv4HcPost(
                self.mxfp_ffn_output[0],
                self.next_residual,
                self.post,
                self.comb,
                self.residual,
                launcher=self.launcher,
                packed_coefficients=self.mhc_output_metadata,
            ),
            self.policy.hc_post(
                self.config.hidden_size, self.config.hc_mult
            ),
        )
        ffn_input_ready = f"{family.name}.ffn.input.ready"
        quant_records_ready = f"{family.name}.ffn.mx.quant.ready"
        router_scores_ready = f"{family.name}.ffn.router.scores.ready"
        resident_input_ready = f"{family.name}.ffn.mx.input.ready"
        # CTA placements are disjoint at the independent frontier:
        # router [0,128), route 128, split [136,152), and quant [144,152).
        # Quant precedes split on their shared eight CTAs.
        resident_done = f"{family.name}.ffn.mx.resident.done"
        resident_stage = self._stage(
            "ffn.mx.resident",
            resident,
            self.sms,
            base_sm=0,
            wait_group_roles=((resident_input_ready, "input"),),
            release_group=resident_done if overlap_internal_clear else None,
        )
        if overlap_internal_clear:
            first_bar = self.mxfp_internal_barrier_start
            clear_count = self.mxfp_internal_barrier_stop - first_bar
            post = replace(
                post,
                schedule=SchedOverlapAsyncBarrierReload(
                    post.schedule,
                    post.num_sms,
                    self.launcher.bars_src,
                    first_bar,
                    clear_count,
                    post.num_sms,
                    runtime_config.async_barrier_reload_workers,
                ),
                num_sms=self.sms,
                input_role="input",
                base_sm=0,
                wait_for_previous=False,
                wait_group=resident_done,
            )
        stages.extend(
            (
                self._stage(
                    "ffn.hidden.quant_mxfp8",
                    SchedDsv4Mxfp8QuantFfnInput(
                        self.norm_hidden, self.mxfp_input_records
                    ),
                    8,
                    base_sm=self.sms - 8,
                    wait_group=ffn_input_ready,
                    release_group=quant_records_ready,
                ),
            )
        )

        router_weights = self._family_tensors(family, "ffn.gate.weight")
        hash_routing = family.representative < cfg.num_hash_layers
        if hash_routing:
            router_biases = (self.zero_bias,) * len(family.layer_ids)
            hash_rows = torch.stack(
                tuple(self._hash_row(layer_id) for layer_id in family.layer_ids)
            ).contiguous()
        else:
            router_biases = self._family_tensors(family, "ffn.gate.bias")
            hash_rows = self.zero_hash
        stages.append(
            self._stage(
                "ffn.router.prepared",
                SchedLayeredDsv4RouterBf16Gemv(
                    router_weights,
                    self.norm_hidden,
                    router_biases,
                    self.router_prepared,
                    counter_strides=family.counter_strides,
                ),
                128,
                base_sm=0,
                wait_group=ffn_input_ready,
                release_group=router_scores_ready,
            )
        )
        stages.extend(
            (
                self._stage(
                    "ffn.hidden.split_mxfp8",
                    SchedDsv4SplitMxfp8FfnInputRecords(
                        self.mxfp_input_records,
                        self.mxfp_activation_data,
                        self.mxfp_activation_scales,
                    ),
                    16,
                    base_sm=self.sms - 16,
                    wait_group=quant_records_ready,
                    release_group=resident_input_ready,
                ),
                self._stage(
                    "ffn.route.prepared",
                    SchedDsv4RouteTop6(
                        self.router_prepared,
                        None,
                        hash_rows,
                        route_indices,
                        route_weights,
                        hash_routing=hash_routing,
                        route_scale=cfg.route_scale,
                        pretransformed=True,
                        packed_output=route_record,
                        hash_counter_strides=(
                            family.counter_strides if hash_routing else ()
                        ),
                    ),
                    1,
                    base_sm=128,
                    wait_group=router_scores_ready,
                    release_group=resident_input_ready,
                ),
                resident_stage,
                post,
            )
        )
        return tail_stage, stages

    def _build_ffn(
        self, family: LayerFamily
    ) -> tuple[Stage, list[Stage]]:
        return self._build_mxfp_ffn(family)

    def _build_family(self, family: LayerFamily) -> list[Stage]:
        attention, _ = self._build_attention(family)
        tail_stage, ffn = self._build_ffn(family)
        tail_insert_after = max(
            index
            for index, stage in enumerate(attention)
            if stage.name in {
                "attn.hc_pre_rms4096",
                "attn.projections.reset",
            }
        )
        attention.insert(tail_insert_after + 1, tail_stage)
        return attention + ffn

    def _apply_cross_layer_hc_fusion(
        self,
        previous_family: LayerFamily,
        next_family: LayerFamily,
    ) -> None:
        """Fuse each HCA FFN-post into the following CSA projection."""
        previous_stages = self.family_stages[previous_family.representative]
        next_stages = self.family_stages[next_family.representative]
        if previous_stages[-1].name != "ffn.hc_post":
            raise ValueError("cross-layer mHC fusion requires an FFN-post tail")

        packed_weights, metadata_tails = self._fused_hc_projection_operands(
            next_family, "attn"
        )
        norm_weights = self._family_tensors(next_family, "attn_norm.weight")
        previous_ffn_ready = f"{previous_family.name}.ffn.input.ready"
        previous_resident_input_ready = (
            f"{previous_family.name}.ffn.mx.input.ready"
        )
        metadata_ready = f"{next_family.name}.attn.hc.metadata.ready"
        residual_ready = f"{next_family.name}.attn.hc.residual.ready"
        attention_input_ready = f"{next_family.name}.attn.input.ready"

        tail_copy = SchedCopy(
            (
                TmaLoad1D(metadata_tails[0]),
                TmaStore1D(self.mhc_fused_metadata_tail),
            ),
            size=self.mhc_fused_metadata_tail.numel()
            * self.mhc_fused_metadata_tail.element_size(),
        )
        tail_copy = self._layered(
            tail_copy,
            next_family,
            metadata_tails,
        )
        tail_stage = self._stage(
            "attn.hc_metadata_tail",
            tail_copy,
            1,
            base_sm=self.sms - 1,
            wait_for_previous=False,
            wait_group=previous_ffn_ready,
            release_group=metadata_ready,
        )
        previous_pre_index = next(
            index
            for index, stage in enumerate(previous_stages)
            if stage.name == "ffn.hc_pre_rms4096"
        )
        packed_record = self.mxfp_ffn_output[:5]
        residual_pack = SchedCopy(
            (
                TmaLoad1D(self.next_residual),
                TmaStore1D(packed_record[1:]),
            ),
            size=(
                self.next_residual.numel()
                * self.next_residual.element_size()
            ),
        )
        residual_pack_stage = self._stage(
            "ffn.hc_post_input_pack",
            residual_pack,
            1,
            base_sm=133,
            wait_for_previous=False,
            wait_group=previous_ffn_ready,
            release_group=previous_resident_input_ready,
        )
        previous_stages[previous_pre_index + 1:previous_pre_index + 1] = [
            tail_stage,
            residual_pack_stage,
        ]

        previous_stages.pop()
        if previous_stages[-1].name != "ffn.mx.resident":
            raise ValueError(
                "cross-layer mHC fusion requires a resident FFN tail"
            )

        project_index = next(
            index
            for index, stage in enumerate(next_stages)
            if stage.name == "attn.hc_project"
        )
        if next_stages[project_index + 1].name != "attn.hc_pre_rms4096":
            raise ValueError(
                "cross-layer mHC fusion requires adjacent attention project/pre"
            )
        del next_stages[project_index:project_index + 2]

        fused_project = SchedDsv4Fp32Bf16Gemv(
            packed_weights[0],
            self.residual.reshape(-1),
            self.mixes,
            fused_post_input_record=packed_record,
            fused_post_output=self.residual,
            fused_partial_metadata=self.mhc_fused_metadata,
            packed_coefficients=self.mhc_output_metadata,
            launcher=self.launcher,
            prefetch_operands_before_resident_reset=True,
        )
        fused_project = self._layered(
            fused_project,
            next_family,
            packed_weights,
        )
        fused_stage = self._stage(
            "ffn.hc_post_next_attn.hc_project",
            fused_project,
            SchedDsv4Fp32Bf16Gemv.FUSED_TASK_SMS,
            base_sm=0,
            input_role="record",
            prefetch_before_wait=True,
            prefetch_before_resident_reset=True,
            release_group_roles=(
                (metadata_ready, "metadata"),
                (residual_ready, "residual"),
            ),
        )

        pre = SchedDsv4HcPreRms(
            self.residual,
            self.mixes,
            metadata_tails[0][:3],
            metadata_tails[0][3:27],
            norm_weights[0],
            self.norm_hidden,
            self.post,
            self.comb,
            residual_square_sum=self.mhc_fused_residual_square_sum,
            packed_metadata=self.mhc_fused_metadata,
            packed_output=self.mhc_packed_output,
            split_metadata_splits=SchedDsv4Fp32Bf16Gemv.FUSED_SPLITS,
        )
        pre = self._layered(pre, next_family, norm_weights)
        pre_stage = self._stage(
            "attn.hc_pre_rms4096",
            pre,
            1,
            base_sm=128,
            wait_for_previous=False,
            wait_group_roles=(
                (metadata_ready, "metadata"),
                (residual_ready, "residual"),
            ),
            release_group=attention_input_ready,
        )
        next_stages[project_index:project_index] = [fused_stage, pre_stage]

    def _apply_loopback_hc_fusion(
        self,
        layer2_family: LayerFamily,
        hca_family: LayerFamily,
        csa_family: LayerFamily,
    ) -> None:
        """Fuse layer-2/CSA post work into the following repeated HCA."""
        layer2_stages = self.family_stages[layer2_family.representative]
        hca_stages = self.family_stages[hca_family.representative]
        csa_stages = self.family_stages[csa_family.representative]
        if (
            layer2_stages[-1].name != "ffn.hc_post"
            or csa_stages[-1].name != "ffn.hc_post"
        ):
            raise ValueError(
                "loop-back mHC fusion requires layer-2 and CSA FFN-post tails"
            )

        packed_record = self.mxfp_ffn_output[:5]

        def replace_post_with_record_pack(
            family: LayerFamily,
            stages: list[Stage],
        ) -> Stage:
            post_stage = stages.pop()
            pre_index = next(
                index
                for index, stage in enumerate(stages)
                if stage.name == "ffn.hc_pre_rms4096"
            )
            ffn_input_ready = f"{family.name}.ffn.input.ready"
            resident_input_ready = f"{family.name}.ffn.mx.input.ready"
            residual_pack = SchedCopy(
                (
                    TmaLoad1D(self.next_residual),
                    TmaStore1D(packed_record[1:]),
                ),
                size=(
                    self.next_residual.numel()
                    * self.next_residual.element_size()
                ),
            )
            stages[pre_index + 1:pre_index + 1] = [
                self._stage(
                    "ffn.hc_post_input_pack",
                    residual_pack,
                    1,
                    base_sm=133,
                    wait_for_previous=False,
                    wait_group=ffn_input_ready,
                    release_group=resident_input_ready,
                )
            ]
            if stages[-1].name != "ffn.mx.resident":
                raise ValueError(
                    "loop-back mHC fusion requires a resident FFN tail"
                )
            return post_stage

        replace_post_with_record_pack(layer2_family, layer2_stages)
        csa_post = replace_post_with_record_pack(csa_family, csa_stages)
        if not isinstance(csa_post.schedule, SchedOverlapAsyncBarrierReload):
            raise ValueError(
                "loop-back mHC fusion requires the CSA asynchronous clear tail"
            )
        clear_wrapper = csa_post.schedule
        csa_resident_done = f"{csa_family.name}.ffn.mx.resident.done"
        if csa_stages[-1].release_group != csa_resident_done:
            raise ValueError(
                "loop-back internal clear requires the CSA resident release"
            )
        csa_stages[-1] = replace(csa_stages[-1], release_group=None)

        packed_weights, metadata_tails = self._fused_hc_projection_operands(
            hca_family, "attn"
        )
        norm_weights = self._family_tensors(
            hca_family, "attn_norm.weight"
        )
        metadata_ready = f"{hca_family.name}.attn.hc.metadata.ready"
        residual_ready = f"{hca_family.name}.attn.hc.residual.ready"
        attention_input_ready = f"{hca_family.name}.attn.input.ready"
        resident_input_ready = f"{hca_family.name}.ffn.mx.input.ready"

        project_index = next(
            index
            for index, stage in enumerate(hca_stages)
            if stage.name == "attn.hc_project"
        )
        if hca_stages[project_index + 1].name != "attn.hc_pre_rms4096":
            raise ValueError(
                "loop-back mHC fusion requires adjacent HCA project/pre stages"
        )
        del hca_stages[project_index:project_index + 2]

        tail_copy = SchedCopy(
            (
                TmaLoad1D(metadata_tails[0]),
                TmaStore1D(self.mhc_fused_metadata_tail),
            ),
            size=(
                self.mhc_fused_metadata_tail.numel()
                * self.mhc_fused_metadata_tail.element_size()
            ),
        )
        tail_copy = self._layered(
            tail_copy,
            hca_family,
            metadata_tails,
        )
        tail_stage = self._stage(
            "attn.hc_metadata_tail",
            tail_copy,
            1,
            base_sm=self.sms - 1,
            release_group=metadata_ready,
        )

        fused_project = SchedDsv4Fp32Bf16Gemv(
            packed_weights[0],
            self.residual.reshape(-1),
            self.mixes,
            fused_post_input_record=packed_record,
            fused_post_output=self.residual,
            fused_partial_metadata=self.mhc_fused_metadata,
            packed_coefficients=self.mhc_output_metadata,
            launcher=self.launcher,
            profile_operands=self.args.profile_loopback_boundary,
        )
        fused_project = self._layered(
            fused_project,
            hca_family,
            packed_weights,
        )
        fused_project = SchedOverlapAsyncBarrierReload(
            fused_project,
            SchedDsv4Fp32Bf16Gemv.FUSED_TASK_SMS,
            clear_wrapper.bar_source,
            clear_wrapper.first_bar,
            clear_wrapper.count,
            132,
            clear_wrapper.workers,
            special_slot=clear_wrapper.special_slot,
            clear_input_role="metadata",
            skip_initial_loop=True,
        )
        fused_stage = self._stage(
            "ffn.hc_post_next_attn.hc_project",
            fused_project,
            132 + clear_wrapper.workers,
            base_sm=0,
            wait_for_previous=False,
            parallel_with_previous=True,
            release_group_roles=(
                (metadata_ready, "metadata"),
                (residual_ready, "residual"),
                (resident_input_ready, "clear"),
            ),
        )

        pre = SchedDsv4HcPreRms(
            self.residual,
            self.mixes,
            metadata_tails[0][:3],
            metadata_tails[0][3:27],
            norm_weights[0],
            self.norm_hidden,
            self.post,
            self.comb,
            residual_square_sum=self.mhc_fused_residual_square_sum,
            packed_metadata=self.mhc_fused_metadata,
            packed_output=self.mhc_packed_output,
            split_metadata_splits=SchedDsv4Fp32Bf16Gemv.FUSED_SPLITS,
        )
        pre = self._layered(pre, hca_family, norm_weights)
        pre_stage = self._stage(
            "attn.hc_pre_rms4096",
            pre,
            1,
            base_sm=128,
            wait_for_previous=False,
            wait_group_roles=(
                (metadata_ready, "metadata"),
                (residual_ready, "residual"),
            ),
            release_group=attention_input_ready,
        )
        hca_stages[project_index:project_index] = [
            tail_stage,
            fused_stage,
            pre_stage,
        ]

        self.head_stages.insert(
            0,
            self._stage(
                "head.final_hc_post",
                clear_wrapper.inner,
                clear_wrapper.inner_sms,
                base_sm=0,
            ),
        )

    def _build_head(self) -> list[Stage]:
        cfg = self.config
        head_fn = self._tensor("hc_head_fn")
        head_scale = self._tensor("hc_head_scale")
        head_base = self._tensor("hc_head_base")
        self.head_mixes = torch.empty((4,), dtype=torch.float32, device=self.device)
        self.fp8_head = (
            not self.args.bf16_head and self.args.vocab_size == cfg.vocab_size
        )
        self.bf16_umma_head = (
            self.args.bf16_head and self.args.vocab_size == cfg.vocab_size
        )
        self.compact_head = self.fp8_head or self.bf16_umma_head
        self.head_norm = torch.empty(
            ((8, cfg.hidden_size) if self.bf16_umma_head else (cfg.hidden_size,)),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.head_norm_oracle = None
        self.fp8_head_activation_oracle = None
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
                "head.hc_rms",
                SchedDsv4HcHeadRms(
                    self.residual,
                    self.head_mixes,
                    head_scale,
                    head_base,
                    self._tensor("norm.weight"),
                    self.head_norm[0] if self.bf16_umma_head else self.head_norm,
                    rms_epsilon=cfg.rms_epsilon,
                ),
                1,
            ),
        ]
        head_weight = self._tensor("head.weight")[: self.args.vocab_size]
        if self.fp8_head:
            print(
                "DSV4_HEAD_PREPROCESS status=START "
                f"rows={self.args.vocab_size} k={cfg.hidden_size} "
                "format=native_fp8_coupled",
                flush=True,
            )
            preprocess_started = time.monotonic()
            head_weight_fp8, head_weight_scale = quantize_fp8_block128(
                head_weight
            )
            head_m_tiles = self.args.vocab_size // 128
            head_k_tiles = cfg.hidden_size // 128
            self.head_weight_native_fp8 = torch.empty(
                (head_m_tiles, head_k_tiles, 16896),
                dtype=torch.uint8,
                device=self.device,
            )
            runtime.prepack_fp8_checkpoint(
                head_weight_fp8,
                head_weight_scale,
                self.head_weight_native_fp8,
                SchedFp8GemvUmmaCoupled.SCALE_PACK,
            )
            self.head_input_native_fp8 = torch.empty(
                (head_k_tiles, SchedFp8GemvUmmaCoupled.ACTIVATION_TILE_BYTES),
                dtype=torch.uint8,
                device=self.device,
            )
            head_assignment = self.policy.fp8_umma_gemv(
                self.args.vocab_size, cfg.hidden_size
            )
            head_assignment = replace(
                head_assignment,
                num_sms=min(head_assignment.num_sms, head_m_tiles // 2),
                row_alignment=256,
                tile_rows=256,
                tile_k=256,
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
                    self._native_fp8_quant_stage(
                        "head.quant_native_fp8",
                        self.head_norm,
                        self.head_input_native_fp8,
                    ),
                    self._stage(
                        "head.logits.fp8_coupled",
                        SchedFp8GemvUmmaCoupled(
                            self.head_weight_native_fp8,
                            self.head_input_native_fp8,
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
                f"weight_gib={self.head_weight_native_fp8.numel() / (1 << 30):.3f} "
                f"elapsed_s={time.monotonic() - preprocess_started:.3f}",
                flush=True,
            )
            return stages

        if self.bf16_umma_head:
            # Reuse the retained four-output BF16 LM-head task.  The hardware
            # UMMA atom is N8.  The preceding small mHC-head/RMS task emits
            # only row zero; the other seven physical rows remain
            # uninitialized and are ignored.  The ordinary K-major TMA path
            # still performs the proven swizzled N8 load.
            epoch_rows = (
                128
                * Gemv_M128N8Direct4.MNK[0]
                * Gemv_M128N8Direct4.output_groups
            )
            num_epochs = math.ceil(self.args.vocab_size / epoch_rows)
            padded_rows = num_epochs * epoch_rows
            if padded_rows % (128 * Gemv_M128N8Direct4.output_groups):
                raise AssertionError("BF16 LM-head epochs must contain M512 groups")
            print(
                "DSV4_HEAD_PREPROCESS status=START "
                f"rows={self.args.vocab_size} padded_rows={padded_rows} "
                f"k={cfg.hidden_size} format=bf16_tile_major_group4",
                flush=True,
            )
            preprocess_started = time.monotonic()
            epoch_weights = []
            for epoch in range(num_epochs):
                row_start = epoch * epoch_rows
                row_end = min(row_start + epoch_rows, self.args.vocab_size)
                source = head_weight[row_start:row_end]
                if source.shape[0] != epoch_rows:
                    padded = torch.empty(
                        (epoch_rows, cfg.hidden_size),
                        dtype=torch.bfloat16,
                        device=self.device,
                    )
                    # Duplicate a real vocabulary row into the padding.
                    # Equal-value argmax ties select the smaller absolute
                    # index, so padded rows cannot replace their source row.
                    padded[:] = head_weight[0]
                    padded[: source.shape[0]].copy_(source)
                    source = padded
                epoch_weights.append(pack_weight_tile_major(source, 128, 128))
            self.head_weight_bf16_packed = tuple(epoch_weights)
            activation_tma = TmaTensor(
                self.launcher, self.head_norm
            ).wgmma_load(
                Gemv_M128N8Direct4.MNK[1],
                Gemv_M128N8Direct4.MNK[2]
                * Gemv_M128N8Direct4.n_batch,
                Major.K,
            )
            weight_tmas = tuple(
                TmaTensor(self.launcher, weight).wgmma_load_tiled(128, 128)
                for weight in self.head_weight_bf16_packed
            )
            self.head_logits_bf16 = torch.empty(
                (num_epochs, 8, epoch_rows),
                dtype=torch.bfloat16,
                device=self.device,
            )
            argmax_sms = self.sms - 128
            self.head_argmax_partial = torch.empty(
                (num_epochs * argmax_sms, 16),
                dtype=torch.uint8,
                device=self.device,
            )
            self.output_token = torch.empty(
                (1,), dtype=torch.int64, device=self.device
            )
            input_ready = "head.bf16.input.ready"
            partial_ready = "head.bf16.partial.ready"
            logits_ready = tuple(
                f"head.bf16.logits.epoch{epoch}.ready"
                for epoch in range(num_epochs)
            )
            stages[-1] = replace(stages[-1], release_group=input_ready)
            for epoch, weight_tma in enumerate(weight_tmas):
                stages.append(
                    self._stage(
                        f"head.logits.bf16_umma.epoch{epoch}",
                        SchedGemvMGroup(
                            Gemv_M128N8Direct4,
                            (epoch_rows, 8, cfg.hidden_size),
                            (weight_tma, activation_tma),
                            self.head_logits_bf16[epoch],
                            group=False,
                        ),
                        128,
                        base_sm=0,
                        wait_for_previous=False,
                        wait_group=input_ready,
                        release_group=logits_ready[epoch],
                    )
                )
            for epoch in range(num_epochs):
                real_rows = min(
                    epoch_rows,
                    self.args.vocab_size - epoch * epoch_rows,
                )
                partial_base = epoch * argmax_sms
                stages.append(
                    self._stage(
                        f"head.argmax.bf16_umma.epoch{epoch}",
                        SchedArgmaxSmemPartial(
                            self.head_logits_bf16[epoch, 0, :real_rows],
                            self.head_argmax_partial[
                                partial_base : partial_base + argmax_sms
                            ],
                            index_base=epoch * epoch_rows,
                        ),
                        argmax_sms,
                        base_sm=128,
                        wait_for_previous=False,
                        wait_group=logits_ready[epoch],
                        release_group=partial_ready,
                    )
                )
            stages.append(
                self._stage(
                    "head.argmax.bf16_umma",
                    SchedArgmaxSmemReduce(
                        self.head_argmax_partial,
                        self.output_token,
                    ),
                    1,
                    base_sm=128,
                    wait_for_previous=False,
                    wait_group=partial_ready,
                )
            )
            print(
                "DSV4_HEAD_PREPROCESS status=PASS "
                f"weight_gib={sum(weight.numel() * weight.element_size() for weight in self.head_weight_bf16_packed) / (1 << 30):.3f} "
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
        mxfp_reload_start = self.mxfp_internal_barrier_start
        family_barrier_banks = 1
        pair_barrier_banks = (
            2 if runtime_config.async_barrier_reload_enabled else 1
        )
        self.stage_profile_labels: list[str] = []
        self.head_profile_labels: list[str] = []
        self.step_profile_records: list[tuple[int, str, int, int, int]] = []
        self.step_profile_begin_events: dict[int, int] = {}
        self.step_profile_total = 0
        aggregate_labels = (
            "hc_project",
            "hc_pre_rms",
            "hidden_quant",
            "router",
            "route_top6",
            "resident_ffn",
            "hc_post",
        )
        # Slots 2..5 are task-owned resident-FFN timestamps.
        aggregate_base = runtime_config.layer_profile_event_base + 16
        self.ffn_aggregate_events = {
            label: (aggregate_base + 2 * index, aggregate_base + 2 * index + 1)
            for index, label in enumerate(aggregate_labels)
        }
        self.ffn_aggregate_used: set[str] = set()
        # Slots 2..5 are written directly by the resident FFN task.  Keep
        # phase aggregates in a disjoint part of the layer-profile range.
        phase_aggregate_base = runtime_config.layer_profile_event_base + 16
        self.phase_aggregate_events = {
            "attention": (
                phase_aggregate_base,
                phase_aggregate_base + 1,
            ),
            "ffn": (
                phase_aggregate_base + 2,
                phase_aggregate_base + 3,
            ),
        }

        def aggregate_category(name: str) -> str | None:
            exact = {
                "ffn.hc_project": "hc_project",
                "attn.hc_post_ffn.hc_project": "hc_project",
                "ffn.hc_post_next_attn.hc_project": "hc_project",
                "ffn.hc_pre_rms4096": "hc_pre_rms",
                "ffn.hidden.quant_mxfp8": "hidden_quant",
                "ffn.hidden.split_mxfp8": "hidden_quant",
                "ffn.router.prepared": "router",
                "ffn.route.prepared": "route_top6",
                "ffn.mx.resident": "resident_ffn",
                "ffn.hc_post": "hc_post",
            }
            return exact.get(name)

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
                "attn.hc_post_ffn.hc_project",
                "ffn.hc_pre",
                "ffn.route.prepared",
                "ffn.mx.resident",
                "ffn.hc_post",
                "ffn.hc_post_input_pack",
            }:
                return True
            return False

        def queued(
            stage: Stage,
            prefix: str = "",
            *,
            group_namespace: str = "",
            profile_after: bool = False,
            profile_step_event: int | None = None,
            profile_step_begin_event: int | None = None,
            profile_aggregate_events: tuple[int, int] | None = None,
            profile_span_begin: tuple[int, int] | None = None,
            profile_span_end: tuple[int, int] | None = None,
        ) -> SequentialStage:
            nonlocal serial_sm
            def group_name(group: str | None) -> str | None:
                return (
                    f"{group_namespace}{group}"
                    if group is not None
                    else None
                )

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
                parallel_with_previous=stage.parallel_with_previous,
                wait_group=group_name(stage.wait_group),
                release_group=group_name(stage.release_group),
                profile_step_event=profile_step_event,
                profile_step_begin_event=profile_step_begin_event,
                profile_aggregate_events=profile_aggregate_events,
                profile_span_begin=profile_span_begin,
                profile_span_end=profile_span_end,
                prefetch_before_wait=stage.prefetch_before_wait,
                prefetch_before_resident_reset=(
                    stage.prefetch_before_resident_reset
                ),
                wait_group_roles=tuple(
                    (group_name(group), role)
                    for group, role in stage.wait_group_roles
                ),
                release_group_roles=tuple(
                    (group_name(group), role)
                    for group, role in stage.release_group_roles
                ),
            )

        def queued_family(
            family: LayerFamily,
            *,
            enable_profile: bool = True,
            group_namespace: str = "",
        ) -> list[SequentialStage]:
            stages = self.family_stages[family.representative]
            self.step_profile_total = len(stages)
            profile_step_family = (
                family is self.families[0]
                if self.args.profile_fp8_coupled_detail
                else
                (
                    self.args.profile_step_family == "hca"
                    and self.config.attention_kind(family.representative)
                    == "hca"
                    and family.representative >= self.config.num_hash_layers
                )
                if self.args.profile_step_family != "last"
                else (
                    len(self.families) == 1
                    or self.profile_layer_ids[-1] in family.layer_ids
                )
            )
            queued_stages = []
            for index, stage in enumerate(stages):
                stage_profile_after = (
                    enable_profile and profile_stage(stage.name)
                )
                if stage_profile_after:
                    label = (
                        "ffn.outputs_join"
                        if stage.name == "ffn.shared.w2"
                        else stage.name
                    )
                    self.stage_profile_labels.append(label)
                step_event = None
                if (
                    enable_profile
                    and self.args.profile_steps
                    and profile_step_family
                    and self.args.profile_step_start
                    <= index
                    < self.args.profile_step_start + self.args.profile_step_count
                ):
                    step_event = (
                        STEP_PROFILE_EVENT_BASE
                        + index
                        - self.args.profile_step_start
                    )
                step_begin_event = None
                if step_event is not None and self.args.profile_step_frontiers:
                    frontier_base = (
                        LOOPBACK_PROFILE_FRONTIER_BASE
                        if self.args.profile_loopback_boundary
                        else STEP_PROFILE_FRONTIER_BASE
                    )
                    step_begin_event = (
                        frontier_base
                        + index
                        - self.args.profile_step_start
                    )
                aggregate_events = None
                if enable_profile and self.args.profile_ffn_aggregate:
                    category = aggregate_category(stage.name)
                    if category is not None:
                        aggregate_events = self.ffn_aggregate_events[category]
                        self.ffn_aggregate_used.add(category)
                span_begin = None
                span_end = None
                if enable_profile and self.args.profile_phase_aggregate:
                    if stage.name == "attn.hc_project":
                        span_begin = self.phase_aggregate_events["attention"]
                    elif stage.name == "attn.hc_post":
                        span_end = self.phase_aggregate_events["attention"]
                    elif stage.name == "attn.hc_post_ffn.hc_project":
                        span_begin = self.phase_aggregate_events["ffn"]
                        span_end = self.phase_aggregate_events["attention"]
                    elif stage.name == "ffn.hc_project":
                        span_begin = self.phase_aggregate_events["ffn"]
                    elif stage.name == "ffn.hc_post":
                        span_end = self.phase_aggregate_events["ffn"]
                queued_stage = queued(
                    stage,
                    f"{family.name}.",
                    group_namespace=group_namespace,
                    profile_after=(
                        enable_profile
                        and (
                            self.args.profile_layers
                            or self.args.profile_mxfp_ffn_detail
                        )
                        and index + 1 == len(stages)
                    )
                    or stage_profile_after
                    or (
                        self.args.profile_loopback_boundary
                        and family is self.families[3]
                        and index + 1 == len(stages)
                    ),
                    profile_step_event=step_event,
                    profile_step_begin_event=(
                        step_begin_event
                        if step_begin_event is not None
                        else (
                            FP8_COUPLED_STEP_BEGIN_EVENT
                            if self.args.profile_fp8_coupled_detail
                            and stage.name == "attn.q_b"
                            else (
                                FP8_COUPLED_LAYER_BEGIN_EVENT
                                if self.args.profile_fp8_coupled_detail
                                and index == 0
                                else None
                            )
                        )
                    ),
                    profile_aggregate_events=aggregate_events,
                    profile_span_begin=span_begin,
                    profile_span_end=span_end,
                )
                queued_stages.append(queued_stage)
                if step_event is not None:
                    if step_begin_event is not None:
                        self.step_profile_begin_events[index] = step_begin_event
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

        def queued_head() -> list[SequentialStage]:
            queued_stages = []
            head_step_base = self.step_profile_total
            for index, stage in enumerate(self.head_stages):
                profile_after = (
                    self.args.profile_layers
                    and index + 1 < len(self.head_stages)
                )
                if profile_after:
                    self.head_profile_labels.append(stage.name)
                step_index = head_step_base + index
                step_event = None
                if (
                    self.args.profile_steps
                    and self.args.profile_step_family == "last"
                    and self.args.profile_step_start
                    <= step_index
                    < self.args.profile_step_start + self.args.profile_step_count
                ):
                    step_event = (
                        STEP_PROFILE_EVENT_BASE
                        + step_index
                        - self.args.profile_step_start
                    )
                step_begin_event = None
                if step_event is not None and self.args.profile_step_frontiers:
                    step_begin_event = (
                        STEP_PROFILE_FRONTIER_BASE
                        + step_index
                        - self.args.profile_step_start
                    )
                queued_stage = queued(
                    stage,
                    profile_after=profile_after,
                    profile_step_event=step_event,
                    profile_step_begin_event=step_begin_event,
                )
                queued_stages.append(queued_stage)
                if step_event is not None:
                    if step_begin_event is not None:
                        self.step_profile_begin_events[step_index] = (
                            step_begin_event
                        )
                    self.step_profile_records.append(
                        (
                            step_index,
                            stage.name,
                            step_event,
                            queued_stage.base_sm,
                            queued_stage.num_sms,
                        )
                    )
            self.step_profile_total = head_step_base + len(self.head_stages)
            return queued_stages

        self.launcher.i(
            SchedDsv4PreloadRopeTables(self.resident_rope_tables).place(
                self.sms
            )
        )
        if self.args.layers == 1:
            family = self.families[0]
            stages = queued_family(family)
            # The looped multi-layer image resets the persistent MXFP rings
            # at every block tail.  Preserve the same direct, FFN-completion-
            # dependent reset in the one-layer diagnostic image so repeated
            # launches do not inherit the previous launch's full/empty phase.
            stages[-1] = replace(
                stages[-1], reset_mxfp_resident_after=True
            )
            stages.extend(queued_head())
            self.program = SequentialProgram(
                self.launcher,
                stages,
                balance_load_ports=True,
            )
            logical_stages = len(stages)
            queue_stages = logical_stages
        elif self.args.layers == 2 and self.args.unroll_two_layers:
            # Diagnostic only: duplicate the same layer-0 command body with
            # independent dependency barriers.  This preserves one resident
            # kernel and identical task placement while removing LOOPC,
            # LOOPM, and the loop-wide dependency-barrier reload.  The MXFP
            # rings still require their direct tail-dependent phase reset.
            family = self.families[0]
            family_serial_sm = serial_sm
            first_stages = queued_family(
                family,
                # Layer profiling needs one frontier for each duplicated
                # body. Other profiling modes intentionally remain attached
                # only to the second body so their event IDs stay unique.
                enable_profile=self.args.profile_layers,
                group_namespace="unroll0.",
            )
            first_stages[-1] = replace(
                first_stages[-1], reset_mxfp_resident_after=True
            )
            serial_sm = family_serial_sm
            second_stages = queued_family(
                family,
                group_namespace="unroll1.",
            )
            stages = first_stages + second_stages
            stages.extend(queued_head())
            self.program = SequentialProgram(
                self.launcher,
                stages,
                balance_load_ports=True,
            )
            logical_stages = len(stages)
            queue_stages = logical_stages
        elif self.args.layers == 2 and len(self.families) == 1:
            family = self.families[0]
            family_stages = queued_family(family)
            if self.args.profile_stages:
                labels = tuple(self.stage_profile_labels)
                self.stage_profile_labels = [
                    f"iteration{iteration}.{label}"
                    for iteration in range(len(self.profile_layer_ids))
                    for label in labels
                ]
            head_stages = queued_head()
            blocks = [
                SequentialBlock(
                    family.name,
                    family_stages,
                    repeat=len(self.profile_layer_ids),
                    barrier_banks=family_barrier_banks,
                    reload_barrier_start=mxfp_reload_start,
                    reload_mxfp_resident=True,
                    elide_terminal_reload=bool(head_stages),
                ),
            ]
            if head_stages:
                blocks.append(
                    SequentialBlock("head", head_stages, reload_after=False)
                )
            blocks = tuple(blocks)
            self.program = LoopedSequentialProgram(
                self.launcher, blocks, balance_load_ports=True
            )
            logical_stages = sum(
                len(block.stages) * block.repeat for block in blocks
            )
            queue_stages = sum(len(block.stages) for block in blocks)
        elif self.args.layers == 2:
            # Diagnostic HCA->CSA (or any heterogeneous adjacent pair): use
            # exactly the same concatenated pair body and loop-tail reload as
            # the production 43-layer block, but execute one pair only.
            first_family, second_family = self.families
            first_stages = queued_family(first_family)
            first_stages[-1] = replace(
                first_stages[-1], reset_mxfp_resident_after=True
            )
            pair_stages = first_stages + queued_family(second_family)
            if (
                self.args.profile_stages
                and self.args.two_layer_pair_repeats > 1
            ):
                labels = tuple(self.stage_profile_labels)
                self.stage_profile_labels = [
                    f"pair{pair_index}.{label}"
                    for pair_index in range(self.args.two_layer_pair_repeats)
                    for label in labels
                ]
            head_stages = queued_head()
            blocks = [
                SequentialBlock(
                    f"{first_family.name}.{second_family.name}",
                    pair_stages,
                    repeat=self.args.two_layer_pair_repeats,
                    barrier_banks=pair_barrier_banks,
                    reload_barrier_start=(
                        None
                        if runtime_config.async_barrier_reload_enabled
                        else mxfp_reload_start
                    ),
                    reload_mxfp_resident=True,
                    elide_terminal_reload=bool(head_stages),
                    async_reload_after=(
                        runtime_config.async_barrier_reload_enabled
                    ),
                    async_reload_worker_base=(
                        132 if self.args.loopback_hc_fusion else 32
                    ),
                ),
            ]
            if head_stages:
                blocks.append(
                    SequentialBlock("head", head_stages, reload_after=False)
                )
            blocks = tuple(blocks)
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
            hca_stages = queued_family(hca)
            hca_stages[-1] = replace(
                hca_stages[-1], reset_mxfp_resident_after=True
            )
            pair_stages = hca_stages + queued_family(csa)
            head_stages = queued_head()
            blocks = [
                SequentialBlock(
                    swa.name,
                    swa_stages,
                    repeat=2,
                    barrier_banks=family_barrier_banks,
                    reload_barrier_start=mxfp_reload_start,
                    reload_mxfp_resident=True,
                ),
                SequentialBlock(
                    layer2.name,
                    layer2_stages,
                    reload_barrier_start=mxfp_reload_start,
                    reload_mxfp_resident=True,
                ),
                SequentialBlock(
                    "layers3-42.hca_csa_score",
                    pair_stages,
                    repeat=20,
                    barrier_banks=pair_barrier_banks,
                    reload_barrier_start=(
                        None
                        if runtime_config.async_barrier_reload_enabled
                        else mxfp_reload_start
                    ),
                    reload_mxfp_resident=True,
                    elide_terminal_reload=bool(head_stages),
                    async_reload_after=(
                        runtime_config.async_barrier_reload_enabled
                    ),
                    async_reload_worker_base=(
                        132 if self.args.loopback_hc_fusion else 32
                    ),
                ),
            ]
            if head_stages:
                blocks.append(
                    SequentialBlock("head", head_stages, reload_after=False)
                )
            blocks = tuple(blocks)
            self.program = LoopedSequentialProgram(
                self.launcher, blocks, balance_load_ports=True
            )
            logical_stages = sum(
                len(block.stages) * block.repeat for block in blocks
            )
            queue_stages = sum(len(block.stages) for block in blocks)
        self.launcher.s(self.program)
        if os.environ.get("DAE_DUMP_COUPLED_PHASES"):
            segments = getattr(self.program, "segments", (self.program,))
            for segment_index, segment in enumerate(segments):
                phases = tuple(segment.coupled_fp8_final_phases)
                counts = {
                    phase: phases.count(phase)
                    for phase in sorted(set(phases))
                }
                print(
                    "DSV4_COUPLED_PHASES "
                    f"segment={segment_index} counts={counts} "
                    f"values={','.join(str(phase) for phase in phases)}",
                    flush=True,
                )
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
            "ffn=mxfp4_mxfp8_routed_resident "
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
        if (
            self.args.profile_layers
            and self.program.profile_event_count
            != len(self.profile_layer_ids) + len(self.head_profile_labels)
        ):
            raise AssertionError(
                "internal layer/head counter does not cover every requested boundary"
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
        if (
            self.args.profile_layers
            or self.args.profile_stages
            or self.args.profile_ffn_aggregate
            or self.args.profile_phase_aggregate
            or self.args.profile_attention_detail
            or self.args.profile_mxfp_ffn_basic
            or self.args.profile_mxfp_ffn_detail
        ):
            self.launcher.profile.zero_()
        if self._l2_scrub is not None:
            self._l2_scrub.add_(1)
        torch.cuda.synchronize(self.device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        self.launcher.launch(synchronize=False)
        end.record()
        end.synchronize()
        if self.compact_head:
            return int(self.output_token[0].item()), start.elapsed_time(end), torch.empty(0)
        logits_cpu = self.logits.cpu()
        logits_fp32 = logits_cpu.float()
        if not bool(torch.isfinite(logits_fp32).all().item()):
            raise AssertionError("one-launch checkpoint logits are not finite")
        token = int(torch.argmax(logits_fp32).item())
        return token, start.elapsed_time(end), logits_fp32

    def device_frontier_ms(self) -> float:
        """Return the last completed kernel's device-only grid envelope."""
        frontiers = self.launcher.profile[:, :2].cpu()
        start = max(int(value) for value in frontiers[:, 0])
        end = max(int(value) for value in frontiers[:, 1])
        if end < start:
            raise RuntimeError("device termination frontier precedes startup")
        return (end - start) / 1.0e6

    def validate_compact_head(
        self, token: int, *, require_reference: bool = True
    ) -> None:
        if not self.compact_head:
            return
        active_head_norm = (
            self.head_norm[0] if self.bf16_umma_head else self.head_norm
        )
        if self.head_norm_oracle is not None and not torch.equal(
            active_head_norm, self.head_norm_oracle
        ):
            mismatch = active_head_norm != self.head_norm_oracle
            first = int(mismatch.nonzero()[0].item())
            delta = (
                active_head_norm.float() - self.head_norm_oracle.float()
            ).abs()
            raise AssertionError(
                "BF16 head input changed between launches: "
                f"mismatches={int(mismatch.count_nonzero().item())} "
                f"max_abs={float(delta.max().item()):.6f} "
                f"first_index={first} "
                f"actual={float(active_head_norm[first].item()):.6f} "
                f"expected={float(self.head_norm_oracle[first].item()):.6f}"
            )
        if self.bf16_umma_head:
            reference_logits = torch.mv(
                self._tensor("head.weight")[: self.config.vocab_size],
                self.head_norm[0],
            )
            reference_token = int(torch.argmax(reference_logits).item())
            if token != reference_token and require_reference:
                raise AssertionError(
                    "BF16 UMMA head selects "
                    f"token {token}, reference BF16 GEMV selects "
                    f"{reference_token}"
                )
            print(
                "DSV4_HEAD_REFERENCE "
                f"status={'PASS' if token == reference_token else 'DIAGNOSTIC'} "
                f"output_token={token} reference_token={reference_token} "
                "format=bf16_umma_group4",
                flush=True,
            )
            return
        if self.fp8_head_activation_oracle is not None and not torch.equal(
            self.head_input_native_fp8, self.fp8_head_activation_oracle
        ):
            mismatch = (
                self.head_input_native_fp8 != self.fp8_head_activation_oracle
            )
            mismatch_indices = mismatch.nonzero()
            first = tuple(int(value) for value in mismatch_indices[0].tolist())
            raise AssertionError(
                "FP8 head packed activation changed between launches: "
                f"mismatches={int(mismatch.count_nonzero().item())} "
                f"first_index={first} "
                f"actual={int(self.head_input_native_fp8[first].item())} "
                f"expected={int(self.fp8_head_activation_oracle[first].item())}"
            )
        resident_logits = self.logits.float()
        if not bool(torch.isfinite(resident_logits).all().item()):
            raise AssertionError("FP8 head logits are not finite")
        resident_token = int(torch.argmax(resident_logits).item())
        if token != resident_token:
            raise AssertionError(
                "FP8 head argmax emitted "
                f"token {token}, resident logits select {resident_token}"
            )
        reference_logits = torch.mv(
            self._tensor("head.weight")[: self.config.vocab_size],
            self.head_norm,
        )
        reference_token = int(torch.argmax(reference_logits).item())
        if resident_token != reference_token and require_reference:
            raise AssertionError(
                "FP8 head logits select "
                f"token {resident_token}, reference BF16 GEMV selects "
                f"{reference_token}"
            )
        print(
            "DSV4_HEAD_REFERENCE "
            f"status={'PASS' if resident_token == reference_token else 'DIAGNOSTIC'} "
            f"output_token={token} reference_token={reference_token}",
            flush=True,
        )

    def capture_repeat_state(self) -> dict[str, torch.Tensor]:
        names = (
            "mhc_packed_output",
            "hidden",
            "hidden_native_fp8",
            "q_rank",
            "q_rank_norm",
            "q_rank_native_fp8",
            "q",
            "q_norm",
            "q_rope",
            "kv",
            "kv_norm",
            "attention_output",
            "attention_inverse",
            "o_group_native_fp8",
            "o_rank",
            "o_rank_native_fp8",
            "branch",
            "router_prepared",
            "route_record",
            "mxfp_input_records",
            "mxfp_middle_records",
            "mxfp_ffn_output",
            "next_residual",
            "residual",
            "head_mixes",
            "head_norm",
            "head_input_native_fp8",
            "logits",
            "output_token",
        )
        return {
            name: getattr(self, name).clone()
            for name in names
            if isinstance(getattr(self, name, None), torch.Tensor)
        }

    def report_repeat_state(
        self, oracle: dict[str, torch.Tensor], iteration: int
    ) -> None:
        for name, expected in oracle.items():
            actual = getattr(self, name)
            if torch.equal(actual, expected):
                continue
            mismatch = actual != expected
            count = int(mismatch.count_nonzero().item())
            if actual.is_floating_point():
                max_abs = float(
                    (actual.float() - expected.float()).abs().max().item()
                )
            else:
                max_abs = -1.0
            print(
                "DSV4_REPEAT_STATE "
                f"iteration={iteration} name={name} exact=false "
                f"mismatches={count} max_abs={max_abs:.6f}",
                flush=True,
            )

    def report_projection_diagnostics(self) -> None:
        """Compare resident Q_b output with its raw-checkpoint FP8 oracle."""
        # The resident buffers contain the final executed layer after a
        # multi-layer launch, so diagnose them against that layer's weights.
        layer_id = self.families[-1].layer_ids[-1]
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
        profile_layer_ids = self.profile_layer_ids
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
        elif self.args.layers == 2 and len(self.families) == 1:
            reload_after_layers = self.profile_layer_ids
        elif self.args.layers == 2:
            reload_after_layers = tuple(self.profile_layer_ids[1::2])
        else:
            reload_after_layers = (0, 1, 2, *range(4, self.args.layers, 2))
        reload_durations = []
        reload_spreads = []
        reload_slowest_vcores = []
        internal_span = end_frontier - start_frontier
        for reload_index, layer_id in enumerate(reload_after_layers):
            event_id = runtime_config.reload_profile_event_base + reload_index
            values = [int(value) for value in profile[:, event_id]]
            # Reload timing is diagnostic-only and some balanced LDU streams
            # do not place the profiling form on port 0. Treat a missing or
            # unsubtracted globaltimer value as unavailable instead of
            # corrupting the otherwise valid layer-frontier report.
            if (
                any(value == 0 for value in values)
                or max(values) > internal_span
            ):
                reload_durations.append(None)
                reload_spreads.append(None)
                reload_slowest_vcores.append(None)
            else:
                reload_durations.append(max(values))
                reload_spreads.append(max(values) - min(values))
                reload_slowest_vcores.append(
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
                f"frontier_us={(boundary - start_frontier) / 1.0e3:.3f} "
                f"frontier_spread_us={spread / 1.0e3:.3f} "
                f"frontier_vcore={frontier_vcore} "
                f"frontier_physical_sm={physical_sm_ids[frontier_vcore]}",
                flush=True,
            )
            if layer_id in reload_after_layers:
                reload_elapsed = reload_durations[reload_index]
                if reload_elapsed is None:
                    print(
                        "DSV4_RELOAD_SERVICE "
                        f"after_layer={layer_id} status=UNAVAILABLE",
                        flush=True,
                    )
                else:
                    reload_total += reload_elapsed
                    slowest_vcore = reload_slowest_vcores[reload_index]
                    assert slowest_vcore is not None
                    print(
                        "DSV4_RELOAD_SERVICE "
                        f"after_layer={layer_id} "
                        f"barriers={'pair' if layer_id >= 4 else 'family'} "
                        f"elapsed_ms={reload_elapsed / 1.0e6:.6f} "
                        f"frontier_spread_us={reload_spreads[reload_index] / 1.0e3:.3f} "
                        f"slowest_vcore={slowest_vcore} "
                        "frontier_physical_sm="
                        f"{physical_sm_ids[slowest_vcore]}",
                        flush=True,
                    )
                reload_index += 1
        head_start = previous
        head_event_base = (
            runtime_config.layer_profile_event_base + len(profile_layer_ids)
        )
        for index, label in enumerate(self.head_profile_labels):
            values = [
                int(value) for value in profile[:, head_event_base + index]
            ]
            if any(value == 0 for value in values):
                raise RuntimeError(
                    f"head profile boundary {label!r} was not recorded"
                )
            boundary = max(values)
            elapsed = boundary - previous
            if elapsed < 0:
                raise RuntimeError("head profile frontiers are not monotonic")
            print(
                "DSV4_HEAD_STAGE_TIME "
                f"stage={label} elapsed_ms={elapsed / 1.0e6:.6f} "
                f"frontier_spread_us={(max(values) - min(values)) / 1.0e3:.3f}",
                flush=True,
            )
            previous = boundary
        if self.head_stages:
            end_values = [int(value) for value in profile[:, 1]]
            final_head_elapsed = end_frontier - previous
            if final_head_elapsed < 0:
                raise RuntimeError("head termination frontier is not monotonic")
            print(
                "DSV4_HEAD_STAGE_TIME "
                f"stage={self.head_stages[-1].name} "
                f"elapsed_ms={final_head_elapsed / 1.0e6:.6f} "
                f"frontier_spread_us={(max(end_values) - min(end_values)) / 1.0e3:.3f}",
                flush=True,
            )
        head_elapsed = end_frontier - head_start
        if head_elapsed < 0:
            raise RuntimeError("head profile frontier precedes the final layer")
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
        physical_sm_ids = [
            int(profile[vcore, counter_base + 25])
            for vcore in range(profile.shape[0])
        ]
        placement_signature_vcores = (0, 1, 2, 3, 16, 31, 101, 116)
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
            f"reload_service_sum_ms={reload_total / 1.0e6:.6f} "
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
        step_end_frontiers = {}
        step_end_by_sm = {}
        if profile is None:
            profile = self.launcher.profile.cpu()
        magic = 0x4454524B50524631
        if any(int(value) != magic for value in profile[:, 127]):
            raise RuntimeError(
                "step profiling requires a runtime built with track_profile=1"
            )

        summed_local_elapsed_ns = 0
        summed_wait_ns = 0
        counter_base = runtime_config.track_profile_event_base
        sm_clock_ghz = []
        for vcore in range(profile.shape[0]):
            elapsed_ns = int(profile[vcore, 1]) - int(profile[vcore, 0])
            elapsed_cycles = int(profile[vcore, counter_base + 27]) - int(
                profile[vcore, counter_base + 26]
            )
            if elapsed_ns > 0:
                sm_clock_ghz.append(elapsed_cycles / elapsed_ns)
        physical_sm_ids = [
            int(profile[vcore, counter_base + 25])
            for vcore in range(profile.shape[0])
        ]
        grid_start = max(int(value) for value in profile[:, 0])
        if self.args.profile_loopback_boundary:
            # The last recorded HCA step belongs to layer 41.  Its preceding
            # loop iteration completed at CSA layer 40, the nineteenth of the
            # twenty repeated CSA boundaries recorded below.
            boundary_event = (
                runtime_config.layer_profile_event_base
                + LOOPBACK_PROFILE_CSA_COMPLETIONS
                - 2
            )
            boundary_values = [
                int(value) for value in profile[:, boundary_event]
            ]
            if any(value == 0 for value in boundary_values):
                raise RuntimeError(
                    "loopback boundary profile did not record CSA layer 40"
                )
            boundary = max(boundary_values)
            print(
                "DSV4_LOOPBACK_BOUNDARY "
                "previous_layer=40 next_layer=41 "
                f"frontier_us={(max(boundary_values) - grid_start) / 1.0e3:.3f} "
                f"frontier_spread_us={(max(boundary_values) - min(boundary_values)) / 1.0e3:.3f} "
                f"sample_index={sample_index if sample_index is not None else -1}",
                flush=True,
            )
            operand_frontiers = []
            operand_spreads = []
            for event in range(25, 30):
                values = [int(profile[sm, event]) for sm in range(128)]
                if any(value == 0 for value in values):
                    raise RuntimeError(
                        f"loopback operand event {event} was not recorded"
                    )
                operand_frontiers.append((max(values) - boundary) / 1.0e3)
                operand_spreads.append((max(values) - min(values)) / 1.0e3)
            print(
                "DSV4_LOOPBACK_OPERANDS "
                f"enter_us={operand_frontiers[0]:.3f} "
                f"weight_us={operand_frontiers[1]:.3f} "
                f"coefficients_us={operand_frontiers[2]:.3f} "
                f"record_us={operand_frontiers[3]:.3f} "
                f"all_inputs_us={operand_frontiers[4]:.3f} "
                f"record_spread_us={operand_spreads[3]:.3f} "
                f"sample_index={sample_index if sample_index is not None else -1}",
                flush=True,
            )
        placement_signature_vcores = (0, 1, 2, 3, 16, 31, 101, 116)
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
            frontier_summary = ""
            if self.args.profile_step_frontiers:
                begin_event = self.step_profile_begin_events[step_index]
                elapsed_by_sm = {
                    sm: (sample_elapsed_ns, sample_wait_ns)
                    for sm, sample_elapsed_ns, sample_wait_ns in samples
                }
                begin_samples = [
                    (sm, int(profile[sm, begin_event]))
                    for sm in elapsed_by_sm
                ]
                if any(timestamp == 0 for _, timestamp in begin_samples):
                    raise RuntimeError(
                        f"step begin event {begin_event} for {name!r} was not recorded"
                    )
                begin_values = [timestamp for _, timestamp in begin_samples]
                ready_values = [
                    timestamp + elapsed_by_sm[sm][1]
                    for sm, timestamp in begin_samples
                ]
                end_values = [
                    timestamp + elapsed_by_sm[sm][0]
                    for sm, timestamp in begin_samples
                ]
                latest_end_sm, latest_end = max(
                    (
                        (sm, timestamp + elapsed_by_sm[sm][0])
                        for sm, timestamp in begin_samples
                    ),
                    key=lambda sample: sample[1],
                )
                step_end_frontiers[step_index] = latest_end
                step_end_by_sm[step_index] = {
                    sm: timestamp + elapsed_by_sm[sm][0]
                    for sm, timestamp in begin_samples
                }
                frontier_summary = (
                    f" begin_min_us={(min(begin_values) - grid_start) / 1.0e3:.3f}"
                    f" begin_max_us={(max(begin_values) - grid_start) / 1.0e3:.3f}"
                    f" begin_spread_us={(max(begin_values) - min(begin_values)) / 1.0e3:.3f}"
                    f" ready_frontier_us={(max(ready_values) - grid_start) / 1.0e3:.3f}"
                    f" end_frontier_us={(latest_end - grid_start) / 1.0e3:.3f}"
                    f" end_spread_us={(max(end_values) - min(end_values)) / 1.0e3:.3f}"
                    f" latest_end_sm={latest_end_sm}"
                    f" latest_end_physical_sm={physical_sm_ids[latest_end_sm]}"
                )
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
                f"m2c_wait_pct={100.0 * wait_ns / elapsed_ns if elapsed_ns else 0.0:.3f}"
                f"{frontier_summary}",
                flush=True,
            )
            if (
                self.args.profile_loopback_boundary
                and name == "ffn.hc_post_next_attn.hc_project"
                and self.args.profile_step_frontiers
            ):
                operand_events = tuple(
                    [int(profile[sm, event]) for event in range(25, 30)]
                    for sm in elapsed_by_sm
                )
                operand_by_sm = {
                    sm: values
                    for sm, values in zip(elapsed_by_sm, operand_events)
                }
                latest = sorted(
                    (
                        (
                            timestamp + elapsed_by_sm[sm][0],
                            sm,
                            timestamp,
                            elapsed_by_sm[sm][1],
                            elapsed_by_sm[sm][0],
                        )
                        for sm, timestamp in begin_samples
                    ),
                    reverse=True,
                )[:12]
                print(
                    "DSV4_LOOPBACK_PROJECTION_TAIL "
                    "end_rank="
                    + ",".join(
                        f"{sm}:{physical_sm_ids[sm]}:"
                        f"{(begin - boundary) / 1.0e3:.3f}:"
                        f"{(operand_by_sm[sm][1] - boundary) / 1.0e3:.3f}:"
                        f"{(operand_by_sm[sm][3] - boundary) / 1.0e3:.3f}:"
                        f"{wait_ns / 1.0e3:.3f}:"
                        f"{(elapsed_ns - wait_ns) / 1.0e3:.3f}:"
                        f"{(end - boundary) / 1.0e3:.3f}"
                        for end, sm, begin, wait_ns, elapsed_ns in latest
                    )
                    + " fields=vcore:physical_sm:begin_us:weight_us:record_us:"
                    "m2c_wait_us:active_us:end_us"
                    f" sample_index={sample_index if sample_index is not None else -1}",
                    flush=True,
                )
        resident_step_indices = [
            step_index
            for step_index, name, _, _, _ in self.step_profile_records
            if name == "ffn.mx.resident"
        ]
        if resident_step_indices:
            if len(resident_step_indices) != 1:
                raise RuntimeError("resident FFN step profiling is ambiguous")
            resident_step_index = resident_step_indices[0]
            resident_base = self.sms - 112
            output_profile = profile[
                resident_base : resident_base + 112,
                FFN_OUTPUT_PROFILE_EVENT_BASE :
                FFN_OUTPUT_PROFILE_EVENT_BASE + 3,
            ]
            if bool((output_profile == 0).any().item()):
                raise RuntimeError("resident FFN output publication was not recorded")
            allocation = [int(value) for value in output_profile[:, 0]]
            dequeue = [int(value) for value in output_profile[:, 1]]
            publication = [int(value) for value in output_profile[:, 2]]
            compute_end = step_end_frontiers[resident_step_index]
            service = [
                end - begin for begin, end in zip(dequeue, publication)
            ]
            print(
                "DSV4_FFN_OUTPUT_PUBLICATION "
                f"allocation_frontier_us={(max(allocation) - grid_start) / 1.0e3:.3f} "
                f"dequeue_frontier_us={(max(dequeue) - grid_start) / 1.0e3:.3f} "
                f"publication_frontier_us={(max(publication) - grid_start) / 1.0e3:.3f} "
                f"compute_end_to_dequeue_us={(max(dequeue) - compute_end) / 1.0e3:.3f} "
                f"compute_end_to_publication_us={(max(publication) - compute_end) / 1.0e3:.3f} "
                f"stu_service_median_us={statistics.median(service) / 1.0e3:.3f} "
                f"stu_service_max_us={max(service) / 1.0e3:.3f}",
                flush=True,
            )
            history_profile = profile[
                resident_base : resident_base + 112,
                STU_HISTORY_EVENT_BASE :
                STU_HISTORY_EVENT_BASE + 3 * STU_HISTORY_COMMANDS + 1,
            ]
            history_counts = [
                int(value)
                for value in history_profile[:, 3 * STU_HISTORY_COMMANDS]
            ]
            if any(history_counts):
                critical_local = max(
                    range(len(dequeue)), key=dequeue.__getitem__
                )
                critical_sm = resident_base + critical_local
                critical_compute_end = step_end_by_sm[resident_step_index][critical_sm]
                pop_begin = [
                    int(value)
                    for value in profile[
                        resident_base : resident_base + 112,
                        STU_RAW_POP_BEGIN_EVENT,
                    ]
                ]
                service_identity = profile[
                    resident_base : resident_base + 112,
                    STU_RAW_SERVICE_IDENTITY_EVENT,
                ]
                output_tokens = profile[
                    resident_base : resident_base + 112,
                    STU_RAW_OUTPUT_TOKEN_EVENT,
                ]
                raw_ptr_matches = profile[
                    resident_base : resident_base + 112,
                    STU_RAW_PTR_MATCH_EVENT_BASE :
                    STU_RAW_PTR_MATCH_EVENT_BASE + 4,
                ]
                raw_ptrs = profile[
                    resident_base : resident_base + 112,
                    STU_RAW_PTR_EVENT_BASE :
                    STU_RAW_PTR_EVENT_BASE + 4,
                ]
                raw_arrivals = profile[
                    resident_base : resident_base + 112,
                    STU_RAW_ARRIVAL_EVENT_BASE :
                    STU_RAW_ARRIVAL_EVENT_BASE + 4,
                ]
                raw_posts = profile[
                    resident_base : resident_base + 112,
                    STU_RAW_POST_EVENT_BASE :
                    STU_RAW_POST_EVENT_BASE + 4,
                ]
                pointer_matches = [
                    len({int(value) for value in row}) == 1
                    for row in raw_ptrs
                ]
                lane_pointer_matches = [
                    all(int(value) == 0xFFFFFFFF for value in row)
                    for row in raw_ptr_matches
                ]
                service_identity_matches = [
                    (int(identity) & 0xFFFFFFFF) == int(ptr_row[0])
                    and (int(identity) >> 32) == int(output_token)
                    for identity, output_token, ptr_row in zip(
                        service_identity, output_tokens, raw_ptrs
                    )
                ]
                arrival_spreads = [
                    max(int(value) for value in row) -
                    min(int(value) for value in row)
                    for row in raw_arrivals
                ]
                arrival_to_dequeue = [
                    target_dequeue - max(int(value) for value in row)
                    for target_dequeue, row in zip(dequeue, raw_arrivals)
                ]
                post_to_dequeue = [
                    target_dequeue - max(int(value) for value in row)
                    for target_dequeue, row in zip(dequeue, raw_posts)
                ]
                pop_wait = [
                    target_dequeue - begin
                    for target_dequeue, begin in zip(dequeue, pop_begin)
                ]
                critical_ptrs = ",".join(
                    str(int(value)) for value in raw_ptrs[critical_local]
                )
                critical_ptr_masks = ",".join(
                    f"0x{int(value):08x}"
                    for value in raw_ptr_matches[critical_local]
                )
                critical_identity = int(service_identity[critical_local])
                critical_arrivals = [
                    int(value) for value in raw_arrivals[critical_local]
                ]
                critical_posts = [
                    int(value) for value in raw_posts[critical_local]
                ]
                print(
                    "DSV4_FFN_OUTPUT_C2M_ARRIVAL "
                    f"pointer_match_sms={sum(pointer_matches)}/{len(pointer_matches)} "
                    f"lane_pointer_match_sms="
                    f"{sum(lane_pointer_matches)}/{len(lane_pointer_matches)} "
                    f"service_identity_match_sms="
                    f"{sum(service_identity_matches)}/{len(service_identity_matches)} "
                    f"arrival_spread_median_us="
                    f"{statistics.median(arrival_spreads) / 1.0e3:.3f} "
                    f"arrival_spread_max_us={max(arrival_spreads) / 1.0e3:.3f} "
                    f"last_arrival_to_dequeue_median_us="
                    f"{statistics.median(arrival_to_dequeue) / 1.0e3:.3f} "
                    f"last_arrival_to_dequeue_max_us="
                    f"{max(arrival_to_dequeue) / 1.0e3:.3f} "
                    f"last_post_to_dequeue_median_us="
                    f"{statistics.median(post_to_dequeue) / 1.0e3:.3f} "
                    f"last_post_to_dequeue_max_us="
                    f"{max(post_to_dequeue) / 1.0e3:.3f} "
                    f"pop_wait_median_us="
                    f"{statistics.median(pop_wait) / 1.0e3:.3f} "
                    f"pop_wait_max_us={max(pop_wait) / 1.0e3:.3f} "
                    f"critical_sm={critical_sm} "
                    f"critical_ptrs={critical_ptrs} "
                    f"critical_ptr_masks={critical_ptr_masks} "
                    f"critical_output_token={int(output_tokens[critical_local])} "
                    f"critical_service_slot={critical_identity >> 32} "
                    f"critical_service_queue={critical_identity & 0xFFFFFFFF} "
                    f"critical_pop_begin_from_compute_end_us="
                    f"{(pop_begin[critical_local] - critical_compute_end) / 1.0e3:.3f} "
                    f"critical_arrival_spread_us="
                    f"{(max(critical_arrivals) - min(critical_arrivals)) / 1.0e3:.3f} "
                    f"critical_last_arrival_from_compute_end_us="
                    f"{(max(critical_arrivals) - critical_compute_end) / 1.0e3:.3f} "
                    f"critical_last_arrival_to_dequeue_us="
                    f"{(dequeue[critical_local] - max(critical_arrivals)) / 1.0e3:.3f} "
                    f"critical_last_post_to_dequeue_us="
                    f"{(dequeue[critical_local] - max(critical_posts)) / 1.0e3:.3f}",
                    flush=True,
                )
                opcode_names = {}
                for name in dir(runtime.opcode):
                    if not name.startswith("OP_"):
                        continue
                    value = getattr(runtime.opcode, name)
                    if isinstance(value, int):
                        opcode_names.setdefault(int(value) >> 6, name)
                for distance in range(STU_HISTORY_COMMANDS, 0, -1):
                    samples = []
                    for local_sm, count in enumerate(history_counts):
                        if count < distance:
                            continue
                        event = 3 * (count - distance)
                        samples.append(
                            (
                                local_sm,
                                int(history_profile[local_sm, event + 0]),
                                int(history_profile[local_sm, event + 1]),
                                int(history_profile[local_sm, event + 2]),
                            )
                        )
                    if not samples:
                        continue
                    opcode_values = [sample[1] for sample in samples]
                    service_begin = [sample[2] for sample in samples]
                    service_end = [sample[3] for sample in samples]
                    durations = [
                        end - begin
                        for begin, end in zip(service_begin, service_end)
                    ]
                    opcode_counts = {}
                    for value in opcode_values:
                        name = opcode_names.get(value, f"mop_{value}")
                        opcode_counts[name] = opcode_counts.get(name, 0) + 1
                    critical_sample = next(
                        (
                            sample
                            for sample in samples
                            if sample[0] == critical_local
                        ),
                        None,
                    )
                    critical_summary = ""
                    if critical_sample is not None:
                        critical_begin = critical_sample[2]
                        critical_end = critical_sample[3]
                        critical_summary = (
                            f" critical_begin_from_compute_end_us="
                            f"{(critical_begin - critical_compute_end) / 1.0e3:.3f}"
                            f" critical_end_from_compute_end_us="
                            f"{(critical_end - critical_compute_end) / 1.0e3:.3f}"
                            f" critical_service_us="
                            f"{(critical_end - critical_begin) / 1.0e3:.3f}"
                        )
                    print(
                        "DSV4_FFN_OUTPUT_PRECEDING "
                        f"rank={-distance} samples={len(samples)} "
                        "opcodes="
                        + ",".join(
                            f"{name}:{count}"
                            for name, count in sorted(opcode_counts.items())
                        )
                        + " "
                        f"service_median_us={statistics.median(durations) / 1.0e3:.3f} "
                        f"service_max_us={max(durations) / 1.0e3:.3f} "
                        f"critical_sm={critical_sm}"
                        f"{critical_summary}",
                        flush=True,
                    )
        print(
            "DSV4_STEP_PROFILE_SUMMARY "
            f"profiled_layer="
            f"{self.layer_ids[-1]} "
            f"window_start={self.args.profile_step_start} "
            f"window_steps={len(self.step_profile_records)} "
            f"layer_steps={self.step_profile_total} "
            f"summed_local_elapsed_us={summed_local_elapsed_ns / 1.0e3:.3f} "
            f"summed_local_m2c_wait_us={summed_wait_ns / 1.0e3:.3f} "
            f"sm_clock_ghz_min={min(sm_clock_ghz):.3f} "
            f"sm_clock_ghz_median={statistics.median(sm_clock_ghz):.3f} "
            f"sm_clock_ghz_max={max(sm_clock_ghz):.3f} "
            "physical_sm_signature="
            + ",".join(
                str(physical_sm_ids[vcore])
                for vcore in placement_signature_vcores
            )
            + " "
            f"sample_index={sample_index if sample_index is not None else -1} "
            f"sample_cuda_ms={sample_cuda_ms if sample_cuda_ms is not None else -1.0:.6f}",
            flush=True,
        )
    def report_fp8_coupled_detail_profile(
        self,
        profile: torch.Tensor | None = None,
        *,
        sample_index: int | None = None,
        sample_cuda_ms: float | None = None,
    ) -> None:
        """Decompose the selected layer's Q-a coupled-ring service."""
        if not self.args.profile_fp8_coupled_detail:
            return
        if profile is None:
            profile = self.launcher.profile.cpu()
        if runtime_config.num_profile_events < 160:
            raise RuntimeError(
                "FP8 coupled detail profiling requires "
                "fp8_coupled_detail_profile=1"
            )
        magic = 0x4454524B50524631
        if any(int(value) != magic for value in profile[:, 127]):
            raise RuntimeError(
                "FP8 coupled detail profiling requires track_profile=1"
            )

        q_record = next(
            record
            for record in self.step_profile_records
            if record[1] == "attn.q_b"
        )
        _, _, step_event, base_sm, num_sms = q_record
        ldu_base = runtime_config.detail_profile_event_base
        commands = 8
        coupled_load_opcode = (
            (int(runtime.opcode.OP_TMA_LOAD_MX_COUPLED_STREAM) >> 6)
            | 0x800
        )
        opcode_names = {}
        for name in dir(runtime.opcode):
            if not name.startswith("OP_"):
                continue
            value = getattr(runtime.opcode, name)
            if isinstance(value, int):
                opcode_names.setdefault(int(value) >> 6, name)

        def signed_wrapped_delta(end: int, begin: int) -> int:
            return ((end - begin + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)

        records = []
        trace = {}
        for sm in range(base_sm, base_sm + num_sms):
            ldu = []
            for port in range(2):
                port_records = []
                for command in range(commands):
                    packed = int(
                        profile[
                            sm,
                            ldu_base + port * commands + command,
                        ].item()
                    )
                    if packed == 0:
                        break
                    begin = packed & 0xFFFFFFFF
                    duration = (packed >> 32) & 0xFFFFF
                    normalized_opcode = (packed >> 52) & 0xFFF
                    port_records.append(
                        (begin, duration, normalized_opcode)
                    )
                q_slots = [
                    index
                    for index, item in enumerate(port_records)
                    if item[2] == coupled_load_opcode
                ]
                if not port_records or not q_slots:
                    raise RuntimeError(
                        f"LDU{port} rolling prefix for SM {sm} has "
                        f"{len(port_records)} records and Q-a slots {q_slots}; "
                        f"opcodes={[item[2] for item in port_records]}"
                    )
                q_slot = q_slots[0]
                # The producer stops at the first Q-a command. Slots beyond
                # it may contain an older sample because the diagnostic path
                # deliberately avoids clearing global memory in the LDU.
                port_records = port_records[: q_slot + 1]
                ldu.append(port_records)

            step_packed = int(profile[sm, step_event])
            step_elapsed = step_packed & 0xFFFFFFFF
            step_wait = (step_packed >> 32) & 0xFFFFFFFF
            step_begin = int(
                profile[sm, FP8_COUPLED_STEP_BEGIN_EVENT]
            ) & 0xFFFFFFFF
            if step_begin == 0:
                raise RuntimeError(
                    f"missing Q-a begin timestamp for SM {sm}"
                )
            q_commands = []
            q_gaps = []
            for port in range(2):
                q_index = next(
                    (
                        index
                        for index, item in enumerate(ldu[port])
                        if item[2] == coupled_load_opcode
                    ),
                    None,
                )
                if q_index is None:
                    raise RuntimeError(
                        f"LDU{port} Q-a command lies beyond the "
                        f"{commands}-command trace on SM {sm}"
                    )
                q_commands.append(ldu[port][q_index])
                previous_end = (
                    ldu[port][q_index - 1][0]
                    + ldu[port][q_index - 1][1]
                    if q_index > 0
                    else ldu[port][q_index][0]
                ) & 0xFFFFFFFF
                q_gaps.append(
                    signed_wrapped_delta(ldu[port][q_index][0], previous_end)
                )
                for command, (begin, duration, normalized_opcode) in enumerate(
                    ldu[port][: q_index + 1]
                ):
                    key = (
                        port,
                        command - q_index,
                        normalized_opcode,
                    )
                    trace.setdefault(key, []).append(
                        (
                            signed_wrapped_delta(begin, step_begin),
                            duration,
                        )
                    )
            records.append(
                {
                    "sm": sm,
                    "step": step_elapsed,
                    "step_wait": step_wait,
                    "step_active": max(0, step_elapsed - step_wait),
                    "ldu0_begin": signed_wrapped_delta(
                        q_commands[0][0], step_begin
                    ),
                    "ldu1_begin": signed_wrapped_delta(
                        q_commands[1][0], step_begin
                    ),
                    "ldu0_begin_raw": q_commands[0][0],
                    "ldu1_begin_raw": q_commands[1][0],
                    "ldu0_service": q_commands[0][1],
                    "ldu1_service": q_commands[1][1],
                    "ldu0_gap": q_gaps[0],
                    "ldu1_gap": q_gaps[1],
                }
            )
            records[-1]["ldu0_end"] = (
                records[-1]["ldu0_begin"]
                + records[-1]["ldu0_service"]
            )
            records[-1]["ldu1_end"] = (
                records[-1]["ldu1_begin"]
                + records[-1]["ldu1_service"]
            )
            records[-1]["post_issue_tail"] = (
                records[-1]["step"]
                - max(records[-1]["ldu0_end"], records[-1]["ldu1_end"])
            )

            wait_base = ldu_base + 2 * commands
            source_base = wait_base + 6
            for port in range(2):
                source = int(profile[sm, source_base + port].item())
                records[-1][f"ldu{port}_source_begin"] = (
                    signed_wrapped_delta(source & 0xFFFFFFFF, step_begin)
                )
                records[-1][f"ldu{port}_source_begin_raw"] = (
                    source & 0xFFFFFFFF
                )
                records[-1][f"ldu{port}_gate_wait"] = (
                    records[-1][f"ldu{port}_source_begin"]
                    - records[-1][f"ldu{port}_begin"]
                )
                records[-1][f"ldu{port}_source_wait"] = (
                    source >> 32
                ) & 0xFFFFF
                state = int(profile[sm, wait_base + port * 3].item())
                records[-1][f"ldu{port}_phase_base"] = state & 0xFFFFFFFF
                records[-1][f"ldu{port}_pair_count"] = state >> 32
                for pair in range(2):
                    packed = int(
                        profile[
                            sm,
                            wait_base + port * 3 + 1 + pair,
                        ].item()
                    )
                    records[-1][f"ldu{port}_pair{pair}_begin"] = (
                        signed_wrapped_delta(packed & 0xFFFFFFFF, step_begin)
                    )
                    records[-1][f"ldu{port}_pair{pair}_wait"] = (
                        packed >> 32
                    ) & 0xFFFFF
                    records[-1][f"ldu{port}_pair{pair}_expected_ready"] = (
                        packed >> 52
                    ) & 1
                    records[-1][f"ldu{port}_pair{pair}_opposite_ready"] = (
                        packed >> 53
                    ) & 1
                    records[-1][f"ldu{port}_pair{pair}_stage"] = (
                        packed >> 54
                    ) & 1
                    records[-1][f"ldu{port}_pair{pair}_phase"] = (
                        packed >> 55
                    ) & 1

        def values(name: str) -> list[int]:
            return [int(record[name]) for record in records]

        # This diagnostic is temporarily armed on the K1024 Q-b stream.  It
        # intentionally bypasses the older Q-a/hidden-quant handoff report,
        # whose producer mapping does not describe Q-b.
        critical = max(records, key=lambda record: record["step"])
        critical_tail = sorted(
            records,
            key=lambda record: record["step_active"],
            reverse=True,
        )[:12]
        print(
            "DSV4_FP8_COUPLED_DETAIL "
            "stage=attn.q_b "
            f"active_sms={len(records)} "
            f"step_us={max(values('step')) / 1.0e3:.3f} "
            f"step_median_us={statistics.median(values('step')) / 1.0e3:.3f} "
            f"m2c_wait_us={critical['step_wait'] / 1.0e3:.3f} "
            f"compute_active_us={critical['step_active'] / 1.0e3:.3f} "
            f"compute_active_median_us="
            f"{statistics.median(values('step_active')) / 1.0e3:.3f} "
            f"critical_sm={critical['sm']} "
            f"critical_ldu0_begin_us={critical['ldu0_begin'] / 1.0e3:.3f} "
            f"critical_ldu1_begin_us={critical['ldu1_begin'] / 1.0e3:.3f} "
            f"critical_ldu0_service_us={critical['ldu0_service'] / 1.0e3:.3f} "
            f"critical_ldu1_service_us={critical['ldu1_service'] / 1.0e3:.3f} "
            f"critical_post_issue_tail_us="
            f"{critical['post_issue_tail'] / 1.0e3:.3f} "
            "top_sm:active_us:ldu0_begin_us:ldu0_service_us:"
            "ldu1_begin_us:ldu1_service_us:post_issue_us="
            + ",".join(
                f"{record['sm']}:{record['step_active'] / 1.0e3:.3f}:"
                f"{record['ldu0_begin'] / 1.0e3:.3f}:"
                f"{record['ldu0_service'] / 1.0e3:.3f}:"
                f"{record['ldu1_begin'] / 1.0e3:.3f}:"
                f"{record['ldu1_service'] / 1.0e3:.3f}:"
                f"{record['post_issue_tail'] / 1.0e3:.3f}"
                for record in critical_tail
            ),
            flush=True,
        )
        for port in range(2):
            print(
                "DSV4_FP8_COUPLED_SOURCE_LOAD "
                f"port={port} "
                f"command_begin_min_us={min(values(f'ldu{port}_begin')) / 1.0e3:.3f} "
                f"command_begin_median_us="
                f"{statistics.median(values(f'ldu{port}_begin')) / 1.0e3:.3f} "
                f"command_begin_max_us={max(values(f'ldu{port}_begin')) / 1.0e3:.3f} "
                f"gate_wait_median_us="
                f"{statistics.median(values(f'ldu{port}_gate_wait')) / 1.0e3:.3f} "
                f"source_wait_median_us="
                f"{statistics.median(values(f'ldu{port}_source_wait')) / 1.0e3:.3f} "
                f"service_median_us="
                f"{statistics.median(values(f'ldu{port}_service')) / 1.0e3:.3f}",
                flush=True,
            )
        for band_start in range(0, len(records), 32):
            band = records[band_start : band_start + 32]
            print(
                "DSV4_FP8_COUPLED_BAND "
                f"start_sm={band_start} stop_sm={band_start + len(band)} "
                f"active_median_us="
                f"{statistics.median(record['step_active'] for record in band) / 1.0e3:.3f} "
                f"active_max_us="
                f"{max(record['step_active'] for record in band) / 1.0e3:.3f} "
                f"ldu0_service_median_us="
                f"{statistics.median(record['ldu0_service'] for record in band) / 1.0e3:.3f} "
                f"ldu1_service_median_us="
                f"{statistics.median(record['ldu1_service'] for record in band) / 1.0e3:.3f} "
                f"ldu0_pair0_wait_median_us="
                f"{statistics.median(record['ldu0_pair0_wait'] for record in band) / 1.0e3:.3f} "
                f"ldu0_pair1_wait_median_us="
                f"{statistics.median(record['ldu0_pair1_wait'] for record in band) / 1.0e3:.3f} "
                f"ldu1_pair0_wait_median_us="
                f"{statistics.median(record['ldu1_pair0_wait'] for record in band) / 1.0e3:.3f} "
                f"ldu1_pair1_wait_median_us="
                f"{statistics.median(record['ldu1_pair1_wait'] for record in band) / 1.0e3:.3f} "
                f"phase0={sorted(set(record['ldu0_phase_base'] for record in band))} "
                f"phase1={sorted(set(record['ldu1_phase_base'] for record in band))} "
                f"post_issue_median_us="
                f"{statistics.median(record['post_issue_tail'] for record in band) / 1.0e3:.3f}",
                flush=True,
            )
        return

        quant_store_event = ldu_base + 24
        quant_sms = self.config.hidden_size // 256
        quant_store_raw = [
            int(profile[sm, quant_store_event].item()) & 0xFFFFFFFF
            for sm in range(quant_sms)
        ]
        if any(timestamp == 0 for timestamp in quant_store_raw):
            raise RuntimeError("missing hidden-quant STU completion timestamp")
        quant_anchor = quant_store_raw[0]
        quant_store_offsets = [
            signed_wrapped_delta(timestamp, quant_anchor)
            for timestamp in quant_store_raw
        ]
        layer_begin_raw = [
            int(profile[sm, FP8_COUPLED_LAYER_BEGIN_EVENT].item())
            & 0xFFFFFFFF
            for sm in range(self.sms)
        ]
        if any(timestamp == 0 for timestamp in layer_begin_raw):
            raise RuntimeError("missing selected-layer begin timestamp")
        layer_anchor = layer_begin_raw[0]
        layer_begin_offsets = [
            signed_wrapped_delta(timestamp, layer_anchor)
            for timestamp in layer_begin_raw
        ]
        layer_origin = min(layer_begin_offsets)
        reset_record = next(
            record for record in self.step_profile_records if record[0] == 0
        )
        reset_event = reset_record[2]
        reset_elapsed = []
        reset_wait = []
        for sm in range(self.sms):
            packed = int(profile[sm, reset_event].item())
            reset_elapsed.append(packed & 0xFFFFFFFF)
            reset_wait.append((packed >> 32) & 0xFFFFFFFF)
        reset_active_start = [
            layer_begin_offsets[sm] - layer_origin + reset_wait[sm]
            for sm in range(self.sms)
        ]
        reset_compute_end = [
            layer_begin_offsets[sm] - layer_origin + reset_elapsed[sm]
            for sm in range(self.sms)
        ]
        reset_alloc_event = ldu_base + 26
        reset_alloc_begin_raw = [
            int(profile[sm, reset_alloc_event].item()) & 0xFFFFFFFF
            for sm in range(self.sms)
        ]
        reset_alloc_end_raw = [
            int(profile[sm, reset_alloc_event + 1].item()) & 0xFFFFFFFF
            for sm in range(self.sms)
        ]
        reset_store_end_raw = [
            int(profile[sm, reset_alloc_event + 2].item()) & 0xFFFFFFFF
            for sm in range(self.sms)
        ]
        reload_end_raw = [
            int(profile[sm, reset_alloc_event + 3].item()) & 0xFFFFFFFF
            for sm in range(self.sms)
        ]
        if any(
            timestamp == 0
            for timestamp in (
                *reset_alloc_begin_raw,
                *reset_alloc_end_raw,
                *reset_store_end_raw,
                *reload_end_raw,
            )
        ):
            raise RuntimeError("missing reset allocation/STU timestamp")
        reset_alloc_begin = [
            signed_wrapped_delta(timestamp, layer_anchor) - layer_origin
            for timestamp in reset_alloc_begin_raw
        ]
        reset_alloc_end = [
            signed_wrapped_delta(timestamp, layer_anchor) - layer_origin
            for timestamp in reset_alloc_end_raw
        ]
        reset_store_end = [
            signed_wrapped_delta(timestamp, layer_anchor) - layer_origin
            for timestamp in reset_store_end_raw
        ]
        reset_alloc_service = [
            signed_wrapped_delta(end, begin)
            for begin, end in zip(
                reset_alloc_begin_raw, reset_alloc_end_raw
            )
        ]
        reload_end = [
            signed_wrapped_delta(timestamp, layer_anchor) - layer_origin
            for timestamp in reload_end_raw
        ]
        layer_begin_to_ldu_reload_end = [
            signed_wrapped_delta(reload, begin)
            for begin, reload in zip(layer_begin_raw, reload_end_raw)
        ]
        reload_to_reset_alloc = [
            signed_wrapped_delta(allocation, reload)
            for reload, allocation in zip(
                reload_end_raw, reset_alloc_begin_raw
            )
        ]
        quant_offsets_from_layer = [
            signed_wrapped_delta(timestamp, layer_anchor) - layer_origin
            for timestamp in quant_store_raw
        ]
        pair_counts = set(values("ldu1_pair_count"))
        if len(pair_counts) != 1:
            raise RuntimeError(
                f"Q-a tasks disagree on K-pair count: {sorted(pair_counts)}"
            )
        pair_count = pair_counts.pop()
        total_pairs = self.config.hidden_size // 256
        split_count = total_pairs // pair_count
        if total_pairs % pair_count or num_sms % split_count:
            raise RuntimeError("Q-a split placement cannot be mapped to quant shards")
        output_pairs = num_sms // split_count
        global_ready = max(quant_store_offsets)
        local_gate_savings = []
        for local_task, record in enumerate(records):
            split = local_task // output_pairs
            producer_start = split * pair_count
            local_ready = max(
                quant_store_offsets[
                    producer_start : producer_start + pair_count
                ]
            )
            arrival = signed_wrapped_delta(
                record["ldu1_begin_raw"], quant_anchor
            )
            local_gate_savings.append(
                max(arrival, global_ready) - max(arrival, local_ready)
            )
        source_offsets_from_quant = [
            signed_wrapped_delta(
                record["ldu1_source_begin_raw"], quant_anchor
            )
            for record in records
        ]
        late_layer_sms = sorted(
            range(self.sms),
            key=lambda sm: layer_begin_offsets[sm],
            reverse=True,
        )[:16]
        late_reset_sms = sorted(
            range(self.sms),
            key=lambda sm: reset_active_start[sm],
            reverse=True,
        )[:16]
        print(
            "DSV4_FP8_QUANT_HANDOFF "
            f"producer_sms={quant_sms} "
            f"producer_finish_spread_us="
            f"{(max(quant_store_offsets) - min(quant_store_offsets)) / 1.0e3:.3f} "
            f"layer_begin_spread_us="
            f"{(max(layer_begin_offsets) - layer_origin) / 1.0e3:.3f} "
            f"reset_active_start_spread_us="
            f"{(max(reset_active_start) - min(reset_active_start)) / 1.0e3:.3f} "
            f"reset_compute_end_spread_us="
            f"{(max(reset_compute_end) - min(reset_compute_end)) / 1.0e3:.3f} "
            f"reset_alloc_begin_spread_us="
            f"{(max(reset_alloc_begin) - min(reset_alloc_begin)) / 1.0e3:.3f} "
            f"reset_alloc_service_median_us="
            f"{statistics.median(reset_alloc_service) / 1.0e3:.3f} "
            f"reset_alloc_service_max_us="
            f"{max(reset_alloc_service) / 1.0e3:.3f} "
            f"reset_alloc_end_spread_us="
            f"{(max(reset_alloc_end) - min(reset_alloc_end)) / 1.0e3:.3f} "
            f"reset_store_end_spread_us="
            f"{(max(reset_store_end) - min(reset_store_end)) / 1.0e3:.3f} "
            f"reload_end_spread_us="
            f"{(max(reload_end) - min(reload_end)) / 1.0e3:.3f} "
            f"layer_begin_to_ldu_reload_end_min_us="
            f"{min(layer_begin_to_ldu_reload_end) / 1.0e3:.3f} "
            f"layer_begin_to_ldu_reload_end_median_us="
            f"{statistics.median(layer_begin_to_ldu_reload_end) / 1.0e3:.3f} "
            f"layer_begin_to_ldu_reload_end_max_us="
            f"{max(layer_begin_to_ldu_reload_end) / 1.0e3:.3f} "
            f"reload_to_alloc_min_us="
            f"{min(reload_to_reset_alloc) / 1.0e3:.3f} "
            f"reload_to_alloc_median_us="
            f"{statistics.median(reload_to_reset_alloc) / 1.0e3:.3f} "
            f"reload_to_alloc_max_us="
            f"{max(reload_to_reset_alloc) / 1.0e3:.3f} "
            f"quant_finish_frontier_us="
            f"{max(quant_offsets_from_layer) / 1.0e3:.3f} "
            f"begin_0_15_median_us="
            f"{(statistics.median(layer_begin_offsets[0:16]) - layer_origin) / 1.0e3:.3f} "
            f"begin_16_31_median_us="
            f"{(statistics.median(layer_begin_offsets[16:32]) - layer_origin) / 1.0e3:.3f} "
            f"begin_32_151_median_us="
            f"{(statistics.median(layer_begin_offsets[32:]) - layer_origin) / 1.0e3:.3f} "
            f"global_release_lag_us="
            f"{(min(source_offsets_from_quant) - global_ready) / 1.0e3:.3f} "
            f"shard_gate_saving_median_us="
            f"{statistics.median(local_gate_savings) / 1.0e3:.3f} "
            f"shard_gate_saving_max_us="
            f"{max(local_gate_savings) / 1.0e3:.3f} "
            f"pair_count={pair_count} output_pairs={output_pairs} "
            "late_sm:begin_us="
            + ",".join(
                f"{sm}:{(layer_begin_offsets[sm] - layer_origin) / 1.0e3:.3f}"
                for sm in late_layer_sms
            )
            + " "
            "late_reset_sm:active_start_us="
            + ",".join(
                f"{sm}:{reset_active_start[sm] / 1.0e3:.3f}"
                for sm in late_reset_sms
            )
            + " "
            f"sample_index={sample_index if sample_index is not None else -1}",
            flush=True,
        )

        critical = max(records, key=lambda record: record["step"])
        critical_tail = sorted(
            records,
            key=lambda record: record["step_active"],
            reverse=True,
        )[:8]
        print(
            "DSV4_FP8_COUPLED_DETAIL "
            "stage=attn.q_b "
            f"active_sms={len(records)} "
            f"step_us={max(values('step')) / 1.0e3:.3f} "
            f"step_median_us={statistics.median(values('step')) / 1.0e3:.3f} "
            f"m2c_wait_us={critical['step_wait'] / 1.0e3:.3f} "
            f"compute_active_us={critical['step_active'] / 1.0e3:.3f} "
            f"compute_active_median_us={statistics.median(values('step_active')) / 1.0e3:.3f} "
            f"ldu0_begin_median_us={statistics.median(values('ldu0_begin')) / 1.0e3:.3f} "
            f"ldu1_begin_median_us={statistics.median(values('ldu1_begin')) / 1.0e3:.3f} "
            f"ldu0_service_median_us={statistics.median(values('ldu0_service')) / 1.0e3:.3f} "
            f"ldu1_service_median_us={statistics.median(values('ldu1_service')) / 1.0e3:.3f} "
            f"ldu0_preceding_gap_median_us={statistics.median(values('ldu0_gap')) / 1.0e3:.3f} "
            f"ldu1_preceding_gap_median_us={statistics.median(values('ldu1_gap')) / 1.0e3:.3f} "
            f"post_issue_tail_median_us="
            f"{statistics.median(values('post_issue_tail')) / 1.0e3:.3f} "
            f"critical_sm={critical['sm']} "
            f"critical_ldu0_service_us={critical['ldu0_service'] / 1.0e3:.3f} "
            f"critical_ldu1_service_us={critical['ldu1_service'] / 1.0e3:.3f} "
            f"critical_post_issue_tail_us="
            f"{critical['post_issue_tail'] / 1.0e3:.3f} "
            "top_sm:active_us:ldu0_us:ldu1_us:post_issue_us="
            + ",".join(
                f"{record['sm']}:{record['step_active'] / 1.0e3:.3f}:"
                f"{record['ldu0_service'] / 1.0e3:.3f}:"
                f"{record['ldu1_service'] / 1.0e3:.3f}:"
                f"{record['post_issue_tail'] / 1.0e3:.3f}"
                for record in critical_tail
            )
            + " "
            f"sample_index={sample_index if sample_index is not None else -1} "
            f"sample_cuda_ms={sample_cuda_ms if sample_cuda_ms is not None else -1.0:.6f}",
            flush=True,
        )
        for port in range(2):
            command_anchor = records[0][f"ldu{port}_begin_raw"]
            command_offsets = [
                signed_wrapped_delta(
                    record[f"ldu{port}_begin_raw"], command_anchor
                )
                for record in records
            ]
            source_anchor = records[0][f"ldu{port}_source_begin_raw"]
            source_offsets = [
                signed_wrapped_delta(
                    record[f"ldu{port}_source_begin_raw"], source_anchor
                )
                for record in records
            ]
            print(
                "DSV4_FP8_COUPLED_SOURCE_LOAD "
                f"port={port} "
                f"command_begin_spread_us="
                f"{(max(command_offsets) - min(command_offsets)) / 1.0e3:.3f} "
                f"source_begin_spread_us="
                f"{(max(source_offsets) - min(source_offsets)) / 1.0e3:.3f} "
                f"gate_wait_min_us={min(values(f'ldu{port}_gate_wait')) / 1.0e3:.3f} "
                f"gate_wait_median_us={statistics.median(values(f'ldu{port}_gate_wait')) / 1.0e3:.3f} "
                f"gate_wait_max_us={max(values(f'ldu{port}_gate_wait')) / 1.0e3:.3f} "
                f"begin_median_us={statistics.median(values(f'ldu{port}_source_begin')) / 1.0e3:.3f} "
                f"wait_median_us={statistics.median(values(f'ldu{port}_source_wait')) / 1.0e3:.3f} "
                f"wait_max_us={max(values(f'ldu{port}_source_wait')) / 1.0e3:.3f} "
                f"sample_index={sample_index if sample_index is not None else -1}",
                flush=True,
            )
            for pair in range(2):
                wait_key = f"ldu{port}_pair{pair}_wait"
                begin_key = f"ldu{port}_pair{pair}_begin"
                expected_key = f"ldu{port}_pair{pair}_expected_ready"
                opposite_key = f"ldu{port}_pair{pair}_opposite_ready"
                stage_key = f"ldu{port}_pair{pair}_stage"
                phase_key = f"ldu{port}_pair{pair}_phase"
                print(
                    "DSV4_FP8_COUPLED_EMPTY_WAIT "
                    f"port={port} pair={pair} "
                    f"begin_median_us={statistics.median(values(begin_key)) / 1.0e3:.3f} "
                    f"wait_median_us={statistics.median(values(wait_key)) / 1.0e3:.3f} "
                    f"wait_max_us={max(values(wait_key)) / 1.0e3:.3f} "
                    f"expected_ready={sum(values(expected_key))}/{len(records)} "
                    f"opposite_ready={sum(values(opposite_key))}/{len(records)} "
                    f"stages={sorted(set(values(stage_key)))} "
                    f"phases={sorted(set(values(phase_key)))} "
                    f"phase_bases={sorted(set(values(f'ldu{port}_phase_base')))} "
                    f"pair_counts={sorted(set(values(f'ldu{port}_pair_count')))} "
                    f"sample_index={sample_index if sample_index is not None else -1}",
                    flush=True,
                )
        for (port, command, normalized_opcode), samples in sorted(
            trace.items()
        ):
            offsets = [sample[0] for sample in samples]
            durations = [sample[1] for sample in samples]
            ends = [
                offset + duration
                for offset, duration in samples
            ]
            opcode_name = (
                "OP_TMA_LOAD_MX_COUPLED_STREAM_FP8"
                if normalized_opcode == coupled_load_opcode
                else opcode_names.get(
                    normalized_opcode, f"mop_{normalized_opcode}"
                )
            )
            print(
                "DSV4_LDU_PREFIX_COMMAND "
                f"port={port} command={command} "
                f"opcode={opcode_name} "
                f"samples={len(samples)} "
                f"begin_median_us={statistics.median(offsets) / 1.0e3:.3f} "
                f"end_median_us={statistics.median(ends) / 1.0e3:.3f} "
                f"service_median_us={statistics.median(durations) / 1.0e3:.3f} "
                f"sample_index={sample_index if sample_index is not None else -1}",
                flush=True,
            )

    def report_attention_detail_profile(
        self,
        profile: torch.Tensor | None = None,
        *,
        sample_index: int | None = None,
        sample_cuda_ms: float | None = None,
    ) -> None:
        if not self.args.profile_attention_detail:
            return
        if profile is None:
            profile = self.launcher.profile.cpu()
        magic = 0x4454524B50524631
        if any(int(value) != magic for value in profile[:, 127]):
            raise RuntimeError(
                "attention detail profiling requires track_profile=1"
            )
        # Attention-detail events are overwritten by each execution of the
        # same physical producer SMs.  In a two-layer diagnostic the retained
        # records therefore belong to the final family, which is also the
        # production HCA -> CSA handoff we need to inspect.
        family = self.families[-1]
        kind = self.config.attention_kind(family.representative)
        rows = self.attention_indices_by_kind[kind].numel()
        num_splits = (rows + 63) // 64
        producer_base = (
            self.args.sms - num_splits
            if self.args.sms >= 152
            else 0
        )
        detail = profile[producer_base : producer_base + num_splits].numpy()
        detail_base = runtime_config.detail_profile_event_base
        event_names = (
            (2, "enter"),
            (3, "operands"),
            (4, "qk"),
            (5, "output"),
            (6, "softmax"),
            (20, "metadata"),
            (7, "pv0-done"),
            (30, "pv0-alloc"),
            (34, "pv0-tmem"),
            (8, "pv0-store"),
            (9, "pv0-reuse"),
            (10, "pv1-done"),
            (31, "pv1-alloc"),
            (35, "pv1-tmem"),
            (11, "pv1-store"),
            (12, "pv1-reuse"),
            (13, "pv2-done"),
            (32, "pv2-alloc"),
            (36, "pv2-tmem"),
            (14, "pv2-store"),
            (15, "pv2-reuse"),
            (16, "pv3-done"),
            (33, "pv3-alloc"),
            (37, "pv3-tmem"),
            (17, "pv3-store"),
            (18, "pv3-reuse"),
            (19, "published"),
            (21, "done"),
        )
        pieces = ["enter=0.000/0.000/0/0.000"]
        previous_event = 2
        for event_id, label in event_names[1:]:
            ns_deltas = []
            ns_offsets = []
            cycle_deltas = []
            for row in detail:
                begin = int(row[detail_base + 2])
                previous = int(row[detail_base + previous_event])
                current = int(row[detail_base + event_id])
                ns_deltas.append(
                    ((current & 0xFFFFFFFF) - (previous & 0xFFFFFFFF))
                    & 0xFFFFFFFF
                )
                ns_offsets.append(
                    ((current & 0xFFFFFFFF) - (begin & 0xFFFFFFFF))
                    & 0xFFFFFFFF
                )
                cycle_deltas.append(
                    (((current >> 32) & 0xFFFFFFFF)
                     - ((previous >> 32) & 0xFFFFFFFF))
                    & 0xFFFFFFFF
                )
            median_ns = statistics.median(ns_deltas)
            median_cycles = statistics.median(cycle_deltas)
            pieces.append(
                f"{label}={median_ns / 1.0e3:.3f}/"
                f"{statistics.median(ns_offsets) / 1.0e3:.3f}/"
                f"{median_cycles:.0f}/"
                f"{median_cycles / median_ns if median_ns else 0.0:.3f}"
            )
            previous_event = event_id
        print(
            "DSV4_ATTN_DETAIL_PROFILE "
            f"kind={kind} rows={rows} splits={num_splits} "
            "delta_us/median_offset_us/cycles/effective_ghz "
            + " ".join(pieces)
            + f" sample_index={sample_index if sample_index is not None else -1}"
            + f" sample_cuda_ms={sample_cuda_ms if sample_cuda_ms is not None else -1.0:.6f}",
            flush=True,
        )
        q_store_before = [
            int(value) & 0xFFFFFFFF
            for value in profile[: self.config.num_heads, detail_base + 46]
            if int(value) != 0
        ]
        q_store_after = [
            int(value) & 0xFFFFFFFF
            for value in profile[: self.config.num_heads, detail_base + 47]
            if int(value) != 0
        ]
        if q_store_before and q_store_after:
            attention_enter = int(detail[0, detail_base + 2]) & 0xFFFFFFFF

            def store_offset(value: int) -> int:
                delta = (value - attention_enter) & 0xFFFFFFFF
                return delta - (1 << 32) if delta >= 1 << 31 else delta

            before_offsets = [store_offset(value) for value in q_store_before]
            after_offsets = [store_offset(value) for value in q_store_after]
            print(
                "DSV4_ATTN_Q_STORE_FRONTIER "
                f"kind={kind} samples={len(before_offsets)} "
                f"before_atomic_min_us={min(before_offsets) / 1.0e3:.3f} "
                f"before_atomic_median_us="
                f"{statistics.median(before_offsets) / 1.0e3:.3f} "
                f"before_atomic_max_us={max(before_offsets) / 1.0e3:.3f} "
                f"after_atomic_min_us={min(after_offsets) / 1.0e3:.3f} "
                f"after_atomic_median_us="
                f"{statistics.median(after_offsets) / 1.0e3:.3f} "
                f"after_atomic_max_us={max(after_offsets) / 1.0e3:.3f}",
                flush=True,
            )
        full_row_groups, residual_rows = divmod(rows, 8)
        base_groups, extra_groups = divmod(full_row_groups, num_splits)
        split_lengths = [
            8 * (base_groups + (split < extra_groups))
            for split in range(num_splits)
        ]
        split_lengths[-1] += residual_rows
        enter_times = [
            int(row[detail_base + 2]) & 0xFFFFFFFF for row in detail
        ]
        enter_anchor = enter_times[0]
        row_start = 0
        for split, (row, split_length) in enumerate(
            zip(detail, split_lengths, strict=True)
        ):
            enter = int(row[detail_base + 2]) & 0xFFFFFFFF
            operands = int(row[detail_base + 3]) & 0xFFFFFFFF
            ring_token = int(row[detail_base + 0]) & 0xFFFFFFFF
            q_token = int(row[detail_base + 1]) & 0xFFFFFFFF
            ring0_ready = int(row[detail_base + 38]) & 0xFFFFFFFF
            ring1_ready = int(row[detail_base + 39]) & 0xFFFFFFFF
            published = int(row[detail_base + 19]) & 0xFFFFFFFF
            done = int(row[detail_base + 21]) & 0xFFFFFFFF
            enter_offset = (enter - enter_anchor) & 0xFFFFFFFF
            if enter_offset >= 1 << 31:
                enter_offset -= 1 << 32
            print(
                "DSV4_ATTN_DETAIL_SPLIT "
                f"kind={kind} split={split} row_start={row_start} "
                f"rows={split_length} "
                f"contains_current={int(row_start + split_length == rows)} "
                f"enter_offset_us={enter_offset / 1.0e3:.3f} "
                f"ring_token_us={((ring_token - enter) & 0xFFFFFFFF) / 1.0e3:.3f} "
                f"q_token_us={((q_token - enter) & 0xFFFFFFFF) / 1.0e3:.3f} "
                f"ring0_ready_us={((ring0_ready - enter) & 0xFFFFFFFF) / 1.0e3:.3f} "
                f"ring1_ready_us={((ring1_ready - enter) & 0xFFFFFFFF) / 1.0e3:.3f} "
                f"operand_wait_us={((operands - enter) & 0xFFFFFFFF) / 1.0e3:.3f} "
                f"active_to_publish_us={((published - operands) & 0xFFFFFFFF) / 1.0e3:.3f} "
                f"active_to_done_us={((done - operands) & 0xFFFFFFFF) / 1.0e3:.3f}",
                flush=True,
            )
            def ldu_offset(event_id: int) -> float:
                value = int(row[detail_base + event_id]) & 0xFFFFFFFF
                if value == 0:
                    return -1.0
                return ((value - enter) & 0xFFFFFFFF) / 1.0e3

            print(
                "DSV4_ATTN_LDU_DETAIL "
                f"kind={kind} split={split} "
                f"ring_begin_us={ldu_offset(40):.3f} "
                f"ring_dependency_us={ldu_offset(41):.3f} "
                f"ring_issue_us={ldu_offset(42):.3f} "
                f"q_begin_us={ldu_offset(43):.3f} "
                f"q_dependency_us={ldu_offset(44):.3f} "
                f"q_issue_us={ldu_offset(45):.3f}",
                flush=True,
            )
            row_start += split_length

    def report_mxfp_ffn_detail_profile(
        self,
        profile: torch.Tensor | None = None,
        *,
        sample_index: int | None = None,
        sample_cuda_ms: float | None = None,
    ) -> None:
        if not self.args.profile_mxfp_ffn_detail:
            return
        if profile is None:
            profile = self.launcher.profile.cpu()
        magic = 0x4454524B50524631
        if any(int(value) != magic for value in profile[:, 127]):
            raise RuntimeError(
                "MXFP FFN detail profiling requires "
                "mxfp_ffn_detail_profile=1"
            )
        # The detail build leaves reload timing disabled, so this compact
        # event range is available even in the full repeated-layer image.
        base = runtime_config.reload_profile_event_base
        event_names = (
            "allocator-linear1",
            "allocator-down-weight",
            "allocator-down-activation",
            "ldu0-linear1-begin",
            "ldu0-linear1-end",
            "ldu0-down-begin",
            "ldu0-down-ready",
            "ldu0-down-end",
            "ldu1-activation-begin",
            "ldu1-poll-ready",
            "ldu1-activation-end",
            "compute-begin",
            "compute-linear1-end",
            "compute-end",
            "ldu0-previous-begin",
            "ldu0-previous-end",
        )
        resident_base = self.sms - 112
        detail = profile[resident_base : resident_base + 112]
        physical_sm_ids = [
            int(value)
            for value in profile[:, runtime_config.track_profile_event_base + 25]
        ]
        unavailable_events: set[int] = set()
        for event_offset, name in enumerate(event_names):
            if event_offset in unavailable_events:
                continue
            values = [int(value) for value in detail[:, base + event_offset]]
            if any(value == 0 for value in values):
                raise RuntimeError(
                    f"MXFP FFN detail event {name!r} was not recorded"
                )
        compute_begin = [
            int(value) for value in detail[:, base + 11]
        ]
        final_layer_start = (
            max(int(value) for value in profile[:, 0])
            if len(self.profile_layer_ids) == 1
            else max(
                int(value)
                for value in profile[
                    :,
                    runtime_config.layer_profile_event_base
                    + len(self.profile_layer_ids)
                    - 2,
                ]
            )
        )
        final_layer_end = max(
            int(value)
            for value in profile[
                :,
                runtime_config.layer_profile_event_base
                + len(self.profile_layer_ids)
                - 1,
            ]
        )
        print(
            "DSV4_MXFP_FFN_DETAIL_LAYER "
            f"layer={self.profile_layer_ids[-1]} "
            f"start_to_linear1_begin_us="
            f"{(min(int(value) for value in detail[:, base + 3]) - final_layer_start) / 1.0e3:.3f} "
            f"start_to_compute_begin_us="
            f"{(min(compute_begin) - final_layer_start) / 1.0e3:.3f} "
            f"layer_span_us={(final_layer_end - final_layer_start) / 1.0e3:.3f}",
            flush=True,
        )

        def percentile(values: list[int], fraction: float) -> float:
            ordered = sorted(values)
            position = fraction * (len(ordered) - 1)
            lower = int(math.floor(position))
            upper = int(math.ceil(position))
            if lower == upper:
                return float(ordered[lower])
            weight = position - lower
            return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

        for event_offset, name in enumerate(event_names):
            if event_offset in unavailable_events:
                continue
            offsets = [
                int(value) - origin
                for value, origin in zip(
                    detail[:, base + event_offset], compute_begin
                )
            ]
            print(
                "DSV4_MXFP_FFN_DETAIL_EVENT "
                f"name={name} "
                f"min_offset_us={min(offsets) / 1.0e3:.3f} "
                f"median_offset_us={statistics.median(offsets) / 1.0e3:.3f} "
                f"p95_offset_us={percentile(offsets, 0.95) / 1.0e3:.3f} "
                f"max_offset_us={max(offsets) / 1.0e3:.3f}",
                flush=True,
            )

        intervals = (
            ("linear1-command-queue", 0, 3),
            ("down-weight-command-queue", 1, 5),
            ("down-activation-command-queue", 2, 8),
            ("ldu0-linear1", 3, 4),
            ("ldu0-linear1-input-dependency", 3, 21),
            ("ldu0-linear1-post-dependency", 21, 4),
            ("ldu0-down-handoff-wait", 5, 6),
            ("ldu0-down-service", 6, 7),
            ("ldu1-poll-wait", 8, 9),
            ("ldu1-activation-service", 9, 10),
            ("compute-linear1", 11, 12),
            ("compute-down", 12, 13),
            ("compute-total", 11, 13),
            ("ldu0-previous-service", 14, 15),
            ("previous-to-linear1", 15, 3),
        )
        for name, begin_offset, end_offset in intervals:
            if begin_offset in unavailable_events or end_offset in unavailable_events:
                continue
            interval_base = 16 if 21 in (begin_offset, end_offset) else 0
            interval_detail = detail[interval_base:]
            durations = [
                int(end) - int(begin)
                for begin, end in zip(
                    interval_detail[:, base + begin_offset],
                    interval_detail[:, base + end_offset],
                )
            ]
            if min(durations) < 0:
                raise RuntimeError(
                    f"MXFP FFN interval {name!r} has reversed timestamps"
                )
            slowest_local = max(
                range(len(durations)), key=durations.__getitem__
            )
            slowest_vcore = resident_base + interval_base + slowest_local
            print(
                "DSV4_MXFP_FFN_DETAIL_INTERVAL "
                f"name={name} min_us={min(durations) / 1.0e3:.3f} "
                f"median_us={statistics.median(durations) / 1.0e3:.3f} "
                f"p95_us={percentile(durations, 0.95) / 1.0e3:.3f} "
                f"max_us={max(durations) / 1.0e3:.3f} "
                f"slowest_vcore={slowest_vcore} "
                f"slowest_physical_sm={physical_sm_ids[slowest_vcore]}",
                flush=True,
            )
        opcode_counts: dict[int, int] = {}
        opcode_vcores: dict[int, list[int]] = {}
        for local_vcore, value in enumerate(detail[:, base + 16]):
            opcode_value = int(value)
            opcode_counts[opcode_value] = opcode_counts.get(opcode_value, 0) + 1
            opcode_vcores.setdefault(opcode_value, []).append(
                resident_base + local_vcore
            )
        print(
            "DSV4_MXFP_FFN_DETAIL_PREVIOUS_OPCODES counts="
            + ",".join(
                f"0x{opcode_value:04x}:{count}"
                for opcode_value, count in sorted(opcode_counts.items())
            ),
            flush=True,
        )
        print(
            "DSV4_MXFP_FFN_DETAIL_PREVIOUS_VCORES groups="
            + ";".join(
                f"0x{opcode_value:04x}:"
                + ",".join(str(vcore) for vcore in vcores)
                for opcode_value, vcores in sorted(opcode_vcores.items())
            ),
            flush=True,
        )
        duration_counters = (
            ("ldu0-linear1-prologue", 17),
            ("ldu0-linear1-stage-empty-wait", 18),
            ("compute-linear1-weight-full-wait", 19),
            ("compute-linear1-umma-full-wait", 20),
            ("ldu0-down-stage-empty-wait", 22),
            ("ldu1-down-stage-empty-wait", 23),
            ("compute-down-weight-full-wait", 24),
            ("compute-down-operand-full-wait", 25),
            ("compute-down-umma-full-wait", 26),
        )
        for name, event_offset in duration_counters:
            durations = [
                int(value)
                for value in detail[:, base + event_offset]
            ]
            slowest_local = max(
                range(len(durations)), key=durations.__getitem__
            )
            slowest_vcore = resident_base + slowest_local
            top_locals = sorted(
                range(len(durations)),
                key=durations.__getitem__,
                reverse=True,
            )[:8]
            print(
                "DSV4_MXFP_FFN_DETAIL_COUNTER "
                f"name={name} min_us={min(durations) / 1.0e3:.3f} "
                f"median_us={statistics.median(durations) / 1.0e3:.3f} "
                f"p95_us={percentile(durations, 0.95) / 1.0e3:.3f} "
                f"stddev_us={statistics.pstdev(durations) / 1.0e3:.3f} "
                f"max_us={max(durations) / 1.0e3:.3f} "
                f"slowest_vcore={slowest_vcore} "
                f"slowest_physical_sm={physical_sm_ids[slowest_vcore]} "
                "top_vcore:physical_sm:us="
                + ",".join(
                    f"{resident_base + local}:"
                    f"{physical_sm_ids[resident_base + local]}:"
                    f"{durations[local] / 1.0e3:.3f}"
                    for local in top_locals
                ),
                flush=True,
            )

        def unpack_task_half(value: int, task_order: int) -> int:
            return (value >> (32 * task_order)) & 0xFFFFFFFF

        packed_phase_counters = (
            ("compute-down-to-umma-done", 27),
            ("compute-down-epilogue", 28),
            ("compute-down-reduction-wait", 29),
            ("compute-down-output-tma", 30),
        )
        # Preserve the historical per-worker sums for direct comparison with
        # older detail traces, while decoding each task below.
        for name, event_offset in packed_phase_counters:
            durations = [
                unpack_task_half(int(value), 0)
                + unpack_task_half(int(value), 1)
                for value in detail[:, base + event_offset]
            ]
            print(
                "DSV4_MXFP_FFN_DETAIL_COUNTER "
                f"name={name} min_us={min(durations) / 1.0e3:.3f} "
                f"median_us={statistics.median(durations) / 1.0e3:.3f} "
                f"p95_us={percentile(durations, 0.95) / 1.0e3:.3f} "
                f"stddev_us={statistics.pstdev(durations) / 1.0e3:.3f} "
                f"max_us={max(durations) / 1.0e3:.3f}",
                flush=True,
            )

        down_tasks: list[dict[str, int | str]] = []
        placed_residents = [
            schedule
            for schedule in self.program.placed_schedules
            if isinstance(
                schedule, SchedLayeredMxfp4Mxfp8RoutedResidentFfn
            )
        ]
        if not placed_residents:
            raise RuntimeError("placed MXFP resident schedule was not retained")
        final_resident = placed_residents[-1].placed_resident
        task_queues = final_resident.task_queues
        metadata_variants = final_resident.down_metadata_variants
        for vcore in range(self.sms):
            packed_phases = [
                int(profile[vcore, base + event_offset])
                for _, event_offset in packed_phase_counters
            ]
            packed_begins = int(profile[vcore, base + 31])
            down_origin = int(profile[vcore, base + 12])
            for task_order in range(2):
                phases = [
                    unpack_task_half(value, task_order)
                    for value in packed_phases
                ]
                if task_order == 1 and not any(phases):
                    continue
                if self.sms not in (112, 152):
                    raise RuntimeError(
                        "MXFP FFN task detail expects 112 or 152 workers"
                    )
                metadata_record = task_queues[vcore][task_order]
                output_task, split_variant = divmod(
                    metadata_record, metadata_variants
                )
                if split_variant == 0:
                    task_class = (
                        "shared-full"
                        if output_task < 32
                        else "routed-full"
                    )
                elif split_variant == 1:
                    task_class = (
                        "shared-split-first"
                        if output_task < 32
                        else "routed-split-first"
                    )
                else:
                    task_class = (
                        "shared-split-final"
                        if output_task < 32
                        else "routed-split-final"
                    )
                task_begin = down_origin + unpack_task_half(
                    packed_begins, task_order
                )
                down_tasks.append(
                    {
                        "vcore": vcore,
                        "physical_sm": physical_sm_ids[vcore],
                        "order": task_order,
                        "output_task": output_task,
                        "tile": output_task % 32,
                        "class": task_class,
                        "begin": task_begin,
                        "to_umma": phases[0],
                        "epilogue": phases[1],
                        "reduction_wait": phases[2],
                        "output_tma": phases[3],
                        "finish": task_begin + sum(phases),
                    }
                )
        task_origin = min(int(task["begin"]) for task in down_tasks)
        for task_class in (
            "shared-full",
            "routed-full",
            "routed-split-first",
            "routed-split-final",
        ):
            group = [task for task in down_tasks if task["class"] == task_class]
            if not group:
                continue
            totals = [
                int(task["finish"]) - int(task["begin"])
                for task in group
            ]
            starts = [int(task["begin"]) - task_origin for task in group]
            waits = [int(task["reduction_wait"]) for task in group]
            print(
                "DSV4_MXFP_FFN_DOWN_CLASS "
                f"class={task_class} count={len(group)} "
                f"start_median_us={statistics.median(starts) / 1.0e3:.3f} "
                f"total_median_us={statistics.median(totals) / 1.0e3:.3f} "
                f"to_umma_median_us="
                f"{statistics.median(int(task['to_umma']) for task in group) / 1.0e3:.3f} "
                f"reduction_wait_median_us={statistics.median(waits) / 1.0e3:.3f} "
                f"reduction_wait_p95_us={percentile(waits, 0.95) / 1.0e3:.3f} "
                f"output_tma_median_us="
                f"{statistics.median(int(task['output_tma']) for task in group) / 1.0e3:.3f}",
                flush=True,
            )

        shared_output_done = {
            int(task["tile"]): int(task["finish"])
            for task in down_tasks
            if task["class"] == "shared-full"
        }
        predicted_waits: list[int] = []
        measured_waits: list[int] = []
        residuals: list[int] = []
        for task in down_tasks:
            if task["class"] == "shared-full":
                continue
            shared_done = shared_output_done.get(int(task["tile"]))
            if shared_done is None:
                continue
            epilogue_done = (
                int(task["begin"])
                + int(task["to_umma"])
                + int(task["epilogue"])
            )
            predicted = max(0, shared_done - epilogue_done)
            measured = int(task["reduction_wait"])
            predicted_waits.append(predicted)
            measured_waits.append(measured)
            residuals.append(measured - predicted)
        print(
            "DSV4_MXFP_FFN_DOWN_DEPENDENCY "
            f"tasks={len(measured_waits)} "
            f"shared_frontier_prediction_median_us="
            f"{statistics.median(predicted_waits) / 1.0e3:.3f} "
            f"measured_wait_median_us="
            f"{statistics.median(measured_waits) / 1.0e3:.3f} "
            f"publication_residual_median_us="
            f"{statistics.median(residuals) / 1.0e3:.3f}",
            flush=True,
        )
        for physical_expert in range(7):
            expert_tasks = [
                task
                for task in down_tasks
                if int(task["output_task"]) // 32 == physical_expert
            ]
            if not expert_tasks:
                continue
            finishes = [
                int(task["finish"]) - task_origin for task in expert_tasks
            ]
            starts = [
                int(task["begin"]) - task_origin for task in expert_tasks
            ]
            print(
                "DSV4_MXFP_FFN_DOWN_EXPERT "
                f"physical_expert={physical_expert} "
                f"records={len(expert_tasks)} "
                f"start_median_us={statistics.median(starts) / 1.0e3:.3f} "
                f"finish_median_us={statistics.median(finishes) / 1.0e3:.3f} "
                f"finish_p95_us={percentile(finishes, 0.95) / 1.0e3:.3f} "
                f"finish_max_us={max(finishes) / 1.0e3:.3f}",
                flush=True,
            )
        for task_class in (
            "routed-full",
            "routed-split-first",
            "routed-split-final",
        ):
            class_predicted: list[int] = []
            class_measured: list[int] = []
            for task in down_tasks:
                if task["class"] != task_class:
                    continue
                shared_done = shared_output_done[int(task["tile"])]
                epilogue_done = (
                    int(task["begin"])
                    + int(task["to_umma"])
                    + int(task["epilogue"])
                )
                class_predicted.append(max(0, shared_done - epilogue_done))
                class_measured.append(int(task["reduction_wait"]))
            print(
                "DSV4_MXFP_FFN_DOWN_DEPENDENCY_CLASS "
                f"class={task_class} tasks={len(class_measured)} "
                f"shared_frontier_prediction_median_us="
                f"{statistics.median(class_predicted) / 1.0e3:.3f} "
                f"measured_wait_median_us="
                f"{statistics.median(class_measured) / 1.0e3:.3f}",
                flush=True,
            )
        critical_tasks = sorted(
            down_tasks, key=lambda task: int(task["finish"]), reverse=True
        )[:8]
        print(
            "DSV4_MXFP_FFN_DOWN_CRITICAL tasks="
            + ",".join(
                f"{task['vcore']}:{task['physical_sm']}:"
                f"{task['order']}:{task['output_task']}:"
                f"{task['class']}:"
                f"finish={(int(task['finish']) - task_origin) / 1.0e3:.3f}:"
                f"start={(int(task['begin']) - task_origin) / 1.0e3:.3f}:"
                f"umma={int(task['to_umma']) / 1.0e3:.3f}:"
                f"epilogue={int(task['epilogue']) / 1.0e3:.3f}:"
                f"reduce={int(task['reduction_wait']) / 1.0e3:.3f}:"
                f"output={int(task['output_tma']) / 1.0e3:.3f}"
                for task in critical_tasks
            ),
            flush=True,
        )
        first_tasks = {
            int(task["vcore"]): task
            for task in down_tasks
            if int(task["order"]) == 0
        }
        second_tasks = {
            int(task["vcore"]): task
            for task in down_tasks
            if int(task["order"]) == 1
        }
        no_second = sorted(set(first_tasks) - set(second_tasks))
        latest_second = sorted(
            second_tasks.values(),
            key=lambda task: int(task["finish"]),
            reverse=True,
        )[:16]
        latest_first = sorted(
            first_tasks.values(),
            key=lambda task: int(task["finish"]),
            reverse=True,
        )[:16]
        print(
            "DSV4_MXFP_FFN_DOWN_PLACEMENT no_second="
            + ",".join(
                f"{vcore}:"
                f"{(int(first_tasks[vcore]['finish']) - task_origin) / 1.0e3:.3f}"
                for vcore in no_second
            )
            + " latest_second="
            + ",".join(
                f"{task['vcore']}:"
                f"{(int(task['finish']) - task_origin) / 1.0e3:.3f}"
                for task in latest_second
            ),
            flush=True,
        )
        print(
            "DSV4_MXFP_FFN_DOWN_FIRST_LATEST tasks="
            + ",".join(
                f"{task['vcore']}:{task['physical_sm']}:"
                f"{(int(task['finish']) - task_origin) / 1.0e3:.3f}"
                for task in latest_first
            ),
            flush=True,
        )

        tile_finish = {
            tile: max(
                int(task["finish"])
                for task in down_tasks
                if int(task["tile"]) == tile
            )
            for tile in range(32)
        }
        block_finish = {
            block: max(
                tile_finish[2 * block],
                tile_finish[2 * block + 1],
            )
            for block in range(16)
        }
        global_finish = max(block_finish.values())
        block_leads = [
            global_finish - block_finish[block]
            for block in range(16)
        ]
        print(
            "DSV4_MXFP_FFN_DOWN_BLOCK_FRONTIER "
            f"global_finish_us={(global_finish - task_origin) / 1.0e3:.3f} "
            f"lead_min_us={min(block_leads) / 1.0e3:.3f} "
            f"lead_median_us={statistics.median(block_leads) / 1.0e3:.3f} "
            f"lead_max_us={max(block_leads) / 1.0e3:.3f} "
            "blocks="
            + ",".join(
                f"{block}:"
                f"{(block_finish[block] - task_origin) / 1.0e3:.3f}:"
                f"{(global_finish - block_finish[block]) / 1.0e3:.3f}"
                for block in range(16)
            ),
            flush=True,
        )

        frontier_events = (
            ("compute-begin", 11),
            ("ldu0-linear1-after-dependency", 21),
            ("ldu0-linear1-end", 4),
            ("compute-linear1-end", 12),
            ("ldu0-down-end", 7),
            ("compute-end", 13),
        )
        for name, event_offset in frontier_events:
            frontier_base = 16 if event_offset == 21 else 0
            values = [
                int(value)
                for value in detail[frontier_base:, base + event_offset]
            ]
            origin = min(values)
            offsets = [value - origin for value in values]
            latest_local = max(
                range(len(values)), key=values.__getitem__
            )
            latest_vcore = resident_base + frontier_base + latest_local
            print(
                "DSV4_MXFP_FFN_DETAIL_FRONTIER "
                f"name={name} "
                f"median_from_first_us="
                f"{statistics.median(offsets) / 1.0e3:.3f} "
                f"p95_from_first_us={percentile(offsets, 0.95) / 1.0e3:.3f} "
                f"spread_us={max(offsets) / 1.0e3:.3f} "
                f"latest_vcore={latest_vcore} "
                f"latest_physical_sm={physical_sm_ids[latest_vcore]}",
                flush=True,
            )
        dependency_ready = [
            int(value) for value in detail[16:, base + 21]
        ]
        linear1_complete = [
            int(value) for value in detail[:, base + 12]
        ]
        compute_complete = [
            int(value) for value in detail[:, base + 13]
        ]
        ready_origin = min(dependency_ready)
        print(
            "DSV4_MXFP_FFN_DETAIL_CRITICAL "
            f"dependency_spread_us="
            f"{(max(dependency_ready) - ready_origin) / 1.0e3:.3f} "
            f"ready_to_linear1_end_us="
            f"{(max(linear1_complete) - ready_origin) / 1.0e3:.3f} "
            f"ready_to_compute_end_us="
            f"{(max(compute_complete) - ready_origin) / 1.0e3:.3f}",
            flush=True,
        )
        effective_sm_clock = [
            (int(clock_end) - int(clock_start))
            / (int(timer_end) - int(timer_start))
            for timer_start, timer_end, clock_start, clock_end in zip(
                profile[:, 0], profile[:, 1], profile[:, 122], profile[:, 123]
            )
            if int(timer_end) > int(timer_start)
        ]
        print(
            "DSV4_MXFP_FFN_DETAIL_SUMMARY "
            f"workers={detail.shape[0]} "
            "effective_sm_clock_median_ghz="
            f"{statistics.median(effective_sm_clock):.6f} "
            f"sample_index={sample_index if sample_index is not None else -1} "
            f"sample_cuda_ms={sample_cuda_ms if sample_cuda_ms is not None else -1.0:.6f}",
            flush=True,
        )

    def report_mxfp_ffn_basic_profile(
        self,
        profile: torch.Tensor | None = None,
        *,
        sample_index: int | None = None,
        sample_cuda_ms: float | None = None,
    ) -> None:
        """Report the production resident-FFN timestamps in events 2--5.

        The routed resident handler always records these four events when the
        runtime is built without ``DAE_TRACK_PROFILE``.  Reading them here
        therefore does not add device-side profiling work or alter the image.
        """
        if not self.args.profile_mxfp_ffn_basic:
            return
        if profile is None:
            profile = self.launcher.profile.cpu()
        workers = profile[: self.sms]
        event_names = {
            2: "task-begin",
            3: "task-end",
            4: "linear1-end/down-begin",
            5: "down-end",
        }
        for event_id, name in event_names.items():
            if any(int(value) == 0 for value in workers[:, event_id]):
                raise RuntimeError(
                    f"production MXFP FFN event {name!r} was not recorded"
                )

        def percentile(values: list[int], fraction: float) -> float:
            ordered = sorted(values)
            position = fraction * (len(ordered) - 1)
            lower = int(math.floor(position))
            upper = int(math.ceil(position))
            if lower == upper:
                return float(ordered[lower])
            weight = position - lower
            return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

        def interval(label: str, rows: torch.Tensor, begin: int, end: int) -> None:
            durations = [
                int(stop) - int(start)
                for start, stop in zip(rows[:, begin], rows[:, end])
            ]
            if min(durations) < 0:
                raise RuntimeError(f"production MXFP FFN interval {label!r} reversed")
            print(
                "DSV4_MXFP_FFN_BASIC_INTERVAL "
                f"name={label} min_us={min(durations) / 1.0e3:.3f} "
                f"median_us={statistics.median(durations) / 1.0e3:.3f} "
                f"p95_us={percentile(durations, 0.95) / 1.0e3:.3f} "
                f"max_us={max(durations) / 1.0e3:.3f}",
                flush=True,
            )

        linear1_base = self.sms - 112
        linear1 = workers[linear1_base:]
        task_begin = [int(value) for value in workers[:, 2]]
        task_end = [int(value) for value in workers[:, 3]]
        linear1_begin = [int(value) for value in linear1[:, 2]]
        linear1_end = [int(value) for value in linear1[:, 4]]
        down_begin = [int(value) for value in workers[:, 4]]
        down_end = [int(value) for value in workers[:, 5]]
        kernel_start = max(int(value) for value in workers[:, 0])
        kernel_end = max(int(value) for value in workers[:, 1])
        print(
            "DSV4_MXFP_FFN_BASIC_FRONTIER "
            f"workers={len(workers)} linear1_workers={len(linear1)} "
            f"kernel_to_task_begin_us={(min(task_begin) - kernel_start) / 1.0e3:.3f} "
            f"task_entry_spread_us={(max(task_begin) - min(task_begin)) / 1.0e3:.3f} "
            f"task_envelope_us={(max(task_end) - min(task_begin)) / 1.0e3:.3f} "
            f"linear1_envelope_us={(max(linear1_end) - min(linear1_begin)) / 1.0e3:.3f} "
            f"linear1_finish_spread_us={(max(linear1_end) - min(linear1_end)) / 1.0e3:.3f} "
            f"down_envelope_us={(max(down_end) - min(down_begin)) / 1.0e3:.3f} "
            f"down_finish_spread_us={(max(down_end) - min(down_end)) / 1.0e3:.3f} "
            f"task_end_to_kernel_end_us={(kernel_end - max(task_end)) / 1.0e3:.3f} "
            f"sample_index={sample_index if sample_index is not None else -1} "
            f"sample_cuda_ms={sample_cuda_ms if sample_cuda_ms is not None else -1.0:.6f}",
            flush=True,
        )
        interval("task", workers, 2, 3)
        interval("linear1", linear1, 2, 4)
        interval("down", workers, 4, 5)

    def report_ffn_aggregate_profile(
        self,
        profile: torch.Tensor | None = None,
        *,
        sample_index: int | None = None,
        sample_cuda_ms: float | None = None,
    ) -> None:
        if not self.args.profile_ffn_aggregate:
            return
        if profile is None:
            profile = self.launcher.profile.cpu()
        magic = 0x4454524B50524631
        if any(int(value) != magic for value in profile[:, 127]):
            raise RuntimeError(
                "aggregate FFN profiling requires aggregate_profile=1"
            )
        for label, (_, aggregate_event) in self.ffn_aggregate_events.items():
            if label not in self.ffn_aggregate_used:
                continue
            samples = []
            for sm, value in enumerate(profile[:, aggregate_event]):
                packed = int(value)
                count = (packed >> 48) & 0xFFFF
                if count == 0:
                    continue
                total_ns = packed & 0xFFFFFFFF
                maximum_ns = ((packed >> 32) & 0xFFFF) * 32
                samples.append((sm, count, total_ns, maximum_ns))
            if not samples:
                raise RuntimeError(
                    f"aggregate FFN profile category {label!r} was not recorded"
                )
            total_occurrences = sum(sample[1] for sample in samples)
            total_ns = sum(sample[2] for sample in samples)
            critical = max(samples, key=lambda sample: sample[2] / sample[1])
            print(
                "DSV4_FFN_AGGREGATE_TIME "
                f"name={label} active_sms={len(samples)} "
                f"occurrences={total_occurrences} "
                f"grid_mean_us={total_ns / total_occurrences / 1.0e3:.3f} "
                f"slowest_sm={critical[0]} "
                f"slowest_sm_mean_us={critical[2] / critical[1] / 1.0e3:.3f} "
                f"max_occurrence_us={max(sample[3] for sample in samples) / 1.0e3:.3f}",
                flush=True,
            )
        start_frontier = max(int(value) for value in profile[:, 0])
        end_frontier = max(int(value) for value in profile[:, 1])
        print(
            "DSV4_FFN_AGGREGATE_SUMMARY "
            f"categories={len(self.ffn_aggregate_used)} "
            f"internal_span_ms={(end_frontier - start_frontier) / 1.0e6:.6f} "
            f"sample_index={sample_index if sample_index is not None else -1} "
            f"sample_cuda_ms={sample_cuda_ms if sample_cuda_ms is not None else -1.0:.6f}",
            flush=True,
        )

    def report_phase_aggregate_profile(
        self,
        profile: torch.Tensor | None = None,
        *,
        sample_index: int | None = None,
        sample_cuda_ms: float | None = None,
    ) -> None:
        if not self.args.profile_phase_aggregate:
            return
        if profile is None:
            profile = self.launcher.profile.cpu()
        magic = 0x4454524B50524631
        if any(int(value) != magic for value in profile[:, 127]):
            raise RuntimeError(
                "aggregate phase profiling requires aggregate_profile=1"
            )
        for label, (_, aggregate_event) in self.phase_aggregate_events.items():
            samples = []
            for sm, value in enumerate(profile[:, aggregate_event]):
                packed = int(value)
                count = (packed >> 48) & 0xFFFF
                if count == 0:
                    continue
                total_ns = packed & 0xFFFFFFFF
                maximum_ns = ((packed >> 32) & 0xFFFF) * 32
                samples.append((sm, count, total_ns, maximum_ns))
            if not samples:
                raise RuntimeError(f"phase profile {label!r} was not recorded")
            critical = max(samples, key=lambda sample: sample[2])
            print(
                "DSV4_PHASE_AGGREGATE_TIME "
                f"name={label} active_sms={len(samples)} "
                f"layers={critical[1]} slowest_sm={critical[0]} "
                f"slowest_sm_total_ms={critical[2] / 1.0e6:.6f} "
                f"slowest_sm_mean_us={critical[2] / critical[1] / 1.0e3:.3f} "
                f"max_occurrence_us={max(sample[3] for sample in samples) / 1.0e3:.3f}",
                flush=True,
            )
        start_frontier = max(int(value) for value in profile[:, 0])
        end_frontier = max(int(value) for value in profile[:, 1])
        print(
            "DSV4_PHASE_AGGREGATE_SUMMARY "
            f"internal_span_ms={(end_frontier - start_frontier) / 1.0e6:.6f} "
            f"sample_index={sample_index if sample_index is not None else -1} "
            f"sample_cuda_ms={sample_cuda_ms if sample_cuda_ms is not None else -1.0:.6f}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--mxfp-ffn-root",
        default=None,
        help=(
            "directory containing offline layer-XXX.safetensors MXFP FFN "
            "images; defaults beneath the checkpoint"
        ),
    )
    parser.add_argument("--layers", type=int, choices=(1, 2, 43), default=1)
    parser.add_argument(
        "--repeat-same-layer",
        action="store_true",
        help=(
            "with --layers=2, reuse layer-0 tensor addresses in both loop "
            "iterations for a working-set/orchestration diagnostic"
        ),
    )
    parser.add_argument(
        "--unroll-two-layers",
        action="store_true",
        help=(
            "diagnostically duplicate the same layer-0 body without LOOPC, "
            "LOOPM, or LDU barrier reload; requires --layers=2 and "
            "--repeat-same-layer"
        ),
    )
    parser.add_argument(
        "--single-layer-id",
        type=int,
        default=0,
        help="checkpoint layer to use when --layers=1",
    )
    parser.add_argument(
        "--two-layer-start-id",
        type=int,
        default=0,
        help=(
            "first checkpoint layer for a two-layer orchestration diagnostic; "
            "the second layer is adjacent unless --repeat-same-layer is set"
        ),
    )
    parser.add_argument(
        "--disable-cross-layer-hc-fusion",
        action="store_true",
        help="diagnostically restore the standalone HCA-post to CSA-pre boundary",
    )
    parser.add_argument(
        "--loopback-hc-fusion",
        action="store_true",
        help="diagnostically fuse layer-2/CSA post into each following HCA",
    )
    parser.add_argument(
        "--two-layer-pair-repeats",
        type=int,
        default=1,
        help=(
            "repeat the selected two-layer body while reusing its tensors; "
            "diagnoses loop/orchestration scaling independently of working set"
        ),
    )
    parser.add_argument("--token-id", type=int, default=791)
    parser.add_argument(
        "--context-length",
        type=int,
        default=1,
        help=(
            "timed decode context in [1,1024]; contexts above one use a "
            "deterministic resident prefix while the current KV/compressed "
            "rows are produced inside the launch"
        ),
    )
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument(
        "--bf16-head",
        action="store_true",
        help="retain the checkpoint BF16 vocabulary head instead of offline FP8",
    )
    parser.add_argument("--sms", type=int, default=152)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument(
        "--cold-l2-scrub-mib",
        type=int,
        default=0,
        help="evict cached data before each out-of-interval timed launch",
    )
    parser.add_argument("--expected-token-id", type=int)
    parser.add_argument(
        "--allow-token-variation",
        action="store_true",
        help=(
            "diagnostically retain all numerical samples when a truncated "
            "model crosses a near-tied argmax"
        ),
    )
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
        default="splitk",
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
        "--qkv-projection-schedule",
        choices=("overlap", "q-first"),
        default="overlap",
        help=(
            "statically release Q-a and KV together or attach KV's load "
            "command to Q-a's direct completion dependency"
        ),
    )
    parser.add_argument(
        "--projection-reset-sms",
        type=int,
        default=0,
        help=(
            "statically shard the split-K projection reset across this many "
            "SMs; zero retains the full-grid diagnostic baseline"
        ),
    )
    parser.add_argument(
        "--projection-reset-position",
        choices=("after-input", "layer-first"),
        default="after-input",
        help=(
            "queue the independent split-K output reset after attention "
            "input production or first in the loop-local layer body"
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
        "--validate-each-launch",
        action="store_true",
        help=(
            "diagnostically validate every emitted FP8-head token against "
            "resident logits and the independent BF16 head"
        ),
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
        "--profile-ffn-aggregate",
        action="store_true",
        help=(
            "passively aggregate per-SM FFN stage durations across the full "
            "resident loop without adding dependency barriers"
        ),
    )
    parser.add_argument(
        "--profile-phase-aggregate",
        action="store_true",
        help="passively aggregate whole attention and FFN spans",
    )
    parser.add_argument(
        "--profile-attention-detail",
        action="store_true",
        help="report native split-attention task timestamps without step wrappers",
    )
    parser.add_argument(
        "--profile-mxfp-ffn-detail",
        action="store_true",
        help="report allocator/LDU/compute timing for the resident MXFP FFN",
    )
    parser.add_argument(
        "--profile-mxfp-ffn-basic",
        action="store_true",
        help=(
            "report the production resident-FFN timestamps already emitted "
            "in profile events 2--5"
        ),
    )
    parser.add_argument(
        "--profile-fp8-coupled-detail",
        action="store_true",
        help=(
            "decompose selected-layer Q-a LDU issue timing and barrier waits; "
            "requires a one- or two-layer run and the FP8 coupled detail runtime"
        ),
    )
    parser.add_argument(
        "--profile-step-start",
        type=int,
        default=0,
        help="first queued layer step included by --profile-steps",
    )
    parser.add_argument(
        "--profile-step-frontiers",
        action="store_true",
        help=(
            "augment a bounded step window with absolute per-SM begin, ready, "
            "and completion frontiers"
        ),
    )
    parser.add_argument(
        "--profile-loopback-boundary",
        action="store_true",
        help=(
            "with the first HCA step window, also record the immediately "
            "preceding CSA completion frontier"
        ),
    )
    parser.add_argument(
        "--profile-step-count",
        type=int,
        default=(
            runtime_config.reload_profile_event_base
            - STEP_PROFILE_EVENT_BASE
        ),
        help="number of queued layer steps included by --profile-steps",
    )
    parser.add_argument(
        "--profile-step-family",
        choices=("last", "hca"),
        default="last",
        help="select the production family whose queued steps receive timestamps",
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
    if args.layers != 2 and args.two_layer_start_id != 0:
        parser.error("two-layer-start-id is only valid with --layers=2")
    if args.layers != 2 and args.two_layer_pair_repeats != 1:
        parser.error("two-layer-pair-repeats is only valid with --layers=2")
    if args.loopback_hc_fusion and (
        args.layers != cfg.num_layers
        or args.disable_cross_layer_hc_fusion
    ):
        parser.error(
            "--loopback-hc-fusion requires the full model and forward fusion"
        )
    if not 1 <= args.two_layer_pair_repeats <= 20:
        parser.error("two-layer-pair-repeats must be in [1,20]")
    if args.layers == 2:
        max_start = cfg.num_layers - (1 if args.repeat_same_layer else 2)
        if not 0 <= args.two_layer_start_id <= max_start:
            parser.error("two-layer-start-id is outside the transformer")
    if args.repeat_same_layer and args.layers != 2:
        parser.error("--repeat-same-layer requires --layers=2")
    if args.unroll_two_layers and not (
        args.layers == 2
        and args.repeat_same_layer
        and args.two_layer_pair_repeats == 1
    ):
        parser.error(
            "--unroll-two-layers requires --layers=2, --repeat-same-layer, "
            "and --two-layer-pair-repeats=1"
        )
    if not 1 <= args.context_length <= 1024:
        parser.error("context-length must be in [1,1024]")
    if not 1 <= args.vocab_size <= cfg.vocab_size:
        parser.error("vocab-size must be in [1,129280]")
    if args.sms <= 0 or args.iterations <= 0 or args.warmup < 0:
        parser.error("sms/iterations must be positive and warmup non-negative")
    if not 0 <= args.projection_reset_sms <= args.sms:
        parser.error("projection-reset-sms must be zero or within the resident grid")
    if args.resident_reserve_gib < 0:
        parser.error("resident-reserve-gib must be non-negative")
    if args.cold_l2_scrub_mib < 0:
        parser.error("cold-l2-scrub-mib must be non-negative")
    profile_modes = sum(
        (
            args.profile_layers,
            args.profile_stages,
            args.profile_steps,
            args.profile_ffn_aggregate,
            args.profile_phase_aggregate,
            args.profile_attention_detail,
            args.profile_mxfp_ffn_basic,
            args.profile_mxfp_ffn_detail,
        )
    )
    if profile_modes > 1:
        parser.error("profiling modes are mutually exclusive")
    if args.profile_preattention_only and not args.profile_stages:
        parser.error("--profile-preattention-only requires --profile-stages")
    if args.profile_stages and args.layers not in (1, 2):
        parser.error("stage profiling requires --layers 1 or 2")
    if args.profile_attention_detail and args.layers not in (1, 2):
        parser.error(
            "attention detail profiling requires --layers 1 or 2"
        )
    if args.profile_mxfp_ffn_detail and args.layers not in (1, 2, 43):
        parser.error("MXFP FFN detail profiling supports 1, 2, or 43 layers")
    if args.profile_mxfp_ffn_basic and args.layers != 1:
        parser.error("production MXFP FFN basic profiling requires --layers 1")
    if args.profile_steps and args.layers not in (1, 2, 43):
        parser.error(
            "step profiling supports one layer or the final logical family "
            "of a repeated image"
        )
    if args.profile_fp8_coupled_detail and (
        args.layers not in (1, 2)
        or not args.profile_steps
        or not (
            args.profile_step_start
            <= 4
            < args.profile_step_start + args.profile_step_count
        )
    ):
        parser.error(
            "--profile-fp8-coupled-detail requires one or two layers and a "
            "--profile-steps window containing step 4"
        )
    step_capacity = (
        runtime_config.reload_profile_event_base
        - STEP_PROFILE_EVENT_BASE
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
    if args.profile_step_frontiers and not args.profile_steps:
        parser.error("--profile-step-frontiers requires --profile-steps")
    if args.profile_loopback_boundary and not (
        args.profile_steps
        and args.profile_step_frontiers
        and args.profile_step_family == "hca"
        and args.profile_step_start == 0
        and args.profile_step_count <= (
            25 - LOOPBACK_PROFILE_FRONTIER_BASE
        )
        and args.layers == cfg.num_layers
        and args.loopback_hc_fusion
    ):
        parser.error(
            "--profile-loopback-boundary requires the full loopback image, "
            "HCA step frontiers starting at zero, and a non-overlapping window"
        )
    if args.profile_step_frontiers and args.profile_fp8_coupled_detail:
        parser.error(
            "--profile-step-frontiers and --profile-fp8-coupled-detail "
            "use the same begin-event range"
        )
    step_frontier_capacity = (
        STEP_PROFILE_EVENT_BASE - STEP_PROFILE_FRONTIER_BASE
    )
    if (
        args.profile_step_frontiers
        and args.profile_step_count > step_frontier_capacity
    ):
        parser.error(
            "--profile-step-frontiers supports at most "
            f"{step_frontier_capacity} steps"
        )
    if args.profile_all_samples and not profile_modes:
        parser.error("--profile-all-samples requires a profiling mode")

    device = torch.device("cuda")
    build_started = time.monotonic()
    flow = ResidentOneLaunchDecode(args, device)
    dump_sm = os.environ.get("DAE_DUMP_MEMORY_SM")
    if dump_sm is not None:
        flow.launcher.prepare_launch()
        for segment_index, segment in enumerate(
            getattr(flow.program, "segments", (flow.program,))
        ):
            for stage_index, stage in enumerate(segment.stages):
                print(
                    "DSV4_STAGE_PLAN "
                    f"segment={segment_index} stage={stage_index} "
                    f"name={stage.name} base_sm={stage.base_sm} "
                    f"num_sms={stage.num_sms} "
                    f"wait_previous={str(stage.wait_for_previous).lower()} "
                    f"wait_group={stage.wait_group or '-'} "
                    f"release_group={stage.release_group or '-'} "
                    f"prefetch={str(stage.prefetch_before_wait).lower()}",
                    flush=True,
                )
        for sm in (int(value) for value in dump_sm.split(",")):
            for pc, inst in enumerate(flow.launcher.builder[sm].built_cinsts):
                print(
                    "DSV4_COMPUTE_INST "
                    f"sm={sm} pc={pc} op={inst.compute_operator_name()} "
                    f"args={inst.args}",
                    flush=True,
                )
            for pc, inst in enumerate(flow.launcher.builder[sm].built_minsts):
                address = sum(int(coord) << (16 * index)
                              for index, coord in enumerate(inst.cords))
                print(
                    "DSV4_MEMORY_INST "
                    f"sm={sm} pc={pc} opcode=0x{inst.opcode:04x} "
                    f"slot={inst.num_slots & 0x3f} "
                    f"bar={inst.num_slots >> 6} arg={inst.arg} "
                    f"size={inst.size} address=0x{address:x} "
                    f"annotation={inst.annotation}",
                    flush=True,
                )
        return
    torch.cuda.synchronize(device)
    build_seconds = time.monotonic() - build_started
    prime_token, prime_ms, prime_logits = flow.run_once()
    flow.validate_compact_head(
        prime_token, require_reference=not args.validate_each_launch
    )
    if flow.compact_head and args.validate_each_launch:
        flow.head_norm_oracle = (
            flow.head_norm[0] if flow.bf16_umma_head else flow.head_norm
        ).clone()
    if flow.fp8_head and args.validate_each_launch:
        flow.fp8_head_activation_oracle = flow.head_input_native_fp8.clone()
    repeat_state_oracle = (
        flow.capture_repeat_state() if args.validate_each_launch else None
    )
    if args.diagnose_projections:
        flow.report_projection_diagnostics()
    if args.expected_token_id is not None and prime_token != args.expected_token_id:
        raise AssertionError(
            f"prime launch emitted token {prime_token}, "
            f"expected {args.expected_token_id}"
        )
    if prime_logits.numel():
        logit_digest = hashlib.sha256(
            prime_logits.contiguous().numpy().tobytes()
        ).hexdigest()[:16]
        prime_top_values, prime_top_indices = torch.topk(prime_logits, 2)
        logit_summary = (
            f"logit_min={float(prime_logits.min().item()):.6f} "
            f"logit_max={float(prime_logits.max().item()):.6f} "
            f"logit_sha256_64={logit_digest} "
            f"top1={int(prime_top_indices[0].item())}:"
            f"{float(prime_top_values[0].item()):.6f} "
            f"top2={int(prime_top_indices[1].item())}:"
            f"{float(prime_top_values[1].item()):.6f} "
            f"top_margin={float((prime_top_values[0] - prime_top_values[1]).item()):.6f}"
        )
    else:
        logit_summary = (
            "logits=bf16_umma_argmax"
            if flow.bf16_umma_head
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
    device_frontier_timings = []
    repeat_logit_max_abs = []
    repeat_logit_mean_abs = []
    profile_samples = []
    reference_token = None
    logits = None
    for iteration in range(args.iterations):
        token, elapsed_ms, logits = flow.run_once()
        if repeat_state_oracle is not None:
            flow.report_repeat_state(repeat_state_oracle, iteration)
        if args.validate_each_launch:
            flow.validate_compact_head(token)
        timings.append(elapsed_ms)
        device_frontier_ms = flow.device_frontier_ms()
        device_frontier_timings.append(device_frontier_ms)
        if logits.numel() and prime_logits.numel():
            repeat_delta = (logits - prime_logits).abs()
            repeat_logit_max_abs.append(float(repeat_delta.max().item()))
            repeat_logit_mean_abs.append(float(repeat_delta.mean().item()))
        if profile_modes:
            profile_samples.append(flow.launcher.profile.cpu().clone())
        print(
            "DSV4_ONE_LAUNCH_SAMPLE "
            f"iteration={iteration} elapsed_ms={elapsed_ms:.6f} "
            f"device_frontier_ms={device_frontier_ms:.6f} "
            f"output_token={token}",
            flush=True,
        )
        if reference_token is None:
            reference_token = token
        elif token != reference_token:
            if logits.numel():
                top_values, top_indices = torch.topk(logits, 2)
                top_summary = (
                    f", current_top1={int(top_indices[0].item())}:"
                    f"{float(top_values[0].item()):.6f}"
                    f", current_top2={int(top_indices[1].item())}:"
                    f"{float(top_values[1].item()):.6f}"
                )
            else:
                top_summary = ""
            message = (
                "one-launch checkpoint token is not repeatable: "
                f"reference={reference_token}, current={token}{top_summary}"
            )
            if args.allow_token_variation:
                print(
                    "DSV4_TOKEN_VARIATION status=DIAGNOSTIC " + message,
                    flush=True,
                )
            else:
                raise AssertionError(message)
    if args.expected_token_id is not None and reference_token != args.expected_token_id:
        raise AssertionError(
            f"checkpoint emitted token {reference_token}, "
            f"expected {args.expected_token_id}"
        )
    # In diagnostic BF16-reduction mode the resident buffers belong to the
    # final timed launch, not the first token selected as the repeatability
    # reference.  Validate the token that corresponds to those buffers.
    compact_validation_token = (
        token if args.allow_token_variation else reference_token
    )
    flow.validate_compact_head(compact_validation_token)
    if reference_token != prime_token:
        if not args.allow_token_variation:
            raise AssertionError(
                "one-launch checkpoint token changed between prime and timed "
                f"launches: prime={prime_token}, timed={reference_token}"
            )
    assert logits is not None
    if profile_modes:
        if args.profile_mxfp_ffn_detail and len(profile_samples) > 1:
            base = runtime_config.reload_profile_event_base
            task0_finishes: dict[int, list[int]] = {
                vcore: [] for vcore in range(args.sms)
            }
            task1_finishes: dict[int, list[int]] = {
                vcore: [] for vcore in range(args.sms)
            }
            for profile in profile_samples:
                sample_tasks = []
                for vcore in range(args.sms):
                    down_origin = int(profile[vcore, base + 12])
                    packed_begins = int(profile[vcore, base + 31])
                    packed_phases = [
                        int(profile[vcore, base + offset])
                        for offset in range(27, 31)
                    ]
                    for task_order in range(2):
                        phases = [
                            (value >> (32 * task_order)) & 0xFFFFFFFF
                            for value in packed_phases
                        ]
                        if task_order == 1 and not any(phases):
                            continue
                        begin = down_origin + (
                            (packed_begins >> (32 * task_order)) & 0xFFFFFFFF
                        )
                        sample_tasks.append(
                            (vcore, task_order, begin + sum(phases))
                        )
                origin = min(task[2] for task in sample_tasks)
                for vcore, task_order, finish in sample_tasks:
                    destination = (
                        task0_finishes if task_order == 0 else task1_finishes
                    )
                    destination[vcore].append(finish - origin)
            late_second = sorted(
                (
                    (statistics.median(samples), vcore)
                    for vcore, samples in task1_finishes.items()
                    if samples
                ),
                reverse=True,
            )[:24]
            late_first = sorted(
                (
                    (statistics.median(samples), vcore)
                    for vcore, samples in task0_finishes.items()
                    if samples
                ),
                reverse=True,
            )[:24]
            no_second = [
                vcore for vcore, samples in task1_finishes.items() if not samples
            ]
            print(
                "DSV4_MXFP_FFN_DOWN_PLACEMENT_AGGREGATE samples="
                f"{len(profile_samples)} no_second="
                + ",".join(
                    f"{vcore}:"
                    f"{statistics.median(task0_finishes[vcore]) / 1.0e3:.3f}"
                    for vcore in no_second
                )
                + " latest_second="
                + ",".join(
                    f"{vcore}:{finish / 1.0e3:.3f}"
                    for finish, vcore in late_second
                )
                + " latest_first="
                + ",".join(
                    f"{vcore}:{finish / 1.0e3:.3f}"
                    for finish, vcore in late_first
                ),
                flush=True,
            )
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
            if args.profile_fp8_coupled_detail:
                reporter = flow.report_fp8_coupled_detail_profile
            elif args.profile_layers:
                reporter = flow.report_layer_profile
            elif args.profile_stages:
                reporter = flow.report_stage_profile
            elif args.profile_ffn_aggregate:
                reporter = flow.report_ffn_aggregate_profile
            elif args.profile_phase_aggregate:
                reporter = flow.report_phase_aggregate_profile
            elif args.profile_attention_detail:
                reporter = flow.report_attention_detail_profile
            elif args.profile_mxfp_ffn_basic:
                reporter = flow.report_mxfp_ffn_basic_profile
            elif args.profile_mxfp_ffn_detail:
                reporter = flow.report_mxfp_ffn_detail_profile
            else:
                reporter = flow.report_step_profile
            reporter(
                profile_samples[sample_index],
                sample_index=sample_index,
                sample_cuda_ms=timings[sample_index],
            )
    repeat_logit_summary = (
        "repeat_logit_max_abs="
        f"{max(repeat_logit_max_abs):.6f} "
        "repeat_logit_mean_abs="
        f"{statistics.mean(repeat_logit_mean_abs):.6f}"
        if repeat_logit_max_abs
        else (
            "repeat_logits=bf16_umma_argmax"
            if flow.bf16_umma_head
            else "repeat_logits=fp8_argmax"
        )
    )
    print(
        "DSV4_ONE_LAUNCH_DECODE status=PASS model_launches=1 gpu=1 "
        f"layers={args.layers} token_id={args.token_id} "
        f"context={args.context_length} position={args.context_length - 1} "
        f"attention={args.attention_mode} "
        "ffn=mxfp4_mxfp8_routed_resident "
        f"index_selection={args.index_selection_mode} "
        f"gated_pool={args.gated_pool_mode} "
        f"prefix_cache={'current_token' if args.context_length == 1 else 'deterministic_seeded'} "
        f"vocab={args.vocab_size} output_token={reference_token} "
        f"build_s={build_seconds:.3f} min_ms={min(timings):.6f} "
        f"median_ms={statistics.median(timings):.6f} "
        f"max_ms={max(timings):.6f} "
        f"device_frontier_min_ms={min(device_frontier_timings):.6f} "
        "device_frontier_median_ms="
        f"{statistics.median(device_frontier_timings):.6f} "
        f"device_frontier_max_ms={max(device_frontier_timings):.6f} "
        f"{logit_summary} {repeat_logit_summary}",
        flush=True,
    )


if __name__ == "__main__":
    main()
