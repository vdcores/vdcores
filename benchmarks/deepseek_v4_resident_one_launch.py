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
    apply_partial_rope_512_64,
    deepseek_v4_rope_bank,
    deepseek_v4_rope_table,
    hc_post_reference,
    hc_pre_reference,
    pack_gated_pool_history,
    sparse_attention_512_reference,
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
    LduReloadBarriers,
    MemoryInstruction,
    TmaLoad1D,
    TmaStore1D,
    TmaTensor,
)
from dae.routing import IndexedLoadTable
from dae.runtime import config as runtime_config
from dae.schedule import (
    LayerStateSchedule,
    LayeredSchedule,
    SchedDsv4AttentionContext1Fp8Sm100,
    SchedLayeredDsv4AttentionContext1Fp8Sm100,
    SchedArgmaxSmemPartial,
    SchedArgmaxSmemReduce,
    SchedDsv4AttentionSplit64UmmaSm100,
    SchedLayeredDsv4AttentionSplit64UmmaSm100,
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
    SchedDsv4CompressorStateStore,
    SchedDsv4GatedPoolRmsRope,
    SchedDsv4GatedPoolPacked8Shard128,
    SchedDsv4GatedPoolPacked8HistoryState,
    SchedDsv4GatedPoolTailRmsPartial,
    SchedDsv4Hadamard,
    SchedDsv4HcHeadRms,
    SchedDsv4HcPost,
    SchedDsv4HcPreRms,
    SchedDsv4IndexScore,
    SchedDsv4IndexedGather512,
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
HC_GLOBAL_RECORD_WAIT_BEGIN_EVENT = 50
HC_GLOBAL_RECORD_WAIT_VALUE_EVENT = 51
HC_GLOBAL_RECORD_WAIT_END_EVENT = 52
HC_GLOBAL_RECORD_COMMAND_END_EVENT = 53
HC_GLOBAL_RESIDENT_COMPUTE_DONE_EVENT = 54
HC_GLOBAL_RAW_PREVIOUS_VALUE_EVENT = 58
HC_GLOBAL_RELOAD_BEGIN_EVENT = 59
HC_GLOBAL_RELOAD_VALUE_EVENT = 60
HC_GLOBAL_RELOAD_READY_EVENT = 61
HC_GLOBAL_RELOAD_STORE_EVENT = 62
HC_GLOBAL_RELOAD_END_EVENT = 63
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
    join_completion: bool = False


class ResidentOneLaunchDecode:
    def __init__(
        self,
        args: argparse.Namespace,
        device: torch.device,
        *,
        weight_source: "ResidentOneLaunchDecode | None" = None,
        live_state=None,
        dynamic_max_position: int | None = None,
        dynamic_variant: str | None = None,
    ):
        self.args = args
        self.device = device
        self.weight_source = weight_source
        self.live_state = live_state
        self.config = DeepSeekV4FlashConfig()
        self.dynamic_max_position = (
            None
            if dynamic_max_position is None
            else int(dynamic_max_position)
        )
        self.dynamic_variant = dynamic_variant
        self.dynamic_position = self.dynamic_max_position is not None
        if self.dynamic_position:
            if self.live_state is None:
                raise ValueError("dynamic decode requires persistent live state")
            if self.dynamic_variant not in {
                "context1",
                "normal",
                "csa_first",
                "csa_short",
                "csa",
                "hca",
                "indexed_normal",
                "indexed_csa",
                "indexed_hca",
            }:
                raise ValueError("unknown reusable decode structural variant")
            if not args.context_length - 1 <= self.dynamic_max_position:
                raise ValueError("dynamic maximum precedes its template position")
        elif self.dynamic_variant is not None:
            raise ValueError("a dynamic variant requires dynamic_max_position")
        self.position_counter_reg = 2 if self.dynamic_position else None
        self._dynamic_store_rules: list[tuple[torch.Tensor, tuple]] = []
        self._dynamic_position_updates: dict[tuple, tuple] = {}
        self._active_dynamic_position: int | None = None
        if args.stop_after_layer is not None:
            self.layer_ids = tuple(range(args.stop_after_layer + 1))
        elif args.layers == 1:
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
        if self.weight_source is not None:
            self.launcher.tma_descriptor_cache = (
                self.weight_source.launcher.tma_descriptor_cache
            )
        self.checkpoint = self._load_checkpoint()
        self.families = self._families()
        self._hash_rows: dict[int, torch.Tensor] = {}
        if self.weight_source is None:
            self._fused_bf16_weight_cache: dict[
                tuple, tuple[torch.Tensor, ...]
            ] = {}
            self._fused_hc_projection_cache: dict[
                tuple, tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]
            ] = {}
            self._derived_tensor_cache: dict[tuple, object] = {}
            self._live_workspace_cache: dict[tuple, object] = {}
        else:
            self._fused_bf16_weight_cache = (
                self.weight_source._fused_bf16_weight_cache
            )
            self._fused_hc_projection_cache = (
                self.weight_source._fused_hc_projection_cache
            )
            self._derived_tensor_cache = (
                self.weight_source._derived_tensor_cache
            )
            self._live_workspace_cache = (
                self.weight_source._live_workspace_cache
            )
        self._allocate_state()
        self._reuse_live_workspace()
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
        self.resident_rope_packed = None
        self._dynamic_rope_banks = None
        if self.dynamic_position:
            self.resident_rope_packed = self._dynamic_arena(
                ("rope_record", self.dynamic_variant),
                (len(self.resident_rope_tables), 32, 2),
                dtype=torch.float32,
            )
            banks = []
            for compressed in (False, True):
                cache_key = (
                    "dynamic_rope_bank",
                    self.dynamic_max_position,
                    compressed,
                )
                bank = self._derived_tensor_cache.get(cache_key)
                if bank is None:
                    bank = deepseek_v4_rope_bank(
                        self.dynamic_max_position + 1,
                        compressed=compressed,
                        config=self.config,
                        device="cpu",
                    )
                    bank = bank.to(self.device)
                    self._derived_tensor_cache[cache_key] = bank
                banks.append(bank)
            self._dynamic_rope_banks = tuple(banks)
        self.family_stages = {
            family.representative: self._build_family(family)
            for family in self.families
        }
        # Keep prefix correctness runs identical to the production body for
        # every complete HCA->CSA pair.  An odd terminal HCA remains a
        # standalone tail because its fused post is owned by the next (omitted)
        # CSA layer.  This changes no production command and adds no diagnostic
        # copy stage before the selected layer boundary.
        cross_layer_hc_pair = (
            args.layers == 43
            and (
                args.stop_after_layer is None
                or args.stop_after_layer >= 4
            )
        ) or (
            args.stop_after_layer is None
            and args.layers == 2
            and args.two_layer_start_id == 3
            and not args.repeat_same_layer
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
            if args.stop_after_cross_layer_hc_write:
                next_stages = self.family_stages[
                    next_family.representative
                ]
                writer_index = next(
                    index
                    for index, stage in enumerate(next_stages)
                    if stage.name
                    == "ffn.hc_post_next_attn.hc_project"
                )
                del next_stages[writer_index + 1 :]
        if args.stop_after_stage is not None:
            stages = self.family_stages[self.families[0].representative]
            if args.stop_after_stage >= len(stages):
                raise ValueError(
                    "stop-after-stage is outside the selected layer body: "
                    f"stage={args.stop_after_stage} count={len(stages)}"
                )
            del stages[args.stop_after_stage + 1 :]
        if (
            args.stop_after_layer is None
            and not args.stop_after_cross_layer_hc_write
            and not args.omit_head
        ):
            self.head_stages = self._build_head()
        else:
            # Prefix correctness and the cross-layer writer diagnostic read
            # their terminal tensors directly from HBM.  Neither needs a
            # copy task or diagnostic head operator.
            self.fp8_head = False
            self.bf16_umma_head = False
            self.compact_head = False
            self.head_stages = []
            self.logits = torch.empty(0, dtype=torch.bfloat16, device=device)
        if args.loopback_hc_fusion:
            terminal_hca_family = (
                self.families[4]
                if args.stop_after_layer is not None
                and args.stop_after_layer % 2 == 1
                else None
            )
            self._apply_loopback_hc_fusion(
                self.families[1],
                self.families[2],
                self.families[3],
                terminal_hca_family,
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
        if self.weight_source is not None:
            source = self.weight_source
            if source.device != self.device or source.layer_ids != self.layer_ids:
                raise ValueError(
                    "a shared resident weight source must use the same device "
                    "and transformer layers"
                )
            self.mxfp_ffn_root = source.mxfp_ffn_root
            self.mxfp_ffn_images = source.mxfp_ffn_images
            if hasattr(source, "_mxfp_ffn_stacked_storage"):
                self._mxfp_ffn_stacked_storage = (
                    source._mxfp_ffn_stacked_storage
                )
            print(
                "DSV4_ONE_LAUNCH_RESIDENT status=REUSED "
                f"layers={len(self.layer_ids)}",
                flush=True,
            )
            return source.checkpoint
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

    def _reuse_live_workspace(self) -> None:
        """Alias serial token scratch while retaining persistent decode state.

        A prepared live-decode image is executed only after the preceding
        token has completed.  Its large projection and reduction arenas can
        therefore use the same storage as every other reusable flow.  Cache
        rows, compressor histories/destinations, attention indices, and RoPE
        tables remain private views because their addresses or values encode
        the absolute decode position.
        """

        if self.live_state is None:
            return
        position_owned = {
            "_derived_tensor_cache",
            "_fused_bf16_weight_cache",
            "_fused_hc_projection_cache",
            "_hash_rows",
            "_live_workspace_cache",
            "attention_cache",
            "attention_indices_by_kind",
            "attention_plans",
            "attention_pool_history_packed",
            "attention_pool_history_scores",
            "attention_pool_history_values",
            "compress_rope",
            "compressed_output_rope",
            "current_compressed_rows",
            "current_kv_rows",
            "decode_position",
            "index_cache",
            "index_pool_history_scores",
            "index_pool_history_values",
            "main_rope",
            "mxfp_ffn_images",
        }

        def reuse(value, path):
            if isinstance(value, torch.Tensor):
                if value.device != self.device:
                    return value
                key = (
                    path,
                    value.dtype,
                    tuple(value.shape),
                    tuple(value.stride()),
                    value.storage_offset(),
                )
                existing = self._live_workspace_cache.get(key)
                if existing is None:
                    self._live_workspace_cache[key] = value
                    return value
                return existing
            if isinstance(value, dict):
                return type(value)(
                    (key, reuse(child, (*path, key)))
                    for key, child in value.items()
                )
            if isinstance(value, list):
                return [
                    reuse(child, (*path, index))
                    for index, child in enumerate(value)
                ]
            if isinstance(value, tuple):
                return tuple(
                    reuse(child, (*path, index))
                    for index, child in enumerate(value)
                )
            return value

        for name, value in tuple(vars(self).items()):
            if name in position_owned:
                continue
            setattr(self, name, reuse(value, (name,)))

    def _families(self) -> tuple[LayerFamily, ...]:
        if self.args.stop_after_layer is not None:
            stop = self.args.stop_after_layer
            families = [
                LayerFamily(
                    "prefix.swa_hash",
                    tuple(range(min(stop + 1, 2))),
                    ((0, 1),) if stop >= 1 else (),
                )
            ]
            if stop >= 2:
                families.append(LayerFamily("prefix.csa_hash", (2,)))
            paired_hca = tuple(range(3, stop, 2))
            paired_csa = tuple(range(4, stop + 1, 2))
            if paired_hca:
                families.extend(
                    (
                        LayerFamily(
                            "prefix.hca_score",
                            paired_hca,
                            ((0, 1), (1, 2))
                            if len(paired_hca) > 1
                            else (),
                        ),
                        LayerFamily(
                            "prefix.csa_score",
                            paired_csa,
                            ((0, 1), (1, 2))
                            if len(paired_csa) > 1
                            else (),
                        ),
                    )
                )
            if stop >= 3 and stop % 2 == 1:
                families.append(
                    LayerFamily("prefix.hca_tail_score", (stop,))
                )
            return tuple(families)
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

    def _dynamic_arena(self, key, shape, *, dtype):
        cache_key = ("dynamic_live_arena", key, tuple(shape), dtype)
        arena = self._derived_tensor_cache.get(cache_key)
        if arena is None:
            arena = torch.empty(shape, dtype=dtype, device=self.device)
            self._derived_tensor_cache[cache_key] = arena
        return arena

    def _family_ape_rows(
        self,
        family: LayerFamily,
        suffix: str,
        position_in_group: int,
    ) -> tuple[torch.Tensor, ...]:
        sources = self._family_tensors(family, suffix)
        if not self.dynamic_position:
            return tuple(source[position_in_group] for source in sources)
        bank_key = (
            "dynamic_ape_bank",
            suffix,
            tuple(source.data_ptr() for source in sources),
        )
        bank = self._derived_tensor_cache.get(bank_key)
        if bank is None:
            bank = torch.stack(sources).contiguous()
            self._derived_tensor_cache[bank_key] = bank
        arena = self._dynamic_arena(
            ("ape_row", suffix, tuple(family.layer_ids)),
            (len(sources), bank.shape[-1]),
            dtype=bank.dtype,
        )
        self._dynamic_position_updates[("ape", suffix, family.representative)] = (
            arena,
            bank,
        )
        return tuple(arena.unbind(0))

    def _register_dynamic_store(self, tensor: torch.Tensor, *offsets) -> None:
        if self.dynamic_position:
            self._dynamic_store_rules.append((tensor, tuple(offsets)))

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
        direct_output: bool = False,
        prefetch_before_wait: bool = False,
        prefetch_before_resident_reset: bool = False,
        wait_group_roles: tuple[tuple[str, str], ...] = (),
        release_group_roles: tuple[tuple[str, str], ...] = (),
        join_completion: bool = False,
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
            join_completion=join_completion,
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

    def _activate_live_family_state(
        self, family: LayerFamily
    ) -> tuple[tuple[torch.Tensor, tuple[torch.Tensor, ...]], ...]:
        """Bind one compact family body to its layer-private decode state."""
        if self.live_state is None:
            return ()
        cfg = self.config
        kind = cfg.attention_kind(family.representative)
        plan = self.attention_plans[kind]
        position = self.decode_position
        caches = tuple(
            self.live_state.attention_cache(layer_id, plan.compressed_rows)
            for layer_id in family.layer_ids
        )
        self.attention_cache[kind] = caches[0]
        window_row = 0 if self.dynamic_position else position % cfg.sliding_window
        self.current_kv_rows[kind] = caches[0][window_row : window_row + 1]
        if self.dynamic_position:
            row_bytes = cfg.head_dim * caches[0].element_size()
            self._register_dynamic_store(
                self.current_kv_rows[kind],
                (self.position_counter_reg, row_bytes, 0, 7),
            )
        if plan.should_compress:
            if self.dynamic_position and kind == "csa" and (
                self.dynamic_variant in ("csa_first", "csa_short")
            ):
                compressed_row = 1
                compressed_offsets = (
                    (self.position_counter_reg, row_bytes),
                    (self.position_counter_reg, row_bytes, 2, 0),
                )
            elif self.dynamic_position:
                compressed_row = cfg.sliding_window
                compressed_offsets = ((
                    self.position_counter_reg,
                    row_bytes,
                    2 if kind == "csa" else 7,
                    0,
                ),)
            else:
                compressed_row = cfg.sliding_window + plan.compressed_rows - 1
                compressed_offsets = ()
            self.current_compressed_rows[kind] = caches[0][
                compressed_row : compressed_row + 1
            ]
            if self.dynamic_position:
                self._register_dynamic_store(
                    self.current_compressed_rows[kind],
                    *compressed_offsets,
                )

        groups: list[tuple[torch.Tensor, tuple[torch.Tensor, ...]]] = [
            (caches[0], caches)
        ]
        if kind in ("csa", "hca"):
            input_arena = self.live_compressor_input_arenas[
                family.representative
            ]
            input_rows = tuple(input_arena.unbind(0))
            groups.append((input_rows[0], input_rows))
            projection_arena = self.live_compressor_projection_arenas[
                family.representative
            ]
            projection_rows = tuple(projection_arena.unbind(0))
            groups.append((projection_rows[0], projection_rows))
            if kind == "csa":
                index_projection_arena = (
                    self.live_index_compressor_projection_arenas[
                        family.representative
                    ]
                )
                index_projection_rows = tuple(
                    index_projection_arena.unbind(0)
                )
                groups.append(
                    (index_projection_rows[0], index_projection_rows)
                )
        if kind == "csa":
            index_caches = tuple(
                self.live_state.index_cache(layer_id, plan.compressed_rows)
                for layer_id in family.layer_ids
            )
            self.index_cache = index_caches[0]
            self.current_index_compressed = (
                self.index_cache[:1]
                if self.dynamic_position
                else self.index_cache[-1:]
            )
            if self.dynamic_position and plan.should_compress:
                self._register_dynamic_store(
                    self.current_index_compressed,
                    (
                        self.position_counter_reg,
                        cfg.index_head_dim
                        * self.current_index_compressed.element_size(),
                        2,
                        0,
                    ),
                )
            if plan.should_compress:
                if self.dynamic_position:
                    history_rows = (
                        3 if self.dynamic_variant == "csa_first" else 7
                    )
                    offsets = [
                        self.live_state.layer_offsets[layer_id][1]
                        for layer_id in family.layer_ids
                    ]
                    if offsets != list(range(offsets[0], offsets[0] + len(offsets))):
                        raise ValueError("CSA live-state layers must be contiguous")
                    state_slice = slice(offsets[0], offsets[-1] + 1)
                    dynamic_histories = []
                    for label, storage in (
                        ("attention_values", self.live_state.csa_pool_values),
                        ("attention_scores", self.live_state.csa_pool_scores),
                        ("index_values", self.live_state.index_pool_values),
                        ("index_scores", self.live_state.index_pool_scores),
                    ):
                        width = storage.shape[-1]
                        arena = self._dynamic_arena(
                            ("csa_history", label, tuple(family.layer_ids), history_rows),
                            (len(family.layer_ids), history_rows, width),
                            dtype=storage.dtype,
                        )
                        self._dynamic_position_updates[
                            ("csa_history", label, family.representative)
                        ] = (
                            arena,
                            storage[state_slice],
                            history_rows,
                            4 if self.dynamic_variant == "csa_first" else 0,
                        )
                        rows = tuple(arena.unbind(0))
                        groups.append((rows[0], rows))
                        dynamic_histories.append(rows[0])
                    (
                        self.attention_pool_history_values[kind],
                        self.attention_pool_history_scores[kind],
                        self.index_pool_history_values,
                        self.index_pool_history_scores,
                    ) = dynamic_histories
                else:
                    attention_histories = tuple(
                        self.live_state.csa_pool_history(layer_id, position)
                        for layer_id in family.layer_ids
                    )
                    self.attention_pool_history_values[kind] = (
                        attention_histories[0][0]
                    )
                    self.attention_pool_history_scores[kind] = (
                        attention_histories[0][1]
                    )
                    index_histories = tuple(
                        self.live_state.csa_pool_history(
                            layer_id, position, index=True
                        )
                        for layer_id in family.layer_ids
                    )
                    self.index_pool_history_values = index_histories[0][0]
                    self.index_pool_history_scores = index_histories[0][1]
            attention_storage = tuple(
                self.live_state.csa_pool_storage(layer_id)
                for layer_id in family.layer_ids
            )
            index_storage = tuple(
                self.live_state.csa_pool_storage(layer_id, index=True)
                for layer_id in family.layer_ids
            )
            if plan.compressed_rows:
                groups.append((index_caches[0], index_caches))
            groups.extend(
                (
                    (
                        attention_storage[0][0],
                        tuple(item[0] for item in attention_storage),
                    ),
                    (
                        attention_storage[0][1],
                        tuple(item[1] for item in attention_storage),
                    ),
                    (
                        index_storage[0][0],
                        tuple(item[0] for item in index_storage),
                    ),
                    (
                        index_storage[0][1],
                        tuple(item[1] for item in index_storage),
                    ),
                )
            )
        elif kind == "hca":
            if plan.should_compress:
                histories = tuple(
                    self.live_state.hca_pool_history(layer_id, position)
                    for layer_id in family.layer_ids
                )
                self.attention_pool_history_values[kind] = histories[0][0]
                self.attention_pool_history_scores[kind] = histories[0][1]
            storage = tuple(
                self.live_state.hca_pool_storage(layer_id)
                for layer_id in family.layer_ids
            )
            groups.extend(
                (
                    (storage[0][0], tuple(item[0] for item in storage)),
                    (storage[0][1], tuple(item[1] for item in storage)),
                )
            )
        return tuple(groups)

    def _allocate_state(self) -> None:
        cfg, d = self.config, self.device
        embedding = self._tensor("embed.weight")[self.args.token_id]
        self.initial_residual = embedding.reshape(1, -1).repeat(cfg.hc_mult, 1)
        self.hidden = torch.empty((cfg.hidden_size,), dtype=torch.bfloat16, device=d)
        # Keep the two cross-layer handoff directions in disjoint storage.
        # The 20 FP32 post/comb coefficients must remain live until the next
        # layer consumes them, while that next layer is free to produce its
        # own normalized hidden and coefficients.  Padding this tiny arena is
        # cheaper and clearer than adding a copy or another runtime edge.
        mhc_packed_items = cfg.hidden_size + 40
        mhc_arena_stride = (mhc_packed_items + 63) // 64 * 64
        self.mhc_packed_output_arenas = torch.empty(
            (2, mhc_arena_stride), dtype=torch.bfloat16, device=d
        )
        self.mhc_packed_outputs = self.mhc_packed_output_arenas[
            :, :mhc_packed_items
        ]
        self.norm_hiddens = self.mhc_packed_outputs[:, :cfg.hidden_size]
        self.mhc_output_metadatas = self.mhc_packed_outputs[
            :, cfg.hidden_size:
        ].view(torch.float32)
        self.posts = self.mhc_output_metadatas[:, :4]
        self.combs = self.mhc_output_metadatas[:, 4:].view(2, 4, 4)
        direct_projection_views = {}
        if self.direct_splitk_bf16:
            projection_rows = (
                cfg.q_lora_rank
                + cfg.head_dim
                + cfg.num_heads * cfg.head_dim
                + cfg.index_heads * cfg.index_head_dim
                + cfg.o_groups * cfg.o_lora_rank
            )
            # Keep every BF16 split-K reduction destination in one reset span.
            # The final H elements are o_b's branch row; the following four
            # residual rows remain outside the clear while preserving the
            # contiguous [branch,residual0..3] HC-post input contract.
            self.splitk_output_storage = torch.empty(
                (projection_rows + (1 + cfg.hc_mult) * cfg.hidden_size,),
                dtype=torch.bfloat16,
                device=d,
            )
            projection_arena = self.splitk_output_storage[:projection_rows]
            self.attention_post_input_record = self.splitk_output_storage[
                projection_rows:
            ].view(1 + cfg.hc_mult, cfg.hidden_size)
            self.splitk_output_arena = self.splitk_output_storage[
                : projection_rows + cfg.hidden_size
            ]
            offset = 0

            def direct_view(name, shape):
                nonlocal offset
                elements = math.prod(shape)
                view = projection_arena[offset : offset + elements].view(
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
            self.splitk_output_storage = None
            self.splitk_output_arena = None
            self.attention_post_input_record = torch.empty(
                (1 + cfg.hc_mult, cfg.hidden_size),
                dtype=torch.bfloat16,
                device=d,
            )
        self.branch = self.attention_post_input_record[0]
        self.residual = self.attention_post_input_record[1:]
        self.next_residual = torch.empty_like(self.residual)
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
        if self.dynamic_position and self.dynamic_variant != "context1":
            for kind, layer_id in representatives.items():
                maximum = build_layer_decode_plan(
                    layer_id, self.dynamic_max_position, cfg
                )
                template = self.attention_plans[kind]
                self.attention_plans[kind] = replace(
                    template,
                    compressed_rows=maximum.compressed_rows,
                    compressed_selected=maximum.compressed_selected,
                    requires_index_selection=maximum.requires_index_selection,
                    attention_candidates=maximum.attention_candidates,
                )
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
            valid_window = min(
                cfg.sliding_window,
                (
                    self.dynamic_max_position + 1
                    if self.dynamic_position
                    and self.dynamic_variant != "context1"
                    else self.args.context_length
                ),
            )
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
            window_row = (
                0
                if self.dynamic_position
                else self.decode_position % cfg.sliding_window
            )
            self.current_kv_rows[kind] = cache[window_row : window_row + 1]
            if plan.should_compress:
                compressed_row = (
                    cfg.sliding_window
                    if self.dynamic_position
                    else cfg.sliding_window + plan.compressed_rows - 1
                )
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
                if (
                    plan.compress_ratio != 4
                    or plan.compressed_rows == 1
                    or self.dynamic_variant == "csa_first"
                )
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
        self.live_compressor_projection_arenas = {}
        self.live_index_compressor_projection_arenas = {}
        self.live_compressor_input_arenas = {}
        if self.live_state is not None:
            for family in self.families:
                kind = cfg.attention_kind(family.representative)
                if kind not in ("csa", "hca"):
                    continue
                self.live_compressor_input_arenas[
                    family.representative
                ] = torch.empty(
                    (len(family.layer_ids), cfg.hidden_size),
                    dtype=torch.bfloat16,
                    device=d,
                )
                width = cfg.head_dim * (2 if kind == "csa" else 1)
                self.live_compressor_projection_arenas[
                    family.representative
                ] = torch.zeros(
                    (len(family.layer_ids), 2 * width),
                    dtype=torch.float32,
                    device=d,
                )
                if kind == "csa":
                    self.live_index_compressor_projection_arenas[
                        family.representative
                    ] = torch.zeros(
                        (
                            len(family.layer_ids),
                            4 * cfg.index_head_dim,
                        ),
                        dtype=torch.float32,
                        device=d,
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
        self.attention_gather_workspace = (
            torch.empty(
                (csa_plan.attention_candidates, cfg.head_dim),
                dtype=torch.bfloat16,
                device=d,
            )
            if csa_plan.requires_index_selection
            else None
        )
        if csa_plan.should_compress:
            index_pool_rows = (
                csa_plan.compress_ratio
                if (
                    csa_plan.compressed_rows == 1
                    or self.dynamic_variant == "csa_first"
                )
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
        self.live_hash_rows = None
        if self.live_state is not None:
            if self.layer_ids != tuple(range(cfg.num_layers)):
                raise ValueError("live decode currently requires all 43 layers")
            if self.live_state.config != cfg:
                raise ValueError("live decode state uses a different model config")
            if self.args.context_length > self.live_state.max_seq_len:
                raise ValueError("live decode context exceeds state capacity")
            self.live_hash_rows = torch.zeros(
                (cfg.num_hash_layers, 8), dtype=torch.int32, device=d
            )
            for layer_id in range(cfg.num_hash_layers):
                source = self._tensor(
                    f"layers.{layer_id}.ffn.gate.tid2eid"
                )[self.args.token_id]
                self.live_hash_rows[
                    layer_id, : cfg.experts_per_token
                ].copy_(source.to(torch.int32))

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
        # Down produces an N=8 UMMA tile although only lane zero is the model
        # branch.  Give the other seven lanes padded rows so lane zero can
        # directly share a persistent contiguous [branch, residual0..3]
        # cross-layer record without letting those unused lanes overwrite it.
        # The TMA view remains [8, H], with a five-row physical N stride.
        cross_layer_record_rows = 1 + cfg.hc_mult
        down_output_lanes = 8
        self.mxfp_ffn_output_arenas = torch.empty(
            (
                2,
                (down_output_lanes - 1) * cross_layer_record_rows + 1,
                cfg.hidden_size,
            ),
            dtype=torch.bfloat16,
            device=d,
        )
        self.mhc_cross_layer_input_records = self.mxfp_ffn_output_arenas[
            :, :cross_layer_record_rows
        ]
        self.mhc_boundary_record_snapshot = None
        self.mhc_boundary_coefficients_snapshot = None
        self.mhc_consumed_record_capture = None
        self.mhc_consumed_weight_capture = None
        self.mhc_consumed_coefficient_capture = None
        self.mhc_fused_weight_reference = None
        if self.args.diagnose_cross_layer_hc_boundary:
            self.mhc_boundary_record_snapshot = torch.empty(
                (cross_layer_record_rows, cfg.hidden_size),
                dtype=torch.bfloat16,
                device=d,
            )
            self.mhc_boundary_coefficients_snapshot = torch.empty(
                (20,), dtype=torch.float32, device=d
            )
        if (
            self.args.profile_cross_layer_hc_barrier
            or self.args.stop_after_cross_layer_hc_write
        ):
            self.mhc_consumed_record_capture = torch.full(
                (
                    SchedDsv4Fp32Bf16Gemv.FUSED_TASK_SMS,
                    5,
                    SchedDsv4Fp32Bf16Gemv.FUSED_TILE_HIDDEN,
                ),
                float("nan"),
                dtype=torch.bfloat16,
                device=d,
            )
            self.mhc_consumed_weight_capture = torch.full(
                (
                    SchedDsv4Fp32Bf16Gemv.FUSED_TASK_SMS,
                    SchedDsv4Fp32Bf16Gemv.FUSED_OUTPUTS_PER_TASK,
                    4,
                    SchedDsv4Fp32Bf16Gemv.FUSED_TILE_HIDDEN,
                ),
                float("nan"),
                dtype=torch.float32,
                device=d,
            )
            self.mhc_consumed_coefficient_capture = torch.full(
                (SchedDsv4Fp32Bf16Gemv.FUSED_TASK_SMS, 20),
                float("nan"),
                dtype=torch.float32,
                device=d,
            )
        self.mxfp_ffn_outputs = tuple(
            torch.as_strided(
                arena,
                size=(down_output_lanes, cfg.hidden_size),
                stride=(cross_layer_record_rows * cfg.hidden_size, 1),
            )
            for arena in self.mxfp_ffn_output_arenas
        )
        self.mxfp_ffn_output = self.mxfp_ffn_outputs[1]

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
        self.mxfp_output_tmas = tuple(
            TmaTensor(self.launcher, output).m128n8_output("reduce")
            for output in self.mxfp_ffn_outputs
        )
        self._mxfp_runtime_layers: dict[
            tuple[int, int], MxfpFfnRuntimeLayer
        ] = {}

    @staticmethod
    def _f32_bits(value: float) -> int:
        return struct.unpack("<I", struct.pack("<f", float(value)))[0]

    def _mxfp_output_set(self, layer_id: int) -> int:
        # HCA feeds the following CSA; every CSA (including layer 2) feeds the
        # following HCA.  Keep those two persistent handoff directions in
        # separate arenas so a producer can never overwrite the other edge.
        return int(self.config.attention_kind(layer_id) != "hca")

    def _mhc_outputs(self, layer_id: int):
        output_set = self._mxfp_output_set(layer_id)
        return (
            self.mhc_packed_outputs[output_set],
            self.norm_hiddens[output_set],
            self.mhc_output_metadatas[output_set],
            self.posts[output_set],
            self.combs[output_set],
        )

    def _mxfp_runtime_layer(
        self, layer_id: int, barrier_set: int
    ) -> MxfpFfnRuntimeLayer:
        key = (int(layer_id), int(barrier_set))
        existing = self._mxfp_runtime_layers.get(key)
        if existing is not None:
            return existing
        image = self.mxfp_ffn_images[layer_id]
        output_set = self._mxfp_output_set(layer_id)
        mxfp_output_tma = self.mxfp_output_tmas[output_set]
        mxfp_ffn_output = self.mxfp_ffn_outputs[output_set]
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
                    | (mxfp_output_tma.arg << 16)
                    | (task << 32)
                )
                down_records[record, 4] = (
                    ready_bars[expert * 16]
                    | (zero_ready[output_tile] << 32)
                )
                down_records[record, 5] = self._f32_bits(1.0)
                down_records[record, 6] = mxfp_ffn_output.data_ptr()
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
            derived_tensor_cache=self._derived_tensor_cache,
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
                derived_tensor_cache=self._derived_tensor_cache,
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
            derived_tensor_cache=self._derived_tensor_cache,
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
            derived_tensor_cache=self._derived_tensor_cache,
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
        direct_output: bool = False,
    ) -> Stage:
        weights = self._fused_bf16_weights(family, suffixes)
        rows, k = weights.shape[-2:]
        layer_indexed_output = output.ndim == 2
        if layer_indexed_output:
            if tuple(output.shape) != (len(family.layer_ids), rows):
                raise ValueError(
                    "layered grouped BF16 output must be [layers,fused rows]"
                )
            output_matrix = output.reshape(
                len(family.layer_ids), rows // 128, 128
            )
        else:
            if output.numel() != rows:
                raise ValueError(
                    "grouped BF16 projection output must match fused rows"
                )
            output_matrix = output.reshape(rows // 128, 128)
        weight_tma = TmaTensor(
            self.launcher, weights
        ).wgmma_load(128, 128, Major.K)
        output_reduce_tensor = TmaTensor(self.launcher, output_matrix)
        output_action = "store" if direct_output else "reduce"
        output_reduce = (
            output_reduce_tensor.batched_rowmajor_2d(output_action, 4, 128)
            if layer_indexed_output
            else output_reduce_tensor.rowmajor_2d(output_action, 4, 128)
        )
        schedule = SchedDsv4Bf16GemvGroup4SplitK(
            weights,
            weight_tma,
            source.reshape(-1),
            output_reduce,
            split_k,
            layer_indexed_weight=weights.ndim == 3,
            layer_indexed_output=layer_indexed_output,
            direct_output=direct_output,
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
        (
            mhc_packed_output,
            norm_hidden,
            mhc_output_metadata,
            post,
            comb,
        ) = self._mhc_outputs(family.representative)
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
            norm_hidden,
            post,
            comb,
            residual_square_sum=self.mhc_residual_square_sum,
            packed_metadata=self.mhc_packed_metadata,
            packed_output=mhc_packed_output,
            zero_fp32_output=zero_fp32_output,
        )
        pre = self._layered(pre, family, norm_weights)
        pre_stage = self._stage(f"{branch_name}.hc_pre_rms4096", pre)
        post_stage = self._stage(
            f"{branch_name}.hc_post",
            SchedDsv4HcPost(
                branch,
                residual,
                post,
                comb,
                output_residual,
                launcher=self.launcher,
                packed_coefficients=mhc_output_metadata,
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
        (
            mhc_packed_output,
            norm_hidden,
            mhc_output_metadata,
            post,
            comb,
        ) = self._mhc_outputs(family.representative)
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
            packed_coefficients=mhc_output_metadata,
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
            norm_hidden,
            post,
            comb,
            residual_square_sum=self.mhc_fused_residual_square_sum,
            packed_metadata=self.mhc_fused_metadata,
            packed_output=mhc_packed_output,
            split_metadata_splits=SchedDsv4Fp32Bf16Gemv.FUSED_SPLITS,
        )
        pre = self._layered(pre, family, norm_weights)
        compressor_reuse_wait = (
            (
                (
                    f"{family.name}.attn.compressor.projection.ready",
                    "reuse",
                ),
            )
            if self.live_state is not None
            and self.config.attention_kind(family.representative)
            in ("csa", "hca")
            else ()
        )
        pre_stage = self._stage(
            "ffn.hc_pre_rms4096",
            pre,
            1,
            base_sm=128,
            wait_for_previous=False,
            wait_group_roles=(
                (metadata_ready, "metadata"),
                (residual_ready, "residual"),
            ) + compressor_reuse_wait,
            release_group=ffn_input_ready,
        )
        return tail_stage, [fused_stage, pre_stage]

    def _build_attention(
        self, family: LayerFamily
    ) -> tuple[list[Stage], Stage]:
        cfg = self.config
        layer_id = family.representative
        _, norm_hidden, _, _, _ = self._mhc_outputs(layer_id)
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
        compressor_input_ready = (
            f"{family.name}.attn.compressor.input.ready"
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
        index_q_ready = f"{family.name}.index.q.ready"
        index_selection_ready = f"{family.name}.index.selection.ready"
        attention_gather_ready = f"{family.name}.attn.gather.ready"
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
        live_compressor_state = (
            self.live_state is not None and kind in ("csa", "hca")
        )
        if live_compressor_state and kind == "csa":
            # Both compressor projections read the same normalized hidden.
            # Their STU completion tails therefore publish one shared
            # read-lifetime edge before the FFN mHC pre-task may overwrite it.
            index_compressor_projection_ready = compressor_projection_ready
        if live_compressor_state and not use_grouped_preattention:
            raise ValueError(
                "live decode requires the production grouped pre-attention path"
            )
        project_compressor = plan.should_compress or live_compressor_state
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
                and project_compressor
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
                    norm_hidden,
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
                    norm_hidden,
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
        compressor_source = norm_hidden
        if live_compressor_state:
            compressor_input_arena = self.live_compressor_input_arenas[
                family.representative
            ]
            compressor_source = compressor_input_arena[0]
            stages.append(
                self._stage(
                    "attn.compressor.input_snapshot",
                    SchedCopy(
                        (
                            TmaLoad1D(norm_hidden),
                            TmaStore1D(compressor_source),
                        )
                    ),
                    1,
                    base_sm=self.sms - 1,
                    wait_for_previous=False,
                    wait_group=attention_input_ready,
                    release_group=compressor_input_ready,
                )
            )
        if (
            use_grouped_preattention
            and kind in ("csa", "hca")
            and project_compressor
        ):
            width = cfg.head_dim * (2 if kind == "csa" else 1)
            if live_compressor_state:
                grouped_fused_output = (
                    self.live_compressor_projection_arenas[
                        family.representative
                    ]
                )
                fused_output = grouped_fused_output[0]
                if len(family.layer_ids) == 1:
                    grouped_fused_output = fused_output
            else:
                fused_output = self.compress_fused_projection[: 2 * width]
                grouped_fused_output = fused_output
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
            if live_compressor_state:
                reset = SchedDsv4ZeroFill(
                    self.zero_fill_gate, fused_output
                )
                stages.append(
                    self._stage(
                        "attn.compressor.projection_reset",
                        reset,
                        compressor_sms,
                        base_sm=compressor_base,
                        wait_for_previous=False,
                        wait_group=attention_input_ready,
                        release_group=compressor_input_ready,
                    )
                )
            else:
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
                    compressor_source,
                    grouped_fused_output,
                    split_k=16,
                    base_sm=compressor_projection_base,
                    wait_group=(
                        compressor_input_ready
                        if live_compressor_state
                        else compressor_reset_ready
                    ),
                    release_group=compressor_projection_ready,
                )
            )
        if (
            use_grouped_preattention
            and kind == "csa"
            and project_compressor
        ):
            if live_compressor_state:
                grouped_fused_index_output = (
                    self.live_index_compressor_projection_arenas[
                        family.representative
                    ]
                )
                fused_index_output = grouped_fused_index_output[0]
                if len(family.layer_ids) == 1:
                    grouped_fused_index_output = fused_index_output
            else:
                fused_index_output = (
                    self.index_compress_fused_projection.reshape(-1)
                )
                grouped_fused_index_output = fused_index_output
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
            if live_compressor_state:
                reset = SchedDsv4ZeroFill(
                    self.zero_fill_gate, fused_index_output
                )
                stages.append(
                    self._stage(
                        "index.compressor.projection_reset",
                        reset,
                        index_compressor_sms,
                        base_sm=index_compressor_base,
                        wait_for_previous=False,
                        wait_group=attention_input_ready,
                        release_group=compressor_input_ready,
                    )
                )
            else:
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
                    compressor_source,
                    grouped_fused_index_output,
                    split_k=16,
                    base_sm=index_compressor_base,
                    wait_group=(
                        compressor_input_ready
                        if live_compressor_state
                        else index_compressor_reset_ready
                    ),
                    release_group=index_compressor_projection_ready,
                )
            )
        if live_compressor_state:
            position_in_group = self.decode_position % plan.compress_ratio
            ape_tensors = self._family_ape_rows(
                family, "attn.compressor.ape", position_in_group
            )
            if kind == "csa":
                value_rows = (
                    compress_values[: cfg.head_dim],
                    compress_values[cfg.head_dim : 2 * cfg.head_dim],
                )
                score_rows = (
                    compress_scores[: cfg.head_dim],
                    compress_scores[cfg.head_dim : 2 * cfg.head_dim],
                )
                bias_rows = tuple(
                    tuple(
                        ape[
                            half * cfg.head_dim : (half + 1) * cfg.head_dim
                        ]
                        for ape in ape_tensors
                    )
                    for half in range(2)
                )
                if self.dynamic_position:
                    storage_values, storage_scores = (
                        self.live_state.csa_pool_storage(
                            family.representative
                        )
                    )
                    flat_values = storage_values.reshape(-1, cfg.head_dim)
                    flat_scores = storage_scores.reshape(-1, cfg.head_dim)
                    ordinary_base = 4
                    destinations = (
                        flat_values[8],
                        flat_scores[8],
                        flat_values[ordinary_base],
                        flat_scores[ordinary_base],
                    )
                    row_bytes = cfg.head_dim * 4
                    overlap_offsets = (
                        (self.position_counter_reg, row_bytes, 0, 2),
                        (self.position_counter_reg, -8 * row_bytes, 2, 1),
                    )
                    ordinary_offsets = (
                        (self.position_counter_reg, row_bytes, 0, 2),
                    )
                    ordinary_offsets += ((
                        self.position_counter_reg,
                        8 * row_bytes,
                        2,
                        1,
                    ),)
                    self._register_dynamic_store(
                        destinations[0], *overlap_offsets
                    )
                    self._register_dynamic_store(
                        destinations[1], *overlap_offsets
                    )
                    self._register_dynamic_store(
                        destinations[2], *ordinary_offsets
                    )
                    self._register_dynamic_store(
                        destinations[3], *ordinary_offsets
                    )
                else:
                    destinations = self.live_state.csa_pool_destinations(
                        family.representative, self.decode_position
                    )
                state_store = SchedDsv4CompressorStateStore(
                    value_rows,
                    score_rows,
                    (bias_rows[0][0], bias_rows[1][0]),
                    (destinations[0], destinations[2]),
                    (destinations[1], destinations[3]),
                )
                state_store = self._layered(
                    state_store, family, bias_rows[0], bias_rows[1]
                )
                state_store_sms = 2
            else:
                bias_rows = tuple(
                    ape[: cfg.head_dim]
                    for ape in ape_tensors
                )
                if self.dynamic_position:
                    storage_values, storage_scores = (
                        self.live_state.hca_pool_storage(
                            family.representative
                        )
                    )
                    destination_values = storage_values[0]
                    destination_scores = storage_scores[0]
                    row_bytes = cfg.head_dim * 4
                    offsets = ((
                        self.position_counter_reg,
                        row_bytes,
                        0,
                        7,
                    ),)
                    self._register_dynamic_store(
                        destination_values, *offsets
                    )
                    self._register_dynamic_store(
                        destination_scores, *offsets
                    )
                else:
                    destination_values, destination_scores = (
                        self.live_state.hca_pool_destination(
                            family.representative, self.decode_position
                        )
                    )
                state_store = SchedDsv4CompressorStateStore(
                    (compress_values[: cfg.head_dim],),
                    (compress_scores[: cfg.head_dim],),
                    (bias_rows[0],),
                    (destination_values,),
                    (destination_scores,),
                )
                state_store = self._layered(
                    state_store, family, bias_rows
                )
                state_store_sms = 1
            stages.append(
                self._stage(
                    "attn.compressor.state_store",
                    state_store,
                    state_store_sms,
                    base_sm=self.sms - 4,
                    wait_for_previous=False,
                    wait_group=compressor_projection_ready,
                    join_completion=True,
                )
            )
        if live_compressor_state and kind == "csa":
            index_ape_tensors = self._family_ape_rows(
                family,
                "attn.indexer.compressor.ape",
                position_in_group,
            )
            index_bias_rows = tuple(
                tuple(
                    ape[
                        half * cfg.index_head_dim :
                        (half + 1) * cfg.index_head_dim,
                    ]
                    for ape in index_ape_tensors
                )
                for half in range(2)
            )
            if self.dynamic_position:
                storage_values, storage_scores = (
                    self.live_state.csa_pool_storage(
                        family.representative, index=True
                    )
                )
                flat_values = storage_values.reshape(
                    -1, cfg.index_head_dim
                )
                flat_scores = storage_scores.reshape(
                    -1, cfg.index_head_dim
                )
                ordinary_base = 4
                index_destinations = (
                    flat_values[8],
                    flat_scores[8],
                    flat_values[ordinary_base],
                    flat_scores[ordinary_base],
                )
                row_bytes = cfg.index_head_dim * 4
                overlap_offsets = (
                    (self.position_counter_reg, row_bytes, 0, 2),
                    (self.position_counter_reg, -8 * row_bytes, 2, 1),
                )
                ordinary_offsets = (
                    (self.position_counter_reg, row_bytes, 0, 2),
                )
                ordinary_offsets += ((
                    self.position_counter_reg,
                    8 * row_bytes,
                    2,
                    1,
                ),)
                self._register_dynamic_store(
                    index_destinations[0], *overlap_offsets
                )
                self._register_dynamic_store(
                    index_destinations[1], *overlap_offsets
                )
                self._register_dynamic_store(
                    index_destinations[2], *ordinary_offsets
                )
                self._register_dynamic_store(
                    index_destinations[3], *ordinary_offsets
                )
            else:
                index_destinations = self.live_state.csa_pool_destinations(
                    family.representative, self.decode_position, index=True
                )
            index_state_store = SchedDsv4CompressorStateStore(
                (
                    fused_index_output[: cfg.index_head_dim],
                    fused_index_output[
                        cfg.index_head_dim : 2 * cfg.index_head_dim
                    ],
                ),
                (
                    fused_index_output[
                        2 * cfg.index_head_dim : 3 * cfg.index_head_dim
                    ],
                    fused_index_output[
                        3 * cfg.index_head_dim : 4 * cfg.index_head_dim
                    ],
                ),
                (index_bias_rows[0][0], index_bias_rows[1][0]),
                (index_destinations[0], index_destinations[2]),
                (index_destinations[1], index_destinations[3]),
            )
            index_state_store = self._layered(
                index_state_store,
                family,
                index_bias_rows[0],
                index_bias_rows[1],
            )
            stages.append(
                self._stage(
                    "index.compressor.state_store",
                    index_state_store,
                    2,
                    base_sm=self.sms - 2,
                    wait_for_previous=False,
                    wait_group=index_compressor_projection_ready,
                    join_completion=True,
                )
            )
        if (
            use_grouped_preattention
            and kind in ("csa", "hca")
            and plan.should_compress
        ):
            ape_tensors = self._family_ape_rows(
                family,
                "attn.compressor.ape",
                self.decode_position % plan.compress_ratio,
            )
            tail_offset = cfg.head_dim if plan.compress_ratio == 4 else 0
            ape_rows = tuple(
                ape[tail_offset : tail_offset + cfg.head_dim]
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
                self.live_state is None
                and (
                    self.args.gated_pool_mode == "packed"
                or (
                    self.args.gated_pool_mode == "auto"
                    and plan.compress_ratio == 128
                )
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
            index_ape_tensors = self._family_ape_rows(
                family,
                "attn.indexer.compressor.ape",
                self.decode_position % plan.compress_ratio,
            )
            index_ape_rows = tuple(
                ape[cfg.index_head_dim : 2 * cfg.index_head_dim]
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
                self.current_index_compressed,
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
                    norm_hidden,
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
                        release_group=(
                            index_selection_input_join
                            if fuse_index_q_splitk_epilogue
                            else index_q_ready
                        ),
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
                        release_group=index_q_ready,
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
                        release_group=index_q_ready,
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
                        wait_group=index_q_ready,
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
                            position_counter_reg=self.position_counter_reg,
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
                        release_group=index_selection_ready,
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
                        norm_hidden,
                        compress_values[:width],
                    )
                )
                stages.append(
                    self._bf16_linear_stage(
                        "attn.compressor.wgate",
                        family,
                        "attn.compressor.wgate.weight",
                        norm_hidden,
                        compress_scores[:width],
                        wait_for_previous=False,
                    )
                )
            if plan.should_compress:
                ape_tensors = self._family_ape_rows(
                    family,
                    "attn.compressor.ape",
                    self.decode_position % plan.compress_ratio,
                )
                tail_offset = cfg.head_dim if plan.compress_ratio == 4 else 0
                ape_rows = tuple(
                    ape[tail_offset : tail_offset + cfg.head_dim]
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
                    self.live_state is None
                    and (
                        self.args.gated_pool_mode == "packed"
                    or (
                        self.args.gated_pool_mode == "auto"
                        and plan.compress_ratio == 128
                    )
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
                        norm_hidden,
                        self.index_head_weights,
                    )
                )
            if not use_grouped_preattention and plan.should_compress:
                stages.append(
                    self._bf16_linear_stage(
                        "index.compressor.wkv",
                        family,
                        "attn.indexer.compressor.wkv.weight",
                        norm_hidden,
                        index_compress_values,
                    )
                )
                stages.append(
                    self._bf16_linear_stage(
                        "index.compressor.wgate",
                        family,
                        "attn.indexer.compressor.wgate.weight",
                        norm_hidden,
                        index_compress_scores,
                        wait_for_previous=False,
                    )
                )
            if plan.should_compress and not use_grouped_preattention:
                index_ape_tensors = self._family_ape_rows(
                    family,
                    "attn.indexer.compressor.ape",
                    self.decode_position % plan.compress_ratio,
                )
                index_ape_rows = tuple(
                    ape[cfg.index_head_dim : 2 * cfg.index_head_dim]
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
                    self.current_index_compressed,
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
                            position_counter_reg=self.position_counter_reg,
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
                        release_group=index_selection_ready,
                    )
                )

        attention_kv = self.attention_cache[kind]
        gathered_attention = (
            use_split_umma_attention and plan.requires_index_selection
        )
        if gathered_attention:
            if kind != "csa" or self.attention_gather_workspace is None:
                raise ValueError("indexed UMMA gather requires CSA workspace")
            indices = self.attention_indices_by_kind[kind]
            if self.live_state is not None:
                source_layers = tuple(
                    self.live_state.attention_cache(
                        active_layer_id, plan.compressed_rows
                    )
                    for active_layer_id in family.layer_ids
                )
            else:
                source_layers = (self.attention_cache[kind],)
            indexed_tables = tuple(
                IndexedLoadTable(source, indices) for source in source_layers
            )
            gather = SchedDsv4IndexedGather512(
                source_layers[0],
                indices,
                self.attention_gather_workspace,
                indexed_table=indexed_tables[0],
            )
            if len(source_layers) > 1:
                gather = LayeredSchedule(
                    gather,
                    ((
                        indexed_tables[0].state,
                        tuple(table.state for table in indexed_tables),
                    ),),
                    counter_strides=family.counter_strides,
                )
            gather_dependencies = ()
            if use_grouped_preattention and plan.should_compress:
                gather_dependencies = (
                    (compressor_output_ready, "kv"),
                    (index_selection_ready, "indices"),
                )
            stages.append(
                self._stage(
                    "attn.indexed_gather",
                    gather,
                    min(128, indices.numel()),
                    wait_group=(
                        None
                        if gather_dependencies
                        else index_selection_ready
                    ),
                    wait_group_roles=gather_dependencies,
                    release_group=attention_gather_ready,
                )
            )
            attention_kv = self.attention_gather_workspace

        sinks = self._family_tensors(family, "attn.attn_sink")
        attention_rows = self.attention_indices_by_kind[kind].numel()
        o_a_split_k = 4 if self.args.context_length == cfg.sliding_window else 2
        if self.args.attention_mode == "umma-split" and not split_o_a:
            raise ValueError(
                "UMMA split attention requires native split-K O_a"
            )
        if use_split_umma_attention and not gathered_attention and (
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
                if self.live_state is not None and len(family.layer_ids) > 1:
                    context1 = SchedLayeredDsv4AttentionContext1Fp8Sm100(
                        context1,
                        tuple(
                            self.live_state.attention_cache(
                                layer_id, plan.compressed_rows
                            )[
                                self.decode_position % cfg.sliding_window :
                                self.decode_position % cfg.sliding_window + 1
                            ]
                            for layer_id in family.layer_ids
                        ),
                        counter_strides=family.counter_strides,
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
                self.launcher, attention_kv
            ).wgmma_load(64, 512, Major.K)
            kv_v_tma = TmaTensor(
                self.launcher, attention_kv
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
                attention_kv,
                attention_rows,
                partials,
                metadata,
                q_tma=q_tma,
                kv_tma=kv_tma,
                kv_v_tma=kv_v_tma,
                partial_tma=partial_tma,
                gate_kv_last_split_only=(
                    not gathered_attention
                    and use_grouped_preattention
                    and kind in ("csa", "hca")
                    and plan.should_compress
                ),
                position_counter_reg=(
                    None if gathered_attention else self.position_counter_reg
                ),
                attention_kind=(
                    None
                    if gathered_attention or not self.dynamic_position
                    else kind
                ),
            )
            if (
                not gathered_attention
                and self.live_state is not None
                and len(family.layer_ids) > 1
            ):
                producer = SchedLayeredDsv4AttentionSplit64UmmaSm100(
                    producer,
                    tuple(
                        self.live_state.attention_cache(
                            layer_id, plan.compressed_rows
                        )
                        for layer_id in family.layer_ids
                    ),
                    counter_strides=family.counter_strides,
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
                            (attention_gather_ready, "kv"),
                        )
                        if gathered_attention
                        else (
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
                    position_counter_reg=(
                        None if gathered_attention else self.position_counter_reg
                    ),
                    attention_kind=(
                        None
                        if gathered_attention or not self.dynamic_position
                        else kind
                    ),
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

    @staticmethod
    def reusable_variant_for_position(position: int) -> str:
        position = int(position)
        if position == 0:
            return "context1"
        if position == 3:
            return "csa_first"
        if (position + 1) % 128 == 0:
            variant = "hca"
        elif (position + 1) % 4 == 0:
            variant = "csa_short" if position < 127 else "csa"
        else:
            variant = "normal"
        return f"indexed_{variant}" if position >= 2051 else variant

    def set_decode_position(self, position: int) -> None:
        """Retarget one reusable live image without rebuilding its schedule."""

        if not self.dynamic_position:
            if int(position) != self.decode_position:
                raise RuntimeError("fixed-position image cannot be retargeted")
            return
        position = int(position)
        if not 0 <= position <= self.dynamic_max_position:
            raise ValueError("decode position exceeds the reusable image range")
        expected = self.reusable_variant_for_position(position)
        if expected != self.dynamic_variant:
            raise ValueError(
                f"position {position} requires {expected}, not "
                f"{self.dynamic_variant}"
            )
        self.live_state.prepare_decode_position(position)
        self.launcher.set_loop_counter(self.position_counter_reg, position)

        main_bank, compressed_bank = self._dynamic_rope_banks
        self.resident_rope_packed[0].copy_(main_bank[position])
        self.resident_rope_packed[1].copy_(compressed_bank[position])
        for kind in ("csa", "hca"):
            table = self.compressed_output_rope.get(kind)
            if table is None:
                continue
            table_id = self.resident_rope_table_ids[table.data_ptr()]
            output_position = (
                position - self.attention_plans[kind].compress_ratio + 1
            )
            self.resident_rope_packed[table_id].copy_(
                compressed_bank[output_position]
            )

        for key, update in self._dynamic_position_updates.items():
            if key[0] == "ape":
                arena, bank = update
                arena.copy_(bank[:, position % bank.shape[1]])
            elif key[0] == "csa_history":
                arena, storage, rows, row_start = update
                bank = (position // 4) & 1
                arena.copy_(
                    storage[:, bank, row_start : row_start + rows]
                )
            else:
                raise RuntimeError(f"unknown dynamic-position update {key[0]}")
        self._active_dynamic_position = position

    def set_input_token(self, token_id: int) -> None:
        """Update the token-dependent inputs of a prepared live image."""
        if self.live_state is None:
            raise RuntimeError("set_input_token is only valid for live decode")
        if self.dynamic_position and self._active_dynamic_position is None:
            raise RuntimeError("set_decode_position must precede token input")
        if not 0 <= int(token_id) < self.config.vocab_size:
            raise ValueError("input token is outside the vocabulary")
        embedding = self._tensor("embed.weight")[int(token_id)]
        self.initial_residual.copy_(
            embedding.reshape(1, -1).expand(self.config.hc_mult, -1)
        )
        for layer_id in range(self.config.num_hash_layers):
            source = self._tensor(
                f"layers.{layer_id}.ffn.gate.tid2eid"
            )[int(token_id)]
            self.live_hash_rows[
                layer_id, : self.config.experts_per_token
            ].copy_(source.to(torch.int32))

    def _build_mxfp_ffn(
        self, family: LayerFamily
    ) -> tuple[Stage, list[Stage]]:
        cfg = self.config
        (
            _,
            norm_hidden,
            mhc_output_metadata,
            post,
            comb,
        ) = self._mhc_outputs(family.representative)
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
        output_sets = {
            self._mxfp_output_set(layer_id)
            for layer_id in family.layer_ids
        }
        if len(output_sets) != 1:
            raise ValueError("one MXFP family must share one output direction")
        mxfp_ffn_output = self.mxfp_ffn_outputs[output_sets.pop()]
        # Keep the legacy diagnostic alias pointed at the final family built.
        self.mxfp_ffn_output = mxfp_ffn_output
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
        physical_key = (
            "mxfp_ffn_physical_gate_up_v1",
            representative.image.linear1_weights.data_ptr(),
            representative.image.linear1_scales.data_ptr(),
        )
        physical_gate_up = self._derived_tensor_cache.get(physical_key)
        if physical_gate_up is None:
            linear1_physical = representative.image.linear1_weights[:112]
            linear1_scale_physical = representative.image.linear1_scales[:112]
            physical_gate_up = (
                linear1_physical[:, :8].contiguous(),
                linear1_physical[:, 8:].contiguous(),
                linear1_scale_physical[:, :8].contiguous(),
                linear1_scale_physical[:, 8:].contiguous(),
            )
            self._derived_tensor_cache[physical_key] = physical_gate_up
        gate_weight, up_weight, gate_scale, up_scale = physical_gate_up
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
            mxfp_ffn_output,
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
                if (
                    self.args.profile_steps
                    or (
                        self.args.profile_cross_layer_hc_barrier
                        and family.representative == self.layer_ids[0]
                    )
                )
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
                mxfp_ffn_output[0],
                self.next_residual,
                post,
                comb,
                self.residual,
                launcher=self.launcher,
                packed_coefficients=mhc_output_metadata,
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
                        norm_hidden, self.mxfp_input_records
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
            if self.live_hash_rows is None:
                hash_rows = torch.stack(
                    tuple(
                        self._hash_row(layer_id)
                        for layer_id in family.layer_ids
                    )
                ).contiguous()
            else:
                first = family.layer_ids[0]
                if family.layer_ids != tuple(
                    range(first, first + len(family.layer_ids))
                ):
                    raise ValueError("live hash families must be contiguous")
                hash_rows = self.live_hash_rows[
                    first : first + len(family.layer_ids)
                ]
        else:
            router_biases = self._family_tensors(family, "ffn.gate.bias")
            hash_rows = self.zero_hash
        stages.append(
            self._stage(
                "ffn.router.prepared",
                SchedLayeredDsv4RouterBf16Gemv(
                    router_weights,
                    norm_hidden,
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
        live_state_groups = self._activate_live_family_state(family)
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
        if live_state_groups and (
            len(family.layer_ids) > 1 or self.dynamic_position
        ):
            attention = [
                replace(
                    stage,
                    schedule=LayerStateSchedule(
                        stage.schedule,
                        live_state_groups,
                        counter_strides=family.counter_strides,
                        store_offset_rules=self._dynamic_store_rules,
                    ),
                )
                for stage in attention
            ]
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

        _, _, previous_coefficients, _, _ = self._mhc_outputs(
            previous_family.representative
        )
        (
            next_packed_output,
            next_norm_hidden,
            _,
            next_post,
            next_comb,
        ) = self._mhc_outputs(next_family.representative)

        packed_weights, metadata_tails = self._fused_hc_projection_operands(
            next_family, "attn"
        )
        if (
            self.args.profile_cross_layer_hc_barrier
            or self.args.stop_after_cross_layer_hc_write
        ):
            self.mhc_fused_weight_reference = packed_weights[0]
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
        packed_record = self.mhc_cross_layer_input_records[
            self._mxfp_output_set(previous_family.representative)
        ]
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

        boundary_capture_stages = []
        if self.args.diagnose_cross_layer_hc_boundary:
            record_snapshot = self.mhc_boundary_record_snapshot
            coefficients_snapshot = self.mhc_boundary_coefficients_snapshot
            record_snapshot_load = TmaLoad1D(packed_record)
            record_snapshot_load.annotation[
                "prefetch_before_resident_reset"
            ] = True
            boundary_capture_stages.extend(
                (
                    self._stage(
                        "debug.hc_boundary.record_snapshot",
                        SchedCopy(
                            (
                                record_snapshot_load,
                                TmaStore1D(record_snapshot),
                            )
                        ),
                        1,
                        base_sm=self.sms - 1,
                        input_role="load",
                        prefetch_before_wait=True,
                        prefetch_before_resident_reset=True,
                    ),
                    self._stage(
                        "debug.hc_boundary.coefficients_snapshot",
                        SchedCopy(
                            (
                                TmaLoad1D(previous_coefficients),
                                TmaStore1D(coefficients_snapshot),
                            )
                        ),
                        1,
                        base_sm=self.sms - 1,
                    ),
                )
            )

        fused_project = SchedDsv4Fp32Bf16Gemv(
            packed_weights[0],
            self.residual.reshape(-1),
            self.mixes,
            fused_post_input_record=packed_record,
            fused_post_output=self.residual,
            fused_partial_metadata=self.mhc_fused_metadata,
            packed_coefficients=previous_coefficients,
            launcher=self.launcher,
            prefetch_operands_before_resident_reset=(
                not self.args.diagnose_cross_layer_hc_boundary
            ),
            profile_operands=(
                self.args.profile_cross_layer_hc_barrier
                or self.args.stop_after_cross_layer_hc_write
            ),
            captured_record=self.mhc_consumed_record_capture,
            captured_weight=self.mhc_consumed_weight_capture,
            captured_coefficients=self.mhc_consumed_coefficient_capture,
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
            prefetch_before_wait=(
                not self.args.diagnose_cross_layer_hc_boundary
            ),
            prefetch_before_resident_reset=(
                not self.args.diagnose_cross_layer_hc_boundary
            ),
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
            next_norm_hidden,
            next_post,
            next_comb,
            residual_square_sum=self.mhc_fused_residual_square_sum,
            packed_metadata=self.mhc_fused_metadata,
            packed_output=next_packed_output,
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
        next_stages[project_index:project_index] = [
            *boundary_capture_stages,
            fused_stage,
            pre_stage,
        ]

    def _apply_loopback_hc_fusion(
        self,
        layer2_family: LayerFamily,
        hca_family: LayerFamily,
        csa_family: LayerFamily,
        terminal_hca_family: LayerFamily | None = None,
    ) -> None:
        """Fuse layer-2/CSA post work into each following HCA.

        A production-prefix diagnostic may end on an HCA after the repeated
        HCA/CSA body.  In that case ``terminal_hca_family`` consumes the final
        CSA record with the same fused projection and asynchronous barrier
        reload as production, while retaining its own ordinary FFN HC-post as
        the observable terminal boundary.
        """
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

        _, _, loopback_coefficients, _, _ = self._mhc_outputs(
            csa_family.representative
        )

        loopback_output_sets = {
            self._mxfp_output_set(family.representative)
            for family in (layer2_family, csa_family)
        }
        if len(loopback_output_sets) != 1:
            raise ValueError(
                "layer-2 and CSA loopback must share one output direction"
            )
        packed_record = self.mhc_cross_layer_input_records[
            loopback_output_sets.pop()
        ]

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

        def replace_hca_input(
            family: LayerFamily,
            stages: list[Stage],
            *,
            skip_initial_loop: bool,
        ) -> None:
            (
                hca_packed_output,
                hca_norm_hidden,
                _,
                hca_post,
                hca_comb,
            ) = self._mhc_outputs(family.representative)
            packed_weights, metadata_tails = (
                self._fused_hc_projection_operands(family, "attn")
            )
            norm_weights = self._family_tensors(
                family, "attn_norm.weight"
            )
            metadata_ready = f"{family.name}.attn.hc.metadata.ready"
            residual_ready = f"{family.name}.attn.hc.residual.ready"
            attention_input_ready = f"{family.name}.attn.input.ready"
            resident_input_ready = f"{family.name}.ffn.mx.input.ready"

            project_index = next(
                index
                for index, stage in enumerate(stages)
                if stage.name == "attn.hc_project"
            )
            if stages[project_index + 1].name != "attn.hc_pre_rms4096":
                raise ValueError(
                    "loop-back mHC fusion requires adjacent HCA project/pre "
                    "stages"
                )
            del stages[project_index:project_index + 2]

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
                family,
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
                packed_coefficients=loopback_coefficients,
                launcher=self.launcher,
                profile_operands=self.args.profile_loopback_boundary,
            )
            fused_project = self._layered(
                fused_project,
                family,
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
                skip_initial_loop=skip_initial_loop,
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
                hca_norm_hidden,
                hca_post,
                hca_comb,
                residual_square_sum=self.mhc_fused_residual_square_sum,
                packed_metadata=self.mhc_fused_metadata,
                packed_output=hca_packed_output,
                split_metadata_splits=(
                    SchedDsv4Fp32Bf16Gemv.FUSED_SPLITS
                ),
            )
            pre = self._layered(pre, family, norm_weights)
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
            stages[project_index:project_index] = [
                tail_stage,
                fused_stage,
                pre_stage,
            ]

        replace_hca_input(
            hca_family,
            hca_stages,
            skip_initial_loop=True,
        )
        if terminal_hca_family is not None:
            replace_hca_input(
                terminal_hca_family,
                self.family_stages[terminal_hca_family.representative],
                skip_initial_loop=False,
            )
        else:
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
                            derived_tensor_cache=self._derived_tensor_cache,
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
            if (
                self.weight_source is not None
                and hasattr(self.weight_source, "head_weight_bf16_packed")
                and self.weight_source.args.vocab_size == self.args.vocab_size
            ):
                self.head_weight_bf16_packed = (
                    self.weight_source.head_weight_bf16_packed
                )
            else:
                epoch_weights = []
                for epoch in range(num_epochs):
                    row_start = epoch * epoch_rows
                    row_end = min(
                        row_start + epoch_rows, self.args.vocab_size
                    )
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
                    epoch_weights.append(
                        pack_weight_tile_major(source, 128, 128)
                    )
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
                join_completion=stage.join_completion,
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
            SchedDsv4PreloadRopeTables(
                self.resident_rope_tables,
                packed_tables=self.resident_rope_packed,
            ).place(self.sms)
        )
        if self.args.stop_after_layer is not None:
            families = {family.name: family for family in self.families}
            blocks = []

            swa = families["prefix.swa_hash"]
            blocks.append(
                SequentialBlock(
                    swa.name,
                    queued_family(swa),
                    repeat=len(swa.layer_ids),
                    barrier_banks=family_barrier_banks,
                    reload_barrier_start=mxfp_reload_start,
                    reload_mxfp_resident=True,
                )
            )

            layer2 = families.get("prefix.csa_hash")
            if layer2 is not None:
                blocks.append(
                    SequentialBlock(
                        layer2.name,
                        queued_family(layer2),
                        reload_barrier_start=mxfp_reload_start,
                        reload_mxfp_resident=True,
                    )
                )

            hca = families.get("prefix.hca_score")
            csa = families.get("prefix.csa_score")
            if hca is not None:
                if csa is None or len(hca.layer_ids) != len(csa.layer_ids):
                    raise AssertionError("prefix HCA/CSA families are not paired")
                pair_has_consumer = (
                    "prefix.hca_tail_score" in families
                    or bool(self.head_stages)
                )
                hca_stages = queued_family(hca)
                hca_stages[-1] = replace(
                    hca_stages[-1], reset_mxfp_resident_after=True
                )
                blocks.append(
                    SequentialBlock(
                        "prefix.hca_csa_score",
                        hca_stages + queued_family(csa),
                        repeat=len(hca.layer_ids),
                        barrier_banks=pair_barrier_banks,
                        reload_barrier_start=(
                            None
                            if runtime_config.async_barrier_reload_enabled
                            else mxfp_reload_start
                        ),
                        reload_mxfp_resident=True,
                        elide_terminal_reload=pair_has_consumer,
                        async_reload_after=(
                            runtime_config.async_barrier_reload_enabled
                        ),
                        async_reload_worker_base=32,
                    )
                )

            hca_tail = families.get("prefix.hca_tail_score")
            if hca_tail is not None:
                blocks.append(
                    SequentialBlock(
                        hca_tail.name,
                        queued_family(hca_tail),
                        reload_barrier_start=mxfp_reload_start,
                        reload_mxfp_resident=True,
                    )
                )

            terminal_stages = queued_head()
            if terminal_stages:
                blocks.append(
                    SequentialBlock(
                        "prefix.terminal_hc_post",
                        terminal_stages,
                        reload_after=False,
                    )
                )

            self.program = LoopedSequentialProgram(
                self.launcher, tuple(blocks), balance_load_ports=True
            )
            logical_stages = sum(
                len(block.stages) * block.repeat for block in blocks
            )
            queue_stages = sum(len(block.stages) for block in blocks)
        elif self.args.stop_after_cross_layer_hc_write:
            # Execute the real integrated HCA->CSA producer chain, then stop
            # immediately after the fused projection's writeback tails.  This
            # leaves its residual and 16-byte-per-SM metadata records as the
            # final HBM generation for exact readback.
            first_family, second_family = self.families
            first_stages = queued_family(first_family, enable_profile=False)
            first_stages[-1] = replace(
                first_stages[-1], reset_mxfp_resident_after=True
            )
            second_stages = queued_family(second_family, enable_profile=False)
            stages = first_stages + second_stages
            self.program = SequentialProgram(
                self.launcher,
                stages,
                balance_load_ports=True,
            )
            logical_stages = len(stages)
            queue_stages = logical_stages
        elif self.args.layers == 1:
            family = self.families[0]
            stages = queued_family(family)
            # The looped multi-layer image resets the persistent MXFP rings
            # at every block tail.  Preserve the same direct, FFN-completion-
            # dependent reset in the one-layer diagnostic image so repeated
            # launches do not inherit the previous launch's full/empty phase.
            if not self.args.omit_head:
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
            f"model_launches=1 layers={len(self.layer_ids)} "
            f"stop_after_layer={self.args.stop_after_layer if self.args.stop_after_layer is not None else -1} "
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
        if self.args.stop_after_cross_layer_hc_write:
            # Poison only the partial-record region before the diagnostic
            # launch.  The tail has its own real producer.  This makes any
            # missing 16-byte writer visible without adding work to the
            # persistent kernel or relying on allocator-cache contents.
            self.mhc_fused_metadata[
                : SchedDsv4Fp32Bf16Gemv.FUSED_SPLITS
                * SchedDsv4Fp32Bf16Gemv.FUSED_RECORD_STRIDE
            ].view(torch.int32).fill_(0x7FFFFFFF)
            self.mhc_consumed_record_capture.fill_(float("nan"))
            self.mhc_consumed_weight_capture.fill_(float("nan"))
            self.mhc_consumed_coefficient_capture.fill_(float("nan"))
        if (
            self.args.profile_layers
            or self.args.profile_stages
            or self.args.profile_ffn_aggregate
            or self.args.profile_phase_aggregate
            or self.args.profile_attention_detail
            or self.args.profile_mxfp_ffn_basic
            or self.args.profile_mxfp_ffn_detail
            or self.args.profile_cross_layer_hc_barrier
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
        if (
            self.args.stop_after_layer is not None
            or self.args.stop_after_cross_layer_hc_write
            or self.args.omit_head
        ):
            return -1, start.elapsed_time(end), torch.empty(0)
        if self.compact_head:
            return int(self.output_token[0].item()), start.elapsed_time(end), torch.empty(0)
        logits_cpu = self.logits.cpu()
        logits_fp32 = logits_cpu.float()
        if not bool(torch.isfinite(logits_fp32).all().item()):
            if (
                self.args.hidden_reference is not None
                or self.args.dump_final_hidden is not None
            ):
                return -1, start.elapsed_time(end), logits_fp32
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
            "mhc_packed_outputs",
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
            first = tuple(
                int(value)
                for value in mismatch.nonzero()[0].tolist()
            )
            if actual.is_floating_point():
                max_abs = float(
                    (actual.float() - expected.float()).abs().max().item()
                )
                actual_nonfinite = int(
                    (~torch.isfinite(actual)).count_nonzero().item()
                )
                expected_nonfinite = int(
                    (~torch.isfinite(expected)).count_nonzero().item()
                )
            else:
                max_abs = -1.0
                actual_nonfinite = 0
                expected_nonfinite = 0
            print(
                "DSV4_REPEAT_STATE "
                f"iteration={iteration} name={name} exact=false "
                f"mismatches={count} max_abs={max_abs:.6f} "
                f"actual_nonfinite={actual_nonfinite} "
                f"expected_nonfinite={expected_nonfinite} "
                f"first_index={first} "
                f"actual_first={actual[first].item()} "
                f"expected_first={expected[first].item()}",
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

    @staticmethod
    def _native_fp8_data_reference(source: torch.Tensor) -> torch.Tensor:
        """Return the active 1 KiB data half of each scale-pack-2 tile."""
        quantized, _ = quantize_fp8_block128(source.reshape(-1))
        tiles = source.numel() // 128
        logical = quantized.view(torch.uint8).reshape(tiles, 8, 16)
        expected = torch.empty(
            (tiles, 8, 8, 16), dtype=torch.uint8, device=source.device
        )
        for row in range(8):
            for source_chunk in range(8):
                expected[:, row, source_chunk ^ row].copy_(
                    logical[:, source_chunk]
                )
        return expected.reshape(tiles, 1024)

    @staticmethod
    def _mxfp8_ffn_input_reference(
        source: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return packed K512 data and the active scale bytes."""
        records = source.numel() // 512
        groups = source.float().reshape(records, 16, 32)
        requested = (
            groups.abs().amax(dim=-1) / 448.0
        ).clamp_min(2.0**-127)
        exponents = torch.ceil(torch.log2(requested)).clamp(-127, 127)
        scales = torch.exp2(exponents)
        quantized = (
            (groups / scales.unsqueeze(-1))
            .clamp(-448.0, 448.0)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
            .reshape(records, 4, 8, 16)
        )
        packed = torch.empty(
            (records, 4, 8, 8, 16),
            dtype=torch.uint8,
            device=source.device,
        )
        for row in range(8):
            for source_chunk in range(8):
                packed[:, :, row, source_chunk ^ row].copy_(
                    quantized[:, :, source_chunk]
                )
        scale_bytes = (exponents.to(torch.int16) + 127).to(torch.uint8)
        active_scales = scale_bytes.reshape(records, 4, 4)
        active_scales = active_scales.unsqueeze(2).expand(
            records, 4, 8, 4
        ).contiguous()
        return (
            packed.reshape(records, 4096),
            active_scales.reshape(records, 4, 32),
        )

    @staticmethod
    def _rms_reference(
        source: torch.Tensor,
        epsilon: float,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized = source.float() * torch.rsqrt(
            source.float().square().mean(dim=-1, keepdim=True) + epsilon
        )
        if weight is not None:
            normalized *= weight.float()
        return normalized.to(torch.bfloat16)

    @staticmethod
    def _report_numeric_boundary(
        name: str,
        actual: torch.Tensor,
        expected: torch.Tensor,
    ) -> None:
        actual_fp32 = actual.float().reshape(-1)
        expected_fp32 = expected.float().reshape(-1)
        delta = actual_fp32 - expected_fp32
        expected_norm = float(torch.linalg.vector_norm(expected_fp32).item())
        delta_norm = float(torch.linalg.vector_norm(delta).item())
        cosine = float(
            torch.nn.functional.cosine_similarity(
                actual_fp32.reshape(1, -1),
                expected_fp32.reshape(1, -1),
            ).item()
        )
        print(
            "DSV4_ATTENTION_CORRECTNESS "
            f"stage={name} kind=numeric "
            f"rel_l2={delta_norm / max(expected_norm, 1.0e-30):.9f} "
            f"cosine={cosine:.9f} "
            f"mean_abs={float(delta.abs().mean().item()):.9f} "
            f"max_abs={float(delta.abs().max().item()):.9f} "
            f"actual_nonfinite={int((~torch.isfinite(actual_fp32)).sum().item())} "
            f"expected_nonfinite={int((~torch.isfinite(expected_fp32)).sum().item())}",
            flush=True,
        )

    @staticmethod
    def _report_exact_boundary(
        name: str,
        actual: torch.Tensor,
        expected: torch.Tensor,
    ) -> None:
        mismatch = actual.reshape(-1) != expected.reshape(-1)
        count = int(mismatch.sum().item())
        if count:
            first_flat = int(mismatch.nonzero()[0].item())
            first_index = tuple(
                int(value)
                for value in torch.unravel_index(
                    torch.tensor(first_flat, device=actual.device),
                    actual.shape,
                )
            )
            first_detail = (
                f" first_index={first_index} "
                f"actual_first={actual[first_index].item()} "
                f"expected_first={expected[first_index].item()}"
            )
        else:
            first_detail = ""
        print(
            "DSV4_ATTENTION_CORRECTNESS "
            f"stage={name} kind=exact mismatches={count} "
            f"elements={actual.numel()} "
            f"fraction={count / actual.numel():.9f}{first_detail}",
            flush=True,
        )

    def report_attention_correctness(self) -> None:
        """Compare one resident attention path with the checkpoint oracle.

        This is entirely host-side post-launch diagnosis.  It adds no task,
        instruction, write, or synchronization to the resident kernel.
        """
        if len(self.layer_ids) != 1:
            raise ValueError("attention correctness diagnosis requires one layer")
        layer_id = self.layer_ids[0]
        prefix = f"layers.{layer_id}"
        attn_prefix = f"{prefix}.attn"
        epsilon = self.config.rms_epsilon
        disk_checkpoint = DeepSeekV4Checkpoint(
            self.args.checkpoint, self.config
        )

        residual = self.initial_residual
        functions = self._tensor(f"{prefix}.hc_attn_fn")
        scales = self._tensor(f"{prefix}.hc_attn_scale")
        bases = self._tensor(f"{prefix}.hc_attn_base")
        mixes = functions.float() @ residual.reshape(-1).float()
        hidden, attn_post_coefficients, attn_comb = hc_pre_reference(
            residual, mixes, scales, bases
        )
        normalized = self._rms_reference(
            hidden,
            epsilon,
            self._tensor(f"{prefix}.attn_norm.weight"),
        )
        expected_hidden_native = self._native_fp8_data_reference(normalized)
        self._report_exact_boundary(
            "attn_hidden_native_fp8_data",
            self.hidden_native_fp8[:, :1024],
            expected_hidden_native,
        )

        def fp8_linear(name: str, source: torch.Tensor) -> torch.Tensor:
            activation, activation_scale = quantize_fp8_block128(
                source.reshape(-1)
            )
            linear = disk_checkpoint.load_fp8_linear(
                name, device=self.device
            )
            return (
                dequantize_fp8_block128(linear.weight, linear.scale)
                @ dequantize_fp8_block128(activation, activation_scale)
            ).to(torch.bfloat16)

        q_rank = fp8_linear(f"{attn_prefix}.wq_a", normalized)
        self._report_numeric_boundary("q_a", self.q_rank, q_rank)
        q_norm_weight = self._tensor(f"{attn_prefix}.q_norm.weight")
        q_rank_normalized = self._rms_reference(
            q_rank, epsilon, q_norm_weight
        )
        expected_q_rank_native = self._native_fp8_data_reference(
            q_rank_normalized
        )
        self._report_exact_boundary(
            "q_rank_native_fp8_data_e2e",
            self.q_rank_native_fp8[:, :1024],
            expected_q_rank_native,
        )

        q = fp8_linear(f"{attn_prefix}.wq_b", q_rank_normalized).reshape(
            self.config.num_heads, self.config.head_dim
        )
        self._report_numeric_boundary("q_b_e2e", self.q, q)
        actual_q_rank_normalized = self._rms_reference(
            self.q_rank, epsilon, q_norm_weight
        )
        q_local = fp8_linear(
            f"{attn_prefix}.wq_b", actual_q_rank_normalized
        ).reshape_as(self.q)
        self._report_numeric_boundary("q_b_local", self.q, q_local)

        kv = fp8_linear(f"{attn_prefix}.wkv", normalized)
        self._report_numeric_boundary("kv", self.kv, kv)
        rope_table = (
            self.main_rope
            if self.config.attention_kind(layer_id) == "swa"
            else self.compress_rope
        )
        kv_rope = apply_partial_rope_512_64(
            self._rms_reference(
                kv,
                epsilon,
                self._tensor(f"{attn_prefix}.kv_norm.weight"),
            ).reshape(1, self.config.head_dim),
            rope_table,
        )
        kind = self.config.attention_kind(layer_id)
        self._report_numeric_boundary(
            "kv_rms_rope", self.current_kv_rows[kind], kv_rope
        )

        q_rope = apply_partial_rope_512_64(
            self._rms_reference(q, epsilon), rope_table
        )
        indices = torch.zeros((1,), dtype=torch.int32, device=self.device)
        sink = self._tensor(f"{attn_prefix}.attn_sink")
        attended = sparse_attention_512_reference(
            q_rope, kv_rope, indices, sink
        )
        attention_inverse = apply_partial_rope_512_64(
            attended, rope_table, inverse=True
        )
        expected_o_native = self._native_fp8_data_reference(
            attention_inverse
        ).reshape(self.config.num_heads, 4, 1024)
        actual_o_native = self.o_group_native_fp8.view(
            self.config.num_heads, 4, 2048
        )[:, :, :1024]
        self._report_exact_boundary(
            "context1_attention_native_fp8_data",
            actual_o_native,
            expected_o_native,
        )

        def context1_local_values(score_denominator: float) -> torch.Tensor:
            q_values = self.q.float()
            q_values *= torch.rsqrt(
                q_values.square().mean(dim=1, keepdim=True)
                + float.fromhex("0x1.0cp-20")
            )
            q_values = apply_partial_rope_512_64(
                q_values, rope_table
            ).to(torch.bfloat16).float()
            current_kv = self.current_kv_rows[kind].float()
            score = (q_values * current_kv).sum(dim=1) / score_denominator
            probability = torch.sigmoid(score - sink)
            values = current_kv.expand(self.config.num_heads, -1)
            values = values * probability[:, None]
            return apply_partial_rope_512_64(
                values, rope_table, inverse=True
            )

        for name, denominator in (
            ("context1_local_task_scale", math.sqrt(512.0)),
            ("context1_local_old_scale", math.sqrt(128.0)),
        ):
            local_native = self._native_fp8_data_reference(
                context1_local_values(denominator)
            ).reshape(self.config.num_heads, 4, 1024)
            self._report_exact_boundary(
                name, actual_o_native, local_native
            )

        grouped = attention_inverse.reshape(self.config.o_groups, -1)
        wo_a = disk_checkpoint.load_fp8_linear(
            f"{attn_prefix}.wo_a", device=self.device
        )
        o_rank = torch.empty_like(self.o_rank)
        for group in range(self.config.o_groups):
            activation, activation_scale = quantize_fp8_block128(
                grouped[group]
            )
            start = group * self.config.o_lora_rank
            stop = start + self.config.o_lora_rank
            o_rank[group].copy_(
                (
                    dequantize_fp8_block128(
                        wo_a.weight[start:stop],
                        wo_a.scale[start // 128 : stop // 128],
                    )
                    @ dequantize_fp8_block128(
                        activation, activation_scale
                    )
                ).to(torch.bfloat16)
            )
        self._report_numeric_boundary("o_a_e2e", self.o_rank, o_rank)

        o_b = fp8_linear(f"{attn_prefix}.wo_b", o_rank.reshape(-1))
        self._report_numeric_boundary("o_b_e2e", self.branch, o_b)
        o_b_local = fp8_linear(
            f"{attn_prefix}.wo_b", self.o_rank.reshape(-1)
        )
        self._report_numeric_boundary("o_b_local", self.branch, o_b_local)

        attention_post = hc_post_reference(
            o_b,
            residual,
            attn_post_coefficients,
            attn_comb,
        )
        self._report_numeric_boundary(
            "attention_hc_post_e2e", self.next_residual, attention_post
        )
        attention_post_local = hc_post_reference(
            self.branch,
            residual,
            attn_post_coefficients,
            attn_comb,
        )
        self._report_numeric_boundary(
            "attention_hc_post_local",
            self.next_residual,
            attention_post_local,
        )

        ffn_functions = self._tensor(f"{prefix}.hc_ffn_fn")
        ffn_scales = self._tensor(f"{prefix}.hc_ffn_scale")
        ffn_bases = self._tensor(f"{prefix}.hc_ffn_base")
        ffn_norm_weight = self._tensor(f"{prefix}.ffn_norm.weight")

        def ffn_hc_pre_reference(source: torch.Tensor):
            ffn_mixes = ffn_functions.float() @ source.reshape(-1).float()
            ffn_hidden, ffn_post, ffn_comb = hc_pre_reference(
                source, ffn_mixes, ffn_scales, ffn_bases
            )
            return (
                self._rms_reference(
                    ffn_hidden, epsilon, ffn_norm_weight
                ),
                ffn_post,
                ffn_comb,
            )

        (
            ffn_normalized,
            ffn_post_reference,
            ffn_comb_reference,
        ) = ffn_hc_pre_reference(attention_post)
        ffn_normalized_local, ffn_post_local, ffn_comb_local = (
            ffn_hc_pre_reference(self.next_residual)
        )
        _, actual_ffn_normalized, _, actual_ffn_post, actual_ffn_comb = (
            self._mhc_outputs(layer_id)
        )
        self._report_numeric_boundary(
            "ffn_hc_pre_rms_e2e",
            actual_ffn_normalized,
            ffn_normalized,
        )
        self._report_numeric_boundary(
            "ffn_hc_pre_rms_local",
            actual_ffn_normalized,
            ffn_normalized_local,
        )
        self._report_numeric_boundary(
            "ffn_hc_pre_post_coefficients_local",
            actual_ffn_post,
            ffn_post_local,
        )
        self._report_numeric_boundary(
            "ffn_hc_pre_comb_local",
            actual_ffn_comb,
            ffn_comb_local,
        )

        router_weight = self._tensor(f"{prefix}.ffn.gate.weight")

        def prepared_router_reference(source: torch.Tensor) -> torch.Tensor:
            logits = router_weight.float() @ source.reshape(-1).float()
            original = torch.nn.functional.softplus(logits).sqrt()
            if layer_id < self.config.num_hash_layers:
                selection = original
            else:
                selection = original + self._tensor(
                    f"{prefix}.ffn.gate.bias"
                ).float()
            return torch.stack((original, selection), dim=1)

        router_prepared = prepared_router_reference(ffn_normalized)
        router_prepared_local = prepared_router_reference(
            actual_ffn_normalized
        )
        self._report_numeric_boundary(
            "router_prepared_e2e",
            self.router_prepared,
            router_prepared,
        )
        self._report_numeric_boundary(
            "router_prepared_local",
            self.router_prepared,
            router_prepared_local,
        )

        if layer_id < self.config.num_hash_layers:
            expected_indices = self.checkpoint.load_tensor_slice(
                f"{prefix}.ffn.gate.tid2eid",
                self.args.token_id,
                device=self.device,
            ).to(torch.int32)
        else:
            expected_indices = torch.topk(
                router_prepared_local[:, 1],
                self.config.experts_per_token,
            ).indices.to(torch.int32)
        active_experts = self.config.experts_per_token
        expected_indices = expected_indices[:active_experts]
        self._report_exact_boundary(
            "route_top6_indices",
            self.route_indices[:active_experts],
            expected_indices,
        )
        selected_original = router_prepared_local[
            expected_indices.to(torch.int64), 0
        ]
        expected_route_weights = (
            selected_original
            / selected_original.sum()
            * self.config.route_scale
        )
        if layer_id < self.config.num_hash_layers:
            reference_route_indices = expected_indices
        else:
            reference_route_indices = torch.topk(
                router_prepared[:, 1], active_experts
            ).indices.to(torch.int32)
        reference_selected_original = router_prepared[
            reference_route_indices.to(torch.int64), 0
        ]
        reference_route_weights = (
            reference_selected_original
            / reference_selected_original.sum()
            * self.config.route_scale
        )
        self._report_numeric_boundary(
            "route_top6_weights_local",
            self.route_weights[:active_experts],
            expected_route_weights,
        )
        route_words = self.route_record.view(torch.int32)
        expected_linear1_bases = (
            expected_indices + 1
        ) * 16
        expected_down_bases = (
            expected_indices + 1
        ) * 32
        self._report_exact_boundary(
            "route_linear1_task_bases",
            route_words[16 : 16 + active_experts],
            expected_linear1_bases,
        )
        self._report_exact_boundary(
            "route_down_task_bases",
            route_words[24 : 24 + active_experts],
            expected_down_bases,
        )

        expected_mxfp_data, expected_mxfp_scales = (
            self._mxfp8_ffn_input_reference(actual_ffn_normalized)
        )
        active_scale_indices = (
            torch.arange(8, device=self.device).reshape(-1, 1) * 16
            + torch.arange(4, device=self.device).reshape(1, -1)
        ).reshape(-1)
        actual_record_scales = self.mxfp_input_records[:, 4096:].reshape(
            8, 4, 512
        )[:, :, active_scale_indices]
        actual_split_scales = self.mxfp_activation_scales.reshape(
            8, 4, 512
        )[:, :, active_scale_indices]
        self._report_exact_boundary(
            "mxfp8_input_record_data",
            self.mxfp_input_records[:, :4096],
            expected_mxfp_data,
        )
        self._report_exact_boundary(
            "mxfp8_input_record_scales",
            actual_record_scales,
            expected_mxfp_scales,
        )
        self._report_exact_boundary(
            "mxfp8_input_split_data",
            self.mxfp_activation_data,
            expected_mxfp_data,
        )
        self._report_exact_boundary(
            "mxfp8_input_split_scales",
            actual_split_scales,
            expected_mxfp_scales,
        )
        self.attention_correctness_reference = {
            "attention_post": attention_post,
            "ffn_normalized": ffn_normalized,
            "ffn_post": ffn_post_reference,
            "ffn_comb": ffn_comb_reference,
            "route_indices": reference_route_indices,
            "route_weights": reference_route_weights,
        }

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

    def report_cross_layer_hc_barrier_profile(
        self,
        profile: torch.Tensor | None = None,
        *,
        sample_index: int | None = None,
        sample_cuda_ms: float | None = None,
    ) -> None:
        if not self.args.profile_cross_layer_hc_barrier:
            return
        if profile is None:
            profile = self.launcher.profile.cpu()
        magic = 0x4454524B50524631
        if any(int(value) != magic for value in profile[:, 127]):
            raise RuntimeError(
                "cross-layer HC barrier profiling requires track_profile=1"
            )

        producer_events = (
            HC_GLOBAL_RESIDENT_COMPUTE_DONE_EVENT,
            FFN_OUTPUT_PROFILE_EVENT_BASE,
            FFN_OUTPUT_PROFILE_EVENT_BASE + 1,
            FFN_OUTPUT_PROFILE_EVENT_BASE + 2,
            HC_GLOBAL_RAW_PREVIOUS_VALUE_EVENT,
        )
        record_events = (
            HC_GLOBAL_RECORD_WAIT_BEGIN_EVENT,
            HC_GLOBAL_RECORD_WAIT_VALUE_EVENT,
            HC_GLOBAL_RECORD_WAIT_END_EVENT,
            HC_GLOBAL_RECORD_COMMAND_END_EVENT,
            25,
            26,
            27,
            28,
            29,
        )
        reload_events = (
            HC_GLOBAL_RELOAD_BEGIN_EVENT,
            HC_GLOBAL_RELOAD_VALUE_EVENT,
            HC_GLOBAL_RELOAD_READY_EVENT,
            HC_GLOBAL_RELOAD_STORE_EVENT,
            HC_GLOBAL_RELOAD_END_EVENT,
        )
        if any(
            int(profile[sm, event]) == 0
            for sm in range(self.sms)
            for event in producer_events + reload_events
        ):
            raise RuntimeError("HC producer/reload trace is incomplete")
        if any(
            int(profile[sm, event]) == 0
            for sm in range(SchedDsv4Fp32Bf16Gemv.FUSED_TASK_SMS)
            for event in record_events
        ):
            raise RuntimeError("HC record-consumer trace is incomplete")

        def values(event, count=self.sms):
            return [int(profile[sm, event]) for sm in range(count)]

        raw_previous_packed = values(HC_GLOBAL_RAW_PREVIOUS_VALUE_EVENT)
        raw_bars = [packed >> 32 for packed in raw_previous_packed]
        raw_previous = [packed & 0xFFFFFFFF for packed in raw_previous_packed]
        expected_previous = list(range(1, self.sms + 1))
        exact_decrement_generation = sorted(raw_previous) == expected_previous
        record_values_packed = values(
            HC_GLOBAL_RECORD_WAIT_VALUE_EVENT,
            SchedDsv4Fp32Bf16Gemv.FUSED_TASK_SMS,
        )
        record_bars = [packed >> 32 for packed in record_values_packed]
        record_initial = [packed & 0xFFFFFFFF for packed in record_values_packed]
        reload_values_packed = values(HC_GLOBAL_RELOAD_VALUE_EVENT)
        reload_bars = [packed >> 32 for packed in reload_values_packed]
        reload_initial = [packed & 0xFFFFFFFF for packed in reload_values_packed]
        one_bar = len(set(raw_bars + record_bars + reload_bars)) == 1

        raw_after = values(FFN_OUTPUT_PROFILE_EVENT_BASE + 2)
        final_raw_sm = raw_previous.index(1) if 1 in raw_previous else -1
        final_raw_time = raw_after[final_raw_sm] if final_raw_sm >= 0 else 0
        record_wait_end = values(
            HC_GLOBAL_RECORD_WAIT_END_EVENT,
            SchedDsv4Fp32Bf16Gemv.FUSED_TASK_SMS,
        )
        record_command_end = values(
            HC_GLOBAL_RECORD_COMMAND_END_EVENT,
            SchedDsv4Fp32Bf16Gemv.FUSED_TASK_SMS,
        )
        record_compute_ready = values(
            28, SchedDsv4Fp32Bf16Gemv.FUSED_TASK_SMS
        )
        reload_ready = values(HC_GLOBAL_RELOAD_READY_EVENT)
        reload_end = values(HC_GLOBAL_RELOAD_END_EVENT)
        zero_consumed_before_reload = (
            not any(reload_initial)
            and min(reload_ready) >= max(record_command_end)
        )
        status = (
            "PASS"
            if exact_decrement_generation
            and one_bar
            and zero_consumed_before_reload
            else "FAIL"
        )
        grid_start = max(int(value) for value in profile[:, 0])
        relative_us = lambda timestamp: (timestamp - grid_start) / 1.0e3
        print(
            "DSV4_HC_GLOBAL_BARRIER_PROFILE "
            f"status={status} bar={raw_bars[0] if raw_bars else -1} "
            f"producer_count={len(raw_previous)} "
            f"previous_min={min(raw_previous)} previous_max={max(raw_previous)} "
            f"previous_unique={len(set(raw_previous))} "
            f"record_initial_min={min(record_initial)} "
            f"record_initial_max={max(record_initial)} "
            f"record_initial_zero={sum(value == 0 for value in record_initial)} "
            f"reload_entry_nonzero={sum(value != 0 for value in reload_initial)} "
            f"compute_done_frontier_us={relative_us(max(values(HC_GLOBAL_RESIDENT_COMPUTE_DONE_EVENT))):.3f} "
            f"raw_zero_sm={final_raw_sm} "
            f"raw_zero_recorded_us={relative_us(final_raw_time):.3f} "
            f"record_zero_frontier_us={relative_us(max(record_wait_end)):.3f} "
            f"record_issue_frontier_us={relative_us(max(record_command_end)):.3f} "
            f"record_compute_ready_frontier_us={relative_us(max(record_compute_ready)):.3f} "
            f"reload_ready_first_us={relative_us(min(reload_ready)):.3f} "
            f"reload_ready_frontier_us={relative_us(max(reload_ready)):.3f} "
            f"reload_end_frontier_us={relative_us(max(reload_end)):.3f} "
            f"sample_index={sample_index if sample_index is not None else -1} "
            f"sample_cuda_ms={sample_cuda_ms if sample_cuda_ms is not None else -1.0:.6f}",
            flush=True,
        )
        if status != "PASS":
            raise RuntimeError("VDcores cross-layer global-barrier trace failed")

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


def build_argument_parser() -> argparse.ArgumentParser:
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
        "--stop-after-layer",
        type=int,
        help=(
            "build the production-model prefix through this zero-based layer, "
            "omit the head, and read the existing post-layer residual from HBM"
        ),
    )
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
        "--diagnose-cross-layer-hc-boundary",
        action="store_true",
        help=(
            "copy the HCA record and coefficients into dedicated HBM at the "
            "fused boundary, then make the fused consumer read the snapshots"
        ),
    )
    parser.add_argument(
        "--inspect-cross-layer-hc-barrier",
        action="store_true",
        help="print the fused record's VDcores global-counter contract and exit",
    )
    parser.add_argument(
        "--profile-cross-layer-hc-barrier",
        action="store_true",
        help=(
            "trace the live VDcores producer counter, fused record load, and "
            "first barrier reload in a fused two-layer HCA-to-CSA run"
        ),
    )
    parser.add_argument(
        "--stop-after-cross-layer-hc-write",
        action="store_true",
        help=(
            "run the integrated HCA-to-CSA chain only through its fused HC "
            "writer, leaving that writer's residual and partial metadata as "
            "the final HBM generation for exact readback"
        ),
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
        "--hidden-reference",
        help=(
            "host-side diagnostic file from deepseek_v4_checkpoint_decode.py; "
            "one-layer runs consume that layer's recorded input and compare "
            "their post-layer residual, while 43-layer runs compare the final residual"
        ),
    )
    parser.add_argument(
        "--dump-final-hidden",
        help="write the final residual and normalized LM-head input after the run",
    )
    parser.add_argument(
        "--omit-head",
        action="store_true",
        help=(
            "diagnostically stop after the requested transformer body without "
            "adding LM-head stages"
        ),
    )
    parser.add_argument(
        "--stop-after-stage",
        type=int,
        help=(
            "with one layer and --omit-head, terminate after this zero-based "
            "existing stage without inserting a capture operator"
        ),
    )
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
        "--diagnose-attention-correctness",
        action="store_true",
        help=(
            "post-launch compare every retained one-layer attention boundary "
            "with a direct checkpoint/PyTorch oracle"
        ),
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
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    cfg = DeepSeekV4FlashConfig()
    if not 0 <= args.token_id < cfg.vocab_size:
        parser.error("token-id is outside the vocabulary")
    if not 0 <= args.single_layer_id < cfg.num_layers:
        parser.error("single-layer-id is outside the transformer")
    if args.stop_after_layer is not None:
        if args.layers != cfg.num_layers:
            parser.error("--stop-after-layer requires --layers=43")
        if not 0 <= args.stop_after_layer < cfg.num_layers:
            parser.error("stop-after-layer is outside the transformer")
        if args.loopback_hc_fusion and args.stop_after_layer < 4:
            parser.error(
                "a loopback-fused prefix requires a terminal layer at or "
                "after the first HCA/CSA pair"
            )
        if args.expected_token_id is not None:
            parser.error("--expected-token-id requires the LM head")
    if args.layers != 1 and args.single_layer_id != 0:
        parser.error("single-layer-id is only valid with --layers=1")
    if args.stop_after_stage is not None and not (
        args.layers == 1
        and args.omit_head
        and args.stop_after_stage >= 0
    ):
        parser.error(
            "--stop-after-stage requires --layers=1, --omit-head, and a "
            "non-negative stage index"
        )
    if args.diagnose_attention_correctness and (
        args.layers != 1
        or not {"q_a", "q_b", "kv", "o_a"}.issubset(
            {
                item.strip()
                for item in args.fp8_splitk_components.split(",")
                if item.strip()
            }
            if args.fp8_splitk_components.strip() != "all"
            else {"q_a", "q_b", "kv", "index_q_b", "o_a", "o_b"}
        )
    ):
        parser.error(
            "--diagnose-attention-correctness requires one layer and the "
            "native q_a/q_b/kv/o_a path"
        )
    if args.layers != 2 and args.two_layer_start_id != 0:
        parser.error("two-layer-start-id is only valid with --layers=2")
    if args.diagnose_cross_layer_hc_boundary and (
        args.layers != 2 or args.disable_cross_layer_hc_fusion
    ):
        parser.error(
            "--diagnose-cross-layer-hc-boundary requires a fused two-layer run"
        )
    if args.stop_after_cross_layer_hc_write and (
        args.layers != 2
        or args.two_layer_start_id != 3
        or args.repeat_same_layer
        or args.disable_cross_layer_hc_fusion
        or args.two_layer_pair_repeats != 1
    ):
        parser.error(
            "--stop-after-cross-layer-hc-write requires the fused layer-3 "
            "HCA to layer-4 CSA two-layer run"
        )
    if (
        args.stop_after_cross_layer_hc_write
        and args.expected_token_id is not None
    ):
        parser.error(
            "--expected-token-id is unavailable when stopping at the HC writer"
        )
    if args.inspect_cross_layer_hc_barrier and (
        args.layers != 2 or args.disable_cross_layer_hc_fusion
    ):
        parser.error(
            "--inspect-cross-layer-hc-barrier requires a fused two-layer run"
        )
    if args.layers != 2 and args.two_layer_pair_repeats != 1:
        parser.error("two-layer-pair-repeats is only valid with --layers=2")
    if args.loopback_hc_fusion and (
        args.layers != cfg.num_layers
        or args.disable_cross_layer_hc_fusion
    ):
        parser.error(
            "--loopback-hc-fusion requires the full/prefix model and forward "
            "fusion"
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
            args.profile_cross_layer_hc_barrier,
        )
    )
    if profile_modes > 1:
        parser.error("profiling modes are mutually exclusive")
    if args.stop_after_layer is not None and profile_modes:
        parser.error("--stop-after-layer is a hidden-state diagnostic, not profiling")
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
    if args.profile_cross_layer_hc_barrier and (
        args.layers != 2
        or args.disable_cross_layer_hc_fusion
        or cfg.attention_kind(args.two_layer_start_id) != "hca"
        or cfg.attention_kind(args.two_layer_start_id + 1) != "csa"
    ):
        parser.error(
            "cross-layer HC barrier profiling requires a fused adjacent "
            "HCA-to-CSA two-layer run"
        )
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
    if args.inspect_cross_layer_hc_barrier:
        record_commands = []
        for sm, instructions in enumerate(flow.program.instructions):
            for pc, inst in enumerate(instructions):
                if (
                    isinstance(inst, MemoryInstruction)
                    and inst.annotation.get("fused_hc_record")
                ):
                    record_commands.append((sm, pc, inst))
        if not record_commands:
            raise RuntimeError("fused HC record command was not found")
        for record_bar in sorted(
            {inst.num_slots >> 6 for _, _, inst in record_commands}
        ):
            selected_records = [
                item
                for item in record_commands
                if item[2].num_slots >> 6 == record_bar
            ]
            opcode_counts = {}
            reloads = []
            async_reloads = []
            reload_opcode = runtime.opcode.OP_LDU_RELOAD_BARRIERS & ~0x3F
            async_reload_opcode = (
                runtime.opcode.OP_LDU_ASYNC_RELOAD_BARRIERS & ~0x3F
            )
            for sm, instructions in enumerate(flow.program.instructions):
                for pc, inst in enumerate(instructions):
                    if not isinstance(inst, MemoryInstruction):
                        continue
                    if inst.opcode & 0x10 and inst.num_slots >> 6 == record_bar:
                        key = (
                            inst.opcode & ~0x3F,
                            bool(inst.opcode & 0x1),
                            bool(inst.opcode & 0x2),
                        )
                        opcode_counts[key] = opcode_counts.get(key, 0) + 1
                    base_opcode = inst.opcode & ~0x3F
                    if base_opcode == reload_opcode:
                        source_first_bar = (
                            inst.arg & LduReloadBarriers.FIRST_BAR_MASK
                        )
                        completion_bar = inst.num_slots >> 6
                        first_bar = completion_bar + 1 - inst.size
                        if first_bar <= record_bar < first_bar + inst.size:
                            reloads.append(
                                (
                                    sm,
                                    pc,
                                    first_bar,
                                    inst.size,
                                    completion_bar,
                                    source_first_bar,
                                )
                            )
                    elif base_opcode == async_reload_opcode:
                        first_bar = inst.arg & ((1 << 10) - 1)
                        count = inst.size & ((1 << 6) - 1)
                        input_bar = inst.size >> 6
                        if first_bar <= record_bar < first_bar + count:
                            async_reloads.append(
                                (
                                    sm,
                                    pc,
                                    first_bar,
                                    count,
                                    input_bar,
                                    inst.num_slots >> 6,
                                    bool(inst.opcode & 0x4),
                                )
                            )
            bar_source_value = int(
                flow.launcher.bars_src.view(torch.uint32)[record_bar].item()
            )
            live_bar_value = int(
                flow.launcher.bars.view(torch.uint32)[record_bar].item()
            )
            async_dependency_details = {}
            for input_bar in sorted({item[4] for item in async_reloads}):
                producer_opcodes = {}
                for instructions in flow.program.instructions:
                    for inst in instructions:
                        if (
                            isinstance(inst, MemoryInstruction)
                            and inst.opcode & 0x10
                            and inst.opcode & 0x2
                            and inst.num_slots >> 6 == input_bar
                        ):
                            opcode_key = inst.opcode & ~0x3F
                            producer_opcodes[opcode_key] = (
                                producer_opcodes.get(opcode_key, 0) + 1
                            )
                async_dependency_details[input_bar] = (
                    flow.launcher.bar_values.get(input_bar),
                    int(
                        flow.launcher.bars_src.view(torch.uint32)[
                            input_bar
                        ].item()
                    ),
                    sorted(producer_opcodes.items()),
                )
            raw_opcode = runtime.opcode.OP_ALLOC_WB_RAW_ADDRESS & ~0x3F
            raw_reuse_counts = {}
            raw_special_slots = set()
            for instructions in flow.program.instructions:
                memory_stream = [
                    inst
                    for inst in instructions
                    if isinstance(inst, MemoryInstruction)
                ]
                for memory_pc, inst in enumerate(memory_stream):
                    if (
                        inst.opcode & ~0x3F != raw_opcode
                        or not inst.opcode & 0x2
                        or not inst.opcode & 0x10
                        or inst.num_slots >> 6 != record_bar
                    ):
                        continue
                    special_slot = inst.num_slots & 0x3F
                    raw_special_slots.add(special_slot)
                    reuse = None
                    for next_pc in range(memory_pc + 1, len(memory_stream)):
                        candidate = memory_stream[next_pc]
                        if candidate.num_slots & 0x3F == special_slot:
                            reuse = (
                                next_pc - memory_pc,
                                candidate.opcode & ~0x3F,
                                candidate.opcode & 0x3F,
                                candidate.num_slots >> 6,
                            )
                            break
                    raw_reuse_counts[reuse] = raw_reuse_counts.get(reuse, 0) + 1
            print(
                "DSV4_HC_RECORD_BARRIER "
                f"bar={record_bar} "
                f"initial={flow.launcher.bar_values[record_bar]} "
                f"source={bar_source_value} live_before_launch={live_bar_value} "
                f"record_loads={len(selected_records)} "
                f"prefetch_loads={sum(bool(inst.annotation.get('prefetch_before_resident_reset')) for _, _, inst in selected_records)} "
                f"record_sms={selected_records[0][0]}-{selected_records[-1][0]} "
                f"raw_special_slots={sorted(raw_special_slots)} "
                f"raw_next_reuse={sorted(raw_reuse_counts.items(), key=lambda item: str(item[0]))} "
                f"opcode_counts={sorted(opcode_counts.items())} "
                f"reloads={len(reloads)} reload_sample={reloads[:4]} "
                f"async_reloads={len(async_reloads)} "
                f"async_reload_sample={async_reloads[:4]} "
                f"async_dependencies={async_dependency_details}",
                flush=True,
            )
        return
    hidden_reference = None
    if args.hidden_reference is not None:
        hidden_reference = torch.load(
            args.hidden_reference, map_location="cpu", weights_only=True
        )
        expected_reference_format = (
            "mxfp4",
            "mxfp8_e4m3_group32",
        )
        actual_reference_format = (
            hidden_reference.get("ffn_weight_format"),
            hidden_reference.get("ffn_activation_format"),
        )
        if actual_reference_format != expected_reference_format:
            raise ValueError(
                "hidden reference does not use the production FFN formats: "
                f"expected={expected_reference_format} "
                f"actual={actual_reference_format}; regenerate it with "
                "deepseek_v4_checkpoint_torch_reference.py"
            )
        if args.stop_after_layer is not None:
            reference_input = hidden_reference["pre_layer"][0]
        elif args.layers == 1:
            reference_input = hidden_reference["pre_layer"][
                args.single_layer_id
            ]
        elif args.layers == 2:
            reference_input = hidden_reference["pre_layer"][
                flow.layer_ids[0]
            ]
        elif args.layers == cfg.num_layers:
            reference_input = hidden_reference["pre_layer"][0]
        else:
            parser.error(
                "--hidden-reference supports one layer, an adjacent pair, "
                "a stopped prefix, or all 43 layers"
            )
        if reference_input.shape != flow.initial_residual.shape:
            raise ValueError(
                "hidden reference input shape does not match the resident image: "
                f"reference={tuple(reference_input.shape)} "
                f"resident={tuple(flow.initial_residual.shape)}"
            )
        flow.initial_residual.copy_(reference_input.to(device=device))
    if os.environ.get("DAE_AUDIT_HC_METADATA_HBM") == "1":
        flow.launcher.prepare_launch()
        target_start = flow.mhc_fused_metadata.data_ptr()
        target_bytes = (
            flow.mhc_fused_metadata.numel()
            * flow.mhc_fused_metadata.element_size()
        )
        target_end = target_start + target_bytes
        flag_mask = (1 << 6) - 1
        explicit_store_ops = {
            getattr(runtime.opcode, name) & ~flag_mask
            for name in (
                "OP_ALLOC_WB_TMA_STORE_1D",
                "OP_ALLOC_WB_STU_STORE_1D",
            )
        }
        raw_store_op = (
            runtime.opcode.OP_ALLOC_WB_RAW_ADDRESS & ~flag_mask
        )
        overlaps = []
        raw_overlaps = []
        descriptor_overlaps = []
        for sm, builder in enumerate(flow.launcher.builder):
            for pc, inst in enumerate(builder.built_minsts):
                base_opcode = inst.opcode & ~flag_mask
                address = sum(
                    int(coord) << (16 * index)
                    for index, coord in enumerate(inst.cords)
                )
                if base_opcode in explicit_store_ops:
                    start = address
                    end = start + int(inst.size)
                    if max(start, target_start) < min(end, target_end):
                        overlaps.append(
                            (
                                sm,
                                pc,
                                base_opcode,
                                start - target_start,
                                int(inst.size),
                            )
                        )
                elif base_opcode == raw_store_op:
                    if target_start <= address < target_end:
                        raw_overlaps.append(
                            (sm, pc, address - target_start, int(inst.size))
                        )

                tensor = getattr(inst, "mat", None)
                mode = getattr(inst, "mode", None)
                if (
                    isinstance(tensor, torch.Tensor)
                    and mode in ("store", "reduce")
                ):
                    start = tensor.data_ptr()
                    end = start + tensor.numel() * tensor.element_size()
                    if max(start, target_start) < min(end, target_end):
                        descriptor_overlaps.append(
                            (
                                sm,
                                pc,
                                base_opcode,
                                start - target_start,
                                end - start,
                                mode,
                            )
                        )

        expected_partial_offsets = {
            (split * SchedDsv4Fp32Bf16Gemv.FUSED_RECORD_STRIDE + group * 4)
            * 4
            for split in range(SchedDsv4Fp32Bf16Gemv.FUSED_SPLITS)
            for group in range(SchedDsv4Fp32Bf16Gemv.FUSED_GROUPS)
        }
        partial_offsets = [
            offset
            for _, _, _, offset, size in overlaps
            if size == 16 and offset < 16 * 32 * 4
        ]
        tail_offset = 16 * 32 * 4
        tail_writes = [
            item
            for item in overlaps
            if item[3] == tail_offset
            and item[4] == SchedDsv4Fp32Bf16Gemv.FUSED_TAIL_ITEMS * 4
        ]
        unexpected = [
            item
            for item in overlaps
            if not (
                item[4] == 16
                and item[3] in expected_partial_offsets
            )
            and item not in tail_writes
        ]
        partial_write_counts = {
            offset: partial_offsets.count(offset)
            for offset in expected_partial_offsets
        }
        zero_writes = [item for item in overlaps if item[3] == 0]
        hc_writer_stages = [
            (segment_index, stage_index, stage.name)
            for segment_index, segment in enumerate(
                getattr(flow.program, "segments", (flow.program,))
            )
            for stage_index, stage in enumerate(segment.stages)
            if (
                "attn.hc_post_ffn.hc_project" in stage.name
                or "ffn.hc_post_next_attn.hc_project" in stage.name
            )
        ]
        generations = len(hc_writer_stages)
        complete_generations = (
            generations > 0
            and all(
                count == generations
                for count in partial_write_counts.values()
            )
            and len(tail_writes) == generations
        )
        passed = (
            generations == 3
            and complete_generations
            and not raw_overlaps
            and not descriptor_overlaps
            and not unexpected
        )
        print(
            "DSV4_HC_METADATA_HBM_AUDIT "
            f"target=0x{target_start:x} bytes={target_bytes} "
            f"generations={generations} "
            f"partial_writes={len(partial_offsets)} "
            f"partial_unique={len(set(partial_offsets))} "
            f"partial_missing="
            f"{len(expected_partial_offsets - set(partial_offsets))} "
            f"partial_count_values={sorted(set(partial_write_counts.values()))} "
            f"tail_writes={len(tail_writes)} "
            f"zero_writes={zero_writes} "
            f"tail_write_records={tail_writes} "
            f"hc_writer_stages={hc_writer_stages} "
            f"raw_overlaps={raw_overlaps} "
            f"descriptor_overlaps={descriptor_overlaps} "
            f"unexpected={unexpected} status="
            f"{'PASS' if passed else 'FAIL'}",
            flush=True,
        )
        return
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
    if args.diagnose_attention_correctness:
        print(
            "DSV4_ATTENTION_CORRECTNESS_SCOPE scope=prime",
            flush=True,
        )
        flow.report_attention_correctness()
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
    elif (
        args.stop_after_layer is not None
        or args.stop_after_cross_layer_hc_write
        or args.omit_head
    ):
        logit_summary = (
            "logits=omitted_hc_writer_stop"
            if args.stop_after_cross_layer_hc_write
            else (
                "logits=omitted_prefix_stop"
                if args.stop_after_layer is not None
                else "logits=omitted_diagnostic_head"
            )
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
    for _ in range(0 if args.omit_head else args.warmup):
        token, _, _ = flow.run_once()
        if args.expected_token_id is not None and token != args.expected_token_id:
            raise AssertionError(
                f"warmup emitted token {token}, expected {args.expected_token_id}"
            )

    timings = [prime_ms] if args.omit_head else []
    device_frontier_timings = (
        [flow.device_frontier_ms()] if args.omit_head else []
    )
    repeat_logit_max_abs = []
    repeat_logit_mean_abs = []
    profile_samples = []
    reference_token = prime_token if args.omit_head else None
    token = prime_token
    logits = prime_logits if args.omit_head else None
    if args.omit_head:
        print(
            "DSV4_ONE_LAUNCH_SAMPLE "
            "iteration=0 source=prime_single_launch "
            f"elapsed_ms={prime_ms:.6f} "
            f"device_frontier_ms={device_frontier_timings[0]:.6f} "
            f"output_token={prime_token}",
            flush=True,
        )
    for iteration in range(0 if args.omit_head else args.iterations):
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
    if args.diagnose_attention_correctness:
        print(
            "DSV4_ATTENTION_CORRECTNESS_SCOPE scope=final_timed",
            flush=True,
        )
        flow.report_attention_correctness()
    if hidden_reference is not None and args.stop_after_stage is None:
        if args.stop_after_layer is not None:
            reference_output = hidden_reference["post_layer"][
                args.stop_after_layer
            ].float()
            scope = "prefix"
            reference_layer = args.stop_after_layer
        elif args.layers == 1:
            reference_output = hidden_reference["post_layer"][
                args.single_layer_id
            ].float()
            scope = "layer"
            reference_layer = args.single_layer_id
        elif args.layers == 2:
            reference_output = hidden_reference["post_layer"][
                flow.layer_ids[-1]
            ].float()
            scope = "pair"
            reference_layer = flow.layer_ids[-1]
        else:
            reference_output = hidden_reference["final_residual"].float()
            scope = "full"
            reference_layer = -1
        resident_output = flow.residual.detach().cpu().float()
        delta = resident_output - reference_output
        reference_norm = float(torch.linalg.vector_norm(reference_output).item())
        delta_norm = float(torch.linalg.vector_norm(delta).item())
        cosine = float(
            torch.nn.functional.cosine_similarity(
                resident_output.reshape(1, -1),
                reference_output.reshape(1, -1),
            ).item()
        )
        print(
            "DSV4_HIDDEN_COMPARE status=DIAGNOSTIC "
            f"scope={scope} layer={reference_layer} "
            f"rel_l2={delta_norm / max(reference_norm, 1.0e-30):.9f} "
            f"cosine={cosine:.9f} "
            f"mean_abs={float(delta.abs().mean().item()):.9f} "
            f"max_abs={float(delta.abs().max().item()):.9f}",
            flush=True,
        )
    if args.dump_final_hidden is not None:
        _, final_ffn_normalized, _, _, _ = flow._mhc_outputs(
            flow.layer_ids[-1]
        )
        payload = {
            "residual": flow.residual.detach().cpu(),
            "next_residual": flow.next_residual.detach().cpu(),
            "mxfp_ffn_output": flow.mxfp_ffn_output.detach().cpu(),
            "ffn_normalized": final_ffn_normalized.detach().cpu(),
            "route_records": flow.route_records.detach().cpu(),
            "router_prepared": flow.router_prepared.detach().cpu(),
            "mxfp_input_records": flow.mxfp_input_records.detach().cpu(),
            "mxfp_activation_data": flow.mxfp_activation_data.detach().cpu(),
            "mxfp_activation_scales": (
                flow.mxfp_activation_scales.detach().cpu()
            ),
            "mxfp_middle_records": flow.mxfp_middle_records.detach().cpu(),
            "mhc_cross_layer_input_records": (
                flow.mhc_cross_layer_input_records.detach().cpu()
            ),
            "mhc_output_metadatas": (
                flow.mhc_output_metadatas.detach().cpu()
            ),
            "mhc_fused_metadata": flow.mhc_fused_metadata.detach().cpu(),
            "mhc_packed_metadata": flow.mhc_packed_metadata.detach().cpu(),
            "stop_after_layer": torch.tensor(
                -1
                if args.stop_after_layer is None
                else args.stop_after_layer,
                dtype=torch.int64,
            ),
            "stop_after_cross_layer_hc_write": torch.tensor(
                int(args.stop_after_cross_layer_hc_write),
                dtype=torch.int64,
            ),
        }
        if args.layers == 1:
            payload.update(
                {
                    "layer_ids": torch.tensor(
                        flow.layer_ids, dtype=torch.int64
                    ),
                    "route_record": flow.route_record.detach().cpu(),
                }
            )
            correctness_reference = getattr(
                flow, "attention_correctness_reference", None
            )
            if correctness_reference is not None:
                payload.update(
                    {
                        f"reference_{name}": tensor.detach().cpu()
                        for name, tensor in correctness_reference.items()
                    }
                )
        if flow.mhc_boundary_record_snapshot is not None:
            payload.update(
                {
                    "mhc_boundary_record_snapshot": (
                        flow.mhc_boundary_record_snapshot.detach().cpu()
                    ),
                    "mhc_boundary_coefficients_snapshot": (
                        flow.mhc_boundary_coefficients_snapshot.detach().cpu()
                    ),
                }
            )
        if flow.mhc_consumed_record_capture is not None:
            payload["mhc_consumed_record_capture"] = (
                flow.mhc_consumed_record_capture.detach().cpu()
            )
            payload["mhc_consumed_weight_capture"] = (
                flow.mhc_consumed_weight_capture.detach().cpu()
            )
            payload["mhc_consumed_coefficient_capture"] = (
                flow.mhc_consumed_coefficient_capture.detach().cpu()
            )
            payload["mhc_fused_weight_reference"] = (
                flow.mhc_fused_weight_reference.detach().cpu()
            )
        if (
            args.stop_after_layer is None
            and not args.stop_after_cross_layer_hc_write
            and not args.omit_head
        ):
            active_head_norm = (
                flow.head_norm[0] if flow.bf16_umma_head else flow.head_norm
            )
            payload.update(
                {
                    "head_norm": active_head_norm.detach().cpu(),
                    "output_token": torch.tensor(
                        reference_token, dtype=torch.int64
                    ),
                }
            )
        torch.save(payload, args.dump_final_hidden)
        print(
            "DSV4_FINAL_HIDDEN_DUMP status=PASS "
            f"path={args.dump_final_hidden}",
            flush=True,
        )
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
            if args.profile_cross_layer_hc_barrier:
                reporter = flow.report_cross_layer_hc_barrier_profile
            elif args.profile_fp8_coupled_detail:
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
            "repeat_logits=omitted_hc_writer_stop"
            if args.stop_after_cross_layer_hc_write
            else (
                "repeat_logits=omitted_prefix_stop"
                if args.stop_after_layer is not None
                else (
                    "repeat_logits=omitted_diagnostic_head"
                    if args.omit_head
                    else (
                        "repeat_logits=bf16_umma_argmax"
                        if flow.bf16_umma_head
                        else "repeat_logits=fp8_argmax"
                    )
                )
            )
        )
    )
    print(
        "DSV4_ONE_LAUNCH_DECODE status=PASS model_launches=1 gpu=1 "
        f"layers={len(flow.layer_ids)} token_id={args.token_id} "
        f"stop_after_layer={args.stop_after_layer if args.stop_after_layer is not None else -1} "
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
