from . import runtime
from .instruction_utils import decode_opcode, dedcode_opcode
from .instructions import *
from .runtime import config, opcode
from .tma_utils import *
from .core import CoreConfig, CoreKind, KernelVariant, parse_kernel_variant

import copy
import os
from enum import Enum
from math import prod

import numpy as np
import torch


def extract_compute_operator_names(launcher) -> list[str]:
    operator_names = []
    seen = set()
    for builder in launcher.builder:
        for inst in [*builder.built_cinsts, *builder.cinsts]:
            name = inst.compute_operator_name()
            if name in seen:
                continue
            seen.add(name)
            operator_names.append(name)
    return operator_names


class SMInstructionBuilder:
    def __init__(self, sm_id : int):
        self.sm_id = sm_id
        self.cinsts = []
        self.minsts = []
        self.comminsts = []
        self.poolinsts = []

        self.built_cinsts = []
        self.built_minsts = []
        self.built_comminsts = []
        self.built_poolinsts = []

    def add(self, inst : Instruction):
        # flatten list of instructions
        if inst is None:
            pass
        elif isinstance(inst, list):
            for subi in inst:
                self.add(subi)
        elif hasattr(inst, "expand_instructions"):
            self.add(inst.expand_instructions())
        # expand callable to each SM
        elif callable(inst):
            sminst = inst(self.sm_id)
            self.add(sminst)
        # add memory or compute instruction
        elif isinstance(inst, ComputeInstruction):
            self.cinsts.append(inst)
        elif isinstance(inst, MemoryInstruction):
            self.minsts.append(inst)
        elif isinstance(inst, CommunicationInstruction):
            self.comminsts.append(inst)
        elif isinstance(inst, PoolInstruction):
            self.poolinsts.append(inst)
        else:
            raise ValueError("Unknown instruction type", inst)

    def add_compute(self, inst : ComputeInstruction):
        self.cinsts.append(inst)

    def add_memory(self, inst : MemoryInstruction):
        self.minsts.append(inst)

    def add_communication(self, inst: CommunicationInstruction):
        self.comminsts.append(inst)

    def add_pool(self, inst: PoolInstruction):
        self.poolinsts.append(inst)

    def build(self,
        ctensor : torch.Tensor, cptrs: list[int],
        mtensor : torch.Tensor, mptrs: list[int],
        commtensor : torch.Tensor, commptrs: list[int],
        pooltensor : torch.Tensor, poolptrs: list[int]):
        # TODO(zhiyuang): now we only keep this check for not submitting "too many"
        #                 insts, but not 100% safe it won't overwrite
        assert len(self.cinsts) <= ctensor.shape[0]
        assert len(self.minsts) <= mtensor.shape[0]
        assert len(self.comminsts) <= commtensor.shape[0]
        assert len(self.poolinsts) <= pooltensor.shape[0]
        for i, inst in enumerate(self.cinsts):
            inst.tensor(ctensor[cptrs[self.sm_id],...])
            cptrs[self.sm_id] = (cptrs[self.sm_id] + 1) % ctensor.shape[0]
        for i, inst in enumerate(self.minsts):
            inst.tensor(mtensor[mptrs[self.sm_id],...])
            mptrs[self.sm_id] = (mptrs[self.sm_id] + 1) % mtensor.shape[0]
        for inst in self.comminsts:
            inst.tensor(commtensor[commptrs[self.sm_id], ...])
            commptrs[self.sm_id] = (
                commptrs[self.sm_id] + 1
            ) % commtensor.shape[0]
        for inst in self.poolinsts:
            inst.tensor(pooltensor[poolptrs[self.sm_id], ...])
            poolptrs[self.sm_id] = (
                poolptrs[self.sm_id] + 1
            ) % pooltensor.shape[0]

        # after building, clear the inst list to avoid duplicate build
        self.built_cinsts += self.cinsts
        self.built_minsts += self.minsts
        self.built_comminsts += self.comminsts
        self.built_poolinsts += self.poolinsts
        self.cinsts = []
        self.minsts = []
        self.comminsts = []
        self.poolinsts = []

class ResourceGroup:
    def __init__(self, name, repeat = 1):
        self.name = name
        self.repeat = repeat
        self.tmas = {}
        self.bars = {}

        self.built = False
        self.launcher = None
        self.tma_insts = {}
        self.bar_ids = {}
        self.bar_instances = {}

    def addTma(self, name: str, matList, tmaFn):
        if isinstance(matList, torch.Tensor):
            assert len(matList.shape) > 1, "matList must be a list of matrices or a 3D tensor"
            matList = [matList[i,...] for i in range(matList.shape[0])]

        assert len(matList) == self.repeat, f"tmaList length {len(matList)} does not match group size {self.repeat}"
        assert name not in self.tmas, f"TMA with name {name} already exists in the group"
        self.tmas[name] = (matList, tmaFn)

    def addBarrier(self, name: str, bar_count = None):
        if bar_count is not None:
            assert isinstance(bar_count, int), "bar_count must be an int or None"
        assert name not in self.bars, f"Barrier with name {name} already exists in the group"
        self.bars[name] = {
            "count": bar_count,
            "late_bind": bar_count is None,
        }

    def bindBarrier(self, name: str, bar_count: int):
        assert isinstance(bar_count, int), "bar_count must be an int"
        assert name in self.bars, f"Barrier with name {name} does not exist in the group"

        bar_info = self.bars[name]
        if not bar_info["late_bind"]:
            raise ValueError(f"Barrier {name} was declared with an eager count and cannot be rebound")
        if bar_info["count"] is not None:
            raise ValueError(f"Barrier {name} has already been bound")

        bar_info["count"] = bar_count
        if self.built:
            for bar_id in self.bar_instances[name]:
                self.launcher.set_bar(bar_id, bar_count)

    def bindBarriersFromCounts(self, bar_counts: dict[int, int]):
        unresolved = []
        for name, bar_info in self.bars.items():
            if not bar_info["late_bind"] or bar_info["count"] is not None:
                continue

            matched_counts = {
                bar_counts[bar_id]
                for bar_id in self.bar_instances.get(name, [])
                if bar_id in bar_counts
            }
            if len(matched_counts) == 0:
                unresolved.append(name)
                continue
            if len(matched_counts) != 1:
                raise ValueError(f"Barrier {name} observed inconsistent release counts: {sorted(matched_counts)}")

            self.bindBarrier(name, matched_counts.pop())

        if unresolved:
            raise ValueError(f"Could not infer release counts for late-bound barriers in group {self.name}: {unresolved}")

    def get_shift(self):
        return len(self.tmas), len(self.bars)
    def get(self, name: str):
        assert self.built, "ResourceGroup must be built before getting resource ids"
        if name in self.tmas:
            return self.tma_insts[name]
        elif name in self.bars:
            return self.bar_ids[name]
        else:
            raise ValueError(f"Resource with name {name} not found in the group")
    def next(self, name: str, nnext : int = 1):
        assert self.built, "ResourceGroup must be built before getting resource ids"
        bar_id = self.get(name)
        return bar_id + len(self.bars) * nnext
    def over(self, name: str):
        return self.next(name, self.repeat)
    def __getitem__(self, name: str):
        return self.get(name)

    def range_bars(self):
        assert self.built, "ResourceGroup must be built before getting start id"
        bars_min = min(self.bar_ids.values())
        bars_max = max(self.bar_ids.values())
        assert bars_min % 4 == 0, "bar ids must be aligned to 16 bytes for efficient encoding"
        while bars_max % 4 != 3:
            bars_max += 1
        return bars_min, bars_max + 1

    # TODO(zhiyuang): make this a dae callback
    def build(self, launcher):
        if self.built:
            return
        self.launcher = launcher

        for i in range(self.repeat):
            for name, (matList, tmaFn) in self.tmas.items():
                tmaInst = TmaTensor(launcher, matList[i])
                tma_id = tmaFn(tmaInst)
                if self.repeat > 1:
                    tmaInst.group()
                tmaInst.annotation['group'] = self.name
                tmaInst.annotation['tensor'] = name
                if i == 0:
                    self.tma_insts[name] = tmaInst

        # Schedules may still use next()/over() even when repeat == 1, so always
        # materialize the extra barrier instance for the "after the last repeat"
        # state instead of special-casing single-repeat groups.
        num_bar_repeat = self.repeat + 1
        for i in range(num_bar_repeat):
            for name, bar_info in self.bars.items():
                bar_id = launcher.new_bar(bar_info["count"])
                self.bar_instances.setdefault(name, []).append(bar_id)
                if i == 0:
                    self.bar_ids[name] = bar_id
        
        # align to 16 bytes
        while launcher.num_bars % 4 != 0:
            launcher.new_bar(0)

        self.built = True

class Launcher:
    def __init__(
        self,
        num_sms: int = 1,
        device='cuda',
        *,
        signal_array: torch.Tensor | None = None,
        benchmark_barrier=None,
        kernel_variant: KernelVariant | str | int = KernelVariant.AUTO,
    ):
        self.smem_size = 202 * 1024 # 202 KB
        self.num_sms = num_sms
        self.device = device
        self.signal_array = self._validate_signal_array(signal_array)
        self.kernel_variant = parse_kernel_variant(kernel_variant)
        if benchmark_barrier is not None and not callable(benchmark_barrier):
            raise TypeError("benchmark_barrier must be callable or None")
        self._benchmark_barrier = benchmark_barrier

        self.max_insts = config.max_insts
        self.max_comm_insts = config.max_comm_insts
        self.max_pool_insts = config.max_pool_insts
        self.builder = [SMInstructionBuilder(sm_id=i) for i in range(num_sms)]
        self.profile = torch.empty((num_sms, config.num_profile_events), dtype=torch.uint64, device=self.device)

        self.cinsts = torch.empty((num_sms, self.max_insts, 8), dtype=torch.uint8)
        self.minsts = torch.empty((num_sms, self.max_insts, 16), dtype=torch.uint8)
        # COMM_TERMINATE is zero, so untouched communication streams terminate
        # immediately without changing existing programs.
        self.comminsts = torch.zeros(
            (num_sms, self.max_comm_insts, 16), dtype=torch.uint8
        )
        # Ordinary compute/memory programs keep one inert slot per block.
        # Expand to the compile-time PoolInst stride only when a pool program
        # is actually assembled, so the new queue has no allocation/copy cost
        # for existing operators.
        self.poolinsts = torch.zeros((num_sms, 1, 16), dtype=torch.uint8)
        self.cptrs = [0 for _ in range(num_sms)]
        self.mptrs = [0 for _ in range(num_sms)]
        self.commptrs = [0 for _ in range(num_sms)]
        self.poolptrs = [0 for _ in range(num_sms)]
        self.core_configs: list[CoreConfig | None] = [None] * num_sms

        self.tmas = []

        self.need_instruction_build = True

        self.num_bars = 0
        self.bar_values = {}
        self._late_barriers_bound = False

        self.bars = torch.zeros(config.max_bars, 4, dtype=torch.uint8, device=self.device)
        self.bars_src = torch.zeros(config.max_bars, 4, dtype=torch.uint8, device=self.device)
        self._bar_source_dirty = True

        self.resource_groups = {
            'default': ResourceGroup('default')
        }

        self._cache_window_override = None
        self._cache_window_requests = []
        self._cache_window_disabled = False
        # Immutable instruction/core/TMA state is materialized on the device
        # once and reused across launches. Schedule edits invalidate it.
        self._launch_packet = None

        runtime.set_smem_size(self.smem_size)

    def _validate_signal_array(
        self, signal_array: torch.Tensor | None
    ) -> torch.Tensor | None:
        if signal_array is None:
            return None
        if not isinstance(signal_array, torch.Tensor):
            raise TypeError("signal_array must be a torch.Tensor or None")
        if not signal_array.is_cuda:
            raise ValueError("signal_array must be a CUDA tensor")
        if signal_array.dtype != torch.uint64:
            raise ValueError("signal_array must have dtype torch.uint64")
        if signal_array.ndim != 1:
            raise ValueError("signal_array must be rank-1")
        if not signal_array.is_contiguous():
            raise ValueError("signal_array must be contiguous")

        launcher_device = torch.device(self.device)
        if (
            launcher_device.index is not None
            and signal_array.device.index != launcher_device.index
        ):
            raise ValueError(
                f"signal_array is on {signal_array.device}, not {launcher_device}"
            )
        return signal_array

    # resource management functions
    def add_group(self, name, size):
        assert name not in self.resource_groups, f"Resource group with name {name} already exists"
        self.resource_groups[name] = ResourceGroup(name, size)
        return self.resource_groups[name]
    def get_group(self, name = 'default'):
        assert name in self.resource_groups, f"Resource group with name {name} does not exist"
        return self.resource_groups[name]
    def build_groups(self):
        for group in self.resource_groups.values():
            group.build(self)

    def new_bar(self, value: int | None):
        bar_id = self.num_bars
        self.bar_values[bar_id] = value
        self.num_bars += 1
        self._bar_source_dirty = True
        return bar_id
    def set_bar(self, bar_id: int, value: int):
        assert bar_id in self.bar_values, f"bar_id {bar_id} does not exist"
        assert isinstance(value, int), "bar value must be an int"
        self.bar_values[bar_id] = value
        self._bar_source_dirty = True
    def new_tma(self, desc: torch.Tensor) -> int:
        self.tmas.append(desc)
        self._launch_packet = None
        return len(self.tmas) - 1

    # instruction management
    def _projected_cptrs(self):
        return [
            (self.cptrs[i] + len(self.builder[i].cinsts)) % self.max_insts
            for i in range(self.num_sms)
        ]

    def _projected_mptrs(self):
        return [
            (self.mptrs[i] + len(self.builder[i].minsts)) % self.max_insts
            for i in range(self.num_sms)
        ]

    def copy_cptrs(self):
        return self._projected_cptrs()
    def copy_mptrs(self):
        return self._projected_mptrs()

    def _projected_commptrs(self):
        return [
            (self.commptrs[i] + len(self.builder[i].comminsts))
            % self.max_comm_insts
            for i in range(self.num_sms)
        ]

    def copy_commptrs(self):
        return self._projected_commptrs()

    def set_core(self, sm_id: int, core: CoreConfig) -> None:
        if not 0 <= sm_id < self.num_sms:
            raise IndexError("sm_id is outside the launched block range")
        if not isinstance(core, CoreConfig):
            raise TypeError("core must be a dae.core.CoreConfig")
        self.core_configs[sm_id] = core
        self._launch_packet = None

    def _ensure_pool_instruction_storage(self) -> None:
        has_pool_program = any(
            builder.poolinsts or builder.built_poolinsts
            for builder in self.builder
        )
        if not has_pool_program or self.poolinsts.shape[1] == self.max_pool_insts:
            return
        expanded = torch.zeros(
            (self.num_sms, self.max_pool_insts, 16), dtype=torch.uint8
        )
        expanded[:, : self.poolinsts.shape[1], :] = self.poolinsts
        self.poolinsts = expanded

    def build_instructions(self):
        if self.need_instruction_build:
            self._ensure_pool_instruction_storage()
            self._launch_packet = None
            for i in range(self.num_sms):
                self.builder[i].build(
                    self.cinsts[i,...],
                    self.cptrs,
                    self.minsts[i,...],
                    self.mptrs,
                    self.comminsts[i, ...],
                    self.commptrs,
                    self.poolinsts[i, ...],
                    self.poolptrs,
                )
            self.need_instruction_build = False

    def _cache_window_dict(self, tensor, *, hit_ratio, hit_policy, miss_policy, num_bytes=None):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("tensor must be a torch.Tensor")
        if num_bytes is not None:
            num_bytes = int(num_bytes)
            if num_bytes <= 0:
                raise ValueError("num_bytes must be positive when provided")
        return {
            "tensor": tensor,
            "hit_ratio": float(hit_ratio),
            "hit_policy": int(hit_policy),
            "miss_policy": int(miss_policy),
            "num_bytes": num_bytes,
        }

    def _flatten_cache_tensors(self, *tensors):
        for tensor in tensors:
            if tensor is None:
                continue
            if isinstance(tensor, (list, tuple)):
                yield from self._flatten_cache_tensors(*tensor)
            elif isinstance(tensor, torch.Tensor):
                yield tensor
            else:
                raise ValueError("tensor must be a torch.Tensor or a list/tuple of torch.Tensor")

    def set_cache_window(self, tensor, *, hit_ratio=1.0, hit_policy=2, miss_policy=0, num_bytes=None):
        self._cache_window_disabled = False
        self._cache_window_override = self._cache_window_dict(
            tensor,
            hit_ratio=hit_ratio,
            hit_policy=hit_policy,
            miss_policy=miss_policy,
            num_bytes=num_bytes,
        )

    def clear_cache_window(self):
        self._cache_window_disabled = False
        self._cache_window_override = None
        self._cache_window_requests.clear()

    def disable_cache_window(self):
        self._cache_window_disabled = True
        self._cache_window_override = None
        self._cache_window_requests.clear()

    def set_persistent(self, *tensors, num_bytes=None, hit_ratio=1.0):
        self._cache_window_disabled = False
        for tensor in self._flatten_cache_tensors(*tensors):
            self._cache_window_requests.append(
                self._cache_window_dict(
                    tensor,
                    hit_ratio=hit_ratio,
                    hit_policy=2,
                    miss_policy=0,
                    num_bytes=num_bytes,
                )
            )

    def set_streaming(self, *tensors, num_bytes=None):
        self._cache_window_disabled = False
        for tensor in self._flatten_cache_tensors(*tensors):
            self._cache_window_requests.append(
                self._cache_window_dict(
                    tensor,
                    hit_ratio=0.0,
                    hit_policy=0,
                    miss_policy=1,
                    num_bytes=num_bytes,
                )
            )

    def _select_requested_cache_window(self):
        if self._cache_window_override is not None:
            return self._cache_window_override
        if not self._cache_window_requests:
            return None

        selection = os.environ.get("DAE_CACHE_REQUEST_SELECTION", "last").strip().lower()
        if selection == "first":
            return self._cache_window_requests[0]
        if selection == "last":
            return self._cache_window_requests[-1]
        if selection == "largest":
            return max(
                self._cache_window_requests,
                key=lambda req: req["tensor"].numel() * req["tensor"].element_size(),
            )
        raise ValueError(f"Unsupported DAE_CACHE_REQUEST_SELECTION={selection!r}")

    def _select_launch_cache_window(
        self, *, bars, tma, cinsts, minsts, comminsts, poolinsts
    ):
        if self._cache_window_disabled:
            return None
        requested = self._select_requested_cache_window()
        if requested is not None:
            return requested

        target = os.environ.get("DAE_LAUNCH_CACHE_WINDOW", "minsts").strip().lower()
        if target in {"", "none", "off"}:
            return None

        target_map = {
            "bars": bars,
            "tma": tma,
            "cinsts": cinsts,
            "minsts": minsts,
            "comminsts": comminsts,
            "poolinsts": poolinsts,
            "cinsts_tail": cinsts[-4 * self.max_insts :],
            "minsts_tail": minsts[-4 * self.max_insts :],
        }
        if target not in target_map:
            raise ValueError(
                "Unsupported DAE_LAUNCH_CACHE_WINDOW="
                f"{target!r} (expected none/bars/tma/cinsts/minsts/comminsts/"
                "poolinsts/"
                "cinsts_tail/minsts_tail)"
            )

        num_bytes_raw = os.environ.get("DAE_LAUNCH_CACHE_WINDOW_BYTES")
        num_bytes = None if num_bytes_raw in (None, "") else int(num_bytes_raw)
        return self._cache_window_dict(
            target_map[target],
            hit_ratio=1.0,
            hit_policy=2,
            miss_policy=0,
            num_bytes=num_bytes,
        )

    def i(self, *insts):
        """Add instructions to all SM builders."""
        for inst in insts:
            for b in self.builder:
                b.add(inst)
        self.need_instruction_build = True
        self._launch_packet = None

    def collect_barrier_release_counts(self, *insts):
        counts = {}

        def merge(new_counts):
            for bar_id, count in new_counts.items():
                counts[bar_id] = counts.get(bar_id, 0) + count

        def collect(inst):
            if inst is None:
                return
            if isinstance(inst, list):
                for sub_inst in inst:
                    collect(sub_inst)
                return
            if hasattr(inst, "collect_barrier_release_counts"):
                merge(inst.collect_barrier_release_counts())
                return

        for inst in insts:
            collect(inst)

        return counts

    def bind_late_barrier_counts(self, *insts):
        if self._late_barriers_bound:
            return

        bar_counts = self.collect_barrier_release_counts(*insts)
        for group in self.resource_groups.values():
            group.bindBarriersFromCounts(bar_counts)

        self._late_barriers_bound = True

    def s(self, *schedules):
        self.i(*schedules, TerminateC(), TerminateM())

    def num_insts(self):
        ci, mi = 0, 0
        for b in self.builder:
            ci += len(b.cinsts)
            mi += len(b.minsts)
        return ci / self.num_sms, mi / self.num_sms

    def _resolve_core_configs(self) -> list[CoreConfig]:
        resolved = []
        for sm_id, builder in enumerate(self.builder):
            pool_insts = builder.built_poolinsts
            comm_insts = builder.built_comminsts
            pool_headers = [
                inst
                for inst in pool_insts
                if getattr(inst, "selects_pool_execute_warp", False)
            ]

            core = self.core_configs[sm_id]
            if core is None:
                if pool_insts:
                    core = CoreConfig.pool()
                else:
                    core = CoreConfig.compute_memory(
                        communication_warps=1 if comm_insts else 0
                    )

            if core.kind == CoreKind.POOL:
                if not pool_insts:
                    raise ValueError(f"pool core {sm_id} has no PoolInst")
                if len(pool_headers) != 1:
                    raise ValueError(
                        f"pool core {sm_id} requires one program header"
                    )
                if comm_insts:
                    raise ValueError(
                        f"pool core {sm_id} cannot execute an ordinary CommInst stream"
                    )
            elif pool_insts:
                raise ValueError(
                    f"block {sm_id} contains PoolInst but is not configured as a pool core"
                )

            if core.kind == CoreKind.COMPUTE_MEMORY:
                required_comm_warps = 1 if comm_insts else 0
                if core.communication_warps < required_comm_warps:
                    raise ValueError(
                        f"block {sm_id} has CommInst instructions but no communication warp"
                    )
            resolved.append(core)
        return resolved

    def _resolve_pool_inst_opcode(self) -> int:
        """Select the one pool-program execute-warp type in this launch.

        Operation PoolInsts do not participate in assembly selection. Different
        blocks may execute the same program type, but one CUDA grid cannot
        contain different execute-warp types: blockDim, registers, and static
        shared memory are properties of the compiled kernel.
        """
        opcodes = {
            int(inst.opcode)
            for builder in self.builder
            for inst in builder.built_poolinsts
            if getattr(inst, "selects_pool_execute_warp", False)
        }
        if not opcodes:
            return 0
        if len(opcodes) != 1:
            raise ValueError(
                "one VDCores launch can assemble only one PoolInst "
                f"execute-warp type; requested opcodes {sorted(opcodes)}"
            )

        opcode = opcodes.pop()
        execute_warp_types = getattr(runtime, "pool_execute_warp_types", {})
        supported = {int(value) for value in execute_warp_types.keys()}
        if opcode not in supported:
            raise ValueError(
                f"PoolInst opcode {opcode} has no compiled execute-warp type"
            )
        return opcode

    def _select_kernel_variant(
        self, core_configs: list[CoreConfig]
    ) -> KernelVariant:
        variant = self.kernel_variant
        if variant == KernelVariant.AUTO:
            if all(
                core.kind == CoreKind.COMPUTE_MEMORY
                and core.load_warps == 2
                and core.communication_warps == 0
                for core in core_configs
            ):
                variant = KernelVariant.COMPUTE_MEMORY
            elif all(
                core.kind == CoreKind.COMPUTE_MEMORY
                and core.load_warps == 1
                and core.communication_warps == 0
                for core in core_configs
            ):
                variant = KernelVariant.COMPUTE_MEMORY_ONE_LOAD
            elif all(core.kind == CoreKind.POOL for core in core_configs):
                variant = KernelVariant.POOL
            elif any(core.communication_warps for core in core_configs):
                if any(core.kind == CoreKind.POOL for core in core_configs):
                    raise ValueError(
                        "PoolInst and ordinary CommInst warp types use separate "
                        "specialized runtime assemblies"
                    )
                variant = KernelVariant.RUNTIME_COMMUNICATION
            else:
                variant = KernelVariant.RUNTIME

        if variant == KernelVariant.COMPUTE_MEMORY:
            valid = all(
                core.kind == CoreKind.COMPUTE_MEMORY
                and core.load_warps == 2
                and core.communication_warps == 0
                for core in core_configs
            )
        elif variant == KernelVariant.COMPUTE_MEMORY_ONE_LOAD:
            valid = all(
                core.kind == CoreKind.COMPUTE_MEMORY
                and core.load_warps == 1
                and core.communication_warps == 0
                for core in core_configs
            )
        elif variant == KernelVariant.POOL:
            valid = all(core.kind == CoreKind.POOL for core in core_configs)
        elif variant == KernelVariant.RUNTIME:
            valid = not any(core.communication_warps for core in core_configs)
        else:
            valid = (
                variant == KernelVariant.RUNTIME_COMMUNICATION
                and not any(core.kind == CoreKind.POOL for core in core_configs)
            )
        if not valid:
            raise ValueError(
                f"core configurations are incompatible with kernel variant {variant.name}"
            )
        return variant

    def _pack_core_configs(
        self, core_configs: list[CoreConfig], variant: KernelVariant
    ) -> torch.Tensor:
        runtime_warps = int(
            config.runtime_communication_core_warps
            if variant == KernelVariant.RUNTIME_COMMUNICATION
            else config.runtime_core_warps
        )
        payload = b"".join(
            core.pack(runtime_core_warps=runtime_warps) for core in core_configs
        )
        return torch.tensor(list(payload), dtype=torch.uint8).view(
            self.num_sms, config.core_config_bytes
        ).to(self.device)

    def _prepare_launch_packet(self):
        """Materialize immutable VDCores launch state once on the device."""

        self.build_instructions()
        if self._launch_packet is not None:
            return self._launch_packet

        core_configs = self._resolve_core_configs()
        kernel_variant = self._select_kernel_variant(core_configs)
        pool_inst_opcode = self._resolve_pool_inst_opcode()
        core_config_tensor = self._pack_core_configs(
            core_configs, kernel_variant
        )

        communication_instructions = [
            inst
            for builder in self.builder
            for inst in builder.built_comminsts
        ]
        pool_instructions = [
            inst
            for builder in self.builder
            for inst in builder.built_poolinsts
        ]
        signal_requiring_instructions = [
            inst
            for inst in communication_instructions
            if getattr(inst, "requires_signal_array", False)
        ]
        signal_requiring_instructions += [
            inst
            for inst in pool_instructions
            if getattr(inst, "requires_signal_array", False)
        ]
        if communication_instructions and not bool(config.nvshmem_enabled):
            raise RuntimeError(
                "ordinary communication instructions require "
                "`make nvshmem-pyext`"
            )
        if pool_instructions and not bool(
            getattr(config, "pool_enabled", False)
        ):
            raise RuntimeError(
                "pool instructions require a compiled PoolInst backend"
            )
        if signal_requiring_instructions and self.signal_array is None:
            raise ValueError(
                "communication and memory-pool instructions require "
                "Launcher(signal_array=...)"
            )

        supported_compute_ops = getattr(
            runtime, "supported_compute_ops", None
        )
        if supported_compute_ops is not None:
            required_compute_ops = self.compute_operator_names()
            supported_compute_ops = set(supported_compute_ops)
            missing_compute_ops = [
                name
                for name in required_compute_ops
                if name not in supported_compute_ops
            ]
            if missing_compute_ops:
                rebuild_list = ",".join(required_compute_ops)
                raise ValueError(
                    "Launcher requires compute operators that are not "
                    "compiled into dae.runtime: "
                    f"{missing_compute_ops}. Rebuild with "
                    f"DAE_COMPUTE_OPS={rebuild_list} or a superset."
                )

        cinsts = self.cinsts.to(self.device).view(
            self.num_sms * self.max_insts, 8
        )
        minsts = self.minsts.to(self.device).view(
            self.num_sms * self.max_insts, 16
        )
        comminsts = self.comminsts.to(self.device).view(
            self.num_sms * self.max_comm_insts, 16
        )
        poolinsts = self.poolinsts.to(self.device).view(-1, 16)
        if len(self.tmas) == 0:
            tma = torch.empty((4, 128), dtype=torch.uint8, device=self.device)
        else:
            tma = torch.stack(self.tmas).to(self.device)
        profile = self.profile.view(torch.uint8).view(
            self.num_sms * config.num_profile_events, 8
        )
        self._launch_packet = {
            "kernel_variant": kernel_variant,
            "pool_inst_opcode": pool_inst_opcode,
            "core_config_tensor": core_config_tensor,
            "cinsts": cinsts,
            "minsts": minsts,
            "comminsts": comminsts,
            "poolinsts": poolinsts,
            "tma": tma,
            "profile": profile,
        }
        return self._launch_packet

    def _reset_barriers(self) -> None:
        """Restore all VDCores counters with one device copy per launch."""

        if self._bar_source_dirty:
            source = torch.zeros(
                (config.max_bars, 4), dtype=torch.uint8
            )
            source_values = source.view(torch.uint32)
            for bar_id, value in self.bar_values.items():
                source_values[bar_id] = value
            self.bars_src.copy_(source)
            self._bar_source_dirty = False
        self.bars.copy_(self.bars_src)

    def launch(self):
        packet = self._prepare_launch_packet()
        kernel_variant = packet["kernel_variant"]
        pool_inst_opcode = packet["pool_inst_opcode"]
        core_config_tensor = packet["core_config_tensor"]
        cinsts = packet["cinsts"]
        minsts = packet["minsts"]
        comminsts = packet["comminsts"]
        poolinsts = packet["poolinsts"]
        tma = packet["tma"]
        profile = packet["profile"]

        unbound_bar_ids = [
            bar_id
            for bar_id, value in self.bar_values.items()
            if value is None
        ]
        if unbound_bar_ids:
            raise ValueError(
                "Cannot launch with unbound barrier counts: "
                f"{unbound_bar_ids}"
            )

        stream = torch.cuda.current_stream().cuda_stream
        # TODO(zhiyuang): check this?

        self._reset_barriers()

        # print("bars before launch:", self.bar_values)

        runtime.reset_cache_policy(stream)
        cache_window = self._select_launch_cache_window(
            bars=self.bars,
            tma=tma,
            cinsts=cinsts,
            minsts=minsts,
            comminsts=comminsts,
            poolinsts=poolinsts,
        )
        if cache_window is not None:
            runtime.set_cache_policy(
                cache_window["tensor"],
                stream,
                cache_window["hit_ratio"],
                cache_window["hit_policy"],
                cache_window["miss_policy"],
                cache_window["num_bytes"] if cache_window["num_bytes"] is not None else -1,
            )

        ret = runtime.launch_dae(
            self.num_sms, self.smem_size,
            cinsts, minsts, comminsts, poolinsts, tma,
            self.bars, profile,
            stream, self.signal_array, core_config_tensor, int(kernel_variant),
            pool_inst_opcode,
        )
        assert ret == 0

    def compute_operator_names(self) -> list[str]:
        return extract_compute_operator_names(self)

    def benchmark_barrier(self):
        """Synchronize launch participants before a measured iteration."""

        if self._benchmark_barrier is not None:
            return self._benchmark_barrier()
        return None
    
    def bench(self, iterations : int = 100,
                    total_bytes : int | None = None, total_flops : int | None = None):
        execution_times = []
        duration_ns = np.zeros(self.num_sms, dtype=np.float64)
        for i in range(iterations):
            self.benchmark_barrier()
            self.launch()

            # fetch profile data
            profile_data = self.profile[:,0:2].cpu().numpy()
            duration_ns += profile_data[:,1] - profile_data[:,0]
            execution_times.append(profile_data[:,1].max() - profile_data[:,0].min())

        execution_times = np.asarray(execution_times, dtype=np.float64)
        avg_duration_per_sm = duration_ns / iterations
        min_execution_time = float(execution_times.min())
        median_execution_time = float(np.median(execution_times))
        mean_execution_time = float(execution_times.mean())
        max_execution_time = float(execution_times.max())

        print(f"Benchmark Results on {self.num_sms} SMs and {iterations} iterations:")
        # for sm_id, dur in enumerate(avg_duration_per_sm):
        #     print(f"SM {sm_id:3d} | Avg Duration (ns): {dur:10.2f}")
        print(f"Min execution time (ns): {min_execution_time:.2f}")
        print(f"Median execution time (ns): {median_execution_time:.2f}")
        print(f"Average execution time (ns): {mean_execution_time:.2f}")
        print(f"Max execution time (ns): {max_execution_time:.2f}")

        if total_bytes is not None:
            bandwidth = total_bytes / (mean_execution_time / 1e9) / (1024 **3) # GB/s
            print(f"Effective Bandwidth (GB/s): {bandwidth:.2f}")
        if total_flops is not None:
            gflops = total_flops / mean_execution_time / 1e6
            print(f"Effective GFLOPS: {gflops:.2f}")
