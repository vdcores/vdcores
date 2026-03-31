from . import runtime
from .instruction_utils import decode_opcode, dedcode_opcode
from .instructions import *
from .runtime import config, opcode
from .tma_utils import *

import copy
import os
from enum import Enum
from math import prod

import numpy as np
import torch


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value not in ("0", "false", "False", "no", "NO")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _align_up(value: int, align: int) -> int:
    if align <= 1:
        return value
    return ((value + align - 1) // align) * align


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

        self.built_cinsts = []
        self.built_minsts = []

    def add(self, inst : Instruction):
        # flatten list of instructions
        if inst is None:
            pass
        elif isinstance(inst, list):
            for subi in inst:
                self.add(subi)
        # expand callable to each SM
        elif callable(inst):
            sminst = inst(self.sm_id)
            self.add(sminst)
        # add memory or compute instruction
        elif isinstance(inst, ComputeInstruction):
            self.cinsts.append(inst)
        elif isinstance(inst, MemoryInstruction):
            self.minsts.append(inst)
        else:
            raise ValueError("Unknown instruction type", inst)

    def add_compute(self, inst : ComputeInstruction):
        self.cinsts.append(inst)

    def add_memory(self, inst : MemoryInstruction):
        self.minsts.append(inst)

    def build(self,
        ctensor : torch.Tensor, cptrs: list[int],
        mtensor : torch.Tensor, mptrs: list[int]):
        # TODO(zhiyuang): now we only keep this check for not submitting "too many"
        #                 insts, but not 100% safe it won't overwrite
        assert len(self.cinsts) <= ctensor.shape[0]
        assert len(self.minsts) <= mtensor.shape[0]
        for i, inst in enumerate(self.cinsts):
            inst.tensor(ctensor[cptrs[self.sm_id],...])
            cptrs[self.sm_id] = (cptrs[self.sm_id] + 1) % ctensor.shape[0]
        for i, inst in enumerate(self.minsts):
            inst.tensor(mtensor[mptrs[self.sm_id],...])
            mptrs[self.sm_id] = (mptrs[self.sm_id] + 1) % mtensor.shape[0]

        # after building, clear the inst list to avoid duplicate build
        self.built_cinsts += self.cinsts
        self.built_minsts += self.minsts
        self.cinsts = []
        self.minsts = []

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
        tma_shift = len(self.tmas)
        bar_shift = len(self.bars)
        if self.launcher is not None:
            tma_shift *= self.launcher.tma_id_stride
            bar_shift *= self.launcher.bar_id_stride
        return tma_shift, bar_shift
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
        if name in self.bars:
            instances = self.bar_instances[name]
            assert 0 <= nnext < len(instances), f"next({name}, {nnext}) exceeds allocated barrier instances"
            return instances[nnext]
        bar_id = self.get(name)
        return bar_id + self.get_shift()[0] * nnext
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
    def __init__(self, num_sms : int = 1, device = 'cuda'):
        self.smem_size = 202 * 1024 # 202 KB
        self.num_sms = num_sms
        self.device = device

        self.max_insts = config.max_insts
        self.builder = [SMInstructionBuilder(sm_id=i) for i in range(num_sms)]
        self.profile = torch.empty((num_sms, config.num_profile_events), dtype=torch.uint64, device=self.device)

        self.cinsts = torch.empty((num_sms, self.max_insts, 8), dtype=torch.uint8)
        self.minsts = torch.empty((num_sms, self.max_insts, 16), dtype=torch.uint8)
        self.cptrs = [0 for _ in range(num_sms)]
        self.mptrs = [0 for _ in range(num_sms)]

        self.bar_id_stride = _env_int("DAE_LAUNCHER_BAR_ID_STRIDE", 1)
        self.tma_id_stride = _env_int("DAE_LAUNCHER_TMA_ID_STRIDE", 1)
        if self.bar_id_stride <= 0:
            raise ValueError(f"DAE_LAUNCHER_BAR_ID_STRIDE must be positive, got {self.bar_id_stride}")
        if self.tma_id_stride <= 0:
            raise ValueError(f"DAE_LAUNCHER_TMA_ID_STRIDE must be positive, got {self.tma_id_stride}")

        self.tmas = []
        self.num_tmas = 0

        self.need_instruction_build = True

        self.num_bars = 0
        self.bar_values = {}
        self._late_barriers_bound = False

        self.bars = torch.zeros(config.max_bars, 4, dtype=torch.uint8, device=self.device)
        self.bars_src = torch.zeros(config.max_bars, 4, dtype=torch.uint8, device=self.device)

        self.resource_groups = {
            'default': ResourceGroup('default')
        }
        self.launch_cache_window = None
        self.internal_cache_windows_enabled = True

        runtime.set_smem_size(self.smem_size)

    def _dense_tma_tensor(self):
        if len(self.tmas) == 0:
            return torch.empty((4, 128), dtype=torch.uint8, device=self.device)

        template = next((desc for desc in self.tmas if desc is not None), None)
        if template is None:
            return torch.empty((4, 128), dtype=torch.uint8, device=self.device)

        host_tmas = []
        zero_desc = torch.zeros_like(template)
        for desc in self.tmas:
            host_tmas.append(desc if desc is not None else zero_desc)
        return torch.stack(host_tmas).to(self.device)

    def _pack_metadata_window(self, tma: torch.Tensor):
        align_bytes = max(1, _env_int("DAE_LAUNCHER_METADATA_ALIGN_BYTES", 128))
        front_pad_bytes = max(0, _env_int("DAE_LAUNCHER_METADATA_FRONT_PAD_BYTES", 0))
        gap_bytes = max(0, _env_int("DAE_LAUNCHER_METADATA_GAP_BYTES", 128))
        order = os.environ.get("DAE_LAUNCHER_METADATA_ORDER", "bars,tma").replace("+", ",").replace(" ", "")
        if order not in ("bars,tma", "tma,bars"):
            raise ValueError(
                f"Unsupported DAE_LAUNCHER_METADATA_ORDER={order!r}; expected 'bars,tma' or 'tma,bars'"
            )

        bars_bytes = self.bars.numel() * self.bars.element_size()
        tma_bytes = tma.numel() * tma.element_size()
        total_bytes = front_pad_bytes
        offsets = {}

        for name, size in ((item, bars_bytes if item == "bars" else tma_bytes) for item in order.split(",")):
            total_bytes = _align_up(total_bytes, align_bytes)
            offsets[name] = total_bytes
            total_bytes += size
            total_bytes += gap_bytes

        total_bytes = _align_up(total_bytes, align_bytes)

        packed = torch.empty(total_bytes, dtype=torch.uint8, device=self.device)
        packed.zero_()
        packed[offsets["bars"] : offsets["bars"] + bars_bytes].copy_(self.bars.view(-1))
        packed[offsets["tma"] : offsets["tma"] + tma_bytes].copy_(tma.view(-1))

        bars_view = packed[offsets["bars"] : offsets["bars"] + bars_bytes].view_as(self.bars)
        tma_view = packed[offsets["tma"] : offsets["tma"] + tma_bytes].view_as(tma)
        return packed, bars_view, tma_view

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
        if bar_id >= config.max_bars:
            raise ValueError(
                f"Barrier id {bar_id} exceeds config.max_bars={config.max_bars}. "
                f"Reduce DAE_LAUNCHER_BAR_ID_STRIDE or the number of allocated barriers."
            )
        self.bar_values[bar_id] = value
        self.num_bars += self.bar_id_stride
        return bar_id
    def set_bar(self, bar_id: int, value: int):
        assert bar_id in self.bar_values, f"bar_id {bar_id} does not exist"
        assert isinstance(value, int), "bar value must be an int"
        self.bar_values[bar_id] = value
    def new_tma(self, desc: torch.Tensor) -> int:
        tma_id = self.num_tmas
        if tma_id >= config.max_tmas:
            raise ValueError(
                f"TMA descriptor id {tma_id} exceeds config.max_tmas={config.max_tmas}. "
                f"Reduce DAE_LAUNCHER_TMA_ID_STRIDE or the number of allocated descriptors."
            )
        if tma_id >= len(self.tmas):
            self.tmas.extend([None] * (tma_id - len(self.tmas) + 1))
        self.tmas[tma_id] = desc
        self.num_tmas += self.tma_id_stride
        return tma_id

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

    def build_instructions(self):
        if self.need_instruction_build:
            for i in range(self.num_sms):
                self.builder[i].build(
                    self.cinsts[i,...],
                    self.cptrs,
                    self.minsts[i,...],
                    self.mptrs,
                )
            self.need_instruction_build = False

    def set_persistent(self, *tensors):
        stream = torch.cuda.current_stream().cuda_stream
        for tensor in tensors:
            runtime.set_cache_policy(tensor, stream, 1.0, 2, 0)
    def set_streaming(self, *tensors):
        stream = torch.cuda.current_stream().cuda_stream
        for tensor in tensors:
            if isinstance(tensor, list):
                for t in tensor:
                    runtime.set_cache_policy(t, stream, 0, 0, 1)
            elif isinstance(tensor, torch.Tensor):
                runtime.set_cache_policy(tensor, stream, 0, 0, 1)
            else:
                raise ValueError("tensor must be a torch.Tensor or a list of torch.Tensor")

    def enable_internal_cache_windows(self, enabled: bool):
        self.internal_cache_windows_enabled = enabled

    def clear_launch_cache_window(self):
        self.launch_cache_window = None

    def set_launch_cache_window(self, tensor: torch.Tensor, hit_ratio: float, hit_policy: int, miss_policy: int):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("launch cache window tensor must be a torch.Tensor")
        self.launch_cache_window = (tensor, hit_ratio, hit_policy, miss_policy)

    def set_launch_persistent(self, tensor: torch.Tensor, hit_ratio: float = 1.0):
        self.set_launch_cache_window(tensor, hit_ratio, 2, 0)

    def set_launch_streaming(self, tensor: torch.Tensor):
        self.set_launch_cache_window(tensor, 0.0, 0, 1)

    def i(self, *insts):
        """Add instructions to all SM builders."""
        for inst in insts:
            for b in self.builder:
                b.add(inst)
        self.need_instruction_build = True

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

    def launch(self):
        self.build_instructions()

        supported_compute_ops = getattr(runtime, "supported_compute_ops", None)
        if supported_compute_ops is not None:
            required_compute_ops = self.compute_operator_names()
            supported_compute_ops = set(supported_compute_ops)
            missing_compute_ops = [name for name in required_compute_ops if name not in supported_compute_ops]
            if missing_compute_ops:
                rebuild_list = ",".join(required_compute_ops)
                raise ValueError(
                    "Launcher requires compute operators that are not compiled into dae.runtime: "
                    f"{missing_compute_ops}. Rebuild with DAE_COMPUTE_OPS={rebuild_list} or a superset."
                )

        unbound_bar_ids = [bar_id for bar_id, value in self.bar_values.items() if value is None]
        if unbound_bar_ids:
            raise ValueError(f"Cannot launch with unbound barrier counts: {unbound_bar_ids}")

        # Load the model using the runtime
        cinsts = self.cinsts.to(self.device).view(self.num_sms * self.max_insts, 8)
        minsts = self.minsts.to(self.device).view(self.num_sms * self.max_insts, 16)

        stream = torch.cuda.current_stream().cuda_stream
        # TODO(zhiyuang): check this?

        # init the bars based on dict
        bar_int_view = self.bars.view(torch.uint32)
        bar_src_int_view = self.bars_src.view(torch.uint32)
        for bar_id, value in self.bar_values.items():
            bar_int_view[bar_id] = value
            bar_src_int_view[bar_id] = value

        # print("bars before launch:", self.bar_values)

        tma = self._dense_tma_tensor()
        profile = self.profile.view(torch.uint8).view(self.num_sms * config.num_profile_events, 8)
        bars_for_launch = self.bars
        tma_for_launch = tma
        metadata_window = None

        internal_cache_windows_enabled = self.internal_cache_windows_enabled
        if _env_flag("DAE_LAUNCHER_INTERNAL_CACHE_WINDOW", internal_cache_windows_enabled):
            cache_mode = os.environ.get("DAE_LAUNCHER_INTERNAL_CACHE_MODE", "metadata")
            if cache_mode == "metadata":
                metadata_window, bars_for_launch, tma_for_launch = self._pack_metadata_window(tma)
                runtime.set_cache_policy(metadata_window, stream, 1, 2, 0)
                runtime.set_cache_policy(cinsts, stream, 0, 0, 1)
                runtime.set_cache_policy(minsts, stream, 0, 0, 1)
            elif cache_mode == "legacy":
                runtime.set_cache_policy(self.bars, stream, 1, 2, 0)
                runtime.set_cache_policy(tma, stream, 1, 2, 0)
                for i in range(self.num_sms // 4):
                    runtime.set_cache_policy(cinsts[i*4*self.max_insts:(i+1)*4*self.max_insts], stream, 1, 2, 0)
                    runtime.set_cache_policy(minsts[i*4*self.max_insts:(i+1)*4*self.max_insts], stream, 1, 2, 0)
            else:
                raise ValueError(
                    f"Unsupported DAE_LAUNCHER_INTERNAL_CACHE_MODE={cache_mode!r}; "
                    "expected 'metadata' or 'legacy'"
                )
        else:
            runtime.clear_cache_policy(stream)

        if self.launch_cache_window is not None:
            tensor, hit_ratio, hit_policy, miss_policy = self.launch_cache_window
            runtime.set_cache_policy(tensor, stream, hit_ratio, hit_policy, miss_policy)

        ret = runtime.launch_dae(
            self.num_sms, self.smem_size,
            cinsts, minsts, tma_for_launch,
            bars_for_launch, profile,
            stream
        )
        assert ret == 0

    def compute_operator_names(self) -> list[str]:
        return extract_compute_operator_names(self)
    
    def bench(self, iterations : int = 100,
                    total_bytes : int | None = None, total_flops : int | None = None):
        duration_ns = torch.zeros(self.num_sms, dtype=torch.uint64)
        execution_time = 0.0
        for i in range(iterations):
            self.profile.zero_()
            self.launch()

            # fetch profile data
            profile_data = self.profile[:,0:2].cpu().numpy()
            invalid_mask = profile_data[:,1] <= profile_data[:,0]
            if invalid_mask.any():
                bad_sms = np.nonzero(invalid_mask)[0].tolist()
                raise RuntimeError(
                    "Invalid DAE profile timestamps during bench iteration "
                    f"{i}: SMs {bad_sms} reported end <= start"
                )
            duration_ns += (profile_data[:,1] - profile_data[:,0])
            execution_time += profile_data[:,1].max() - profile_data[:,0].min()
        # print("SM durations (ns):", duration_ns)
        print(f"Benchmark Results on {self.num_sms} SMs and {iterations} iterations:")
        avg_duration_ns = (duration_ns.double() / iterations).mean()
        print(f"Average duration (ns): {avg_duration_ns:.2f}")
        avg_execution_time = execution_time / iterations
        print(f"Average execution time (ns): {avg_execution_time:.2f}")

        # print(duration_ns)


        if total_bytes is not None:
            bandwidth = total_bytes / (avg_duration_ns / 1e9) / (1024 **3) # GB/s
            print(f"Effective Bandwidth (GB/s): {bandwidth:.2f}")
        if total_flops is not None:
            gflops = total_flops / avg_duration_ns / 1e6
            print(f"Effective GFLOPS: {gflops:.2f}")
