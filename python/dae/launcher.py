from . import runtime
from .runtime import opcode, config
from .tma_utils import *
import numpy as np
import torch
import copy
from math import prod
from enum import Enum

def dedcode_opcode(op: int) -> str:
    for name, value in vars(opcode).items():
        if value == op:
            return name
    # TODO(zhiyuang): try wb variant
    op = op | 2
    for name, value in vars(opcode).items():
        if value == op:
            return name

    return f"UNKNOWN_OPCODE[0x{op:04x}]"

class Instruction:
    def tensor(self, tensor: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError()

class ComputeInstruction(Instruction):
    def __init__(self, opcode : int, args : list[int]):
        self.opcode = opcode
        self.args = args
    def tensor(self, tensor: torch.Tensor | None = None) -> torch.Tensor:
        if tensor is None:
            tensor = torch.empty((4,), dtype=torch.uint16)
        else:
            tensor = tensor.view(torch.uint16)
            assert tensor.numel() == 4

        tensor[0] = self.opcode
        assert len(self.args) <= 3
        for i in range(len(self.args)):
            assert self.args[i] >= 0 and self.args[i] < 2**16, "args must be uint16"
            tensor[i+1] = self.args[i]
        return tensor.view(torch.uint8)
    def __repr__(self):
        return f"ComputeInstruction(opcode={dedcode_opcode(self.opcode)}, args={self.args})"

class TerminateC(ComputeInstruction):
    def __init__(self):
        super().__init__(opcode=opcode.OP_TERMINATEC, args=[])

class Gemv_M64N8(ComputeInstruction):
    MNK = (64, 8, 256)
    n_batch = 4
    def __init__(self, kTiles: int, nprefeth = 0, residual: bool = False):
        super().__init__(opcode=opcode.OP_GEMV_M64N8, args=[kTiles, nprefeth])
class Gemv_M128N8(ComputeInstruction):
    MNK = (128, 8, 128)
    n_batch = 4
    def __init__(self, kTiles: int, nprefeth = 0, residual: bool = False):
        super().__init__(opcode=opcode.OP_GEMV_M128N8, args=[kTiles, nprefeth])
class Gemv_M64N8_ROPE_128(ComputeInstruction):
    MNK = (64, 8, 256)
    n_batch = 4
    def __init__(self, kTiles: int, hist_len: int, head_dim_ofst: int):
        super().__init__(opcode=opcode.OP_GEMV_M64N8_ROPE_128, args=[kTiles, hist_len, head_dim_ofst])
class Gemv_M192N16(ComputeInstruction):
    MNK = (192, 8, 128)
    def __init__(self, kTiles: int):
        super().__init__(opcode=opcode.OP_GEMV_M192, args=[kTiles])
class WGMMA_64x256x64_F16(ComputeInstruction):
    MNK = (64, 64, 256)
    def __init__(self, mTiles, kTiles, residual: bool = False):
        residual_flag = 1 if residual else 0
        # currently we only support WMMA 16x16x16 FP16
        super().__init__(opcode=opcode.OP_WGMMA_M64N256K16_F16, args=[mTiles, kTiles, residual_flag])
class WGMMA_64x256x64_BF16(ComputeInstruction):
    MNK = (64, 64, 128)
    def __init__(self, mTiles, kTiles, residual: bool = False):
        residual_flag = 1 if residual else 0
        # currently we only support WMMA 16x16x16 FP16
        super().__init__(opcode=opcode.OP_WGMMA_M64N256K16_BF16, args=[mTiles, kTiles, residual_flag])
    
class ROPE_INTERLEAVE_512(ComputeInstruction):
    def __init__(self):
        super().__init__(opcode=opcode.OP_ROPE_INTERLEAVE_512, args=[])

class ATTENTION_M64N64K16_F16_F32_64_64_hdim(ComputeInstruction):
    HEAD_DIM = 128
    def __init__(self, num_kv_block: int, last_kv_active_token_len: int, need_norm: bool = True, need_rope: bool = True):
        # pack need_norm and need_rope into a uint16 arg
        need_flag = (need_norm << 0) | (need_rope << 1)
        super().__init__(opcode=opcode.OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim, args=[num_kv_block, last_kv_active_token_len, need_flag])

class SILU_MUL_SHARED_BF16_K_4096_INTER(ComputeInstruction):
    def __init__(self, num_token):
        super().__init__(opcode=opcode.OP_SILU_MUL_SHARED_BF16_K_4096_INTER, args=[num_token])
class SILU_MUL_SHARED_BF16_K_64_SW128(ComputeInstruction):
    def __init__(self, num_token):
        super().__init__(opcode=opcode.OP_SILU_MUL_SHARED_BF16_K_64_SW128, args=[num_token])

class RMS_NORM_F16_K_4096(ComputeInstruction):
    def __init__(self, num_token: int, epsilon: float):
        # pack epsilon as int16
        epsilon_int = torch.tensor(epsilon, dtype=torch.bfloat16) \
                        .view(torch.uint16) \
                        .item()
        super().__init__(opcode=opcode.OP_RMS_NORM_F16_K_4096, args=[num_token, epsilon_int])
class RMS_NORM_F16_K_4096_SMEM(ComputeInstruction):
    def __init__(self, num_token: int, epsilon: float):
        # pack epsilon as int16
        epsilon_int = torch.tensor(epsilon, dtype=torch.bfloat16) \
                        .view(torch.uint16) \
                        .item()
        super().__init__(opcode=opcode.OP_RMS_NORM_F16_K_4096_SMEM, args=[num_token, epsilon_int])
class RMS_NORM_F16_K_128_SMEM(ComputeInstruction):
    def __init__(self, num_token: int, epsilon: float):
        # pack epsilon as int16
        epsilon_int = torch.tensor(epsilon, dtype=torch.bfloat16) \
                        .view(torch.uint16) \
                        .item()
        super().__init__(opcode=opcode.OP_RMS_NORM_F16_K_128_SMEM, args=[num_token, epsilon_int])
class ARGMAX_PARTIAL_bf16_1152_50688_132(ComputeInstruction):
    CHUNK_SIZE = 1152
    I_STRIDE = 50688
    SMS = 132
    def __init__(self, num_active_token: int):
        super().__init__(opcode=opcode.OP_ARGMAX_PARTIAL_bf16_1152_50688_132, args=[num_active_token])

class ARGMAX_REDUCE_bf16_1152_132(ComputeInstruction):
    CHUNK_SIZE = 1152
    SMS = 132
    def __init__(self, num_active_token: int):
        super().__init__(opcode=opcode.OP_ARGMAX_REDUCE_bf16_1152_132, args=[num_active_token])

class ARGMAX_PARTIAL_bf16_1024_65536_128(ComputeInstruction):
    CHUNK_SIZE = 1024
    I_STRIDE = 65536
    SMS = 128
    def __init__(self, num_active_token: int):
        super().__init__(opcode=opcode.OP_ARGMAX_PARTIAL_bf16_1024_65536_128, args=[num_active_token])
class ARGMAX_REDUCE_bf16_1024_128(ComputeInstruction):
    CHUNK_SIZE = 1024
    SMS = 128
    def __init__(self, num_active_token: int):
        super().__init__(opcode=opcode.OP_ARGMAX_REDUCE_bf16_1024_128, args=[num_active_token])

class Dummy(ComputeInstruction):
    def __init__(self, iters : int):
        super().__init__(opcode=opcode.OP_DUMMY, args=[iters])
class Copy(ComputeInstruction):
    def __init__(self, iters : int, size: int):
        assert size % 4 == 0, "Copy size must be multiple of 4 bytes (size of uint32)"
        super().__init__(opcode=opcode.OP_COPY, args=[iters, size // 4])
class LoopC(ComputeInstruction):
    def __init__(self, count: int, pc: int):
        super().__init__(opcode=opcode.OP_LOOPC, args=[count, pc])

    @classmethod
    def toNext(cls, ptrs, count):
        def smfunc(sm_id: int):
            pc = ptrs[sm_id]
            return cls(count, pc)
        return smfunc

class MemoryInstruction(Instruction):
    def __init__(self, opcode : int, num_slots: int, arg: int, size: int,
            cords : list[int] = [], address: int = None):
        self.opcode = opcode
        self.num_slots = num_slots
        self.arg = arg
        self.size = size
        # pad cord to 4 elements
        self.set_cords(cords)
        self.annotation = dict()
        if address is not None:
            addr_bytes = address.to_bytes(8, byteorder='little')
            for i in range(4):
                self.cords[i] = int.from_bytes(addr_bytes[i*2:i*2+2], byteorder='little')

    # modification functions. those function will MODIFY the instance
    def set_cords(self, cords : list[int]):
        assert len(cords) <=4, "Maximum 4 cords are supported"
        self.cords = cords + [0] * (4 - len(cords))
        for i in range(4):
            assert self.cords[i] >= 0 and self.cords[i] < 2**16-1, "cord values must be a uint16"
    def delta(self, delta):
        cords = []
        if isinstance(delta, int):
            addr = cords2addr(self.cords)
            self.cords = addr2cords(addr + delta)
        elif isinstance(delta, list):
            cords = delta
            assert len(cords) <=4, "Maximum 4 cords are supported"
            cords = cords + [0] * (4 - len(cords))

            for i in range(4):
                self.cords[i] = self.cords[i] + cords[i]
        else:
            raise ValueError("delta must be int or list[int]")

        return self
    # linked and group currently use the same flag slot
    def group(self, enable=True):
        if enable:
            self.opcode = self.opcode | 4
        return self
    def jump(self):
        self.opcode = self.opcode | 8
        return self
    def bar(self, bar_id: int | None = None):
        if bar_id is not None:
            self.opcode = self.opcode | 16
            self.num_slots = self.num_slots | (bar_id << 6)
        return self
    # this mainly used to control if RawAddr needs ld bar
    def writeback(self):
        self.opcode = self.opcode | 2
        return self
    def port(self, port_id: int):
        if port_id == 0:
            pass
        elif port_id == 1:
            self.opcode = self.opcode | 32
        else:
            raise ValueError("Only port 0 and 1 are supported")
        return self

    # copy function that will not modify the instance
    def copy(self):
        other = MemoryInstruction(
            opcode=self.opcode,
            num_slots=self.num_slots,
            arg=self.arg,
            size=self.size,
            cords=self.cords.copy()
        )
        other.annotation = self.annotation.copy()
        return other

    # serialization function
    def tensor(self, tensor: torch.Tensor | None = None) -> torch.Tensor:
        if tensor is None:
            tensor = torch.empty((8,), dtype=torch.uint16)
        else:
            tensor = tensor.view(torch.uint16)
            assert tensor.numel() == 8

        tensor[0] = self.opcode
        tensor[1] = self.size
        tensor[2] = self.num_slots
        tensor[3] = self.arg
        for i in range(4):
            tensor[4 + i] = self.cords[i]
        return tensor.view(torch.uint8)
    def __repr__(self):
        flags = []
        opcode = self.opcode
        num_slots = self.num_slots
        if opcode & 8:
            flags.append("JUMP")
            opcode = opcode & (~8)
        if opcode & 4:
            flags.append("GROUP")
            opcode = opcode & (~4)
        if opcode & 16:
            bar_id = num_slots >> 6
            num_slots = num_slots & 0x3F
            flags.append(f"BAR[{bar_id}]")
            opcode = opcode & (~16)
        if opcode & 2:
            flags.append("WB")
            opcode = opcode & (~2)
        if opcode & 32:
            flags.append("PORT1")
            opcode = opcode & (~32)
        
        return f"MemoryInstruction(opcode={dedcode_opcode(opcode)}, num_slots={num_slots}, arg={self.arg}, size={self.size}, cords={self.cords}, flags={flags}, anno={self.annotation})"

class TerminateM(MemoryInstruction):
    def __init__(self):
        super().__init__(opcode=opcode.OP_TERMINATE, num_slots=0, arg=0, size=0, address=0)
class LoopM(MemoryInstruction):
    """
    This is a combond operation, will take care of loop registers and information to 
    be updated alone the loop.
    current information include:
    - cords[1]:   accumulator registers (gpr[1]) to be cleared at the beginning of each loop iteration
    - cords[2:3]: resource group shift after each loop iteration
    """
    def __init__(self, count: int, pc: int, reg=0, bar_shift: int = 0, tma_shift: int = 0, resource_group = None):
        if resource_group is not None:
            tma_shift, bar_shift = resource_group.get_shift()

        assert reg >= 0 and reg <32, "reg must be in [0,31]"
        assert tma_shift < 2 ** 16, "tma_shift must be less than 65536"
        assert bar_shift < 2 ** 10, "bar_shift must be less than 1024"
        bar_shift_mask = bar_shift << 6
        super().__init__(opcode=opcode.OP_LOOP, num_slots=reg, arg=0, size=count,
                         cords=[pc, 0, bar_shift_mask, tma_shift])

    # TODO(zhiyuang): support smid region
    @classmethod
    def toNext(cls, ptrs, count: int, **kwargs):
        def smfunc(sm_id: int):
            pc = ptrs[sm_id]
            return cls(count, pc, **kwargs)
        return smfunc

class RepeatM(MemoryInstruction):
    def __init__(self, count: int, reg: int = 0, reg_end = None, delta_addr: int | None = None, delta_cords = []):
        if reg_end is None:
            reg_end = reg + 1
        assert reg >= 0 and reg <32, "reg must be in [0,31]"
        assert reg_end >= 0 and reg_end <=32, "reg_end must be in [0,32]"
        assert reg_end > reg, "reg_end must be greater than reg"
        super().__init__(opcode=opcode.OP_REPEAT, num_slots=(reg_end << 8) | reg, arg=0, size=count, address=delta_addr, cords=delta_cords)

    @classmethod
    def onSync(cls, bar_inst_offset: int, bar_id: int | None, count: int, *steps, asyncPort: bool = True):
        if bar_id is None:
            return cls.on(count, *steps)

        port = 1 if asyncPort else 0

        insts = []
        for i, (inst, _) in enumerate(steps):
            if i == bar_inst_offset:
                inst.port(port)
            new_inst = inst.copy()
            if i == bar_inst_offset:
                new_inst.bar(bar_id)
            insts.append(new_inst)
        insts += cls.on1(count, *steps)
        return insts

    @classmethod
    def on1(cls, count: int, *steps):
        assert count > 0, "count must be greater than 0 to use on1"
        # transform the steps
        new_steps = []
        for inst, delta in steps:
            new_steps.append((inst.delta(delta), delta))
        return cls.on(count - 1, *new_steps)

    @classmethod
    def on(cls, count : int, *steps):
        """
        count: number of repeats, from 0
        """
        insts = []
        if len(steps) == 0:
            return insts
        if count == 0:
            return []

        # TODO(zhiyuang): move this optimization to "compiler" side
        regcords = []
        for i, (inst, delta) in enumerate(steps):
            # c = count if i == len(steps) - 1 else 0
            cords = []
            if isinstance(delta, list):
                cords = delta
            elif isinstance(delta, int):
                cords = addr2cords(delta)
            else:
                raise ValueError("delta must be int or list[int]")

            if len(regcords) > 0 and regcords[-1][-1] == cords:
                # extend previous repeat
                regcords[-1][1] = i+1
            else:
                regcords.append([i, i+1, cords])

        # TODO(zhiyuang): move this to "compiler" optimization side?
        # optimzie for no repeats
        if count > 1:
            for reg_start, reg_end, delta_cords in regcords:
                insts += [cls(0, reg=reg_start, reg_end=reg_end, delta_cords=delta_cords)]
            insts[-1].size = count
            
        for inst, _ in steps:
            insts.append(inst)

        if count > 1:
            # enables last jump
            insts[-1].jump()
        return insts
class RawAddress(MemoryInstruction):
    def __init__(self, tensor : torch.Tensor, slot_id : int):
        assert tensor.device.type == 'cuda'
        address = tensor.data_ptr() # assume we know what we are doing

        assert slot_id >= config.num_slots and slot_id < config.num_slots + config.num_special_slots, f"slot_id must be in the range of special slots [{config.num_slots}, {config.num_slots + config.num_special_slots - 1}]"
        super().__init__(opcode=opcode.OP_ALLOC_WB_RAW_ADDRESS, num_slots=slot_id, arg=slot_id, size=0, address=address)
class IssueBarrier(MemoryInstruction):
    def __init__(self, bar: int):
        super().__init__(opcode=opcode.OP_ISSUE_BARRIER, num_slots=0, arg=0, size=0, address=0)
        self.bar(bar)
class CC0(MemoryInstruction):
    def __init__(self, tokens: torch.Tensor, idx: int):
        addr = get_tensor_address(tokens[idx])
        super().__init__(opcode=opcode.OP_CC0, num_slots=0, arg=0, size=0, address=addr)

class RegStore(MemoryInstruction):
    def __init__(self, reg_id: int, shape: torch.Tensor = None, size = None):
        if size is None:
            assert shape is not None, "Either shape or size must be provided for RegStore"
            size = shape.numel() * shape.element_size()
        assert size is not None, "Size must be provided for RegStore"

        numSlots = bytes2slots(size)
        super().__init__(opcode=opcode.OP_ALLOC_WB_REG_STORE, num_slots=numSlots, arg=0, size=reg_id, address=0)

        # TODO(zhiyuang): trick the typecheck system
        self.mode = 'reduce'
    def cord(self, *args):
        # this is a local store; no matter how to cord it it will stay the same
        return self
class RegLoad(MemoryInstruction):
    def __init__(self, reg_id: int, slot_id = None):
        if slot_id is None:
            slot_id = reg_id
        assert slot_id < config.num_special_slots, f"slot_id must be less than {config.num_special_slots} for RegLoad"
        numSlots = config.num_slots + slot_id
        super().__init__(opcode=opcode.OP_ALLOC_REG_LOAD, num_slots=numSlots, arg=0, size=reg_id, address=0)
    def cord(self, *args):
        # this is a local store; no matter how to cord it it will stay the same
        return self
class TmaLoad1D(MemoryInstruction):
    def __init__(self, src : torch.Tensor, bytes : int | None = None, numSlots : int | None = None):
        address = get_tensor_address(src)
        if bytes is None:
            bytes = src.numel() * src.element_size()
        if numSlots is None:
            numSlots = bytes2slots(bytes)
        super().__init__(opcode=opcode.OP_ALLOC_TMA_LOAD_1D, num_slots=numSlots, arg=0, size=bytes, address=address)
    def cord(self, addr):
        new_inst = copy.copy(self)
        new_inst.delta(addr)
        return new_inst
class TmaStore1D(MemoryInstruction):
    def __init__(self, dst : torch.Tensor, bytes : int | None = None, numSlots : int | None = None):
        address = get_tensor_address(dst)
        if bytes is None:
            bytes = dst.numel() * dst.element_size()
        if numSlots is None:
            numSlots = bytes2slots(bytes)
        super().__init__(opcode=opcode.OP_ALLOC_WB_TMA_STORE_1D, num_slots=numSlots, arg=0, size=bytes, address=address)
    def cord(self, addr):
        new_inst = copy.copy(self)
        new_inst.delta(addr)
        return new_inst
class TmaTensor(MemoryInstruction):
    def __init__(self, launcher, mat : torch.Tensor):
        super().__init__(opcode=0, num_slots=0, arg=0, size=0, cords=[])
        self.launcher = launcher
        self.mat = mat
        self.cord_func = None
    def _rank2opcode(self, rank : int, action : str) -> int:
        opcode_map = {
            "reduce": {
                2: opcode.OP_ALLOC_WB_TMA_REDUCE_ADD_2D,
                3: opcode.OP_ALLOC_WB_TMA_REDUCE_ADD_3D,
            },
            "load": {
                1: opcode.OP_ALLOC_TMA_LOAD_TENSOR_1D,
                2: opcode.OP_ALLOC_TMA_LOAD_2D,
                3: opcode.OP_ALLOC_TMA_LOAD_3D,
                4: opcode.OP_ALLOC_TMA_LOAD_4D,
                5: opcode.OP_ALLOC_TMA_LOAD_5D_FIX0
            },
            "store": {
                2: opcode.OP_ALLOC_WB_TMA_STORE_2D,
                3: opcode.OP_ALLOC_WB_TMA_STORE_3D,
                4: opcode.OP_ALLOC_WB_TMA_STORE_4D,
                5: opcode.OP_ALLOC_WB_TMA_STORE_5D_FIX0,
            }
        }
        try:
            return opcode_map[action][rank]
        except KeyError:
            raise ValueError(f"Unsupported rank {rank} and action {action}")
        
    # from this function, we use runtime-majorness
    def _build(self, action, tileM, tileN, tma_func, cord_func_builder):
        self.mode = action
        self.size = self.mat.element_size() * tileM * tileN
        self.num_slots = bytes2slots(self.size)
        rank, desc = tma_func(self.mat, tileM, tileN)
        self.rank = rank
        self.opcode = self._rank2opcode(rank, action)
        self.cord_func = cord_func_builder(self.mat, rank)

        mangled_name = f"TmaTensor_{action}_M{tileM}_N{tileN}_{hex(self.mat.data_ptr())}"

        self.desc = desc
        if isinstance(self.launcher, Launcher):
            self.arg = self.launcher.new_tma(desc)
        else:
            raise ValueError("launcher must be ResourceGroup or Launcher")

        return self

    def cord2tma(self, *cords):
        if self.cord_func is None:
            raise ValueError("cord_func is not set, please call wgmma_load/wgmma_store first")
        return self.cord_func(*cords)
    def cord(self, *cords):
        cords = self.cord2tma(*cords)
        inst = copy.copy(self)
        inst.set_cords(cords)
        return inst
    # implemented build functions
    def tensor1d(self, action: str, size : int):
        actions = ["load"]
        assert action in actions, f"action must be one of {actions}, got {action}"
        return self._build(action, size, 1, build_tma_1d, cord_func_tma_1d)
    def wgmma(self, action : str, tileN: int, tileM: int, major: Major):
        actions = ["load", "store", "reduce"]
        assert action in actions, f"action must be one of {actions}, got {action}"
        if (major == Major.K):
            return self._build(action, tileM, tileN, build_tma_wgmma_kmajor, cord_func_2d_kmajor)
        else:
            return self._build(action, tileM, tileN, build_tma_wgmma_mnmajor, cord_func_2d_mnmajor)
    def wgmma_load(self, tileN: int, tileM: int, major: Major):
        return self.wgmma("load", tileN, tileM, major)
    def wgmma_store(self, tileN: int, tileM: int, major: Major):
        return self.wgmma("store", tileN, tileM, major)

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

        # TODO(zhiyuang): include over group by default
        num_bar_repeat = 1 if self.repeat == 1 else self.repeat + 1
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

        self.tmas = []

        self.need_instruction_build = True

        self.num_bars = 0
        self.bar_values = {}
        self._late_barriers_bound = False

        self.bars = torch.zeros(config.max_bars, 4, dtype=torch.uint8, device=self.device)
        self.bars_src = torch.zeros(config.max_bars, 4, dtype=torch.uint8, device=self.device)

        self.resource_groups = {
            'default': ResourceGroup('default')
        }

        runtime.set_smem_size(self.smem_size)

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
        return bar_id
    def set_bar(self, bar_id: int, value: int):
        assert bar_id in self.bar_values, f"bar_id {bar_id} does not exist"
        assert isinstance(value, int), "bar value must be an int"
        self.bar_values[bar_id] = value
    def new_tma(self, desc: torch.Tensor) -> int:
        self.tmas.append(desc)
        return len(self.tmas) - 1

    # instruction management
    def copy_cptrs(self):
        self.build_instructions()
        return self.cptrs.copy()
    def copy_mptrs(self):
        self.build_instructions()
        return self.mptrs.copy()

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

        if len(self.tmas) == 0:
            tma = torch.empty((4, 128), dtype=torch.uint8, device=self.device)
        else:
            tma = torch.stack(self.tmas).to(self.device)
        profile = self.profile.view(torch.uint8).view(self.num_sms * config.num_profile_events, 8)

        runtime.set_cache_policy(self.bars, stream, 1, 2, 0)
        runtime.set_cache_policy(tma, stream, 1, 2, 0)
        for i in range(self.num_sms // 4):
            runtime.set_cache_policy(cinsts[i*4*self.max_insts:(i+1)*4*self.max_insts], stream, 1, 2, 0)
            runtime.set_cache_policy(minsts[i*4*self.max_insts:(i+1)*4*self.max_insts], stream, 1, 2, 0)

        ret = runtime.launch_dae(
            self.num_sms, self.smem_size,
            cinsts, minsts, tma,
            self.bars, profile,
            stream
        )
        assert ret == 0
    
    def bench(self, iterations : int = 100,
                    total_bytes : int | None = None, total_flops : int | None = None):
        duration_ns = torch.zeros(self.num_sms, dtype=torch.uint64)
        execution_time = 0.0
        for i in range(iterations):
            self.launch()

            # fetch profile data
            profile_data = self.profile[:,0:2].cpu().numpy()
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
