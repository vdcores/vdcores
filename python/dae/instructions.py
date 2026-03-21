import copy

import torch

from .instruction_utils import decode_opcode, dedcode_opcode, encode_bfloat16_u16
from .runtime import config, opcode
from .tma_utils import (
    Major,
    addr2cords,
    build_tma_1d,
    build_tma_wgmma_kmajor,
    build_tma_wgmma_mnmajor,
    bytes2slots,
    cord_func_2d_kmajor,
    cord_func_2d_mnmajor,
    cord_func_tma_1d,
    cords2addr,
    get_tensor_address,
)


class Instruction:
    def tensor(self, tensor: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError()


class ComputeInstruction(Instruction):
    def __init__(self, opcode: int, args: list[int]):
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
        for i, arg in enumerate(self.args):
            assert 0 <= arg < 2**16, "args must be uint16"
            tensor[i + 1] = arg
        return tensor.view(torch.uint8)

    def __repr__(self):
        return f"ComputeInstruction(opcode={decode_opcode(self.opcode)}, args={self.args})"


class TerminateC(ComputeInstruction):
    def __init__(self):
        super().__init__(opcode=opcode.OP_TERMINATEC, args=[])


class Gemv_M64N8(ComputeInstruction):
    MNK = (64, 8, 256)
    n_batch = 4

    def __init__(self, kTiles: int, nprefeth=0, residual: bool = False):
        super().__init__(opcode=opcode.OP_GEMV_M64N8, args=[kTiles, nprefeth])


class Gemv_M128N8(ComputeInstruction):
    MNK = (128, 8, 128)
    n_batch = 4

    def __init__(self, kTiles: int, nprefeth=0, residual: bool = False):
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

class Gemv_M64N8_MMA(ComputeInstruction):
    MNK = (64, 8, 256)
    n_batch = 1
    def __init__(self, kTiles: int):
        super().__init__(opcode=opcode.OP_GEMV_M64N8_MMA, args=[kTiles])


class WGMMA_64x256x64_F16(ComputeInstruction):
    MNK = (64, 64, 256)

    def __init__(self, mTiles, kTiles, residual: bool = False):
        residual_flag = 1 if residual else 0
        super().__init__(opcode=opcode.OP_WGMMA_M64N256K16_F16, args=[mTiles, kTiles, residual_flag])


class WGMMA_64x256x64_BF16(ComputeInstruction):
    MNK = (64, 64, 128)

    def __init__(self, mTiles, kTiles, residual: bool = False):
        residual_flag = 1 if residual else 0
        super().__init__(opcode=opcode.OP_WGMMA_M64N256K16_BF16, args=[mTiles, kTiles, residual_flag])


class ROPE_INTERLEAVE_512(ComputeInstruction):
    def __init__(self):
        super().__init__(opcode=opcode.OP_ROPE_INTERLEAVE_512, args=[])


class ATTENTION_M64N64K16_F16_F32_64_64_hdim(ComputeInstruction):
    HEAD_DIM = 128

    def __init__(self, num_kv_block: int, last_kv_active_token_len: int, need_norm: bool = True, need_rope: bool = True):
        need_flag = (need_norm << 0) | (need_rope << 1)
        super().__init__(
            opcode=opcode.OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim,
            args=[num_kv_block, last_kv_active_token_len, need_flag],
        )


class SILU_MUL_SHARED_BF16_K_4096_INTER(ComputeInstruction):
    def __init__(self, num_token):
        super().__init__(opcode=opcode.OP_SILU_MUL_SHARED_BF16_K_4096_INTER, args=[num_token])


class SILU_MUL_SHARED_BF16_K_64_SW128(ComputeInstruction):
    def __init__(self, num_token):
        super().__init__(opcode=opcode.OP_SILU_MUL_SHARED_BF16_K_64_SW128, args=[num_token])


class RMS_NORM_F16_K_4096(ComputeInstruction):
    def __init__(self, num_token: int, epsilon: float):
        super().__init__(opcode=opcode.OP_RMS_NORM_F16_K_4096, args=[num_token, encode_bfloat16_u16(epsilon)])


class RMS_NORM_F16_K_4096_SMEM(ComputeInstruction):
    def __init__(self, num_token: int, epsilon: float):
        super().__init__(opcode=opcode.OP_RMS_NORM_F16_K_4096_SMEM, args=[num_token, encode_bfloat16_u16(epsilon)])


class RMS_NORM_F16_K_128_SMEM(ComputeInstruction):
    def __init__(self, num_token: int, epsilon: float):
        super().__init__(opcode=opcode.OP_RMS_NORM_F16_K_128_SMEM, args=[num_token, encode_bfloat16_u16(epsilon)])


class RMS_NORM_F16_K_2048_SMEM(ComputeInstruction):
    def __init__(self, num_token: int, epsilon: float):
        super().__init__(opcode=opcode.OP_RMS_NORM_F16_K_2048_SMEM, args=[num_token, encode_bfloat16_u16(epsilon)])


def select_attention_decode_instruction(head_dim: int):
    if head_dim == ATTENTION_M64N64K16_F16_F32_64_64_hdim.HEAD_DIM:
        return ATTENTION_M64N64K16_F16_F32_64_64_hdim
    raise NotImplementedError(
        f"Missing attention decode kernel support for head_dim={head_dim}. "
        "Add a dedicated opcode/instruction path before launching this model."
    )


def select_rms_glob_instruction(hidden_size: int):
    if hidden_size == 4096:
        return RMS_NORM_F16_K_4096
    raise NotImplementedError(
        f"Missing global RMS kernel support for hidden_size={hidden_size}. "
        "Add a dedicated opcode/instruction path before launching this model."
    )


def select_rms_smem_instruction(hidden_size: int):
    if hidden_size == 4096:
        return RMS_NORM_F16_K_4096_SMEM
    if hidden_size == 2048:
        return RMS_NORM_F16_K_2048_SMEM
    if hidden_size == 128:
        return RMS_NORM_F16_K_128_SMEM
    raise NotImplementedError(
        f"Missing shared-memory RMS kernel support for hidden_size={hidden_size}. "
        "Add a dedicated opcode/instruction path before launching this model."
    )


def ensure_cc0_supported_hidden_size(hidden_size: int):
    if hidden_size == 4096:
        return
    raise NotImplementedError(
        f"Missing CC0 embedding-stride support for hidden_size={hidden_size}. "
        "Parameterize the memory op before launching this model."
    )


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
    def __init__(self, iters: int):
        super().__init__(opcode=opcode.OP_DUMMY, args=[iters])


class Copy(ComputeInstruction):
    def __init__(self, iters: int, size: int):
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
    def __init__(
        self,
        opcode: int,
        num_slots: int,
        arg: int,
        size: int,
        cords: list[int] = [],
        address: int | None = None,
    ):
        self.opcode = opcode
        self.num_slots = num_slots
        self.arg = arg
        self.size = size
        self.set_cords(cords)
        self.annotation = {}
        if address is not None:
            addr_bytes = address.to_bytes(8, byteorder="little")
            for i in range(4):
                self.cords[i] = int.from_bytes(addr_bytes[i * 2 : i * 2 + 2], byteorder="little")

    def set_cords(self, cords: list[int]):
        assert len(cords) <= 4, "Maximum 4 cords are supported"
        self.cords = cords + [0] * (4 - len(cords))
        for i in range(4):
            assert 0 <= self.cords[i] < 2**16 - 1, "cord values must be a uint16"

    def delta(self, delta):
        if isinstance(delta, int):
            addr = cords2addr(self.cords)
            self.cords = addr2cords(addr + delta)
        elif isinstance(delta, list):
            cords = delta
            assert len(cords) <= 4, "Maximum 4 cords are supported"
            cords = cords + [0] * (4 - len(cords))

            for i in range(4):
                self.cords[i] = self.cords[i] + cords[i]
        else:
            raise ValueError("delta must be int or list[int]")

        return self

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

    def writeback(self):
        self.opcode = self.opcode | 2
        return self

    def port(self, port_id: int):
        if port_id == 0:
            return self
        if port_id == 1:
            self.opcode = self.opcode | 32
            return self
        raise ValueError("Only port 0 and 1 are supported")

    def copy(self):
        other = MemoryInstruction(
            opcode=self.opcode,
            num_slots=self.num_slots,
            arg=self.arg,
            size=self.size,
            cords=self.cords.copy(),
        )
        other.annotation = self.annotation.copy()
        return other

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
        opcode_value = self.opcode
        num_slots = self.num_slots
        if opcode_value & 8:
            flags.append("JUMP")
            opcode_value = opcode_value & (~8)
        if opcode_value & 4:
            flags.append("GROUP")
            opcode_value = opcode_value & (~4)
        if opcode_value & 16:
            bar_id = num_slots >> 6
            num_slots = num_slots & 0x3F
            flags.append(f"BAR[{bar_id}]")
            opcode_value = opcode_value & (~16)
        if opcode_value & 2:
            flags.append("WB")
            opcode_value = opcode_value & (~2)
        if opcode_value & 32:
            flags.append("PORT1")
            opcode_value = opcode_value & (~32)

        return (
            "MemoryInstruction("
            f"opcode={decode_opcode(opcode_value)}, num_slots={num_slots}, "
            f"arg={self.arg}, size={self.size}, cords={self.cords}, "
            f"flags={flags}, anno={self.annotation})"
        )


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

    def __init__(
        self,
        count: int,
        pc: int,
        reg=0,
        bar_shift: int = 0,
        tma_shift: int = 0,
        resource_group=None,
    ):
        if resource_group is not None:
            tma_shift, bar_shift = resource_group.get_shift()

        assert 0 <= reg < 32, "reg must be in [0,31]"
        assert tma_shift < 2**16, "tma_shift must be less than 65536"
        assert bar_shift < 2**10, "bar_shift must be less than 1024"
        bar_shift_mask = bar_shift << 6
        super().__init__(
            opcode=opcode.OP_LOOP,
            num_slots=reg,
            arg=0,
            size=count,
            cords=[pc, 0, bar_shift_mask, tma_shift],
        )

    @classmethod
    def toNext(cls, ptrs, count: int, **kwargs):
        def smfunc(sm_id: int):
            pc = ptrs[sm_id]
            return cls(count, pc, **kwargs)

        return smfunc


class RepeatM(MemoryInstruction):
    def __init__(
        self,
        count: int,
        reg: int = 0,
        reg_end=None,
        delta_addr: int | None = None,
        delta_cords=[],
    ):
        if reg_end is None:
            reg_end = reg + 1
        assert 0 <= reg < 32, "reg must be in [0,31]"
        assert 0 <= reg_end <= 32, "reg_end must be in [0,32]"
        assert reg_end > reg, "reg_end must be greater than reg"
        super().__init__(
            opcode=opcode.OP_REPEAT,
            num_slots=(reg_end << 8) | reg,
            arg=0,
            size=count,
            address=delta_addr,
            cords=delta_cords,
        )

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
        new_steps = []
        for inst, delta in steps:
            new_steps.append((inst.delta(delta), delta))
        return cls.on(count - 1, *new_steps)

    @classmethod
    def on(cls, count: int, *steps):
        insts = []
        if len(steps) == 0:
            return insts
        if count == 0:
            return []

        regcords = []
        for i, (inst, delta) in enumerate(steps):
            if isinstance(delta, list):
                cords = delta
            elif isinstance(delta, int):
                cords = addr2cords(delta)
            else:
                raise ValueError("delta must be int or list[int]")

            if len(regcords) > 0 and regcords[-1][-1] == cords:
                regcords[-1][1] = i + 1
            else:
                regcords.append([i, i + 1, cords])

        if count > 1:
            for reg_start, reg_end, delta_cords in regcords:
                insts += [cls(0, reg=reg_start, reg_end=reg_end, delta_cords=delta_cords)]
            insts[-1].size = count

        for inst, _ in steps:
            insts.append(inst)

        if count > 1:
            insts[-1].jump()
        return insts


class RawAddress(MemoryInstruction):
    def __init__(self, tensor: torch.Tensor, slot_id: int):
        assert tensor.device.type == "cuda"
        address = tensor.data_ptr()

        min_slot = config.num_slots
        max_slot = config.num_slots + config.num_special_slots - 1
        assert min_slot <= slot_id <= max_slot, (
            f"slot_id must be in the range of special slots [{min_slot}, {max_slot}]"
        )
        super().__init__(
            opcode=opcode.OP_ALLOC_WB_RAW_ADDRESS,
            num_slots=slot_id,
            arg=slot_id,
            size=0,
            address=address,
        )


class IssueBarrier(MemoryInstruction):
    def __init__(self, bar: int):
        super().__init__(opcode=opcode.OP_ISSUE_BARRIER, num_slots=0, arg=0, size=0, address=0)
        self.bar(bar)


class CC0(MemoryInstruction):
    def __init__(self, tokens: torch.Tensor, idx: int):
        addr = get_tensor_address(tokens[idx])
        super().__init__(opcode=opcode.OP_CC0, num_slots=0, arg=0, size=0, address=addr)


class RegStore(MemoryInstruction):
    def __init__(self, reg_id: int, shape: torch.Tensor = None, size=None):
        if size is None:
            assert shape is not None, "Either shape or size must be provided for RegStore"
            size = shape.numel() * shape.element_size()
        assert size is not None, "Size must be provided for RegStore"

        num_slots = bytes2slots(size)
        super().__init__(opcode=opcode.OP_ALLOC_WB_REG_STORE, num_slots=num_slots, arg=0, size=reg_id, address=0)

        self.mode = "reduce"

    def cord(self, *args):
        return self


class RegLoad(MemoryInstruction):
    def __init__(self, reg_id: int, slot_id=None):
        if slot_id is None:
            slot_id = reg_id
        assert slot_id < config.num_special_slots, (
            f"slot_id must be less than {config.num_special_slots} for RegLoad"
        )
        num_slots = config.num_slots + slot_id
        super().__init__(opcode=opcode.OP_ALLOC_REG_LOAD, num_slots=num_slots, arg=0, size=reg_id, address=0)

    def cord(self, *args):
        return self


class TmaLoad1D(MemoryInstruction):
    def __init__(self, src: torch.Tensor, bytes: int | None = None, numSlots: int | None = None):
        address = get_tensor_address(src)
        if bytes is None:
            bytes = src.numel() * src.element_size()
        if numSlots is None:
            numSlots = bytes2slots(bytes)
        super().__init__(
            opcode=opcode.OP_ALLOC_TMA_LOAD_1D,
            num_slots=numSlots,
            arg=0,
            size=bytes,
            address=address,
        )

    def cord(self, addr):
        new_inst = copy.copy(self)
        new_inst.delta(addr)
        return new_inst


class TmaStore1D(MemoryInstruction):
    def __init__(self, dst: torch.Tensor, bytes: int | None = None, numSlots: int | None = None):
        address = get_tensor_address(dst)
        if bytes is None:
            bytes = dst.numel() * dst.element_size()
        if numSlots is None:
            numSlots = bytes2slots(bytes)
        super().__init__(
            opcode=opcode.OP_ALLOC_WB_TMA_STORE_1D,
            num_slots=numSlots,
            arg=0,
            size=bytes,
            address=address,
        )

    def cord(self, addr):
        new_inst = copy.copy(self)
        new_inst.delta(addr)
        return new_inst


class TmaTensor(MemoryInstruction):
    def __init__(self, launcher, mat: torch.Tensor):
        super().__init__(opcode=0, num_slots=0, arg=0, size=0, cords=[])
        self.launcher = launcher
        self.mat = mat
        self.cord_func = None

    def _rank2opcode(self, rank: int, action: str) -> int:
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
                5: opcode.OP_ALLOC_TMA_LOAD_5D_FIX0,
            },
            "store": {
                2: opcode.OP_ALLOC_WB_TMA_STORE_2D,
                3: opcode.OP_ALLOC_WB_TMA_STORE_3D,
                4: opcode.OP_ALLOC_WB_TMA_STORE_4D,
                5: opcode.OP_ALLOC_WB_TMA_STORE_5D_FIX0,
            },
        }
        try:
            return opcode_map[action][rank]
        except KeyError as exc:
            raise ValueError(f"Unsupported rank {rank} and action {action}") from exc

    def _build(self, action, tileM, tileN, tma_func, cord_func_builder):
        self.mode = action
        self.size = self.mat.element_size() * tileM * tileN
        self.num_slots = bytes2slots(self.size)
        rank, desc = tma_func(self.mat, tileM, tileN)
        self.rank = rank
        self.opcode = self._rank2opcode(rank, action)
        self.cord_func = cord_func_builder(self.mat, rank)
        self.desc = desc

        if not hasattr(self.launcher, "new_tma"):
            raise ValueError("launcher must expose new_tma()")
        self.arg = self.launcher.new_tma(desc)

        return self

    def cord2tma(self, *cords):
        if self.cord_func is None:
            raise ValueError("cord_func is not set, please call wgmma_load/wgmma_store first")
        return self.cord_func(*cords)

    def cord(self, *cords):
        inst = copy.copy(self)
        inst.set_cords(self.cord2tma(*cords))
        return inst

    def tensor1d(self, action: str, size: int):
        actions = ["load"]
        assert action in actions, f"action must be one of {actions}, got {action}"
        return self._build(action, size, 1, build_tma_1d, cord_func_tma_1d)

    def wgmma(self, action: str, tileN: int, tileM: int, major: Major):
        actions = ["load", "store", "reduce"]
        assert action in actions, f"action must be one of {actions}, got {action}"
        if major == Major.K:
            return self._build(action, tileM, tileN, build_tma_wgmma_kmajor, cord_func_2d_kmajor)
        return self._build(action, tileM, tileN, build_tma_wgmma_mnmajor, cord_func_2d_mnmajor)

    def wgmma_load(self, tileN: int, tileM: int, major: Major):
        return self.wgmma("load", tileN, tileM, major)

    def wgmma_store(self, tileN: int, tileM: int, major: Major):
        return self.wgmma("store", tileN, tileM, major)


__all__ = [
    "decode_opcode",
    "dedcode_opcode",
    "Instruction",
    "ComputeInstruction",
    "TerminateC",
    "Gemv_M64N8",
    "Gemv_M128N8",
    "Gemv_M64N8_ROPE_128",
    "Gemv_M192N16",
    "Gemv_M64N8_MMA",
    "WGMMA_64x256x64_F16",
    "WGMMA_64x256x64_BF16",
    "ROPE_INTERLEAVE_512",
    "ATTENTION_M64N64K16_F16_F32_64_64_hdim",
    "SILU_MUL_SHARED_BF16_K_4096_INTER",
    "SILU_MUL_SHARED_BF16_K_64_SW128",
    "RMS_NORM_F16_K_4096",
    "RMS_NORM_F16_K_4096_SMEM",
    "RMS_NORM_F16_K_128_SMEM",
    "RMS_NORM_F16_K_2048_SMEM",
    "select_attention_decode_instruction",
    "select_rms_glob_instruction",
    "select_rms_smem_instruction",
    "ensure_cc0_supported_hidden_size",
    "ARGMAX_PARTIAL_bf16_1152_50688_132",
    "ARGMAX_REDUCE_bf16_1152_132",
    "ARGMAX_PARTIAL_bf16_1024_65536_128",
    "ARGMAX_REDUCE_bf16_1024_128",
    "Dummy",
    "Copy",
    "LoopC",
    "MemoryInstruction",
    "TerminateM",
    "LoopM",
    "RepeatM",
    "RawAddress",
    "IssueBarrier",
    "CC0",
    "RegStore",
    "RegLoad",
    "TmaLoad1D",
    "TmaStore1D",
    "TmaTensor",
]
