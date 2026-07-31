import copy

import torch

from .instruction_utils import (
    compute_operator_name as format_compute_operator_name,
    decode_opcode,
    dedcode_opcode,
    encode_bfloat16_u16,
    encode_compute_instruction_tensor,
    normalize_compute_opcode_reference,
    resolve_compute_opcode_value,
)
from .op_families import ComputeOpFamilyRef, family_ref
from .runtime import comm_opcode, config, opcode, pool_opcode
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
    def __init__(self, opcode: int | str | ComputeOpFamilyRef, args: list[int]):
        self.opcode, self.op_family_name = normalize_compute_opcode_reference(opcode)
        self.args = args

    def opcode_value(self) -> int:
        return resolve_compute_opcode_value(self.opcode, self.op_family_name)

    def compute_operator_name(self) -> str:
        return format_compute_operator_name(self.opcode, self.op_family_name)

    def tensor(self, tensor: torch.Tensor | None = None) -> torch.Tensor:
        return encode_compute_instruction_tensor(self.opcode, self.op_family_name, self.args, tensor)

    def __repr__(self):
        return f"ComputeInstruction(opcode={self.compute_operator_name()}, args={self.args})"


class TerminateC(ComputeInstruction):
    def __init__(self):
        super().__init__(opcode=opcode.OP_TERMINATEC, args=[])


class Gemv_M64N8(ComputeInstruction):
    MNK = (64, 8, 256)
    n_batch = 4

    def __init__(self, kTiles: int, nprefeth=0, residual: bool = False):
        super().__init__(opcode=family_ref("GEMV_WGMMA", M=64, N=8, K=256, BLOAD=4, RESIDUAL=residual), args=[kTiles, nprefeth])

class Gemv_M64N8K64(ComputeInstruction):
    MNK = (64, 8, 64)
    n_batch = 1

    def __init__(self, kTiles: int, nprefeth=0, residual: bool = False):
        super().__init__(opcode=family_ref("GEMV_WGMMA", M=64, N=8, K=64, BLOAD=1, RESIDUAL=residual), args=[kTiles, nprefeth])

class Gemv_M64N8K128(ComputeInstruction):
    MNK = (64, 8, 128)
    n_batch = 1

    def __init__(self, kTiles: int, nprefeth=0, residual: bool = False):
        super().__init__(opcode=family_ref("GEMV_WGMMA", M=64, N=8, K=128, BLOAD=1, RESIDUAL=residual), args=[kTiles, nprefeth])

class Gemv_M64N8B2(ComputeInstruction):
    MNK = (64, 8, 256)
    n_batch = 2

    def __init__(self, kTiles: int, nprefeth=0, residual: bool = False):
        super().__init__(opcode=family_ref("GEMV_WGMMA", M=64, N=8, K=256, BLOAD=2, RESIDUAL=residual), args=[kTiles, nprefeth])

class Gemv_M128N8(ComputeInstruction):
    MNK = (128, 8, 128)
    n_batch = 4

    def __init__(self, kTiles: int, nprefeth=0, residual: bool = False):
        super().__init__(opcode=family_ref("GEMV_WGMMA", M=128, N=8, K=128, BLOAD=4, RESIDUAL=residual), args=[kTiles, nprefeth])

class Gemm_M64N64(ComputeInstruction):
    MNK = (64, 64, 128)
    n_batch = 1

    def __init__(self, kTiles: int, residual: bool = False):
        super().__init__(opcode=opcode.OP_GEMM_M64N64, args=[kTiles])

class Gemm_M64N64K64(ComputeInstruction):
    MNK = (64, 64, 64)
    n_batch = 1

    def __init__(self, kTiles: int, residual: bool = False):
        super().__init__(opcode=opcode.OP_GEMM_M64N64K64, args=[kTiles])


class Gemm_M64N128K64(ComputeInstruction):
    MNK = (64, 128, 64)
    n_batch = 1

    def __init__(self, kTiles: int, residual: bool = False):
        super().__init__(opcode=opcode.OP_GEMM_M64N128K64, args=[kTiles])


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
        super().__init__(opcode=family_ref("GEMV_MMA", M=64, N=8, K=256), args=[kTiles])


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


ATTENTION_DYNAMIC_LAST_KV_LEN_FLAG = 0x4
ATTENTION_DYNAMIC_NUM_KV_BLOCKS_FLAG = 0x8
ATTENTION_BLOCK_COUNTER_SHIFT = 4
ATTENTION_BLOCK_COUNTER_MASK = 0xF
ATTENTION_COUNTER_SHIFT = 8


def _encode_attention_runtime_flags(
    need_norm: bool,
    need_rope: bool,
    seq_len_counter_reg: int | None = None,
    num_kv_block_counter_reg: int | None = None,
) -> int:
    flags = 0
    if need_norm:
        flags |= 1
    if need_rope:
        flags |= 2
    if seq_len_counter_reg is not None:
        assert 0 <= seq_len_counter_reg < 256, "seq_len_counter_reg must fit in 8 bits"
        flags |= ATTENTION_DYNAMIC_LAST_KV_LEN_FLAG
        flags |= seq_len_counter_reg << ATTENTION_COUNTER_SHIFT
    if num_kv_block_counter_reg is not None:
        assert 0 <= num_kv_block_counter_reg <= ATTENTION_BLOCK_COUNTER_MASK, "num_kv_block_counter_reg must fit in 4 bits"
        flags |= ATTENTION_DYNAMIC_NUM_KV_BLOCKS_FLAG
        flags |= num_kv_block_counter_reg << ATTENTION_BLOCK_COUNTER_SHIFT
    return flags

def _encode_attention_qkv_workload_flag(num_active_q: int, last_kv_active_token_len: int) -> int:
    return num_active_q | (last_kv_active_token_len << 8)

class ATTENTION_M64N64K16_F16_F32_64_64_hdim(ComputeInstruction):
    HEAD_DIM = 128

    def __init__(self, num_kv_block: int, num_active_q: int, last_kv_active_token_len: int, need_norm: bool = True, need_rope: bool = True, seq_len_counter_reg: int | None = None, num_kv_block_counter_reg: int | None = None):
        super().__init__(
            opcode=opcode.OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim,
            args=[
                num_kv_block, 
                _encode_attention_qkv_workload_flag(num_active_q, last_kv_active_token_len), 
                _encode_attention_runtime_flags(need_norm, need_rope, seq_len_counter_reg, num_kv_block_counter_reg)
            ],
        )


class ATTENTION_M64N64K16_F16_F32_64_64_hdim_MMA(ComputeInstruction):
    HEAD_DIM = 128

    def __init__(self, num_kv_block: int, num_active_q: int, last_kv_active_token_len: int, need_norm: bool = True, need_rope: bool = True, seq_len_counter_reg: int | None = None, num_kv_block_counter_reg: int | None = None):
        super().__init__(
            opcode=opcode.OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim_MMA,
            args=[
                num_kv_block,
                _encode_attention_qkv_workload_flag(num_active_q, last_kv_active_token_len),
                _encode_attention_runtime_flags(need_norm, need_rope, seq_len_counter_reg, num_kv_block_counter_reg),
            ],
        )


class ATTENTION_M64N64K16_F16_F32_64_64_hdim64(ComputeInstruction):
    HEAD_DIM = 64

    def __init__(self, num_kv_block: int, num_active_q: int, last_kv_active_token_len: int, need_norm: bool = True, need_rope: bool = True, seq_len_counter_reg: int | None = None, num_kv_block_counter_reg: int | None = None):
        super().__init__(
            opcode=opcode.OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim64,
            args=[
                num_kv_block, 
                _encode_attention_qkv_workload_flag(num_active_q, last_kv_active_token_len), 
                _encode_attention_runtime_flags(need_norm, need_rope, seq_len_counter_reg, num_kv_block_counter_reg)
            ],
        )


class ATTENTION_M64N64K16_F16_F32_64_64_hdim64_MMA(ComputeInstruction):
    HEAD_DIM = 64

    def __init__(self, num_kv_block: int, num_active_q: int, last_kv_active_token_len: int, need_norm: bool = True, need_rope: bool = True, seq_len_counter_reg: int | None = None, num_kv_block_counter_reg: int | None = None):
        super().__init__(
            opcode=opcode.OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim64_MMA,
            args=[
                num_kv_block,
                _encode_attention_qkv_workload_flag(num_active_q, last_kv_active_token_len),
                _encode_attention_runtime_flags(need_norm, need_rope, seq_len_counter_reg, num_kv_block_counter_reg),
            ],
        )


class ATTENTION_M64N64K16_F16_F32_64_64_hdim_split(ComputeInstruction):
    HEAD_DIM = 128
    def __init__(self, num_kv_block: int, split_idx: int, num_active_q: int, last_kv_active_token_len: int, kv_start_idx: int, need_norm: bool = True, need_rope: bool = True):
        assert split_idx < 16, "split_idx must be less than 16 to fit in the instruction encoding"
        # pack need_norm and need_rope into a uint16 arg
        arg0 = num_kv_block | (split_idx << 12)
        arg1 = num_active_q | (last_kv_active_token_len << 8)
        arg2 = kv_start_idx # make this 16bit to support long seq
        super().__init__(
            opcode=opcode.OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim_split, 
            args=[arg0, arg1, arg2]
        )


class ATTENTION_M64N64K16_F16_F32_64_64_hdim_split_MMA(ComputeInstruction):
    HEAD_DIM = 128
    def __init__(self, num_kv_block: int, split_idx: int, num_active_q: int, last_kv_active_token_len: int, kv_start_idx: int, need_norm: bool = True, need_rope: bool = True):
        assert split_idx < 16, "split_idx must be less than 16 to fit in the instruction encoding"
        arg0 = num_kv_block | (split_idx << 12)
        arg1 = num_active_q | (last_kv_active_token_len << 8)
        arg2 = kv_start_idx
        super().__init__(
            opcode=opcode.OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim_split_MMA,
            args=[arg0, arg1, arg2]
        )

class ATTN_SPLIT_POST_REDUCE(ComputeInstruction):
    HEAD_DIM = 128
    Q_TILE = 4
    def __init__(self, num_split: int):
        super().__init__(opcode=opcode.OP_ATTN_SPLIT_POST_REDUCE, args=[num_split])

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


class RMS_NORM_F16_K_5120_SMEM(ComputeInstruction):
    def __init__(self, num_token: int, epsilon: float):
        super().__init__(opcode=opcode.OP_RMS_NORM_F16_K_5120_SMEM, args=[num_token, encode_bfloat16_u16(epsilon)])


def select_attention_decode_instruction(head_dim: int):
    if head_dim == ATTENTION_M64N64K16_F16_F32_64_64_hdim.HEAD_DIM:
        return ATTENTION_M64N64K16_F16_F32_64_64_hdim
    if head_dim == ATTENTION_M64N64K16_F16_F32_64_64_hdim64.HEAD_DIM:
        return ATTENTION_M64N64K16_F16_F32_64_64_hdim64_MMA
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
    if hidden_size == 5120:
        return RMS_NORM_F16_K_5120_SMEM
    if hidden_size == 128:
        return RMS_NORM_F16_K_128_SMEM
    raise NotImplementedError(f"Missing shared-memory RMS kernel support for hidden_size={hidden_size}. Add a dedicated opcode/instruction path before launching this model.")


def ensure_cc0_supported_hidden_size(hidden_size: int):
    row_bytes = hidden_size * 2
    if row_bytes > 0:
        return
    raise NotImplementedError(f"Missing CC0 embedding-stride support for hidden_size={hidden_size}. Parameterize the memory op before launching this model.")


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
    def __init__(self, count: int, pc: int, reg: int = 0):
        assert 0 <= reg < 2**16, "reg must fit in uint16"
        super().__init__(opcode=opcode.OP_LOOPC, args=[count, pc, reg])

    @classmethod
    def toNext(cls, ptrs, count, reg: int = 0):
        def smfunc(sm_id: int):
            pc = ptrs[sm_id]
            return cls(count, pc, reg=reg)

        return smfunc


class CommunicationInstruction(Instruction):
    """A 16-byte instruction consumed only by the VDCores communication warp."""

    requires_communication_core = True
    requires_signal_array = False

    def __init__(
        self,
        opcode: int,
        *,
        size: int = 0,
        arg0: int = 0,
        arg1: int = 0,
        address: int = 0,
    ):
        for name, value in (("opcode", opcode), ("size", size), ("arg0", arg0), ("arg1", arg1)):
            if not 0 <= int(value) < 2**16:
                raise ValueError(f"{name} must fit in uint16")
        if not 0 <= int(address) < 2**64:
            raise ValueError("address must fit in uint64")
        self.opcode = int(opcode)
        self.size = int(size)
        self.arg0 = int(arg0)
        self.arg1 = int(arg1)
        self.address = int(address)

    def tensor(self, tensor: torch.Tensor | None = None) -> torch.Tensor:
        if tensor is None:
            tensor = torch.empty((8,), dtype=torch.uint16)
        else:
            tensor = tensor.view(torch.uint16)
            assert tensor.numel() == 8
        tensor[0] = self.opcode
        tensor[1] = self.size
        tensor[2] = self.arg0
        tensor[3] = self.arg1
        address_words = addr2cords(self.address)
        for index in range(4):
            tensor[4 + index] = address_words[index]
        return tensor.view(torch.uint8)

    def __repr__(self):
        return (
            "CommunicationInstruction("
            f"opcode={self.opcode}, size={self.size}, arg0={self.arg0}, "
            f"arg1={self.arg1}, address=0x{self.address:x})"
        )


class PoolInstruction(Instruction):
    """A 16-byte instruction consumed by a compiled multi-warp pool core.

    Slot zero selects the precompiled VDCores assembly. The remaining
    instructions form that assembly's immutable operation queue.
    """

    requires_pool_core = True
    requires_signal_array = True
    # A bare PoolInstruction remains an assembly header. Operation subclasses
    # opt out explicitly so existing custom pool executors keep their API.
    selects_pool_execute_warp = True

    def __init__(
        self,
        opcode: int,
        *,
        size: int = 0,
        arg0: int = 0,
        arg1: int = 0,
        address: int = 0,
    ):
        for name, value in (
            ("opcode", opcode),
            ("size", size),
            ("arg0", arg0),
            ("arg1", arg1),
        ):
            if not 0 <= int(value) < 2**16:
                raise ValueError(f"{name} must fit in uint16")
        if not 0 <= int(address) < 2**64:
            raise ValueError("address must fit in uint64")
        self.opcode = int(opcode)
        self.size = int(size)
        self.arg0 = int(arg0)
        self.arg1 = int(arg1)
        self.address = int(address)

    def tensor(self, tensor: torch.Tensor | None = None) -> torch.Tensor:
        if tensor is None:
            tensor = torch.empty((8,), dtype=torch.uint16)
        else:
            tensor = tensor.view(torch.uint16)
            assert tensor.numel() == 8
        tensor[0] = self.opcode
        tensor[1] = self.size
        tensor[2] = self.arg0
        tensor[3] = self.arg1
        address_words = addr2cords(self.address)
        for index in range(4):
            tensor[4 + index] = address_words[index]
        return tensor.view(torch.uint8)

    def __repr__(self):
        return (
            "PoolInstruction("
            f"opcode={self.opcode}, size={self.size}, arg0={self.arg0}, "
            f"arg1={self.arg1}, address=0x{self.address:x})"
        )


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
            assert 0 <= self.cords[i] < 2**16, "cord values must be a uint16"

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


class CounterOffsetMemoryInstruction:
    def __init__(self, counter_reg: int, inst: MemoryInstruction, delta):
        offsets = [(counter_reg, delta)]
        if isinstance(inst, CounterOffsetMemoryInstruction):
            self.inst = inst.inst
            self.offsets = inst.offsets + offsets
        else:
            self.inst = inst
            self.offsets = offsets

    def expand_instructions(self):
        return RepeatM.offsetByCounters(self.offsets, self.inst)

    def bar(self, *args, **kwargs):
        self.inst.bar(*args, **kwargs)
        return self

    def group(self, *args, **kwargs):
        self.inst.group(*args, **kwargs)
        return self

    def jump(self, *args, **kwargs):
        self.inst.jump(*args, **kwargs)
        return self

    def port(self, *args, **kwargs):
        self.inst.port(*args, **kwargs)
        return self

    def writeback(self, *args, **kwargs):
        self.inst.writeback(*args, **kwargs)
        return self

    def copy(self):
        new_inst = CounterOffsetMemoryInstruction(self.offsets[0][0], self.inst.copy(), self.offsets[0][1])
        new_inst.offsets = self.offsets.copy()
        return new_inst

    def __getattr__(self, name):
        return getattr(self.inst, name)

    def __repr__(self):
        return f"CounterOffsetMemoryInstruction(offsets={self.offsets}, inst={self.inst!r})"


class RepeatM(MemoryInstruction):
    COUNTER_MODE_FLAG = 0x8000
    COUNT_COUNTER_MODE_FLAG = 0x4000
    ACCUMULATE_MODE_FLAG = 0x2000
    COUNTER_REG_MASK = 0x00FF

    def __init__(
        self,
        count: int,
        reg: int = 0,
        reg_end=None,
        delta_addr: int | None = None,
        delta_cords=[],
        counter_reg: int | None = None,
        count_counter_reg: int | None = None,
        accumulate: bool = False,
    ):
        if reg_end is None:
            reg_end = reg + 1
        assert 0 <= reg < 32, "reg must be in [0,31]"
        assert 0 <= reg_end <= 32, "reg_end must be in [0,32]"
        assert reg_end > reg, "reg_end must be greater than reg"
        arg = 0
        encoded_counter_reg = None
        if counter_reg is not None:
            assert 0 <= counter_reg <= self.COUNTER_REG_MASK, "counter_reg must fit in the REPEAT counter field"
            encoded_counter_reg = counter_reg
            arg |= self.COUNTER_MODE_FLAG
        if count_counter_reg is not None:
            assert 0 <= count_counter_reg <= self.COUNTER_REG_MASK, "count_counter_reg must fit in the REPEAT counter field"
            if encoded_counter_reg is not None and encoded_counter_reg != count_counter_reg:
                raise ValueError("REPEAT can only encode one counter register")
            encoded_counter_reg = count_counter_reg
            arg |= self.COUNT_COUNTER_MODE_FLAG
        if accumulate:
            arg |= self.ACCUMULATE_MODE_FLAG
        if encoded_counter_reg is not None:
            arg |= encoded_counter_reg
        super().__init__(
            opcode=opcode.OP_REPEAT,
            num_slots=(reg_end << 8) | reg,
            arg=arg,
            size=count,
            address=delta_addr,
            cords=delta_cords,
        )

    @classmethod
    def byCounter(cls, counter_reg: int, *steps):
        insts = []
        if len(steps) == 0:
            return insts

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

        insts.append(cls(1, reg=0, reg_end=32, delta_cords=[0], counter_reg=counter_reg))
        for reg_start, reg_end, delta_cords in regcords:
            insts += [cls(1, reg=reg_start, reg_end=reg_end, delta_cords=delta_cords, counter_reg=counter_reg)]
        for inst, _ in steps:
            insts.append(inst)
        return insts

    @classmethod
    def offsetByCounter(cls, counter_reg: int, inst, delta):
        return cls.byCounter(counter_reg, (inst, delta))

    @classmethod
    def offsetByCounters(cls, counter_offsets, inst):
        offsets = [(counter_reg, delta) for counter_reg, delta in counter_offsets]
        if len(offsets) == 0:
            return inst
        target_reg = len(offsets)
        assert target_reg < 32, "offsetByCounters can combine at most 31 counter offsets"

        insts = [cls(1, reg=0, reg_end=32, delta_cords=[0])]
        for counter_reg, delta in offsets:
            if isinstance(delta, list):
                delta_cords = delta
                delta_addr = None
            elif isinstance(delta, int):
                delta_cords = []
                delta_addr = delta
            else:
                raise ValueError("delta must be int or list[int]")
            insts.append(cls(
                1,
                reg=target_reg,
                reg_end=target_reg + 1,
                delta_addr=delta_addr,
                delta_cords=delta_cords,
                counter_reg=counter_reg,
                accumulate=True,
            ))
        insts.append(inst)
        return insts

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
    def on(cls, count: int, *steps, count_counter_reg: int | None = None):
        insts = []
        if len(steps) == 0:
            return insts
        if count == 0 and count_counter_reg is None:
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

        if count > 1 or count_counter_reg is not None:
            for reg_start, reg_end, delta_cords in regcords:
                insts += [cls(0, reg=reg_start, reg_end=reg_end, delta_cords=delta_cords, count_counter_reg=count_counter_reg)]
            insts[-1].size = count

        for inst, _ in steps:
            insts.append(inst)

        if count > 1 or count_counter_reg is not None:
            insts[-1].jump()
        return insts


class RawAddress(MemoryInstruction):
    MAX_WRITEBACK_SLOT = 30

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

    def writeback(self):
        slot_id = self.arg
        if slot_id > self.MAX_WRITEBACK_SLOT:
            raise ValueError(
                "RawAddress writeback requires a one-hot c2m mask and "
                f"therefore supports slot_id <= {self.MAX_WRITEBACK_SLOT}; "
                f"got {slot_id}"
            )
        return super().writeback()


class IssueBarrier(MemoryInstruction):
    def __init__(self, bar: int):
        super().__init__(opcode=opcode.OP_ISSUE_BARRIER, num_slots=0, arg=0, size=0, address=0)
        self.bar(bar)


def _control_pointer(value: torch.Tensor | int, name: str) -> int:
    if isinstance(value, torch.Tensor):
        return get_tensor_address(value)
    if not isinstance(value, int):
        raise TypeError(f"{name} must be a CUDA tensor or integer address")
    if not 0 <= value < 2**64:
        raise ValueError(f"{name} must fit in uint64")
    return value


class TerminateComm(CommunicationInstruction):
    def __init__(self):
        super().__init__(comm_opcode.COMM_TERMINATE)


class CommWaitBarrier(CommunicationInstruction):
    def __init__(self, bar: int):
        if not 0 <= bar < 2**16:
            raise ValueError("bar must fit in uint16")
        super().__init__(comm_opcode.COMM_WAIT_BARRIER, size=bar)


class CommRecordEvent(CommunicationInstruction):
    def __init__(self, event: int):
        if not 0 <= event < config.num_profile_events:
            raise ValueError("event is outside the VDCores profile buffer")
        super().__init__(comm_opcode.COMM_RECORD_EVENT, size=event)


class NvshmemPut(CommunicationInstruction):
    """Issue-#25 same-symmetric-address PUT with an explicit signal id."""

    requires_signal_array = True

    def __init__(
        self,
        address: torch.Tensor | int,
        nbytes: int,
        target_pe: int,
        signal_id: int = 0,
    ):
        if not 0 < nbytes < 2**32:
            raise ValueError("nbytes must be in [1, 2**32)")
        if not 0 <= target_pe < 2**8:
            raise ValueError("target_pe must fit in 8 bits")
        if not 0 <= signal_id < 2**8:
            raise ValueError("signal_id must fit in 8 bits")
        super().__init__(
            opcode=comm_opcode.COMM_NVSHMEM_PUT,
            arg0=nbytes >> 16,
            arg1=(signal_id << 8) | target_pe,
            size=nbytes & 0xFFFF,
            address=_control_pointer(address, "address"),
        )


class NvshmemWait(CommunicationInstruction):
    requires_signal_array = True

    def __init__(self, signal_id: int = 0, value: int = 1):
        if not 0 <= signal_id < 2**16:
            raise ValueError("signal_id must fit in 16 bits")
        if not 0 <= value < 2**64:
            raise ValueError("value must fit in uint64")
        super().__init__(
            opcode=comm_opcode.COMM_NVSHMEM_WAIT,
            size=signal_id,
            address=value,
        )


class MemoryPoolSubmit(CommunicationInstruction):
    requires_signal_array = True

    def __init__(
        self,
        request: torch.Tensor | int,
        pool_pe: int,
        submit_signal: int,
    ):
        if not 0 <= pool_pe < 2**16:
            raise ValueError("pool_pe must fit in 16 bits")
        if not 0 <= submit_signal < 2**16:
            raise ValueError("submit_signal must fit in 16 bits")
        super().__init__(
            opcode=comm_opcode.COMM_MEMORY_POOL_SUBMIT,
            arg0=pool_pe,
            size=submit_signal,
            address=_control_pointer(request, "request"),
        )


class MemoryPoolWait(CommunicationInstruction):
    requires_signal_array = True

    def __init__(self, request: torch.Tensor | int, *, pool_pe: int):
        if not 0 <= pool_pe < 2**16:
            raise ValueError("pool_pe must fit in 16 bits")
        super().__init__(
            opcode=comm_opcode.COMM_MEMORY_POOL_WAIT,
            arg0=pool_pe,
            address=_control_pointer(request, "request"),
        )


class MemoryPoolRun(CommunicationInstruction):
    requires_signal_array = True

    def __init__(self, config_tensor: torch.Tensor | int, expected_requests: int):
        if not 0 <= expected_requests < 2**32:
            raise ValueError("expected_requests must fit in uint32")
        super().__init__(
            opcode=comm_opcode.COMM_MEMORY_POOL_RUN,
            arg0=expected_requests >> 16,
            size=expected_requests & 0xFFFF,
            address=_control_pointer(config_tensor, "config_tensor"),
        )


class PoolSliceExchange(PoolInstruction):
    """Configure a pool-slice PoolInst program.

    This first stream entry selects the specialized runtime and carries its
    rank-local configuration. DynamicRead operations follow it in the same
    PoolInst stream.
    """

    requires_signal_array = True
    selects_pool_execute_warp = True
    wire_opcode = getattr(pool_opcode, "POOL_SLICE_EXCHANGE", 1)

    def __init__(
        self,
        config_tensor: torch.Tensor | int,
        *,
        write_barrier: int,
        dispatch_barrier_base: int,
        compute_barrier_base: int,
    ):
        if not 0 <= write_barrier < 2**16:
            raise ValueError("write_barrier must fit in uint16")
        if not 0 <= dispatch_barrier_base < 2**16:
            raise ValueError("dispatch_barrier_base must fit in uint16")
        if not 0 <= compute_barrier_base < 2**16:
            raise ValueError("compute_barrier_base must fit in uint16")
        super().__init__(
            self.wire_opcode,
            size=write_barrier,
            arg0=dispatch_barrier_base,
            arg1=compute_barrier_base,
            address=_control_pointer(config_tensor, "config_tensor"),
        )


class PoolSliceWeightedExchange(PoolSliceExchange):
    """Run the same gathered-read protocol with compiled inline EP combine."""

    wire_opcode = getattr(pool_opcode, "POOL_SLICE_WEIGHTED_EXCHANGE", 2)


class PoolSliceHostWeightedExchange(PoolSliceExchange):
    """Run weighted PoolInst with Grace verbs payload delivery only."""

    wire_opcode = getattr(pool_opcode, "POOL_SLICE_HOST_WEIGHTED_EXCHANGE", 3)


class PoolSliceMultimemExchange(PoolSliceExchange):
    """Run the unified scheduler-worker protocol with GB300 multimem reduce."""

    wire_opcode = getattr(pool_opcode, "POOL_SLICE_MULTIMEM_EXCHANGE", 5)


class PoolSliceGinWeightedExchange(PoolSliceExchange):
    """Run weighted PoolInst with the compile-time NCCL GIN transport."""

    wire_opcode = getattr(pool_opcode, "POOL_SLICE_GIN_WEIGHTED_EXCHANGE", 4)


class PoolSliceDynamicReadCopy(PoolInstruction):
    """Persistent per-expert gathered read in a pool-slice instruction queue."""

    wire_opcode = 0x100
    selects_pool_execute_warp = False

    def __init__(
        self,
        config_tensor: torch.Tensor | int,
        *,
        local_reader: int,
        write_barrier: int,
        dispatch_barrier_base: int,
    ):
        super().__init__(
            self.wire_opcode,
            size=local_reader,
            arg0=write_barrier,
            arg1=dispatch_barrier_base,
            address=_control_pointer(config_tensor, "config_tensor"),
        )


class PoolSliceDynamicReadReduceAdd(PoolInstruction):
    """Persistent combine-plan read using the shared pool worker CTAs."""

    wire_opcode = 0x101
    selects_pool_execute_warp = False

    def __init__(
        self,
        config_tensor: torch.Tensor | int,
        *,
        plan_rank: int,
        compute_barrier_base: int,
    ):
        super().__init__(
            self.wire_opcode,
            size=plan_rank,
            arg0=compute_barrier_base,
            address=_control_pointer(config_tensor, "config_tensor"),
        )


class CC0(MemoryInstruction):
    def __init__(self, tokens: torch.Tensor, idx: int, hidden_size: int = 4096, dtype_size: int = 2):
        addr = get_tensor_address(tokens[idx])
        row_bytes = hidden_size * dtype_size
        if row_bytes <= 0:
            raise ValueError(f"CC0 requires a positive embedding row size in bytes, got {row_bytes}")
        if (row_bytes & (row_bytes - 1)) == 0:
            shift = row_bytes.bit_length() - 1
            super().__init__(opcode=opcode.OP_CC0, num_slots=0, arg=shift, size=0, address=addr)
            return
        super().__init__(opcode=opcode.OP_CC0_ROW_BYTES, num_slots=0, arg=0, size=row_bytes, address=addr)


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
                1: opcode.OP_ALLOC_WB_TMA_STORE_1D,
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
        actions = ["load", "store"]
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
    "Gemv_M64N8K64",
    "Gemv_M64N8K128",
    "Gemv_M64N8B2",
    "Gemm_M64N64",
    "Gemm_M64N64K64",
    "Gemm_M64N128K64",
    "Gemv_M64N8_ROPE_128",
    "Gemv_M192N16",
    "Gemv_M64N8_MMA",
    "WGMMA_64x256x64_F16",
    "WGMMA_64x256x64_BF16",
    "ROPE_INTERLEAVE_512",
    "ATTENTION_M64N64K16_F16_F32_64_64_hdim",
    "ATTENTION_M64N64K16_F16_F32_64_64_hdim_MMA",
    "ATTENTION_M64N64K16_F16_F32_64_64_hdim64",
    "ATTENTION_M64N64K16_F16_F32_64_64_hdim64_MMA",
    "ATTENTION_M64N64K16_F16_F32_64_64_hdim_split",
    "ATTENTION_M64N64K16_F16_F32_64_64_hdim_split_MMA",
    "ATTN_SPLIT_POST_REDUCE",
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
    "CommunicationInstruction",
    "PoolInstruction",
    "TerminateComm",
    "CommWaitBarrier",
    "CommRecordEvent",
    "MemoryInstruction",
    "TerminateM",
    "LoopM",
    "CounterOffsetMemoryInstruction",
    "RepeatM",
    "RawAddress",
    "IssueBarrier",
    "NvshmemPut",
    "NvshmemWait",
    "MemoryPoolSubmit",
    "MemoryPoolWait",
    "MemoryPoolRun",
    "PoolSliceExchange",
    "PoolSliceWeightedExchange",
    "PoolSliceHostWeightedExchange",
    "PoolSliceMultimemExchange",
    "PoolSliceGinWeightedExchange",
    "PoolSliceDynamicReadCopy",
    "PoolSliceDynamicReadReduceAdd",
    "CC0",
    "RegStore",
    "RegLoad",
    "TmaLoad1D",
    "TmaStore1D",
    "TmaTensor",
]
