from . import runtime
from .runtime import opcode
import numpy as np
import torch
import copy
from math import prod
from enum import Enum

class Major(Enum):
    MN = 1
    K = 2

def get_tensor_address(tensor: torch.Tensor) -> int:
    assert tensor.is_contiguous()
    assert tensor.device.type == 'cuda'

    return tensor.data_ptr()

def addr2cords(address: int) -> list[int]:
    addr_bytes = address.to_bytes(8, byteorder='little')
    cords = []
    for i in range(4):
        cords.append(int.from_bytes(addr_bytes[i*2:i*2+2], byteorder='little'))
    return cords
def cords2addr(cords: list[int]) -> int:
    addr_bytes = bytearray(8)
    for i in range(4):
        addr_bytes[i*2:i*2+2] = cords[i].to_bytes(2, byteorder='little')
    return int.from_bytes(addr_bytes, byteorder='little')

def bytes2slots(bytes: int) -> int:
    return (bytes + runtime.config.slot_size - 1) // runtime.config.slot_size

#####
# Cord adapters
#####

class CordAdapter:
    def __init__(self, inner):
        self.inner = inner

    def cord(self, *cords):
        raise NotImplementedError()

    def __getattr__(self, name):
        return getattr(self.inner, name)

class StaticCordAdapter(CordAdapter):
    def cord(self, *cords):
        return self.inner

def wrap_static(*tmas):
    return tuple(StaticCordAdapter(tma) for tma in tmas)

class ToConvertedCordAdapter(CordAdapter):
    def __init__(self, inner, convert):
        super().__init__(inner)
        self.convert = convert

    def cord(self, *cords):
        converted = self.convert(*cords)
        return self.inner.cord(*converted)

class ToLinearCordAdapter(ToConvertedCordAdapter):
    def __init__(self, inner, delta: int):
        super().__init__(inner, lambda sm: (sm * delta,))

class ToRopeTableCordAdapter(ToConvertedCordAdapter):
    def __init__(self, inner, batch_seq_len: int, tile_repeats: int = 2):
        super().__init__(inner, lambda sm: (sm % tile_repeats, batch_seq_len))

class ToSplitMCordAdapter(ToConvertedCordAdapter):
    def __init__(self, inner, num_sms: int, tile_m: int):
        super().__init__(inner, lambda sm: (0, (sm % num_sms) * tile_m))

class ToAttnKVStoreCordAdapter(ToConvertedCordAdapter):
    def __init__(self, inner, num_sms: int, tile_m: int, position: int):
        super().__init__(
            inner,
            lambda sm: ((sm % num_sms) * tile_m, position, 0),
        )

class ToAttnVStoreCordAdapter(ToConvertedCordAdapter):
    def __init__(self, inner, position: int):
        super().__init__(inner, lambda _, m: (m, position, 0))

class ToAttnCurrentKStore1DAdapter(ToConvertedCordAdapter):
    def __init__(self, inner, position: int, max_seq_len: int, num_kv_heads: int, head_dim: int, dtype_size: int = 2):
        row_elems = num_kv_heads * head_dim
        super().__init__(
            inner,
            lambda sm: (
                (
                    ((sm // num_kv_heads) * max_seq_len + position) * row_elems
                    + (sm % num_kv_heads) * head_dim
                ) * dtype_size,
            ),
        )

#####
# TMA Builders
#####

def cord_id(mat: torch.Tensor, rank: int):
    def cordf(*cords):
        assert len(cords) == rank, f"cords length {len(cords)} does not match expected rank {rank}"
        return list(cords)
    return cordf

def tma_store_attn_kv(mat: torch.Tensor, TileM: int, TileK: int):
    assert TileM == 64, "TileM must be 64 for store K"
    assert TileK == 8, "TileK must be 8 for store K"
    shape = mat.shape
    elsize = mat.element_size()
    assert elsize == 2, "Only support float16/bfloat16 output for store K"
    assert len(shape) == 3, "Only support 3D input for store K"

    global_dims = [shape[-1], shape[-2], shape[-3]]
    global_strides = [elsize * shape[-1], elsize * shape[-1] * shape[-2]]
    box_dims = [TileM, 1, TileK]
    box_strides = [1, 1, 1]

    return len(global_dims), runtime.build_tma_desc(
        mat,
        global_dims,
        global_strides,
        box_dims,
        box_strides,
        128,
        0
    )

# for building table
def tma_load_tbl(mat: torch.Tensor, TileM: int, TileN: int):
    assert mat.element_size() == 2, "Only support float16/bfloat16 output"
    head_dim = mat.shape[-1]
    MAX_SEQ_LEN = mat.shape[0]
    assert TileM == 64, "TileM must be 64 for rope table"
    assert head_dim % 64 == 0, "rope table head_dim must be a multiple of 64"

    # assign a different ROPE table for each single req in a batch of TileN
    s = mat.element_size()
    tile_repeats = head_dim // 64
    rope_tile_width = 64
    # repeat for tileN times
    glob_dims = [rope_tile_width, TileN, tile_repeats, MAX_SEQ_LEN]
    glob_strides = [head_dim * s, rope_tile_width * s, head_dim * TileN * s]
    box_dims = [rope_tile_width, TileN, 1, 1]
    rank = len(glob_dims)
    box_strides = [1] * rank
    return rank, runtime.build_tma_desc(
        mat,
        glob_dims,
        glob_strides,
        box_dims,
        box_strides,
        128,
        0
    )

def cord_load_tbl(mat: torch.Tensor, rank: int):
    assert rank == 4, "Only support 4D tma load for rope table"
    # cords: [half : 2, batch_seq : max_seq_len]
    def cfunc(*cords):
        assert len(cords) == 2, f"cords length {len(cords)} should be 2 for rope table"
        return [0, 0, cords[0], cords[1]]
    return cfunc

# 1d cord funcs
def build_tma_1d(mat: torch.Tensor, size1: int, size2: int):
    global_size = mat.numel() 
    size = size1 * size2
    assert global_size % 64 == 0, "Global size must be multiple of 64 bytes for 1d TMA"
    assert size % 64 == 0, "Tile size must be multiple of 64 bytes for 1d TMA"

    global_dims = [64, global_size // 64]
    global_strides = [64 * 2]
    box_dims = [64, size // 64]
    box_strides = [1, 1]

    return len(global_dims), runtime.build_tma_desc(
        mat,
        global_dims,
        global_strides,
        box_dims,
        box_strides,
        0, 0
    )

def cord_func_tma_1d(mat: torch.Tensor, rank: int):
    assert rank == 2, f"Rank {rank} is not supported for 1d cord function"
    def cord_func(addr):
        assert addr % 64 == 0, "Address must be aligned to 64 bytes for 1d cord function"
        return [0, addr // 64]
    return cord_func


def build_tma_rowmajor_2d(
    mat: torch.Tensor, tile_rows: int, tile_cols: int
):
    """Build an unswizzled 2D descriptor for a row-major shared tile."""
    assert mat.ndim == 2, "row-major 2D TMA requires a rank-2 tensor"
    rows, cols = mat.shape
    element_size = mat.element_size()
    assert 0 < tile_rows <= rows and 0 < tile_cols <= cols
    return 2, runtime.build_tma_desc(
        mat,
        [cols, rows],
        [cols * element_size],
        [tile_cols, tile_rows],
        [1, 1],
        0,
        0,
    )


def cord_func_rowmajor_2d(mat: torch.Tensor, rank: int):
    assert rank == 2

    def cord_func(row: int, col: int):
        return [col, row]

    return cord_func

# pytorch-major cord functions
def cord_func_2d_mnmajor(mat: torch.Tensor, rank : int):
    def cord_func(*cords):
        assert len(mat.shape) == len(cords), f"cords length {len(cords)} does not match mat rank {len(mat.shape)}"
        rest = np.dot(mat.stride()[:-2], cords[:-2]) // prod(mat.shape[-2:])
        rest = int(rest)

        if rank == 2: # (K, M)
            return [cords[-1], cords[-2]]
        elif rank == 3: #(...rest, K, M)
            return [cords[-1], cords[-2], rest]
        elif rank == 4: # (B, C, M//blockM, K//blockK)
            return [0, 0, cords[-1] // 64, cords[-2] // 8]
        elif rank == 5: # (B, C, M//blockM, K//blockK, 1)
            # fix0 for first dim, so we return dim 1-4
            return [0, cords[-1] // 64, cords[-2] // 8, rest]
        else:
            raise ValueError(f"Unsupported rank {rank} for mn-major cord function")
    return cord_func
        
def build_tma_wgmma_mnmajor(mat: torch.Tensor, tileM : int, tileK : int):
    # build 4d by default
    assert len(mat.shape) >= 2, "Input matrix must be at least 2D"
    K, M = mat.shape[-2], mat.shape[-1]
    elsize = mat.element_size()

    blockM = 128 // elsize
    blockK = 8

    assert tileM % blockM == 0, "tileM must be multiple of blockM"

    if blockM != tileM:
        global_dims = [blockM, blockK, M // blockM, K // blockK]
        global_strides = [M * elsize, blockM * elsize, M * blockK * elsize]
        box_dims = [blockM, blockK, tileM // blockM, tileK // blockK]
    else:
        global_dims = [M, K]
        global_strides = [M * elsize]
        box_dims = [tileM, tileK]
    
    if len(mat.shape) > 2:
        global_dims.append(prod(mat.shape[:-2]))
        global_strides.append(K * M * elsize)
        box_dims.append(1)
    rank = len(global_dims)
    box_strides = [1] * rank

    # print(f"build_tma_wgmma_mnmajor(tileMN={tileM},tileK={tileK}) = {{")
    # print("  global_dims:", global_dims)
    # print("  global_strides:", global_strides)
    # print("  box_dims:", box_dims)
    # print("  box_strides:", box_strides)
    # print("}")

    return rank, runtime.build_tma_desc(
        mat,
        global_dims,
        global_strides,
        box_dims,
        box_strides,
        128,
        0
    )


def build_tma_wgmma_mnmajor_m128n8(mat: torch.Tensor, tileM: int, tileK: int):
    """Build a reducible rank-3 descriptor for an SM100 M128xN8 output tile."""
    assert mat.dim() == 2, "M128N8 output currently requires a 2D [N, M] tensor"
    assert mat.element_size() == 2, "M128N8 output requires a 16-bit element type"
    n, m = mat.shape
    assert tileM == 128 and tileK == 8
    assert n == 8 and m % 64 == 0

    element_size = mat.element_size()
    global_dims = [64, 8, m // 64]
    global_strides = [m * element_size, 64 * element_size]
    box_dims = [64, 8, 2]
    rank = len(global_dims)
    return rank, runtime.build_tma_desc(
        mat,
        global_dims,
        global_strides,
        box_dims,
        [1] * rank,
        128,
        0,
    )


def cord_func_m128n8_output(mat: torch.Tensor, rank: int):
    assert rank == 3
    def cord_func(*cords):
        assert len(cords) == 2, f"expected [N, M] coordinates, got {cords}"
        n, m = cords
        assert n == 0, "M128N8 output tiles cover the full N=8 dimension"
        return [0, 0, m // 64]
    return cord_func


def build_tma_wgmma_mnmajor_m128n8_grouped(
    mat: torch.Tensor, tileM: int, tileK: int, *, output_groups: int
):
    """Pack strided M128xN8 outputs into one 8 KiB reduction transaction."""
    assert mat.dim() == 2
    assert mat.element_size() == 2
    n, m = mat.shape
    assert n == 8 and tileK == 8
    assert output_groups == 4
    assert tileM == 128 * output_groups
    assert m % tileM == 0

    m_groups = m // tileM
    output_group_stride = m_groups * 128
    element_size = mat.element_size()
    global_dims = [64, 8, 2 * m_groups, output_groups]
    global_strides = [
        m * element_size,
        64 * element_size,
        output_group_stride * element_size,
    ]
    box_dims = [64, 8, 2, output_groups]
    return 4, runtime.build_tma_desc(
        mat,
        global_dims,
        global_strides,
        box_dims,
        [1] * 4,
        128,
        0,
    )


def cord_func_m128n8_grouped_output(
    mat: torch.Tensor, rank: int, *, output_groups: int
):
    assert rank == 4
    assert mat.shape[1] % (128 * output_groups) == 0

    def cord_func(*cords):
        assert len(cords) == 2
        n, m = cords
        assert n == 0 and m % 128 == 0
        return [0, 0, 2 * (m // 128), 0]

    return cord_func

# pytorch-major cord functions
def cord_func_2d_kmajor(mat: torch.Tensor, rank : int):
    block_k = 128 // mat.element_size()

    def cord_func(*cords):
        assert len(mat.shape) == len(cords), f"cords length {len(cords)} does not match mat rank {len(mat.shape)}"
        rest = np.dot(mat.stride()[:-2], cords[:-2]) // prod(mat.shape[-2:])
        rest = int(rest)
        
        if rank == 3: #(...rest, M, K)
            return [0, cords[-2], cords[-1] // block_k]
        elif rank == 4: # (B, C, M//blockM, K//blockK)
            return [0, cords[-2], cords[-1] // block_k, rest]
        else:
            raise ValueError(f"Unsupported rank {rank} for mn-major cord function")
    return cord_func

def build_tma_wgmma_kmajor(mat: torch.Tensor, tileK : int, tileN : int):
    # build 3d by default
    assert len(mat.shape) >= 2, "Input matrix must be at least 2D"
    N, K = mat.shape[-2], mat.shape[-1]
    elsize = mat.element_size()

    blockK = 128 // elsize

    assert tileK % blockK == 0, "tileK must be multiple of blockK"

    global_dims = [blockK, N, K // blockK]
    global_strides = [K *elsize, blockK * elsize]
    box_dims = [blockK, tileN, tileK // blockK]
    box_strides = [1, 1, 1]

    # if n > 2, fold rest dims into a single dimension
    if len(mat.shape) > 2:
        global_dims.append(prod(mat.shape[:-2]))
        global_strides.append(N * K * elsize)
        box_dims.append(1)
        box_strides.append(1)

    # print(f"build_tma_wgmma_kmajor(tileK={tileK},tileMN={tileN}) = {{")
    # print("  global_dims:", global_dims)
    # print("  global_strides:", global_strides)
    # print("  box_dims:", box_dims)
    # print("  box_strides:", box_strides)
    # print("}")
    
    return len(global_dims), runtime.build_tma_desc(
        mat,
        global_dims,
        global_strides,
        box_dims,
        box_strides,
        128,
        0
    )


# Pack complete logical ``(M, K)`` weights so each SM100 UMMA tile is one
# contiguous TMA allocation. The descriptor below interprets this rank-5 view
# without copying it again at launch time.
def pack_weight_tile_major(weight: torch.Tensor, tile_m: int, tile_k: int):
    if not weight.is_contiguous():
        raise ValueError("tile-major weight packing requires contiguous input")
    output_rows, input_cols = weight.shape
    if output_rows % tile_m or input_cols % tile_k or tile_k % 64:
        raise ValueError(
            f"weight {tuple(weight.shape)} is incompatible with M{tile_m}K{tile_k} packing"
        )
    return (
        weight.view(
            output_rows // tile_m,
            tile_m,
            input_cols // tile_k,
            tile_k // 64,
            64,
        )
        .permute(2, 0, 3, 1, 4)
        .contiguous()
    )


def cord_func_2d_tile_major(mat: torch.Tensor, rank: int):
    """Map logical ``(M, K)`` into [K-tile, M-tile, K-block, M-row, 64]."""
    assert rank == 5, f"tile-major weights require a rank-5 TMA map, got {rank}"
    _, _, subblocks, tile_rows, block_k = mat.shape
    tile_k = subblocks * block_k

    def cord_func(m: int, k: int):
        assert m % tile_rows == 0, "tile-major M coordinates must be tile aligned"
        assert k % tile_k == 0, "tile-major K coordinates must be tile aligned"
        return [0, 0, m // tile_rows, k // tile_k]

    return cord_func


def build_tma_wgmma_tile_major(mat: torch.Tensor, tileK: int, tileN: int):
    """Build one contiguous global-memory allocation per logical UMMA tile."""
    assert mat.dim() == 5, (
        "tile-major weights must have shape [K/tileK,M/tileM,tileK/64,tileM,64]"
    )
    k_tiles, m_tiles, subblocks, tile_rows, block_k = mat.shape
    elsize = mat.element_size()
    assert block_k == 128 // elsize == 64
    assert tile_rows == tileN
    assert subblocks * block_k == tileK

    tile_bytes = tile_rows * tileK * elsize
    global_dims = [block_k, tile_rows, subblocks, m_tiles, k_tiles]
    global_strides = [
        block_k * elsize,
        tile_rows * block_k * elsize,
        tile_bytes,
        m_tiles * tile_bytes,
    ]
    box_dims = [block_k, tile_rows, subblocks, 1, 1]
    return 5, runtime.build_tma_desc(
        mat,
        global_dims,
        global_strides,
        box_dims,
        [1] * 5,
        128,
        0,
    )
