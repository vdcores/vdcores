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
# SM Cord helper functions
#####

def sm_cord_1d(delta: int):
    def cordf(sm: int, inst):
        return inst.cord(sm * delta)
    return cordf

def sm_cord_splitM(num_sms: int, tileM: int):
    def cordf(sm: int, inst):
        m = sm % num_sms * tileM
        return inst.cord(0, m)
    return cordf

def sm_cord_store_attn_kv(num_sms: int, tileM: int, position: int):
    def cordf(sm: int, inst):
        m = sm % num_sms * tileM
        return inst.cord(m, position, 0)
    return cordf

def sm_cord_rope(batch_seq_len : int):
    def cordf(sm: int, inst):
        return inst.cord(sm % 2, batch_seq_len)
    return cordf

def conv_m2cord_attn_store_v(position: int):
    def cordf(m, inst):
        return inst.cord(m, position, 0)
    return cordf

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
    # assume headdim = 64
    MAX_SEQ_LEN = mat.shape[0]
    assert TileM == 64, "TileM must be 64 for rope table"
    assert mat.shape[-1] == 128, "mat shape should be head_dim"

    # assign a different ROPE table for each single req in a batch of TileN
    s = mat.element_size()
    # repeat for tileN times
    glob_dims = [64, TileN, 2, MAX_SEQ_LEN]
    glob_strides = [128 * s, 64 * s, 128 * TileN * s]
    box_dims = [64, TileN, 1, 1]
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

# pytorch-major cord functions
def cord_func_2d_kmajor(mat: torch.Tensor, rank : int):
    def cord_func(*cords):
        assert len(mat.shape) == len(cords), f"cords length {len(cords)} does not match mat rank {len(mat.shape)}"
        rest = np.dot(mat.stride()[:-2], cords[:-2]) // prod(mat.shape[-2:])
        rest = int(rest)
        
        if rank == 3: #(...rest, M, K)
            return [0, cords[-2], cords[-1] // 64]
        elif rank == 4: # (B, C, M//blockM, K//blockK)
            return [0, cords[-2], cords[-1] // 64, rest]
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