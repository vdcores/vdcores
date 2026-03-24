import torch
from dae.launcher import *
from dae.runtime import build_tma_desc
import numpy as np

def uniform_rand_scaled(*shape: int, dtype=torch.float16, device='cuda', scale=0.1):
    return (torch.rand(*shape, dtype=dtype, device=device) - 0.5) * scale

# get dims other than the specified index and last one
def get_other_dims(dim: int, i: int):
    return [j for j in range(dim-1) if j != i and j - dim != i]

# mat is always pytorch-major
def cord_func_MN_major(mat: torch.Tensor, rank: int, iK = -2):
    assert iK != -1 and iK < len(mat.shape) - 1, "iK must not be the last dim"

    def cord_func(*cords):
        assert len(mat.shape) == len(cords), f"cords length {len(cords)} does not match mat rank {len(mat.shape)}"
        
        if rank == 2:
            assert iK == -2
            return [cords[-1], cords[-2]]
        else:
            other_dims = get_other_dims(len(mat.shape), iK)
            other_strides = [mat.stride(i) for i in other_dims]
            other_cords = [cords[i] for i in other_dims]

            rest = np.dot(other_strides, other_cords) // mat.stride(other_dims[-1])
            rest = int(rest)

            if rank == 3:
                return [cords[-1], cords[iK], rest]
            elif rank == 4:
                assert iK == -2
                return [0, 0, cords[-1] // 64, cords[-2] // 8]
            elif rank == 5:
                # fix 0 for first dim
                return [0, cords[-1] // 64, cords[iK] // 8, rest]
            else:
                raise ValueError(f"Unsupported rank {rank} for mn-major cord function")
    return cord_func

def build_tma_wgmma_mn(mat: torch.Tensor, tileM: int, tileK: int, iK = -2, swizzle=128):
    assert iK != -1 and iK < len(mat.shape) - 1, "iK must not be the last dim"
    # build 4d by default
    assert len(mat.shape) >= 2, "Input matrix must be at least 2D"
    K, M = mat.shape[iK], mat.shape[-1]
    elsize = mat.element_size()

    blockM = 128 // elsize
    blockK = 8

    assert tileM % blockM == 0, f"tileM {tileM} must be multiple of blockM {blockM}"

    if blockM != tileM:
        global_dims = [blockM, blockK, M // blockM, K // blockK]
        global_strides = [mat.stride(iK), blockM, mat.stride(iK) * blockK]
        box_dims = [blockM, blockK, tileM // blockM, tileK // blockK]
    else:
        global_dims = [M, K]
        global_strides = [mat.stride(iK)]
        box_dims = [tileM, tileK]
    
    if len(mat.shape) > 2:
        # collapse other dims
        # the stride of the collapsed dim is in the unit of stride(inner_most_dim)
        # to bypass glob_dim range check we need to also get logical size = numel / stride(inner_most_dim)
        other_dims = get_other_dims(len(mat.shape), iK)
        inner_most_others = other_dims[-1]
        size_otherthan_k_or_m = mat.numel() // mat.stride(inner_most_others)
        
        global_dims.append(size_otherthan_k_or_m)
        global_strides.append(mat.stride(inner_most_others))
        box_dims.append(1)

    rank = len(global_dims)
    box_strides = [1] * rank
    global_strides = [s * elsize for s in global_strides]

    return rank, runtime.build_tma_desc(
        mat,
        global_dims,
        global_strides,
        box_dims,
        box_strides,
        swizzle,
        0
    )

# pytorch-major cord functions
def cord_func_K_major(mat: torch.Tensor, rank: int, iN = -2):
    assert iN != -1 and iN < len(mat.shape) - 1, "iN must not be the last dim"

    def cord_func(*cords):
        assert len(mat.shape) == len(cords), f"cords length {len(cords)} does not match mat rank {len(mat.shape)}"
        
        if rank == 3:
            assert iN == -2
            return [0, cords[-2], cords[-1] // 64]
        elif rank == 4:
            other_dims = get_other_dims(len(mat.shape), iN)
            other_strides = [mat.stride(i) for i in other_dims]
            other_cords = [cords[i] for i in other_dims]

            rest = np.dot(other_strides, other_cords) // mat.stride(other_dims[-1])
            rest = int(rest)
            return [0, cords[iN], cords[-1] // 64, rest]
        else:
            raise ValueError(f"Unsupported rank {rank} for mn-major cord function")
    return cord_func

def build_tma_wgmma_k(mat: torch.Tensor, tileK: int, tileN: int, iN: int = -2, swizzle=128):
    assert iN != -1 and iN < len(mat.shape) - 1, "iN must not be the last dim"
    N, K = mat.shape[iN], mat.shape[-1]
    elsize = mat.element_size()
    blockK = 128 // elsize

    assert tileK % blockK == 0, "tileK must be multiple of blockK"

    glob_dims = [blockK, N, K // blockK]
    glob_strides = [mat.stride(iN), blockK]
    box_dims = [blockK, tileN, tileK // blockK]

    if len(mat.shape) > 2:
        # collapse other dims
        other_dims = get_other_dims(len(mat.shape), iN)
        inner_most_others = other_dims[-1]
        size_otherthan_k_or_n = mat.numel() // mat.stride(inner_most_others)
        
        # find stride of inner most dims other than K or N
        glob_dims.append(size_otherthan_k_or_n)
        glob_strides.append(mat.stride(inner_most_others))
        box_dims.append(1)
    
    glob_strides = [s * elsize for s in glob_strides]
    rank = len(glob_dims)
    box_strides = [1] * rank
    return rank, runtime.build_tma_desc(
        mat,
        glob_dims,
        glob_strides,
        box_dims,
        box_strides,
        swizzle,
        0
    )