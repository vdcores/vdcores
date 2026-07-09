import argparse

import torch
import torch.nn.functional as F

from dae.launcher import *
from dae.schedule import Schedule, ListSchedule
from dae.util import dae_app, tensor_diff
from manual_sum import manual_reduction

class ManualGemm(Schedule):
    def __init__(self, Atom, MNK, tmas, MNK_base, num_k_folds):
        super().__init__()
        self.Atom = Atom
        self.MNK = MNK
        self.tmas = tmas
        self.MNK_base = MNK_base
        self.num_k_folds = num_k_folds

    def schedule(self, sm: int):
        if sm < 0:
            return []

        BaseM, BaseN, BaseK = self.MNK_base
        TileM, TileN, TileK = self.Atom.MNK
        loadA, loadB, reduceC = self.tmas
        M, N, K = self.MNK

        instructions = []

        m_tiles = M // TileM
        n_tiles = N // TileN

        # Distribute output row tiles across SMs --> each SM handles every num_sms-th M tile
        for m_tile_id in range(sm, m_tiles, self.num_sms):
            m = BaseM + m_tile_id * TileM

            # For the current row tile sweep across all output column tiles.
            for n_tile_id in range(n_tiles):
                n = BaseN + n_tile_id * TileN

                # Accumulate this output tile by reducing over K in TileK-sized chunks
                instructions.append(self.Atom(self.num_k_folds))

                for fold in range(self.num_k_folds):
                    k = BaseK + fold * TileK
                    instructions.append(loadB.cord(n, k))
                    instructions.append(loadA.cord(m, k))

                instructions.append(reduceC.cord(m, n))

        return instructions


gpu = torch.device("cuda")
torch.manual_seed(0)
dtype = torch.bfloat16

Atom = Gemm_M64N64K64
TileM, TileN, TileK = Atom.MNK

# MLP dimensions
M = TileM * 128
N = TileN * 8
num_k_folds = 16
K = TileK * num_k_folds
num_sms = 128

tp_size = 2
n_chunk_tiles = 4

assert K % TileK == 0
assert M % TileM == 0
assert N % TileN == 0
assert N % TileK == 0

# First GEMM tensors:
#
# matC = matA @ matB.t()
# matA: M x K
# matB: N x K
# matC: M x N

matA = torch.rand(M, K, dtype=dtype, device=gpu) - 0.5
matB = torch.rand(N, K, dtype=dtype, device=gpu) - 0.5
matC = torch.zeros(M, N, dtype=dtype, device=gpu)

# First GEMM:
# N-split / column-parallel
#
# Computes:
# matC = matA @ matB.t()

n_tiles = N // TileN

assert n_tiles % tp_size == 0
n_tiles_per_rank = n_tiles // tp_size

matC.zero_()
torch.cuda.synchronize()

for tp_rank in range(tp_size): # Pretending here that the work is split across multiple “parallel workers” or “virtual GPUs.” Each one is a rank.
    rank_n_start_tile = tp_rank * n_tiles_per_rank # Split the work across ranks
    rank_n_end_tile = rank_n_start_tile + n_tiles_per_rank # Split the work across ranks

    for n_base_tile in range(rank_n_start_tile, rank_n_end_tile, n_chunk_tiles):
        cur_n_tiles = min(n_chunk_tiles, rank_n_end_tile - n_base_tile)
        N_chunk = cur_n_tiles * TileN
        BaseN = n_base_tile * TileN

        dae = Launcher(num_sms, device=gpu)

        loadA = TmaTensor(dae, matA).wgmma_load(TileM, TileK, Major.K)
        loadB = TmaTensor(dae, matB).wgmma_load(TileN, TileK * Atom.n_batch, Major.K)
        reduceC = TmaTensor(dae, matC).wgmma("reduce", TileM, TileN, Major.K)

        gemm = ManualGemm(
            Atom,
            MNK=(M, N_chunk, K),
            tmas=(loadA, loadB, reduceC),
            MNK_base=(0, BaseN, 0),
            num_k_folds=num_k_folds,
        ).place(num_sms, base_sm=0)

        dae.i(
            gemm,
            TerminateC(),
            TerminateM(),
        )

        dae_app(dae)

torch.cuda.synchronize()

hidden_ref = matA @ matB.t()
hidden_res = matC

tensor_diff("first_gemm_hidden", hidden_ref, hidden_res)

# Activation:
# matSilu = silu(matC)

matSilu = F.silu(matC).contiguous()
matSilu_ref = F.silu(hidden_ref).contiguous()

tensor_diff("silu_hidden", matSilu_ref, matSilu)

# Second GEMM tensors
#
# res = matSilu @ matD.t()
# matSilu: M x N
# matD:    N_out x N
# res:     M x N_out

N_out = TileN * 8
matD = torch.rand(N_out, N, dtype=dtype, device=gpu) - 0.5

assert N_out % TileN == 0

# For the second GEMM, the K dimension is the hidden dimension N.
num_k_folds_second = N // TileK

assert num_k_folds_second % tp_size == 0

k_folds_per_rank = num_k_folds_second // tp_size
K_shard = k_folds_per_rank * TileK

n_tiles_second = N_out // TileN

# Each TP rank computes a partial full output.
matC_partials = [
    torch.zeros(M, N_out, dtype=dtype, device=gpu)
    for _ in range(tp_size)
]

torch.cuda.synchronize()

# Second GEMM:
# K-split / row-parallel
#
# Computes partials:
# matC_partial_rank = matSilu[:, K_rank] @ matD[:, K_rank].t()
#
# Then:
# res = manual_reduction(matC_partials)

for tp_rank in range(tp_size):
    rank_k_start_fold = tp_rank * k_folds_per_rank
    BaseK = rank_k_start_fold * TileK

    matC_partial = matC_partials[tp_rank]

    for n_base_tile in range(0, n_tiles_second, n_chunk_tiles):
        cur_n_tiles = min(n_chunk_tiles, n_tiles_second - n_base_tile)
        N_chunk = cur_n_tiles * TileN
        BaseN = n_base_tile * TileN

        dae = Launcher(num_sms, device=gpu)

        loadA = TmaTensor(dae, matSilu).wgmma_load(TileM, TileK, Major.K)
        loadB = TmaTensor(dae, matD).wgmma_load(TileN, TileK * Atom.n_batch, Major.K)
        reduceC = TmaTensor(dae, matC_partial).wgmma("reduce", TileM, TileN, Major.K)

        gemm = ManualGemm(
            Atom,
            MNK=(M, N_chunk, K_shard),
            tmas=(loadA, loadB, reduceC),
            MNK_base=(0, BaseN, BaseK),
            num_k_folds=k_folds_per_rank,
        ).place(num_sms, base_sm=0)

        dae.i(
            gemm,
            TerminateC(),
            TerminateM(),
        )

        dae_app(dae)

torch.cuda.synchronize()

# Final partial-output sum using explicit CUDA reduction kernel
assert tp_size == 2

res = torch.zeros(M, N_out, dtype=dtype, device=gpu)

assert matC_partials[0].is_contiguous()
assert matC_partials[1].is_contiguous()
assert res.is_contiguous()

manual_reduction(
    matC_partials[0],
    matC_partials[1],
    res,
)

torch.cuda.synchronize()

# Optional sanity check against old PyTorch/Python sum
res_sum = sum(matC_partials)
tensor_diff("manual_reduction_vs_sum", res_sum, res)

# Full PyTorch reference
ref = F.silu(matA @ matB.t()) @ matD.t()

tensor_diff("full_mlp", ref, res)