import argparse

import torch

from dae.launcher import *
from dae.schedule import SchedGemm
from dae.util import dae_app, tensor_diff


gpu = torch.device("cuda")
torch.manual_seed(0)
dtype = torch.bfloat16

Atom = Gemm_M64N64
TileM, TileN, TileK = Atom.MNK

M = 64
N = 64
K = 4096
num_sms = 32

assert K % TileK == 0
assert M % TileM == 0

matA = torch.rand(M, K, dtype=dtype, device=gpu) - 0.5
matB = torch.rand(N, K, dtype=dtype, device=gpu) - 0.5
matC = torch.zeros(M, N, dtype=dtype, device=gpu)

dae = Launcher(num_sms, device=gpu)

loadA = TmaTensor(dae, matA).wgmma_load(TileM, TileK, Major.K)
loadB = TmaTensor(dae, matB).wgmma_load(TileN, TileK * Atom.n_batch, Major.K)
storeC = TmaTensor(dae, matC).wgmma_store(TileM, TileN, Major.K)
reduceC = TmaTensor(dae, matC).wgmma("reduce", TileM, TileN, Major.K)

store_tensor = reduceC 

gemm = SchedGemm(
    Atom,
    MNK=(M, N, K),
    tmas=(loadA, loadB, store_tensor),
).place(num_sms)

dae.i(
    gemm,
    TerminateC(),
    TerminateM(),
)

ref = matA @ matB.t()
res = matC

dae_app(dae)

tensor_diff(Atom.__name__, ref, res)
