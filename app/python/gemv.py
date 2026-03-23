import argparse

import torch

from dae.launcher import *
from dae.schedule import SchedGemv
from dae.util import dae_app, tensor_diff


gpu = torch.device("cuda")
torch.manual_seed(0)
dtype = torch.bfloat16

Atom = Gemv_M64N8
Atom.n_batch = 2
TileM, N, TileK = Atom.MNK

M = 64
K = 4096
num_sms = 8

assert K % TileK == 0
assert M % TileM == 0

matA = torch.rand(M, K, dtype=dtype, device=gpu) - 0.5
matB = torch.rand(N, K, dtype=dtype, device=gpu) - 0.5
matC = torch.zeros(N, M, dtype=dtype, device=gpu)

dae = Launcher(num_sms, device=gpu)

loadA = TmaTensor(dae, matA).wgmma_load(TileM, TileK, Major.K)
loadB = TmaTensor(dae, matB).wgmma_load(N, TileK * Atom.n_batch, Major.K)
reduceC = TmaTensor(dae, matC).wgmma("reduce", N, TileM, Major.MN)

store_tensor = reduceC 

gemv = SchedGemv(
    Atom,
    MNK=(M, N, K),
    tmas=(loadA, loadB, store_tensor),
).place(num_sms)

dae.i(
    gemv,
    TerminateC(),
    TerminateM(),
)

ref = matA @ matB.t()
res = matC.t()

dae_app(dae)

tensor_diff(Atom.__name__, ref, res)
