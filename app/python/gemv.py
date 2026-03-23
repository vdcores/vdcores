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
TileM, TileN, TileK = Atom.MNK

M = 64
K = 4096
N = 64
num_sms = N // TileN * 8

assert K % TileK == 0
assert M % TileM == 0

matA = torch.rand(M, K, dtype=dtype, device=gpu) - 0.5
matB = torch.rand(N, K, dtype=dtype, device=gpu) - 0.5
matC = torch.zeros(N, M, dtype=dtype, device=gpu)

dae = Launcher(num_sms, device=gpu)

loadA = TmaTensor(dae, matA).wgmma_load(TileM, TileK, Major.K)

insts = []
for i in range(N // TileN):
    sm_for_me = num_sms // (N // TileN)
    loadB = TmaTensor(dae, matB[i*TileN:(i+1)*TileN]).wgmma_load(TileN, TileK * Atom.n_batch, Major.K)
    reduceC = TmaTensor(dae, matC[i*TileN:(i+1)*TileN]).wgmma("reduce", TileN, TileM, Major.MN)

    store_tensor = reduceC 

    gemv = SchedGemv(
        Atom,
        MNK=(M, TileN, K),
        tmas=(loadA, loadB, store_tensor),
    ).place(sm_for_me, i * sm_for_me)
    insts.append(gemv)

dae.i(
    insts,
    TerminateC(),
    TerminateM(),
)

ref = matA @ matB.t()
res = matC.t()

dae_app(dae)

tensor_diff(Atom.__name__, ref, res)
