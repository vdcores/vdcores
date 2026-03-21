import os

import torch
from dae.launcher import *
from dae.util import *

gpu = torch.device("cuda")
torch.manual_seed(0)

Atom = Gemv_M64N8_MMA

TileM, N, TileK = Atom.MNK
M = int(os.getenv("GEMV_M", "4096"))
K = int(os.getenv("GEMV_K", "4096"))
num_sms = int(os.getenv("GEMV_SMS", str(M // TileM)))

assert M % TileM == 0, f"M={M} must be a multiple of {TileM}"
assert K % TileK == 0, f"K={K} must be a multiple of {TileK}"
assert num_sms == M // TileM, f"num_sms={num_sms} must match M/TileM={M // TileM} for this harness"

matA = torch.rand(M, K, dtype=torch.bfloat16, device=gpu) - 0.5
matB = torch.rand(N, K, dtype=torch.bfloat16, device=gpu) - 0.5
matC = torch.zeros(N, M, dtype=torch.bfloat16, device=gpu)

dae = Launcher(num_sms, device=gpu)

loadA = TmaTensor(dae, matA).wgmma_load(TileM, TileK, Major.K)
loadB = TmaTensor(dae, matB).wgmma_load(N, TileK, Major.K)
storeC = TmaTensor(dae, matC).wgmma_store(N, TileM, Major.MN)


def sm_task(sm: int):
    m = sm * TileM
    return [
        Atom(K // TileK),
        RepeatM.on(
            K // TileK,
            (loadB.cord(0, 0), loadB.cord2tma(0, TileK)),
            (loadA.cord(m, 0), loadA.cord2tma(0, TileK)),
        ),
        storeC.cord(0, m).bar(),
    ]


dae.i(
    sm_task,
    TerminateC(),
    TerminateM(),
)

ref = matA @ matB.t()

dae_app(dae)

tensor_diff("GEMV MMA M64N8", ref, matC.t())
