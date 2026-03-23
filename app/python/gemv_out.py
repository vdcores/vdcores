import os

import torch
from dae.launcher import *
from dae.tma_utils import StaticCordAdapter
from dae.util import *
from dae.schedule import *
from dae.model import *

gpu = torch.device("cuda")
torch.manual_seed(0)

use_scale = os.getenv("GEMV_SCALE", "0") == "1"
Atom = Gemv_M64N8_MMA_SCALE if use_scale else Gemv_M64N8

TileM, N, TileK = Atom.MNK
M = int(os.getenv("GEMV_M", "4096"))
K = int(os.getenv("GEMV_K", "4096"))
num_sms = int(os.getenv("GEMV_SMS", "128"))

assert M % TileM == 0, f"M={M} must be a multiple of {TileM}"
assert K % TileK == 0, f"K={K} must be a multiple of {TileK}"

matA = torch.rand(M, K, dtype=torch.bfloat16, device=gpu) - 0.5
matB = torch.rand(N, K, dtype=torch.bfloat16, device=gpu) - 0.5
matC = torch.zeros(N, M, dtype=torch.bfloat16, device=gpu)
matScale = torch.rand(32, dtype=torch.bfloat16, device=gpu) if use_scale else None

dae = Launcher(num_sms, device=gpu)

if use_scale:
    loadA = TmaTensor(dae, matA).wgmma_load(TileM, TileK, Major.K)
    loadB = TmaTensor(dae, matB).wgmma_load(N, TileK, Major.K)
    storeC = TmaTensor(dae, matC).wgmma_store(N, TileM, Major.MN)
    loadScale = StaticCordAdapter(TmaLoad1D(matScale, bytes=64))
    layer = SchedGemvMmaScale(
        Atom,
        (M, N, K),
        (loadA, loadB, loadScale, storeC),
    ).place(num_sms)
    ref = matA @ (matB * matScale[:N, None]).t()
else:
    layer = GemvLayer(dae, Atom, "out_proj", (matA, matB, matC))
    ref = matA @ matB.t()

dae.s(layer if use_scale else layer.schedule().place(num_sms))

dae_app(dae)

if use_scale:
    tensor_diff("GEMV MMA SCALE M64N8", ref, matC.t())
else:
    layer.diff()
