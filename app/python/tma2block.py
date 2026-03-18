import torch
from dae.launcher import *
from dae.util import *

gpu = torch.device("cuda")

Atom = Gemv_M128N16
TileM, N, TileK = Atom.MNK
M, K = 12288, 4096

print("Matrix size:", M, "x", K, "with tile", TileM, "x", TileK)

num_sms = 96

matIn = torch.rand(M, K, dtype=torch.float16, device=gpu) - 0.5
# This matrix convert tma_nd to tma_1d
blockOut = torch.zeros(M // TileM, K // TileK, TileM * TileK, dtype=torch.float16, device=gpu)

dae = Launcher(num_sms, device=gpu)

load = TmaTensor(dae, matIn).wgmma_load(TileM, TileK, Major.K)

def sm_task(sm: int):
    m = TileM * sm
    store = TmaStore1D(blockOut[sm, 0, ...])
    return [
        Dummy(K // TileK * 2),
        RepeatM.on(K // TileK,
            (load.cord(m, 0), load.cord2tma(0, TileK)),
            (store, store.size)
        ),
    ]

dae.i(
    sm_task,
    # GlobalBarrier(num_sms, 0),

    TerminateC(),
    TerminateM(),
)


dae_app(dae, total_bytes = matIn.nbytes)

print("matIn:", matIn.nbytes)
print("blockOut:", blockOut.nbytes)