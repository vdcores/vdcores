import torch
from dae.launcher import *
from dae.util import *

gpu = torch.device("cuda")

Atom = Gemv_M64N16

TileM, N, TileK = Atom.MNK
M, K = 4096 * 3, 4096

assert K % TileK == 0
assert M % TileM == 0
num_sms = 128

matA = torch.rand(M, K, dtype=torch.float16, device=gpu) - 0.5
matB = torch.rand(N, K, dtype=torch.float16, device=gpu) - 0.5
matC = torch.zeros(N, M, dtype=torch.float16, device=gpu)

dae = Launcher(num_sms, device=gpu)

n_batch = 4

loadA = TmaTensor(dae, matA).wgmma_load(TileM, TileK, Major.K)
loadB = TmaTensor(dae, matB).wgmma_load(N, TileK * n_batch, Major.K)
storeC = TmaTensor(dae, matC).wgmma_store(N, TileM, Major.MN)
reduceC = TmaTensor(dae, matC).wgmma("reduce", N, TileM, Major.MN)


def sm_task(sm: int):
    m = sm * TileM
    # first, do 8192 x 4096
    insts =  [
        Atom(K // TileK),
        # original schedule
        [
            [
                loadB.cord(0, k),
                loadA.cord(m, k + TileK * 0),
                loadA.cord(m, k + TileK * 1),
                loadA.cord(m, k + TileK * 2),
                loadA.cord(m, k + TileK * 3),
            ]
            for k in range(0, K, TileK * n_batch)
        ],
        storeC.cord(0, m),
    ]
    # then, do a second 4096 x 4096
    m = 8192

    m_offset = sm // 2 * TileM
    k_offset = (sm % 2) * K // 2

    m = m + m_offset
    
    insts += [
        Atom(K // TileK // 2),
        # original schedule
        [
            [
                loadB.cord(0, k_offset + k),
                loadA.cord(m, k_offset + k + TileK * 0),
                loadA.cord(m, k_offset + k + TileK * 1),
                loadA.cord(m, k_offset + k + TileK * 2),
                loadA.cord(m, k_offset + k + TileK * 3),
            ]
            for k in range(0, K // 2, TileK * n_batch)
        ],
        reduceC.cord(0, m).bar(),
    ]

    return insts

    
dae.i(
    sm_task,
    GlobalBarrier(num_sms, reduceC),

    TerminateC(),
    TerminateM(),
)

print(f"GEMV M64N16 on [M={M} x N={N} x K={K}]:")
print(f"loadA size: {loadA.size // 1024} KB, loadB size: {loadB.size // 1024} KB, storeC size: {storeC.size // 1024} KB")

sm_bytes = (loadA.size * 4 + loadB.size) * K // TileK // 4 + storeC.size
dae.bench(1, total_bytes = sm_bytes * num_sms)

print("inst size:", dae.num_insts())
print("theory load speed:", (matA.nbytes + matB.nbytes + matC.nbytes) / 1024 ** 3 / 3700 * 1e6, "us")
print("theory load speed (no L2):", (matA.nbytes + matB.nbytes * num_sms * 2 + matC.nbytes) / 1024 ** 3 / 3700 * 1e6, "us")

ref = matA @ matB.t()
res = matC.t()

tensor_diff("GEMV M128N16", ref, res)