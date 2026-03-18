import torch
from dae.launcher import *
from dae.util import *

gpu = torch.device("cuda")

Atom = Gemv_M128N16
TileM, N, TileK = Atom.MNK
M, K = 12288, 4096

num_sms = 96

matA = torch.rand(M, K, dtype=torch.float16, device=gpu) - 0.5
matB = torch.rand(N, K, dtype=torch.float16, device=gpu) - 0.5
matC = torch.zeros(N, M, dtype=torch.float16, device=gpu)

dae = Launcher(num_sms, device=gpu)

n_batch = 4
loadA = TmaTensor(dae, matA).wgmma_load(TileM, TileK, Major.K)
loadB = TmaTensor(dae, matB).wgmma_load(N, TileK * 4, Major.K)
storeC = TmaTensor(dae, matC).wgmma_store(N, TileM, Major.MN)

num_loads = K // TileK 
def sm_task(sm: int):
    m_total = M // num_sms
    assert m_total % TileM == 0
    m_start = sm * m_total
    insts = []
    for m in range(m_start, m_start + m_total, TileM):
        insts += [
            Dummy(num_loads),
            RepeatM.on(K // TileK // n_batch,
                # [loadB.cord(0, 0), loadB.cord2tma(0, n_batch * TileK)],
                *[
                    [loadA.cord(m, TileK * i), loadA.cord2tma(0, n_batch * TileK)]
                    for i in range(n_batch)
                ]
            ),
            # storeC.cord(0, m).bar(),
        ]
    return insts

dae.i(
    sm_task,

    TerminateC(),
    TerminateM(),
)

# dae.launch()
dae.bench(1, total_bytes = matA.nbytes)