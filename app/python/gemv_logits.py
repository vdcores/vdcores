import torch
from dae.launcher import *
from dae.util import *

# a 9 epoch schedule
nlogits = 151936
epoch = 9
num_sms = 132

gpu = torch.device("cuda")
Atom = Gemv_M128N8

TileM, N, TileK = Atom.MNK
M, K = epoch * num_sms * TileM, 4096

print(f"nlogits: {nlogits}, epoch: {epoch}, M: {M}, N: {N}, K: {K}")

m_per_epoch = num_sms * TileM

assert num_sms <= 132 # max sm count for HX00

matA = torch.rand(epoch, m_per_epoch, K, dtype=torch.float16, device=gpu) - 0.5
matB = torch.rand(N, K, dtype=torch.float16, device=gpu) - 0.5
matC = torch.zeros(epoch, N, m_per_epoch, dtype=torch.float16, device=gpu)

dae = Launcher(num_sms, device=gpu)

n_batch = 4

loadA = TmaTensor(dae, matA).wgmma_load(TileM, TileK, Major.K)
loadB = TmaTensor(dae, matB).wgmma_load(N, TileK * n_batch, Major.K)
storeC = TmaTensor(dae, matC).wgmma_store(N, TileM, Major.MN)

def mk_sm_task(e: int):
    def sm_task(sm: int):
        m = sm * TileM 
        return [
            Atom(K // TileK),
            RepeatM.on(K // TileK // n_batch,
                (loadB.cord(0, 0), loadB.cord2tma(0, n_batch * TileK)),
                (loadA.cord(e, m, TileK * 0), loadA.cord2tma(0, 0, n_batch * TileK)),
                (loadA.cord(e, m, TileK * 1), loadA.cord2tma(0, 0, n_batch * TileK)),
                (loadA.cord(e, m, TileK * 2), loadA.cord2tma(0, 0, n_batch * TileK)),
                (loadA.cord(e, m, TileK * 3), loadA.cord2tma(0, 0, n_batch * TileK)),
            ),
            storeC.cord(e, 0, m),
        ]
    return sm_task
    
dae.i(
    [ mk_sm_task(e) for e in range(epoch) ],

    TerminateC(),
    TerminateM(),
)

# ref = matA.view(M, K) @ matB.t()
# res = matC.reshape(M, N)

print(f"GEMV M128N8 on [M={M} x N={N} x K={K}]")
print(f"loadA size: {loadA.size // 1024} KB, loadB size: {loadB.size // 1024} KB, storeC size: {storeC.size // 1024} KB")
# tensor_diff("GEMV M64N16", ref, res)

dae.bench(1)

print("inst size:", dae.num_insts())
print("theory load speed:", (matA.nbytes + matB.nbytes + matC.nbytes) / 1024 ** 3 / 3700 * 1e6, "us")
print("theory load speed (no L2):", (matA.nbytes + matB.nbytes * num_sms * epoch + matC.nbytes) / 1024 ** 3 / 3700 * 1e6, "us")
# print("stall instructions", dae.profile[:,3].cpu().numpy().mean())
