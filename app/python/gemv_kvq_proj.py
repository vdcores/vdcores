import torch
from dae.launcher import *
from dae.util import *
from dae.schedule import SchedGemv

gpu = torch.device("cuda")

Atom = Gemv_M64N16

TileM, N, TileK = Atom.MNK
M, K = 6144, 4096
nstage = 2

assert K % TileK == 0
assert M % TileM == 0
num_sms = 128

matA = torch.rand(M, K, dtype=torch.bfloat16, device=gpu) - 0.5
matB = torch.rand(N, K, dtype=torch.bfloat16, device=gpu) - 0.5
matC = torch.zeros(N, M, dtype=torch.bfloat16, device=gpu)

dae = Launcher(num_sms, device=gpu)

n_batch = 4

loadA = TmaTensor(dae, matA).wgmma_load(TileM, TileK, Major.K)
loadB = TmaTensor(dae, matB).wgmma_load(N, TileK * n_batch, Major.K)
reduceC = TmaTensor(dae, matC).wgmma("reduce", N, TileM, Major.MN)

    
dae.i(
    # sm tasks for 
    SchedGemv(Atom, num_sms, (4096, N, K), (loadA, loadB, reduceC)),

    SchedGemv(Atom, num_sms, ((4096, 2048), N, K), (loadA, loadB, reduceC)),
    # GlobalBarrier(num_sms, reduceC.bar()),

    TerminateC(),
    TerminateM(),
)

print(f"GEMV M64N16 on [M={M} x N={N} x K={K}], stage={nstage}:")
print(f"loadA size: {loadA.size // 1024} KB, loadB size: {loadB.size // 1024} KB, storeC size: {reduceC.size // 1024} KB")

print("inst size:", dae.num_insts())
print("theory load speed:", (matA.nbytes + matB.nbytes + matC.nbytes) / 1024 ** 3 / 3700 * 1e6, "us")
print("theory load speed (no L2):", (matA.nbytes + matB.nbytes * num_sms + matC.nbytes) / 1024 ** 3 / 3700 * 1e6, "us")

ref = matA @ matB.t()
res = matC.t()

dae_app(dae)

tensor_diff("GEMV M64N16", ref, res)

