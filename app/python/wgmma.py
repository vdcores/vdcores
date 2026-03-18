import torch
from dae.launcher import *

gpu = torch.device("cuda")
MMAKernel = WGMMA_64x256x64_F16

# M, N, K = 8192, 2048, 4096
M, N, K = 4096, 64, 4096
TileM, TileN, TileK = MMAKernel.MNK

assert M % TileM == 0
num_sms = M // TileM
assert num_sms <= 132 # max sm count for HX00

matA = torch.rand(K, M, dtype=torch.float16, device=gpu) - 0.5
matB = torch.rand(K, N, dtype=torch.float16, device=gpu) - 0.5
matC = torch.zeros(N, M, dtype=torch.float16, device=gpu)

dae = Launcher(num_sms, device=gpu)

loadA = TmaTensor(dae, matA).wgmma_load(TileK, TileM, Major.MN)
loadB = TmaTensor(dae, matB).wgmma_load(TileK, TileN, Major.MN)
storeC = TmaTensor(dae, matC).wgmma_store(TileN, TileM, Major.MN)

def sm_task(sm: int):
    m = sm * TileM
    insts = []
    for n in range(0, N, TileN):
        insts += [
            RepeatM.on(K // TileK,
                (loadA.cord(0, m), loadA.cord2tma(TileK, 0)),
                (loadB.cord(0, n), loadB.cord2tma(TileK, 0))
            ),
            storeC.cord(n, m)
        ]
    return insts
    
dae.i(
    sm_task,

    MMAKernel(N // TileN, K // TileK),
    TerminateC(),
    WriteBarrier(),
    TerminateM(),
)

dae.launch()

ref = matA.t() @ matB
res = matC.t()

print("Result Shape:", ref.shape)
print(f"Ave Diff: {((ref - res).abs().mean() / ref.abs().mean()).item() * 100} %. ")

sm_bytes = ((loadA.size + loadB.size) * K // TileK + storeC.size) * (N // TileN)
dae.bench(total_bytes = sm_bytes * num_sms)

print("total instrucions:", dae.profile[:,2].cpu().numpy().mean())
print("stall instructions", dae.profile[:,3].cpu().numpy().mean())
