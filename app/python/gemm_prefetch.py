import torch
from dae.launcher import *

gpu = torch.device("cuda")
MMAKernel = GemmPrefetchA

M, N, K = 4096, 64, 4096
TileM, TileN, TileK = MMAKernel.MNK

assert M % TileM == 0
assert N == TileN, "This kernel only supports N == TileN"
num_sms = M // TileM
assert num_sms <= 132 # max sm count for HX00

matA = torch.rand(K, M, dtype=torch.float16, device=gpu) - 0.5
matB = torch.rand(K, N, dtype=torch.float16, device=gpu) - 0.5
matC = torch.zeros(N, M, dtype=torch.float16, device=gpu)

dae = Launcher(num_sms, device=gpu)

loadA = TmaTensor(dae, matA).wgmma_load(TileK, TileM, Major.MN)
loadB = TmaTensor(dae, matB).wgmma_load(TileK, TileN, Major.MN)
storeC = TmaTensor(dae, matC).wgmma_store(TileN, TileM, Major.MN)

def build_task(n_prefetch: int):
    def sm_task(sm: int):
        m = sm * TileM
        n = 0

        insts = []
        # prefetch A tiles
        for i in range(n_prefetch):
            insts += [ loadA.cord(i * TileK, m) ]
        # stream other tiles with prefetching
        for i in range(K // TileK):
            if i >= n_prefetch:
                insts += [ loadA.cord(i * TileK, m)]
            insts += [loadB.cord(i * TileK, n)]
        insts += [
            storeC.cord(n, m),
            MMAKernel(n_prefetch, K // TileK)
        ]
        return insts
    return sm_task
    
#TODO(zhiyuang): stuck on prefetch > 4
dae.i(
    build_task(n_prefetch=4),

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
