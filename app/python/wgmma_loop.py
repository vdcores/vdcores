import torch
from dae.launcher import *
from dae.util import *

gpu = torch.device("cuda")
MMAKernel = WGMMA_64x256x64_F16

# M, N, K = 64, 64, 256
M, N, K = 8192, 64, 4096
TileM, TileN, TileK = MMAKernel.MNK

assert M % TileM == 0
assert N % TileN == 0
assert K % TileK == 0

num_sms = M // TileM
assert num_sms <= 132 # max sm count for HX00

BATCH = 4

matA = torch.rand(BATCH, K, M, dtype=torch.float16, device=gpu) - 0.5
matB = torch.rand(BATCH, K, N, dtype=torch.float16, device=gpu) - 0.5
matC = torch.zeros(BATCH, N, M, dtype=torch.float16, device=gpu)

dae = Launcher(num_sms, device=gpu)

# push TMAs into DAE
loadAs = []
loadBs = []
storeCs = []
for b in range(BATCH):
    loadAs .append(TmaTensor(dae, matA[b]).wgmma_load(TileK, TileM, Major.MN))
    loadBs .append(TmaTensor(dae, matB[b]).wgmma_load(TileK, TileN, Major.MN))
    storeCs.append(TmaTensor(dae, matC[b]).wgmma_store(TileN, TileM, Major.MN))

def sm_task(sm: int):
    m = sm * TileM
    insts = []
    for b in range(BATCH):
        insts += [MMAKernel(N // TileN, K // TileK)],
        for n in range(0, N, TileN):
            insts += [
                RepeatM.on(K // TileK,
                    (loadAs[b].cord(0, m), loadAs[b].cord2tma(TileK, 0)),
                    (loadBs[b].cord(0, n), loadBs[b].cord2tma(TileK, 0))
                ),
                storeCs[b].cord(n, m)
            ]
    return insts

def loop_task(sm: int):
    m = sm * TileM
    insts = []
    for b in range(BATCH):
        insts += [MMAKernel(N // TileN, K // TileK)],

    minsts = []
    for n in range(0, N, TileN):
        minsts += [
            RepeatM.on(K // TileK,
                (loadAs[0].cord(0, m).group(), loadAs[0].cord2tma(TileK, 0)),
                (loadBs[0].cord(0, n).group(), loadBs[0].cord2tma(TileK, 0))
            ),
            storeCs[0].cord(n, m).group()
        ]
    insts += minsts
    # TODO(zhiyuang): compute this value
    insts += [LoopM(BATCH, 1, resource_group_shift=3, gpr_start=0, gpr_end=2)]
    return insts
    
dae.i(
    loop_task,
    # sm_task,

    TerminateC(),
    WriteBarrier(),
    TerminateM(),
)

dae.launch()

ref = matA.transpose(-2,-1) @ matB
res = matC.transpose(-2,-1)

print("Result Shape:", ref.shape)
tensor_diff("output", ref, res)

sm_bytes = ((loadAs[0].size + loadBs[0].size) * K // TileK + storeCs[0].size) * (N // TileN) * BATCH
dae.bench(total_bytes = sm_bytes * num_sms)

