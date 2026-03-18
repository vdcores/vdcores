import torch
import copy
from math import sqrt
from dae.launcher import *
from dae.util import *

gpu = torch.device("cuda")

Q_SEQ_LEN, KV_SEQ_LEN = 64, 64
HEAD_DIM = 128
NUM_REQ, NUM_HEAD = 1, 32

QTile = 64
KVTile = 64

attn_sms = NUM_HEAD * NUM_REQ

# FFN parameters
MMAKernel = GemmPrefetchA

M, N, K = 4096, 64, 4096
TileM, TileN, TileK = MMAKernel.MNK
ffn_sms = M // TileM

num_sms = attn_sms + ffn_sms
assert num_sms <= 132 # max sm count for HX00

dae = Launcher(num_sms, device=gpu)

matQ = torch.rand(NUM_REQ, NUM_HEAD, Q_SEQ_LEN, HEAD_DIM, dtype=torch.float16, device=gpu) - 0.5
matK = torch.rand(NUM_REQ, NUM_HEAD, KV_SEQ_LEN, HEAD_DIM, dtype=torch.float16, device=gpu) - 0.5
matV = torch.rand(NUM_REQ, NUM_HEAD, KV_SEQ_LEN, HEAD_DIM, dtype=torch.float16, device=gpu) - 0.5
matO = torch.zeros(NUM_REQ, NUM_HEAD, Q_SEQ_LEN, HEAD_DIM, dtype=torch.float16, device=gpu)

tQ = TmaTensor(dae, matQ).wgmma_load(QTile, HEAD_DIM, Major.K)
tK = TmaTensor(dae, matK).wgmma_load(KVTile, HEAD_DIM, Major.K)
tV = TmaTensor(dae, matV).wgmma_load(KVTile, HEAD_DIM, Major.MN)
tO = TmaTensor(dae, matO).wgmma_store(QTile, HEAD_DIM, Major.K)


matA = torch.rand(K, M, dtype=torch.float16, device=gpu) - 0.5
matB = torch.rand(K, N, dtype=torch.float16, device=gpu) - 0.5
matC = torch.zeros(N, M, dtype=torch.float16, device=gpu)

loadA = TmaTensor(dae, matA).wgmma_load(TileK, TileM, Major.MN)
loadB = TmaTensor(dae, matB).wgmma_load(TileK, TileN, Major.MN)
storeC = TmaTensor(dae, matC).wgmma_store(TileN, TileM, Major.MN)

# there could be multiple schedules, for example, we can partition the whole SMs
def task_attention(sm: int):
    head = sm % NUM_HEAD
    req = sm // NUM_HEAD

    insts = []
    for q in range(0, Q_SEQ_LEN, QTile):
        insts += [
            ATTENTION_M64N64K16_F16_F32_64_64_hdim(KV_SEQ_LEN),
            tQ.cord(req, head, q, 0),
            tO.cord(req, head, q, 0).bar().linked(),
            RepeatM.on(KV_SEQ_LEN // KVTile,
                [tK.cord(req, head, 0, 0), tK.cord2tma(0, 0, KVTile, 0)],
                [tV.cord(req, head, 0, 0), tV.cord2tma(0, 0, KVTile, 0)],
            ),
        ]
    insts += [ GlobalBarrier(attn_sms, tO) ]
    return insts

nPrefetch = 3
def task_ffn_prefetch(sm: int):
    m = sm * TileM
    n = 0 # GEMV kernel

    insts = []
    for k in range(nPrefetch):
        insts += [ loadA.cord(k * TileK, m) ]
    # sync with attention SMs when ready to load B
    insts += [ GlobalBarrier(attn_sms, tO) ]
    for k in range(K // TileK):
        if k >= nPrefetch:
            insts += [ loadA.cord(k * TileK, m) ]
        insts += [ loadB.cord(k * TileK, n) ]
    insts += [
        storeC.cord(n, m),
        MMAKernel(nPrefetch, K // TileK),
        WriteBarrier()
    ]
    return insts

def task_ffn_naive(sm: int):
    m = sm * TileM
    n = 0
    insts = [
        GlobalBarrier(attn_sms, tO),
        RepeatM.on(K // TileK,
            (loadA.cord(0, m), loadA.cord2tma(TileK, 0)),
            (loadB.cord(0, n), loadB.cord2tma(TileK, 0))
        ),
        storeC.cord(n, m),
        WGMMA_64x256x64_F16(1, K // TileK),
        WriteBarrier(),
    ]
    return insts

# TODO(zhiyuang): DSMEM schedule?

# simple scheduler just partition the SMs
def task(sm: int):
    if sm < attn_sms:
        return task_attention(sm)
    else:
        return task_ffn_prefetch(sm - attn_sms)
    
dae.i(
    task,

    TerminateC(),
    TerminateM(),
)

print("Launching Attention DAE...")

dae.launch()

def attention():
    S = matQ @ matK.transpose(-1, -2)
    P = torch.softmax(S, dim=-1)
    return P @ matV

refO = attention()
O = matO

refC = matA.t() @ matB
resC = matC.t()

tensor_diff("DAE", refO, O)
tensor_diff("FFN", refC, resC)

total_bytes = (matQ.nbytes + matO.nbytes) + (matK.nbytes + matV.nbytes) * (Q_SEQ_LEN // QTile)
print(f"Total bytes processed: {total_bytes / (1024**2)} MB")
dae.bench(total_bytes = total_bytes)
