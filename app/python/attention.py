import torch
import copy
from math import sqrt
from dae.launcher import *
from dae.util import *

gpu = torch.device("cuda")

# Q_SEQ_LEN, KV_SEQ_LEN = 1024, 2048
Q_SEQ_LEN, KV_SEQ_LEN = 64, 64
HEAD_DIM = 128
NUM_REQ, NUM_HEAD = 8, 16
# NUM_REQ, NUM_HEAD = 1, 32

QTile = 64
KVTile = 64

num_sms = NUM_HEAD * NUM_REQ
assert num_sms <= 132 # max sm count for HX00

dae = Launcher(num_sms, device=gpu)

matQ = torch.rand(NUM_REQ, NUM_HEAD, Q_SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16, device=gpu) - 0.5
matK = torch.rand(NUM_REQ, NUM_HEAD, KV_SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16, device=gpu) - 0.5
matV = torch.rand(NUM_REQ, NUM_HEAD, KV_SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16, device=gpu) - 0.5
matO = torch.zeros(NUM_REQ, NUM_HEAD, Q_SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16, device=gpu)

tQ = TmaTensor(dae, matQ).wgmma_load(QTile, HEAD_DIM, Major.K)
tK = TmaTensor(dae, matK).wgmma_load(KVTile, HEAD_DIM, Major.K)
tV = TmaTensor(dae, matV).wgmma_load(KVTile, HEAD_DIM, Major.MN)
tO = TmaTensor(dae, matO).wgmma_store(QTile, HEAD_DIM, Major.K)

def sm_task(sm: int):
    head = sm % NUM_HEAD
    req = sm // NUM_HEAD

    insts = []
    for q in range(0, Q_SEQ_LEN, QTile):
        insts += [
            ATTENTION_M64N64K16_F16_F32_64_64_hdim(KV_SEQ_LEN),
            tQ.cord(req, head, q, 0),
            RepeatM.on(KV_SEQ_LEN // KVTile,
                [tK.cord(req, head, 0, 0), tK.cord2tma(0, 0, KVTile, 0)],
                [tV.cord(req, head, 0, 0), tV.cord2tma(0, 0, KVTile, 0)],
            ),
            tO.cord(req, head, q, 0),
        ]
    return insts

dae.i(
    sm_task,

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

tensor_diff("DAE", refO, O)

total_bytes = (matQ.nbytes + matO.nbytes) + (matK.nbytes + matV.nbytes) * (Q_SEQ_LEN // QTile)
print(f"Total bytes processed: {total_bytes / (1024**2)} MB")
dae.bench(total_bytes = total_bytes)
