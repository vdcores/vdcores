import torch
import copy
from math import sqrt
from dae.launcher import *
from dae.util import *
from dae.runtime import opcode, build_tma_desc 

gpu = torch.device("cuda")

Q_SEQ_LEN, KV_SEQ_LEN = 64, 64
HEAD_DIM = 128
NUM_REQ, NUM_HEAD = 1, 1

QTile = 64
KVTile = 64

num_sms = NUM_HEAD * NUM_REQ
assert num_sms <= 1 # max sm count for HX00

dae = Launcher(num_sms, device=gpu)

matQ = torch.rand(NUM_REQ, NUM_HEAD, Q_SEQ_LEN, HEAD_DIM, dtype=torch.float16, device=gpu) - 0.5
matK = torch.rand(NUM_REQ, NUM_HEAD, KV_SEQ_LEN, HEAD_DIM, dtype=torch.float16, device=gpu) - 0.5
matV = torch.rand(NUM_REQ, NUM_HEAD, KV_SEQ_LEN, HEAD_DIM, dtype=torch.float16, device=gpu) - 0.5
matO = torch.zeros(NUM_REQ, NUM_HEAD, Q_SEQ_LEN, HEAD_DIM, dtype=torch.float16, device=gpu)

tQ = TmaTensor(dae, matQ).wgmma_load(QTile, HEAD_DIM, Major.K)
tK = TmaTensor(dae, matK).wgmma_load(KVTile, HEAD_DIM, Major.K)
tV = TmaTensor(dae, matV).wgmma_load(KVTile, HEAD_DIM, Major.MN)
tO = TmaTensor(dae, matO).wgmma_store(QTile, HEAD_DIM, Major.K)

# debug tensors
matR = torch.zeros(NUM_REQ, NUM_HEAD, KV_SEQ_LEN, Q_SEQ_LEN, dtype=torch.float16, device=gpu)
tR = TmaTensor(dae, matR).wgmma_store(KVTile, QTile, Major.MN)

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
                # [tR.cord(req, head, 0, q), tR.cord2tma(0, 0, KVTile, 0)],
                [tV.cord(req, head, 0, 0), tV.cord2tma(0, 0, KVTile, 0)],
            ),
            tO.cord(req, head, q, 0),
            ]
    return insts

dae.i(
    sm_task,

    TerminateC(),
    WriteBarrier(),
    TerminateM(),
)

print("Launching Attention DAE...")

dae.launch()

def attention(req, head):
    Q = matQ[req, head]
    K = matK[req, head]
    S = Q @ K.T # [SEQ_LEN, SEQ_LEN]
    RM = torch.max(S, dim=-1, keepdim=True).values # row max
    P_online = torch.exp(S - RM) # online softmax
    P = torch.softmax(S, dim=-1)
    V = matV[req, head]
    O = P @ V
    O_before_scale = P_online @ V
    return S, RM, P_online, P, O, O_before_scale

def attention_online(req, head):
    Q = matQ[req, head]
    K = matK[req, head]
    V = matV[req, head]
    O = torch.zeros(Q_SEQ_LEN, HEAD_DIM, dtype=torch.float16, device=gpu)
    row_max_tmp = None
    row_sum_tmp = None
    for qi in range(Q_SEQ_LEN // QTile):
        Q_block = Q[qi * QTile: (qi + 1) * QTile] # [QTile, HEAD_DIM]
        O_block = torch.zeros(QTile, HEAD_DIM, dtype=torch.float16, device=gpu)
        for ki in range(KV_SEQ_LEN // KVTile):
            K_block = K[ki * KVTile: (ki + 1) * KVTile] # [KVTile, HEAD_DIM]
            V_block = V[ki * KVTile: (ki + 1) * KVTile] # [KVTile, HEAD_DIM]
            S_block = Q_block.float() @ K_block.T.float() # [QTile, KVTile] f32 as accumulate type
            
            if ki == 0:
                row_max = torch.max(S_block, dim=-1, keepdim=True).values
                exp_S = torch.exp(S_block - row_max)
                row_sum = torch.sum(exp_S, dim=-1, keepdim=True)
                row_max_tmp = row_max
                row_sum_tmp = row_sum
            else:
                # new_row_max should be max(old_row_max, S_block)
                new_row_max = torch.max(torch.cat([row_max, torch.max(S_block, dim=-1, keepdim=True).values], dim=-1), dim=-1, keepdim=True).values
                score_scaler = torch.exp(row_max - new_row_max)
                exp_S = torch.exp(S_block - new_row_max)
                row_sum = row_sum * score_scaler + torch.sum(exp_S, dim=-1, keepdim=True)
                row_max = new_row_max
                O_block = O_block * score_scaler
            O_block += exp_S @ V_block.float()
        O[qi * QTile: (qi + 1) * QTile] = O_block / row_sum
    return O, row_sum, S_block, row_max, O_block, row_max_tmp, exp_S, row_sum_tmp

refS, refRM, refP_online, refP, refO, refO_before_scale = attention(0,0)
refonline, row_sum, S_block, row_max, online_O_before_scale, ref_first_rowmax, exp_S, ref_row_sum_tmp = attention_online(0,0)

print("Reference O:", refO)
print("RefOnline O:", refonline)
print("DAE O:", matO[0,0])

tensor_diff("DAE and RefOnline", refonline, matO[0,0])

# print("Ref before scale O:", online_O_before_scale)
# last_dae_block = matO[0, 0, (SEQ_LEN - QTile): SEQ_LEN]
# print("Result before scale O:", last_dae_block)
# print(f"Ave Diff before scale: {((online_O_before_scale - last_dae_block).abs().mean() / online_O_before_scale.abs().mean()).item() * 100} %. ")

# daeQK = matR[0,0].T
# print("Ref QK block:", refS.shape, daeQK.shape)
# tensor_diff("DAE QK and ref QK", refS, daeQK)

# row_max = row_max.squeeze()
# print("Reference ROW MAX:", row_max)
# print("DAE ROW MAX:", matTmpM[0])
# print(f"Ave Diff ROW MAX: {((row_max - matTmpM[0]).abs().mean() / row_max.abs().mean()).item() * 100:.2f} %. ")

# old_rowmax = ref_first_rowmax.squeeze()
# print("Reference first block ROW MAX:", old_rowmax)
# print("qk tile:", S_block[1])
# print("dae qk tile:", daeQK[1])

# tmp_Ponline = matTmpPOnline.T
# print("Reference exp P online first row: ", exp_S[0])
# print("Result P online:   ", tmp_Ponline[0])
# print(f"Ave Diff: {((exp_S - tmp_Ponline).abs().mean() / exp_S.abs().mean()).item() * 100} %. ")
# print("row sum first row: ", exp_S[0].sum())

# row_sum = row_sum.squeeze()
# print("Ref row sum:", row_sum)
# print("DAE Result row sum:", matTmpS[0])
# print(f"Ave Diff row sum: {((row_sum - matTmpS[0]).abs().mean() / row_sum.abs().mean()).item() * 100:.2f} %. ")


# print("Ref O first row:", refO[7])
# print("DAE O first row:", matO[0,0,7])
