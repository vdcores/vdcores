import torch
import copy
from math import sqrt
from functools import partial
from dae.launcher import *
from dae.util import *
from dae.runtime import opcode, build_tma_desc 
from qwen3.utils import *

gpu = torch.device("cuda")
torch.manual_seed(0)

active_new_len = 8 # only first 8 tokens are real
Q_SEQ_LEN, KV_SEQ_LEN = 16, 64
HEAD_DIM = 128
HIDDEN_SIZE = 4096
NUM_REQ = 1
NUM_Q_HEAD = 32
NUM_KV_HEAD = 8
HEAD_GROUP_SIZE = NUM_Q_HEAD // NUM_KV_HEAD

QTile = 16
KVTile = 64

num_sms = NUM_KV_HEAD * NUM_REQ
# num_sms = 1
assert num_sms <= 132 # max sm count for HX00

dae = Launcher(num_sms, device=gpu)

matQ = torch.rand(NUM_REQ * Q_SEQ_LEN, NUM_Q_HEAD * HEAD_DIM, dtype=torch.bfloat16, device=gpu) - 0.5
matK = torch.rand(NUM_REQ * KV_SEQ_LEN, NUM_KV_HEAD * HEAD_DIM, dtype=torch.bfloat16, device=gpu) - 0.5
matV = torch.rand(NUM_REQ * KV_SEQ_LEN, NUM_KV_HEAD * HEAD_DIM, dtype=torch.bfloat16, device=gpu) - 0.5
matO = torch.zeros(NUM_REQ * Q_SEQ_LEN, HIDDEN_SIZE, dtype=torch.bfloat16, device=gpu)

# interleaved QKV
matQ_attn_view = matQ.view(NUM_REQ, Q_SEQ_LEN * HEAD_GROUP_SIZE, NUM_KV_HEAD, HEAD_DIM)
matK_attn_view = matK.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)
matV_attn_view = matV.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)
matO_attn_view = matO.view(NUM_REQ, Q_SEQ_LEN * HEAD_GROUP_SIZE, NUM_KV_HEAD, HEAD_DIM)

tma_builder_K_inter = partial(build_tma_wgmma_k, iN = -3, swizzle=0)
tma_builder_K = partial(build_tma_wgmma_k, iN = -3)
cord_func_K = partial(cord_func_K_major, iN=-3)

tma_builder_MN_inter = partial(build_tma_wgmma_mn, iK = -3, swizzle=0)
tma_builder_MN = partial(build_tma_wgmma_mn, iK = -3)
cord_func_MN = partial(cord_func_MN_major, iK=-3)

tQ = TmaTensor(dae, matQ_attn_view)._build("load", HEAD_DIM, HEAD_GROUP_SIZE * QTile, tma_builder_K, cord_func_K)
tK = TmaTensor(dae, matK_attn_view)._build("load", HEAD_DIM, KVTile, tma_builder_K, cord_func_K)
tV = TmaTensor(dae, matV_attn_view)._build("load", HEAD_DIM, KVTile, tma_builder_MN, cord_func_MN)
tO = TmaTensor(dae, matO_attn_view)._build("store", HEAD_DIM, HEAD_GROUP_SIZE * QTile, tma_builder_K, cord_func_K)

# debug tensors
matR = torch.zeros(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, Q_SEQ_LEN * HEAD_GROUP_SIZE, dtype=torch.float16, device=gpu)
tR = TmaTensor(dae, matR)._build("store", QTile * HEAD_GROUP_SIZE, KVTile, tma_builder_MN, cord_func_MN)

matRP = torch.zeros(NUM_REQ, Q_SEQ_LEN * HEAD_GROUP_SIZE, NUM_KV_HEAD, KV_SEQ_LEN, dtype=torch.float16, device=gpu)
tRP = TmaTensor(dae, matRP)._build("store", QTile * HEAD_GROUP_SIZE, KVTile, tma_builder_K, cord_func_K)

matRPV = torch.zeros(NUM_REQ, Q_SEQ_LEN * HEAD_GROUP_SIZE, NUM_KV_HEAD, HEAD_DIM, dtype=torch.float16, device=gpu)
tRPV = TmaTensor(dae, matRPV)._build("store", HEAD_DIM, HEAD_GROUP_SIZE * QTile, tma_builder_K, cord_func_K)

need_norm = False
need_rope = False

rope_table = torch.ones(1024, HEAD_DIM, dtype=torch.bfloat16, device=gpu) - 0.5

def sm_task(sm: int):
    head = sm % NUM_KV_HEAD
    req = sm // NUM_KV_HEAD

    insts = []
    for q in range(0, active_new_len, QTile):
        insts += [
            ATTENTION_M64N64K16_F16_F32_64_64_hdim(min(active_new_len-q, QTile), hist_len=q, need_norm=need_norm, need_rope=need_rope),
            tQ.cord(req, q * HEAD_GROUP_SIZE, head, 0),
            RawAddress(rope_table, 24) if need_rope else [],
            RepeatM.on((active_new_len + KVTile - 1) // KVTile,
                [tK.cord(req, 0, head, 0), tK.cord2tma(0, KVTile, 0, 0)],
                [tV.cord(req, 0, head, 0), tV.cord2tma(0, KVTile, 0, 0)],
            ),
            tO.cord(req, q * HEAD_GROUP_SIZE, head, 0),
        ]
    return insts

dae.i(
    sm_task,

    TerminateC(),
    TerminateM(),
)

# print("Launching Attention DAE...")

dae.launch()

def attention():
    # ---- 1. QK norm and rope ------------------
    fQ = matQ.view(NUM_REQ * Q_SEQ_LEN * NUM_Q_HEAD, HEAD_DIM)
    fK = matK.view(NUM_REQ * KV_SEQ_LEN * NUM_KV_HEAD, HEAD_DIM)
    if need_norm:
        # only normalize the first 8 tokens, the rest are padding tokens
        normQ = fQ[:active_new_len * NUM_Q_HEAD].pow(2).mean(dim=-1, keepdim=True)
        normQ = fQ[:active_new_len * NUM_Q_HEAD] * torch.rsqrt(normQ + 1.0)

        normK = fK[:active_new_len * NUM_KV_HEAD].pow(2).mean(dim=-1, keepdim=True)
        normK = fK[:active_new_len * NUM_KV_HEAD] * torch.rsqrt(normK + 1.0)
    else:
        normQ = fQ[:active_new_len * NUM_Q_HEAD]
        normK = fK[:active_new_len * NUM_KV_HEAD]
    
    if need_rope:
        # rope_table: [1024, HEAD_DIM], interleaved as [cos_0, sin_0, cos_1, sin_1, ...]
        # rotation: (x[2i], x[2i+1]) -> (x*cos - y*sin, x*sin + y*cos)

        # Q: [NUM_REQ * Q_SEQ_LEN * NUM_Q_HEAD, HEAD_DIM]
        normQ_4d = normQ.view(NUM_REQ, active_new_len, NUM_Q_HEAD, HEAD_DIM).float()
        q_even = normQ_4d[..., 0::2]                                        # [B, Q, H, D/2]
        q_odd  = normQ_4d[..., 1::2]
        cos_q = rope_table[:active_new_len, 0::2].float()[None, :, None, :]     # [1, Q, 1, D/2]
        sin_q = rope_table[:active_new_len, 1::2].float()[None, :, None, :]
        normQ = torch.stack([q_even * cos_q - q_odd * sin_q,
                             q_even * sin_q + q_odd * cos_q], dim=-1) \
                     .flatten(-2).to(matQ.dtype) \
                     .view(NUM_REQ * active_new_len * NUM_Q_HEAD, HEAD_DIM)

        # K: [NUM_REQ * KV_SEQ_LEN * NUM_KV_HEAD, HEAD_DIM]
        normK_4d = normK.view(NUM_REQ, active_new_len, NUM_KV_HEAD, HEAD_DIM).float()
        k_even = normK_4d[..., 0::2]                                        # [B, KV, H, D/2]
        k_odd  = normK_4d[..., 1::2]
        cos_k = rope_table[:active_new_len, 0::2].float()[None, :, None, :]    # [1, KV, 1, D/2]
        sin_k = rope_table[:active_new_len, 1::2].float()[None, :, None, :]
        normK = torch.stack([k_even * cos_k - k_odd * sin_k,
                             k_even * sin_k + k_odd * cos_k], dim=-1) \
                     .flatten(-2).to(matK.dtype) \
                     .view(NUM_REQ * active_new_len * NUM_KV_HEAD, HEAD_DIM)
    
    fQ[:active_new_len * NUM_Q_HEAD] = normQ
    fK[:active_new_len * NUM_KV_HEAD] = normK

    # ---- 2. Slice Q, K, V -------------------------------------------------

    Q = matQ.view(NUM_REQ, Q_SEQ_LEN, HEAD_GROUP_SIZE, NUM_KV_HEAD, HEAD_DIM)  # (B, T, G, KV, D)
    K = matK.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)  # (B, T, KV, D)
    V = matV.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)   # (B, T, KV, D)

    # ---- 3. Reorder for attention ----------------------------------------

    Q = Q.permute(0, 3, 2, 1, 4)            # (B, KV, GROUP, T, D)
    K = K.permute(0, 2, 1, 3).unsqueeze(2) # (B, KV, 1, T, D)
    V = V.permute(0, 2, 1, 3).unsqueeze(2)

    # ---- 4. Scaled dot-product attention ---------------------------------

    S = torch.matmul(Q.float(), K.float().transpose(-1, -2)) # [B, KV, GROUP, Qlen, KVlen] f32 as accumulate type
    P = torch.softmax(S, dim=-1)
    O = torch.matmul(P, V.float()) # [B, KV, GROUP, Qlen, D]

    # ---- 5. Restore to (B, T, Q_HEAD * HEAD_DIM) ----------------------------------

    O = O.permute(0, 3, 2, 1, 4) \
         .reshape(NUM_REQ * Q_SEQ_LEN, NUM_Q_HEAD * HEAD_DIM)

    return S, P, O


def attention_online(req, head):
    Q = matQ_attn_view[req, :, head] # [Q_SEQ_LEN * HEAD_GROUP_SIZE, HEAD_DIM]
    K = matK_attn_view[req, :, head] # [KV_SEQ_LEN, HEAD_DIM]
    V = matV_attn_view[req, :, head] # [KV_SEQ_LEN, HEAD_DIM]
    O = torch.zeros(Q_SEQ_LEN * HEAD_GROUP_SIZE, HEAD_DIM, dtype=torch.float16, device=gpu)

    qk = torch.zeros(Q_SEQ_LEN * HEAD_GROUP_SIZE, KV_SEQ_LEN, dtype=torch.float32, device=gpu)
    p = torch.zeros(Q_SEQ_LEN * HEAD_GROUP_SIZE, KV_SEQ_LEN, dtype=torch.float32, device=gpu)
    row_maxs = torch.zeros(Q_SEQ_LEN * HEAD_GROUP_SIZE, dtype=torch.float32, device=gpu)
    pv = torch.zeros(Q_SEQ_LEN * HEAD_GROUP_SIZE, HEAD_DIM, dtype=torch.float32, device=gpu)
    row_sums = torch.zeros(Q_SEQ_LEN * HEAD_GROUP_SIZE, dtype=torch.float32, device=gpu)

    for qi in range(Q_SEQ_LEN // QTile):
        Q_block = Q[qi * QTile * HEAD_GROUP_SIZE: (qi + 1) * QTile * HEAD_GROUP_SIZE] # [QTile * HEAD_GROUP_SIZE, HEAD_DIM]
        # print("Q_block shape: ", Q_block.shape)
        # for r in range(4):
        #     for c in range(128):
        #         print(f"Q_block[{r},{c}]={Q_block[r,c].item()}")

        O_block = torch.zeros(QTile * HEAD_GROUP_SIZE, HEAD_DIM, dtype=torch.float16, device=gpu)
        for ki in range(KV_SEQ_LEN // KVTile):
            K_block = K[ki * KVTile: (ki + 1) * KVTile] # [KVTile, HEAD_DIM]
            V_block = V[ki * KVTile: (ki + 1) * KVTile] # [KVTile, HEAD_DIM]
            S_block = Q_block.float() @ K_block.T.float() # [QTile * HEAD_GROUP_SIZE, KVTile] f32 as accumulate type
            qk[qi * QTile * HEAD_GROUP_SIZE: (qi + 1) * QTile * HEAD_GROUP_SIZE, ki * KVTile: (ki + 1) * KVTile] = S_block
            # mask S_block column after active kv len to -inf
            for c in range(KVTile):
                if False and ki * KVTile + c >= 16:
                    S_block[:, c] = -float('inf')
            
            if ki == 0:
                row_max = torch.max(S_block, dim=-1, keepdim=True).values
                row_maxs[qi * QTile * HEAD_GROUP_SIZE: (qi + 1) * QTile * HEAD_GROUP_SIZE] = row_max.squeeze()
                exp_S = torch.exp(S_block - row_max)
                row_sum = torch.sum(exp_S, dim=-1, keepdim=True)
            else:
                # new_row_max should be max(old_row_max, S_block)
                new_row_max = torch.max(torch.cat([row_max, torch.max(S_block, dim=-1, keepdim=True).values], dim=-1), dim=-1, keepdim=True).values
                score_scaler = torch.exp(row_max - new_row_max)
                exp_S = torch.exp(S_block - new_row_max)
                row_sum = row_sum * score_scaler + torch.sum(exp_S, dim=-1, keepdim=True)
                row_max = new_row_max
                O_block = O_block * score_scaler
            p[qi * QTile * HEAD_GROUP_SIZE: (qi + 1) * QTile * HEAD_GROUP_SIZE, ki * KVTile: (ki + 1) * KVTile] = exp_S
            pv_block = exp_S @ V_block.float() # [QTile * HEAD_GROUP_SIZE, HEAD_DIM]
            pv[qi * QTile * HEAD_GROUP_SIZE: (qi + 1) * QTile * HEAD_GROUP_SIZE] = pv_block
            O_block += pv_block.float()
        O[qi * QTile * HEAD_GROUP_SIZE: (qi + 1) * QTile * HEAD_GROUP_SIZE] = O_block / row_sum
        row_sums[qi * QTile * HEAD_GROUP_SIZE: (qi + 1) * QTile * HEAD_GROUP_SIZE] = row_sum.squeeze()
    return O, O_block, qk, p, row_maxs, pv, row_sums

refS, refP, refO = attention()
# refonline, online_O_before_scale, qk_online, p_online, rowmax_s, pv_online, row_sums = attention_online(0,0)

refO_detail = refO.view(NUM_REQ, Q_SEQ_LEN, HEAD_GROUP_SIZE, NUM_KV_HEAD, HEAD_DIM)
# tensor_diff("Ref and RefOnline", refO_detail[0,:,:,0], refonline.view(Q_SEQ_LEN, HEAD_GROUP_SIZE, HEAD_DIM))
# tensor_diff("Ref and DAE", refO_detail, matO_attn_view.view(NUM_REQ, Q_SEQ_LEN, HEAD_GROUP_SIZE, NUM_KV_HEAD, HEAD_DIM))
# print("ref rope token[0][:128]")
# print(rope_table[15,:128])

tensor_diff("Ref and DAE", refO_detail, matO_attn_view.view(NUM_REQ, Q_SEQ_LEN, HEAD_GROUP_SIZE, NUM_KV_HEAD, HEAD_DIM))

dae_app(dae)

# daeQK = matR[0, :, 0].view(KV_SEQ_LEN, Q_SEQ_LEN, HEAD_GROUP_SIZE)
# ref_online_s = qk_online.T.view(KV_SEQ_LEN, Q_SEQ_LEN, HEAD_GROUP_SIZE)
# tensor_diff("DAE QK and ref QK", ref_online_s, daeQK)

# print("Ref qk row 9th:", qk_online[8])
# print("Ref thread 0 max:", rowmax_s[[0, 8, 64, 72, 128, 136, 192, 200]])

# daeP = matRP[0, :, 0]
# ref_online_p = p_online
# tensor_diff("DAE exp and Ref exp", ref_online_p, daeP)
# print("Ref first exp P row:", ref_online_p[0])
# print("DAE first exp P row:", daeP[0])

# dae_pv = matRPV[0, :, 0]
# ref_online_pv = pv_online
# tensor_diff("DAE pv and Ref pv", ref_online_pv, dae_pv)
# print("Ref first pv row:", ref_online_pv[0])
# print("DAE first pv row:", dae_pv[0])

# print("Ref thread 0 row sum: ", row_sums[[0, 8, 64, 72, 128, 136, 192, 200]])

# print("Ref O first row first head: ", refonline[0])
# print("DAE O first row first head: ", matO_attn_view[0,0,0])
