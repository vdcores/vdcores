import torch
import copy
from math import sqrt
from dae.launcher import *
from dae.util import *
from dae.runtime import opcode, build_tma_desc 
from qwen3.utils import *
from dae.model import *

gpu = torch.device("cuda")
torch.manual_seed(0)

Q_SEQ_LEN, KV_SEQ_LEN = 16, 64
HEAD_DIM = 128
NUM_REQ = 1
NUM_Q_HEAD = 32
NUM_KV_HEAD = 8
HEAD_GROUP_SIZE = NUM_Q_HEAD // NUM_KV_HEAD

QTile = 16
KVTile = 64

num_sms = NUM_KV_HEAD * NUM_REQ
assert num_sms <= 132 # max sm count for HX00

dae = Launcher(num_sms, device=gpu)

matQ = torch.rand(NUM_REQ, Q_SEQ_LEN, NUM_Q_HEAD * HEAD_DIM, dtype=torch.bfloat16, device=gpu) - 0.5
matK = torch.rand(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD * HEAD_DIM, dtype=torch.bfloat16, device=gpu) - 0.5
matV = torch.rand(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD * HEAD_DIM, dtype=torch.bfloat16, device=gpu) - 0.5
matO = torch.zeros(NUM_REQ, Q_SEQ_LEN, NUM_Q_HEAD * HEAD_DIM, dtype=torch.bfloat16, device=gpu)

gqa = GQALayer(dae, "gqa", (matQ, matK, matV, matO))

dae.s(
    gqa.schedule().o_bar(0)
)

dae_app(dae)

# def attention():
#     # ---- 1. Unflatten the interleaved sequence dimension ------------------

#     # ---- 2. Slice Q, K, V -------------------------------------------------

#     Q = matQ.view(NUM_REQ, Q_SEQ_LEN, HEAD_GROUP_SIZE, NUM_KV_HEAD, HEAD_DIM)  # (B, T, G, KV, D)
#     K = matK.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)  # (B, T, KV, D)
#     V = matV.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)   # (B, T, KV, D)

#     # ---- 3. Reorder for attention ----------------------------------------

#     Q = Q.permute(0, 3, 2, 1, 4)            # (B, KV, GROUP, T, D)
#     K = K.permute(0, 2, 1, 3).unsqueeze(2) # (B, KV, 1, T, D)
#     V = V.permute(0, 2, 1, 3).unsqueeze(2)

#     # ---- 4. Scaled dot-product attention ---------------------------------

#     S = torch.matmul(Q.float(), K.float().transpose(-1, -2)) # [B, KV, GROUP, Qlen, KVlen] f32 as accumulate type
#     P = torch.softmax(S, dim=-1)
#     O = torch.matmul(P, V.float()) # [B, KV, GROUP, Qlen, D]

#     # ---- 5. Restore to (B, T, Q_HEAD * HEAD_DIM) ----------------------------------

#     O = O.permute(0, 3, 2, 1, 4) \
#          .reshape(NUM_REQ * Q_SEQ_LEN, NUM_Q_HEAD * HEAD_DIM)

#     return S, P, O


# def attention_online(req, head):
#     Q = matQ_attn_view[req, :, head] # [Q_SEQ_LEN * HEAD_GROUP_SIZE, HEAD_DIM]
#     K = matK_attn_view[req, :, head] # [KV_SEQ_LEN, HEAD_DIM]
#     V = matV_attn_view[req, :, head] # [KV_SEQ_LEN, HEAD_DIM]
#     O = torch.zeros(Q_SEQ_LEN * HEAD_GROUP_SIZE, HEAD_DIM, dtype=torch.float16, device=gpu)

#     qk = torch.zeros(Q_SEQ_LEN * HEAD_GROUP_SIZE, KV_SEQ_LEN, dtype=torch.float32, device=gpu)
#     p = torch.zeros(Q_SEQ_LEN * HEAD_GROUP_SIZE, KV_SEQ_LEN, dtype=torch.float32, device=gpu)
#     row_maxs = torch.zeros(Q_SEQ_LEN * HEAD_GROUP_SIZE, dtype=torch.float32, device=gpu)
#     pv = torch.zeros(Q_SEQ_LEN * HEAD_GROUP_SIZE, HEAD_DIM, dtype=torch.float32, device=gpu)
#     row_sums = torch.zeros(Q_SEQ_LEN * HEAD_GROUP_SIZE, dtype=torch.float32, device=gpu)

#     for qi in range(Q_SEQ_LEN // QTile):
#         Q_block = Q[qi * QTile * HEAD_GROUP_SIZE: (qi + 1) * QTile * HEAD_GROUP_SIZE] # [QTile * HEAD_GROUP_SIZE, HEAD_DIM]
#         # print("Q_block shape: ", Q_block.shape)
#         # for r in range(4):
#         #     for c in range(128):
#         #         print(f"Q_block[{r},{c}]={Q_block[r,c].item()}")

#         O_block = torch.zeros(QTile * HEAD_GROUP_SIZE, HEAD_DIM, dtype=torch.float16, device=gpu)
#         for ki in range(KV_SEQ_LEN // KVTile):
#             K_block = K[ki * KVTile: (ki + 1) * KVTile] # [KVTile, HEAD_DIM]
#             V_block = V[ki * KVTile: (ki + 1) * KVTile] # [KVTile, HEAD_DIM]
#             S_block = Q_block.float() @ K_block.T.float() # [QTile * HEAD_GROUP_SIZE, KVTile] f32 as accumulate type
#             qk[qi * QTile * HEAD_GROUP_SIZE: (qi + 1) * QTile * HEAD_GROUP_SIZE, ki * KVTile: (ki + 1) * KVTile] = S_block
#             # mask S_block column after active kv len to -inf
#             for c in range(KVTile):
#                 if False and ki * KVTile + c >= 16:
#                     S_block[:, c] = -float('inf')
            
#             if ki == 0:
#                 row_max = torch.max(S_block, dim=-1, keepdim=True).values
#                 row_maxs[qi * QTile * HEAD_GROUP_SIZE: (qi + 1) * QTile * HEAD_GROUP_SIZE] = row_max.squeeze()
#                 exp_S = torch.exp(S_block - row_max)
#                 row_sum = torch.sum(exp_S, dim=-1, keepdim=True)
#             else:
#                 # new_row_max should be max(old_row_max, S_block)
#                 new_row_max = torch.max(torch.cat([row_max, torch.max(S_block, dim=-1, keepdim=True).values], dim=-1), dim=-1, keepdim=True).values
#                 score_scaler = torch.exp(row_max - new_row_max)
#                 exp_S = torch.exp(S_block - new_row_max)
#                 row_sum = row_sum * score_scaler + torch.sum(exp_S, dim=-1, keepdim=True)
#                 row_max = new_row_max
#                 O_block = O_block * score_scaler
#             p[qi * QTile * HEAD_GROUP_SIZE: (qi + 1) * QTile * HEAD_GROUP_SIZE, ki * KVTile: (ki + 1) * KVTile] = exp_S
#             pv_block = exp_S @ V_block.float() # [QTile * HEAD_GROUP_SIZE, HEAD_DIM]
#             pv[qi * QTile * HEAD_GROUP_SIZE: (qi + 1) * QTile * HEAD_GROUP_SIZE] = pv_block
#             O_block += pv_block.float()
#         O[qi * QTile * HEAD_GROUP_SIZE: (qi + 1) * QTile * HEAD_GROUP_SIZE] = O_block / row_sum
#         row_sums[qi * QTile * HEAD_GROUP_SIZE: (qi + 1) * QTile * HEAD_GROUP_SIZE] = row_sum.squeeze()
#     return O, O_block, qk, p, row_maxs, pv, row_sums

# refS, refP, refO = attention()
# refonline, online_O_before_scale, qk_online, p_online, rowmax_s, pv_online, row_sums = attention_online(0,0)

# refO_detail = refO.view(NUM_REQ, Q_SEQ_LEN, HEAD_GROUP_SIZE, NUM_KV_HEAD, HEAD_DIM)
# tensor_diff("Ref and RefOnline", refO_detail[0,:,:,0], refonline.view(Q_SEQ_LEN, HEAD_GROUP_SIZE, HEAD_DIM))
# tensor_diff("Ref and DAE", refO_detail, matO_attn_view.view(NUM_REQ, Q_SEQ_LEN, HEAD_GROUP_SIZE, NUM_KV_HEAD, HEAD_DIM))
