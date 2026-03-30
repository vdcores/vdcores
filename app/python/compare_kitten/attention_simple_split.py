import torch
import copy
from math import sqrt
from dae.launcher import *
from dae.util import *
from dae.runtime import opcode, build_tma_desc
from split_sched import SchedAttentionSplit
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from qwen3.utils import *

gpu = torch.device("cuda")
torch.manual_seed(0)

KV_SEQ_LEN = 65536
HEAD_DIM = 128
HIDDEN_SIZE = 1024
NUM_Q_HEAD = 8
NUM_KV_HEAD = 1
HEAD_GROUP_SIZE = NUM_Q_HEAD // NUM_KV_HEAD
MAX_SPLIT = 128
seq_lengths = [512] * 64
NUM_REQ = len(seq_lengths)

assert HIDDEN_SIZE == NUM_KV_HEAD * HEAD_GROUP_SIZE * HEAD_DIM, "Q size must match HIDDEN SIZE"

QTile = 64 // HEAD_GROUP_SIZE
KVTile = 64

split_kv = 1
assert split_kv <= MAX_SPLIT
num_sms = 128

dae = Launcher(num_sms, device=gpu)

matQ = torch.rand(NUM_REQ, HIDDEN_SIZE, dtype=torch.bfloat16, device=gpu) - 0.5
matK = torch.rand(NUM_REQ * KV_SEQ_LEN, NUM_KV_HEAD * HEAD_DIM, dtype=torch.bfloat16, device=gpu) - 0.5
matV = torch.rand(NUM_REQ * KV_SEQ_LEN, NUM_KV_HEAD * HEAD_DIM, dtype=torch.bfloat16, device=gpu) - 0.5
matO = torch.zeros(NUM_REQ, HIDDEN_SIZE, dtype=torch.bfloat16, device=gpu)
matO_split = torch.zeros(split_kv, NUM_REQ, HIDDEN_SIZE, dtype=torch.bfloat16, device=gpu)
matP = torch.zeros(NUM_REQ, MAX_SPLIT, NUM_KV_HEAD, HEAD_GROUP_SIZE, dtype=torch.float, device=gpu)

# interleaved QKV
matQ_attn_view = matQ.view(NUM_REQ, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)
matK_attn_view = matK.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)
matV_attn_view = matV.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)
matO_attn_view = matO.view(NUM_REQ, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)
matO_split_attn_view = matO_split.view(split_kv, NUM_REQ, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)
matO_attn_Q_view = matO.view(NUM_REQ, NUM_Q_HEAD, HEAD_DIM)

matQK = torch.zeros(NUM_REQ, NUM_KV_HEAD, 64, 64, dtype=torch.bfloat16, device=gpu)

matO_split_load_view = matO_split.view(split_kv, NUM_REQ, NUM_Q_HEAD, HEAD_DIM)

need_norm = False
need_rope = False
sms_per_req = num_sms // NUM_REQ
assert sms_per_req > 0 and sms_per_req * NUM_REQ == num_sms, "num_sms must divide evenly across requests for this demo"

tasks = [
    SchedAttentionSplit(
        dae=dae,
        req_id=req,
        split_level=split_kv,
        num_sms=sms_per_req,
        base_sm=req * sms_per_req,
        seq_length=seq_lengths[req],
        matQ=matQ,
        matK=matK,
        matV=matV,
        matO=matO,
        matO_split=matO_split,
        matP=matP,
        need_norm=need_norm,
        need_rope=need_rope,
        kv_tile=KVTile,
    )
    for req in range(NUM_REQ)
]

split_q_tile = schedulers[0].split_q_tile
SPLITS_PER_POST_LOAD = schedulers[0].splits_per_post_load
print(f"split_q_tile: {split_q_tile}, SPLITS_PER_POST_LOAD: {SPLITS_PER_POST_LOAD}")

def split_bounds(seq_length: int, split_stage: int):
    num_kv_block = (seq_length + KVTile - 1) // KVTile
    num_block_per_split = num_kv_block // split_kv
    kv_start_block = split_stage * num_block_per_split
    kv_start = kv_start_block * KVTile
    kv_end = kv_start + num_block_per_split * KVTile
    total_active = min(max(seq_length - kv_start, 0), kv_end - kv_start)
    split_last_active_kv_len = total_active % KVTile
    if total_active > 0 and split_last_active_kv_len == 0:
        split_last_active_kv_len = KVTile
    return num_kv_block, num_block_per_split, kv_start_block, kv_start, kv_end, total_active, split_last_active_kv_len

dae.i(
    [t.schedule for t in tasks],   

    TerminateC(),
    TerminateM(),
)

# print("Launching Attention DAE...")

dae_app(dae)

def gqa_ref():
    Q = matQ.view(NUM_REQ, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)     # [B, Hkv, G, D]
    K = matK.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)     # [B, S, Hkv, D]
    V = matV.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)     # [B, S, Hkv, D]

    # move K/V to [B, Hkv, S, D]
    K = K.permute(0, 2, 1, 3)       # [B, Hkv, S, D]
    V = V.permute(0, 2, 1, 3)       # [B, Hkv, S, D]

    # scores = Q @ K^T
    # Q: [B, Hkv, G, D]
    # K.transpose(-1, -2): [B, Hkv, D, S]
    # result: [B, Hkv, G, S]
    QK = torch.matmul(Q, K.transpose(-1, -2)) / sqrt(HEAD_DIM)
    # apply mask according to lsat_active_kv_len
    active_kv_len = torch.tensor(seq_lengths, device=gpu, dtype=torch.long)
    mask = torch.arange(KV_SEQ_LEN, device=gpu)[None, None, None, :] >= active_kv_len[:, None, None, None]
    QK = QK.masked_fill(mask, float("-inf"))

    # softmax on sequence dimension
    attn = torch.softmax(QK, dim=-1)   # [B, Hkv, G, S]

    # output = attn @ V
    return QK, torch.matmul(attn, V)


def split_ref(split_stage):
    """Per-split reference: each split computes local softmax only over its own KV slice.
    Returns O_local = softmax_local(Q @ K_split^T / sqrt(D)) @ V_split  [B, Hkv, G, D]
    and     lse     = max_local + log(sum_local)                         [B, Hkv, G]
    """
    Q = matQ.view(NUM_REQ, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)
    K = matK.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM).permute(0, 2, 1, 3)
    V = matV.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM).permute(0, 2, 1, 3)
    ref_o = torch.zeros(NUM_REQ, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM, dtype=torch.bfloat16, device=gpu)
    ref_lse = torch.full((NUM_REQ, NUM_KV_HEAD, HEAD_GROUP_SIZE), float("-inf"), dtype=torch.float32, device=gpu)

    scale = 1.0 / sqrt(HEAD_DIM)
    for req in range(NUM_REQ):
        _, num_block_per_split, _, kv_start, kv_end, total_active, _ = split_bounds(seq_lengths[req], split_stage)
        if total_active == 0 or num_block_per_split == 0:
            continue

        k_split = K[req:req + 1, :, kv_start:kv_end, :]
        v_split = V[req:req + 1, :, kv_start:kv_end, :]
        q = Q[req:req + 1]

        qk = torch.matmul(q * scale, k_split.transpose(-1, -2))
        split_span = kv_end - kv_start
        mask = torch.arange(split_span, device=gpu)[None, None, None, :] >= total_active
        qk = qk.masked_fill(mask, float("-inf"))

        qk_f = qk.float()
        row_max = qk_f.amax(dim=-1)
        qk_exp2 = torch.exp2(qk_f - row_max.unsqueeze(-1))
        row_sum = qk_exp2.sum(dim=-1)
        ref_lse[req] = (row_max + torch.log2(row_sum)).squeeze(0)
        attn = (qk_exp2 / row_sum.unsqueeze(-1)).to(v_split.dtype)
        ref_o[req] = torch.matmul(attn, v_split).to(torch.bfloat16).squeeze(0)

    return ref_o, ref_lse


# for s in range(split_kv):
#     ref_split_o, ref_split_lse = split_ref(s)
#     tensor_diff(f"Split {s} O", ref_split_o, matO_split_attn_view[s], threshold=3.0)

#     ref_split_lse_view = ref_split_lse.permute(1, 0, 2)
#     tensor_diff(f"Split {s} LSE", ref_split_lse_view, matP[:, s, : :].float())

refQK, refO = gqa_ref()
tensor_diff("Ref and DAE", refO, matO_attn_view)
