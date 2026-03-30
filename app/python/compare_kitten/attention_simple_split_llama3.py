import torch
from math import sqrt

from dae.launcher import *
from dae.schedule import SchedAttentionSplit
from dae.tma_utils import (
    cord_split_load_k,
    cord_split_load_o,
    cord_split_load_q,
    cord_split_load_v,
    tma_split_load_k,
    tma_split_load_o,
    tma_split_load_q,
    tma_split_load_v,
)
from dae.util import dae_app, tensor_diff


gpu = torch.device("cuda")
torch.manual_seed(0)

KV_SEQ_LEN = 2048
HEAD_DIM = 128
NUM_Q_HEAD = 32
NUM_KV_HEAD = 8
HEAD_GROUP_SIZE = NUM_Q_HEAD // NUM_KV_HEAD
HIDDEN_SIZE = NUM_Q_HEAD * HEAD_DIM
NUM_REQ = 8
KV_TILE = 64

ATTN_SPLIT_KV = 2
ATTN_SPLIT_SMS_PER_REQ = 16
ATTN_SPLIT_Q_TILE = 4
ATTN_SPLITS_PER_POST_LOAD = 2

seq_lengths = [512, 640, 768, 896, 1024, 1152, 1280, 1408]
assert len(seq_lengths) == NUM_REQ

num_sms = NUM_REQ * ATTN_SPLIT_SMS_PER_REQ
dae = Launcher(num_sms, device=gpu)

matQ = torch.rand(NUM_REQ, HIDDEN_SIZE, dtype=torch.bfloat16, device=gpu) - 0.5
matK = torch.rand(NUM_REQ * KV_SEQ_LEN, NUM_KV_HEAD * HEAD_DIM, dtype=torch.bfloat16, device=gpu) - 0.5
matV = torch.rand(NUM_REQ * KV_SEQ_LEN, NUM_KV_HEAD * HEAD_DIM, dtype=torch.bfloat16, device=gpu) - 0.5
matO = torch.zeros(NUM_REQ, HIDDEN_SIZE, dtype=torch.bfloat16, device=gpu)
matO_split = torch.zeros(
    ATTN_SPLIT_KV,
    NUM_REQ,
    NUM_KV_HEAD,
    HEAD_GROUP_SIZE,
    HEAD_DIM,
    dtype=torch.bfloat16,
    device=gpu,
)
matP = torch.zeros(
    ATTN_SPLIT_KV,
    NUM_REQ,
    NUM_KV_HEAD,
    HEAD_GROUP_SIZE,
    dtype=torch.float,
    device=gpu,
)

matQ_attn_view = matQ.view(NUM_REQ, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)
matK_attn_view = matK.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)
matV_attn_view = matV.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)
matO_attn_view = matO.view(NUM_REQ, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)
matO_split_load_view = matO_split.view(ATTN_SPLIT_KV, NUM_REQ, NUM_Q_HEAD, HEAD_DIM)

tasks = []
for req in range(NUM_REQ):
    tQ = TmaTensor(dae, matQ_attn_view[req:req + 1])._build("load", HEAD_DIM, 64, tma_split_load_q, cord_split_load_q)
    tK = TmaTensor(dae, matK_attn_view[req:req + 1])._build("load", HEAD_DIM, KV_TILE, tma_split_load_k, cord_split_load_k)
    tV = TmaTensor(dae, matV_attn_view[req:req + 1])._build("load", HEAD_DIM, KV_TILE, tma_split_load_v, cord_split_load_v)
    tO_split = TmaTensor(dae, matO_split_load_view[:, req:req + 1])._build(
        "load",
        ATTN_SPLITS_PER_POST_LOAD,
        HEAD_DIM * ATTN_SPLIT_Q_TILE,
        tma_split_load_o,
        cord_split_load_o,
    )
    tasks.append(
        SchedAttentionSplit(
            dae=dae,
            seq_len=seq_lengths[req],
            KV_BLOCK_SIZE=KV_TILE,
            NUM_Q_HEADS=NUM_Q_HEAD,
            NUM_KV_HEADS=NUM_KV_HEAD,
            split_kv=ATTN_SPLIT_KV,
            split_q_tile=ATTN_SPLIT_Q_TILE,
            splits_per_post_load=ATTN_SPLITS_PER_POST_LOAD,
            matO=matO_attn_view[req],
            matO_split=matO_split[:, req],
            matP=matP[:, req],
            tmas=(tQ, tK, tV, tO_split),
        ).place(ATTN_SPLIT_SMS_PER_REQ, req * ATTN_SPLIT_SMS_PER_REQ)
    )

dae.i(
    [task.schedule for task in tasks],
    TerminateC(),
    TerminateM(),
)

print(
    "llama3 split mvp:",
    f"reqs={NUM_REQ}",
    f"q_heads={NUM_Q_HEAD}",
    f"kv_heads={NUM_KV_HEAD}",
    f"split_kv={ATTN_SPLIT_KV}",
    f"sms_per_req={ATTN_SPLIT_SMS_PER_REQ}",
)

dae_app(dae)


def gqa_ref():
    q = matQ.view(NUM_REQ, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)
    k = matK.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM).permute(0, 2, 1, 3)
    v = matV.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM).permute(0, 2, 1, 3)

    qk = torch.matmul(q, k.transpose(-1, -2)) / sqrt(HEAD_DIM)
    active_kv_len = torch.tensor(seq_lengths, device=gpu, dtype=torch.long)
    mask = torch.arange(KV_SEQ_LEN, device=gpu)[None, None, None, :] >= active_kv_len[:, None, None, None]
    qk = qk.masked_fill(mask, float("-inf"))
    attn = torch.softmax(qk, dim=-1)
    return torch.matmul(attn, v)


refO = gqa_ref()
tensor_diff("Ref and DAE", refO, matO_attn_view, threshold=3.0)
