import torch
from math import sqrt

from dae.launcher import *
from dae.schedule import SchedAttentionSplit
from dae.model import (
    cord_split_load_k,
    cord_split_load_o,
    cord_split_load_q,
    cord_split_load_v,
    tma_split_load_k,
    tma_split_load_o,
    tma_split_load_q,
    tma_split_load_v,
    calc_split_meta,
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
MAX_SPLIT = 128

seq_lengths = [2048] * NUM_REQ

assert len(seq_lengths) == NUM_REQ

num_sms = 128
dae = Launcher(num_sms, device=gpu)

matQ = torch.rand(NUM_REQ, HIDDEN_SIZE, dtype=torch.bfloat16, device=gpu) - 0.5
matK = torch.rand(NUM_REQ * KV_SEQ_LEN, NUM_KV_HEAD * HEAD_DIM, dtype=torch.bfloat16, device=gpu) - 0.5
matV = torch.rand(NUM_REQ * KV_SEQ_LEN, NUM_KV_HEAD * HEAD_DIM, dtype=torch.bfloat16, device=gpu) - 0.5
matO = torch.zeros(NUM_REQ, HIDDEN_SIZE, dtype=torch.bfloat16, device=gpu)
matO_attn_view = matO.view(NUM_REQ, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)

matO_split = torch.zeros(
    MAX_SPLIT,
    NUM_REQ,
    NUM_KV_HEAD,
    HEAD_GROUP_SIZE,
    HEAD_DIM,
    dtype=torch.bfloat16,
    device=gpu,
)
matP = torch.zeros(
    NUM_REQ,
    MAX_SPLIT,
    NUM_KV_HEAD,
    HEAD_GROUP_SIZE,
    dtype=torch.float,
    device=gpu,
)

matQ_attn_view = matQ.view(NUM_REQ, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)
matK_attn_view = matK.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)
matV_attn_view = matV.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)

matO_split_post_load_view = matO_split.view(MAX_SPLIT, NUM_REQ, NUM_Q_HEAD, HEAD_DIM)
matO_post_store_view = matO.view(NUM_REQ, NUM_Q_HEAD, HEAD_DIM)

tasks = []
base_sm = 0
for req in range(NUM_REQ):
    split_kv, split_q_tile, splits_per_post_load = calc_split_meta(
        16,
        NUM_Q_HEAD,
        NUM_KV_HEAD,
        HEAD_DIM,
        (seq_lengths[req] + KV_TILE - 1) // KV_TILE,
    )

    tQ = TmaTensor(dae, matQ_attn_view[req:req + 1])._build("load", HEAD_DIM, 64, tma_split_load_q, cord_split_load_q)
    tK = TmaTensor(dae, matK_attn_view[req:req + 1])._build("load", HEAD_DIM, KV_TILE, tma_split_load_k, cord_split_load_k)
    tV = TmaTensor(dae, matV_attn_view[req:req + 1])._build("load", HEAD_DIM, KV_TILE, tma_split_load_v, cord_split_load_v)

    matO_split_store_req = matO_split[:split_kv, req:req + 1]
    matO_split_post_load_req = matO_split_post_load_view[:split_kv, req:req + 1]
    matO_post_store_req = matO_post_store_view[req]

    tO_split_post_load = TmaTensor(dae, matO_split_post_load_req)._build(
        "load",
        splits_per_post_load,
        split_q_tile * HEAD_DIM,
        tma_split_load_o,
        cord_split_load_o,
    )
    bar = dae.new_bar(NUM_KV_HEAD * split_kv)

    tasks.append(
        SchedAttentionSplit(
            dae=dae,
            seq_len=seq_lengths[req],
            KV_BLOCK_SIZE=KV_TILE,
            NUM_Q_HEADS=NUM_Q_HEAD,
            NUM_KV_HEADS=NUM_KV_HEAD,
            split_kv=split_kv,
            split_q_tile=split_q_tile,
            splits_per_post_load=splits_per_post_load,
            matO=matO_post_store_req,
            matO_split=matO_split_store_req,
            matP=matP[req],
            tmas=(tQ, tK, tV, tO_split_post_load),
        ).place(split_kv * NUM_KV_HEAD, base_sm).bar('o_split', bar)
    )
    base_sm += split_kv * NUM_KV_HEAD

dae.i(
    tasks,
    TerminateC(),
    TerminateM(),
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
