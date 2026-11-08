# M2 Part B: decode attention at Llama 3.1 70B's shape, checked against PyTorch.
#
# 70B uses grouped-query attention: 64 query heads share 8 KV heads, so each
# KV head serves 8 query heads (8B: 4). Runs the same path sched_70b.py uses
# (SchedAttentionDecoding + tma_gqa_load_q) on its own.
#
#   python m2_attention_70b.py -l                  correctness
#   python m2_attention_70b.py -b 100              correctness + timing
#   SEQ_LEN=64 python m2_attention_70b.py -l       exactly one full KV block
#   SEQ_LEN=512 python m2_attention_70b.py -b 100  longest context the app allows
import math
import os
from functools import partial

import torch
from dae.launcher import *
from dae.schedule import SchedAttentionDecoding
from dae.model import (tma_gqa_load_q, cord_gqa_load_q,
                       build_tma_wgmma_k, cord_func_K_major,
                       build_tma_wgmma_mn, cord_func_MN_major)
from dae.util import dae_app, tensor_diff

gpu = torch.device("cuda")
torch.manual_seed(0)
dtype = torch.bfloat16

# same values as sched_70b.py
N = 8                                  # requests, one new token each
NUM_Q_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128
GROUP = NUM_Q_HEADS // NUM_KV_HEADS    # 8 query heads per KV head
KV_BLOCK = 64
MAX_SEQ_LEN = 512
SEQ_LEN = int(os.environ.get("SEQ_LEN", "200"))  # 200 = 3 full blocks + a partial one
assert 1 <= SEQ_LEN <= MAX_SEQ_LEN

num_sms = N * NUM_KV_HEADS             # one SM per (request, KV head) = 64
dae = Launcher(num_sms, device=gpu)

# Q = the new token's query (already roped); K/V = the cache
matQ = (torch.rand(N, NUM_Q_HEADS * HEAD_DIM, device=gpu) - 0.5).to(dtype)
matK = (torch.rand(N, MAX_SEQ_LEN, NUM_KV_HEADS * HEAD_DIM, device=gpu) - 0.5).to(dtype)
matV = (torch.rand(N, MAX_SEQ_LEN, NUM_KV_HEADS * HEAD_DIM, device=gpu) - 0.5).to(dtype)
# cache slots past SEQ_LEN get huge values: if the kernel reads them, the answer is visibly wrong
matK[:, SEQ_LEN:] = 50.0
matV[:, SEQ_LEN:] = 50.0
matO = torch.full((N, NUM_Q_HEADS * HEAD_DIM), float("nan"), dtype=dtype, device=gpu)

# same views and TMA setup as sched_70b.py
viewQ = matQ.view(N, NUM_KV_HEADS, GROUP, HEAD_DIM)
viewK = matK.view(N, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM)
viewV = matV.view(N, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM)
viewO = matO.view(N, NUM_KV_HEADS, GROUP, HEAD_DIM)

tma_builder_K = partial(build_tma_wgmma_k, iN=-3)
cord_func_K = partial(cord_func_K_major, iN=-3)
tma_builder_MN = partial(build_tma_wgmma_mn, iK=-3)
cord_func_MN = partial(cord_func_MN_major, iK=-3)

loadQ = TmaTensor(dae, viewQ)._build("load", HEAD_DIM, 64, tma_gqa_load_q, cord_gqa_load_q)
loadK = TmaTensor(dae, viewK)._build("load", HEAD_DIM, KV_BLOCK, tma_builder_K, cord_func_K)
loadV = TmaTensor(dae, viewV)._build("load", HEAD_DIM, KV_BLOCK, tma_builder_MN, cord_func_MN)

attn = SchedAttentionDecoding(
    reqs=N, seq_len=SEQ_LEN,
    KV_BLOCK_SIZE=KV_BLOCK,
    NUM_KV_HEADS=NUM_KV_HEADS,
    matO=viewO,
    tmas=(loadQ, loadK, loadV),
).place(num_sms)

dae.s(attn)
dae_app(dae)


def ref():
    q = matQ.float().view(N, NUM_Q_HEADS, 1, HEAD_DIM)
    k = matK[:, :SEQ_LEN].float().view(N, SEQ_LEN, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    v = matV[:, :SEQ_LEN].float().view(N, SEQ_LEN, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    # query head h reads KV head h // GROUP, same as HuggingFace's repeat_kv
    k = k.repeat_interleave(GROUP, dim=1)
    v = v.repeat_interleave(GROUP, dim=1)
    scores = q @ k.transpose(-1, -2) / math.sqrt(HEAD_DIM)
    out = torch.softmax(scores, dim=-1) @ v
    return out.view(N, NUM_Q_HEADS * HEAD_DIM).to(dtype)


assert not matO.isnan().any(), "some outputs never written (run with -l or -b)"
tensor_diff(f"GQA decode attention, seq_len={SEQ_LEN}", ref(), matO)