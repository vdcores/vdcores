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

KV_SEQ_LEN = 64
HEAD_DIM = 128
HIDDEN_SIZE = 4096
NUM_REQ = 1
NUM_Q_HEAD = 32
NUM_KV_HEAD = 8
HEAD_GROUP_SIZE = NUM_Q_HEAD // NUM_KV_HEAD

assert HIDDEN_SIZE == NUM_KV_HEAD * HEAD_GROUP_SIZE * HEAD_DIM, "Q size must match HIDDEN SIZE"

QTile = 16
KVTile = 64

num_sms = NUM_KV_HEAD * NUM_REQ
assert num_sms <= 132 # max sm count for HX00

dae = Launcher(num_sms, device=gpu)

matQ = torch.rand(NUM_REQ, HIDDEN_SIZE, dtype=torch.bfloat16, device=gpu) - 0.5
matK = torch.rand(NUM_REQ * KV_SEQ_LEN, NUM_KV_HEAD * HEAD_DIM, dtype=torch.bfloat16, device=gpu) - 0.5
matV = torch.rand(NUM_REQ * KV_SEQ_LEN, NUM_KV_HEAD * HEAD_DIM, dtype=torch.bfloat16, device=gpu) - 0.5
matO = torch.zeros(NUM_REQ, HIDDEN_SIZE, dtype=torch.bfloat16, device=gpu)

# interleaved QKV
matQ_attn_view = matQ.view(NUM_REQ, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)
matK_attn_view = matK.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)
matV_attn_view = matV.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)
matO_attn_view = matO.view(NUM_REQ, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)

matQK = torch.zeros(NUM_REQ, NUM_KV_HEAD, 64, 64, dtype=torch.bfloat16, device=gpu)

tma_builder_K_inter = partial(build_tma_wgmma_k, iN = -3, swizzle=0)
tma_builder_K = partial(build_tma_wgmma_k, iN = -3)
cord_func_K = partial(cord_func_K_major, iN=-3)

tma_builder_MN_inter = partial(build_tma_wgmma_mn, iK = -3, swizzle=0)
tma_builder_MN = partial(build_tma_wgmma_mn, iK = -3)
cord_func_MN = partial(cord_func_MN_major, iK=-3)

def tma_load_o(mat: torch.Tensor, tileK: int, tileN: int):
    # [HEAD_DIM[0], HEAD_GROUP_SIZE, REP * HEAD_DIM[1] * NUM_KV_HEAD]
    assert mat.element_size() == 2, "Only support float16/bfloat16 output"
    assert tileK == 128 and tileN == 64, "tile must be 128x64"

    # this will dup for 16 times, due to 0 in strides, do not know how tma engine will handle it
    glob_dims = [64, 4, 16, 2, NUM_REQ * NUM_KV_HEAD]
    glob_strides = [128 * 2, 0, 64 * 2, HEAD_DIM * HEAD_GROUP_SIZE * 2]
    box_dims = [64, 4, 16, 2, 1]

    rank = len(glob_dims)
    box_strides = [1] * rank

    return rank, runtime.build_tma_desc(
        mat,
        glob_dims,
        glob_strides,
        box_dims,
        box_strides,
        128,
        0
    )

def cord_load_o(mat: torch.Tensor, rank: int):
    assert rank == 5, "Only support 5D TMA load for load Q"
    def cfunc(*cords):
        assert len(cords) == 2, f"cords should be (req, head), but got {cords}"
        return [0, 0, 0, cords[0] * NUM_KV_HEAD + cords[1]]
    return cfunc


tQ = TmaTensor(dae, matQ_attn_view)._build("load", HEAD_DIM, 64, tma_load_o, cord_load_o)
tK = TmaTensor(dae, matK_attn_view)._build("load", HEAD_DIM, KVTile, tma_builder_K, cord_func_K)
tV = TmaTensor(dae, matV_attn_view)._build("load", HEAD_DIM, KVTile, tma_builder_MN, cord_func_MN)

need_norm = False
need_rope = False

NUM_KV_BLOCK = (KV_SEQ_LEN + KVTile - 1) // KVTile
last_active_kv_len = 48
assert last_active_kv_len <= KVTile

def sm_task(sm: int):
    head = sm % NUM_KV_HEAD
    req = sm // NUM_KV_HEAD

    insts = [
        ATTENTION_M64N64K16_F16_F32_64_64_hdim(NUM_KV_BLOCK, last_active_kv_len, need_norm=need_norm, need_rope=need_rope),
        tQ.cord(req, head),
        RepeatM.on(NUM_KV_BLOCK,
            [tK.cord(req, 0, head, 0), tK.cord2tma(0, KVTile, 0, 0)],
            [tV.cord(req, 0, head, 0), tV.cord2tma(0, KVTile, 0, 0)],
        ),
        # here we override the allocator, to allocate enough space in the smem
        # but we will only write back the first 128*16*2 bytes to the output mat
        TmaStore1D(matO_attn_view[req, head, ...], numSlots = 2)
    ]
    return insts

dae.i(
    sm_task,

    TerminateC(),
    TerminateM(),
)

# print("Launching Attention DAE...")

dae.launch()

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
    QK = torch.matmul(Q, K.transpose(-1, -2))
    # apply mask according to lsat_active_kv_len
    total_active_KV_len = (NUM_KV_BLOCK-1) * KVTile + last_active_kv_len
    mask = torch.arange(KV_SEQ_LEN, device=gpu)[None, None, None, :] >= total_active_KV_len
    QK = QK.masked_fill(mask, float("-inf"))

    # softmax on sequence dimension
    attn = torch.softmax(QK, dim=-1)   # [B, Hkv, G, S]

    # output = attn @ V
    return QK, torch.matmul(attn, V)

refQK, refO = gqa_ref()
tensor_diff("Ref and DAE", refO[0], matO_attn_view[0])

dae_app(dae)
