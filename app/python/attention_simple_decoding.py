import torch
import copy
import os
from math import sqrt
from functools import partial
from dae.launcher import *
from dae.util import *
from dae.runtime import opcode, build_tma_desc 
from qwen3.utils import *
import sys

gpu = torch.device("cuda")
torch.manual_seed(0)

KV_SEQ_LEN = 2048
HEAD_DIM = 128
HIDDEN_SIZE = 4096
NUM_REQ = 2
NUM_Q_HEAD = 32
NUM_KV_HEAD = 8
HEAD_GROUP_SIZE = NUM_Q_HEAD // NUM_KV_HEAD
seq_lengths = [2048, 1024]

assert HIDDEN_SIZE == NUM_KV_HEAD * HEAD_GROUP_SIZE * HEAD_DIM, "Q size must match HIDDEN SIZE"
assert len(seq_lengths) == NUM_REQ, "Length of seq_lengths must match NUM_REQ"
for seq_len in seq_lengths:
    assert seq_len <= KV_SEQ_LEN, "Sequence length must be less than or equal to KV_SEQ_LEN"

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


def tma_load_k(mat: torch.Tensor, tileK: int, tileN: int):
    R, S, H, D = mat.shape
    glob_dims = [64, S, 2, H, R]
    elsize = mat.element_size()
    glob_strides = [d * elsize for d in [mat.stride(1), 64, mat.stride(2), mat.stride(0)]]
    box_dims = [64, tileN, 2, 1, 1]
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

def cord_load_k(mat: torch.Tensor, rank: int):
    assert rank == 5, "Only support 5D TMA load for K/V"
    def cfunc(*cords):
        assert len(cords) == 3, f"cords should be (req, seq, head), but got {cords}"
        r, s, h = cords
        return [s, 0, h, r]
    return cfunc

def tma_load_v(mat: torch.Tensor, tileM: int, tileK: int):
    # mat: [R, S, H, D]
    R, S, H, D = mat.shape
    elsize = mat.element_size()

    assert D == HEAD_DIM
    assert tileM == HEAD_DIM
    assert tileK == KVTile
    assert D % 64 == 0
    assert S % 8 == 0

    M_total = H * D  # fold head into M

    glob_dims = [64, 8, M_total // 64, S // 8, R]
    glob_strides = [
        mat.stride(1),      # seq stride
        64,                 # next 64 elems in folded M
        mat.stride(1) * 8,  # next 8 seq elems
        mat.stride(0),      # next request
    ]
    glob_strides = [s * elsize for s in glob_strides]

    box_dims = [64, 8, tileM // 64, tileK // 8, 1]
    rank = len(glob_dims)
    box_strides = [1] * rank

    return rank, runtime.build_tma_desc(
        mat,
        glob_dims,
        glob_strides,
        box_dims,
        box_strides,
        128,
        0,
    )

def cord_load_v(mat: torch.Tensor, rank: int):
    assert rank == 5, "Only support 5D TMA load for V"
    def cfunc(*cords):
        # cords: (req, seq, head)
        assert len(cords) == 3, f"cords should be (req, seq, head), but got {cords}"
        r, s, h = cords
        return [0, h * (HEAD_DIM // 64), s // 8, r]
    return cfunc


tQ = TmaTensor(dae, matQ_attn_view)._build("load", HEAD_DIM, 64, tma_load_o, cord_load_o)
tK = TmaTensor(dae, matK_attn_view)._build("load", HEAD_DIM, KVTile, tma_load_k, cord_load_k)
tV = TmaTensor(dae, matV_attn_view)._build("load", HEAD_DIM, KVTile, tma_load_v, cord_load_v)

need_norm = False
need_rope = False

<<<<<<< HEAD
ATTENTION_IMPL = os.environ.get("ATTENTION_IMPL", "hopper").lower()
if ATTENTION_IMPL == "mma":
    attention_inst = ATTENTION_M64N64K16_F16_F32_64_64_hdim_MMA
elif ATTENTION_IMPL in ("hopper", "gmma", ""):
    attention_inst = ATTENTION_M64N64K16_F16_F32_64_64_hdim
else:
    raise ValueError(f"Unsupported ATTENTION_IMPL={ATTENTION_IMPL!r}; expected 'hopper' or 'mma'")

NUM_KV_BLOCK = (KV_SEQ_LEN + KVTile - 1) // KVTile
last_active_kv_len = 48
=======
last_active_kv_len = 64
>>>>>>> 3f67a0b (fix tma util collapsed dim)
assert last_active_kv_len <= KVTile

def sm_task(sm: int):
    head = sm % NUM_KV_HEAD
    req = sm // NUM_KV_HEAD
    seq_length = seq_lengths[req]
    num_kv_block = (seq_length + KVTile - 1) // KVTile

    insts = [
<<<<<<< HEAD
        attention_inst(NUM_KV_BLOCK, last_active_kv_len, need_norm=need_norm, need_rope=need_rope),
=======
        ATTENTION_M64N64K16_F16_F32_64_64_hdim(num_kv_block, last_active_kv_len, need_norm=need_norm, need_rope=need_rope),
>>>>>>> 3f67a0b (fix tma util collapsed dim)
        tQ.cord(req, head),
        RepeatM.on(num_kv_block,
            [tK.cord(req, 0, head), tK.cord2tma(0, KVTile, 0)],
            [tV.cord(req, 0, head), tV.cord2tma(0, KVTile, 0)],
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
    QK = torch.matmul(Q, K.transpose(-1, -2)) / sqrt(HEAD_DIM)
    # Apply a per-request causal length mask so each request can expose a different
    # active KV span while sharing the same backing K/V buffers.
    active_kv_len = torch.tensor(seq_lengths, device=gpu, dtype=torch.long)
    mask = torch.arange(KV_SEQ_LEN, device=gpu)[None, None, None, :] >= active_kv_len[:, None, None, None]
    QK = QK.masked_fill(mask, float("-inf"))

    # softmax on sequence dimension
    attn = torch.softmax(QK, dim=-1)   # [B, Hkv, G, S]

    # output = attn @ V
    return QK, torch.matmul(attn, V)

refQK, refO = gqa_ref()
tensor_diff("Ref and DAE", refO, matO_attn_view)

dae_app(dae)
