import torch
import copy
from math import sqrt
from functools import partial
from dae.launcher import *
from dae.util import *
from dae.runtime import opcode, build_tma_desc 
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
seq_lengths = [65536] * 1
NUM_REQ = len(seq_lengths)

assert HIDDEN_SIZE == NUM_KV_HEAD * HEAD_GROUP_SIZE * HEAD_DIM, "Q size must match HIDDEN SIZE"

QTile = 64 // HEAD_GROUP_SIZE
KVTile = 64

split_kv = 128
assert split_kv <= MAX_SPLIT
num_sms = NUM_KV_HEAD * NUM_REQ * split_kv
assert num_sms <= 132 # max sm count for HX00
POST_SPLIT_LOAD_LIMIT_BYTES = 32 * 1024
SPLITS_PER_POST_LOAD = min(max(1, POST_SPLIT_LOAD_LIMIT_BYTES // (HEAD_GROUP_SIZE * HEAD_DIM * 2)), split_kv)
assert split_kv % SPLITS_PER_POST_LOAD == 0, "For simplicity we require split_kv to be divisible by SPLITS_PER_POST_LOAD"
print(f"num_sms: {num_sms}, splits per post load: {SPLITS_PER_POST_LOAD}")

dae = Launcher(num_sms, device=gpu)

matQ = torch.rand(NUM_REQ, HIDDEN_SIZE, dtype=torch.bfloat16, device=gpu) - 0.5
matK = torch.rand(NUM_REQ * KV_SEQ_LEN, NUM_KV_HEAD * HEAD_DIM, dtype=torch.bfloat16, device=gpu) - 0.5
matV = torch.rand(NUM_REQ * KV_SEQ_LEN, NUM_KV_HEAD * HEAD_DIM, dtype=torch.bfloat16, device=gpu) - 0.5
matO = torch.zeros(NUM_REQ, HIDDEN_SIZE, dtype=torch.bfloat16, device=gpu)
matO_split = torch.zeros(split_kv, NUM_REQ, HIDDEN_SIZE, dtype=torch.bfloat16, device=gpu)
matP = torch.zeros(NUM_KV_HEAD, NUM_REQ, MAX_SPLIT, HEAD_GROUP_SIZE, dtype=torch.float, device=gpu)

# interleaved QKV
matQ_attn_view = matQ.view(NUM_REQ, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)
matK_attn_view = matK.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)
matV_attn_view = matV.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)
matO_attn_view = matO.view(NUM_REQ, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)
matO_split_attn_view = matO_split.view(split_kv, NUM_REQ, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)

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

    # this will dup for QTile times, due to 0 in strides, do not know how tma engine will handle it
    glob_dims = [64, HEAD_GROUP_SIZE, QTile, 2, NUM_REQ * NUM_KV_HEAD]
    glob_strides = [128 * 2, 0, 64 * 2, HEAD_DIM * HEAD_GROUP_SIZE * 2]
    box_dims = [64, HEAD_GROUP_SIZE, QTile, 2, 1]

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


def tma_load_split_attn(mat: torch.Tensor, tileS, tileQ):
    assert tileS <= split_kv
    assert tileQ <= HEAD_GROUP_SIZE
    S, R, H, G, D = mat.shape
    permute = [4, 3, 0, 2, 1] # [D, G, S, H, R]
    glob_dims = [mat.shape[i] for i in permute]
    glob_strides = [mat.stride(i) * mat.element_size() for i in permute[1:]]
    box_dims = [D, tileQ, tileS, 1, 1]
    rank = len(glob_dims)
    box_strides = [1] * rank
    return rank, runtime.build_tma_desc(
        mat,
        glob_dims,
        glob_strides,
        box_dims,
        box_strides,
        0,
        0
    )

def cord_load_split_attn(mat: torch.Tensor, rank: int):
    assert rank == 5, "Only support 5D TMA load for split attn output"
    def cfunc(*cords):
        assert len(cords) == 4, f"cords should be (head, req, q_start, split_start), but got {cords}"
        return [cords[2], cords[3], cords[0], cords[1]]
    return cfunc

tO_split = TmaTensor(dae, matO_split_attn_view)._build(
    "load", 
    SPLITS_PER_POST_LOAD, 
    HEAD_DIM,
    tma_load_split_attn, 
    cord_load_split_attn,
)

need_norm = False
need_rope = False

attn_bar = dae.new_bar(num_sms)

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


def sm_task(sm: int):
    if sm >= num_sms:
        return []
    split_stage = sm // (NUM_KV_HEAD * NUM_REQ)
    ofst_in_group = sm % (NUM_KV_HEAD * NUM_REQ)
    head = ofst_in_group % NUM_KV_HEAD
    req = ofst_in_group // NUM_KV_HEAD
    seq_length = seq_lengths[req]
    _, num_block_per_split, kv_start_block, kv_start_idx, _, total_active, split_last_active_kv_len = split_bounds(seq_length, split_stage)
    if total_active == 0:
        return []
    insts = [
        ATTENTION_M64N64K16_F16_F32_64_64_hdim_split(num_block_per_split, HEAD_GROUP_SIZE, split_last_active_kv_len, kv_start_block, need_norm=need_norm, need_rope=need_rope),
        tQ.cord(req, head),
        RepeatM.on(num_block_per_split,
            [tK.cord(req, kv_start_idx, head), tK.cord2tma(0, KVTile, 0)],
            [tV.cord(req, kv_start_idx, head), tV.cord2tma(0, KVTile, 0)],
        ),
        # here we override the allocator, to allocate enough space in the smem
        # but we will only write back the first 128*16*2 bytes to the output mat
        TmaStore1D(matO_split_attn_view[split_stage, req, head, ...], numSlots = 2),
        TmaStore1D(matP[head, req, split_stage]).bar(attn_bar),
    ]

    if sm >= NUM_KV_HEAD * NUM_REQ * HEAD_GROUP_SIZE:
        return insts
    post_sm = sm
    q_idx = post_sm // (NUM_KV_HEAD * NUM_REQ)
    ofst_in_group = post_sm % (NUM_KV_HEAD * NUM_REQ)
    head = ofst_in_group % NUM_KV_HEAD
    req = ofst_in_group // NUM_KV_HEAD
    insts += [
        ATTN_SPLIT_POST_REDUCE(SPLITS_PER_POST_LOAD, split_kv, num_q=1),
        TmaLoad1D(matP[head, req, :split_kv]).bar(attn_bar),
        RepeatM.on((split_kv + SPLITS_PER_POST_LOAD - 1) // SPLITS_PER_POST_LOAD,
            [tO_split.cord(head, req, q_idx, 0), tO_split.cord2tma(0, 0, 0, SPLITS_PER_POST_LOAD)]
        ),
        TmaStore1D(matO_attn_view[req, head, q_idx:q_idx+1, ...]),
    ]
    return insts

dae.i(
    sm_task,

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
#     tensor_diff(f"Split {s} O", ref_split_o, matO_split_attn_view[s])

#     ref_split_lse_view = ref_split_lse.permute(1, 0, 2)
#     tensor_diff(f"Split {s} LSE", ref_split_lse_view, matP[:, :, s, :].float())

refQK, refO = gqa_ref()
tensor_diff("Ref and DAE", refO, matO_attn_view)
