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

num_sms = NUM_Q_HEAD * NUM_REQ
assert num_sms <= 132 # max sm count for HX00

dae = Launcher(num_sms, device=gpu)

matQ = torch.rand(NUM_REQ, HIDDEN_SIZE, dtype=torch.bfloat16, device=gpu) - 0.5
matK = torch.rand(NUM_REQ * KV_SEQ_LEN, NUM_KV_HEAD * HEAD_DIM, dtype=torch.bfloat16, device=gpu) - 0.5
matV = torch.rand(NUM_REQ * KV_SEQ_LEN, NUM_KV_HEAD * HEAD_DIM, dtype=torch.bfloat16, device=gpu) - 0.5
matO = torch.zeros(NUM_REQ, HIDDEN_SIZE, dtype=torch.bfloat16, device=gpu)

# interleaved QKV
matQ_attn_view = matQ.view(NUM_REQ, NUM_Q_HEAD, HEAD_DIM)
matK_attn_view = matK.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)
matV_attn_view = matV.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)
matO_attn_view = matO.view(NUM_REQ, NUM_Q_HEAD, HEAD_DIM)

matQK = torch.zeros(NUM_REQ, NUM_KV_HEAD, 64, 64, dtype=torch.bfloat16, device=gpu)


def env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def build_interleaved_rope_rows(max_seq_len: int, head_dim: int, rope_theta: float, device, dtype):
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    rope = torch.empty(max_seq_len, head_dim, device=device, dtype=dtype)
    rope[:, 0::2] = freqs.cos().to(dtype=dtype)
    rope[:, 1::2] = freqs.sin().to(dtype=dtype)
    return rope


def apply_rms_affine_rope_heads(hidden_states: torch.Tensor, weight: torch.Tensor, rope_row: torch.Tensor, eps: float):
    hidden_states = hidden_states.float()
    variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    hidden_states = hidden_states * weight.float().view(*([1] * (hidden_states.ndim - 1)), -1)
    even = hidden_states[..., 0::2]
    odd = hidden_states[..., 1::2]
    cos = rope_row[..., 0::2].float()
    sin = rope_row[..., 1::2].float()
    return torch.stack(
        (even * cos - odd * sin, even * sin + odd * cos),
        dim=-1,
    ).flatten(-2).to(dtype=weight.dtype)

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
    glob_dims = [64, 1, 64, 2, NUM_REQ * NUM_Q_HEAD]
    glob_strides = [128 * 2, 0, 64 * 2, HEAD_DIM * 2]
    box_dims = [64, 1, 64, 2, 1]

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
        return [0, 0, 0, cords[0] * NUM_Q_HEAD + cords[1]]
    return cfunc


tQ = TmaTensor(dae, matQ_attn_view)._build("load", HEAD_DIM, 64, tma_load_o, cord_load_o)
tK = TmaTensor(dae, matK_attn_view)._build("load", HEAD_DIM, KVTile, tma_builder_K, cord_func_K)
tV = TmaTensor(dae, matV_attn_view)._build("load", HEAD_DIM, KVTile, tma_builder_MN, cord_func_MN)

need_norm = env_flag("ATTENTION_NEED_NORM", True)
need_rope = env_flag("ATTENTION_NEED_ROPE", True)
if need_norm != need_rope:
    raise ValueError("attention_simple_decoding.py mirrors the fused Qwen decode path, so norm and rope must be enabled together")

ATTENTION_IMPL = os.environ.get("ATTENTION_IMPL", "hopper").lower()
if ATTENTION_IMPL == "mma":
    attention_inst = ATTENTION_M64N64K16_F16_F32_64_64_hdim_MMA
elif ATTENTION_IMPL in ("hopper", "gmma", ""):
    attention_inst = ATTENTION_M64N64K16_F16_F32_64_64_hdim
else:
    raise ValueError(f"Unsupported ATTENTION_IMPL={ATTENTION_IMPL!r}; expected 'hopper' or 'mma'")

NUM_KV_BLOCK = (KV_SEQ_LEN + KVTile - 1) // KVTile
last_active_kv_len = 48
assert last_active_kv_len <= KVTile
total_active_kv_len = (NUM_KV_BLOCK - 1) * KVTile + last_active_kv_len
token_pos = total_active_kv_len - 1

rope_theta = float(os.environ.get("ROPE_THETA", "1000000.0"))
q_norm_weight = 0.75 + 0.5 * torch.rand(HEAD_DIM, dtype=torch.bfloat16, device=gpu)
k_norm_weight = 0.75 + 0.5 * torch.rand(HEAD_DIM, dtype=torch.bfloat16, device=gpu)
rope_table = build_interleaved_rope_rows(KV_SEQ_LEN, HEAD_DIM, rope_theta, gpu, torch.bfloat16)
matSideInput = torch.empty(KV_SEQ_LEN, 3 * HEAD_DIM, dtype=torch.bfloat16, device=gpu)
matSideInput[:, 0:HEAD_DIM] = q_norm_weight.view(1, HEAD_DIM)
matSideInput[:, HEAD_DIM:2 * HEAD_DIM] = k_norm_weight.view(1, HEAD_DIM)
matSideInput[:, 2 * HEAD_DIM:3 * HEAD_DIM] = rope_table

tSideInput = TmaTensor(dae, matSideInput).tensor1d("load", 3 * HEAD_DIM)
current_k_store = [
    TmaTensor(dae, matK.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD * HEAD_DIM)[req]).tensor1d("store", HEAD_DIM)
    for req in range(NUM_REQ)
]

if need_norm and need_rope and token_pos > 0:
    matK_attn_view[:, :token_pos] = apply_rms_affine_rope_heads(
        matK_attn_view[:, :token_pos],
        k_norm_weight,
        rope_table[:token_pos].view(1, token_pos, 1, HEAD_DIM),
        eps=1.0e-6,
    )

def sm_task(sm: int):
    head = sm % NUM_Q_HEAD
    req = sm // NUM_Q_HEAD
    kv_head = head // HEAD_GROUP_SIZE

    insts = [
        attention_inst(NUM_KV_BLOCK, last_active_kv_len, need_norm=need_norm, need_rope=need_rope),
        tSideInput.cord(token_pos * 3 * HEAD_DIM) if need_norm and need_rope else [],
        current_k_store[req].cord((token_pos * NUM_KV_HEAD + kv_head) * HEAD_DIM) if need_norm and need_rope else [],
        tQ.cord(req, head),
        RepeatM.on(NUM_KV_BLOCK - 1,
            [tK.cord(req, 0, kv_head, 0), tK.cord2tma(0, KVTile, 0, 0)],
            [tV.cord(req, 0, kv_head, 0), tV.cord2tma(0, KVTile, 0, 0)],
        ),
        tK.cord(req, KVTile * (NUM_KV_BLOCK - 1), kv_head, 0),
        tV.cord(req, KVTile * (NUM_KV_BLOCK - 1), kv_head, 0),
        # here we override the allocator, to allocate enough space in the smem
        # but we will only write back the first 128*16*2 bytes to the output mat
        TmaStore1D(matO_attn_view[req, head, ...])
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
    if need_norm and need_rope:
        Q = apply_rms_affine_rope_heads(
            Q.view(NUM_REQ, 1, NUM_Q_HEAD, HEAD_DIM),
            q_norm_weight,
            rope_table[token_pos].view(1, 1, 1, HEAD_DIM),
            eps=1.0e-6,
        ).view(NUM_REQ, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)

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
    mask = torch.arange(KV_SEQ_LEN, device=gpu)[None, None, None, :] >= total_active_kv_len
    QK = QK.masked_fill(mask, float("-inf"))

    # softmax on sequence dimension
    attn = torch.softmax(QK, dim=-1)   # [B, Hkv, G, S]

    # output = attn @ V
    return QK, torch.matmul(attn, V)

refQK, refO = gqa_ref()
tensor_diff("Ref and DAE", refO.view(matO_attn_view.shape), matO_attn_view)

