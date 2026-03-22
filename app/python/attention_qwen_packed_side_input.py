import math
from functools import partial

import torch

from dae.launcher import *
from dae.runtime import build_tma_desc
from dae.util import dae_app, tensor_diff
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
TOKEN_POS = 7
ROPE_THETA = 1_000_000.0

assert HIDDEN_SIZE == NUM_KV_HEAD * HEAD_GROUP_SIZE * HEAD_DIM

KVTile = 64
num_sms = NUM_KV_HEAD * NUM_REQ
assert num_sms <= 132

dae = Launcher(num_sms, device=gpu)


def rope_cos_sin(position, head_dim, rope_theta, device):
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    freqs = position * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_qwen_rotate_half_rope(x, position, head_dim, rope_theta):
    cos, sin = rope_cos_sin(position, head_dim, rope_theta, x.device)
    return (x.float() * cos) + (rotate_half(x.float()) * sin)


matQ = torch.rand(NUM_REQ, HIDDEN_SIZE, dtype=torch.bfloat16, device=gpu) - 0.5
matK = torch.zeros(NUM_REQ * KV_SEQ_LEN, NUM_KV_HEAD * HEAD_DIM, dtype=torch.bfloat16, device=gpu)
matV = torch.zeros(NUM_REQ * KV_SEQ_LEN, NUM_KV_HEAD * HEAD_DIM, dtype=torch.bfloat16, device=gpu)
matO = torch.zeros(NUM_REQ, HIDDEN_SIZE, dtype=torch.bfloat16, device=gpu)

q_norm_weight = torch.rand(HEAD_DIM, dtype=torch.bfloat16, device=gpu) + 0.5
k_norm_weight = torch.rand(HEAD_DIM, dtype=torch.bfloat16, device=gpu) + 0.5
rope_row = torch.empty(HEAD_DIM, dtype=torch.bfloat16, device=gpu)
cos, sin = rope_cos_sin(TOKEN_POS, HEAD_DIM, ROPE_THETA, gpu)
rope_row[: HEAD_DIM // 2] = cos[: HEAD_DIM // 2].to(dtype=torch.bfloat16)
rope_row[HEAD_DIM // 2 :] = sin[: HEAD_DIM // 2].to(dtype=torch.bfloat16)
matSideInput = torch.cat((q_norm_weight, k_norm_weight, rope_row)).contiguous()

current_k = torch.rand(NUM_REQ, NUM_KV_HEAD, HEAD_DIM, dtype=torch.bfloat16, device=gpu) - 0.5
current_v = torch.rand(NUM_REQ, NUM_KV_HEAD, HEAD_DIM, dtype=torch.bfloat16, device=gpu) - 0.5
matK.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)[:, 0] = current_k
matV.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)[:, 0] = current_v

# interleaved QKV
matQ_attn_view = matQ.view(NUM_REQ, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)
matK_attn_view = matK.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)
matV_attn_view = matV.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)
matO_attn_view = matO.view(NUM_REQ, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)

tma_builder_K = partial(build_tma_wgmma_k, iN=-3)
cord_func_K = partial(cord_func_K_major, iN=-3)
tma_builder_MN = partial(build_tma_wgmma_mn, iK=-3)
cord_func_MN = partial(cord_func_MN_major, iK=-3)


def tma_load_q(mat: torch.Tensor, tileK: int, tileN: int):
    assert mat.element_size() == 2
    assert tileK == 128 and tileN == 64

    glob_dims = [64, 4, 16, 2, NUM_REQ * NUM_KV_HEAD]
    glob_strides = [128 * 2, 0, 64 * 2, HEAD_DIM * HEAD_GROUP_SIZE * 2]
    box_dims = [64, 4, 16, 2, 1]
    box_strides = [1] * len(glob_dims)

    return len(glob_dims), build_tma_desc(
        mat,
        glob_dims,
        glob_strides,
        box_dims,
        box_strides,
        128,
        0,
    )


def cord_load_q(mat: torch.Tensor, rank: int):
    assert rank == 5

    def cfunc(*cords):
        assert len(cords) == 2
        return [0, 0, 0, cords[0] * NUM_KV_HEAD + cords[1]]

    return cfunc


tQ = TmaTensor(dae, matQ_attn_view)._build("load", HEAD_DIM, 64, tma_load_q, cord_load_q)
tK = TmaTensor(dae, matK_attn_view)._build("load", HEAD_DIM, KVTile, tma_builder_K, cord_func_K)
tV = TmaTensor(dae, matV_attn_view)._build("load", HEAD_DIM, KVTile, tma_builder_MN, cord_func_MN)
side_input = TmaLoad1D(matSideInput, bytes=matSideInput.numel() * matSideInput.element_size())

NUM_KV_BLOCK = 1
last_active_kv_len = 1


def sm_task(sm: int):
    head = sm % NUM_KV_HEAD
    req = sm // NUM_KV_HEAD
    return [
        ATTENTION_M64N64K16_F16_F32_64_64_hdim(
            NUM_KV_BLOCK,
            last_active_kv_len,
            need_norm=True,
            need_rope=True,
        ),
        side_input,
        tQ.cord(req, head),
        tK.cord(req, 0, head, 0),
        tV.cord(req, 0, head, 0),
        TmaStore1D(matO_attn_view[req, head, ...], numSlots=2),
    ]


dae.i(
    sm_task,
    TerminateC(),
    TerminateM(),
)

dae.launch()


def rms_affine(x, weight):
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return (x * torch.rsqrt(variance + 1.0e-6)) * weight.float()


def reference_attention():
    q = matQ.view(NUM_REQ, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM).float()
    k = matK.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM).permute(0, 2, 1, 3).float()
    v = matV.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM).permute(0, 2, 1, 3).float()

    q = rms_affine(q, q_norm_weight).reshape(NUM_REQ, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)
    q = apply_qwen_rotate_half_rope(q, TOKEN_POS, HEAD_DIM, ROPE_THETA)

    k = rms_affine(k, k_norm_weight).reshape(NUM_REQ, NUM_KV_HEAD, KV_SEQ_LEN, HEAD_DIM)
    k_active = k[:, :, :last_active_kv_len]
    k[:, :, :last_active_kv_len] = apply_qwen_rotate_half_rope(k_active, TOKEN_POS, HEAD_DIM, ROPE_THETA)

    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(HEAD_DIM)
    mask = torch.arange(KV_SEQ_LEN, device=gpu)[None, None, None, :] >= last_active_kv_len
    scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v)
    return out.to(dtype=torch.bfloat16)


refO = reference_attention()
tensor_diff("Packed-side-input attention", refO[0], matO_attn_view[0])

dae_app(dae)
