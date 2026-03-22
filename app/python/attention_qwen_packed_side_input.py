import argparse
import math
import sys
from functools import partial

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-pos", type=int, default=7)
    parser.add_argument("--cache-mode", choices=("single", "empty-prefix"), default="single")
    parser.add_argument("--rope-mode", choices=("qwen", "llama-interleaved"), default="qwen")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable-norm", action="store_true")
    parser.add_argument("--disable-rope", action="store_true")
    parser.add_argument("--active-len", type=int, default=None)
    parser.add_argument("--prefix-fill", choices=("zeros", "random"), default="zeros")
    parser.add_argument("--reference-no-scale", action="store_true")
    parsed_args, remaining_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining_argv]
    return parsed_args


args = parse_args()

from dae.launcher import *
from dae.runtime import build_tma_desc
from dae.util import dae_app, tensor_diff
from qwen3.utils import *

gpu = torch.device("cuda")
torch.manual_seed(args.seed)

KV_SEQ_LEN = 64
HEAD_DIM = 128
HIDDEN_SIZE = 4096
NUM_REQ = 1
NUM_Q_HEAD = 32
NUM_KV_HEAD = 8
HEAD_GROUP_SIZE = NUM_Q_HEAD // NUM_KV_HEAD
TOKEN_POS = args.token_pos
ROPE_THETA = 1_000_000.0

assert HIDDEN_SIZE == NUM_KV_HEAD * HEAD_GROUP_SIZE * HEAD_DIM
assert 0 <= TOKEN_POS < KV_SEQ_LEN, f"--token-pos must be in [0, {KV_SEQ_LEN - 1}] for this harness"

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


def apply_interleaved_rope(x, position, head_dim, rope_theta):
    cos_half, sin_half = rope_cos_sin(position, head_dim, rope_theta, x.device)
    cos = cos_half[: head_dim // 2]
    sin = sin_half[: head_dim // 2]
    even = x.float()[..., 0::2]
    odd = x.float()[..., 1::2]
    return torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2)


def interleave_channels_to_half_split(x):
    return torch.cat((x[..., 0::2], x[..., 1::2]), dim=-1).contiguous()


def mean_relative_diff_pct(t1, t2, ref=None):
    if ref is None:
        ref = t1
    denom = ref.abs().float().mean().item()
    if denom == 0:
        denom = 1.0
    return (t1 - t2).abs().float().mean().item() / denom * 100.0


q_source = torch.rand(NUM_REQ, HIDDEN_SIZE, dtype=torch.bfloat16, device=gpu) - 0.5
k_source = torch.rand(NUM_REQ, NUM_KV_HEAD, HEAD_DIM, dtype=torch.bfloat16, device=gpu) - 0.5
v_source = torch.rand(NUM_REQ, NUM_KV_HEAD, HEAD_DIM, dtype=torch.bfloat16, device=gpu) - 0.5
prefix_k_source = torch.rand(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM, dtype=torch.bfloat16, device=gpu) - 0.5
prefix_v_source = torch.rand(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM, dtype=torch.bfloat16, device=gpu) - 0.5
q_norm_weight_source = torch.rand(HEAD_DIM, dtype=torch.bfloat16, device=gpu) + 0.5
k_norm_weight_source = torch.rand(HEAD_DIM, dtype=torch.bfloat16, device=gpu) + 0.5

if args.rope_mode == "llama-interleaved":
    matQ = interleave_channels_to_half_split(q_source.view(NUM_REQ, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)).view(
        NUM_REQ, HIDDEN_SIZE
    )
    current_k = interleave_channels_to_half_split(k_source)
    q_norm_weight = interleave_channels_to_half_split(q_norm_weight_source.view(1, 1, HEAD_DIM)).view(HEAD_DIM)
    k_norm_weight = interleave_channels_to_half_split(k_norm_weight_source.view(1, 1, HEAD_DIM)).view(HEAD_DIM)
else:
    matQ = q_source.clone()
    current_k = k_source.clone()
    q_norm_weight = q_norm_weight_source.clone()
    k_norm_weight = k_norm_weight_source.clone()
current_v = v_source.clone()

matK = torch.zeros(NUM_REQ * KV_SEQ_LEN, NUM_KV_HEAD * HEAD_DIM, dtype=torch.bfloat16, device=gpu)
matV = torch.zeros(NUM_REQ * KV_SEQ_LEN, NUM_KV_HEAD * HEAD_DIM, dtype=torch.bfloat16, device=gpu)
matO = torch.zeros(NUM_REQ, HIDDEN_SIZE, dtype=torch.bfloat16, device=gpu)

active_index = 0 if args.cache_mode == "single" else TOKEN_POS
last_active_kv_len = 1 if args.cache_mode == "single" else TOKEN_POS + 1
if args.active_len is not None:
    if args.active_len <= 0 or args.active_len > KV_SEQ_LEN:
        raise ValueError(f"--active-len must be in [1, {KV_SEQ_LEN}]")
    if active_index >= args.active_len:
        raise ValueError("--active-len must be greater than active_index")
    last_active_kv_len = args.active_len

rope_row = torch.empty(HEAD_DIM, dtype=torch.bfloat16, device=gpu)
cos, sin = rope_cos_sin(TOKEN_POS, HEAD_DIM, ROPE_THETA, gpu)
rope_row[: HEAD_DIM // 2] = cos[: HEAD_DIM // 2].to(dtype=torch.bfloat16)
rope_row[HEAD_DIM // 2 :] = sin[: HEAD_DIM // 2].to(dtype=torch.bfloat16)
side_inputs = []
if not args.disable_norm:
    side_inputs.extend((q_norm_weight, k_norm_weight))
if not args.disable_rope:
    side_inputs.append(rope_row)
matSideInput = torch.cat(side_inputs).contiguous() if side_inputs else None

matK.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)[:, active_index] = current_k
matV.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)[:, active_index] = current_v
if args.cache_mode == "empty-prefix" and args.prefix_fill == "random" and active_index > 0:
    matK.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)[:, :active_index] = prefix_k_source[:, :active_index]
    matV.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)[:, :active_index] = prefix_v_source[:, :active_index]

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
side_input = None
if matSideInput is not None:
    side_input = TmaLoad1D(matSideInput, bytes=matSideInput.numel() * matSideInput.element_size())

NUM_KV_BLOCK = 1


def sm_task(sm: int):
    head = sm % NUM_KV_HEAD
    req = sm // NUM_KV_HEAD
    insts = [
        ATTENTION_M64N64K16_F16_F32_64_64_hdim(
            NUM_KV_BLOCK,
            last_active_kv_len,
            need_norm=not args.disable_norm,
            need_rope=not args.disable_rope,
        ),
    ]
    if side_input is not None:
        insts.append(side_input)
    insts.extend(
        [
            tQ.cord(req, head),
            tK.cord(req, 0, head, 0),
            tV.cord(req, 0, head, 0),
            TmaStore1D(matO_attn_view[req, head, ...], numSlots=2),
        ]
    )
    return insts


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


def apply_selected_rope(x, position):
    if args.rope_mode == "llama-interleaved":
        return apply_interleaved_rope(x, position, HEAD_DIM, ROPE_THETA)
    return apply_qwen_rotate_half_rope(x, position, HEAD_DIM, ROPE_THETA)


def reference_attention(cache_mode, override_active_index=None, override_active_len=None):
    q = q_source.view(NUM_REQ, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM).float()
    k = torch.zeros(NUM_REQ, NUM_KV_HEAD, KV_SEQ_LEN, HEAD_DIM, dtype=torch.float32, device=gpu)
    v = torch.zeros(NUM_REQ, NUM_KV_HEAD, KV_SEQ_LEN, HEAD_DIM, dtype=torch.float32, device=gpu)

    if cache_mode == "single":
        active_slot = 0 if override_active_index is None else override_active_index
        k[:, :, active_slot] = k_source.float()
        v[:, :, active_slot] = v_source.float()
        active_len = 1 if override_active_len is None else override_active_len
    else:
        active_slot = TOKEN_POS if override_active_index is None else override_active_index
        if args.prefix_fill == "random" and active_slot > 0:
            k[:, :, :active_slot] = prefix_k_source[:, :active_slot].permute(0, 2, 1, 3).float()
            v[:, :, :active_slot] = prefix_v_source[:, :active_slot].permute(0, 2, 1, 3).float()
        k[:, :, active_slot] = k_source.float()
        v[:, :, active_slot] = v_source.float()
        active_len = TOKEN_POS + 1 if override_active_len is None else override_active_len

    if not args.disable_norm:
        q = rms_affine(q, q_norm_weight_source)
    if not args.disable_rope:
        q = apply_selected_rope(q, TOKEN_POS)
    q = q.reshape(NUM_REQ, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)

    if not args.disable_norm:
        k = rms_affine(k, k_norm_weight_source)
    k = k.reshape(NUM_REQ, NUM_KV_HEAD, KV_SEQ_LEN, HEAD_DIM)
    if not args.disable_rope:
        k[:, :, :active_len] = apply_selected_rope(k[:, :, :active_len], TOKEN_POS)

    scores = torch.matmul(q, k.transpose(-1, -2))
    if not args.reference_no_scale:
        scores = scores / math.sqrt(HEAD_DIM)
    mask = torch.arange(KV_SEQ_LEN, device=gpu)[None, None, None, :] >= active_len
    scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v)
    return out.to(dtype=torch.bfloat16)


ref_exact = reference_attention(args.cache_mode, override_active_len=last_active_kv_len)
tensor_diff("Kernel vs exact reference", ref_exact[0], matO_attn_view[0])

if args.cache_mode == "empty-prefix":
    ref_hf_single = reference_attention("single")
    hf_style_diff = mean_relative_diff_pct(ref_hf_single[0], matO_attn_view[0], ref=ref_hf_single[0])
    print(
        f"[debug] empty-prefix vs single-token reference at token_pos={TOKEN_POS}, "
        f"rope_mode={args.rope_mode}: {hf_style_diff:.3f}%"
    )
    tensor_diff("Empty-prefix vs single-token reference", ref_hf_single[0], matO_attn_view[0], ref=ref_hf_single[0])
    candidate_diffs = []
    for candidate_idx in range(last_active_kv_len):
        ref_candidate = reference_attention(
            "empty-prefix",
            override_active_index=candidate_idx,
            override_active_len=last_active_kv_len,
        )
        candidate_diffs.append(
            (
                candidate_idx,
                mean_relative_diff_pct(ref_candidate[0], matO_attn_view[0], ref=ref_candidate[0]),
            )
        )
    best_candidate_idx, best_candidate_diff = min(candidate_diffs, key=lambda item: item[1])
    print(
        f"[debug] best matching active slot within active_len={last_active_kv_len}: "
        f"slot={best_candidate_idx}, diff={best_candidate_diff:.3f}%"
    )
    print(f"[debug] candidate slot diffs: {candidate_diffs}")

print(
    f"[debug] token_pos={TOKEN_POS} cache_mode={args.cache_mode} rope_mode={args.rope_mode} "
    f"prefix_fill={args.prefix_fill} "
    f"reference_no_scale={args.reference_no_scale} "
    f"need_norm={not args.disable_norm} need_rope={not args.disable_rope} "
    f"active_index={active_index} active_len={last_active_kv_len}"
)
dae_app(dae)
