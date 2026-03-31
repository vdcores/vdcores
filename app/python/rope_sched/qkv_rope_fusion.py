import argparse
import sys

import torch
import torch.nn.functional as F

from dae.launcher import *
from dae.schedule import *
from dae.tma_utils import (
    Major,
    ToAttnKVStoreCordAdapter,
    ToAttnVStoreCordAdapter,
    ToRopeTableCordAdapter,
    ToSplitMCordAdapter,
    cord_id,
    cord_load_tbl,
    tma_load_tbl,
    tma_store_attn_kv,
)
from dae.util import dae_app, tensor_diff


HIDDEN = 4096
HEAD_DIM = 128
NUM_Q_HEAD = 32
NUM_KV_HEAD = 8
N = 8
MAX_SEQ_LEN = 512
TOKEN_POS = 0
DTYPE = torch.bfloat16
FULL_SMS = 132


def parse_args():
    raw_argv = sys.argv[1:]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=N)
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    parser.add_argument("--head-dim", type=int, default=HEAD_DIM)
    parser.add_argument("--num-q-head", type=int, default=NUM_Q_HEAD)
    parser.add_argument("--num-kv-head", type=int, default=NUM_KV_HEAD)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--token-pos", type=int, default=TOKEN_POS)
    parser.add_argument("--rope-theta", type=float, default=500000.0)
    parsed_args, remaining_argv = parser.parse_known_args()
    if parsed_args.correctness and not any(arg in ("-l", "--launch", "-b", "--bench") for arg in remaining_argv):
        remaining_argv = [*remaining_argv, "--launch"]
    sys.argv = [sys.argv[0], *remaining_argv]
    return parsed_args


def permute_rope_weight(weight: torch.Tensor, num_heads: int, head_dim: int, hidden: int) -> torch.Tensor:
    return (
        weight.view(num_heads, 2, head_dim // 2, hidden)
        .transpose(1, 2)
        .reshape_as(weight)
        .contiguous()
    )


def permute_rope_activation(activation: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    return (
        activation.view(*activation.shape[:-1], num_heads, 2, head_dim // 2)
        .transpose(-2, -1)
        .reshape_as(activation)
        .contiguous()
    )


def build_rope_table(max_seq_len: int, batch: int, head_dim: int, rope_theta: float, device, dtype):
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    rope_2d = torch.empty(max_seq_len, head_dim, device=device, dtype=dtype)
    rope_2d[:, 0::2] = freqs.cos().to(dtype=dtype)
    rope_2d[:, 1::2] = freqs.sin().to(dtype=dtype)
    return rope_2d[:, None, :].repeat(1, batch, 1).contiguous()


def apply_interleaved_rope(hidden_states: torch.Tensor, rope_row: torch.Tensor) -> torch.Tensor:
    hidden_states_f32 = hidden_states.float()
    cos = rope_row[..., 0::2].float()
    sin = rope_row[..., 1::2].float()
    even = hidden_states_f32[..., 0::2]
    odd = hidden_states_f32[..., 1::2]
    rotated = torch.stack(
        (even * cos - odd * sin, even * sin + odd * cos),
        dim=-1,
    ).flatten(-2)
    return rotated.to(hidden_states.dtype)


def mean_relative_diff_pct(expected: torch.Tensor, actual: torch.Tensor) -> float:
    denom = expected.abs().float().mean().item()
    if denom == 0:
        denom = 1.0
    return (expected - actual).abs().float().mean().item() / denom * 100.0


def run_correctness_check(
    mat_rms_hidden: torch.Tensor,
    mat_q_w: torch.Tensor,
    mat_k_w: torch.Tensor,
    mat_v_w: torch.Tensor,
    mat_rope: torch.Tensor,
    mat_q_out: torch.Tensor,
    mat_k_out: torch.Tensor,
    mat_v_out: torch.Tensor,
    num_q_head: int,
    num_kv_head: int,
    head_dim: int,
    token_pos: int,
):
    q_proj = F.linear(mat_rms_hidden, mat_q_w)
    k_proj = F.linear(mat_rms_hidden, mat_k_w)
    v_proj = F.linear(mat_rms_hidden, mat_v_w)

    q_perm = permute_rope_activation(q_proj, num_q_head, head_dim)
    k_perm = permute_rope_activation(k_proj, num_kv_head, head_dim)
    rope_row = mat_rope[token_pos]

    q_ref = apply_interleaved_rope(
        q_perm.view(mat_rms_hidden.shape[0], num_q_head, head_dim),
        rope_row[:, None, :],
    ).reshape_as(q_proj)
    k_ref = apply_interleaved_rope(
        k_perm.view(mat_rms_hidden.shape[0], num_kv_head, head_dim),
        rope_row[:, None, :],
    ).reshape_as(k_proj)

    q_diff = mean_relative_diff_pct(q_ref, mat_q_out)
    k_diff = mean_relative_diff_pct(k_ref, mat_k_out[:, token_pos])
    v_diff = mean_relative_diff_pct(v_proj, mat_v_out[:, token_pos])

    tensor_diff("q_rope", q_ref, mat_q_out)
    tensor_diff("k_rope", k_ref, mat_k_out[:, token_pos])
    tensor_diff("v_proj", v_proj, mat_v_out[:, token_pos])
    if max(q_diff, k_diff, v_diff) > 5.0:
        raise RuntimeError(
            f"Correctness check failed: q={q_diff:.3f}% k={k_diff:.3f}% v={v_diff:.3f}%"
        )
    print("[correctness] all checks passed")


parsed_args = parse_args()
torch.manual_seed(parsed_args.seed)

if parsed_args.hidden != parsed_args.num_q_head * parsed_args.head_dim:
    raise ValueError("--hidden must equal --num-q-head * --head-dim")
if parsed_args.hidden % 64 != 0:
    raise ValueError("--hidden must be divisible by 64")
if parsed_args.head_dim % 64 != 0:
    raise ValueError("--head-dim must be divisible by 64")
if parsed_args.token_pos < 0 or parsed_args.token_pos >= parsed_args.max_seq_len:
    raise ValueError("--token-pos must be in [0, max_seq_len)")

gpu = torch.device("cuda")
dae = Launcher(FULL_SMS, device=gpu)

qw = parsed_args.num_q_head * parsed_args.head_dim
kw = parsed_args.num_kv_head * parsed_args.head_dim
vw = kw

mat_rms_hidden = torch.rand(parsed_args.tokens, parsed_args.hidden, dtype=DTYPE, device=gpu) - 0.5
mat_q_out = torch.zeros(parsed_args.tokens, qw, dtype=DTYPE, device=gpu)
mat_k_out = torch.zeros(parsed_args.tokens, parsed_args.max_seq_len, kw, dtype=DTYPE, device=gpu)
mat_v_out = torch.zeros(parsed_args.tokens, parsed_args.max_seq_len, vw, dtype=DTYPE, device=gpu)

mat_q_w_raw = torch.rand(qw, parsed_args.hidden, dtype=DTYPE, device=gpu) - 0.5
mat_k_w_raw = torch.rand(kw, parsed_args.hidden, dtype=DTYPE, device=gpu) - 0.5
mat_v_w = torch.rand(vw, parsed_args.hidden, dtype=DTYPE, device=gpu) - 0.5
mat_q_w = permute_rope_weight(mat_q_w_raw, parsed_args.num_q_head, parsed_args.head_dim, parsed_args.hidden)
mat_k_w = permute_rope_weight(mat_k_w_raw, parsed_args.num_kv_head, parsed_args.head_dim, parsed_args.hidden)
mat_rope = build_rope_table(
    parsed_args.max_seq_len,
    parsed_args.tokens,
    parsed_args.head_dim,
    parsed_args.rope_theta,
    gpu,
    DTYPE,
)

dae.set_streaming(mat_q_w, mat_k_w, mat_v_w)

tile_m, _, tile_k = Gemv_M64N8.MNK
t_load_rms = TmaTensor(dae, mat_rms_hidden).wgmma_load(parsed_args.tokens, tile_k * Gemv_M64N8.n_batch, Major.K)
t_load_rope = TmaTensor(dae, mat_rope)._build("load", tile_m, parsed_args.tokens, tma_load_tbl, cord_load_tbl)
t_load_qw = TmaTensor(dae, mat_q_w).wgmma_load(tile_m, tile_k, Major.K)
t_load_kw = TmaTensor(dae, mat_k_w).wgmma_load(tile_m, tile_k, Major.K)
t_load_vw = TmaTensor(dae, mat_v_w).wgmma_load(tile_m, tile_k, Major.K)
t_store_q = TmaTensor(dae, mat_q_out).wgmma("reduce", parsed_args.tokens, tile_m, Major.MN)
t_store_k = TmaTensor(dae, mat_k_out)._build("reduce", 64, parsed_args.tokens, tma_store_attn_kv, cord_id)
t_store_v = TmaTensor(dae, mat_v_out)._build("reduce", 64, parsed_args.tokens, tma_store_attn_kv, cord_id)

reg_store_q = RegStore(0, size=parsed_args.tokens * tile_m * mat_q_out.element_size())
reg_load_q = RegLoad(0)
reg_store_k = RegStore(0, size=parsed_args.tokens * tile_m * mat_k_out.element_size())
reg_load_k = RegLoad(0)

q_proj = SchedGemv(
    Gemv_M64N8,
    MNK=(qw, parsed_args.tokens, parsed_args.hidden),
    tmas=(t_load_qw, t_load_rms, reg_store_q),
)
q_rope = SchedRope(
    ROPE_INTERLEAVE_512,
    tmas=(
        ToRopeTableCordAdapter(t_load_rope, parsed_args.token_pos, tile_repeats=max(1, parsed_args.head_dim // 64)),
        reg_load_q,
        ToSplitMCordAdapter(t_store_q, qw // tile_m, tile_m),
    ),
)

k_proj = SchedGemv(
    Gemv_M64N8,
    MNK=(kw, parsed_args.tokens, parsed_args.hidden),
    tmas=(t_load_kw, t_load_rms, reg_store_k),
)
k_rope = SchedRope(
    ROPE_INTERLEAVE_512,
    tmas=(
        ToRopeTableCordAdapter(t_load_rope, parsed_args.token_pos, tile_repeats=max(1, parsed_args.head_dim // 64)),
        reg_load_k,
        ToAttnKVStoreCordAdapter(t_store_k, kw // tile_m, tile_m, parsed_args.token_pos),
    ),
)

v_proj = SchedGemv(
    Gemv_M64N8,
    MNK=(vw, parsed_args.tokens, parsed_args.hidden),
    tmas=(
        t_load_vw,
        t_load_rms,
        ToAttnVStoreCordAdapter(t_store_v, parsed_args.token_pos),
    ),
)

q_proj = q_proj.place(128)
q_rope = q_rope.place(128)
k_proj = k_proj.place(64, base_sm=64)
k_rope = k_rope.place(64, base_sm=64)
v_proj = v_proj.place(64)

dae.s(
    q_proj,
    q_rope,
    k_proj,
    k_rope,
    v_proj,
)

print(
    "run rope qkv harness "
    f"hidden={parsed_args.hidden} qw={qw} kw={kw} tokens={parsed_args.tokens} token_pos={parsed_args.token_pos}..."
)
dae_app(dae)
if parsed_args.correctness:
    run_correctness_check(
        mat_rms_hidden,
        mat_q_w_raw,
        mat_k_w_raw,
        mat_v_w,
        mat_rope,
        mat_q_out,
        mat_k_out,
        mat_v_out,
        parsed_args.num_q_head,
        parsed_args.num_kv_head,
        parsed_args.head_dim,
        parsed_args.token_pos,
    )
