import torch
import os
from math import sqrt
from functools import partial
from dae import runtime
from dae.launcher import *
from dae.util import *
from qwen3.utils import *

gpu = torch.device("cuda")
torch.manual_seed(0)

ACTIVE_KV_SEQ_LEN = int(os.environ.get("ATTENTION_SEQ_LEN", "48"))
if ACTIVE_KV_SEQ_LEN <= 0:
    raise ValueError("ATTENTION_SEQ_LEN must be positive")
KVTile = int(os.environ.get("ATTENTION_KV_TILE", "64"))
if KVTile not in {64, 128}:
    raise ValueError("ATTENTION_KV_TILE must be 64 or 128")
KV_SEQ_LEN = ((ACTIVE_KV_SEQ_LEN + KVTile - 1) // KVTile) * KVTile
HEAD_DIM = 128
HIDDEN_SIZE = 4096
NUM_REQ = int(os.environ.get("ATTENTION_BATCH", "1"))
if NUM_REQ <= 0:
    raise ValueError("ATTENTION_BATCH must be positive")
NUM_Q_HEAD = 32
NUM_KV_HEAD = 8
HEAD_GROUP_SIZE = NUM_Q_HEAD // NUM_KV_HEAD

assert HIDDEN_SIZE == NUM_KV_HEAD * HEAD_GROUP_SIZE * HEAD_DIM, "Q size must match HIDDEN SIZE"

ATTENTION_TOPOLOGY = os.environ.get("ATTENTION_TOPOLOGY", "head").strip().lower()
if ATTENTION_TOPOLOGY not in {"head", "gqa"}:
    raise ValueError("ATTENTION_TOPOLOGY must be 'head' or 'gqa'")
compute_heads = NUM_Q_HEAD if ATTENTION_TOPOLOGY == "head" else NUM_KV_HEAD
num_sms = compute_heads * NUM_REQ
device_sms = torch.cuda.get_device_properties(gpu).multi_processor_count
assert num_sms <= device_sms, f"attention needs {num_sms} SMs, but the device has {device_sms}"

dae = Launcher(num_sms, device=gpu)

matQ = torch.rand(NUM_REQ, HIDDEN_SIZE, dtype=torch.bfloat16, device=gpu) - 0.5
matK = torch.rand(NUM_REQ, NUM_KV_HEAD, KV_SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16, device=gpu) - 0.5
matV = torch.rand(NUM_REQ, NUM_KV_HEAD, KV_SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16, device=gpu) - 0.5
matO = torch.zeros(NUM_REQ, HIDDEN_SIZE, dtype=torch.bfloat16, device=gpu)

# Q is head-interleaved; the KV cache is head-major so batches and heads are
# contiguous in the collapsed TMA dimension.
matQ_head_view = matQ.view(NUM_REQ, NUM_Q_HEAD, HEAD_DIM)
matQ_gqa_view = matQ.view(NUM_REQ, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)
matK_attn_view = matK
matV_attn_view = matV
matO_head_view = matO.view(NUM_REQ, NUM_Q_HEAD, HEAD_DIM)
matO_gqa_view = matO.view(NUM_REQ, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)
matO_attn_view = matO_head_view if ATTENTION_TOPOLOGY == "head" else matO_gqa_view

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

tma_builder_K = partial(build_tma_wgmma_k, iN = -2)
cord_func_K = partial(cord_func_K_major, iN=-2)

tma_builder_MN = partial(build_tma_wgmma_mn, iK = -2)
cord_func_MN = partial(cord_func_MN_major, iK=-2)

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


def tma_load_gqa_q(mat: torch.Tensor, tileK: int, tileN: int):
    assert mat.element_size() == 2, "Only support float16/bfloat16 Q"
    assert tileK == HEAD_DIM and tileN == 64
    glob_dims = [64, HEAD_GROUP_SIZE, 16, 2, NUM_REQ * NUM_KV_HEAD]
    glob_strides = [HEAD_DIM * 2, 0, 64 * 2, HEAD_DIM * HEAD_GROUP_SIZE * 2]
    box_dims = [64, HEAD_GROUP_SIZE, 16, 2, 1]
    rank = len(glob_dims)
    return rank, runtime.build_tma_desc(
        mat, glob_dims, glob_strides, box_dims, [1] * rank, 128, 0
    )


def cord_load_gqa_q(mat: torch.Tensor, rank: int):
    assert rank == 5
    def cfunc(*cords):
        assert len(cords) == 2, f"cords should be (req, kv_head), but got {cords}"
        return [0, 0, 0, cords[0] * NUM_KV_HEAD + cords[1]]
    return cfunc


if ATTENTION_TOPOLOGY == "head":
    tQ = TmaTensor(dae, matQ_head_view)._build("load", HEAD_DIM, 64, tma_load_o, cord_load_o)
else:
    tQ = TmaTensor(dae, matQ_gqa_view)._build("load", HEAD_DIM, 64, tma_load_gqa_q, cord_load_gqa_q)
tK = TmaTensor(dae, matK_attn_view)._build("load", HEAD_DIM, KVTile, tma_builder_K, cord_func_K)
tV = TmaTensor(dae, matV_attn_view)._build("load", HEAD_DIM, KVTile, tma_builder_MN, cord_func_MN)

need_norm = env_flag("ATTENTION_NEED_NORM", False)
need_rope = env_flag("ATTENTION_NEED_ROPE", False)
if need_norm != need_rope:
    raise ValueError("attention_simple_decoding.py mirrors the fused Qwen decode path, so norm and rope must be enabled together")

ATTENTION_IMPL = os.environ.get("ATTENTION_IMPL", "native").lower()
if ATTENTION_IMPL == "mma":
    attention_inst = ATTENTION_M64N64K16_F16_F32_64_64_hdim_MMA
elif ATTENTION_IMPL in ("native", "hopper", "gmma", ""):
    attention_inst = ATTENTION_M64N64K16_F16_F32_64_64_hdim
else:
    raise ValueError(f"Unsupported ATTENTION_IMPL={ATTENTION_IMPL!r}; expected 'native' or 'mma'")

NUM_KV_BLOCK = (KV_SEQ_LEN + KVTile - 1) // KVTile
last_active_kv_len = ACTIVE_KV_SEQ_LEN - (NUM_KV_BLOCK - 1) * KVTile
assert last_active_kv_len <= KVTile
total_active_kv_len = ACTIVE_KV_SEQ_LEN
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
    TmaTensor(dae, matK_attn_view[req]).tensor1d("store", HEAD_DIM)
    for req in range(NUM_REQ)
]

if need_norm and need_rope and token_pos > 0:
    matK_attn_view[:, :, :token_pos] = apply_rms_affine_rope_heads(
        matK_attn_view[:, :, :token_pos],
        k_norm_weight,
        rope_table[:token_pos].view(1, 1, token_pos, HEAD_DIM),
        eps=1.0e-6,
    )

def sm_task(sm: int):
    compute_head = sm % compute_heads
    req = sm // compute_heads
    if ATTENTION_TOPOLOGY == "head":
        kv_head = compute_head // HEAD_GROUP_SIZE
        q_cord = tQ.cord(req, compute_head)
        output = matO_head_view[req, compute_head, ...]
        num_active_q = 1
    else:
        kv_head = compute_head
        q_cord = tQ.cord(req, kv_head)
        output = matO_gqa_view[req, kv_head, ...]
        num_active_q = HEAD_GROUP_SIZE

    insts = [
        attention_inst(
            NUM_KV_BLOCK, num_active_q, last_active_kv_len,
            need_norm=need_norm, need_rope=need_rope,
            kv_block_size=KVTile,
        ),
        tSideInput.cord(token_pos * 3 * HEAD_DIM) if need_norm and need_rope else [],
        current_k_store[req].cord((kv_head * KV_SEQ_LEN + token_pos) * HEAD_DIM) if need_norm and need_rope else [],
        q_cord,
        RepeatM.on(NUM_KV_BLOCK - 1,
            [tK.cord(req, kv_head, 0, 0), tK.cord2tma(0, 0, KVTile, 0)],
            [tV.cord(req, kv_head, 0, 0), tV.cord2tma(0, 0, KVTile, 0)],
        ),
        tK.cord(req, kv_head, KVTile * (NUM_KV_BLOCK - 1), 0),
        tV.cord(req, kv_head, KVTile * (NUM_KV_BLOCK - 1), 0),
        # here we override the allocator, to allocate enough space in the smem
        # but we will only write back the first 128*16*2 bytes to the output mat
        TmaStore1D(output, numSlots=2)
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

    K = matK_attn_view     # [B, Hkv, S, D]
    V = matV_attn_view     # [B, Hkv, S, D]

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
refO = refO.view(matO_attn_view.shape)
tensor_diff("Ref and DAE", refO, matO_attn_view)
avg_diff_percent = (
    (refO - matO_attn_view).abs().float().mean()
    / refO.abs().float().mean()
    * 100.0
).item()
max_abs_diff = (refO - matO_attn_view).abs().float().max().item()
print(
    "ATTENTION_RESULT "
    f"impl=sm100-{ATTENTION_TOPOLOGY} batch={NUM_REQ} active_seq={ACTIVE_KV_SEQ_LEN} "
    f"kv_tile={KVTile} blocks={NUM_KV_BLOCK} "
    f"avg_diff_percent={avg_diff_percent:.6f} max_abs_diff={max_abs_diff:.6f}"
)
if env_flag("ATTENTION_STRICT", False):
    max_diff_percent = float(os.environ.get("ATTENTION_MAX_DIFF_PERCENT", "1.0"))
    if avg_diff_percent > max_diff_percent:
        raise AssertionError(
            f"attention average difference {avg_diff_percent:.6f}% exceeds "
            f"{max_diff_percent:.6f}%"
        )
