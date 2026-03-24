import torch
import copy
from math import sqrt
from functools import partial
from dataclasses import dataclass
from dae.launcher import *
from dae.util import *
from dae.runtime import opcode, build_tma_desc 
from qwen3.utils import *

gpu = torch.device("cuda")
torch.manual_seed(0)

KV_SEQ_LEN = 2048
HEAD_DIM = 128
HIDDEN_SIZE = 4096
NUM_REQ = 2
NUM_Q_HEAD = 32
NUM_KV_HEAD = 8
HEAD_GROUP_SIZE = NUM_Q_HEAD // NUM_KV_HEAD
MAX_SPLIT = 16
seq_lengths = [2048, 1024]

assert HIDDEN_SIZE == NUM_KV_HEAD * HEAD_GROUP_SIZE * HEAD_DIM, "Q size must match HIDDEN SIZE"
assert len(seq_lengths) == NUM_REQ, "Length of seq_lengths must match NUM_REQ"
for seq_len in seq_lengths:
    assert seq_len <= KV_SEQ_LEN, "Sequence length must be less than or equal to KV_SEQ_LEN"

QTile = 16
KVTile = 64

num_sms = 132

dae = Launcher(num_sms, device=gpu)

matQ = torch.rand(NUM_REQ, HIDDEN_SIZE, dtype=torch.bfloat16, device=gpu) - 0.5
matK = torch.rand(NUM_REQ * KV_SEQ_LEN, NUM_KV_HEAD * HEAD_DIM, dtype=torch.bfloat16, device=gpu) - 0.5
matV = torch.rand(NUM_REQ * KV_SEQ_LEN, NUM_KV_HEAD * HEAD_DIM, dtype=torch.bfloat16, device=gpu) - 0.5
matO = torch.zeros(NUM_REQ, HIDDEN_SIZE, dtype=torch.bfloat16, device=gpu)
matO_split = torch.zeros(MAX_SPLIT, NUM_REQ, HIDDEN_SIZE, dtype=torch.bfloat16, device=gpu)
matP = torch.zeros(NUM_KV_HEAD, NUM_REQ * HEAD_GROUP_SIZE, MAX_SPLIT, dtype=torch.float, device=gpu)

# interleaved QKV
matQ_attn_view = matQ.view(NUM_REQ, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)
matK_attn_view = matK.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)
matV_attn_view = matV.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)
matO_attn_view = matO.view(NUM_REQ, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)
matO_split_attn_view = matO_split.view(MAX_SPLIT, NUM_REQ, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)

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

def tma_load_split_attn(mat: torch.Tensor, tileS, tileO):
    assert tileS == split_kv
    assert tileO == HEAD_GROUP_SIZE * HEAD_DIM
    S, R, H, G, D = mat.shape
    permute = [4, 3, 0, 2, 1] # [D, G, S, H, R]
    glob_dims = [mat.shape[i] for i in permute]
    glob_strides = [mat.stride(i) * mat.element_size() for i in permute[1:]]
    box_dims = [D, G, split_kv, 1, 1]
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
        assert len(cords) == 2, f"cords should be (head, req), but got {cords}"
        return [0, 0, cords[0], cords[1]]
    return cfunc

tQ = TmaTensor(dae, matQ_attn_view)._build("load", HEAD_DIM, 64, tma_load_o, cord_load_o)
tK = TmaTensor(dae, matK_attn_view)._build("load", HEAD_DIM, KVTile, tma_builder_K, cord_func_K)
tV = TmaTensor(dae, matV_attn_view)._build("load", HEAD_DIM, KVTile, tma_builder_MN, cord_func_MN)

need_norm = False
need_rope = False

last_active_kv_len = 64
assert last_active_kv_len <= KVTile

@dataclass 
class SchedPlan:
    request_idx: int
    split_level: int
    attn_num_sms: int
    attn_base_sm: int
    post_num_sms: int
    post_base_sm: int

    def __post_init__(self):
        self.seq_length = seq_lengths[self.request_idx]
        self.num_kv_block = (self.seq_length + KVTile - 1) // KVTile
        self.num_block_per_split = self.num_kv_block // self.split_level
        self.attn_bar = dae.new_bar(NUM_KV_HEAD * self.split_level)
        self.tO_split = TmaTensor(dae, matO_split_attn_view)._build("load", self.split_level, HEAD_GROUP_SIZE*HEAD_DIM, tma_load_split_attn, cord_load_split_attn)

    def sm_attn_task(self, sm: int):
        sm -= self.attn_base_sm
        if sm < 0 or sm >= self.attn_num_sms:
            return []
        insts = []
        for i in range(sm, self.split_level * NUM_KV_HEAD, self.attn_num_sms):
            head = i % NUM_KV_HEAD
            split = i // NUM_KV_HEAD
            kv_start_block = split * self.num_block_per_split
            kv_start_idx = kv_start_block * KVTile
            split_last_active_kv_len = last_active_kv_len if split == self.split_level - 1 else KVTile
            insts.extend([
                ATTENTION_M64N64K16_F16_F32_64_64_hdim_split(self.num_block_per_split, split, HEAD_GROUP_SIZE, split_last_active_kv_len, kv_start_idx, need_norm=need_norm, need_rope=need_rope),
                tQ.cord(self.request_idx, head),
                RepeatM.on(self.num_block_per_split,
                    [tK.cord(self.request_idx, kv_start_idx, head, 0), tK.cord2tma(0, KVTile, 0, 0)],
                    [tV.cord(self.request_idx, kv_start_idx, head, 0), tV.cord2tma(0, KVTile, 0, 0)],
                ),
                # here we override the allocator, to allocate enough space in the smem
                # but we will only write back the first 128*16*2 bytes to the output mat
                TmaStore1D(matO_split_attn_view[split, self.request_idx, head, ...]),
                RawAddress(matP[head, self.request_idx * HEAD_GROUP_SIZE], 24).bar(self.attn_bar).writeback(),
            ])
        return insts

    def sm_post_task(self, sm: int):
        sm -= self.post_base_sm
        if sm < 0 or sm >= self.post_num_sms:
            return []
        insts = []
        for i in range(sm, NUM_KV_HEAD, self.post_num_sms):
            head = i % NUM_KV_HEAD
            insts.extend([
                ATTN_SPLIT_POST_REDUCE(self.split_level),
                RawAddress(matP[head], 25).bar(self.attn_bar),
                # RepeatM(self.split_level, delta_addr=matO.numel() * matO.element_size()),
                # TmaLoad1D(matO_split_attn_view[0, self.request_idx, head, ...]).jump(),
                self.tO_split.cord(head, self.request_idx),
                TmaStore1D(matO_attn_view[self.request_idx, head, ...]),
            ])
        return insts


split_kv = 2
assert split_kv <= MAX_SPLIT

attn_base_sm = 0
post_base_sm = 0
plans = []
for rid in range(NUM_REQ):
    plan = SchedPlan(
        request_idx=rid,
        split_level=split_kv,
        attn_num_sms=NUM_KV_HEAD * split_kv,
        attn_base_sm=attn_base_sm,
        post_num_sms=NUM_KV_HEAD,
        post_base_sm=post_base_sm,
    )
    plans.append(plan)
    attn_base_sm += plan.attn_num_sms
    post_base_sm += plan.post_num_sms

dae.i(
    [plan.sm_attn_task for plan in plans],
    [plan.sm_post_task for plan in plans],

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
    # Apply a per-request causal length mask so each request can expose a different
    # active KV span while sharing the same backing K/V buffers.
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
    num_block_per_split = NUM_KV_BLOCK // split_kv
    kv_start = split_stage * num_block_per_split * KVTile
    kv_end   = kv_start + num_block_per_split * KVTile
    split_last_active = last_active_kv_len if split_stage == split_kv - 1 else KVTile
    total_active = (num_block_per_split - 1) * KVTile + split_last_active

    Q = matQ.view(NUM_REQ, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)
    K = matK.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM).permute(0, 2, 1, 3)
    V = matV.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM).permute(0, 2, 1, 3)

    K_split = K[:, :, kv_start:kv_end, :]   # [B, Hkv, S_split, D]
    V_split = V[:, :, kv_start:kv_end, :]

    scale = 1.0 / sqrt(HEAD_DIM)
    QK = torch.matmul(Q * scale, K_split.transpose(-1, -2))  # [B, Hkv, G, S_split]

    # mask tokens beyond the active length in this split's last block
    S_split = kv_end - kv_start
    mask = torch.arange(S_split, device=gpu)[None, None, None, :] >= total_active
    QK = QK.masked_fill(mask, float("-inf"))

    lse  = torch.logsumexp(QK, dim=-1)    # [B, Hkv, G]
    attn = torch.softmax(QK, dim=-1)      # [B, Hkv, G, S_split]
    O    = torch.matmul(attn, V_split)    # [B, Hkv, G, D]

    return O.bfloat16(), lse


# for s in range(split_kv):
#     ref_O, ref_lse = split_ref(s)
#     # matO_split_attn_view: [split_kv, NUM_REQ, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM]
#     tensor_diff(f"Split {s} O", ref_O[0], matO_split_attn_view[s, 0])
#     # matP: [NUM_KV_HEAD, NUM_REQ * HEAD_GROUP_SIZE, split_kv], written as float32 by kernel
#     ref_lse_hkv = ref_lse[0]  # [Hkv, G]
#     tensor_diff(f"Split {s} LSE", ref_lse_hkv, matP[:, :HEAD_GROUP_SIZE, s].float())

refQK, refO = gqa_ref()
tensor_diff("Ref and DAE", refO, matO_attn_view)
