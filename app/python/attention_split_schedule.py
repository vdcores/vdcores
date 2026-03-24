import torch
import copy
from math import sqrt
from functools import partial
from dataclasses import dataclass
from collections import defaultdict
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

def tma_load_split_attn(mat: torch.Tensor, tileS, tileO, split_kv):
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

TOTAL_ATTN_GROUPS = num_sms // NUM_KV_HEAD


def legal_split_levels(num_kv_blocks: int):
    max_split = min(MAX_SPLIT, TOTAL_ATTN_GROUPS, num_kv_blocks)
    return [split for split in range(1, max_split + 1) if num_kv_blocks % split == 0]


def choose_split_level(num_kv_blocks: int, target_blocks_per_shard: float):
    legal = legal_split_levels(num_kv_blocks)
    for split in legal:
        if num_kv_blocks / split <= target_blocks_per_shard:
            return split
    return legal[-1]


@dataclass
class SchedPlan:
    request_idx: int
    split_level: int
    attn_groups: list[int]
    post_groups: list[int]

    def __post_init__(self):
        self.seq_length = seq_lengths[self.request_idx]
        self.num_kv_block = (self.seq_length + KVTile - 1) // KVTile
        self.num_block_per_split = self.num_kv_block // self.split_level
        assert len(self.attn_groups) == self.split_level, "Need one attention group per split shard"
        self.attn_group_to_split = {group_id: split_idx for split_idx, group_id in enumerate(self.attn_groups)}
        self.post_group_to_lane = {group_id: lane for lane, group_id in enumerate(self.post_groups)}
        self.attn_bar = dae.new_bar(NUM_KV_HEAD * self.split_level)
        self.tO_split = TmaTensor(dae, matO_split_attn_view)._build("load", self.split_level, HEAD_GROUP_SIZE*HEAD_DIM, partial(tma_load_split_attn, split_kv=self.split_level), cord_load_split_attn)

    def sm_attn_task(self, sm: int):
        sm_group = sm // NUM_KV_HEAD
        if sm_group not in self.attn_group_to_split:
            return []
        head = sm % NUM_KV_HEAD
        split = self.attn_group_to_split[sm_group]
        insts = []
        if self.split_level == 1:
            return [
                ATTENTION_M64N64K16_F16_F32_64_64_hdim(self.num_kv_block, last_active_kv_len, need_norm=need_norm, need_rope=need_rope),
                tQ.cord(self.request_idx, head),
                RepeatM.on(self.num_kv_block,
                    [tK.cord(self.request_idx, 0, head, 0), tK.cord2tma(0, KVTile, 0, 0)],
                    [tV.cord(self.request_idx, 0, head, 0), tV.cord2tma(0, KVTile, 0, 0)],
                ),
                TmaStore1D(matO_attn_view[self.request_idx, head, ...], numSlots=2),
            ]

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
            TmaStore1D(matO_split_attn_view[split, self.request_idx, head, ...], numSlots=2),
            RawAddress(matP[head, self.request_idx * HEAD_GROUP_SIZE], 24).bar(self.attn_bar).writeback(),
        ])
        return insts

    def sm_post_task(self, sm: int):
        if self.split_level == 1:
            return []
        sm_group = sm // NUM_KV_HEAD
        if sm_group not in self.post_group_to_lane:
            return []
        head = sm % NUM_KV_HEAD
        insts = []
        insts.extend([
            ATTN_SPLIT_POST_REDUCE(self.split_level),
            RawAddress(matP[head, self.request_idx * HEAD_GROUP_SIZE], 25).bar(self.attn_bar),
            self.tO_split.cord(head, self.request_idx),
            TmaStore1D(matO_attn_view[self.request_idx, head, ...]),
        ])
        return insts


def build_sched_plans():
    num_kv_blocks = [(seq_len + KVTile - 1) // KVTile for seq_len in seq_lengths]
    total_blocks = sum(num_kv_blocks)
    target_blocks_per_shard = total_blocks / TOTAL_ATTN_GROUPS * 1.2

    split_levels = [
        choose_split_level(num_blocks, target_blocks_per_shard)
        for num_blocks in num_kv_blocks
    ]

    attn_group_loads = [0] * TOTAL_ATTN_GROUPS
    attn_group_assignments = defaultdict(list)
    shard_specs = []
    for req_idx, (num_blocks, split_level) in enumerate(zip(num_kv_blocks, split_levels)):
        shard_cost = num_blocks // split_level
        for split_idx in range(split_level):
            shard_specs.append((shard_cost, req_idx, split_idx))

    shard_specs.sort(reverse=True)
    for shard_cost, req_idx, split_idx in shard_specs:
        group_id = min(range(TOTAL_ATTN_GROUPS), key=lambda gid: (attn_group_loads[gid], gid))
        attn_group_loads[group_id] += shard_cost
        attn_group_assignments[req_idx].append((split_idx, group_id))

    post_group_assignments = defaultdict(list)
    rr_group = 0
    for req_idx, split_level in enumerate(split_levels):
        if split_level == 1:
            continue
        post_group_assignments[req_idx].append(rr_group % TOTAL_ATTN_GROUPS)
        rr_group += 1

    plans = []
    for req_idx, split_level in enumerate(split_levels):
        attn_groups = [group_id for _, group_id in sorted(attn_group_assignments[req_idx])]
        post_groups = post_group_assignments[req_idx]
        plans.append(
            SchedPlan(
                request_idx=req_idx,
                split_level=split_level,
                attn_groups=attn_groups,
                post_groups=post_groups,
            )
        )
    return plans, split_levels, attn_group_loads


plans, split_levels, attn_group_loads = build_sched_plans()
print(f"[sched] seq_lengths={seq_lengths}")
print(f"[sched] split_levels={split_levels}, target_blocks_per_shard={sum((seq_len + KVTile - 1) // KVTile for seq_len in seq_lengths) / TOTAL_ATTN_GROUPS:.2f}")
print(f"[sched] attn_group_loads={attn_group_loads}")
for plan in plans:
    print(
        f"[sched] req={plan.request_idx} blocks={plan.num_kv_block} split={plan.split_level} "
        f"attn_groups={plan.attn_groups} post_groups={plan.post_groups}"
    )

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


def split_ref(plan: SchedPlan, split_stage: int):
    """Per-split reference: each split computes local softmax only over its own KV slice.
    Returns O_local = softmax_local(Q @ K_split^T / sqrt(D)) @ V_split  [B, Hkv, G, D]
    and     lse     = max_local + log(sum_local)                         [B, Hkv, G]
    """
    num_block_per_split = plan.num_block_per_split
    kv_start = split_stage * num_block_per_split * KVTile
    kv_end   = kv_start + num_block_per_split * KVTile
    split_last_active = last_active_kv_len if split_stage == plan.split_level - 1 else KVTile
    total_active = (num_block_per_split - 1) * KVTile + split_last_active

    Q = matQ.view(NUM_REQ, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)[plan.request_idx : plan.request_idx + 1]
    K = matK.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM).permute(0, 2, 1, 3)[plan.request_idx : plan.request_idx + 1]
    V = matV.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM).permute(0, 2, 1, 3)[plan.request_idx : plan.request_idx + 1]

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


# for plan in plans:
#     if plan.split_level == 1:
#         continue
#     for s in range(plan.split_level):
#         ref_O, ref_lse = split_ref(plan, s)
#         tensor_diff(f"Req {plan.request_idx} Split {s} O", ref_O[0], matO_split_attn_view[s, plan.request_idx])
#         start = plan.request_idx * HEAD_GROUP_SIZE
#         end = start + HEAD_GROUP_SIZE
#         tensor_diff(f"Req {plan.request_idx} Split {s} LSE", ref_lse[0], matP[:, start:end, s].float())

refQK, refO = gqa_ref()
tensor_diff("Ref and DAE", refO, matO_attn_view)
