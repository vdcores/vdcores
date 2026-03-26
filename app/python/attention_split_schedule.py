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

KV_SEQ_LEN = 32768
HEAD_DIM = 128
HIDDEN_SIZE = 4096
NUM_Q_HEAD = 32
NUM_KV_HEAD = 8
HEAD_GROUP_SIZE = NUM_Q_HEAD // NUM_KV_HEAD
MAX_SPLIT = 64
seq_lengths = [16384] + [1024] * 8
NUM_REQ = len(seq_lengths)

assert HIDDEN_SIZE == NUM_KV_HEAD * HEAD_GROUP_SIZE * HEAD_DIM, "Q size must match HIDDEN SIZE"
assert len(seq_lengths) == NUM_REQ, "Length of seq_lengths must match NUM_REQ"
for seq_len in seq_lengths:
    assert seq_len <= KV_SEQ_LEN, "Sequence length must be less than or equal to KV_SEQ_LEN"

QTile = 16
KVTile = 64

num_sms = 132
schedule_mode = 'vdc' # 'no_split', 'static_split', 'vdc'
static_split_token_size = 512

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
tK = TmaTensor(dae, matK_attn_view)._build("load", HEAD_DIM, KVTile, tma_load_k, cord_load_v)
tV = TmaTensor(dae, matV_attn_view)._build("load", HEAD_DIM, KVTile, tma_load_v, cord_load_v)

need_norm = False
need_rope = False

last_active_kv_len = 64
assert last_active_kv_len <= KVTile

TOTAL_ATTN_GROUPS = num_sms // NUM_KV_HEAD


def legal_split_levels(num_kv_blocks: int):
    max_split = min(MAX_SPLIT, num_kv_blocks)
    return [split for split in range(1, max_split + 1) if num_kv_blocks % split == 0]


def choose_split_level(num_kv_blocks: int, target_blocks_per_shard: float, mode: str = 'vdc'):
    if mode == 'no_split':
        return 1
    if mode == 'static_split':
        chunk_size = static_split_token_size // KVTile
        return max(1, min(MAX_SPLIT, num_kv_blocks // chunk_size))
    legal = legal_split_levels(num_kv_blocks)
    print(target_blocks_per_shard, num_kv_blocks, legal)
    for split in legal:
        if num_kv_blocks / split <= target_blocks_per_shard:
            return split
    return legal[-1]


def calc_group_metrics(group_loads: list[int], full_groups: int):
    total_work = sum(group_loads)
    max_work = max(group_loads, default=0)
    mean_work = total_work / full_groups if full_groups > 0 else 0.0
    imbalance = (max_work / mean_work) if mean_work > 0 else 0.0
    utilization = (total_work / (full_groups * max_work)) if max_work > 0 else 0.0
    return {
        "total_work": total_work,
        "max_work": max_work,
        "mean_work": mean_work,
        "imbalance": imbalance,
        "utilization": utilization,
    }


def next_finer_split(num_kv_blocks: int, current_split: int):
    legal = legal_split_levels(num_kv_blocks)
    idx = legal.index(current_split)
    if idx + 1 >= len(legal):
        return None
    return legal[idx + 1]


def place_split_levels(num_kv_blocks: list[int], split_levels: list[int]):
    attn_group_loads = [0] * TOTAL_ATTN_GROUPS
    attn_group_assignments = defaultdict(list)
    attn_group_waves = [0] * TOTAL_ATTN_GROUPS
    shard_specs = []
    for req_idx, (num_blocks, split_level) in enumerate(zip(num_kv_blocks, split_levels)):
        shard_cost = num_blocks // split_level
        for split_idx in range(split_level):
            shard_specs.append((shard_cost, req_idx, split_idx))

    shard_specs.sort(reverse=True)
    for shard_cost, req_idx, split_idx in shard_specs:
        group_id = min(range(TOTAL_ATTN_GROUPS), key=lambda gid: (attn_group_loads[gid], gid))
        wave_idx = attn_group_waves[group_id]
        attn_group_loads[group_id] += shard_cost
        attn_group_assignments[req_idx].append((split_idx, group_id, wave_idx))
        attn_group_waves[group_id] += 1

    post_group_assignments = defaultdict(list)
    post_group_waves = [0] * TOTAL_ATTN_GROUPS
    rr_group = 0
    for req_idx, split_level in enumerate(split_levels):
        if split_level == 1:
            continue
        group_id = rr_group % TOTAL_ATTN_GROUPS
        wave_idx = post_group_waves[group_id]
        post_group_assignments[req_idx].append((group_id, wave_idx))
        post_group_waves[group_id] += 1
        rr_group += 1

    return attn_group_assignments, post_group_assignments, attn_group_loads, attn_group_waves, post_group_waves


def refine_split_levels(num_kv_blocks: list[int], split_levels: list[int]):
    best_split_levels = split_levels[:]
    (
        best_attn_assignments,
        best_post_assignments,
        best_group_loads,
        best_attn_waves,
        best_post_waves,
    ) = place_split_levels(num_kv_blocks, best_split_levels)
    best_metrics = calc_group_metrics(best_group_loads, TOTAL_ATTN_GROUPS)

    while True:
        shard_sizes = [num_blocks // split for num_blocks, split in zip(num_kv_blocks, best_split_levels)]
        longest_shard = max(shard_sizes)
        candidate_reqs = [idx for idx, shard_size in enumerate(shard_sizes) if shard_size == longest_shard]

        improved = False
        chosen = None
        for req_idx in candidate_reqs:
            finer_split = next_finer_split(num_kv_blocks[req_idx], best_split_levels[req_idx])
            if finer_split is None:
                continue

            candidate_split_levels = best_split_levels[:]
            candidate_split_levels[req_idx] = finer_split
            try:
                candidate = place_split_levels(num_kv_blocks, candidate_split_levels)
            except ValueError:
                continue

            candidate_metrics = calc_group_metrics(candidate[2], TOTAL_ATTN_GROUPS)
            candidate_total_splits = sum(candidate_split_levels)
            best_total_splits = sum(best_split_levels)
            if (
                candidate_metrics["max_work"] < best_metrics["max_work"]
                or (
                    candidate_metrics["max_work"] == best_metrics["max_work"]
                    and candidate_metrics["utilization"] > best_metrics["utilization"] + 1e-4
                )
                or (
                    candidate_metrics["max_work"] == best_metrics["max_work"]
                    and abs(candidate_metrics["utilization"] - best_metrics["utilization"]) <= 1e-4
                    and candidate_total_splits < best_total_splits
                )
            ):
                chosen = (req_idx, candidate_split_levels, candidate, candidate_metrics)
                improved = True
                break

        if not improved:
            break

        _, best_split_levels, candidate, best_metrics = chosen
        (
            best_attn_assignments,
            best_post_assignments,
            best_group_loads,
            best_attn_waves,
            best_post_waves,
        ) = candidate

    return (
        best_split_levels,
        best_attn_assignments,
        best_post_assignments,
        best_group_loads,
        best_attn_waves,
        best_post_waves,
    )


@dataclass
class SchedPlan:
    request_idx: int
    split_level: int
    attn_groups: list[int]
    post_groups: list[int]
    attn_waves: list[int]
    post_waves: list[int]

    def __post_init__(self):
        self.seq_length = seq_lengths[self.request_idx]
        self.num_kv_block = (self.seq_length + KVTile - 1) // KVTile
        self.num_block_per_split = self.num_kv_block // self.split_level
        assert len(self.attn_groups) == self.split_level, "Need one attention group per split shard"
        assert len(self.attn_waves) == self.split_level, "Need one attention wave per split shard"
        assert len(self.post_groups) == len(self.post_waves), "Need one post wave per post group"
        self.attn_group_to_split = {
            group_id: (split_idx, wave_idx)
            for split_idx, (group_id, wave_idx) in enumerate(zip(self.attn_groups, self.attn_waves))
        }
        self.post_group_to_lane = {
            group_id: (lane, wave_idx)
            for lane, (group_id, wave_idx) in enumerate(zip(self.post_groups, self.post_waves))
        }
        self.attn_bar = dae.new_bar(NUM_KV_HEAD * self.split_level)
        self.tO_split = TmaTensor(dae, matO_split_attn_view)._build("load", self.split_level, HEAD_GROUP_SIZE*HEAD_DIM, partial(tma_load_split_attn, split_kv=self.split_level), cord_load_split_attn)

    def sm_attn_task(self, sm: int):
        sm_group = sm // NUM_KV_HEAD
        if sm_group not in self.attn_group_to_split:
            return []
        head = sm % NUM_KV_HEAD
        split, wave_idx = self.attn_group_to_split[sm_group]
        insts = []
        if self.split_level == 1:
            return [
                ATTENTION_M64N64K16_F16_F32_64_64_hdim(self.num_kv_block, last_active_kv_len, need_norm=need_norm, need_rope=need_rope),
                tQ.cord(self.request_idx, head),
                RepeatM.on(self.num_kv_block,
                    [tK.cord(self.request_idx, 0, head), tK.cord2tma(0, KVTile, 0)],
                    [tV.cord(self.request_idx, 0, head), tV.cord2tma(0, KVTile, 0)],
                ),
                TmaStore1D(matO_attn_view[self.request_idx, head, ...]),
            ]

        kv_start_block = split * self.num_block_per_split
        kv_start_idx = kv_start_block * KVTile
        split_last_active_kv_len = last_active_kv_len if split == self.split_level - 1 else KVTile
        insts.extend([
            ATTENTION_M64N64K16_F16_F32_64_64_hdim_split(self.num_block_per_split, split, HEAD_GROUP_SIZE, split_last_active_kv_len, kv_start_idx, need_norm=need_norm, need_rope=need_rope),
            tQ.cord(self.request_idx, head),
            RepeatM.on(self.num_block_per_split,
                [tK.cord(self.request_idx, kv_start_idx, head), tK.cord2tma(0, KVTile, 0)],
                [tV.cord(self.request_idx, kv_start_idx, head), tV.cord2tma(0, KVTile, 0)],
            ),
            TmaStore1D(matO_split_attn_view[split, self.request_idx, head, ...]),
            TmaStore1D(matP[head, self.request_idx * HEAD_GROUP_SIZE: (self.request_idx+1) * HEAD_GROUP_SIZE]).bar(self.attn_bar),
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
            TmaLoad1D(matP[head, self.request_idx * HEAD_GROUP_SIZE: (self.request_idx+1) * HEAD_GROUP_SIZE]).bar(self.attn_bar),
            self.tO_split.cord(head, self.request_idx),
            TmaStore1D(matO_attn_view[self.request_idx, head, ...]),
        ])
        return insts


def build_sched_plans():
    num_kv_blocks = [(seq_len + KVTile - 1) // KVTile for seq_len in seq_lengths]
    total_blocks = sum(num_kv_blocks)
    target_blocks_per_shard = total_blocks / TOTAL_ATTN_GROUPS * 1.2

    split_levels = [
        choose_split_level(num_blocks, target_blocks_per_shard, mode=schedule_mode)
        for num_blocks in num_kv_blocks
    ]
    if schedule_mode == 'vdc':
        (
            split_levels,
            attn_group_assignments,
            post_group_assignments,
            attn_group_loads,
            attn_group_waves,
            post_group_waves,
        ) = refine_split_levels(num_kv_blocks, split_levels)
    else:
        (
            attn_group_assignments,
            post_group_assignments,
            attn_group_loads,
            attn_group_waves,
            post_group_waves,
        ) = place_split_levels(num_kv_blocks, split_levels)

    plans = []
    for req_idx, split_level in enumerate(split_levels):
        sorted_attn = sorted(attn_group_assignments[req_idx])
        attn_groups = [group_id for _, group_id, _ in sorted_attn]
        attn_waves = [wave_idx for _, _, wave_idx in sorted_attn]
        post_groups = [group_id for group_id, _ in post_group_assignments[req_idx]]
        post_waves = [wave_idx for _, wave_idx in post_group_assignments[req_idx]]
        plans.append(
            SchedPlan(
                request_idx=req_idx,
                split_level=split_level,
                attn_groups=attn_groups,
                post_groups=post_groups,
                attn_waves=attn_waves,
                post_waves=post_waves,
            )
        )
    return plans, split_levels, attn_group_loads, attn_group_waves, post_group_waves


plans, split_levels, attn_group_loads, attn_group_waves, post_group_waves = build_sched_plans()
attn_metrics = calc_group_metrics(attn_group_loads, TOTAL_ATTN_GROUPS)
print(f"[sched] seq_lengths={seq_lengths}")
print(f"[sched] split_levels={split_levels}, target_blocks_per_shard={sum((seq_len + KVTile - 1) // KVTile for seq_len in seq_lengths) / TOTAL_ATTN_GROUPS:.2f}")
print(f"[sched] attn_group_loads={attn_group_loads}")
print(f"[sched] attn_group_waves={attn_group_waves}")
print(f"[sched] post_group_waves={post_group_waves}")
print(
    f"[sched] attn_metrics total={attn_metrics['total_work']} "
    f"max={attn_metrics['max_work']} mean={attn_metrics['mean_work']:.2f} "
    f"imbalance={attn_metrics['imbalance']:.3f} utilization={attn_metrics['utilization']:.3f}"
)
for plan in plans:
    print(
        f"[sched] req={plan.request_idx} blocks={plan.num_kv_block} split={plan.split_level} "
        f"attn_groups={plan.attn_groups} attn_waves={plan.attn_waves} "
        f"post_groups={plan.post_groups} post_waves={plan.post_waves}"
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

# refQK, refO = gqa_ref()
# tensor_diff("Ref and DAE", refO, matO_attn_view)
