import torch

from dae import runtime
from dae.launcher import *


def kv_blocks_for_seq(seq_length: int, kv_tile: int) -> int:
    return (seq_length + kv_tile - 1) // kv_tile


def infer_split_kv(num_sms: int, num_kv_head: int, num_kv_block: int, max_split: int) -> int:
    if num_kv_block <= 0:
        return 1
    if num_sms < num_kv_head:
        raise ValueError(f"num_sms={num_sms} must be at least num_kv_head={num_kv_head}")
    return max(1, min(num_sms // num_kv_head, num_kv_block, max_split))


def estimate_request_steps(num_sms: int, num_kv_head: int, num_kv_block: int, max_split: int) -> int:
    split_kv = infer_split_kv(num_sms, num_kv_head, num_kv_block, max_split)
    return (num_kv_block + split_kv - 1) // split_kv


def min_sms_for_target_steps(target_steps: int, num_kv_head: int, num_kv_block: int, max_split: int) -> int | None:
    if num_kv_block <= 0:
        return num_kv_head
    required_split = (num_kv_block + target_steps - 1) // target_steps
    if required_split > max_split:
        return None
    return num_kv_head * max(1, required_split)


class SchedAttentionSplit:
    def __init__(self,
                 dae: Launcher,
                 req_id: int,
                 num_sms: int,
                 base_sm: int,
                 seq_length: int,
                 matQ: torch.Tensor,
                 matK: torch.Tensor,
                 matV: torch.Tensor,
                 matO: torch.Tensor,
                 matO_split: torch.Tensor,
                 matP: torch.Tensor,
                 need_norm: bool = False,
                 need_rope: bool = False,
                 kv_tile: int = 64,
                 post_split_load_limit_bytes: int = 16 * 1024):
        self.dae = dae
        self.req_id = req_id
        self.num_sms = num_sms
        self.base_sm = base_sm
        self.seq_length = seq_length

        self.matQ = matQ
        self.matK = matK
        self.matV = matV
        self.matO = matO
        self.matO_split = matO_split
        self.matP = matP

        self.need_norm = need_norm
        self.need_rope = need_rope
        self.kv_tile = kv_tile
        self.post_split_load_limit_bytes = post_split_load_limit_bytes

        self.num_req = matQ.shape[0]
        self.hidden_size = matQ.shape[1]
        self.max_split = matP.shape[1]
        self.num_kv_head = matP.shape[2]
        self.head_group_size = matP.shape[3]
        self.num_q_head = self.num_kv_head * self.head_group_size
        self.head_dim = matK.shape[1] // self.num_kv_head
        self.kv_seq_len = matK.shape[0] // self.num_req
        self.num_kv_block = (self.seq_length + self.kv_tile - 1) // self.kv_tile

        assert 0 <= self.req_id < self.num_req, f"req_id {self.req_id} out of range for {self.num_req} requests"
        assert self.num_sms > 0, "num_sms must be positive"
        assert self.base_sm >= 0, "base_sm must be non-negative"
        assert self.hidden_size == self.num_q_head * self.head_dim, "Q size must match inferred attention shape"
        assert self.kv_seq_len * self.num_req == matK.shape[0], "matK must reshape cleanly into [req, seq, head, dim]"
        assert matV.shape == matK.shape, "matV must match matK layout"
        assert matO.shape == matQ.shape, "matO must match matQ layout"
        assert self.num_sms >= self.num_kv_head, (
            f"num_sms={self.num_sms} must cover all KV heads ({self.num_kv_head}) before sequence splitting"
        )

        # Prefer head parallelism first since it does not introduce post-reduce overhead.
        # Only the residual per-head budget is used for sequence splitting.
        self.split_kv = infer_split_kv(self.num_sms, self.num_kv_head, self.num_kv_block, self.max_split)
        assert matO_split.shape[0] >= self.split_kv, "matO_split must reserve at least the inferred split count"

        self.q_tile = 64 // self.head_group_size
        self.compute_sms = self.num_kv_head * self.split_kv
        self.split_q_tile = max(1, (self.num_q_head + self.num_sms - 1) // self.num_sms)
        self.num_post_sms = (self.num_q_head + self.split_q_tile - 1) // self.split_q_tile
        self.splits_per_post_load = min(
            max(1, self.post_split_load_limit_bytes // (self.split_q_tile * self.head_dim * 2)),
            self.split_kv,
        )
        # print(f"req_id {self.req_id}: split_kv={self.split_kv}, compute_sms={self.compute_sms}, split_q_tile={self.split_q_tile}, num_post_sms={self.num_post_sms}, splits_per_post_load={self.splits_per_post_load}")
        assert self.split_kv % self.splits_per_post_load == 0, (
            f"split_level {self.split_kv} must be divisible by SPLITS_PER_POST_LOAD {self.splits_per_post_load}"
        )

        self.matQ_attn_view = matQ.view(self.num_req, self.num_kv_head, self.head_group_size, self.head_dim)
        self.matK_attn_view = matK.view(self.num_req, self.kv_seq_len, self.num_kv_head, self.head_dim)
        self.matV_attn_view = matV.view(self.num_req, self.kv_seq_len, self.num_kv_head, self.head_dim)
        self.matO_split_attn_view = matO_split.view(self.max_split, self.num_req, self.num_kv_head, self.head_group_size, self.head_dim)
        self.matO_split_load_view = matO_split.view(self.max_split, self.num_req, self.num_q_head, self.head_dim)
        self.matO_attn_q_view = matO.view(self.num_req, self.num_q_head, self.head_dim)

        self.q_req = self.matQ_attn_view[self.req_id:self.req_id + 1]
        self.k_req = self.matK_attn_view[self.req_id:self.req_id + 1]
        self.v_req = self.matV_attn_view[self.req_id:self.req_id + 1]
        self.o_split_req = self.matO_split_attn_view[:self.split_kv, self.req_id:self.req_id + 1]
        self.o_split_load_req = self.matO_split_load_view[:self.split_kv, self.req_id:self.req_id + 1]
        self.o_post_req = self.matO_attn_q_view[self.req_id]
        self.p_req = self.matP[self.req_id:self.req_id + 1]

        self.attn_bar = self.dae.new_bar(self._active_split_count() * self.num_kv_head)
        self.tQ, self.tK, self.tV, self.tO_split = self._build_tmas()

    def _active_split_count(self):
        active = 0
        for split_stage in range(self.split_kv):
            if self.split_bounds(split_stage)[5] > 0:
                active += 1
        return active

    def split_bounds(self, split_stage: int):
        num_block_per_split = (self.num_kv_block + self.split_kv - 1) // self.split_kv
        kv_start_block = split_stage * num_block_per_split
        kv_start = kv_start_block * self.kv_tile
        kv_end = kv_start + num_block_per_split * self.kv_tile
        total_active = min(max(self.seq_length - kv_start, 0), kv_end - kv_start)
        split_last_active_kv_len = total_active % self.kv_tile
        if total_active > 0 and split_last_active_kv_len == 0:
            split_last_active_kv_len = self.kv_tile
        return (
            self.num_kv_block,
            num_block_per_split,
            kv_start_block,
            kv_start,
            kv_end,
            total_active,
            split_last_active_kv_len,
        )

    def _build_tmas(self):
        def tma_load_q(mat: torch.Tensor, tileK: int, tileN: int):
            assert mat.element_size() == 2, "Only support float16/bfloat16 Q tensors"
            assert tileK == self.head_dim and tileN == 64, "Q tile must be HEAD_DIM x 64"

            glob_dims = [64, self.head_group_size, self.q_tile, 2, mat.shape[0] * mat.shape[1]]
            glob_strides = [128 * 2, 0, 64 * 2, self.head_dim * self.head_group_size * 2]
            box_dims = [64, self.head_group_size, self.q_tile, 2, 1]
            rank = len(glob_dims)
            return rank, runtime.build_tma_desc(
                mat,
                glob_dims,
                glob_strides,
                box_dims,
                [1] * rank,
                128,
                0,
            )

        def cord_load_q(mat: torch.Tensor, rank: int):
            assert rank == 5, "Q load expects a 5D TMA descriptor"
            def cfunc(*cords):
                assert len(cords) == 1, f"Q load expects (head), got {cords}"
                return [0, 0, 0, cords[0]]
            return cfunc

        def tma_load_k(mat: torch.Tensor, tileK: int, tileN: int):
            _, S, H, _ = mat.shape
            glob_dims = [64, S, 2, H, 1]
            elsize = mat.element_size()
            glob_strides = [d * elsize for d in [mat.stride(1), 64, mat.stride(2), mat.stride(0)]]
            box_dims = [64, tileN, 2, 1, 1]
            rank = len(glob_dims)
            return rank, runtime.build_tma_desc(
                mat,
                glob_dims,
                glob_strides,
                box_dims,
                [1] * rank,
                128,
                0,
            )

        def cord_load_k(mat: torch.Tensor, rank: int):
            assert rank == 5, "K load expects a 5D TMA descriptor"
            def cfunc(*cords):
                assert len(cords) == 2, f"K load expects (seq, head), got {cords}"
                s, h = cords
                return [s, 0, h, 0]
            return cfunc

        def tma_load_v(mat: torch.Tensor, tileM: int, tileK: int):
            _, S, H, D = mat.shape
            elsize = mat.element_size()

            assert D == self.head_dim
            assert tileM == self.head_dim
            assert tileK == self.kv_tile
            assert D % 64 == 0
            assert S % 8 == 0

            m_total = H * D
            glob_dims = [64, 8, m_total // 64, S // 8, 1]
            glob_strides = [
                mat.stride(1),
                64,
                mat.stride(1) * 8,
                mat.stride(0),
            ]
            glob_strides = [s * elsize for s in glob_strides]
            box_dims = [64, 8, tileM // 64, tileK // 8, 1]
            rank = len(glob_dims)
            return rank, runtime.build_tma_desc(
                mat,
                glob_dims,
                glob_strides,
                box_dims,
                [1] * rank,
                128,
                0,
            )

        def cord_load_v(mat: torch.Tensor, rank: int):
            assert rank == 5, "V load expects a 5D TMA descriptor"
            def cfunc(*cords):
                assert len(cords) == 2, f"V load expects (seq, head), got {cords}"
                s, h = cords
                return [0, h * (self.head_dim // 64), s // 8, 0]
            return cfunc

        def tma_load_split_attn(mat: torch.Tensor, tileS: int, tileO: int):
            assert tileS == self.splits_per_post_load
            tileQ = tileO // self.head_dim
            assert tileQ == self.split_q_tile, f"tileQ {tileQ} must match split_q_tile {self.split_q_tile}"
            _, _, Q, D = mat.shape
            glob_dims = [D, Q, mat.shape[0], mat.shape[1]]
            glob_strides = [mat.stride(i) * mat.element_size() for i in [2, 0, 1]]
            box_dims = [D, tileQ, tileS, 1]
            rank = len(glob_dims)
            return rank, runtime.build_tma_desc(
                mat,
                glob_dims,
                glob_strides,
                box_dims,
                [1] * rank,
                0,
                0,
            )

        def cord_load_split_attn(mat: torch.Tensor, rank: int):
            assert rank == 4, "Split-attention load expects a 4D TMA descriptor"
            def cfunc(*cords):
                assert len(cords) == 2, f"Split-attention load expects (split, q_head), got {cords}"
                s, hq = cords
                return [0, hq, s, 0]
            return cfunc

        tQ = TmaTensor(self.dae, self.q_req)._build("load", self.head_dim, 64, tma_load_q, cord_load_q)
        tK = TmaTensor(self.dae, self.k_req)._build("load", self.head_dim, self.kv_tile, tma_load_k, cord_load_k)
        tV = TmaTensor(self.dae, self.v_req)._build("load", self.head_dim, self.kv_tile, tma_load_v, cord_load_v)
        tO_split = TmaTensor(self.dae, self.o_split_load_req)._build(
            "load",
            self.splits_per_post_load,
            self.head_dim * self.split_q_tile,
            tma_load_split_attn,
            cord_load_split_attn,
        )
        return tQ, tK, tV, tO_split

    def sm_task(self, sm: int):
        if sm < 0 or sm >= self.num_sms:
            return []

        insts = []
        if sm < self.compute_sms:
            split_stage = sm // self.num_kv_head
            head = sm % self.num_kv_head
            _, num_block_per_split, kv_start_block, kv_start_idx, _, total_active, split_last_active_kv_len = self.split_bounds(split_stage)
            if total_active > 0:
                insts += [
                    ATTENTION_M64N64K16_F16_F32_64_64_hdim_split(
                        num_block_per_split,
                        self.head_group_size,
                        split_last_active_kv_len,
                        kv_start_block,
                        need_norm=self.need_norm,
                        need_rope=self.need_rope,
                    ),
                    self.tQ.cord(head),
                    RepeatM.on(
                        num_block_per_split,
                        [self.tK.cord(kv_start_idx, head), self.tK.cord2tma(self.kv_tile, 0)],
                        [self.tV.cord(kv_start_idx, head), self.tV.cord2tma(self.kv_tile, 0)],
                    ),
                    TmaStore1D(self.o_split_req[split_stage, 0, head, ...], numSlots=2),
                    TmaStore1D(self.p_req[0, split_stage, head]).bar(self.attn_bar),
                ]

        if sm >= self.num_post_sms:
            return insts

        q_ofst = sm * self.split_q_tile
        insts += [
            ATTN_SPLIT_POST_REDUCE(self.split_kv, self.splits_per_post_load, self.split_q_tile, q_ofst, self.num_q_head),
            TmaLoad1D(self.p_req[0, :self.split_kv]).bar(self.attn_bar),
            RepeatM.on(
                self.split_kv // self.splits_per_post_load,
                [self.tO_split.cord(0, q_ofst), self.tO_split.cord2tma(self.splits_per_post_load, 0)],
            ),
            TmaStore1D(self.o_post_req[q_ofst:q_ofst + self.split_q_tile]),
        ]
        return insts

    def schedule(self, sm: int):
        local_sm = sm - self.base_sm
        if local_sm < 0 or local_sm >= self.num_sms:
            return []
        return self.sm_task(local_sm)

    __call__ = schedule


class GlobalSchedAttentionSplit:
    def __init__(self,
                 dae: Launcher,
                 total_sms: int,
                 seq_lengths: list[int] | tuple[int, ...],
                 matQ: torch.Tensor,
                 matK: torch.Tensor,
                 matV: torch.Tensor,
                 matO: torch.Tensor,
                 matO_split: torch.Tensor,
                 matP: torch.Tensor,
                 need_norm: bool = False,
                 need_rope: bool = False,
                 kv_tile: int = 64,
                 post_split_load_limit_bytes: int = 16 * 1024):
        self.dae = dae
        self.total_sms = total_sms
        self.seq_lengths = list(seq_lengths)
        self.matQ = matQ
        self.matK = matK
        self.matV = matV
        self.matO = matO
        self.matO_split = matO_split
        self.matP = matP
        self.need_norm = need_norm
        self.need_rope = need_rope
        self.kv_tile = kv_tile
        self.post_split_load_limit_bytes = post_split_load_limit_bytes

        self.num_req = matQ.shape[0]
        self.max_split = matP.shape[1]
        self.num_kv_head = matP.shape[2]
        self.head_group_size = matP.shape[3]
        self.num_q_head = self.num_kv_head * self.head_group_size
        self.head_dim = matK.shape[1] // self.num_kv_head
        self.num_kv_blocks = [kv_blocks_for_seq(seq_len, self.kv_tile) for seq_len in self.seq_lengths]

        if len(self.seq_lengths) != self.num_req:
            raise ValueError(f"seq_lengths has {len(self.seq_lengths)} entries, expected {self.num_req}")
        min_total_sms = self.num_req * self.num_kv_head
        if self.total_sms < min_total_sms:
            raise ValueError(
                f"total_sms={self.total_sms} is insufficient for {self.num_req} requests; need at least {min_total_sms}"
            )

        self.sms_assignment = self._assign_sms()
        self.schedulers = self._build_request_schedulers()

    def _feasible_assignment_for_steps(self, target_steps: int):
        assignment = []
        for num_blocks in self.num_kv_blocks:
            required_sms = min_sms_for_target_steps(target_steps, self.num_kv_head, num_blocks, self.max_split)
            if required_sms is None:
                return None
            assignment.append(required_sms)
        if sum(assignment) > self.total_sms:
            return None
        return assignment

    def _assign_sms(self):
        lo = 1
        hi = max(max(self.num_kv_blocks, default=1), 1)
        best_assignment = [self.num_kv_head for _ in range(self.num_req)]

        while lo <= hi:
            mid = (lo + hi) // 2
            assignment = self._feasible_assignment_for_steps(mid)
            if assignment is None:
                lo = mid + 1
            else:
                best_assignment = assignment
                hi = mid - 1

        assignment = best_assignment[:]
        remaining = self.total_sms - sum(assignment)

        while remaining >= self.num_kv_head:
            best_req = None
            best_gain = 0
            best_next_steps = None

            for req_id, (cur_sms, num_blocks) in enumerate(zip(assignment, self.num_kv_blocks)):
                cur_steps = estimate_request_steps(cur_sms, self.num_kv_head, num_blocks, self.max_split)
                next_sms = cur_sms + self.num_kv_head
                next_steps = estimate_request_steps(next_sms, self.num_kv_head, num_blocks, self.max_split)
                gain = cur_steps - next_steps
                if gain <= 0:
                    continue
                if gain > best_gain or (gain == best_gain and (best_next_steps is None or next_steps < best_next_steps)):
                    best_req = req_id
                    best_gain = gain
                    best_next_steps = next_steps

            if best_req is None:
                break

            assignment[best_req] += self.num_kv_head
            remaining -= self.num_kv_head

        return assignment

    def _build_request_schedulers(self):
        schedulers = []
        base_sm = 0
        for req_id, (seq_length, num_sms) in enumerate(zip(self.seq_lengths, self.sms_assignment)):
            schedulers.append(
                SchedAttentionSplit(
                    dae=self.dae,
                    req_id=req_id,
                    num_sms=num_sms,
                    base_sm=base_sm,
                    seq_length=seq_length,
                    matQ=self.matQ,
                    matK=self.matK,
                    matV=self.matV,
                    matO=self.matO,
                    matO_split=self.matO_split,
                    matP=self.matP,
                    need_norm=self.need_norm,
                    need_rope=self.need_rope,
                    kv_tile=self.kv_tile,
                    post_split_load_limit_bytes=self.post_split_load_limit_bytes,
                )
            )
            base_sm += num_sms
        return schedulers

    def schedule(self, sm: int):
        for scheduler in self.schedulers:
            if scheduler.base_sm <= sm < scheduler.base_sm + scheduler.num_sms:
                return scheduler.schedule(sm)
        return []

    def describe(self):
        return [
            {
                "req_id": scheduler.req_id,
                "seq_length": scheduler.seq_length,
                "num_sms": scheduler.num_sms,
                "split_kv": scheduler.split_kv,
                "base_sm": scheduler.base_sm,
                "estimated_steps": estimate_request_steps(
                    scheduler.num_sms,
                    scheduler.num_kv_head,
                    scheduler.num_kv_block,
                    scheduler.max_split,
                ),
            }
            for scheduler in self.schedulers
        ]

    __call__ = schedule
