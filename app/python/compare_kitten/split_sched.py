import torch

from dae import runtime
from dae.launcher import *


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
        self.split_kv = max(1, min(self.num_sms // self.num_kv_head, self.num_kv_block, self.max_split))
        assert matO_split.shape[0] >= self.split_kv, "matO_split must reserve at least the inferred split count"

        self.q_tile = 64 // self.head_group_size
        self.compute_sms = self.num_kv_head * self.split_kv
        self.split_q_tile = max(1, (self.num_q_head + self.num_sms - 1) // self.num_sms)
        self.num_post_sms = (self.num_q_head + self.split_q_tile - 1) // self.split_q_tile
        self.splits_per_post_load = min(
            max(1, self.post_split_load_limit_bytes // (self.split_q_tile * self.head_dim * 2)),
            self.split_kv,
        )
        assert self.split_kv % self.splits_per_post_load == 0, (
            f"split_level {self.split_kv} must be divisible by SPLITS_PER_POST_LOAD {self.splits_per_post_load}"
        )

        self.matQ_attn_view = matQ.view(self.num_req, self.num_kv_head, self.head_group_size, self.head_dim)
        self.matK_attn_view = matK.view(self.num_req, self.kv_seq_len, self.num_kv_head, self.head_dim)
        self.matV_attn_view = matV.view(self.num_req, self.kv_seq_len, self.num_kv_head, self.head_dim)
        self.matO_split_attn_view = matO_split.view(self.split_kv, self.num_req, self.num_kv_head, self.head_group_size, self.head_dim)
        self.matO_split_load_view = matO_split.view(self.split_kv, self.num_req, self.num_q_head, self.head_dim)
        self.matO_attn_q_view = matO.view(self.num_req, self.num_q_head, self.head_dim)

        self.q_req = self.matQ_attn_view[self.req_id:self.req_id + 1]
        self.k_req = self.matK_attn_view[self.req_id:self.req_id + 1]
        self.v_req = self.matV_attn_view[self.req_id:self.req_id + 1]
        self.o_split_req = self.matO_split_attn_view[:, self.req_id:self.req_id + 1]
        self.o_split_load_req = self.matO_split_load_view[:, self.req_id:self.req_id + 1]
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
                        [self.tK.cord(kv_start_idx, head), self.tK.cord2tma(0, self.kv_tile, 0)],
                        [self.tV.cord(kv_start_idx, head), self.tV.cord2tma(0, self.kv_tile, 0)],
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
                [self.tO_split.cord(0, q_ofst), self.tO_split.cord2tma(self.splits_per_post_load, 0, 0)],
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
