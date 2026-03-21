import copy as pycopy
import warnings

from .runtime import *
from .launcher import *

class Schedule:
    def __init__(self):
        self.num_sms = None
        self.base_sm = 0
        self._bars = {}

    def _clone(self):
        clone = pycopy.copy(self)
        clone._bars = self._bars.copy()
        return clone

    def _on_place(self):
        pass

    def place(self, num_sms: int, base_sm: int = 0):
        clone = self._clone()
        clone.num_sms = num_sms
        clone.base_sm = base_sm
        clone._on_place()
        return clone

    def bar(self, role: str, bar_id: int):
        self._bars[role] = bar_id
        return self

    def _bar(self, role: str):
        return self._bars.get(role)

    def _require_placed(self):
        if self.num_sms is None:
            raise ValueError(f"{self.__class__.__name__} must be placed before querying barrier release counts")

    def _bar_release_if_present(self, role: str, count: int):
        if self._bar(role) is None:
            return 0
        self._require_placed()
        return count

    def bar_release_count(self, role: str):
        return 0

    def collect_barrier_release_counts(self):
        counts = {}
        for role, bar_id in self._bars.items():
            count = self.bar_release_count(role)
            if count <= 0:
                continue
            counts[bar_id] = counts.get(bar_id, 0) + count
        return counts

    def map_sm(self, sm: int):
        # This function decides how to map SM to SM ID. this can create scheudle
        # that is not strictly round-robin, e.g., for hierarchical scheduling,
        # we may want to map SMs in the same fold together.
        if self.num_sms is None:
            raise ValueError(f"{self.__class__.__name__} must be placed before scheduling")
        sm -= self.base_sm
        if sm < 0 or sm >= self.num_sms:
            return -1
        return sm
    def schedule(self, sm: int):
        raise NotImplementedError("Schedule.schedule() must be implemented by subclass")
    def __call__(self, sm: int):
        mapped_sm = self.map_sm(sm)
        return self.schedule(mapped_sm)


class ListSchedule(Schedule):
    def __init__(self, items, lead_bars=None, tail_bars=None, warn_boundary_bars=False):
        super().__init__()
        self.items = list(items)
        self.lead_bars = set() if lead_bars is None else set(lead_bars)
        self.tail_bars = set() if tail_bars is None else set(tail_bars)
        self.warn_boundary_bars = warn_boundary_bars
        self._warned_boundary_roles = set()

    def _clone(self):
        clone = super()._clone()
        clone.items = [
            item._clone() if isinstance(item, Schedule) else item
            for item in self.items
        ]
        clone.lead_bars = self.lead_bars.copy()
        clone.tail_bars = self.tail_bars.copy()
        clone.warn_boundary_bars = self.warn_boundary_bars
        clone._warned_boundary_roles = self._warned_boundary_roles.copy()
        return clone

    def _schedule_items(self):
        return [item for item in self.items if isinstance(item, Schedule)]

    def warn_on_boundary_bars(self, enable=True):
        self.warn_boundary_bars = enable
        return self

    def _maybe_warn_boundary_bar(self, role: str, num_schedules: int):
        if not self.warn_boundary_bars or num_schedules <= 1:
            return
        if role in self._warned_boundary_roles:
            return
        if role in self.lead_bars or role in self.tail_bars:
            warnings.warn(
                f"ListSchedule bar('{role}', ...) only applies to boundary schedule(s); "
                "this may be insufficient for interior dependencies",
                stacklevel=3,
            )
            self._warned_boundary_roles.add(role)

    def _apply_boundary_bars(self):
        schedules = self._schedule_items()
        if not schedules:
            return

        first = schedules[0]
        last = schedules[-1]
        for role, bar_id in self._bars.items():
            self._maybe_warn_boundary_bar(role, len(schedules))
            applied = False
            if role in self.lead_bars:
                first.bar(role, bar_id)
                applied = True
            if role in self.tail_bars:
                last.bar(role, bar_id)
                applied = True
            if not applied:
                if first is last:
                    first.bar(role, bar_id)
                else:
                    raise ValueError(f"ListSchedule cannot route bar role '{role}'")

    def place(self, num_sms: int, base_sm: int = 0):
        clone = self._clone()
        clone.num_sms = num_sms
        clone.base_sm = base_sm
        clone.items = [
            item.place(num_sms, base_sm) if isinstance(item, Schedule) else item
            for item in clone.items
        ]
        clone._apply_boundary_bars()
        return clone

    def bar(self, role: str, bar_id: int):
        super().bar(role, bar_id)
        self._apply_boundary_bars()
        return self

    def __call__(self, sm: int):
        insts = []
        for item in self.items:
            if callable(item):
                insts.append(item(sm))
            else:
                insts.append(item)
        return insts

    def __getitem__(self, idx):
        return self.items[idx]

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def bar_release_count(self, role: str):
        return sum(
            item.bar_release_count(role)
            for item in self.items
            if isinstance(item, Schedule)
        )

    def collect_barrier_release_counts(self):
        counts = {}
        for item in self.items:
            if not isinstance(item, Schedule):
                continue
            for bar_id, count in item.collect_barrier_release_counts().items():
                counts[bar_id] = counts.get(bar_id, 0) + count
        return counts

class SchedCopy(Schedule):
    def __init__(self,
                 tmas,
                 size = None,
                 before_copy = None,
                 count = 1):
        super().__init__()
        self.tmas = tmas
        self.count = count
        self.before_copy = before_copy

        if size is None:
            assert tmas[0].size == tmas[1].size, "Size must be specified when load and store TMA sizes do not match"
            size = tmas[0].size
        self.size = size

    def schedule(self, sm: int):
        if sm < 0:
            return []

        load, store = self.tmas
        load = load.cord(sm)
        store = store.cord(sm)
        if self.before_copy is not None:
            load.jump()

        return [
            Copy(1, size = self.size),

            self.before_copy,
            load.bar(self._bar("load")),
            store.bar(self._bar("store")),
        ]

    def bar_release_count(self, role: str):
        if role != "store":
            return 0
        return self._bar_release_if_present(role, self.num_sms)

class SchedRope(Schedule):
    def __init__(self, Atom, tmas):
        super().__init__()
        self.Atom = Atom
        self.tmas = tmas

    def schedule(self, sm: int):
        if sm < 0:
            return []
        table, load, store = [tma.cord(sm) for tma in self.tmas]

        return [
            self.Atom(),

            table,
            load,
            store.bar(self._bar("store")).group(),
        ]

    def bar_release_count(self, role: str):
        if role != "store":
            return 0
        return self._bar_release_if_present(role, self.num_sms)

class SchedAttentionDecoding(Schedule):
    # for decoding, the Qtile len will always be 1
    def __init__(self, reqs: int, seq_len: int,
                 KV_BLOCK_SIZE : int, NUM_KV_HEADS : int,
                 matO : torch.Tensor,
                 tmas):
        super().__init__()
        self.reqs = reqs
        self.seq_len = seq_len
        self.num_heads = NUM_KV_HEADS
        self.matO = matO
        self.tmas = tmas
        self.required_sms = reqs * NUM_KV_HEADS
        self.block_size = KV_BLOCK_SIZE

    def _on_place(self):
        assert self.num_sms == self.required_sms, f"SchedAttentionDecoding requires {self.required_sms} SMs, got {self.num_sms}"

    def schedule(self, sm: int):
        if sm < 0:
            return []

        req = sm // self.num_heads
        head = sm % self.num_heads

        tQ, tK, tV = self.tmas

        num_kv_blocks = (self.seq_len + self.block_size - 1) // self.block_size
        seq_len_last_block = self.seq_len % self.block_size

        # we only handle a single Q token here
        insts = [
            ATTENTION_M64N64K16_F16_F32_64_64_hdim(num_kv_blocks, seq_len_last_block, need_norm=False, need_rope=False),
            tQ.cord(req, head).bar(self._bar("q")).group(),
            RepeatM.on(num_kv_blocks - 1,
                # this k-barrier will also barrier following V load
                [tK.cord(req, 0, head, 0).group(), tK.cord2tma(0, self.block_size, 0, 0)],
                [tV.cord(req, 0, head, 0).group(), tV.cord2tma(0, self.block_size, 0, 0)],
            ),
            # TODO(zhiyuang): reuse the accumulator register
            # only the last block has new generated KV cache
            tK.cord(req, self.block_size * (num_kv_blocks - 1), head, 0).bar(self._bar("k")).group(),
            tV.cord(req, self.block_size * (num_kv_blocks - 1), head, 0).group(),
            TmaStore1D(self.matO[req, head, ...], numSlots = 2).bar(self._bar("o")).group(),
        ]
        return insts

    def bar_release_count(self, role: str):
        if role != "o":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedAttention(Schedule):
    def __init__(self,
                 reqs : int,
                 active_new_len: int,
                 cached_seq_len: int,
                 QKVHdim: tuple[int, int, int],
                 QKVTile: tuple[int, int],
                 QKVSeqlen: tuple[int, int],
                 tmas: tuple[TmaTensor],
                 need_norm: bool,
                 need_rope: bool,
                 rope_table: RawAddress):
        super().__init__()
        self.tmas = tmas
        self.reqs = reqs
        self.QKVHdim = QKVHdim
        self.QKVSeqlen = QKVSeqlen
        self.QKVTile = QKVTile
        self.active_new_len = active_new_len
        self.cached_seq_len = cached_seq_len
        self.need_norm = need_norm
        self.need_rope = need_rope
        self.rope_table = rope_table

        self.required_sms = reqs * QKVHdim[1]

    def _on_place(self):
        assert self.num_sms == self.required_sms, f"SchedAttention requires {self.required_sms} SMs, got {self.num_sms}"

    def describe(self):
        print(f"SchedAttention: reqs={self.reqs}, active_new_len={self.active_new_len}, QKVHdim={self.QKVHdim}, QKVSeqlen={self.QKVSeqlen}, QKVTile={self.QKVTile}, sms={self.num_sms}")
    
    def schedule(self, sm: int):
        if sm < 0:
            return []

        tQ, tK, tV, tO = self.tmas
        
        NUM_Q_HEAD, NUM_KV_HEAD, HEAD_DIM = self.QKVHdim
        Q_SEQ_LEN, KV_SEQ_LEN = self.QKVSeqlen
        assert self.active_new_len <= Q_SEQ_LEN, "Active new length cannot exceed maximum Q sequence length"
        # KV_SEQ_LEN assume to be new KV
        assert self.cached_seq_len <= KV_SEQ_LEN, "Cached sequence length cannot exceed maximum KV sequence length"
        QTile, KVTile = self.QKVTile

        HEAD_GROUP_SIZE = NUM_Q_HEAD // NUM_KV_HEAD

        # TODO(zhiyuang): why this mapping?
        head = sm % NUM_KV_HEAD
        req = sm // NUM_KV_HEAD

        insts = []
        for q in range(0, self.active_new_len, QTile):
            insts += [
                ATTENTION_M64N64K16_F16_F32_64_64_hdim(min(self.active_new_len-q, QTile), hist_len=q+self.cached_seq_len, need_norm=self.need_norm, need_rope=self.need_rope),
                tQ.cord(req, q * HEAD_GROUP_SIZE, head, 0).bar(self._bar("q")).group(),
                self.rope_table if self.need_rope else [],
                # FIXME (zijian): this calculation should separate cached kv and new kv
                RepeatM.on((self.cached_seq_len + self.active_new_len + KVTile - 1) // KVTile,
                    # this k-barrier will also barrier following V load
                    [tK.cord(req, 0, head, 0).bar(self._bar("k")).group(), tK.cord2tma(0, KVTile, 0, 0)],
                    [tV.cord(req, 0, head, 0).group(), tV.cord2tma(0, KVTile, 0, 0)],
                ),
                tO.cord(req, q * HEAD_GROUP_SIZE, head, 0).port(1).bar(self._bar("o")).group(),
            ]
        return insts

    def bar_release_count(self, role: str):
        if role != "o":
            return 0
        q_tile = self.QKVTile[0]
        q_iters = (self.active_new_len + q_tile - 1) // q_tile
        return self._bar_release_if_present(role, self.num_sms * q_iters)


class SchedGemv(Schedule):
    def __init__(self, Atom,
                 MNK: tuple[int, int, int],
                 tmas: tuple[TmaTensor],
                 fold : int | None = None,
                 exec = True,
                 prefetch = True,
                 group = True):
        super().__init__()
        self.Atom = Atom
        self.MNK = MNK
        self.tmas = tmas

        TileM, TileN, TileK = Atom.MNK
        # process MNK
        MNK_base = []
        MNK_size = []
        assert len(MNK) == 3, "MNK must be a tuple of 3 dimensions"
        for dim in MNK:
            if isinstance(dim, int):
                MNK_base.append(0)
                MNK_size.append(dim)
            elif isinstance(dim, tuple) and len(dim) == 2:
                base, size = dim
                MNK_base.append(base)
                MNK_size.append(size)
            else:
                raise ValueError(f"Invalid MNK dimension: {dim}")
        
        self.MNK = MNK_size
        self.MNK_base = MNK_base

        self.fold = fold
        self.exec = exec
        self.prefetch = prefetch
        self.group = group
        self.sm_per_fold = None
        self.k_per_fold = None

    def _on_place(self):
        TileM, _, _ = self.Atom.MNK
        M, _, K = self.MNK

        if self.fold is None:
            assert self.num_sms % (M // TileM) == 0, f"SMS must be multiple of M tiles when auto folding, got SMS={self.num_sms}, M={M}, TileM={TileM}"
            self.fold = self.num_sms // (M // TileM)

        self.sm_per_fold = self.num_sms // self.fold
        self.k_per_fold = K // self.fold
        self.validate()
    
    def validate(self):
        # TODO(zhiyuang): more validation on fold?
        TileM, TileN, TileK = self.Atom.MNK
        M, N, K = self.MNK

        assert K % TileK == 0
        assert M % TileM == 0
        assert N % TileN == 0

        assert self.MNK_base[1] == 0, "N dimension must start from 0 for current schedule design"
        assert self.MNK[1] == N, "N dimension must cover the whole range for current schedule design"

        # verify fold
        assert self.num_sms % self.fold == 0
        assert K % self.fold == 0
        assert self.sm_per_fold == (M // TileM), "Invalid fold for given SMS and M size"
        assert self.k_per_fold % TileK == 0, "Invalid fold for given K size"

        # verify storeC. if fold > 1, storeC must be reduction
        assert len(self.tmas) == 3, "Expect at 3 TMA tensors: loadA, loadB, storeC"
        if self.fold > 1:
            assert self.tmas[-1].mode == "reduce", f"storeC must be reduction mode when fold > 1, got mode {self.tmas[-1].mode}"
    
    def schedule(self, sm: int):
        # TODO(zhiyuang): different SM mode?
        if sm < 0:
            return []

        TileM, TileN, TileK = self.Atom.MNK
        baseM, _, baseK = self.MNK_base
        n_batch = self.Atom.n_batch

        loadA, loadB, storeC = self.tmas

        m = baseM + (sm % self.sm_per_fold) * TileM
        k = baseK + (sm // self.sm_per_fold) * self.k_per_fold

        n_repeat = self.k_per_fold // (TileK * n_batch)

        # TODO(zhiyuang): more detailed group control
        load_group = self.group and (self._bar("load") is not None)
        store_group = self.group and (self._bar("store") is not None)

        storeC_cord = storeC.cord(0, m)

        insts = [
            self.Atom(self.k_per_fold // TileK),

            RepeatM.onSync(0, self._bar("load"), n_repeat,
                (loadB.cord(0, k).group(load_group), loadB.cord2tma(0, TileK * n_batch)),
                *[
                    (loadA.cord(m, k + TileK * i).group(load_group), loadA.cord2tma(0, TileK * n_batch))
                    for i in range(n_batch)
                ],
                asyncPort=self.prefetch,
            ),

            storeC_cord.bar(self._bar("store")).group(store_group),
        ]
        return insts
    
    # combinators
    def split(self, dim: int, div: int):
        # create N new schedules that split the given dim by div
        assert dim in (0, 1, 2), "dim must be 0 (M), 1 (N), or 2 (K)"
        assert self.MNK[dim] % div == 0, "Cannot split dimension that is not divisible by div"

        new_schedules = []
        for i in range(div):
            new_MNK = list(self.MNK)
            new_base = list(self.MNK_base)
            size = new_MNK[dim] // div
            base = new_base[dim] + i * size
            new_MNK[dim] = size
            new_base[dim] = base
            new_schedule = SchedGemv(
                self.Atom,
                ((new_base[0], new_MNK[0]),
                 (new_base[1], new_MNK[1]),
                 (new_base[2], new_MNK[2])),
                self.tmas,
                fold=self.fold,
                prefetch=self.prefetch,
                group=self.group,
            )
            new_schedules.append(new_schedule)
        split_schedule = ListSchedule(new_schedules, lead_bars={"load"}, tail_bars={"store"})
        split_schedule._bars = self._bars.copy()
        if self.num_sms is not None:
            split_schedule = split_schedule.place(self.num_sms, self.base_sm)
        else:
            split_schedule._apply_boundary_bars()
        return split_schedule

    def split_M(self, div: int):
        return self.split(0, div)
    def split_K(self, div: int):
        return self.split(2, div)

    def no_prefetch(self):
        self.prefetch = False
        return self

    def bar_release_count(self, role: str):
        if role != "store":
            return 0
        return self._bar_release_if_present(role, self.num_sms)

class SchedGemvRope(Schedule):
    def __init__(self,
                 MNK: tuple[int, int, int],
                 tmas: tuple[TmaTensor],
                 rope_table: RawAddress,
                 hist_seq_len: int,
                 ):
        super().__init__()
        self.Atom = Gemv_M64N8_ROPE_128
        self.MNK = MNK
        self.tmas = tmas

        MNK_base = []
        MNK_size = []
        assert len(MNK) == 3, "MNK must be a tuple of 3 dimensions"
        for dim in MNK:
            if isinstance(dim, int):
                MNK_base.append(0)
                MNK_size.append(dim)
            elif isinstance(dim, tuple) and len(dim) == 2:
                base, size = dim
                MNK_base.append(base)
                MNK_size.append(size)
            else:
                raise ValueError(f"Invalid MNK dimension: {dim}")
        
        self.MNK = MNK_size
        self.MNK_base = MNK_base
        self.rope_table = rope_table
        self.hist_seq_len = hist_seq_len

        self.fold = None
        self.prefetch = True
        self.sm_per_fold = None
        self.k_per_fold = None

    def _on_place(self):
        self.fold = self.num_sms // (self.MNK[0] // self.Atom.MNK[0])
        self.sm_per_fold = self.num_sms // self.fold
        self.k_per_fold = self.MNK[2] // self.fold
        self.validate()
    
    def validate(self):
        TileM, TileN, TileK = self.Atom.MNK
        M, N, K = self.MNK
        assert 128 % TileM == 0, "TileM must divide 128 for rope fusion"

        # verify fold
        assert self.num_sms % (M // TileM) == 0, f"SMS must be multiple of M tiles, got SMS={self.num_sms}, M={M}, TileM={TileM}"
        assert self.num_sms % self.fold == 0
        assert K % self.fold == 0
        assert self.sm_per_fold == (M // TileM), "Invalid fold for given SMS and M size"
        assert self.k_per_fold % TileK == 0, "Invalid fold for given K size"
    
    def schedule(self, sm: int):
        # TODO(zhiyuang): different SM mode?
        if sm < 0:
            return []

        TileM, TileN, TileK = self.Atom.MNK
        baseM, _, baseK = self.MNK_base
        n_batch = self.Atom.n_batch
        loadA, loadB, storeC = self.tmas

        m = baseM + (sm % self.sm_per_fold) * TileM
        k = baseK + (sm // self.sm_per_fold) * self.k_per_fold

        n_repeat = self.k_per_fold // (TileK * n_batch)

        insts = [
            self.Atom(self.k_per_fold // TileK, self.hist_seq_len, m % 128),
            self.rope_table,
            RepeatM.onSync(0, self._bar("load"), n_repeat,
                (loadB.cord(0, k).group(), loadB.cord2tma(0, TileK * n_batch)),
                *[
                    (loadA.cord(m, k + TileK * i).group(), loadA.cord2tma(0, TileK * n_batch))
                    for i in range(n_batch)
                ],
                asyncPort=self.prefetch,
            ),
            storeC.cord(0, self.hist_seq_len, m).bar(self._bar("store")).group(),
        ]
        return insts

    def bar_release_count(self, role: str):
        if role != "store":
            return 0
        return self._bar_release_if_present(role, self.num_sms)

class SchedRMSShared(Schedule):
    def __init__(self,
                 num_token: int,
                 epsilon: float,
                 tmas,
                 hidden_size: int | None = None,
                 group: bool = True,
                 embedding = None):
        super().__init__()
        self.num_token = num_token
        self.epsilon = epsilon
        self.tmas = tmas
        self.hidden_size = hidden_size
        self.group = group
        self.embedding = embedding

    def _on_place(self):
        assert self.num_token % self.num_sms == 0, "Number of tokens must be divisible by number of SMs"
        self.workload_per_sm = self.num_token // self.num_sms

    def _resolve_hidden_size(self):
        if self.hidden_size is not None:
            return self.hidden_size

        weight = self.tmas[0]
        if hasattr(weight, "size") and weight.size % 2 == 0:
            return weight.size // 2
        raise ValueError("SchedRMSShared requires hidden_size or a byte-sized weight TMA")

    def schedule(self, sm):
        if sm < 0:
            return []

        hidden_size = self._resolve_hidden_size()
        per_token_size = hidden_size * 2
        start_token_id = sm * self.workload_per_sm
        weight, load, store = self.tmas

        load = load \
            .cord(per_token_size * start_token_id) \
            .bar(self._bar("input")).group(self.group)
        store = store \
            .cord(per_token_size * start_token_id) \
            .bar(self._bar("output")).group(self.group)
        if self.embedding is not None:
            load.jump()
        
        return [
            select_rms_smem_instruction(hidden_size)(self.workload_per_sm, self.epsilon),
            weight.group(),
            self.embedding,
            load,
            store,
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)

class SchedRMS(Schedule):
    def __init__(self,
                 num_token: int,
                 epsilon: float,
                 input_glob: torch.Tensor,
                 output_glob: torch.Tensor,
                 weights_glob: torch.Tensor | None = None,
                 hidden_size: int | None = None,
                 use_glob: bool = False,
                 group: bool = True,
                 embedding = None):
        super().__init__()
        self.num_token = num_token
        self.epsilon = epsilon
        self.input_glob = input_glob
        self.output_glob = output_glob
        if weights_glob is None:
            weights_glob = torch.ones(
                self.input_glob.shape[-1],
                dtype=self.input_glob.dtype,
                device=self.input_glob.device,
            )
        self.weights_glob = weights_glob
        self.hidden_size = hidden_size if hidden_size is not None else input_glob.shape[-1]
        self.use_glob = use_glob
        self.group = group
        self.embedding = embedding

    def _on_place(self):
        assert self.num_token % self.num_sms == 0, "Number of tokens must be divisible by number of SMs"
        self.workload_per_sm = self.num_token // self.num_sms
        # TODO (zijian): residual store in case when rms starts from SM128, we should consider fuse in the kernel

    def schedule(self, sm):
        if sm < 0:
            return []

        if sm < 128:
            # regular rms path
            start_token_id = sm * self.workload_per_sm
            if self.use_glob:
                weight = RawAddress(self.weights_glob, 26)
                load = RawAddress(self.input_glob[start_token_id:start_token_id+self.workload_per_sm], 24)
                store = RawAddress(self.output_glob[start_token_id:start_token_id+self.workload_per_sm], 25)
                kernel = select_rms_glob_instruction(self.hidden_size)
            else:
                loadTensors = self.input_glob[start_token_id:start_token_id+self.workload_per_sm]
                if len(loadTensors) == 1:
                    loadTensors = loadTensors[0]
                storeTensors = self.output_glob[start_token_id:start_token_id+self.workload_per_sm]
                if len(storeTensors) == 1:
                    storeTensors = storeTensors[0]
                
                weight = TmaLoad1D(self.weights_glob)
                load = TmaLoad1D(loadTensors)
                store = TmaStore1D(storeTensors)
                kernel = select_rms_smem_instruction(self.hidden_size)
                # TODO(zhiyuang): recheck this when refector on repeat is done.
                if self.embedding is not None:
                    load.jump()

            load = load.bar(self._bar("input")).group(self.group)
            store = store.bar(self._bar("output")).group(self.group)
            
            insts = [
                kernel(self.workload_per_sm, self.epsilon),
                weight,
                self.embedding,
                load,
                store,
            ]
        
        return insts

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)

class SchedSiLU(Schedule):
    def __init__(self,
                 base_raw_slot: int,
                 num_token: int,
                 output_size: int,
                 gate_glob: torch.Tensor,
                 up_glob: torch.Tensor,
                 out_glob: torch.Tensor):
        super().__init__()
        self.base_raw_slot = base_raw_slot
        self.num_token = num_token
        self.output_size = output_size
        # NOTE[zijian]: pass in first row only to bypass contiguous check
        self.gate_glob = gate_glob
        self.up_glob = up_glob
        self.out_glob = out_glob

    def _on_place(self):
        assert self.output_size % self.num_sms == 0, "Output size must be divisible by number of SMs"
        self.workload = self.output_size // self.num_sms
    
    def schedule(self, sm):
        if sm < 0:
            return []
        k_offset = sm * self.workload
        gate_addr = RawAddress(self.gate_glob[k_offset:], self.base_raw_slot).bar(self._bar("gate"))
        up_addr = RawAddress(self.up_glob[k_offset:], self.base_raw_slot+1).bar(self._bar("up"))
        out_addr = RawAddress(self.out_glob[k_offset:], self.base_raw_slot+2).bar(self._bar("out")).writeback()
        insts = [
            SILU_MUL_F16_K_12288(self.num_token, self.workload),
            gate_addr,
            up_addr,
            out_addr,
        ]
        return insts

    def bar_release_count(self, role: str):
        if role not in ("gate", "up", "out"):
            return 0
        return self._bar_release_if_present(role, self.num_sms)

    def split(self, div: int, bars: list[tuple[int]]):
        assert self.output_size % div == 0, "Cannot split K dimension evenly"
        new_schedules = []
        new_output_size = self.output_size // div
        for i in range(div):
            gate_bar, up_bar, out_bar = bars[i]
            new_schedule = SchedSiLU(
                self.base_raw_slot + i * 3,
                self.num_token,
                new_output_size,
                self.gate_glob[i * new_output_size:(i + 1) * new_output_size],
                self.up_glob[i * new_output_size:(i + 1) * new_output_size],
                self.out_glob[i * new_output_size:(i + 1) * new_output_size],
            )
            new_schedule.bar("gate", gate_bar).bar("up", up_bar).bar("out", out_bar)
            new_schedules.append(new_schedule)
        split_schedule = ListSchedule(new_schedules, lead_bars={"gate", "up"}, tail_bars={"out"})
        split_schedule._bars = self._bars.copy()
        if self.num_sms is not None:
            split_schedule = split_schedule.place(self.num_sms, self.base_sm)
        else:
            split_schedule._apply_boundary_bars()
        return split_schedule

class SchedSmemSiLUInterleaved(Schedule):
    def __init__(self,
                 num_token: int,
                 gate_glob: torch.Tensor,
                 up_glob: torch.Tensor,
                 out_glob: torch.Tensor):
        super().__init__()
        self.num_token = num_token
        self.gate_glob = gate_glob
        self.up_glob = up_glob
        self.out_glob = out_glob

    def _on_place(self):
        assert self.num_token % self.num_sms == 0, "Number of tokens must be divisible by number of SMs"
        self.tokens_per_sm = self.num_token // self.num_sms

    def schedule(self, sm):
        if sm < 0:
            return []

        start_token_id = sm * self.tokens_per_sm
        end_token_id = (sm + 1) * self.tokens_per_sm
        insts = []
        for i in range(start_token_id, end_token_id):
            gate = TmaLoad1D(self.gate_glob[i])
            if i == start_token_id:
                gate = gate.bar(self._bar("input")).group()

            insts.extend([
                SILU_MUL_SHARED_BF16_K_4096_INTER(1),
                TmaStore1D(self.out_glob[i]).bar(self._bar("output")).group(),
                gate,
                TmaLoad1D(self.up_glob[i]),
            ])
        return insts

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_token)

class SchedRegSiLUFused(Schedule):
    def __init__(self,
                 num_token: int,
                 store_tma: TmaTensor,
                 reg_gate: int,
                 reg_up: int,
                 base_offset: int,
                 stride: int):
        super().__init__()
        self.num_token = num_token
        self.store_tma = store_tma
        self.reg_gate = reg_gate
        self.reg_up = reg_up
        self.base_offset = base_offset
        self.stride = stride

    def schedule(self, sm):
        if sm < 0:
            return []

        return [
            SILU_MUL_SHARED_BF16_K_64_SW128(self.num_token),
            self.store_tma.cord(0, self.base_offset + sm * self.stride).bar(self._bar("output")).group(),
            RegLoad(self.reg_gate),
            RegLoad(self.reg_up),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)

class SchedSmemSiLU_K_4096_N_1(Schedule):
    def __init__(self,
                 gate_tma: TmaLoad1D,
                 up_tma: TmaLoad1D,
                 out_tma: TmaStore1D,
                 base_sm: int,
                 ):
        super().__init__()
        self.base_sm = base_sm

        self.gate_tma = gate_tma
        self.up_tma = up_tma
        self.out_tma = out_tma

    def __call__(self, sm: int):
        return self.schedule(sm)
    
    def schedule(self, sm):
        if sm != self.base_sm:
            # only 1 SM is needed
            return []
        insts = [
            SILU_MUL_SHARED_BF16_K_4096(),
            self.out_tma,
            self.gate_tma,
            self.up_tma,
        ]
        return insts

class SchedArgmax(Schedule):
    def __init__(self,
                 num_token: int,
                 logits_slice: int,
                 num_slice: int,
                 AtomPartial,
                 AtomReduce,
                 matLogits: list[torch.Tensor],
                 matOutVal: torch.Tensor,
                 matOutIdx: torch.Tensor,
                 matFinalOut: torch.Tensor):
        super().__init__()
        self.num_token = num_token
        self.logits_slice = logits_slice
        self.num_slice = num_slice
        self.matLogits = matLogits
        self.matOutVal = matOutVal
        self.matOutIdx = matOutIdx
        self.matFinalOut = matFinalOut
        self.AtomPartial = AtomPartial
        self.AtomReduce = AtomReduce
    
    def _on_place(self):
        self.validate()

    def validate(self):
        assert len(self.matLogits) == self.num_slice, "Number of logits slices must match vocab size and slice size"
        assert self.matOutVal.shape == (self.num_token, self.num_sms)
        assert self.matOutIdx.shape == (self.num_token, self.num_sms)
        assert self.matFinalOut.shape == (self.num_token,)

        sm_per_slice = self.num_sms // self.num_slice
        assert self.num_sms % self.num_slice == 0, "Number of SMs must be divisible by number of slices for current schedule design"
        c_per_sm = self.logits_slice // sm_per_slice 
        assert self.logits_slice % c_per_sm == 0, "Logits slice size must be divisible by chunk size per SM for current schedule design"
        assert self.AtomPartial.CHUNK_SIZE == c_per_sm, f"AtomPartial chunk size missmatch, expected {c_per_sm}, got {self.AtomPartial.CHUNK_SIZE}"
        assert self.AtomPartial.I_STRIDE == self.logits_slice, f"AtomPartial i_stride missmatch, expected {self.logits_slice}, got {self.AtomPartial.I_STRIDE}"
        assert self.AtomPartial.SMS == self.num_sms, f"AtomPartial SMS missmatch, expected {self.num_sms}, got {self.AtomPartial.SMS}"
        assert self.AtomReduce.CHUNK_SIZE == c_per_sm, f"AtomReduce chunk size missmatch, expected {c_per_sm}, got {self.AtomReduce.CHUNK_SIZE}"
        assert self.AtomReduce.SMS == self.num_sms, f"AtomReduce SMS missmatch, expected {self.num_sms}, got {self.AtomReduce.SMS}"

    def schedule(self, sm):
        if sm < 0:
            return []
        # decide which slice
        sm_per_slice = self.num_sms // self.num_slice
        slice_idx = sm // sm_per_slice
        c_per_sm = self.logits_slice // sm_per_slice
        slice_ofst = (sm % sm_per_slice) * c_per_sm
        insts = [
            self.AtomPartial(self.num_token),
            # FIXME(zhiyuang): the index 0 for batched mode?
            RawAddress(self.matLogits[slice_idx][0,slice_ofst], 24).bar(self._bar("load")),
            RawAddress(self.matOutVal[0,sm], 25).bar(self._bar("val")).writeback(),
            RawAddress(self.matOutIdx[0,sm], 26).bar(self._bar("idx")).writeback(),
        ]
        if sm >= self.num_token:
            return insts

        insts += [
            self.AtomReduce(1),

            RawAddress(self.matOutVal[sm], 27).bar(self._bar("val")),
            RawAddress(self.matOutIdx[sm], 28).bar(self._bar("idx")),
            RawAddress(self.matFinalOut[sm], 29).bar(self._bar("final")).writeback(),
        ]
        return insts

    def bar_release_count(self, role: str):
        if role == "val" or role == "idx":
            return self._bar_release_if_present(role, self.num_sms)
        if role == "final":
            return self._bar_release_if_present(role, min(self.num_token, self.num_sms))
        return 0


def interleave(*schedules):
    final = []
    for scheds in zip(*schedules):
        for sched in scheds:
            final.append(sched)
    return final
