from .runtime import *
from .launcher import *

class Schedule:
    def map_sm(self, sm: int):
        # This function decides how to map SM to SM ID. this can create scheudle
        # that is not strictly round-robin, e.g., for hierarchical scheduling,
        # we may want to map SMs in the same fold together.
        return sm
    def schedule(self, sm: int):
        raise NotImplementedError("Schedule.schedule() must be implemented by subclass")
    def __call__(self, sm: int):
        mapped_sm = self.map_sm(sm)
        return self.schedule(mapped_sm)

class SchedCopy(Schedule):
    def __init__(self, num_sms: int,
                 tmas, cords = None,
                 size = None,
                 before_copy = None,
                 count = 1,
                 base_sm = 0):
        self.sms = num_sms
        self.tmas = tmas
        self.cords = cords
        self.count = count
        self.base_sm = base_sm
        if self.cords is None:
            self.cords = [None for _ in range(len(tmas))]
        assert len(self.tmas) == len(self.cords), "Number of TMA tensors must match number of cord specifications"
        self.before_copy = before_copy


        self.barrier_store = None
        self.barrier_load = None
        if size is None:
            assert tmas[0].size == tmas[1].size, "Size must be specified when load and store TMA sizes do not match"
            size = tmas[0].size
        self.size = size

    def map_sm(self, sm):
        sm -= self.base_sm
        if sm < 0 or sm >= self.sms:
            return -1
        else:
            return sm

    def schedule(self, sm: int):
        if sm < 0:
            return []

        load, store = self.tmas
        cord_load, cord_store = self.cords
        load = load if cord_load is None else cord_load(sm, load)
        store = store if cord_store is None else cord_store(sm, store)
        if self.before_copy is not None:
            load.jump()

        return [
            Copy(1, size = self.size),

            self.before_copy,
            load.bar(self.barrier_load),
            store.bar(self.barrier_store),
        ]

    def store_bar(self, bar_id: int):
        self.barrier_store = bar_id
        return self
    def load_bar(self, bar_id: int):
        self.barrier_load = bar_id
        return self

class SchedRope(Schedule):
    def __init__(self, Atom, num_sms, tmas, cords = None,
                 base_sm = 0):
        self.sms = num_sms
        self.Atom = Atom
        self.tmas = tmas
        self.cords = cords
        self.base_sm = base_sm

        if self.cords is None:
            self.cords = [None for _ in range(len(tmas))]
        assert len(self.tmas) == len(self.cords), "Number of TMA tensors must match number of cord specifications"

        self.bar_store = None

    def map_sm(self, sm):
        sm -= self.base_sm
        if sm < 0 or sm >= self.sms:
            return -1
        else:
            return sm

    def schedule(self, sm: int):
        if sm < 0:
            return []
        table, load, store = [
            cord(sm, tma) if cord is not None else tma
            for cord, tma in zip(self.cords, self.tmas)
        ]

        return [
            self.Atom(),

            table,
            load,
            store.bar(self.bar_store).group(),
        ]

    def store_bar(self, bar_id: int):
        self.bar_store = bar_id
        return self

class SchedAttentionDecoding(Schedule):
    # for decoding, the Qtile len will always be 1
    def __init__(self, reqs: int, seq_len: int,
                 KV_BLOCK_SIZE : int, NUM_KV_HEADS : int,
                 matO : torch.Tensor,
                 tmas):
        self.reqs = reqs
        self.seq_len = seq_len
        self.num_heads = NUM_KV_HEADS
        self.matO = matO
        self.tmas = tmas
        self.sms = reqs * NUM_KV_HEADS
        self.block_size = KV_BLOCK_SIZE

        self.q_barrier = None
        self.k_barrier = None
        self.o_barrier = None

    def schedule(self, sm: int):
        if sm >= self.sms:
            return []

        req = sm // self.num_heads
        head = sm % self.num_heads

        tQ, tK, tV = self.tmas

        num_kv_blocks = (self.seq_len + self.block_size - 1) // self.block_size
        seq_len_last_block = self.seq_len % self.block_size

        # we only handle a single Q token here
        insts = [
            ATTENTION_M64N64K16_F16_F32_64_64_hdim(num_kv_blocks, seq_len_last_block, need_norm=False, need_rope=False),
            tQ.cord(req, head).bar(self.q_barrier).group(),
            RepeatM.on(num_kv_blocks - 1,
                # this k-barrier will also barrier following V load
                [tK.cord(req, 0, head, 0).group(), tK.cord2tma(0, self.block_size, 0, 0)],
                [tV.cord(req, 0, head, 0).group(), tV.cord2tma(0, self.block_size, 0, 0)],
            ),
            # TODO(zhiyuang): reuse the accumulator register
            # only the last block has new generated KV cache
            tK.cord(req, self.block_size * (num_kv_blocks - 1), head, 0).bar(self.k_barrier).group(),
            tV.cord(req, self.block_size * (num_kv_blocks - 1), head, 0).group(),
            TmaStore1D(self.matO[req, head, ...], numSlots = 2).bar(self.o_barrier).group(),
        ]
        return insts

    # combinators
    def q_bar(self, bar_id: int):
        self.q_barrier = bar_id
        return self
    def k_bar(self, bar_id: int):
        self.k_barrier = bar_id
        return self
    def o_bar(self, bar_id: int):
        self.o_barrier = bar_id
        return self


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

        self.sms = reqs * QKVHdim[1]

        self.q_barrier = None
        self.k_barrier = None
        self.o_barrier = None

    def describe(self):
        print(f"SchedAttention: reqs={self.reqs}, active_new_len={self.active_new_len}, QKVHdim={self.QKVHdim}, QKVSeqlen={self.QKVSeqlen}, QKVTile={self.QKVTile}, sms={self.sms}")
    
    def schedule(self, sm: int):
        if sm >= self.sms:
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

        port = 0 if self.k_barrier is None else 1

        insts = []
        for q in range(0, self.active_new_len, QTile):
            insts += [
                ATTENTION_M64N64K16_F16_F32_64_64_hdim(min(self.active_new_len-q, QTile), hist_len=q+self.cached_seq_len, need_norm=self.need_norm, need_rope=self.need_rope),
                tQ.cord(req, q * HEAD_GROUP_SIZE, head, 0).bar(self.q_barrier).group(),
                self.rope_table if self.need_rope else [],
                # FIXME (zijian): this calculation should separate cached kv and new kv
                RepeatM.on((self.cached_seq_len + self.active_new_len + KVTile - 1) // KVTile,
                    # this k-barrier will also barrier following V load
                    [tK.cord(req, 0, head, 0).bar(self.k_barrier).group(), tK.cord2tma(0, KVTile, 0, 0)],
                    [tV.cord(req, 0, head, 0).group(), tV.cord2tma(0, KVTile, 0, 0)],
                ),
                tO.cord(req, q * HEAD_GROUP_SIZE, head, 0).port(1).bar(self.o_barrier).group(),
            ]
        return insts

    # combinators
    def q_bar(self, bar_id: int):
        self.q_barrier = bar_id
        return self
    def k_bar(self, bar_id: int):
        self.k_barrier = bar_id
        return self
    def o_bar(self, bar_id: int):
        self.o_barrier = bar_id
        return self


class SchedGemv(Schedule):
    def __init__(self, Atom, sms: int,
                 MNK: tuple[int, int, int],
                 tmas: tuple[TmaTensor],
                 fold : int | None = None,
                 exec = True,
                 prefetch = True,
                 group = True,
                 cordconv = None,
                 base_sm = 0):
        self.Atom = Atom
        self.MNK = MNK
        self.tmas = tmas
        if cordconv is None:
            cordconv = [None for _ in range(len(tmas))]
        assert len(tmas) == len(cordconv), "Number of TMA tensors must match number of cord converters"
        self.cordconv = cordconv

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

        M, N, K = self.MNK
        self.sms = sms
        self.base_sm = base_sm
        
        # optional barrier for vector
        self.ld_barrier = None
        self.st_barrier = None
        self.fold = fold
        self.prefetch = prefetch
        self.group = group

        # only check folds when we want to execute them
        if exec == True:
            # auto detect fold if possible
            if fold is None:
                assert sms % (M // TileM) == 0, f"SMS must be multiple of M tiles when auto folding, got SMS={sms}, M={M}, TileM={TileM}"
                self.fold = sms // (M // TileM)
            else:
                self.fold = fold

            self.sm_per_fold = sms // self.fold
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
        assert self.sms % self.fold == 0
        assert K % self.fold == 0
        assert self.sm_per_fold == (M // TileM), "Invalid fold for given SMS and M size"
        assert self.k_per_fold % TileK == 0, "Invalid fold for given K size"

        # verify storeC. if fold > 1, storeC must be reduction
        assert len(self.tmas) == 3, "Expect at 3 TMA tensors: loadA, loadB, storeC"
        if self.fold > 1:
            assert self.tmas[-1].mode == "reduce", f"storeC must be reduction mode when fold > 1, got mode {self.tmas[-1].mode}"

    def map_sm(self, sm):
        sm -= self.base_sm
        if sm < 0 or sm >= self.sms:
            return -1
        else:
            return sm
    
    def schedule(self, sm: int):
        # TODO(zhiyuang): different SM mode?
        if sm == -1:
            return []

        TileM, TileN, TileK = self.Atom.MNK
        baseM, _, baseK = self.MNK_base
        n_batch = self.Atom.n_batch

        loadA, loadB, storeC = self.tmas
        convA, convB, convC = self.cordconv

        m = baseM + (sm % self.sm_per_fold) * TileM
        k = baseK + (sm // self.sm_per_fold) * self.k_per_fold

        n_repeat = self.k_per_fold // (TileK * n_batch)

        # TODO(zhiyuang): more detailed group control
        load_group = self.group and (self.ld_barrier is not None)
        store_group = self.group and (self.st_barrier is not None)

        if convC is None:
            storeC_cord = storeC.cord(0, m)
        else:
            storeC_cord = convC(m, storeC)

        insts = [
            self.Atom(self.k_per_fold // TileK),

            RepeatM.onSync(0, self.ld_barrier, n_repeat,
                (loadB.cord(0, k).group(load_group), loadB.cord2tma(0, TileK * n_batch)),
                *[
                    (loadA.cord(m, k + TileK * i).group(load_group), loadA.cord2tma(0, TileK * n_batch))
                    for i in range(n_batch)
                ],
                asyncPort=self.prefetch,
            ),

            storeC_cord.bar(self.st_barrier).group(store_group),
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
            # this new schedgemv is against the check
            new_schedule = SchedGemv(
                self.Atom,
                self.sms,
                ((new_base[0], new_MNK[0]),
                 (new_base[1], new_MNK[1]),
                 (new_base[2], new_MNK[2])),
                self.tmas,
                fold=self.fold,
                prefetch=self.prefetch,
                group=self.group,
                base_sm=self.base_sm,
            )
            new_schedules.append(new_schedule)
        return new_schedules

    def split_M(self, div: int):
        return self.split(0, div)
    def split_K(self, div: int):
        return self.split(2, div)

    def load_bar(self, bar_id: int):
        self.ld_barrier = bar_id
        return self
    def store_bar(self, bar_id: int):
        self.st_barrier = bar_id
        return self
    def no_prefetch(self):
        self.prefetch = False
        return self

class SchedGemvRope(Schedule):
    def __init__(self, sms: int, 
                 MNK: tuple[int, int, int],
                 tmas: tuple[TmaTensor],
                 rope_table: RawAddress,
                 hist_seq_len: int,
                 base_sm = 0):
        self.Atom = Gemv_M64N8_ROPE_128
        self.sms = sms
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
        self.base_sm = base_sm

        self.ld_barrier = None
        self.st_barrier = None
        self.fold = sms // (self.MNK[0] // self.Atom.MNK[0])
        self.prefetch = True
        self.sm_per_fold = sms // self.fold
        self.k_per_fold = self.MNK[2] // self.fold
        self.validate()
    
    def validate(self):
        TileM, TileN, TileK = self.Atom.MNK
        M, N, K = self.MNK
        assert 128 % TileM == 0, "TileM must divide 128 for rope fusion"

        # verify fold
        assert self.sms % (M // TileM) == 0, f"SMS must be multiple of M tiles, got SMS={self.sms}, M={M}, TileM={TileM}"
        assert self.sms % self.fold == 0
        assert K % self.fold == 0
        assert self.sm_per_fold == (M // TileM), "Invalid fold for given SMS and M size"
        assert self.k_per_fold % TileK == 0, "Invalid fold for given K size"
    
    def map_sm(self, sm):
        sm -= self.base_sm
        if sm < 0 or sm >= self.sms:
            return -1
        else:
            return sm
    
    def schedule(self, sm: int):
        # TODO(zhiyuang): different SM mode?
        if sm == -1:
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
            RepeatM.onSync(0, self.ld_barrier, n_repeat,
                (loadB.cord(0, k).group(), loadB.cord2tma(0, TileK * n_batch)),
                *[
                    (loadA.cord(m, k + TileK * i).group(), loadA.cord2tma(0, TileK * n_batch))
                    for i in range(n_batch)
                ],
                asyncPort=self.prefetch,
            ),
            storeC.cord(0, self.hist_seq_len, m).bar(self.st_barrier).group(),
        ]
        return insts

    def load_bar(self, bar_id: int):
        self.ld_barrier = bar_id
        return self
    def store_bar(self, bar_id: int):
        self.st_barrier = bar_id
        return self

class SchedRMSShared(Schedule):
    def __init__(self,
                 sms: int,
                 num_token: int,
                 epsilon: float,
                 tmas,
                 base_sm: int = 0,
                 group: bool = True,
                 embedding = None):
        self.sms = sms
        self.num_token = num_token
        self.epsilon = epsilon
        self.tmas = tmas
        self.ld_bar = None
        self.st_bar = None
        self.base_sm = base_sm
        self.group = group
        self.embedding = embedding

        assert num_token % sms == 0, "Number of tokens must be divisible by number of SMs"
        self.workload_per_sm = num_token // sms

    def map_sm(self, sm):
        sm -= self.base_sm
        if sm < 0 or sm >= self.sms:
            return -1
        else:
            return sm

    def schedule(self, sm):
        if sm == -1:
            return []

        per_token_size = 4096 * 2
        start_token_id = sm * self.workload_per_sm
        weight, load, store = self.tmas

        load = load \
            .cord(per_token_size * start_token_id) \
            .bar(self.ld_bar).group(self.group)
        store = store \
            .cord(per_token_size * start_token_id) \
            .bar(self.st_bar).group(self.group)
        if self.embedding is not None:
            load.jump()
        
        return [
            RMS_NORM_F16_K_4096_SMEM(self.workload_per_sm, self.epsilon),
            weight.group(),
            self.embedding,
            load,
            store,
        ]

    def i_bar(self, bar_id: int):
        self.ld_bar = bar_id
        return self
    def o_bar(self, bar_id: int):
        self.st_bar = bar_id
        return self

class SchedRMS(Schedule):
    def __init__(self,
                 sms: int,
                 num_token: int,
                 epsilon: float,
                 weights_glob: torch.Tensor,
                 input_glob: torch.Tensor,
                 output_glob: torch.Tensor,
                 base_sm: int,
                 use_glob: bool = False,
                 group: bool = True,
                 embedding = None):
        self.sms = sms
        self.num_token = num_token
        self.epsilon = epsilon
        self.weights_glob = weights_glob
        self.input_glob = input_glob
        self.output_glob = output_glob
        self.ld_bar = None
        self.st_bar = None
        self.base_sm = base_sm
        self.use_glob = use_glob
        self.group = group
        self.embedding = embedding

        assert num_token % sms == 0, "Number of tokens must be divisible by number of SMs"
        self.workload_per_sm = num_token // sms
        # TODO (zijian): residual store in case when rms starts from SM128, we should consider fuse in the kernel

    def map_sm(self, sm):
        sm -= self.base_sm
        if sm < 0 or sm >= self.sms:
            return -1
        else:
            return sm

    def schedule(self, sm):
        if sm == -1:
            return []

        if sm < 128:
            # regular rms path
            start_token_id = sm * self.workload_per_sm
            if self.use_glob:
                weight = RawAddress(self.weights_glob, 26)
                load = RawAddress(self.input_glob[start_token_id:start_token_id+self.workload_per_sm], 24)
                store = RawAddress(self.output_glob[start_token_id:start_token_id+self.workload_per_sm], 25)
                kernel = RMS_NORM_F16_K_4096
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
                kernel = RMS_NORM_F16_K_4096_SMEM
                # TODO(zhiyuang): recheck this when refector on repeat is done.
                if self.embedding is not None:
                    load.jump()

            load = load.bar(self.ld_bar).group(self.group)
            store = store.bar(self.st_bar).group(self.group)
            
            insts = [
                kernel(self.workload_per_sm, self.epsilon),
                weight,
                self.embedding,
                load,
                store,
            ]
                
        return insts

    def i_bar(self, bar_id: int):
        self.ld_bar = bar_id
        return self

    def o_bar(self, bar_id: int):
        self.st_bar = bar_id
        return self

class SchedSiLU(Schedule):
    def __init__(self,
                 base_raw_slot: int,
                 sms: int,
                 num_token: int,
                 output_size: int,
                 gate_glob: torch.Tensor,
                 up_glob: torch.Tensor,
                 out_glob: torch.Tensor,
                 base_sm: int):
        self.base_raw_slot = base_raw_slot
        self.sms = sms
        self.num_token = num_token
        self.output_size = output_size
        self.workload = output_size // sms
        # NOTE[zijian]: pass in first row only to bypass contiguous check
        self.gate_glob = gate_glob
        self.up_glob = up_glob
        self.out_glob = out_glob
        self.base_sm = base_sm
    
        # NOTE[zijian]: should only need either gate or up
        self.gate_bar = None
        self.up_bar = None
        self.out_bar = None
    
    def map_sm(self, sm):
        sm -= self.base_sm
        if sm < 0 or sm >= self.sms:
            return -1
        else:
            return sm

    def add_gate_bar(self, bar_id: int):
        self.gate_bar = bar_id
        return self
    def add_up_bar(self, bar_id: int):
        self.up_bar = bar_id
        return self
    def add_out_bar(self, bar_id: int):
        self.out_bar = bar_id
        return self
    
    def schedule(self, sm):
        if sm == -1:
            return []
        k_offset = sm * self.workload
        gate_addr = RawAddress(self.gate_glob[k_offset:], self.base_raw_slot).bar(self.gate_bar)
        up_addr = RawAddress(self.up_glob[k_offset:], self.base_raw_slot+1).bar(self.up_bar)
        out_addr = RawAddress(self.out_glob[k_offset:], self.base_raw_slot+2).bar(self.out_bar).writeback()
        insts = [
            SILU_MUL_F16_K_12288(self.num_token, self.workload),
            gate_addr,
            up_addr,
            out_addr,
        ]
        return insts

    def split(self, div: int, bars: list[tuple[int]]):
        assert self.output_size % div == 0, "Cannot split K dimension evenly"
        new_schedules = []
        new_output_size = self.output_size // div
        for i in range(div):
            gate_bar, up_bar, out_bar = bars[i]
            new_schedule = SchedSiLU(
                self.base_raw_slot + i * 3,
                self.sms,
                self.num_token,
                new_output_size,
                self.gate_glob[i * new_output_size:(i + 1) * new_output_size],
                self.up_glob[i * new_output_size:(i + 1) * new_output_size],
                self.out_glob[i * new_output_size:(i + 1) * new_output_size],
                self.base_sm,
            )
            new_schedule.add_gate_bar(gate_bar).add_up_bar(up_bar).add_out_bar(out_bar)
            new_schedules.append(new_schedule)
        return new_schedules

class SchedSmemSiLU_K_4096_N_1(Schedule):
    def __init__(self,
                 gate_tma: TmaLoad1D,
                 up_tma: TmaLoad1D,
                 out_tma: TmaStore1D,
                 base_sm: int,
                 ):
        self.base_sm = base_sm

        self.gate_tma = gate_tma
        self.up_tma = up_tma
        self.out_tma = out_tma
    
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
                 num_sms: int,
                 num_token: int,
                 logits_slice: int,
                 num_slice: int,
                 AtomPartial,
                 AtomReduce,
                 matLogits: list[torch.Tensor],
                 matOutVal: torch.Tensor,
                 matOutIdx: torch.Tensor,
                 matFinalOut: torch.Tensor):
        self.num_token = num_token
        self.num_sms = num_sms
        self.logits_slice = logits_slice
        self.num_slice = num_slice
        self.matLogits = matLogits
        self.matOutVal = matOutVal
        self.matOutIdx = matOutIdx
        self.matFinalOut = matFinalOut
        self.AtomPartial = AtomPartial
        self.AtomReduce = AtomReduce
        self.ld_bar = None
        self.idx_bar = None
        self.val_bar = None
        self.final_bar = None
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

    def ld_barrier(self, bar_id: int):
        self.ld_bar = bar_id
        return self
    def idx_barrier(self, bar_id: int):
        self.idx_bar = bar_id
        return self
    def val_barrier(self, bar_id: int):
        self.val_bar = bar_id
        return self
    def final_barrier(self, bar_id: int):
        self.final_bar = bar_id
        return self

    def schedule(self, sm):
        if sm >= self.num_sms:
            return []
        # decide which slice
        sm_per_slice = self.num_sms // self.num_slice
        slice_idx = sm // sm_per_slice
        c_per_sm = self.logits_slice // sm_per_slice
        slice_ofst = (sm % sm_per_slice) * c_per_sm
        insts = [
            self.AtomPartial(self.num_token),
            # FIXME(zhiyuang): the index 0 for batched mode?
            RawAddress(self.matLogits[slice_idx][0,slice_ofst], 24).bar(self.ld_bar),
            RawAddress(self.matOutVal[0,sm], 25).bar(self.val_bar).writeback(),
            RawAddress(self.matOutIdx[0,sm], 26).bar(self.idx_bar).writeback(),
        ]
        if sm >= self.num_token:
            return insts

        insts += [
            self.AtomReduce(1),

            RawAddress(self.matOutVal[sm], 27).bar(self.val_bar),
            RawAddress(self.matOutIdx[sm], 28).bar(self.idx_bar),
            RawAddress(self.matFinalOut[sm], 29).bar(self.final_bar).writeback(),
        ]
        return insts


def interleave(*schedules):
    final = []
    for scheds in zip(*schedules):
        for sched in scheds:
            final.append(sched)
    return final
