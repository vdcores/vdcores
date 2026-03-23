from .launcher import *
from .schedule import *
from .util import tensor_diff

from functools import partial

def uniform_rand_scaled(*shape: int, dtype=torch.float16, device='cuda', scale=0.1):
    return (torch.rand(*shape, dtype=dtype, device=device) - 0.5) * scale

# get dims other than the specified index and last one
def get_other_dims(dim: int, i: int):
    return [j for j in range(dim-1) if j != i and j - dim != i]

# cord functions for loading Q for GQA
def tma_gqa_load_q(mat: torch.Tensor, tileK: int, tileN: int):
    # [HEAD_DIM[0], HEAD_GROUP_SIZE, REP * HEAD_DIM[1] * NUM_KV_HEAD]
    assert mat.element_size() == 2, "Only support float16/bfloat16 output"
    assert tileN == 64, "tileN must be 64"
    # getting the mat size
    NUM_REQ, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM = mat.shape
    assert tileK == HEAD_DIM, "tileK must match HEAD_DIM"
    assert HEAD_DIM % 64 == 0, "HEAD_DIM must be a multiple of 64"
    assert 64 % HEAD_GROUP_SIZE == 0, "HEAD_GROUP_SIZE must divide the 64-row Q tile"

    # this will dup for 4 times, due to 0 in strides, do not know how tma engine will handle it
    rope_tiles = HEAD_DIM // 64
    q_tile_repeat = 64 // HEAD_GROUP_SIZE
    glob_dims = [64, HEAD_GROUP_SIZE, q_tile_repeat, rope_tiles, NUM_REQ * NUM_KV_HEAD]
    glob_strides = [HEAD_DIM * 2, 0, 64 * 2, HEAD_DIM * HEAD_GROUP_SIZE * 2]
    box_dims = [64, HEAD_GROUP_SIZE, q_tile_repeat, rope_tiles, 1]

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

def cord_gqa_load_q(mat: torch.Tensor, rank: int):
    assert rank == 5, "Only support 5D TMA load for load Q"
    NUM_KV_HEAD = mat.shape[1]

    def cfunc(*cords):
        assert len(cords) == 2, f"cords should be (req, head), but got {cords}"
        return [0, 0, 0, cords[0] * NUM_KV_HEAD + cords[1]]
    return cfunc

# mat is always pytorch-major
def cord_func_MN_major(mat: torch.Tensor, rank: int, iK = -2):
    assert iK != -1 and iK < len(mat.shape) - 1, "iK must not be the last dim"

    def cord_func(*cords):
        assert len(mat.shape) == len(cords), f"cords length {len(cords)} does not match mat rank {len(mat.shape)}"
        
        if rank == 2:
            assert iK == -2
            return [cords[-1], cords[-2]]
        else:
            other_dims = get_other_dims(len(mat.shape), iK)
            other_strides = [mat.stride(i) for i in other_dims]
            other_cords = [cords[i] for i in other_dims]

            rest = np.dot(other_strides, other_cords) // mat.stride(other_dims[-1])
            rest = int(rest)

            if rank == 3:
                return [cords[-1], cords[iK], rest]
            elif rank == 4:
                assert iK == -2
                return [0, 0, cords[-1] // 64, cords[-2] // 8]
            elif rank == 5:
                # fix 0 for first dim
                return [0, cords[-1] // 64, cords[iK] // 8, rest]
            else:
                raise ValueError(f"Unsupported rank {rank} for mn-major cord function")
    return cord_func

def cord_func_MN_major_cord2(mat: torch.Tensor, rank: int, iK = -2):
    assert iK != -1 and iK < len(mat.shape) - 1, "iK must not be the last dim"

    def cord_func(*cords):
        assert len(cords) == 2, f"expect input cords to be length 2, but got {len(cords)}"
        cords = [0] * (len(mat.shape) - 2) + list(cords)
        
        if rank == 2:
            assert iK == -2
            return [cords[-1], cords[-2]]
        else:
            other_dims = get_other_dims(len(mat.shape), iK)
            other_strides = [mat.stride(i) for i in other_dims]
            other_cords = [cords[i] for i in other_dims]

            rest = np.dot(other_strides, other_cords) // mat.stride(other_dims[-1])
            rest = int(rest)

            if rank == 3:
                return [cords[-1], cords[iK], rest]
            elif rank == 4:
                assert iK == -2
                return [0, 0, cords[-1] // 64, cords[-2] // 8]
            elif rank == 5:
                # fix 0 for first dim
                return [0, cords[-1] // 64, cords[iK] // 8, rest]
            else:
                raise ValueError(f"Unsupported rank {rank} for mn-major cord function")
    return cord_func

def build_tma_wgmma_mn(mat: torch.Tensor, tileM: int, tileK: int, iK = -2):
    assert iK != -1 and iK < len(mat.shape) - 1, "iK must not be the last dim"
    # build 4d by default
    assert len(mat.shape) >= 2, "Input matrix must be at least 2D"
    K, M = mat.shape[iK], mat.shape[-1]
    elsize = mat.element_size()

    blockM = 128 // elsize
    blockK = 8

    assert tileM % blockM == 0, f"tileM {tileM} must be multiple of blockM {blockM}"

    if blockM != tileM:
        global_dims = [blockM, blockK, M // blockM, K // blockK]
        global_strides = [mat.stride(iK), blockM, mat.stride(iK) * blockK]
        box_dims = [blockM, blockK, tileM // blockM, tileK // blockK]
    else:
        global_dims = [M, K]
        global_strides = [mat.stride(iK)]
        box_dims = [tileM, tileK]
    
    if len(mat.shape) > 2:
        # collapse other dims
        size_otherthan_k_or_m = 1
        other_dims = get_other_dims(len(mat.shape), iK)
        inner_most_others = other_dims[-1]
        for i in other_dims:
            # consider M = negative index
            size_otherthan_k_or_m *= mat.shape[i]
        
        global_dims.append(size_otherthan_k_or_m)
        global_strides.append(mat.stride(inner_most_others))
        box_dims.append(1)

    rank = len(global_dims)
    box_strides = [1] * rank
    global_strides = [s * elsize for s in global_strides]

    return rank, runtime.build_tma_desc(
        mat,
        global_dims,
        global_strides,
        box_dims,
        box_strides,
        128,
        0
    )

# pytorch-major cord functions
def cord_func_K_major(mat: torch.Tensor, rank: int, iN = -2):
    assert iN != -1 and iN < len(mat.shape) - 1, "iN must not be the last dim"

    def cord_func(*cords):
        assert len(mat.shape) == len(cords), f"cords length {len(cords)} does not match mat rank {len(mat.shape)}"
        
        if rank == 3:
            assert iN == -2
            return [0, cords[-2], cords[-1] // 64]
        elif rank == 4:
            other_dims = get_other_dims(len(mat.shape), iN)
            other_strides = [mat.stride(i) for i in other_dims]
            other_cords = [cords[i] for i in other_dims]

            rest = np.dot(other_strides, other_cords) // mat.stride(other_dims[-1])
            rest = int(rest)
            return [0, cords[iN], cords[-1] // 64, rest]
        else:
            raise ValueError(f"Unsupported rank {rank} for mn-major cord function")
    return cord_func

def build_tma_wgmma_k(mat: torch.Tensor, tileK: int, tileN: int, iN: int = -2):
    assert iN != -1 and iN < len(mat.shape) - 1, "iN must not be the last dim"
    N, K = mat.shape[iN], mat.shape[-1]
    elsize = mat.element_size()
    blockK = 128 // elsize

    assert tileK % blockK == 0, "tileK must be multiple of blockK"

    glob_dims = [blockK, N, K // blockK]
    glob_strides = [mat.stride(iN), blockK]
    box_dims = [blockK, tileN, tileK // blockK]

    if len(mat.shape) > 2:
        # collapse other dims
        size_otherthan_k_or_n = 1
        other_dims = get_other_dims(len(mat.shape), iN)
        inner_most_others = other_dims[-1]
        for i in other_dims:
            size_otherthan_k_or_n *= mat.shape[i]
        
        # find stride of inner most dims other than K or N
        glob_dims.append(size_otherthan_k_or_n)
        glob_strides.append(mat.stride(inner_most_others))
        box_dims.append(1)

    glob_strides = [s * elsize for s in glob_strides]
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

class Layer:
    def __init__(self, dae, name : str):
        self.dae = dae
        self.name = name
    def schedule(self):
        raise NotImplementedError("Layer.schedule() must be implemented by subclass")
    def reference(self):
        raise NotImplementedError("Layer.reference() must be implemented by subclass")
    def diff(self):
        raise NotImplementedError("Layer.diff() must be implemented by subclass")

class GQALayer(Layer):
    def __init__(self, dae, name : str, tensors, active_new_len: int, cached_seq_len: int, need_norm: bool, need_rope: bool):
        super().__init__(dae, name)
        assert len(tensors) == 5, "GQALayer with norm/rope fusion requires 5 tensors (Q, K, V, O, rope_table)"
        self.tensors = tensors
        self.active_new_len = active_new_len
        self.cached_seq_len = cached_seq_len
        self.need_norm = need_norm
        self.need_rope = need_rope
        self._build_dims()
        self.Atom = select_attention_decode_instruction(self.QKVHdim[2])
        self._build_tmas()

    def _build_dims(self):
        matQ, matK, matV, matO, _ = self.tensors
        head_dim = self.Atom.HEAD_DIM

        assert all(t.shape[0] == matQ.shape[0] for t in self.tensors[:-1]), "Batch size (dim 0) must match for all tensors"
        self.num_req = matQ.shape[0]
        assert matQ.shape[1] == matO.shape[1], "Q and O sequence length (dim 1) must match"
        q_seq_len = matQ.shape[1]
        assert matK.shape[1] == matV.shape[1], "K and V sequence length (dim 1) must match"
        kv_seq_len = matK.shape[1]
        assert matQ.shape[2] == matO.shape[2], "Q and O head dim (dim 2) must match"
        num_q_head = matQ.shape[2] // head_dim
        assert matK.shape[2] == matV.shape[2], "K and V head dim (dim 2) must match"
        num_kv_head = matK.shape[2] // head_dim

        self.QKVHdim = (num_q_head, num_kv_head, head_dim)
        self.QKVSeqlen = (q_seq_len, kv_seq_len)
        # TODO(zhiyuang): hard code for now
        self.QKVTile = (16, 64)

    def _build_tmas(self):
        matQ, matK, matV, matO, mat_rope_table = self.tensors
        Q_SEQ_LEN, KV_SEQ_LEN = self.QKVSeqlen
        NUM_Q_HEAD, NUM_KV_HEAD, HEAD_DIM = self.QKVHdim
        HEAD_GROUP_SIZE = NUM_Q_HEAD // NUM_KV_HEAD
        QTile, KVTile = self.QKVTile

        matQ_attn_view = matQ.view(self.num_req, Q_SEQ_LEN * HEAD_GROUP_SIZE, NUM_KV_HEAD, HEAD_DIM)
        matK_attn_view = matK.view(self.num_req, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)
        matV_attn_view = matV.view(self.num_req, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)
        matO_attn_view = matO.view(self.num_req, Q_SEQ_LEN * HEAD_GROUP_SIZE, NUM_KV_HEAD, HEAD_DIM)

        tma_builder_K = partial(build_tma_wgmma_k, iN = -3)
        cord_func_K = partial(cord_func_K_major, iN=-3)

        tma_builder_MN = partial(build_tma_wgmma_mn, iK = -3)
        cord_func_MN = partial(cord_func_MN_major, iK=-3)

        tQ = TmaTensor(self.dae, matQ_attn_view)._build("load", HEAD_DIM, HEAD_GROUP_SIZE * QTile, tma_builder_K, cord_func_K)
        tK = TmaTensor(self.dae, matK_attn_view)._build("load", HEAD_DIM, KVTile, tma_builder_K, cord_func_K)
        tV = TmaTensor(self.dae, matV_attn_view)._build("load", HEAD_DIM, KVTile, tma_builder_MN, cord_func_MN)
        tO = TmaTensor(self.dae, matO_attn_view)._build("store", HEAD_DIM, HEAD_GROUP_SIZE * QTile, tma_builder_K, cord_func_K)

        self.tmas = (tQ, tK, tV, tO)
        self.rope_table_raw = RawAddress(mat_rope_table, 24) 

    def schedule(self):
        return SchedAttention(
            reqs = self.num_req,
            active_new_len=self.active_new_len,
            cached_seq_len=self.cached_seq_len,
            QKVHdim = self.QKVHdim,
            QKVTile = self.QKVTile,
            QKVSeqlen = self.QKVSeqlen,
            tmas = self.tmas,
            need_norm=self.need_norm,
            need_rope=self.need_rope,
            rope_table=self.rope_table_raw
        )

    def reference(self):
        matQ, matK, matV, matO, mat_rope_table = self.tensors
        NUM_REQ = self.num_req
        Q_SEQ_LEN, KV_SEQ_LEN = self.QKVSeqlen
        NUM_Q_HEAD, NUM_KV_HEAD, HEAD_DIM = self.QKVHdim
        HEAD_GROUP_SIZE = NUM_Q_HEAD // NUM_KV_HEAD
        fQ = matQ.view(NUM_REQ * Q_SEQ_LEN * NUM_Q_HEAD, HEAD_DIM)
        fK = matK.view(NUM_REQ * KV_SEQ_LEN * NUM_KV_HEAD, HEAD_DIM)
        active_new_len = self.active_new_len
        if self.need_norm:
            # only normalize the first 8 tokens, the rest are padding tokens
            normQ = fQ[:active_new_len * NUM_Q_HEAD].pow(2).mean(dim=-1, keepdim=True)
            normQ = fQ[:active_new_len * NUM_Q_HEAD] * torch.rsqrt(normQ + 1.0)

            normK = fK[:active_new_len * NUM_KV_HEAD].pow(2).mean(dim=-1, keepdim=True)
            normK = fK[:active_new_len * NUM_KV_HEAD] * torch.rsqrt(normK + 1.0)
        else:
            normQ = fQ[:active_new_len * NUM_Q_HEAD]
            normK = fK[:active_new_len * NUM_KV_HEAD]

        if self.need_rope:
            # rope_table: [1024, HEAD_DIM], interleaved as [cos_0, sin_0, cos_1, sin_1, ...]
            # rotation: (x[2i], x[2i+1]) -> (x*cos - y*sin, x*sin + y*cos)

            # Q: [NUM_REQ * Q_SEQ_LEN * NUM_Q_HEAD, HEAD_DIM]
            normQ_4d = normQ.view(NUM_REQ, active_new_len, NUM_Q_HEAD, HEAD_DIM).float()
            q_even = normQ_4d[..., 0::2]                                        # [B, Q, H, D/2]
            q_odd  = normQ_4d[..., 1::2]
            cos_q = mat_rope_table[:active_new_len, 0::2].float()[None, :, None, :]     # [1, Q, 1, D/2]
            sin_q = mat_rope_table[:active_new_len, 1::2].float()[None, :, None, :]
            normQ = torch.stack([q_even * cos_q - q_odd * sin_q,
                                q_even * sin_q + q_odd * cos_q], dim=-1) \
                        .flatten(-2).to(matQ.dtype) \
                        .view(NUM_REQ * active_new_len * NUM_Q_HEAD, HEAD_DIM)

            # K: [NUM_REQ * KV_SEQ_LEN * NUM_KV_HEAD, HEAD_DIM]
            normK_4d = normK.view(NUM_REQ, active_new_len, NUM_KV_HEAD, HEAD_DIM).float()
            k_even = normK_4d[..., 0::2]                                        # [B, KV, H, D/2]
            k_odd  = normK_4d[..., 1::2]
            cos_k = mat_rope_table[:active_new_len, 0::2].float()[None, :, None, :]    # [1, KV, 1, D/2]
            sin_k = mat_rope_table[:active_new_len, 1::2].float()[None, :, None, :]
            normK = torch.stack([k_even * cos_k - k_odd * sin_k,
                                k_even * sin_k + k_odd * cos_k], dim=-1) \
                        .flatten(-2).to(matK.dtype) \
                        .view(NUM_REQ * active_new_len * NUM_KV_HEAD, HEAD_DIM)
        fQ[:active_new_len * NUM_Q_HEAD] = normQ
        fK[:active_new_len * NUM_KV_HEAD] = normK
        # ---- 2. Slice Q, K, V -------------------------------------------------

        Q = matQ.view(NUM_REQ, Q_SEQ_LEN, HEAD_GROUP_SIZE, NUM_KV_HEAD, HEAD_DIM)  # (B, T, G, KV, D)
        K = matK.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)  # (B, T, KV, D)
        V = matV.view(NUM_REQ, KV_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)   # (B, T, KV, D)

        # ---- 3. Reorder for attention ----------------------------------------

        Q = Q.permute(0, 3, 2, 1, 4)            # (B, KV, GROUP, T, D)
        K = K.permute(0, 2, 1, 3).unsqueeze(2) # (B, KV, 1, T, D)
        V = V.permute(0, 2, 1, 3).unsqueeze(2)

        # ---- 4. Scaled dot-product attention ---------------------------------

        S = torch.matmul(Q.float(), K.float().transpose(-1, -2)) # [B, KV, GROUP, Qlen, KVlen] f32 as accumulate type
        P = torch.softmax(S, dim=-1)
        O = torch.matmul(P, V.float()) # [B, KV, GROUP, Qlen, D]

        # ---- 5. Restore to (B, T, Q_HEAD * HEAD_DIM) ----------------------------------

        O = O.permute(0, 3, 2, 1, 4) \
            .reshape(NUM_REQ, Q_SEQ_LEN, NUM_Q_HEAD * HEAD_DIM)
        return O
    
    def diff(self):
        _, _, _, matO, _ = self.tensors
        ref = self.reference()
        return tensor_diff(f"GQALayer: {self.name}", ref, matO.view(*ref.shape))
        ki = 7
        # tensor_diff("subset", ref[0,0,ki], matO[0,0,128*ki:128*(ki+1)])
        tensor_diff("subset", ref[0,15], matO[0,15])
        # print(ref[0, 11,0])
        # print(matO[0,11,:128])


class GemvLayerBase(Layer):
    def __init__(self, dae, Atom, name : str, MNK, tmas, reduce=True):
        super().__init__(dae, name)
        self.dae = dae
        self.Atom = Atom
        self.reduce = reduce
        self.MNK = MNK
        self.tmas = tmas

    @staticmethod
    def _tma_funcs(self, Atom, reduce):
        return [
            lambda t: t.wgmma_load(Atom.MNK[0], Atom.MNK[2], Major.K),
            lambda t: t.wgmma_load(Atom.MNK[1], Atom.MNK[2] * Atom.n_batch, Major.K),
            lambda t: t.wgmma("reduce" if reduce else "store", Atom.MNK[1], Atom.MNK[0], Major.MN)
        ]

    def description(self):
        M, N, K = self.MNK
        loadA, loadB, reduceC = self.tmas

        print(f"GEMV M64N16 on [M={M} x N={N} x K={K}], SMs={num_sms}:")
        print(f"loadA size: {loadA.size // 1024} KB, loadB size: {loadB.size // 1024} KB, storeC size: {reduceC.size // 1024} KB")
        print("theory load speed:", (matA.nbytes + matB.nbytes + matC.nbytes) / 1024 ** 3 / 3700 * 1e6, "us")
        print("theory load speed (no L2):", (matA.nbytes + matB.nbytes * num_sms + matC.nbytes) / 1024 ** 3 / 3700 * 1e6, "us")


    def schedule(self, num_gemv_sms=None, base_sm=0, **kwargs):
        sched = SchedGemv(self.Atom, self.MNK, self.tmas, **kwargs)
        if num_gemv_sms is not None:
            sched = sched.place(num_gemv_sms, base_sm=base_sm)
        return sched
    def schedule_(self, num_gemv_sms=None, base_sm=0, **kwargs):
        return self.schedule(num_gemv_sms=num_gemv_sms, base_sm=base_sm, exec=False, **kwargs)
    
    def reference(self):
        matA, matB, _ = self.tensors
        return matA @ matB.t()

    def diff(self):
        _, _, matC = self.tensors
        ref = self.reference()
        return tensor_diff(f"GEMV Layer: {self.name}", ref, matC.t())

class GemvLayer(GemvLayerBase):
    def __init__(self, dae, Atom, name : str, tensors, reduce=True):
        assert len(tensors) == 3, "GemvLayer requires 3 TmaTensors (A, B, C)"
        self.dae = dae
        self.name = name

        self.reduce = reduce
        self.Atom = Atom
        self.tensors = tensors
        self._build_MNK()
        self._build_tmas()

    def _build_MNK(self):
        matA, matB, matC = self.tensors
        assert matA.dim() == 2, "matA must be 2D"
        assert matB.dim() == 2, "matB must be 2D"
        assert matC.dim() == 2, "matC must be 2D"
        assert matA.shape[1] == matB.shape[1], "matA and matB K dimensions must match"
        assert matA.shape[0] == matC.shape[1], "mata and matC M dimensions must match"
        assert matB.shape[0] == self.Atom.MNK[1], "matB N dimension must match Atom.N"
        assert matC.shape[0] == self.Atom.MNK[1], "matC N dimension must match Atom.N"

        self.MNK = (matA.shape[0], matB.shape[0], matA.shape[1])

    # TODO(zhiyuang): don't build TMA tensors in this func, try to do on-demand
    #                 but cached in case of multiple schedules
    def _build_tmas(self):
        M, N, K = self.MNK
        TileM, TileN, TileK = self.Atom.MNK

        self.tmas = tuple(
            func(TmaTensor(self.dae, t))
            for func, t
            in zip(GemvLayerBase._tma_funcs(self, self.Atom, self.reduce), self.tensors)
        )

class RMSLayer(Layer):
    def __init__(self, dae, name: str, 
                 input: torch.Tensor, output: torch.Tensor, epsilon: float,
                 num_token=None):
        super().__init__(dae, name)
        # input/output shape: [B_padded, H]
        self.input = input
        self.output = output
        self.epsilon = epsilon
        self.num_token = num_token if num_token is not None else input.shape[0]
    
    def schedule(self, num_sms, base_sm=0, **kwargs):
        return SchedRMS(
            num_token=self.num_token,
            epsilon=self.epsilon,
            input_glob=self.input,
            output_glob=self.output,
            use_glob=False,
            **kwargs
        ).place(num_sms, base_sm=base_sm)

    def reference(self):
        var = self.input.pow(2).mean(dim=-1, keepdim=True)
        X = self.input * torch.rsqrt(var + self.epsilon)
        return X[:self.num_token]
    
    def diff(self):
        ref = self.reference()
        return tensor_diff(f"RMS Layer: {self.name}", ref, self.output[:self.num_token])


def layers_like(constr, *cargs):
    def wrapper(*args, **kwargs):
        return constr(*cargs, *args, **kwargs)
    return wrapper
