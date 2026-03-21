import torch
from dae.launcher import *
from dae.util import *
from dae.model import *

torch.manual_seed(0)
gpu = torch.device("cuda")

num_sms = 128
N, HIDDEN = 8, 4096
HEAD_DIM = 128
TileM, TileN, TileK = Gemv_M64N8.MNK

matM = torch.rand(HIDDEN, HIDDEN, dtype=torch.bfloat16, device=gpu) - 0.5
matV = torch.rand(N, HIDDEN, dtype=torch.bfloat16, device=gpu) - 0.5
matO = torch.zeros(N, HIDDEN, dtype=torch.bfloat16, device=gpu)
# M: [Hq * D, 4096]
max_seq_len = 1024

# format: table[pos] = [cos0, sin0, cos1, sin1, ...], interleaved for better memory access pattern
def build_rope_table(max_seq_len, head_dim, base=10000, disable=False):
    if disable:
        # generate rope table with all cos = 1 and sin = 0 to effectively disable rope
        rope = torch.zeros(max_seq_len, head_dim, dtype=torch.float32, device=gpu)
        rope[:, 0::2] = 1.0
        return rope.to(torch.bfloat16)

    assert head_dim % 2 == 0

    half = head_dim // 2

    # frequency for each dim pair
    inv_freq = base ** (-torch.arange(0, half, dtype=torch.float32) * 2 / head_dim)

    # positions
    pos = torch.arange(max_seq_len, dtype=torch.float32)

    # angles
    angles = torch.outer(pos, inv_freq)  # [seq, half]

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    rope = torch.empty(max_seq_len, head_dim, dtype=torch.float32, device=gpu)

    rope[:, 0::2] = cos
    rope[:, 1::2] = sin
    rope = rope.to(torch.bfloat16)
    return rope

rope_table = build_rope_table(max_seq_len, HEAD_DIM, disable=True)
print("Rope table shape:", rope_table.shape)

assert HEAD_DIM % TileM == 0, "HEAD_DIM must be divisible by TileM for interleaved rope table"
assert TileM >= 64, "TileM must be at least 64"
def tma_load_tbl(mat: torch.Tensor, TileM: int, TileN: int):
    assert mat.element_size() == 2, "Only support float16/bfloat16 output"

    # TODO(zijian): how general multi-request batch should work with smem rope table
    # currently assume tokens are from different requests but all at same seq position
    s = mat.element_size()
    # repeat for tileN times
    glob_dims = [64, TileN, HEAD_DIM // 64, max_seq_len]
    glob_strides = [0, 64 * s, HEAD_DIM * s]
    box_dims = [64, TileN, TileM // 64, 1]
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

def cord_load_tbl(mat: torch.Tensor, rank: int):
    assert rank == 4, "Only support 4D tma load for rope table"
    def cfunc(*cords):
        assert len(cords) == 2, f"cords length {len(cords)} should be 2 for rope table"
        return [0, 0, cords[1], cords[0]]
    return cfunc

dae = Launcher(num_sms, device=gpu)

loadM = TmaTensor(dae, matM).wgmma_load(TileM, TileK, Major.K)
loadV = TmaTensor(dae, matV).wgmma_load(TileN, TileK * Gemv_M64N8.n_batch, Major.K)
# index into the entire table to avoid having different TMA desc across iterations ?
loadRope = TmaTensor(dae, rope_table)._build("load", HEAD_DIM, TileN, tma_load_tbl, cord_load_tbl)

regStoreO = RegStore(0, size = TileM * TileN * matO.element_size())
regLoadO = RegLoad(0)

reduceO = TmaTensor(dae, matO).wgmma('reduce', TileN, TileM, Major.MN)

gemv = SchedGemv(Gemv_M64N8,
    MNK=(HIDDEN, N, HIDDEN),
    tmas=(loadM, loadV, regStoreO)).place(num_sms)

def cord_reduce(sm: int, inst):
    m = sm % 64 * TileM
    return inst.cord(0, m)

cached_seq_len = 7
def cord_table(sm: int, inst):
    m = sm % 64 * TileM
    return inst.cord(cached_seq_len, (m % 128) // 64)

rope = SchedRope( ROPE_INTERLEAVE_512,
   tmas=(loadRope, regLoadO, reduceO),
   cords=(cord_table, None, cord_reduce)
).place(num_sms)

dae.s(
    gemv,
    rope,
)


dae_app(dae)

refO = matV @ matM.T
refO_Q = refO.view(N, 32, 128)
even = refO_Q[..., ::2]
odd = refO_Q[..., 1::2]
cos = rope_table[cached_seq_len, ::2].float()[None,None,:]
sin = rope_table[cached_seq_len, 1::2].float()[None,None,:]
out_even = even * cos - odd * sin
out_odd = even * sin + odd * cos
refO_rope = torch.empty_like(refO)
refO_rope.view(N, 32, 128)[..., ::2] = out_even
refO_rope.view(N, 32, 128)[..., 1::2] = out_odd

# apply rope 
tensor_diff("gemv", refO, matO)
