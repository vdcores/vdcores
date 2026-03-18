import torch
from dae.launcher import *
from dae.util import *
from dae.model import *

torch.manual_seed(0)
gpu = torch.device("cuda")

Atom = Gemv_M64N8_ROPE_128

TileM, N, TileK = Atom.MNK
N = 8
NUM_HEAD, HEAD_DIM = 32, 128
nstage = 2
M = NUM_HEAD * HEAD_DIM
K = 4096
NUM_REQ = N

assert K % TileK == 0
assert M % TileM == 0
num_sms = 128
assert num_sms <= 132 # max sm count for HX00

matA = torch.rand(M, K, dtype=torch.bfloat16, device=gpu) - 0.5
matB = torch.rand(N, K, dtype=torch.bfloat16, device=gpu) - 0.5
# store to some index
hist_req_len = 16
matC = torch.zeros(NUM_REQ, 128, M, dtype=torch.bfloat16, device=gpu)

dae = Launcher(num_sms, device=gpu)

loadA = TmaTensor(dae, matA).wgmma_load(TileM, TileK, Major.K)
loadB = TmaTensor(dae, matB).wgmma_load(N, TileK * 4, Major.K)

tma_builder_MN = partial(build_tma_wgmma_mn, iK = -3)
cord_func_MN = partial(cord_func_MN_major, iK=-3)
storeC = TmaTensor(dae, matC)._build("store", TileM, N, tma_builder_MN, cord_func_MN)
reduceC = TmaTensor(dae, matC)._build("reduce", TileM, N, tma_builder_MN, cord_func_MN)

rope_table = torch.rand(1024, HEAD_DIM, dtype=torch.bfloat16, device=gpu) - 0.5

n_batch = 4
def sm_task(sm: int):
    m_total = M // num_sms
    assert m_total % TileM == 0
    m_start = sm * m_total
    insts = []
    for m in range(m_start, m_start + m_total, TileM):
        insts += [
            Atom(K // TileK, hist_req_len, m % 128),
            RawAddress(rope_table, 24), 
            RepeatM.on(K // TileK // n_batch,
                [loadB.cord(0, 0), loadB.cord2tma(0, n_batch * TileK)],
                *[
                    [loadA.cord(m, TileK * i), loadA.cord2tma(0, n_batch * TileK)]
                    for i in range(n_batch)
                ]
            ),
            storeC.cord(0, hist_req_len, m).bar(),
        ]
    return insts

def mk_sm_reduce(stage: int):
    def sm_reduce(sm: int):
        sm_offset = sm // stage
        sm_stage = sm % stage

        k_total = K // stage
        k_offset = sm_stage * k_total

        m = sm_offset * TileM
        return [
            Atom(k_total // TileK, hist_req_len, m % 128),
            RawAddress(rope_table, 24), 

            RepeatM.on(k_total // TileK // n_batch,
                [loadB.cord(0, k_offset), loadB.cord2tma(0, n_batch * TileK)],
                *[
                    [loadA.cord(m, k_offset + TileK * i), loadA.cord2tma(0, n_batch * TileK)]
                    for i in range(n_batch)
                ]
            ),
            reduceC.cord(0, hist_req_len, m).bar(),
        ]
    return sm_reduce
    
dae.i(
    mk_sm_reduce(nstage),
    # sm_task,

    TerminateC(),
    TerminateM(),
)


O = matA @ matB.t()
# apply rope
O = O.view(NUM_HEAD, HEAD_DIM, N).permute(2, 0, 1)
even = O[..., ::2]
odd = O[..., 1::2]
cos = rope_table[hist_req_len, 0::2].float()[None, None, :]
sin = rope_table[hist_req_len, 1::2].float()[None, None, :]
O = torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1).to(matC.dtype).view(N, M)
res = matC[:,hist_req_len,:]

dae_app(dae)
tensor_diff("GEMV M64N16", O, res)
