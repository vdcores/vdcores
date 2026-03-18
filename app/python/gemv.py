import torch
from dae.launcher import *
from dae.util import *

gpu = torch.device("cuda")

Atom = Gemv_M64N8

TileM, N, TileK = Atom.MNK
M, K = 4096, 4096
nstage = 2

assert K % TileK == 0
assert M % TileM == 0
num_sms = 128
assert num_sms <= 132 # max sm count for HX00

matA = torch.rand(M, K, dtype=torch.bfloat16, device=gpu) - 0.5
matB = torch.rand(N, K, dtype=torch.bfloat16, device=gpu) - 0.5
matC = torch.zeros(N, M, dtype=torch.bfloat16, device=gpu)

dae = Launcher(num_sms, device=gpu)

loadA = TmaTensor(dae, matA).wgmma_load(TileM, TileK, Major.K)
loadB = TmaTensor(dae, matB).wgmma_load(N, TileK * 4, Major.K)
storeC = TmaTensor(dae, matC).wgmma_store(N, TileM, Major.MN)
reduceC = TmaTensor(dae, matC).wgmma("reduce", N, TileM, Major.MN)

n_batch = 4
def sm_task(sm: int):
    m_total = M // num_sms
    assert m_total % TileM == 0
    m_start = sm * m_total
    insts = []
    for m in range(m_start, m_start + m_total, TileM):
        insts += [
            Atom(K // TileK),
            # [
            #     [
            #         loadB.cord(0, k),
            #         [ loadA.cord(m, k + TileK * i) for i in range(n_batch) ]
            #     ]
            #     for k in range(0, K, TileK * n_batch)
            # ],
            RepeatM.on(K // TileK // n_batch,
                [loadB.cord(0, 0), loadB.cord2tma(0, n_batch * TileK)],
                *[
                    [loadA.cord(m, TileK * i), loadA.cord2tma(0, n_batch * TileK)]
                    for i in range(n_batch)
                ]
            ),
            storeC.cord(0, m).bar(),
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
            Atom(k_total // TileK),

            # [
            #     [
            #         loadB.cord(0, k),
            #         [ loadA.cord(m, k + TileK * i) for i in range(n_batch) ]
            #     ]
            #     for k in range(k_offset, k_offset + k_total, TileK * n_batch)
            # ],

            RepeatM.on(k_total // TileK // n_batch,
                [loadB.cord(0, k_offset), loadB.cord2tma(0, n_batch * TileK)],
                *[
                    [loadA.cord(m, k_offset + TileK * i), loadA.cord2tma(0, n_batch * TileK)]
                    for i in range(n_batch)
                ]
            ),
            reduceC.cord(0, m).bar(),
        ]
    return sm_reduce

    
dae.i(
    mk_sm_reduce(nstage),
    # sm_task,

    TerminateC(),
    TerminateM(),
)


ref = matA @ matB.t()
res = matC.t()

dae_app(dae)

print(f"GEMV M64N16 on [M={M} x N={N} x K={K}], stage={nstage}:")
print(f"loadA size: {loadA.size // 1024} KB, loadB size: {loadB.size // 1024} KB, storeC size: {storeC.size // 1024} KB")
tensor_diff("GEMV M64N16", ref, res)

# sm_bytes = (loadA.size * 4 + loadB.size) * K // TileK // 4 + storeC.size

# dae_app(dae)

# print("inst size:", dae.num_insts())
# print("theory load speed:", (matA.nbytes + matB.nbytes + matC.nbytes) / 1024 ** 3 / 3700 * 1e6, "us")
# print("theory load speed (no L2):", (matA.nbytes + matB.nbytes * num_sms + matC.nbytes) / 1024 ** 3 / 3700 * 1e6, "us")
# print("stall instructions", dae.profile[:,3].cpu().numpy().mean())
