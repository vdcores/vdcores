import torch
from dae.launcher import *
from dae.util import *

# this test simulates a multi-stage data-dependent flow
# try to figure out if prefetch is useful in this case

gpu = torch.device("cuda")

TileM, N, TileK = Gemv_M64N16.MNK

HIDDEN, INTERMIDIATE = 4096, 4096 * 3
n_batch = 4

num_sms = 128

matHidden = torch.rand(N, HIDDEN, dtype=torch.float16, device=gpu) - 0.5
matUp = torch.rand(INTERMIDIATE, HIDDEN, dtype=torch.float16, device=gpu) - 0.5
matGate = torch.rand(INTERMIDIATE, HIDDEN, dtype=torch.float16, device=gpu) - 0.5
matInterm = torch.zeros(N, INTERMIDIATE, dtype=torch.float16, device=gpu)
matGateOut = torch.zeros(N, INTERMIDIATE, dtype=torch.float16, device=gpu)
matDown = torch.rand(HIDDEN, INTERMIDIATE, dtype=torch.float16, device=gpu) - 0.5
matOut = torch.zeros(N, HIDDEN, dtype=torch.float16, device=gpu)

# blockDown = torch.rand(128, INTERMIDIATE // 2 // TileK // 2, TileK * TileM, dtype=torch.float16, device=gpu)
# blockGate = torch.rand(128, INTERMIDIATE // TileK, TileK * TileM, dtype=torch.float16, device=gpu)
# blockUp = torch.rand(128, INTERMIDIATE // TileK , TileK * TileM, dtype=torch.float16, device=gpu)
# print(blockDown.shape, blockGate.shape, blockUp.shape)

dae = Launcher(num_sms, device=gpu)

loadHidden = TmaTensor(dae, matHidden).wgmma_load(N, TileK * n_batch, Major.K)
loadUp = TmaTensor(dae, matUp).wgmma_load(TileM, TileK, Major.K)
loadGate = TmaTensor(dae, matGate).wgmma_load(TileM, TileK, Major.K)
reduceInterm = TmaTensor(dae, matInterm).wgmma("reduce", N, TileM, Major.MN)
reduceGateOut = TmaTensor(dae, matGateOut).wgmma("reduce", N, TileM, Major.MN)

loadInterm = TmaTensor(dae, matInterm).wgmma_load(N, TileK * n_batch, Major.K)
loadDown = TmaTensor(dae, matDown).wgmma_load(TileM, TileK, Major.K)
reduceOut = TmaTensor(dae, matOut).wgmma("reduce", N, TileM, Major.MN)

intermPerWave = INTERMIDIATE // 2
def up(sm: int):
    insts = []
    for m_start in range(0, INTERMIDIATE, intermPerWave):
        m = m_start + (sm % 64) * TileM
        # reduce on K schedule
        k_start = HIDDEN // 2 if sm >= 64 else 0
        insts += [
            # scheudle gate gemv
            Gemv_M64N16(intermPerWave // TileK // 2),
            RepeatM.on(intermPerWave // TileK // 2 // n_batch,
                [loadHidden.cord(0, k_start), loadHidden.cord2tma(0, n_batch * TileK)],
                *[
                    (loadGate.cord(m, k_start + i * TileK), loadGate.cord2tma(0, n_batch * TileK))
                    for i in range(n_batch)
                ]
            ),
            reduceGateOut.cord(0, m),

            # schedule interm gemv
            Gemv_M64N16(intermPerWave // TileK // 2),
            RepeatM.on(intermPerWave // TileK // 2 // n_batch,
                [loadHidden.cord(0, k_start), loadHidden.cord2tma(0, n_batch * TileK)],
                *[
                    (loadUp.cord(m, k_start + i * TileK), loadUp.cord2tma(0, n_batch * TileK))
                    for i in range(n_batch)
                ]
            ),
            reduceInterm.cord(0, m),
        ]
    return insts

nstage_down = 2
def down(sm: int):
    stage = sm // 64

    K = INTERMIDIATE // 2
    k_start = stage * K
    
    m = (sm % 64) * TileM

    return [
        Gemv_M64N16(K // TileK),
        
        # rest of the body
        RepeatM.on(K // TileK // n_batch,
            [loadInterm.cord(0, k_start + n_batch * TileK), loadInterm.cord2tma(0, n_batch * TileK)],
            *[
                # (TmaLoad1D(blockDown[sm, i, ...]), n_batch * TileM * TileK * 2)
                (loadDown.cord(m, k_start + i * TileK), loadDown.cord2tma(0, n_batch * TileK))
                for i in range(n_batch)
            ]
        ),
        reduceOut.cord(0, m).bar(),
    ]
    
dae.i(
    up,
    down,

    TerminateC(),
    TerminateM(),
)

print(f"MLP GEMV on [H={HIDDEN}, INTERM={INTERMIDIATE}, N={N}], SMs={num_sms}:")
print("theory load speed:", (matHidden.nbytes / N + matUp.nbytes + matGate.nbytes + matInterm.nbytes / N + matDown.nbytes) / 1024 ** 3 / 3700 * 1e6, "us")
print("theory load speed (no L2):", (matHidden.nbytes * num_sms + matUp.nbytes + matGate.nbytes + matInterm.nbytes * num_sms + matDown.nbytes) / 1024 ** 3 / 3700 * 1e6, "us")
print()

def gemv_ref():
    interm = matHidden @ matUp.T
    out = matOut
    return interm, out

interm, out = gemv_ref()

dae_app(dae)

# tensor_diff("interm", interm, matInterm)
# tensor_diff("out", out, matOut)
