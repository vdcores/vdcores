import torch
from dae.launcher import *
from dae.util import *
from dae.schedule import *

# this test simulates a multi-stage data-dependent flow
# try to figure out if prefetch is useful in this case

dtype = torch.bfloat16
gpu = torch.device("cuda")

TileM, N, TileK = Gemv_M64N16.MNK

HIDDEN, INTERMIDIATE = 4096, 4096 * 3

matHidden = torch.rand(N, HIDDEN, dtype=dtype, device=gpu) - 0.5
matUp = torch.rand(INTERMIDIATE, HIDDEN, dtype=dtype, device=gpu) - 0.5
matGate = torch.rand(INTERMIDIATE, HIDDEN, dtype=dtype, device=gpu) - 0.5
matInterm = torch.zeros(N, INTERMIDIATE, dtype=dtype, device=gpu)
matGateOut = torch.zeros(N, INTERMIDIATE, dtype=dtype, device=gpu)
matDown = torch.rand(HIDDEN, INTERMIDIATE, dtype=dtype, device=gpu) - 0.5
matOut = torch.zeros(N, HIDDEN, dtype=dtype, device=gpu)
matSiLUOut = torch.zeros(N, INTERMIDIATE, dtype=dtype, device=gpu)

dae = Launcher(132, device=gpu)

loadHidden = TmaTensor(dae, matHidden).wgmma_load(N, TileK * Gemv_M64N16.n_batch, Major.K)
loadUp = TmaTensor(dae, matUp).wgmma_load(TileM, TileK, Major.K)
loadGate = TmaTensor(dae, matGate).wgmma_load(TileM, TileK, Major.K)
reduceInterm = TmaTensor(dae, matInterm).wgmma("reduce", N, TileM, Major.MN)
reduceGateOut = TmaTensor(dae, matGateOut).wgmma("reduce", N, TileM, Major.MN)

loadInterm = TmaTensor(dae, matSiLUOut).wgmma_load(N, TileK * Gemv_M64N16.n_batch, Major.K)
loadDown = TmaTensor(dae, matDown).wgmma_load(TileM, TileK, Major.K)
reduceOut = TmaTensor(dae, matOut).wgmma("reduce", N, TileM, Major.MN)

num_gemv_sms = 128
num_silu_sms = 1

up_bars = [dae.new_bar(num_gemv_sms) for _ in range(3)]
down_bars = [dae.new_bar(num_silu_sms) for _ in range(3)]
# _down_bar = dae.new_bar(0)

sched_silus = []
chunk_size = INTERMIDIATE // 3
for i in range(3):
    # assume 1 token
    tmaGate = TmaLoad1D(matGateOut[0,i * chunk_size:(i + 1) * chunk_size]).bar(up_bars[i])
    tmaUp = TmaLoad1D(matInterm[0,i * chunk_size:(i + 1) * chunk_size])
    tmaOut = TmaStore1D(matSiLUOut[0,i * chunk_size:(i + 1) * chunk_size]).bar(down_bars[i])

    sched_silu = SchedSmemSiLU_K_4096_N_1(
        tmaGate,
        tmaUp,
        tmaOut,
        i + 128,
    )
    sched_silus.append(sched_silu)

dae.s(
    interleave(
        SchedGemv(Gemv_M64N16,
            MNK=(INTERMIDIATE, N, HIDDEN),
            tmas=(loadGate, loadHidden, reduceGateOut),
            exec=False).split_M(3).place(num_gemv_sms),
        [s.bar("store", up_bar) for up_bar, s in zip(up_bars, SchedGemv(Gemv_M64N16,
            MNK=(INTERMIDIATE, N, HIDDEN),
            tmas=(loadUp, loadHidden, reduceInterm),
            exec=False).split_M(3).place(num_gemv_sms))]
    ),
    sched_silus,
    [s.bar("load", down_bar) for down_bar, s in zip(down_bars, SchedGemv(Gemv_M64N16,
              MNK=(HIDDEN, N, INTERMIDIATE),
              tmas=(loadDown, loadInterm, reduceOut),
              prefetch=False)
              .split_K(3).place(num_gemv_sms))],
    # SchedGemv(Gemv_M64N16, num_gemv_sms,
    #           MNK=(HIDDEN, N, INTERMIDIATE),
    #           tmas=(loadDown, loadInterm, reduceOut))
    #           .split_K(3),
)

print(f"MLP GEMV on [H={HIDDEN}, INTERM={INTERMIDIATE}, N={N}], SMs={132}:")
print("theory load speed:", ((matHidden.nbytes + matGateOut.nbytes) / N + matUp.nbytes + matGate.nbytes + matDown.nbytes) / 1024 ** 3 / 3700 * 1e6, "us")
print("theory load speed (no L2):", (matHidden.nbytes * num_gemv_sms + matUp.nbytes + matGate.nbytes + matInterm.nbytes * num_gemv_sms + matDown.nbytes) / 1024 ** 3 / 3700 * 1e6, "us")
print()

# def gemv_ref():
#     gate = matHidden @ matGate.T
#     interm = matHidden @ matUp.T
#     out = interm @ matDown.T
#     return interm, gate, out

# interm, gate, out = gemv_ref()

dae_app(dae)

# tensor_diff("interm", interm, matInterm)
# tensor_diff("gate", gate, matGateOut)
# tensor_diff("out", out, matOut)
