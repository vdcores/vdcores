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
num_sms = 128

matHidden = torch.rand(N, HIDDEN, dtype=dtype, device=gpu) - 0.5
matUp = torch.rand(INTERMIDIATE, HIDDEN, dtype=dtype, device=gpu) - 0.5
matGate = torch.rand(INTERMIDIATE, HIDDEN, dtype=dtype, device=gpu) - 0.5
matInterm = torch.zeros(N, INTERMIDIATE, dtype=dtype, device=gpu)
matGateOut = torch.zeros(N, INTERMIDIATE, dtype=dtype, device=gpu)
matDown = torch.rand(HIDDEN, INTERMIDIATE, dtype=dtype, device=gpu) - 0.5
matOut = torch.zeros(N, HIDDEN, dtype=dtype, device=gpu)

dae = Launcher(num_sms, device=gpu)

loadHidden = TmaTensor(dae, matHidden).wgmma_load(N, TileK * Gemv_M64N16.n_batch, Major.K)
loadUp = TmaTensor(dae, matUp).wgmma_load(TileM, TileK, Major.K)
loadGate = TmaTensor(dae, matGate).wgmma_load(TileM, TileK, Major.K)
reduceInterm = TmaTensor(dae, matInterm).wgmma("reduce", N, TileM, Major.MN)
reduceGateOut = TmaTensor(dae, matGateOut).wgmma("reduce", N, TileM, Major.MN)

loadInterm = TmaTensor(dae, matInterm).wgmma_load(N, TileK * Gemv_M64N16.n_batch, Major.K)
loadDown = TmaTensor(dae, matDown).wgmma_load(TileM, TileK, Major.K)
reduceOut = TmaTensor(dae, matOut).wgmma("reduce", N, TileM, Major.MN)

dae.s(
    interleave(
        SchedGemv(Gemv_M64N16, num_sms,
            MNK=(INTERMIDIATE, N, HIDDEN),
            tmas=(loadGate, loadHidden, reduceGateOut),
            exec=False).split_M(3),
        SchedGemv(Gemv_M64N16, num_sms,
            MNK=(INTERMIDIATE, N, HIDDEN),
            tmas=(loadUp, loadHidden, reduceInterm),
            exec=False).split_M(3)
    ),
    SchedGemv(Gemv_M64N16, 128,
              MNK=(HIDDEN, N, INTERMIDIATE),
              tmas=(loadDown, loadInterm, reduceOut))
              .split_K(3),
)

print(f"MLP GEMV on [H={HIDDEN}, INTERM={INTERMIDIATE}, N={N}], SMs={num_sms}:")
print("theory load speed:", ((matHidden.nbytes + matGateOut.nbytes) / N + matUp.nbytes + matGate.nbytes + matDown.nbytes) / 1024 ** 3 / 3700 * 1e6, "us")
print("theory load speed (no L2):", (matHidden.nbytes * num_sms + matUp.nbytes + matGate.nbytes + matInterm.nbytes * num_sms + matDown.nbytes) / 1024 ** 3 / 3700 * 1e6, "us")
print()

def gemv_ref():
    gate = matHidden @ matGate.T
    interm = matHidden @ matUp.T
    out = interm @ matDown.T
    return interm, gate, out

interm, gate, out = gemv_ref()

dae_app(dae)

tensor_diff("interm", interm, matInterm)
tensor_diff("gate", gate, matGateOut)
tensor_diff("out", out, matOut)
