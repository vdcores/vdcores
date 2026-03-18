import torch
from dae.launcher import *
from dae.util import *
from dae.schedule import *
import torch.nn.functional as F

# this test simulates a multi-stage data-dependent flow
# try to figure out if prefetch is useful in this case

torch.manual_seed(0)
dtype = torch.bfloat16
gpu = torch.device("cuda")

TileM, N, TileK = Gemv_M64N8.MNK

HIDDEN, INTERMIDIATE = 4096, 4096 * 3
num_sms = 132

matHidden = torch.rand(N, HIDDEN, dtype=dtype, device=gpu) - 0.5
matUp = torch.rand(INTERMIDIATE, HIDDEN, dtype=dtype, device=gpu) - 0.5
matGate = torch.rand(INTERMIDIATE, HIDDEN, dtype=dtype, device=gpu) - 0.5
matInterm = torch.zeros(N, INTERMIDIATE, dtype=dtype, device=gpu)
matGateOut = torch.zeros(N, INTERMIDIATE, dtype=dtype, device=gpu)
matSiluOut = torch.zeros(N, INTERMIDIATE, dtype=dtype, device=gpu)

matDown = torch.rand(HIDDEN, INTERMIDIATE, dtype=dtype, device=gpu) - 0.5
matOut = torch.zeros(N, HIDDEN, dtype=dtype, device=gpu)

dae = Launcher(num_sms, device=gpu)

loadHidden = TmaTensor(dae, matHidden).wgmma_load(N, TileK * Gemv_M64N8.n_batch, Major.K)
loadUp = TmaTensor(dae, matUp).wgmma_load(TileM, TileK, Major.K)
loadGate = TmaTensor(dae, matGate).wgmma_load(TileM, TileK, Major.K)
storeInterm = TmaTensor(dae, matInterm).wgmma_store(N, TileM, Major.MN)
storeGateOut = TmaTensor(dae, matGateOut).wgmma_store(N, TileM, Major.MN)

storeSilu = TmaTensor(dae, matSiluOut).wgmma_store(N, TileM, Major.MN)
loadSilu = TmaTensor(dae, matSiluOut).wgmma_load(N, TileK * Gemv_M64N8.n_batch, Major.K)
loadDown = TmaTensor(dae, matDown).wgmma_load(TileM, TileK, Major.K)

reduceOut = TmaTensor(dae, matOut).wgmma("reduce", N, TileM, Major.MN)

regGate, regUp = 0, 1

regStoreGate = RegStore(regGate, matGateOut[:,0:TileM])
regStoreUp = RegStore(regUp, matInterm[:,0:TileM])

silu_in_bar = dae.new_bar(128)
silu_out_bar1 = dae.new_bar(N)
silu_out_bar2 = dae.new_bar(128)

def silu1(sm: int):
    sm -= 128
    if sm < 0:
        return []
    insts = []
    start_token_id = sm * (N // 4)
    end_token_id = (sm + 1) * (N // 4)
    for i in range(start_token_id, end_token_id):
        insts.extend([
            SILU_MUL_SHARED_BF16_K_4096_INTER(1),
            TmaStore1D(matSiluOut[i,:4096]).bar(silu_out_bar1),
            TmaLoad1D(matGateOut[i,:4096]).bar(silu_in_bar) if i == start_token_id else TmaLoad1D(matGateOut[i,:4096]),
            TmaLoad1D(matInterm[i,:4096]),
        ])
    return insts

def silu(sm : int):
    if sm >= 128:
        return []

    return [
        SILU_MUL_SHARED_BF16_K_64_SW128(N),

        storeSilu.cord(0, 4096 + sm * TileM).bar(silu_out_bar2),
        RegLoad(regGate), # Load the gate
        RegLoad(regUp), # load the up
    ]


downs = SchedGemv(Gemv_M64N8, 128,
    MNK=(HIDDEN, N, INTERMIDIATE),
    tmas=(loadDown, loadSilu, reduceOut)).split_K(3)
downs[0].load_bar(silu_out_bar1)
downs[1].load_bar(silu_out_bar2)

dae.s(
    SchedGemv(Gemv_M64N8, 64,
        MNK=(4096, N, HIDDEN),
        tmas=(loadGate, loadHidden, storeGateOut)).store_bar(silu_in_bar),
    SchedGemv(Gemv_M64N8, 64,
        MNK=(4096, N, HIDDEN),
        tmas=(loadUp, loadHidden, storeInterm),
        base_sm=64).store_bar(silu_in_bar),
    silu1,
    SchedGemv(Gemv_M64N8, 128,
        MNK=((4096,8192), N, HIDDEN),
        tmas=(loadGate, loadHidden, regStoreGate)),
    SchedGemv(Gemv_M64N8, 128,
        MNK=((4096,8192), N, HIDDEN),
        tmas=(loadUp, loadHidden, regStoreUp)),
    silu,
    downs,
    TerminateC(),
    TerminateM(),
)

print(f"MLP GEMV on [H={HIDDEN}, INTERM={INTERMIDIATE}, N={N}], SMs={num_sms}:")
print()

def gemv_ref():
    gate = matHidden @ matGate.T
    interm = matHidden @ matUp.T
    sinterm = F.silu(gate) * interm
    out = sinterm @ matDown.T
    return interm, gate, out, sinterm

interm, gate, out, sinterm = gemv_ref()

dae_app(dae)

# tensor_diff("interm", interm, matInterm)
# tensor_diff("gate", gate, matGateOut)
tensor_diff("out", out, matOut)
tensor_diff("silu1", sinterm[:,4096:], matSiluOut[:,4096:])
