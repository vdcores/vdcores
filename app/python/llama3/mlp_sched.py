import argparse
import sys

import torch
import torch.nn.functional as F

from dae.launcher import *
from dae.schedule import *
from dae.util import dae_app, tensor_diff


arg_parser = argparse.ArgumentParser(add_help=False)
arg_parser.add_argument("--correctness", action="store_true")
arg_parser.add_argument("--seed", type=int, default=0)
parsed_args, remaining_argv = arg_parser.parse_known_args()
if parsed_args.correctness and not any(arg in ("-l", "--launch", "-b", "--bench") for arg in remaining_argv):
    remaining_argv = [*remaining_argv, "--launch"]
sys.argv = [sys.argv[0], *remaining_argv]

torch.manual_seed(parsed_args.seed)

gpu = torch.device("cuda")
dtype = torch.bfloat16

N = 8
HIDDEN = 4096
INTERMEDIATE = 14336
MLP_A = 2048
MLP_B = 4096
MLP_C = 4096
MLP_D = 4096
MLP_AB = MLP_A + MLP_B
MLP_CD = MLP_C + MLP_D

num_sms = 128
full_sms = 132
dae = Launcher(full_sms, device=gpu)

TileM, _, TileK = Gemv_M64N8.MNK

matHidden = torch.randn(N, HIDDEN, dtype=dtype, device=gpu)
matRMSHidden = matHidden
matGate = torch.randn(INTERMEDIATE, HIDDEN, dtype=dtype, device=gpu)
matUp = torch.randn(INTERMEDIATE, HIDDEN, dtype=dtype, device=gpu)
matDown = torch.randn(HIDDEN, INTERMEDIATE, dtype=dtype, device=gpu)

matInterm = torch.zeros(N, INTERMEDIATE, dtype=dtype, device=gpu)
matGateOut = torch.zeros(N, INTERMEDIATE, dtype=dtype, device=gpu)
matSiLUOut = torch.zeros(N, INTERMEDIATE, dtype=dtype, device=gpu)
matOut = torch.zeros(N, HIDDEN, dtype=dtype, device=gpu)

defaultg = dae.get_group()
defaultg.addBarrier("bar_gateup_a")
defaultg.addBarrier("bar_gateup_b")
defaultg.addBarrier("bar_gateup_c")
defaultg.addBarrier("bar_gateup_d")
defaultg.addBarrier("bar_silu_ab")
defaultg.addBarrier("bar_silu_cd")
defaultg.addBarrier("bar_out")

defaultg.addTma("loadRMSLayer", [matRMSHidden], lambda t: t.wgmma_load(N, TileK * Gemv_M64N8.n_batch, Major.K))
defaultg.addTma("loadSiluLayer", [matSiLUOut], lambda t: t.wgmma_load(N, TileK * Gemv_M64N8.n_batch, Major.K))
defaultg.addTma("storeSiluLayer", [matSiLUOut], lambda t: t.wgmma_store(N, TileM, Major.MN))

defaultg.addTma("storeInterm", [matInterm], lambda t: t.wgmma_store(N, TileM, Major.MN))
defaultg.addTma("storeGateOut", [matGateOut], lambda t: t.wgmma_store(N, TileM, Major.MN))
defaultg.addTma("reduceOut", [matOut], lambda t: t.wgmma("reduce", N, TileM, Major.MN))

defaultg.addTma("loadDown", [matDown], lambda t: t.wgmma_load(TileM, TileK, Major.K))
defaultg.addTma("loadUp", [matUp], lambda t: t.wgmma_load(TileM, TileK, Major.K))
defaultg.addTma("loadGate", [matGate], lambda t: t.wgmma_load(TileM, TileK, Major.K))

dae.set_persistent(matRMSHidden)
dae.set_streaming(matGate, matUp, matDown)

dae.build_groups()

gate_a = SchedGemv(
    Gemv_M64N8,
    MNK=(MLP_A, N, HIDDEN),
    tmas=(defaultg["loadGate"], defaultg["loadRMSLayer"], defaultg["storeGateOut"]),
).bar("store", defaultg["bar_gateup_a"])
up_a = SchedGemv(
    Gemv_M64N8,
    MNK=(MLP_A, N, HIDDEN),
    tmas=(defaultg["loadUp"], defaultg["loadRMSLayer"], defaultg["storeInterm"]),
).bar("store", defaultg["bar_gateup_a"])

gate_b = SchedGemv(
    Gemv_M64N8,
    MNK=((MLP_A, MLP_B), N, HIDDEN),
    tmas=(defaultg["loadGate"], defaultg["loadRMSLayer"], defaultg["storeGateOut"]),
).bar("store", defaultg["bar_gateup_b"])
up_b = SchedGemv(
    Gemv_M64N8,
    MNK=((MLP_A, MLP_B), N, HIDDEN),
    tmas=(defaultg["loadUp"], defaultg["loadRMSLayer"], defaultg["storeInterm"]),
).bar("store", defaultg["bar_gateup_b"])

gate_c = SchedGemv(
    Gemv_M64N8,
    MNK=((MLP_AB, MLP_C), N, HIDDEN),
    tmas=(defaultg["loadGate"], defaultg["loadRMSLayer"], defaultg["storeGateOut"]),
).bar("store", defaultg["bar_gateup_c"])
up_c = SchedGemv(
    Gemv_M64N8,
    MNK=((MLP_AB, MLP_C), N, HIDDEN),
    tmas=(defaultg["loadUp"], defaultg["loadRMSLayer"], defaultg["storeInterm"]),
).bar("store", defaultg["bar_gateup_c"])

gate_d = SchedGemv(
    Gemv_M64N8,
    MNK=((MLP_AB + MLP_C, MLP_D), N, HIDDEN),
    tmas=(defaultg["loadGate"], defaultg["loadRMSLayer"], defaultg["storeGateOut"]),
).bar("store", defaultg["bar_gateup_d"])
up_d = SchedGemv(
    Gemv_M64N8,
    MNK=((MLP_AB + MLP_C, MLP_D), N, HIDDEN),
    tmas=(defaultg["loadUp"], defaultg["loadRMSLayer"], defaultg["storeInterm"]),
).bar("store", defaultg["bar_gateup_d"])

silu_a = SchedSmemSiLUInterleavedK(
    num_token=N,
    gate_glob=matGateOut[:, :MLP_A],
    up_glob=matInterm[:, :MLP_A],
    out_glob=matSiLUOut[:, :MLP_A],
).bar("input", defaultg["bar_gateup_a"]).bar("output", defaultg["bar_silu_ab"])
silu_b = SchedSmemSiLUInterleavedK(
    num_token=N,
    gate_glob=matGateOut[:, MLP_A:MLP_AB],
    up_glob=matInterm[:, MLP_A:MLP_AB],
    out_glob=matSiLUOut[:, MLP_A:MLP_AB],
).bar("input", defaultg["bar_gateup_b"]).bar("output", defaultg["bar_silu_ab"])
silu_c = SchedSmemSiLUInterleavedK(
    num_token=N,
    gate_glob=matGateOut[:, MLP_AB:MLP_AB + MLP_C],
    up_glob=matInterm[:, MLP_AB:MLP_AB + MLP_C],
    out_glob=matSiLUOut[:, MLP_AB:MLP_AB + MLP_C],
).bar("input", defaultg["bar_gateup_c"]).bar("output", defaultg["bar_silu_cd"])
silu_d = SchedSmemSiLUInterleavedK(
    num_token=N,
    gate_glob=matGateOut[:, MLP_AB + MLP_C:INTERMEDIATE],
    up_glob=matInterm[:, MLP_AB + MLP_C:INTERMEDIATE],
    out_glob=matSiLUOut[:, MLP_AB + MLP_C:INTERMEDIATE],
).bar("input", defaultg["bar_gateup_d"]).bar("output", defaultg["bar_silu_cd"])

down_0 = SchedGemv(
    Gemv_M64N8,
    MNK=(HIDDEN, N, MLP_AB),
    tmas=(defaultg["loadDown"], defaultg["loadSiluLayer"], defaultg["reduceOut"]),
).bar("load", defaultg["bar_silu_ab"])
down_1 = SchedGemv(
    Gemv_M64N8,
    MNK=(HIDDEN, N, (MLP_AB, MLP_CD)),
    tmas=(defaultg["loadDown"], defaultg["loadSiluLayer"], defaultg["reduceOut"]),
).bar("load", defaultg["bar_silu_cd"]).bar("store", defaultg["bar_out"])

gate_a = gate_a.place(32)
up_a = up_a.place(32, base_sm=32)
gate_b = gate_b.place(64)
up_b = up_b.place(64)
gate_c = gate_c.place(64)
up_c = up_c.place(64)
gate_d = gate_d.place(64)
up_d = up_d.place(64)
silu_a = silu_a.place(8, base_sm=64)
silu_b = silu_b.place(8, base_sm=64)
silu_c = silu_c.place(8, base_sm=64)
silu_d = silu_d.place(8, base_sm=64)
down_0 = down_0.place(64)
down_1 = down_1.place(64)

dae.bind_late_barrier_counts(
    gate_a,
    up_a,
    gate_b,
    up_b,
    gate_c,
    up_c,
    gate_d,
    up_d,
    silu_a,
    silu_b,
    silu_c,
    silu_d,
    down_0,
    down_1,
)

dae.i(
    gate_a,
    up_a,
    silu_a,
    gate_b,
    up_b,
    silu_b,
    down_0,
    gate_c,
    up_c,
    silu_c,
    gate_d,
    up_d,
    silu_d,
    down_1,
)

print(f"Llama3 MLP sched-only on [N={N}, HIDDEN={HIDDEN}, INTERMEDIATE={INTERMEDIATE}], SMs={full_sms}")
dae.s()
dae_app(dae)

if parsed_args.correctness:
    ref = F.linear(F.silu(F.linear(matHidden, matGate)) * F.linear(matHidden, matUp), matDown)
    tensor_diff("mlp_out", ref, matOut)
