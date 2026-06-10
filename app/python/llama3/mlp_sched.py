import argparse
import sys

import torch
import torch.nn.functional as F

from dae.launcher import *
from dae.schedule import *
from dae.util import dae_app, tensor_diff


arg_parser = argparse.ArgumentParser(add_help=False)
arg_parser.add_argument("--correctness", action="store_true")
arg_parser.add_argument("--skip-down", action="store_true")
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
MLP_TILES = (74, 74, 76)
MLP_0 = MLP_TILES[0] * 64
MLP_1 = MLP_TILES[1] * 64
MLP_2 = MLP_TILES[2] * 64
MLP_01 = MLP_0 + MLP_1

num_sms = 128
full_sms = 132
dae = Launcher(full_sms, device=gpu)

TileM, _, TileK = Gemv_M64N8.MNK
assert MLP_0 + MLP_1 + MLP_2 == INTERMEDIATE

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
defaultg.addBarrier("bar_gateup_0")
defaultg.addBarrier("bar_gateup_1")
defaultg.addBarrier("bar_silu_low")
defaultg.addBarrier("bar_silu_tail")
defaultg.addBarrier("bar_out")

defaultg.addTma("loadRMSLayer", [matRMSHidden], lambda t: t.wgmma_load(N, TileK * Gemv_M64N8.n_batch, Major.K))
defaultg.addTma("loadSiluLayer", [matSiLUOut], lambda t: t.wgmma_load(N, TileK * Gemv_M64N8.n_batch, Major.K))
defaultg.addTma("storeSiluLayer", [matSiLUOut], lambda t: t.wgmma_store(N, TileM, Major.MN))

defaultg.addTma("storeInterm", [matInterm], lambda t: t.wgmma_store(N, TileM, Major.MN))
defaultg.addTma("storeGateOut", [matGateOut], lambda t: t.wgmma_store(N, TileM, Major.MN))
defaultg.addTma("reduceOut", [matOut], lambda t: t.wgmma("reduce", N, TileM, Major.MN))
finish_bytes = 64
finish_elems = finish_bytes // matOut.element_size()
defaultg.addTma("loadFinish", [matOut[:, :finish_elems]], lambda t: t.tensor1d("load", finish_bytes))
defaultg.addTma("storeFinish", [matOut[:, :finish_elems]], lambda t: t.tensor1d("store", finish_bytes))

defaultg.addTma("loadDown", [matDown], lambda t: t.wgmma_load(TileM, TileK, Major.K))
defaultg.addTma("loadUp", [matUp], lambda t: t.wgmma_load(TileM, TileK, Major.K))
defaultg.addTma("loadGate", [matGate], lambda t: t.wgmma_load(TileM, TileK, Major.K))

dae.set_persistent(matRMSHidden)
dae.set_streaming(matGate, matUp, matDown)

dae.build_groups()

reg_gate, reg_up = 0, 1
regStoreGate = RegStore(reg_gate, matGateOut[:, 0:TileM])
regStoreUp = RegStore(reg_up, matInterm[:, 0:TileM])

gate_0 = SchedGemv(
    Gemv_M64N8,
    MNK=(MLP_0, N, HIDDEN),
    tmas=(defaultg["loadGate"], defaultg["loadRMSLayer"], defaultg["storeGateOut"]),
).bar("store", defaultg["bar_gateup_0"]).place(MLP_TILES[0])
up_0 = SchedGemv(
    Gemv_M64N8,
    MNK=(MLP_0, N, HIDDEN),
    tmas=(defaultg["loadUp"], defaultg["loadRMSLayer"], defaultg["storeInterm"]),
).bar("store", defaultg["bar_gateup_0"]).place(MLP_TILES[0])
silu_0 = SchedSmemSiLUInterleavedK(
    num_token=N,
    gate_glob=matGateOut[:, :MLP_0],
    up_glob=matInterm[:, :MLP_0],
    out_glob=matSiLUOut[:, :MLP_0],
).bar("input", defaultg["bar_gateup_0"]).bar("output", defaultg["bar_silu_low"]).place(8, base_sm=64)

gate_1 = SchedGemv(
    Gemv_M64N8,
    MNK=((MLP_0, MLP_1), N, HIDDEN),
    tmas=(defaultg["loadGate"], defaultg["loadRMSLayer"], defaultg["storeGateOut"]),
).bar("store", defaultg["bar_gateup_1"]).place(MLP_TILES[1])
up_1 = SchedGemv(
    Gemv_M64N8,
    MNK=((MLP_0, MLP_1), N, HIDDEN),
    tmas=(defaultg["loadUp"], defaultg["loadRMSLayer"], defaultg["storeInterm"]),
).bar("store", defaultg["bar_gateup_1"]).place(MLP_TILES[1])
silu_1 = SchedSmemSiLUInterleavedK(
    num_token=N,
    gate_glob=matGateOut[:, MLP_0:MLP_01],
    up_glob=matInterm[:, MLP_0:MLP_01],
    out_glob=matSiLUOut[:, MLP_0:MLP_01],
).bar("input", defaultg["bar_gateup_1"]).bar("output", defaultg["bar_silu_low"]).place(8, base_sm=64)

gate_tail = SchedGemv(
    Gemv_M64N8,
    MNK=((MLP_01, MLP_2), N, HIDDEN),
    tmas=(defaultg["loadGate"], defaultg["loadRMSLayer"], regStoreGate),
).place(MLP_TILES[2])
up_tail = SchedGemv(
    Gemv_M64N8,
    MNK=((MLP_01, MLP_2), N, HIDDEN),
    tmas=(defaultg["loadUp"], defaultg["loadRMSLayer"], regStoreUp),
).place(MLP_TILES[2])
silu_tail = SchedRegSiLUFused(
    num_token=N,
    store_tma=defaultg["storeSiluLayer"],
    reg_gate=reg_gate,
    reg_up=reg_up,
    base_offset=MLP_01,
    stride=TileM,
).bar("output", defaultg["bar_silu_tail"]).place(MLP_TILES[2])

DOWN_FOLD_M = 14 * TileM
DOWN_REST_M = HIDDEN - DOWN_FOLD_M

down_low_fold = SchedGemv(
    Gemv_M64N8,
    MNK=(DOWN_FOLD_M, N, 8192),
    tmas=(defaultg["loadDown"], defaultg["loadSiluLayer"], defaultg["reduceOut"]),
).bar("load", defaultg["bar_silu_low"]).place(28)
down_low_rest = SchedGemv(
    Gemv_M64N8,
    MNK=((DOWN_FOLD_M, DOWN_REST_M), N, 8192),
    tmas=(defaultg["loadDown"], defaultg["loadSiluLayer"], defaultg["reduceOut"]),
).bar("load", defaultg["bar_silu_low"]).place(50, base_sm=28)
down_tail_fold = SchedGemv(
    Gemv_M64N8,
    MNK=(DOWN_FOLD_M, N, (8192, INTERMEDIATE - 8192)),
    tmas=(defaultg["loadDown"], defaultg["loadSiluLayer"], defaultg["reduceOut"]),
).bar("load", defaultg["bar_silu_tail"]).bar("store", defaultg["bar_out"]).place(28)
down_tail_rest = SchedGemv(
    Gemv_M64N8,
    MNK=((DOWN_FOLD_M, DOWN_REST_M), N, (8192, INTERMEDIATE - 8192)),
    tmas=(defaultg["loadDown"], defaultg["loadSiluLayer"], defaultg["reduceOut"]),
).bar("load", defaultg["bar_silu_tail"]).bar("store", defaultg["bar_out"]).place(50, base_sm=28)
finish_without_down = SchedCopy(
    tmas=(defaultg["loadFinish"], defaultg["storeFinish"]),
    size=finish_bytes,
).bar("load", defaultg["bar_silu_tail"]).bar("store", defaultg["bar_out"]).place(1)

ordered_schedules = [
    gate_0,
    up_0,
    silu_0,
    gate_1,
    up_1,
    silu_1,
]
if not parsed_args.skip_down:
    ordered_schedules.extend([down_low_fold, down_low_rest])
ordered_schedules.extend([
    gate_tail,
    up_tail,
    silu_tail,
])
if parsed_args.skip_down:
    ordered_schedules.append(finish_without_down)
else:
    ordered_schedules.extend([down_tail_fold, down_tail_rest])

dae.bind_late_barrier_counts(ordered_schedules)
dae.i(ordered_schedules)

mode = "skip-down" if parsed_args.skip_down else "with-down"
print(
    f"Llama3 MLP sched-only ({mode}, two side SiLU chunks, fused SiLU tail) "
    f"on [N={N}, HIDDEN={HIDDEN}, INTERMEDIATE={INTERMEDIATE}], "
    f"split=({MLP_0}, {MLP_1}, {MLP_2}), tile split={MLP_TILES}, SMs={full_sms}"
)
dae.s()
dae_app(dae)

if parsed_args.correctness and not parsed_args.skip_down:
    ref = F.linear(F.silu(F.linear(matHidden, matGate)) * F.linear(matHidden, matUp), matDown)
    tensor_diff("mlp_out", ref, matOut)
elif parsed_args.correctness:
    print("[correctness] skipped because --skip-down leaves matOut incomplete")
