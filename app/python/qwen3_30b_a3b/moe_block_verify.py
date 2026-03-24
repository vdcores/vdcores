import argparse
import sys

import torch
from dae.launcher import *
from dae.schedule import *
from dae.tma_utils import StaticCordAdapter, ToConvertedCordAdapter
from dae.util import dae_app


class IndexedWeightCordAdapter(ToConvertedCordAdapter):
    def __init__(self, inner, prefix):
        super().__init__(inner, lambda *cords: (*prefix, *cords))
        self.prefix = tuple(prefix)

    def cord2tma(self, *cords):
        return self.inner.cord2tma(*self.prefix, *cords)


parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("full", "gate_only", "gate_store", "up_only", "gate_up", "gate_up_silu", "down_only"), default="full")
args, rest_argv = parser.parse_known_args()
sys.argv = [sys.argv[0], *rest_argv]

gpu = torch.device("cuda")
torch.manual_seed(0)

N = 8
HIDDEN = 2048
MOE_INTERMEDIATE = 768
TOP_K = 8
NUM_LAYERS = 1
NUM_EXPERTS = 128
TileM64, _, TileK64 = Gemv_M64N8.MNK
_, _, TileKMma = Gemv_M64N8_MMA_SCALE.MNK

matRMSHidden = torch.rand(N, HIDDEN, dtype=torch.bfloat16, device=gpu) - 0.5
matHidden = torch.zeros(N, HIDDEN, dtype=torch.bfloat16, device=gpu)
matExpertAct = torch.zeros(N, MOE_INTERMEDIATE, dtype=torch.bfloat16, device=gpu)
matGateTile = torch.zeros(N, TileM64, dtype=torch.bfloat16, device=gpu)
matRouterTopKIdx = torch.tensor([[3] + [0] * (TOP_K - 1)], dtype=torch.int32, device=gpu)
matRouterTopKWeight = torch.zeros(TOP_K, 32, dtype=torch.bfloat16, device=gpu)
matRouterTopKWeight[0, 0] = 1

matExpertGateWs = torch.rand(NUM_LAYERS, NUM_EXPERTS, MOE_INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device=gpu) - 0.5
matExpertUpWs = torch.rand(NUM_LAYERS, NUM_EXPERTS, MOE_INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device=gpu) - 0.5
matExpertDownWs = torch.rand(NUM_LAYERS, NUM_EXPERTS, HIDDEN, MOE_INTERMEDIATE, dtype=torch.bfloat16, device=gpu) - 0.5

dae = Launcher(32, device=gpu)
weightg = dae.add_group("weights", 1)
sysg = dae.add_group("system", 1)
sysg.addBarrier("bar_gate")
sysg.addBarrier("bar_up")
sysg.addBarrier("bar_silu")
sysg.addBarrier("bar_done")

weightg.addTma("loadRMSLayer64", [matRMSHidden], lambda t: t.wgmma_load(N, TileK64 * Gemv_M64N8.n_batch, Major.K))
weightg.addTma("reduceHiddenLayer", [matHidden], lambda t: t.wgmma("reduce", N, TileM64, Major.MN))
weightg.addTma("loadExpertGateWs", [matExpertGateWs], lambda t: t.indexed("layer_expert").wgmma_load(TileM64, TileK64, Major.K))
weightg.addTma("loadExpertUpWs", [matExpertUpWs], lambda t: t.indexed("layer_expert").wgmma_load(TileM64, TileK64, Major.K))
weightg.addTma("loadExpertDownWs", [matExpertDownWs], lambda t: t.indexed("layer_expert").wgmma_load(TileM64, TileKMma, Major.K))
weightg.addTma("storeExpertAct", [matExpertAct], lambda t: t.wgmma_store(N, TileM64, Major.MN))
weightg.addTma("storeGateTile", [matGateTile], lambda t: t.wgmma_store(N, TileM64, Major.MN))
weightg.addTma("loadExpertAct", [matExpertAct], lambda t: t.wgmma_load(N, TileKMma, Major.K))
dae.build_groups()

expert_weight = lambda tma: IndexedWeightCordAdapter(tma, (0, 0))
reg_gate, reg_up = 0, 1

gate_proj = SchedGemv(
    Gemv_M64N8,
    MNK=(MOE_INTERMEDIATE, N, HIDDEN),
    tmas=(
        expert_weight(weightg["loadExpertGateWs"]),
        weightg["loadRMSLayer64"],
        RegStore(reg_gate, matExpertAct[:, :TileM64]),
    ),
).bar("store", sysg["bar_gate"]).place(12)

gate_proj_nobar = SchedGemv(
    Gemv_M64N8,
    MNK=(MOE_INTERMEDIATE, N, HIDDEN),
    tmas=(
        expert_weight(weightg["loadExpertGateWs"]),
        weightg["loadRMSLayer64"],
        RegStore(reg_gate, matExpertAct[:, :TileM64]),
    ),
).place(12)

gate_store_proj = SchedGemv(
    Gemv_M64N8,
    MNK=(MOE_INTERMEDIATE, N, HIDDEN),
    tmas=(
        expert_weight(weightg["loadExpertGateWs"]),
        weightg["loadRMSLayer64"],
        weightg["storeGateTile"],
    ),
).bar("store", sysg["bar_gate"]).place(12)

up_proj = SchedGemv(
    Gemv_M64N8,
    MNK=(MOE_INTERMEDIATE, N, HIDDEN),
    tmas=(
        expert_weight(weightg["loadExpertUpWs"]),
        weightg["loadRMSLayer64"],
        RegStore(reg_up, matExpertAct[:, :TileM64]),
    ),
).bar("store", sysg["bar_up"]).place(12)

up_proj_nobar = SchedGemv(
    Gemv_M64N8,
    MNK=(MOE_INTERMEDIATE, N, HIDDEN),
    tmas=(
        expert_weight(weightg["loadExpertUpWs"]),
        weightg["loadRMSLayer64"],
        RegStore(reg_up, matExpertAct[:, :TileM64]),
    ),
).place(12)

silu = SchedRegSiLUFused(
    num_token=N,
    store_tma=weightg["storeExpertAct"],
    reg_gate=reg_gate,
    reg_up=reg_up,
    base_offset=0,
    stride=TileM64,
).bar("output", sysg["bar_silu"]).place(12)

down_proj = SchedGemvMmaScale(
    Gemv_M64N8_MMA_SCALE,
    MNK=(HIDDEN, N, MOE_INTERMEDIATE),
    tmas=(
        expert_weight(weightg["loadExpertDownWs"]),
        weightg["loadExpertAct"],
        StaticCordAdapter(TmaLoad1D(matRouterTopKWeight[0], bytes=64)),
        weightg["reduceHiddenLayer"],
    ),
).bar("load", sysg["bar_silu"]).bar("store", sysg["bar_done"]).place(32)

ops = [
    SetLayerIndex(0),
    LoadExpertIndex(matRouterTopKIdx, 0),
]
late_bar_schedules = []

if args.mode == "gate_store":
    ops.extend([gate_store_proj, IssueBarrier(sysg["bar_gate"])])
    late_bar_schedules.append(gate_store_proj)

if args.mode in ("gate_up", "gate_up_silu", "full"):
    ops.append(gate_proj_nobar)
if args.mode in ("gate_only",):
    ops.append(gate_proj)
    late_bar_schedules.append(gate_proj)
    if args.mode == "gate_only":
        ops.append(IssueBarrier(sysg["bar_gate"]))

if args.mode in ("gate_up", "gate_up_silu", "full"):
    ops.append(up_proj_nobar)
if args.mode in ("up_only",):
    ops.append(up_proj)
    late_bar_schedules.append(up_proj)
    if args.mode == "up_only":
        ops.append(IssueBarrier(sysg["bar_up"]))

if args.mode in ("gate_up", "gate_up_silu", "full"):
    if args.mode == "gate_up":
        ops.extend([
            IssueBarrier(sysg["bar_gate"]),
            IssueBarrier(sysg["bar_up"]),
        ])

if args.mode in ("gate_up_silu", "full"):
    ops.extend([
        silu,
    ])
    late_bar_schedules.append(silu)
    if args.mode == "gate_up_silu":
        ops.append(IssueBarrier(sysg["bar_silu"]))

if args.mode == "down_only":
    down_only = SchedGemvMmaScale(
        Gemv_M64N8_MMA_SCALE,
        MNK=(HIDDEN, N, MOE_INTERMEDIATE),
        tmas=(
            expert_weight(weightg["loadExpertDownWs"]),
            weightg["loadExpertAct"],
            StaticCordAdapter(TmaLoad1D(matRouterTopKWeight[0], bytes=64)),
            weightg["reduceHiddenLayer"],
        ),
    ).bar("store", sysg["bar_done"]).place(32)
    ops.extend([down_only, IssueBarrier(sysg["bar_done"])])
    late_bar_schedules.append(down_only)
elif args.mode == "full":
    ops.extend([down_proj, IssueBarrier(sysg["bar_done"])])
    late_bar_schedules.append(down_proj)

dae.i(*ops)

used_bars = set()
for sched in late_bar_schedules:
    used_bars.update(sched._bars.values())
for name in ("bar_gate", "bar_up", "bar_silu", "bar_done"):
    bar_id = sysg[name]
    if bar_id not in used_bars:
        sysg.bindBarrier(name, 0)

dae.bind_late_barrier_counts(*late_bar_schedules)
dae.s()
dae_app(dae)
