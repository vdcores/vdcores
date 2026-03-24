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
parser.add_argument("--slots", type=int, default=8)
args, rest_argv = parser.parse_known_args()
sys.argv = [sys.argv[0], *rest_argv]

gpu = torch.device("cuda")
torch.manual_seed(0)

N = 8
HIDDEN = 2048
MOE_INTERMEDIATE = 768
NUM_LAYERS = 1
NUM_EXPERTS = 128
TileM64, _, TileK64 = Gemv_M64N8.MNK
_, _, TileKMma = Gemv_M64N8_MMA_SCALE.MNK

matRMSHidden = torch.rand(N, HIDDEN, dtype=torch.bfloat16, device=gpu) - 0.5
matHidden = torch.zeros(N, HIDDEN, dtype=torch.bfloat16, device=gpu)
matExpertAct = [torch.zeros(N, MOE_INTERMEDIATE, dtype=torch.bfloat16, device=gpu) for _ in range(args.slots)]

topk_idx = list(range(args.slots))
matRouterTopKIdx = torch.tensor([topk_idx], dtype=torch.int32, device=gpu)
matRouterTopKWeight = torch.zeros(args.slots, 32, dtype=torch.bfloat16, device=gpu)
for slot in range(args.slots):
    matRouterTopKWeight[slot, 0] = 1

matExpertGateWs = torch.rand(NUM_LAYERS, NUM_EXPERTS, MOE_INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device=gpu) - 0.5
matExpertUpWs = torch.rand(NUM_LAYERS, NUM_EXPERTS, MOE_INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device=gpu) - 0.5
matExpertDownWs = torch.rand(NUM_LAYERS, NUM_EXPERTS, HIDDEN, MOE_INTERMEDIATE, dtype=torch.bfloat16, device=gpu) - 0.5

dae = Launcher(32, device=gpu)
weightg = dae.add_group("weights", 1)
sysg = dae.add_group("system", 1)
sysg.addBarrier("bar_done")
for slot in range(args.slots):
    sysg.addBarrier(f"bar_scale{slot}")

weightg.addTma("loadRMSLayer64", [matRMSHidden], lambda t: t.wgmma_load(N, TileK64 * Gemv_M64N8.n_batch, Major.K))
weightg.addTma("reduceHiddenLayer", [matHidden], lambda t: t.wgmma("reduce", N, TileM64, Major.MN))
weightg.addTma("loadExpertGateWs", [matExpertGateWs], lambda t: t.indexed("layer_expert").wgmma_load(TileM64, TileK64, Major.K))
weightg.addTma("loadExpertUpWs", [matExpertUpWs], lambda t: t.indexed("layer_expert").wgmma_load(TileM64, TileK64, Major.K))
weightg.addTma("loadExpertDownWs", [matExpertDownWs], lambda t: t.indexed("layer_expert").wgmma_load(TileM64, TileKMma, Major.K))
for slot in range(args.slots):
    weightg.addTma(f"storeExpertAct{slot}", [matExpertAct[slot]], lambda t: t.wgmma_store(N, TileM64, Major.MN))
    weightg.addTma(f"loadExpertAct{slot}", [matExpertAct[slot]], lambda t: t.wgmma_load(N, TileKMma, Major.K))
dae.build_groups()

expert_weight = lambda tma: IndexedWeightCordAdapter(tma, (0, 0))
reg_gate, reg_up = 0, 1

ops = [SetLayerIndex(0)]
late_bar_schedules = []

for slot in range(args.slots):
    gate_proj = SchedGemv(
        Gemv_M64N8,
        MNK=(MOE_INTERMEDIATE, N, HIDDEN),
        tmas=(
            expert_weight(weightg["loadExpertGateWs"]),
            weightg["loadRMSLayer64"],
            RegStore(reg_gate, matExpertAct[slot][:, :TileM64]),
        ),
    ).place(12)

    up_proj = SchedGemv(
        Gemv_M64N8,
        MNK=(MOE_INTERMEDIATE, N, HIDDEN),
        tmas=(
            expert_weight(weightg["loadExpertUpWs"]),
            weightg["loadRMSLayer64"],
            RegStore(reg_up, matExpertAct[slot][:, :TileM64]),
        ),
    ).place(12)

    silu = SchedRegSiLUFused(
        num_token=N,
        store_tma=weightg[f"storeExpertAct{slot}"],
        reg_gate=reg_gate,
        reg_up=reg_up,
        base_offset=0,
        stride=TileM64,
    ).bar("output", sysg[f"bar_scale{slot}"]).place(12)

    down_proj = SchedGemvMmaScale(
        Gemv_M64N8_MMA_SCALE,
        MNK=(HIDDEN, N, MOE_INTERMEDIATE),
        tmas=(
            expert_weight(weightg["loadExpertDownWs"]),
            weightg[f"loadExpertAct{slot}"],
            StaticCordAdapter(TmaLoad1D(matRouterTopKWeight[slot], bytes=64)),
            weightg["reduceHiddenLayer"],
        ),
    ).bar("load", sysg[f"bar_scale{slot}"]).place(32)
    if slot == args.slots - 1:
        down_proj.bar("store", sysg["bar_done"])
        late_bar_schedules.append(down_proj)

    ops.extend([
        LoadExpertIndex(matRouterTopKIdx, slot),
        gate_proj,
        up_proj,
        silu,
        down_proj,
    ])
    late_bar_schedules.append(silu)

ops.append(IssueBarrier(sysg["bar_done"]))

dae.i(*ops)
dae.bind_late_barrier_counts(*late_bar_schedules)
dae.s()
dae_app(dae)
