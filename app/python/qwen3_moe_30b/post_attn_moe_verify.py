import argparse
import sys

from dae.launcher import *
from dae.schedule import *
from dae.tma_utils import StaticCordAdapter, ToConvertedCordAdapter
from dae.util import dae_app

from cli import parse_args
from runtime_context import build_runtime_context
from utils import build_tma_stacked_row, cord_func_stacked_row


parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("rms_only", "router_proj_only", "rms_router_proj", "rms_router", "router_only", "router_moe"), default="router_moe")
args, rest_argv = parser.parse_known_args()
sys.argv = [sys.argv[0], *rest_argv]


class IndexedWeightCordAdapter(ToConvertedCordAdapter):
    def __init__(self, inner, prefix):
        super().__init__(inner, lambda *cords: (*prefix, *cords))
        self.prefix = tuple(prefix)

    def cord2tma(self, *cords):
        return self.inner.cord2tma(*self.prefix, *cords)


ctx = build_runtime_context(parse_args())
dae = ctx.dae
N = ctx.N
HIDDEN = ctx.HIDDEN
MOE_INTERMEDIATE = ctx.MOE_INTERMEDIATE
TOP_K = ctx.TOP_K
eps = ctx.eps
rms_sms = ctx.rms_sms
full_sms = ctx.full_sms

matHidden = ctx.matHidden
matRMSHidden = ctx.matRMSHidden
matRMSPostAttnW = ctx.matRMSPostAttnW
matRouterWs = ctx.matRouterWs
matRouterLogits = ctx.matRouterLogits
matRouterTopKIdx = ctx.matRouterTopKIdx
matRouterTopKWeight = ctx.matRouterTopKWeight
matExpertAct = ctx.matExpertAct
matExpertGateWs = ctx.matExpertGateWs
matExpertUpWs = ctx.matExpertUpWs
matExpertDownWs = ctx.matExpertDownWs

weightg = dae.add_group("weights", 1)
sysg = dae.add_group("system", 1)
sysg.addBarrier("bar_post_attn_rms")
sysg.addBarrier("bar_router")
sysg.addBarrier("bar_router_topk")
sysg.addBarrier("bar_done")
for slot in range(TOP_K):
    sysg.addBarrier(f"bar_scale{slot}")

TileM64, _, TileK64 = Gemv_M64N8.MNK
_, _, TileKMma = Gemv_M64N8_MMA_SCALE.MNK

weightg.addTma("loadRMSLayer64", [matRMSHidden], lambda t: t.wgmma_load(N, TileK64 * Gemv_M64N8.n_batch, Major.K))
weightg.addTma("reduceHiddenLayer", [matHidden], lambda t: t.wgmma("reduce", N, TileM64, Major.MN))
weightg.addTma("storeRouterLogits", [matRouterLogits], lambda t: t.wgmma_store(N, TileM64, Major.MN))
weightg.addTma("loadRMSPostAttnW", [matRMSPostAttnW], lambda t: t.indexed("layer")._build("load", HIDDEN, 1, build_tma_stacked_row, cord_func_stacked_row))
weightg.addTma("loadRouterWs", [matRouterWs], lambda t: t.indexed("layer").wgmma_load(TileM64, TileK64, Major.K))
weightg.addTma("loadExpertGateWs", [matExpertGateWs], lambda t: t.indexed("layer_expert").wgmma_load(TileM64, TileK64, Major.K))
weightg.addTma("loadExpertUpWs", [matExpertUpWs], lambda t: t.indexed("layer_expert").wgmma_load(TileM64, TileK64, Major.K))
weightg.addTma("loadExpertDownWs", [matExpertDownWs], lambda t: t.indexed("layer_expert").wgmma_load(TileM64, TileKMma, Major.K))
for slot in range(TOP_K):
    weightg.addTma(f"storeExpertAct{slot}", [matExpertAct[slot]], lambda t: t.wgmma_store(N, TileM64, Major.MN))
    weightg.addTma(f"loadExpertAct{slot}", [matExpertAct[slot]], lambda t: t.wgmma_load(N, TileKMma, Major.K))
dae.build_groups()

dense_weight = lambda tma: IndexedWeightCordAdapter(tma, (0,))
expert_weight = lambda tma: IndexedWeightCordAdapter(tma, (0, 0))

post_attn_rms = SchedRMSShared(
    num_token=N,
    epsilon=eps,
    hidden_size=HIDDEN,
    tmas=(dense_weight(weightg["loadRMSPostAttnW"]), TmaLoad1D(matHidden), TmaStore1D(matRMSHidden)),
).bar("output", sysg["bar_post_attn_rms"]).place(rms_sms)

router_proj = SchedGemv(
    Gemv_M64N8,
    MNK=(128, N, HIDDEN),
    tmas=(dense_weight(weightg["loadRouterWs"]), weightg["loadRMSLayer64"], weightg["storeRouterLogits"]),
).bar("load", sysg["bar_post_attn_rms"]).bar("store", sysg["bar_router"]).place(2)
router_proj_only = SchedGemv(
    Gemv_M64N8,
    MNK=(128, N, HIDDEN),
    tmas=(dense_weight(weightg["loadRouterWs"]), weightg["loadRMSLayer64"], weightg["storeRouterLogits"]),
).bar("store", sysg["bar_router"]).place(2)

router_topk = SchedRouterTopK(
    num_token=N,
    logits_tma=StaticCordAdapter(TmaLoad1D(matRouterLogits)),
    weight_tma=StaticCordAdapter(TmaStore1D(matRouterTopKWeight)),
    idx_addr=StaticCordAdapter(RawAddress(matRouterTopKIdx, 24)),
).bar("input", sysg["bar_router"]).bar("output", sysg["bar_router_topk"]).place(1, base_sm=128)

ops = [SetLayerIndex(0)]
late_bar_schedules = []

if args.mode == "router_proj_only":
    ops.extend([router_proj_only, IssueBarrier(sysg["bar_router"])])
    late_bar_schedules.append(router_proj_only)
elif args.mode == "rms_only":
    ops.append(post_attn_rms)
    late_bar_schedules.append(post_attn_rms)
    ops.append(IssueBarrier(sysg["bar_post_attn_rms"]))
else:
    ops.append(post_attn_rms)
    late_bar_schedules.append(post_attn_rms)
    ops.append(router_proj)
    late_bar_schedules.append(router_proj)
    if args.mode == "rms_router_proj":
        ops.append(IssueBarrier(sysg["bar_router"]))
    else:
        ops.extend([router_topk, IssueBarrier(sysg["bar_router_topk"])])
        late_bar_schedules.append(router_topk)

if args.mode == "router_moe":
    reg_gate, reg_up = 0, 1
    for slot in range(TOP_K):
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
        if slot == TOP_K - 1:
            down_proj.bar("store", sysg["bar_done"])
            late_bar_schedules.append(down_proj)

        ops.extend([LoadExpertIndex(matRouterTopKIdx, slot), gate_proj, up_proj, silu, down_proj])
        late_bar_schedules.append(silu)

    ops.append(IssueBarrier(sysg["bar_done"]))

used_bars = set()
for sched in late_bar_schedules:
    used_bars.update(sched._bars.values())
for name in ("bar_post_attn_rms", "bar_router", "bar_router_topk", "bar_done", *(f"bar_scale{slot}" for slot in range(TOP_K))):
    bar_id = sysg[name]
    if bar_id not in used_bars:
        sysg.bindBarrier(name, 0)

dae.bind_late_barrier_counts(*late_bar_schedules)
dae.i(*ops)
dae.s()
dae_app(dae)
