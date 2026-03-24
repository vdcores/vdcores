from dae.launcher import *
from dae.schedule import *
from dae.tma_utils import ToConvertedCordAdapter
from dae.util import dae_app

from cli import parse_args
from runtime_context import build_runtime_context


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
QW = ctx.QW

matRMSHidden = ctx.matRMSHidden
matqWs = ctx.matqWs
attnQs = ctx.attnQs

weightg = dae.add_group("weights", 1)
sysg = dae.add_group("system", 1)
sysg.addBarrier("bar_q")

TileM64, _, TileK64 = Gemv_M64N8.MNK
weightg.addTma("loadRMSLayer64", [matRMSHidden], lambda t: t.wgmma_load(N, TileK64 * Gemv_M64N8.n_batch, Major.K))
weightg.addTma("loadQW", [matqWs], lambda t: t.indexed("layer").wgmma_load(TileM64, TileK64, Major.K))
weightg.addTma("storeQ", [attnQs], lambda t: t.wgmma("reduce", N, TileM64, Major.MN))
dae.build_groups()

dense_weight = lambda tma: IndexedWeightCordAdapter(tma, (0,))
q_proj = SchedGemv(
    Gemv_M64N8,
    MNK=(QW, N, HIDDEN),
    tmas=(dense_weight(weightg["loadQW"]), weightg["loadRMSLayer64"], weightg["storeQ"]),
).bar("store", sysg["bar_q"]).place(64)

dae.bind_late_barrier_counts(q_proj)
dae.i(SetLayerIndex(0), q_proj, IssueBarrier(sysg["bar_q"]))
dae.s()
dae_app(dae)
