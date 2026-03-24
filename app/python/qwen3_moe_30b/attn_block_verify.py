from functools import partial

from dae.launcher import *
from dae.model import *
from dae.schedule import *
from dae.tma_utils import ToAttnVStoreCordAdapter, ToConvertedCordAdapter
from dae.util import dae_app

from cli import parse_args
from runtime_context import build_runtime_context
from utils import build_tma_stacked_row, build_tma_wgmma_k, build_tma_wgmma_mn, cord_func_K_major, cord_func_MN_major, cord_func_stacked_row


class IndexedWeightCordAdapter(ToConvertedCordAdapter):
    def __init__(self, inner, prefix):
        super().__init__(inner, lambda *cords: (*prefix, *cords))
        self.prefix = tuple(prefix)

    def cord2tma(self, *cords):
        return self.inner.cord2tma(*self.prefix, *cords)


ctx = build_runtime_context(parse_args())
dae = ctx.dae
REQ = ctx.REQ
N = ctx.N
KVBlockSize = ctx.KVBlockSize
MAX_SEQ_LEN = ctx.MAX_SEQ_LEN
HIDDEN = ctx.HIDDEN
HEAD_DIM = ctx.HEAD_DIM
NUM_KV_HEAD = ctx.NUM_KV_HEAD
HEAD_GROUP_SIZE = ctx.HEAD_GROUP_SIZE
QW = ctx.QW
KW = ctx.KW
VW = ctx.VW

matRMSHidden = ctx.matRMSHidden
attnQs = ctx.attnQs
attnKs = ctx.attnKs
attnVs = ctx.attnVs
attnO = ctx.attnO
matQwenSideInputs = ctx.matQwenSideInputs
matqWs = ctx.matqWs
matkWs = ctx.matkWs
matvWs = ctx.matvWs
matOutWs = ctx.matOutWs
token_pos = 0

weightg = dae.add_group("weights", 1)
sysg = dae.add_group("system", 1)
for name in ("bar_q_proj", "bar_qkv_attn", "bar_attn_out", "bar_out"):
    sysg.addBarrier(name)

TileM64, _, TileK64 = Gemv_M64N8.MNK
weightg.addTma("loadRMSLayer64", [matRMSHidden], lambda t: t.wgmma_load(N, TileK64 * Gemv_M64N8.n_batch, Major.K))
weightg.addTma("loadQW", [matqWs], lambda t: t.indexed("layer").wgmma_load(TileM64, TileK64, Major.K))
weightg.addTma("loadKW", [matkWs], lambda t: t.indexed("layer").wgmma_load(TileM64, TileK64, Major.K))
weightg.addTma("loadVW", [matvWs], lambda t: t.indexed("layer").wgmma_load(TileM64, TileK64, Major.K))
weightg.addTma("loadOutWs", [matOutWs], lambda t: t.indexed("layer").wgmma_load(TileM64, TileK64, Major.K))
weightg.addTma("loadQwenSideInput", [matQwenSideInputs], lambda t: t.indexed("layer")._build("load", 3 * HEAD_DIM, 1, build_tma_stacked_row, cord_func_stacked_row))
weightg.addTma("storeQ", [attnQs], lambda t: t.wgmma("reduce", N, TileM64, Major.MN))

tma_builder_MN = partial(build_tma_wgmma_mn, iK=-3)
cord_func_MN = partial(cord_func_MN_major, iK=-3)
tma_builder_K = partial(build_tma_wgmma_k, iN=-3)
cord_func_K = partial(cord_func_K_major, iN=-3)

weightg.addTma("storeK", [attnKs], lambda t: t._build("reduce", 64, N, tma_store_attn_kv, cord_id))
weightg.addTma("storeV", [attnVs], lambda t: t._build("reduce", 64, N, tma_store_attn_kv, cord_id))
matQ_attn_view = attnQs.view(N, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)
matK_attn_view = attnKs.view(N, MAX_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)
matV_attn_view = attnVs.view(N, MAX_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)
matO_attn_view = attnO.view(N, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)
weightg.addTma("loadQ", [matQ_attn_view], lambda t: t._build("load", HEAD_DIM, 64, tma_gqa_load_q, cord_gqa_load_q))
weightg.addTma("loadK", [matK_attn_view], lambda t: t._build("load", HEAD_DIM, KVBlockSize, tma_builder_K, cord_func_K))
weightg.addTma("loadV", [matV_attn_view], lambda t: t._build("load", HEAD_DIM, KVBlockSize, tma_builder_MN, cord_func_MN))
weightg.addTma("loadAttnOLayer", [attnO], lambda t: t.wgmma_load(N, TileK64 * Gemv_M64N8.n_batch, Major.K))
weightg.addTma("reduceHiddenLayer", [ctx.matHidden], lambda t: t.wgmma("reduce", N, TileM64, Major.MN))
dae.build_groups()

dense_weight = lambda tma: IndexedWeightCordAdapter(tma, (0,))
side_input = ToConvertedCordAdapter(
    weightg["loadQwenSideInput"],
    lambda addr: (0, addr * matQwenSideInputs.element_size()),
)
current_k_store = [
    ToConvertedCordAdapter(
        TmaStore1D(attnKs[req], bytes=HEAD_DIM * attnKs.element_size()),
        lambda addr: (addr * attnKs.element_size(),),
    )
    for req in range(REQ)
]

q_proj = SchedGemv(
    Gemv_M64N8,
    MNK=(QW, N, HIDDEN),
    tmas=(dense_weight(weightg["loadQW"]), weightg["loadRMSLayer64"], weightg["storeQ"]),
).bar("store", sysg["bar_q_proj"]).place(64)
k_proj = SchedGemv(
    Gemv_M64N8,
    MNK=(KW, N, HIDDEN),
    tmas=(dense_weight(weightg["loadKW"]), weightg["loadRMSLayer64"], ToAttnVStoreCordAdapter(weightg["storeK"], token_pos)),
).bar("store", sysg["bar_qkv_attn"]).place(16, base_sm=64)
v_proj = SchedGemv(
    Gemv_M64N8,
    MNK=(VW, N, HIDDEN),
    tmas=(dense_weight(weightg["loadVW"]), weightg["loadRMSLayer64"], ToAttnVStoreCordAdapter(weightg["storeV"], token_pos)),
).bar("store", sysg["bar_qkv_attn"]).place(16, base_sm=80)
gqa = SchedAttentionDecoding(
    reqs=N,
    seq_len=token_pos + 1,
    KV_BLOCK_SIZE=KVBlockSize,
    NUM_KV_HEADS=NUM_KV_HEAD,
    matO=matO_attn_view,
    tmas=(weightg["loadQ"], weightg["loadK"], weightg["loadV"]),
    side_input=side_input,
    k_store=current_k_store,
    token_pos=token_pos,
).bar("q", sysg["bar_q_proj"]).bar("k", sysg["bar_qkv_attn"]).bar("o", sysg["bar_attn_out"]).place(N * NUM_KV_HEAD)
out_proj = SchedGemv(
    Gemv_M64N8,
    MNK=(HIDDEN, N, QW),
    tmas=(dense_weight(weightg["loadOutWs"]), weightg["loadAttnOLayer"], weightg["reduceHiddenLayer"]),
).bar("load", sysg["bar_attn_out"]).bar("store", sysg["bar_out"]).place(64)

dae.bind_late_barrier_counts(q_proj, k_proj, v_proj, gqa, out_proj)
dae.i(SetLayerIndex(0), q_proj, k_proj, v_proj, gqa, out_proj, IssueBarrier(sysg["bar_out"]))
dae.s()
dae_app(dae)
