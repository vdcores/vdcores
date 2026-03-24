from functools import partial

from dae.launcher import *
from dae.model import *
from dae.schedule import *
from dae.tma_utils import StaticCordAdapter, ToAttnVStoreCordAdapter, ToConvertedCordAdapter, wrap_static
from dae.util import dae_app

from cli import parse_args
from correctness import run_correctness_check
from runtime_context import build_runtime_context, seed_prefill_kv_cache
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
rms_sms = ctx.rms_sms
full_sms = ctx.full_sms
MAX_SEQ_LEN = ctx.MAX_SEQ_LEN
eps = ctx.eps
HIDDEN = ctx.HIDDEN
MOE_INTERMEDIATE = ctx.MOE_INTERMEDIATE
TOP_K = ctx.TOP_K
EXPERT_BUFFER_COUNT = ctx.EXPERT_BUFFER_COUNT
HEAD_DIM = ctx.HEAD_DIM
NUM_Q_HEAD = ctx.NUM_Q_HEAD
NUM_KV_HEAD = ctx.NUM_KV_HEAD
HEAD_GROUP_SIZE = ctx.HEAD_GROUP_SIZE
QW = ctx.QW
KW = ctx.KW
VW = ctx.VW
num_layers = ctx.num_layers
token_offset, token_pos = 0, ctx.input_token_id_and_pos[0][1]

matTokens = ctx.matTokens
matEmbed = ctx.matEmbed
matHidden = ctx.matHidden
matRMSHidden = ctx.matRMSHidden
attnQs = ctx.attnQs
attnKs = ctx.attnKs
attnVs = ctx.attnVs
attnO = ctx.attnO
matRouterLogits = ctx.matRouterLogits
matRouterTopKIdx = ctx.matRouterTopKIdx
matRouterTopKWeight = ctx.matRouterTopKWeight
matExpertAct = ctx.matExpertAct
matRMSInputW0 = ctx.matRMSInputW0
matRMSInputWLoop = ctx.matRMSInputWLoop
matRMSPostAttnW = ctx.matRMSPostAttnW
matQwenSideInputs = ctx.matQwenSideInputs
matqWs = ctx.matqWs
matkWs = ctx.matkWs
matvWs = ctx.matvWs
matOutWs = ctx.matOutWs
matRouterWs = ctx.matRouterWs
matExpertGateWs = ctx.matExpertGateWs
matExpertUpWs = ctx.matExpertUpWs
matExpertDownWs = ctx.matExpertDownWs
logits_slice = ctx.logits_slice
logits_epoch = ctx.logits_epoch
matLogits = ctx.matLogits
matLogitsW = ctx.matLogitsW
matArgmaxIdx = ctx.matArgmaxIdx
matArgmaxVal = ctx.matArgmaxVal

weightg = dae.add_group("weights", 1)
sysg = dae.add_group("system", 1)
for name in (
    "bar_pre_attn_rms",
    "bar_q_proj",
    "bar_qkv_attn",
    "bar_attn_out",
    "bar_out_mlp",
    "bar_post_attn_rms",
    "bar_router",
    "bar_router_topk",
    "bar_layer",
    "bar_final_rms",
    "bar_logits",
    "bar_argmax_idx",
    "bar_argmax_val",
    "bar_token_finish",
):
    sysg.addBarrier(name)
for buf in range(EXPERT_BUFFER_COUNT):
    sysg.addBarrier(f"bar_scale{buf}")
    sysg.addBarrier(f"bar_down{buf}")

TileM64, _, TileK64 = Gemv_M64N8.MNK
_, _, TileKMma = Gemv_M64N8_MMA_SCALE.MNK

weightg.addTma("loadRMSLayer64", [matRMSHidden], lambda t: t.wgmma_load(N, TileK64 * Gemv_M64N8.n_batch, Major.K))
weightg.addTma("reduceHiddenLayer", [matHidden], lambda t: t.wgmma("reduce", N, TileM64, Major.MN))
weightg.addTma("loadAttnOLayer", [attnO], lambda t: t.wgmma_load(N, TileK64 * Gemv_M64N8.n_batch, Major.K))
weightg.addTma("storeRouterLogits", [matRouterLogits], lambda t: t.wgmma_store(N, TileM64, Major.MN))
weightg.addTma(
    "loadRMSInputWLoop",
    [matRMSInputWLoop],
    lambda t: t.indexed("layer")._build("load", HIDDEN, 1, build_tma_stacked_row, cord_func_stacked_row),
)
weightg.addTma(
    "loadRMSPostAttnW",
    [matRMSPostAttnW],
    lambda t: t.indexed("layer")._build("load", HIDDEN, 1, build_tma_stacked_row, cord_func_stacked_row),
)
weightg.addTma(
    "loadQwenSideInput",
    [matQwenSideInputs],
    lambda t: t.indexed("layer")._build("load", 3 * HEAD_DIM, 1, build_tma_stacked_row, cord_func_stacked_row),
)
weightg.addTma("loadQW", [matqWs], lambda t: t.indexed("layer").wgmma_load(TileM64, TileK64, Major.K))
weightg.addTma("loadKW", [matkWs], lambda t: t.indexed("layer").wgmma_load(TileM64, TileK64, Major.K))
weightg.addTma("loadVW", [matvWs], lambda t: t.indexed("layer").wgmma_load(TileM64, TileK64, Major.K))
weightg.addTma("loadOutWs", [matOutWs], lambda t: t.indexed("layer").wgmma_load(TileM64, TileK64, Major.K))
weightg.addTma("loadRouterWs", [matRouterWs], lambda t: t.indexed("layer").wgmma_load(TileM64, TileK64, Major.K))
weightg.addTma("loadExpertGateWs", [matExpertGateWs], lambda t: t.indexed("layer_expert").wgmma_load(TileM64, TileK64, Major.K))
weightg.addTma("loadExpertUpWs", [matExpertUpWs], lambda t: t.indexed("layer_expert").wgmma_load(TileM64, TileK64, Major.K))
weightg.addTma("loadExpertDownWs", [matExpertDownWs], lambda t: t.indexed("layer_expert").wgmma_load(TileM64, TileKMma, Major.K))
weightg.addTma("storeQ", [attnQs], lambda t: t.wgmma("store", N, TileM64, Major.MN))

tma_builder_MN = partial(build_tma_wgmma_mn, iK=-3)
cord_func_MN = partial(cord_func_MN_major, iK=-3)
tma_builder_K = partial(build_tma_wgmma_k, iN=-3)
cord_func_K = partial(cord_func_K_major, iN=-3)

weightg.addTma("storeK", [attnKs], lambda t: t._build("store", 64, N, tma_store_attn_kv, cord_id))
weightg.addTma("storeV", [attnVs], lambda t: t._build("store", 64, N, tma_store_attn_kv, cord_id))
matQ_attn_view = attnQs.view(N, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)
matK_attn_view = attnKs.view(N, MAX_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)
matV_attn_view = attnVs.view(N, MAX_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM)
matO_attn_view = attnO.view(N, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)
weightg.addTma("loadQ", [matQ_attn_view], lambda t: t._build("load", HEAD_DIM, 64, tma_gqa_load_q, cord_gqa_load_q))
weightg.addTma("loadK", [matK_attn_view], lambda t: t._build("load", HEAD_DIM, KVBlockSize, tma_builder_K, cord_func_K))
weightg.addTma("loadV", [matV_attn_view], lambda t: t._build("load", HEAD_DIM, KVBlockSize, tma_builder_MN, cord_func_MN))
for buf in range(EXPERT_BUFFER_COUNT):
    weightg.addTma(f"storeExpertAct{buf}", [matExpertAct[buf]], lambda t: t.wgmma_store(N, TileM64, Major.MN))
    weightg.addTma(f"loadExpertAct{buf}", [matExpertAct[buf]], lambda t: t.wgmma_load(N, TileKMma, Major.K))
dae.build_groups()

dense_weight = lambda tma: IndexedWeightCordAdapter(tma, (0,))
expert_weight = lambda tma: IndexedWeightCordAdapter(tma, (0, 0))
side_input = ToConvertedCordAdapter(weightg["loadQwenSideInput"], lambda addr: (0, addr * matQwenSideInputs.element_size()))
current_k_store = [
    ToConvertedCordAdapter(
        TmaStore1D(attnKs[req], bytes=HEAD_DIM * attnKs.element_size()),
        lambda addr: (addr * attnKs.element_size(),),
    )
    for req in range(REQ)
]

loadEmbed1D = TmaLoad1D(matEmbed, bytes=HIDDEN * matEmbed.element_size())
loadHidden1D = TmaLoad1D(matHidden, bytes=HIDDEN * matHidden.element_size())
storeHidden1D = TmaStore1D(matHidden, bytes=HIDDEN * matHidden.element_size())
storeRMSHidden1D = TmaStore1D(matRMSHidden, bytes=HIDDEN * matRMSHidden.element_size())

embed_rms = SchedRMSShared(
    num_token=N,
    epsilon=eps,
    hidden_size=HIDDEN,
    tmas=(TmaLoad1D(matRMSInputW0), loadEmbed1D, storeRMSHidden1D),
    embedding=CC0(matTokens[0], token_offset, hidden_size=HIDDEN),
).bar("output", sysg["bar_pre_attn_rms"]).place(rms_sms)
copy_hidden = SchedCopy(
    size=HIDDEN * matHidden.element_size(),
    tmas=wrap_static(loadEmbed1D, storeHidden1D),
    before_copy=CC0(matTokens[0], token_offset, hidden_size=HIDDEN),
).place(N, base_sm=64)

ops = [embed_rms, copy_hidden, IssueBarrier(sysg["bar_pre_attn_rms"])]
late_bar_schedules = [embed_rms, copy_hidden]

for layer_idx in range(num_layers):
    q_proj = SchedGemv(
        Gemv_M64N8,
        MNK=(QW, N, HIDDEN),
        tmas=(dense_weight(weightg["loadQW"]), weightg["loadRMSLayer64"], weightg["storeQ"]),
    ).bar("load", sysg["bar_pre_attn_rms"]).bar("store", sysg["bar_q_proj"]).place(64)
    k_proj = SchedGemv(
        Gemv_M64N8,
        MNK=(KW, N, HIDDEN),
        tmas=(dense_weight(weightg["loadKW"]), weightg["loadRMSLayer64"], ToAttnVStoreCordAdapter(weightg["storeK"], token_pos)),
    ).bar("load", sysg["bar_pre_attn_rms"]).bar("store", sysg["bar_qkv_attn"]).place(16, base_sm=64)
    v_proj = SchedGemv(
        Gemv_M64N8,
        MNK=(VW, N, HIDDEN),
        tmas=(dense_weight(weightg["loadVW"]), weightg["loadRMSLayer64"], ToAttnVStoreCordAdapter(weightg["storeV"], token_pos)),
    ).bar("load", sysg["bar_pre_attn_rms"]).bar("store", sysg["bar_qkv_attn"]).place(16, base_sm=80)
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
    ).bar("load", sysg["bar_attn_out"]).bar("store", sysg["bar_out_mlp"]).place(64)
    post_attn_rms = SchedRMSShared(
        num_token=N,
        epsilon=eps,
        hidden_size=HIDDEN,
        tmas=(dense_weight(weightg["loadRMSPostAttnW"]), loadHidden1D, storeRMSHidden1D),
    ).bar("input", sysg["bar_out_mlp"]).bar("output", sysg["bar_post_attn_rms"]).place(rms_sms)
    router_proj = SchedGemv(
        Gemv_M64N8,
        MNK=(128, N, HIDDEN),
        tmas=(dense_weight(weightg["loadRouterWs"]), weightg["loadRMSLayer64"], weightg["storeRouterLogits"]),
    ).bar("load", sysg["bar_post_attn_rms"]).bar("store", sysg["bar_router"]).place(2)
    router_topk = SchedRouterTopK(
        num_token=N,
        logits_tma=StaticCordAdapter(TmaLoad1D(matRouterLogits)),
        weight_tma=StaticCordAdapter(TmaStore1D(matRouterTopKWeight)),
        idx_addr=StaticCordAdapter(RawAddress(matRouterTopKIdx, 24)),
    ).bar("input", sysg["bar_router"]).bar("output", sysg["bar_router_topk"]).place(1, base_sm=128)

    ops.extend([
        SetLayerIndex(layer_idx),
        q_proj,
        IssueBarrier(sysg["bar_q_proj"]),
        k_proj,
        v_proj,
        IssueBarrier(sysg["bar_qkv_attn"]),
        gqa,
        IssueBarrier(sysg["bar_attn_out"]),
        out_proj,
        IssueBarrier(sysg["bar_out_mlp"]),
        post_attn_rms,
        IssueBarrier(sysg["bar_post_attn_rms"]),
        router_proj,
        IssueBarrier(sysg["bar_router"]),
        router_topk,
        IssueBarrier(sysg["bar_router_topk"]),
    ])
    late_bar_schedules.extend([q_proj, k_proj, v_proj, gqa, out_proj, post_attn_rms, router_proj, router_topk])

    reg_gate, reg_up = 0, 1
    for slot in range(TOP_K):
        buf = slot % EXPERT_BUFFER_COUNT
        expert_input_bar = sysg["bar_post_attn_rms"] if slot < EXPERT_BUFFER_COUNT else sysg[f"bar_down{buf}"]
        gate_proj = SchedGemv(
            Gemv_M64N8,
            MNK=(MOE_INTERMEDIATE, N, HIDDEN),
            tmas=(
                expert_weight(weightg["loadExpertGateWs"]),
                weightg["loadRMSLayer64"],
                RegStore(reg_gate, matExpertAct[buf][:, :TileM64]),
            ),
        ).bar("load", expert_input_bar).place(12)
        up_proj = SchedGemv(
            Gemv_M64N8,
            MNK=(MOE_INTERMEDIATE, N, HIDDEN),
            tmas=(
                expert_weight(weightg["loadExpertUpWs"]),
                weightg["loadRMSLayer64"],
                RegStore(reg_up, matExpertAct[buf][:, :TileM64]),
            ),
        ).bar("load", expert_input_bar).place(12)
        silu = SchedRegSiLUFused(
            num_token=N,
            store_tma=weightg[f"storeExpertAct{buf}"],
            reg_gate=reg_gate,
            reg_up=reg_up,
            base_offset=0,
            stride=TileM64,
        ).bar("output", sysg[f"bar_scale{buf}"]).place(12)
        down_store_bar = sysg["bar_layer"] if slot == TOP_K - 1 else sysg[f"bar_down{buf}"]
        down_proj = SchedGemvMmaScale(
            Gemv_M64N8_MMA_SCALE,
            MNK=(HIDDEN, N, MOE_INTERMEDIATE),
            tmas=(
                expert_weight(weightg["loadExpertDownWs"]),
                weightg[f"loadExpertAct{buf}"],
                StaticCordAdapter(TmaLoad1D(matRouterTopKWeight[slot], bytes=64)),
                weightg["reduceHiddenLayer"],
            ),
        ).bar("load", sysg[f"bar_scale{buf}"]).bar("store", down_store_bar).place(32)
        if slot >= EXPERT_BUFFER_COUNT:
            ops.append(IssueBarrier(sysg[f"bar_down{buf}"]))
        ops.extend([
            LoadExpertIndex(matRouterTopKIdx, slot),
            gate_proj,
            up_proj,
            silu,
            IssueBarrier(sysg[f"bar_scale{buf}"]),
            down_proj,
            IssueBarrier(down_store_bar),
        ])
        late_bar_schedules.extend([silu, down_proj])

    if layer_idx == num_layers - 1:
        final_rms = SchedRMSShared(
            num_token=N,
            epsilon=eps,
            hidden_size=HIDDEN,
            tmas=(dense_weight(weightg["loadRMSInputWLoop"]), loadHidden1D, storeRMSHidden1D),
        ).bar("output", sysg["bar_final_rms"]).place(rms_sms)
        ops.extend([final_rms, IssueBarrier(sysg["bar_final_rms"])])
        late_bar_schedules.append(final_rms)
    else:
        pre_attn_rms = SchedRMSShared(
            num_token=N,
            epsilon=eps,
            hidden_size=HIDDEN,
            tmas=(dense_weight(weightg["loadRMSInputWLoop"]), loadHidden1D, storeRMSHidden1D),
        ).bar("input", sysg["bar_layer"]).bar("output", sysg["bar_pre_attn_rms"]).place(rms_sms)
        ops.extend([pre_attn_rms, IssueBarrier(sysg["bar_pre_attn_rms"])])
        late_bar_schedules.append(pre_attn_rms)

qwen_gemvs = layers_like(GemvLayer, dae, Gemv_M64N8)
logits_proj = []
for i in range(logits_epoch):
    proj = qwen_gemvs(f"logits_proj_{i}", (matLogitsW[i], matRMSHidden, matLogits[i]), reduce=False)
    sched = proj.schedule_(group=False).split_M(6)
    if i == 0:
        sched.bar("load", sysg["bar_final_rms"])
    if i == logits_epoch - 1:
        sched.bar("store", sysg["bar_logits"])
    logits_proj.append(sched.place(full_sms))
argmax = SchedArgmax(
    num_token=N,
    logits_slice=logits_slice,
    num_slice=logits_epoch,
    AtomPartial=ARGMAX_PARTIAL_bf16_1152_50688_132,
    AtomReduce=ARGMAX_REDUCE_bf16_1152_132,
    matLogits=matLogits,
    matOutVal=matArgmaxVal,
    matOutIdx=matArgmaxIdx,
    matFinalOut=matTokens[:, token_offset + 1],
).bar("load", sysg["bar_logits"]).bar("val", sysg["bar_argmax_val"]).bar("idx", sysg["bar_argmax_idx"]).bar("final", sysg["bar_token_finish"]).place(full_sms)

ops.extend([logits_proj, argmax, IssueBarrier(sysg["bar_token_finish"])])
late_bar_schedules.extend([logits_proj, argmax])

for name in ("bar_layer", *(f"bar_down{buf}" for buf in range(EXPERT_BUFFER_COUNT))):
    bar_id = sysg[name]
    if not any(bar_id in sched._bars.values() for sched in late_bar_schedules if hasattr(sched, "_bars")):
        sysg.bindBarrier(name, 0)

seed_prefill_kv_cache(ctx)
matTokens[0, token_offset] = ctx.input_token_id_and_pos[0][0]

dae.bind_late_barrier_counts(*late_bar_schedules)
dae.i(*ops)
dae.s()
dae_app(dae)
if ctx.parsed_args.correctness:
    run_correctness_check(ctx)
