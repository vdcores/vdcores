from functools import partial

import torch
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
matTokens = ctx.matTokens
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
matEmbed = ctx.matEmbed
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
vocab_size = ctx.vocab_size
logits_slice = ctx.logits_slice
logits_epoch = ctx.logits_epoch
matLogits = ctx.matLogits
matLogitsW = ctx.matLogitsW
matArgmaxIdx = ctx.matArgmaxIdx
matArgmaxVal = ctx.matArgmaxVal

defaultg = dae.get_group()
weightg = dae.add_group("weights", 1)
layerg = dae.add_group("layerbars", num_layers)
systemg = dae.add_group("system", 1)

defaultg.addBarrier("bar_embedding", N)
for name in (
    "bar_layer",
    "bar_out_mlp",
    "bar_q_proj",
    "bar_qkv_attn",
    "bar_attn_out",
    "bar_pre_attn_rms",
    "bar_post_attn_rms",
    "bar_router",
    "bar_router_topk",
):
    layerg.addBarrier(name)
for slot in range(TOP_K):
    layerg.addBarrier(f"bar_scale{slot}")
    layerg.addBarrier(f"bar_down{slot}")
for name in ("bar_final_rms", "bar_logits", "bar_argmax_idx", "bar_argmax_val", "bar_token_finish"):
    systemg.addBarrier(name)

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
for buf in range(EXPERT_BUFFER_COUNT):
    weightg.addTma(f"storeExpertAct{buf}", [matExpertAct[buf]], lambda t: t.wgmma_store(N, TileM64, Major.MN))
    weightg.addTma(f"loadExpertAct{buf}", [matExpertAct[buf]], lambda t: t.wgmma_load(N, TileKMma, Major.K))

dae.build_groups()


def schedule_single_token(token_offset: int, token_pos: int):
    next_pre_attn_bar = layerg.next("bar_pre_attn_rms") if num_layers > 1 else layerg["bar_pre_attn_rms"]
    logits_load_bar = layerg.over("bar_pre_attn_rms") if num_layers > 1 else layerg["bar_pre_attn_rms"]

    dense_weight = lambda tma: IndexedWeightCordAdapter(tma, (0,))
    expert_weight = lambda tma: IndexedWeightCordAdapter(tma, (0, 0))
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
    ).bar("output", layerg["bar_pre_attn_rms"])
    copy_hidden = SchedCopy(
        size=HIDDEN * matHidden.element_size(),
        tmas=wrap_static(loadEmbed1D, storeHidden1D),
        before_copy=CC0(matTokens[0], token_offset, hidden_size=HIDDEN),
    )

    pre_attn_rms = SchedRMSShared(
        num_token=N,
        epsilon=eps,
        hidden_size=HIDDEN,
        tmas=(weightg["loadRMSInputWLoop"], loadHidden1D, storeRMSHidden1D),
    ).bar("input", layerg["bar_layer"]).bar("output", next_pre_attn_bar)
    post_attn_rms = SchedRMSShared(
        num_token=N,
        epsilon=eps,
        hidden_size=HIDDEN,
        tmas=(weightg["loadRMSPostAttnW"], loadHidden1D, storeRMSHidden1D),
    ).bar("input", layerg["bar_out_mlp"]).bar("output", layerg["bar_post_attn_rms"])
    final_rms = None
    if num_layers == 1:
        final_rms = SchedRMSShared(
            num_token=N,
            epsilon=eps,
            hidden_size=HIDDEN,
            tmas=(weightg["loadRMSInputWLoop"], loadHidden1D, storeRMSHidden1D),
        ).bar("output", systemg["bar_final_rms"])

    q_proj = SchedGemv(
        Gemv_M64N8,
        MNK=(QW, N, HIDDEN),
        tmas=(dense_weight(weightg["loadQW"]), weightg["loadRMSLayer64"], weightg["storeQ"]),
    ).bar("load", layerg["bar_pre_attn_rms"]).bar("store", layerg["bar_q_proj"])
    k_proj = SchedGemv(
        Gemv_M64N8,
        MNK=(KW, N, HIDDEN),
        tmas=(dense_weight(weightg["loadKW"]), weightg["loadRMSLayer64"], ToAttnVStoreCordAdapter(weightg["storeK"], token_pos)),
    ).bar("load", layerg["bar_pre_attn_rms"]).bar("store", layerg["bar_qkv_attn"])
    v_proj = SchedGemv(
        Gemv_M64N8,
        MNK=(VW, N, HIDDEN),
        tmas=(dense_weight(weightg["loadVW"]), weightg["loadRMSLayer64"], ToAttnVStoreCordAdapter(weightg["storeV"], token_pos)),
    ).bar("load", layerg["bar_pre_attn_rms"]).bar("store", layerg["bar_qkv_attn"])
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
    ).bar("q", layerg["bar_q_proj"]).bar("k", layerg["bar_qkv_attn"]).bar("o", layerg["bar_attn_out"])
    out_proj = SchedGemv(
        Gemv_M64N8,
        MNK=(HIDDEN, N, QW),
        tmas=(dense_weight(weightg["loadOutWs"]), weightg["loadAttnOLayer"], weightg["reduceHiddenLayer"]),
    ).bar("load", layerg["bar_attn_out"]).bar("store", layerg["bar_out_mlp"])

    router_proj = SchedGemv(
        Gemv_M64N8,
        MNK=(128, N, HIDDEN),
        tmas=(dense_weight(weightg["loadRouterWs"]), weightg["loadRMSLayer64"], weightg["storeRouterLogits"]),
    ).bar("load", layerg["bar_post_attn_rms"]).bar("store", layerg["bar_router"])
    router_topk = SchedRouterTopK(
        num_token=N,
        logits_tma=StaticCordAdapter(TmaLoad1D(matRouterLogits)),
        weight_tma=StaticCordAdapter(TmaStore1D(matRouterTopKWeight)),
        idx_addr=StaticCordAdapter(RawAddress(matRouterTopKIdx, 24)),
    ).bar("input", layerg["bar_router"]).bar("output", layerg["bar_router_topk"])

    expert_scheds = []
    expert_scheds.append(IssueBarrier(layerg["bar_router_topk"]))
    reg_gate, reg_up = 0, 1
    for slot in range(TOP_K):
        buf = slot % EXPERT_BUFFER_COUNT
        gate_proj = SchedGemv(
            Gemv_M64N8,
            MNK=(MOE_INTERMEDIATE, N, HIDDEN),
            tmas=(
                expert_weight(weightg["loadExpertGateWs"]),
                weightg["loadRMSLayer64"],
                RegStore(reg_gate, matExpertAct[buf][:, :TileM64]),
            ),
        ).bar("load", layerg["bar_post_attn_rms"])
        up_proj = SchedGemv(
            Gemv_M64N8,
            MNK=(MOE_INTERMEDIATE, N, HIDDEN),
            tmas=(
                expert_weight(weightg["loadExpertUpWs"]),
                weightg["loadRMSLayer64"],
                RegStore(reg_up, matExpertAct[buf][:, :TileM64]),
            ),
        ).bar("load", layerg["bar_post_attn_rms"])
        silu = SchedRegSiLUFused(
            num_token=N,
            store_tma=weightg[f"storeExpertAct{buf}"],
            reg_gate=reg_gate,
            reg_up=reg_up,
            base_offset=0,
            stride=TileM64,
        ).bar("output", layerg[f"bar_scale{slot}"])
        down_proj = SchedGemvMmaScale(
            Gemv_M64N8_MMA_SCALE,
            MNK=(HIDDEN, N, MOE_INTERMEDIATE),
            tmas=(
                expert_weight(weightg["loadExpertDownWs"]),
                weightg[f"loadExpertAct{buf}"],
                StaticCordAdapter(TmaLoad1D(matRouterTopKWeight[slot], bytes=64)),
                weightg["reduceHiddenLayer"],
            ),
        ).bar("load", layerg[f"bar_scale{slot}"])
        down_store_bar = layerg["bar_layer"] if slot == TOP_K - 1 else layerg[f"bar_down{slot}"]
        down_proj.bar("store", down_store_bar)
        expert_scheds.extend([
            LoadExpertIndex(matRouterTopKIdx, slot),
            gate_proj,
            up_proj,
            silu,
            IssueBarrier(layerg[f"bar_scale{slot}"]),
            down_proj,
            IssueBarrier(down_store_bar),
        ])

    qwen_gemvs = layers_like(GemvLayer, dae, Gemv_M64N8)
    logits_proj = []
    for i in range(logits_epoch):
        proj = qwen_gemvs(f"logits_proj_{i}", (matLogitsW[i], matRMSHidden, matLogits[i]), reduce=False)
        sched = proj.schedule_(group=False).split_M(6)
        if i == 0:
            sched.bar("load", systemg["bar_final_rms"] if num_layers == 1 else logits_load_bar)
        if i == logits_epoch - 1:
            sched.bar("store", systemg["bar_logits"])
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
    ).bar("load", systemg["bar_logits"]).bar("val", systemg["bar_argmax_val"]).bar("idx", systemg["bar_argmax_idx"]).bar("final", systemg["bar_token_finish"])

    embed_rms = embed_rms.place(rms_sms)
    copy_hidden = copy_hidden.place(N, base_sm=64)
    q_proj = q_proj.place(64)
    k_proj = k_proj.place(16, base_sm=64)
    v_proj = v_proj.place(16, base_sm=80)
    gqa = gqa.place(N * NUM_KV_HEAD)
    out_proj = out_proj.place(64)
    post_attn_rms = post_attn_rms.place(rms_sms)
    router_proj = router_proj.place(2)
    router_topk = router_topk.place(1, base_sm=128)
    expert_scheds = [
        sched.place(12) if isinstance(sched, SchedGemv) and sched.MNK[0] == MOE_INTERMEDIATE else
        sched.place(12) if isinstance(sched, SchedRegSiLUFused) else
        sched.place(32) if isinstance(sched, SchedGemvMmaScale) else
        sched
        for sched in expert_scheds
    ]
    pre_attn_rms = pre_attn_rms.place(rms_sms)
    if num_layers == 1:
        final_rms = final_rms.place(rms_sms)
    argmax = argmax.place(full_sms)

    if TOP_K > 0:
        layerg.bindBarrier(f"bar_down{TOP_K - 1}", 0)
    if final_rms is None:
        systemg.bindBarrier("bar_final_rms", 0)

    dae.bind_late_barrier_counts(
        embed_rms,
        copy_hidden,
        q_proj,
        k_proj,
        v_proj,
        gqa,
        out_proj,
        post_attn_rms,
        router_proj,
        router_topk,
        expert_scheds,
        final_rms if final_rms is not None else pre_attn_rms,
        logits_proj,
        argmax,
    )

    if final_rms is not None:
        dae.i(
            embed_rms,
            copy_hidden,
            IssueBarrier(layerg["bar_pre_attn_rms"]),
            SetLayerIndex(0),
            q_proj,
            IssueBarrier(layerg["bar_q_proj"]),
            k_proj,
            v_proj,
            IssueBarrier(layerg["bar_qkv_attn"]),
            gqa,
            IssueBarrier(layerg["bar_attn_out"]),
            out_proj,
            IssueBarrier(layerg["bar_out_mlp"]),
            post_attn_rms,
            IssueBarrier(layerg["bar_post_attn_rms"]),
            router_proj,
            IssueBarrier(layerg["bar_router"]),
            router_topk,
            expert_scheds,
            final_rms,
            IssueBarrier(systemg["bar_final_rms"]),
            logits_proj,
            argmax,
        )
    else:
        dae.i(embed_rms, copy_hidden, SetLayerIndex(0))
        dae.i(
            q_proj,
            IssueBarrier(layerg["bar_q_proj"]),
            k_proj,
            v_proj,
            IssueBarrier(layerg["bar_qkv_attn"]),
            gqa,
            IssueBarrier(layerg["bar_attn_out"]),
            out_proj,
            IssueBarrier(layerg["bar_out_mlp"]),
            post_attn_rms,
            IssueBarrier(layerg["bar_post_attn_rms"]),
            router_proj,
            IssueBarrier(layerg["bar_router"]),
            router_topk,
            expert_scheds,
            pre_attn_rms,
            IssueBarrier(next_pre_attn_bar),
            IncLayerIndex(1),
            LoopM.toNext(dae.copy_mptrs(), num_layers, resource_group=layerg),
            LoopC.toNext(dae.copy_cptrs(), num_layers),
            logits_proj,
            argmax,
        )


seed_prefill_kv_cache(ctx)
token_offset, token_pos = 0, ctx.input_token_id_and_pos[0][1]
matTokens[0, token_offset] = ctx.input_token_id_and_pos[0][0]
schedule_single_token(token_offset, token_pos)

print(f"run vdcores with {token_offset + 1} tokens...")
dae.s()
dae_app(dae)
if ctx.parsed_args.correctness:
    run_correctness_check(ctx)
