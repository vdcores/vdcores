from functools import partial

import torch
from dae.launcher import *
from dae.model import *
from dae.schedule import *
from dae.tma_utils import (
    StaticCordAdapter,
    ToAttnVStoreCordAdapter,
)
from dae.util import dae_app
from cli import parse_args
from correctness import run_correctness_check
from runtime_context import build_runtime_context, seed_prefill_kv_cache
from utils import build_tma_wgmma_k, build_tma_wgmma_mn, cord_func_K_major, cord_func_MN_major


ctx = build_runtime_context(parse_args())

dae = ctx.dae
layers = ctx.layers
REQ = ctx.REQ
N = ctx.N
KVBlockSize = ctx.KVBlockSize
rms_sms = ctx.rms_sms
num_sms = ctx.num_sms
full_sms = ctx.full_sms
MAX_SEQ_LEN = ctx.MAX_SEQ_LEN
eps = ctx.eps
HIDDEN = ctx.HIDDEN
INTERMIDIATE = ctx.INTERMIDIATE
HEAD_DIM = ctx.HEAD_DIM
NUM_Q_HEAD = ctx.NUM_Q_HEAD
NUM_KV_HEAD = ctx.NUM_KV_HEAD
HEAD_GROUP_SIZE = ctx.HEAD_GROUP_SIZE
QW = ctx.QW
KW = ctx.KW
VW = ctx.VW
num_layers = ctx.num_layers
prefill_token_id_and_pos = ctx.prefill_token_id_and_pos
input_token_id_and_pos = ctx.input_token_id_and_pos
num_generates = ctx.num_generates
matTokens = ctx.matTokens
matHidden = ctx.matHidden
matRMSHidden = ctx.matRMSHidden
attnQs = ctx.attnQs
attnKs = ctx.attnKs
attnVs = ctx.attnVs
attnO = ctx.attnO
matInterm = ctx.matInterm
matGateOut = ctx.matGateOut
matSiLUOut = ctx.matSiLUOut
matEmbed = ctx.matEmbed
matRMSInputW = ctx.matRMSInputW
matRMSPostAttnW = ctx.matRMSPostAttnW
matQwenSideInputs = ctx.matQwenSideInputs
matqWs = ctx.matqWs
matkWs = ctx.matkWs
matvWs = ctx.matvWs
matOutWs = ctx.matOutWs
matUps = ctx.matUps
matGates = ctx.matGates
matDowns = ctx.matDowns
vocab_size = ctx.vocab_size
logits_slice = ctx.logits_slice
logits_epoch = ctx.logits_epoch
matLogits = ctx.matLogits
matLogitsW = ctx.matLogitsW
matArgmaxIdx = ctx.matArgmaxIdx
matArgmaxVal = ctx.matArgmaxVal

defaultg = dae.get_group()
layerg = dae.add_group("layer", num_layers)
systemg = dae.add_group("system", 1)

defaultg.addBarrier("bar_embedding", N)
systemg.addBarrier("bar_logits")
systemg.addBarrier("bar_argmax_idx")
systemg.addBarrier("bar_argmax_val")
systemg.addBarrier("bar_token_finish")

layerg.addBarrier("bar_layer")
layerg.addBarrier("bar_out_mlp")
layerg.addBarrier("bar_q_proj")
layerg.addBarrier("bar_qkv_attn")
layerg.addBarrier("bar_attn_out")
layerg.addBarrier("bar_rms_layer", 0)
layerg.addBarrier("bar_rms_mlp", 0)
layerg.addBarrier("bar_silu_in")
layerg.addBarrier("bar_silu_out1")
layerg.addBarrier("bar_silu_out2")
layerg.addBarrier("bar_pre_attn_rms")
layerg.addBarrier("bar_post_attn_rms")

TileM, _, TileK = Gemv_M64N8.MNK
layerg.addTma("loadRMSLayer", [matRMSHidden] * num_layers, lambda t: t.wgmma_load(N, TileK * Gemv_M64N8.n_batch, Major.K))
layerg.addTma("reduceHiddenLayer", [matHidden] * num_layers, lambda t: t.wgmma("reduce", N, TileM, Major.MN))
layerg.addTma("loadSiluLayer", [matSiLUOut] * num_layers, lambda t: t.wgmma_load(N, TileK * Gemv_M64N8.n_batch, Major.K))
layerg.addTma("storeSiluLayer", [matSiLUOut] * num_layers, lambda t: t.wgmma_store(N, TileM, Major.MN))
layerg.addTma("loadAttnOLayer", [attnO] * num_layers, lambda t: t.wgmma_load(N, TileK * Gemv_M64N8.n_batch, Major.K))
layerg.addTma("storeInterm", [matInterm] * num_layers, lambda t: t.wgmma_store(N, TileM, Major.MN))
layerg.addTma("storeGateOut", [matGateOut] * num_layers, lambda t: t.wgmma_store(N, TileM, Major.MN))
layerg.addTma("loadRMSInputW", matRMSInputW[1:], lambda t: t.tensor1d("load", HIDDEN))
layerg.addTma("loadRMSPostAttnW", matRMSPostAttnW, lambda t: t.tensor1d("load", HIDDEN))
layerg.addTma("loadQwenSideInput", matQwenSideInputs, lambda t: t.tensor1d("load", 3 * HEAD_DIM))
layerg.addTma("loadOutWs", matOutWs, lambda t: t.wgmma_load(TileM, TileK, Major.K))
layerg.addTma("loadDown", matDowns, lambda t: t.wgmma_load(TileM, TileK, Major.K))
layerg.addTma("loadUp", matUps, lambda t: t.wgmma_load(TileM, TileK, Major.K))
layerg.addTma("loadGate", matGates, lambda t: t.wgmma_load(TileM, TileK, Major.K))

tma_builder_MN = partial(build_tma_wgmma_mn, iK=-3)
cord_func_MN = partial(cord_func_MN_major, iK=-3)
tma_builder_K = partial(build_tma_wgmma_k, iN=-3)
cord_func_K = partial(cord_func_K_major, iN=-3)

TileM, _, TileK = Gemv_M64N8_ROPE_128.MNK
layerg.addTma("loadQW", matqWs, lambda t: t.wgmma_load(TileM, TileK, Major.K))
layerg.addTma("loadKW", matkWs, lambda t: t.wgmma_load(TileM, TileK, Major.K))
layerg.addTma("loadVW", matvWs, lambda t: t.wgmma_load(TileM, TileK, Major.K))
layerg.addTma("storeQ", attnQs, lambda t: t.wgmma("reduce", N, TileM, Major.MN))
layerg.addTma("storeK", attnKs, lambda t: t._build("reduce", 64, N, tma_store_attn_kv, cord_id))
layerg.addTma("storeV", attnVs, lambda t: t._build("reduce", 64, N, tma_store_attn_kv, cord_id))
for req in range(REQ):
    layerg.addTma(f"storeKCurrentReq{req}", [attnK[req] for attnK in attnKs], lambda t: t.tensor1d("store", HEAD_DIM))
matQ_attn_views = [attnQ.view(N, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM) for attnQ in attnQs]
matK_attn_views = [attnK.view(N, MAX_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM) for attnK in attnKs]
matV_attn_views = [attnV.view(N, MAX_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM) for attnV in attnVs]
matO_attn_view = attnO.view(N, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)

layerg.addTma("loadQ", matQ_attn_views, lambda t: t._build("load", HEAD_DIM, 64, tma_gqa_load_q, cord_gqa_load_q))
layerg.addTma("loadK", matK_attn_views, lambda t: t._build("load", HEAD_DIM, KVBlockSize, tma_builder_K, cord_func_K))
layerg.addTma("loadV", matV_attn_views, lambda t: t._build("load", HEAD_DIM, KVBlockSize, tma_builder_MN, cord_func_MN))

dae.build_groups()


def schedule_single_token(token_offset: int, token_pos: int):
    loadEmbed1D = TmaLoad1D(matEmbed, bytes=HIDDEN * 2)
    storeHidden1D = TmaStore1D(matHidden, bytes=HIDDEN * 2)
    loadHidden1D = TmaLoad1D(matHidden, bytes=HIDDEN * 2)
    storeRMSHidden1D = TmaStore1D(matRMSHidden, bytes=HIDDEN * 2)

    embed_rms = SchedRMSShared(
        num_token=N,
        epsilon=eps,
        tmas=(TmaLoad1D(matRMSInputW[0]), loadEmbed1D, storeRMSHidden1D),
        embedding=CC0(matTokens[0], token_offset, hidden_size=HIDDEN),
    ).bar("output", layerg["bar_pre_attn_rms"])
    copy_hidden = SchedCopy(
        size=HIDDEN * matHidden.element_size(),
        tmas=(
            StaticCordAdapter(loadEmbed1D),
            StaticCordAdapter(storeHidden1D),
        ),
        before_copy=CC0(matTokens[0], token_offset, hidden_size=HIDDEN),
    )

    pre_attn_rms = SchedRMSShared(
        num_token=N,
        epsilon=eps,
        tmas=(layerg["loadRMSInputW"].cord(0), loadHidden1D, storeRMSHidden1D),
    ).bar("input", layerg["bar_layer"]).bar("output", layerg.next("bar_pre_attn_rms"))
    post_attn_rms = SchedRMSShared(
        num_token=N,
        epsilon=eps,
        tmas=(layerg["loadRMSPostAttnW"].cord(0), loadHidden1D, storeRMSHidden1D),
    ).bar("input", layerg["bar_out_mlp"]).bar("output", layerg["bar_post_attn_rms"])

    QProj = SchedGemv(
        Gemv_M64N8,
        MNK=(QW, N, HIDDEN),
        tmas=(layerg["loadQW"], layerg["loadRMSLayer"], layerg["storeQ"]),
    ).bar("load", layerg["bar_pre_attn_rms"]).bar("store", layerg["bar_q_proj"])

    KProj = SchedGemv(
        Gemv_M64N8,
        MNK=(KW, N, HIDDEN),
        tmas=(
            layerg["loadKW"],
            layerg["loadRMSLayer"],
            ToAttnVStoreCordAdapter(layerg["storeK"], token_pos),
        ),
    ).bar("store", layerg["bar_qkv_attn"])
    VProj = SchedGemv(
        Gemv_M64N8,
        MNK=(VW, N, HIDDEN),
        tmas=(
            layerg["loadVW"],
            layerg["loadRMSLayer"],
            ToAttnVStoreCordAdapter(layerg["storeV"], token_pos),
        ),
    ).bar("store", layerg["bar_qkv_attn"])
    current_k_store = [layerg[f"storeKCurrentReq{req}"] for req in range(REQ)]

    Gqa = SchedAttentionDecoding(
        reqs=N,
        seq_len=token_pos + 1,
        KV_BLOCK_SIZE=KVBlockSize,
        NUM_KV_HEADS=NUM_KV_HEAD,
        matO=matO_attn_view,
        tmas=(layerg["loadQ"], layerg["loadK"], layerg["loadV"]),
        side_input=layerg["loadQwenSideInput"],
        k_store=current_k_store,
        token_pos=token_pos,
    ).bar("q", layerg["bar_q_proj"]).bar("k", layerg["bar_qkv_attn"]).bar("o", layerg["bar_attn_out"])

    OutProj = SchedGemv(
        Gemv_M64N8,
        MNK=(HIDDEN, N, HIDDEN),
        tmas=(layerg["loadOutWs"], layerg["loadAttnOLayer"], layerg["reduceHiddenLayer"]),
    ).bar("load", layerg["bar_attn_out"]).bar("store", layerg["bar_out_mlp"])

    gate_proj_low = SchedGemv(
        Gemv_M64N8,
        MNK=(4096, N, HIDDEN),
        tmas=(layerg["loadGate"], layerg["loadRMSLayer"], layerg["storeGateOut"]),
    ).bar("load", layerg["bar_post_attn_rms"]).bar("store", layerg["bar_silu_in"])
    up_proj_low = SchedGemv(
        Gemv_M64N8,
        MNK=(4096, N, HIDDEN),
        tmas=(layerg["loadUp"], layerg["loadRMSLayer"], layerg["storeInterm"]),
    ).bar("load", layerg["bar_post_attn_rms"]).bar("store", layerg["bar_silu_in"])

    silu1 = SchedSmemSiLUInterleaved(
        num_token=N,
        gate_glob=matGateOut[:, :4096],
        up_glob=matInterm[:, :4096],
        out_glob=matSiLUOut[:, :4096],
    ).bar("input", layerg["bar_silu_in"]).bar("output", layerg["bar_silu_out1"])

    reg_gate, reg_up = 0, 1
    regStoreGate = RegStore(reg_gate, matGateOut[:, 0:TileM])
    regStoreUp = RegStore(reg_up, matInterm[:, 0:TileM])

    gate_proj_fused = SchedGemv(
        Gemv_M64N8,
        MNK=((4096, 8192), N, HIDDEN),
        tmas=(layerg["loadGate"], layerg["loadRMSLayer"], regStoreGate),
    )
    up_proj_fused = SchedGemv(
        Gemv_M64N8,
        MNK=((4096, 8192), N, HIDDEN),
        tmas=(layerg["loadUp"], layerg["loadRMSLayer"], regStoreUp),
    )
    silu_fused = SchedRegSiLUFused(
        num_token=N,
        store_tma=layerg["storeSiluLayer"],
        reg_gate=reg_gate,
        reg_up=reg_up,
        base_offset=4096,
        stride=TileM,
    ).bar("output", layerg["bar_silu_out2"])

    down_proj_low = SchedGemv(
        Gemv_M64N8,
        MNK=(HIDDEN, N, 4096),
        tmas=(layerg["loadDown"], layerg["loadSiluLayer"], layerg["reduceHiddenLayer"]),
    ).bar("load", layerg["bar_silu_out1"])
    down_proj_high = SchedGemv(
        Gemv_M64N8,
        MNK=(HIDDEN, N, (4096, 8192)),
        tmas=(layerg["loadDown"], layerg["loadSiluLayer"], layerg["reduceHiddenLayer"]),
    ).bar("load", layerg["bar_silu_out2"]).bar("store", layerg["bar_layer"])

    qwen_gemvs = layers_like(GemvLayer, dae, Gemv_M64N8)
    logits_proj = []
    for i in range(logits_epoch):
        proj = qwen_gemvs(f"logits_proj_{i}", (matLogitsW[i], matRMSHidden, matLogits[i]), reduce=False)
        sched = proj.schedule_(group=False).split_M(6)
        if i == 0:
            sched.bar("load", layerg.over("bar_pre_attn_rms"))
            sched[0].no_prefetch()
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

    sstart, send = systemg.range_bars()
    restore_bars_low = SchedCopy(
        tmas=(StaticCordAdapter(TmaLoad1D(dae.bars_src[:sstart])), StaticCordAdapter(TmaStore1D(dae.bars[:sstart]))),
    ).bar("load", layerg.over("bar_pre_attn_rms")).bar("store", systemg["bar_token_finish"])
    restore_bars_high = SchedCopy(
        tmas=(StaticCordAdapter(TmaLoad1D(dae.bars_src[sstart:send])), StaticCordAdapter(TmaStore1D(dae.bars[sstart:send]))),
    )

    embed_rms = embed_rms.place(rms_sms)
    copy_hidden = copy_hidden.place(N, base_sm=64)
    pre_attn_rms = pre_attn_rms.place(rms_sms)
    post_attn_rms = post_attn_rms.place(rms_sms)
    QProj = QProj.place(128)
    KProj = KProj.place(64, base_sm=64)
    VProj = VProj.place(64)
    Gqa = Gqa.place(N * NUM_KV_HEAD)
    OutProj = OutProj.place(num_sms)
    gate_proj_low = gate_proj_low.place(64)
    up_proj_low = up_proj_low.place(64, base_sm=64)
    silu1 = silu1.place(4, base_sm=128)
    gate_proj_fused = gate_proj_fused.place(128)
    up_proj_fused = up_proj_fused.place(128)
    silu_fused = silu_fused.place(128)
    down_proj_low = down_proj_low.place(128)
    down_proj_high = down_proj_high.place(128)
    argmax = argmax.place(full_sms)
    restore_bars_low = restore_bars_low.place(1, base_sm=128)
    restore_bars_high = restore_bars_high.place(1, base_sm=128)

    dae.bind_late_barrier_counts(
        embed_rms,
        copy_hidden,
        restore_bars_high,
        QProj,
        KProj,
        VProj,
        Gqa,
        OutProj,
        post_attn_rms,
        gate_proj_low,
        up_proj_low,
        silu1,
        gate_proj_fused,
        up_proj_fused,
        silu_fused,
        down_proj_low,
        down_proj_high,
        pre_attn_rms,
        logits_proj,
        argmax,
        restore_bars_low,
    )

    dae.i(
        embed_rms,
        copy_hidden,
        restore_bars_high,
    )

    dae.i(
        QProj,
        KProj,
        VProj,
        Gqa,
        OutProj,
        post_attn_rms,
        gate_proj_low,
        up_proj_low,
        silu1,
        gate_proj_fused,
        up_proj_fused,
        silu_fused,
        down_proj_low,
        down_proj_high,
        pre_attn_rms,
        LoopM.toNext(dae.copy_mptrs(), num_layers, resource_group=layerg),
        LoopC.toNext(dae.copy_cptrs(), num_layers),
        logits_proj,
        argmax,
        restore_bars_low,
    )


seed_prefill_kv_cache(ctx)

cur_offset = len(prefill_token_id_and_pos) - 1
cur_pos = prefill_token_id_and_pos[-1][1] if prefill_token_id_and_pos else -1
for token_offset, (token, pos) in enumerate(input_token_id_and_pos, start=len(prefill_token_id_and_pos)):
    matTokens[0, token_offset] = token
    if token_offset > len(prefill_token_id_and_pos):
        dae.i(IssueBarrier(systemg["bar_token_finish"]))
    schedule_single_token(token_offset, pos)
    cur_offset, cur_pos = token_offset, pos

for _ in range(num_generates):
    cur_offset += 1
    cur_pos += 1
    dae.i(IssueBarrier(systemg["bar_token_finish"]))
    schedule_single_token(cur_offset, cur_pos)

print(f"run vdcores with {cur_offset + 1} tokens...")
dae.s()
dae_app(dae)
if ctx.parsed_args.correctness:
    run_correctness_check(ctx)
