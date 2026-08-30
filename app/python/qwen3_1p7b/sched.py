import os
from functools import partial

import torch
from dae.launcher import *
from dae.model import *
from dae.schedule import *
from dae.tma_utils import (
    StaticCordAdapter,
    ToLinearCordAdapter,
    ToSeqMajorAttnKVLoadCordAdapter,
    ToSeqMajorAttnKVStoreCordAdapter,
    ToSeqMajorCurrentKStoreCordAdapter,
    pack_weight_tile_major,
    tma_store_attn_kv_seq_major,
    wrap_static,
)
from dae.util import dae_app
from cli import parse_args
from correctness import run_correctness_check
from runtime_context import build_runtime_context, seed_prefill_kv_cache
from utils import build_tma_wgmma_k, build_tma_wgmma_mn, cord_func_K_major, cord_func_MN_major


ctx = build_runtime_context(parse_args())

dae = ctx.dae
layers = ctx.layers
BATCH = ctx.BATCH
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


LinearAtom = Gemv_M64N8IssuerOnly
TileM, _, TileK = LinearAtom.MNK
print(f"[weights] packing Qwen3-1.7B projections as M{TileM}K{TileK} tiles")
matqWs = [pack_weight_tile_major(weight.contiguous(), TileM, TileK) for weight in matqWs]
matkWs = [pack_weight_tile_major(weight.contiguous(), TileM, TileK) for weight in matkWs]
matvWs = [pack_weight_tile_major(weight.contiguous(), TileM, TileK) for weight in matvWs]
matOutWs = [pack_weight_tile_major(weight.contiguous(), TileM, TileK) for weight in matOutWs]
matUps = [pack_weight_tile_major(weight.contiguous(), TileM, TileK) for weight in matUps]
matGates = [pack_weight_tile_major(weight.contiguous(), TileM, TileK) for weight in matGates]
matDowns = [pack_weight_tile_major(weight.contiguous(), TileM, TileK) for weight in matDowns]
dae.set_streaming(matqWs, matkWs, matvWs, matOutWs, matUps, matGates, matDowns)


DEBUG_STAGE_ORDER = (
    "final_rms",
    "logits",
    "argmax",
    "restore",
    "full",
)


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def env_prefetch_overrides() -> set[str]:
    raw = os.environ.get("QWEN1P7B_NO_PREFETCH", "")
    return {token.strip() for token in raw.split(",") if token.strip()}


MLP_LOW = 4096
MLP_HIGH = INTERMIDIATE - MLP_LOW
if MLP_HIGH <= 0 or MLP_HIGH % 64:
    raise ValueError(f"Expected intermediate size larger than {MLP_LOW}, got {INTERMIDIATE}")
MLP_HIGH_SMS = MLP_HIGH // 64

PREFETCH_OFF = env_prefetch_overrides()
LONG_CONTEXT = bool(prefill_token_id_and_pos)
QPROJ_SMS = env_int(
    "QWEN1P7B_QPROJ_SMS",
    32 if LONG_CONTEXT else min(128, (QW // TileM) * (HIDDEN // 1024)),
)
KPROJ_SMS = env_int(
    "QWEN1P7B_KPROJ_SMS",
    16 if LONG_CONTEXT else min(64, (KW // TileM) * (HIDDEN // 1024)),
)
VPROJ_SMS = env_int(
    "QWEN1P7B_VPROJ_SMS", min(64, (VW // TileM) * (HIDDEN // 1024))
)
OUTPROJ_SMS = env_int(
    "QWEN1P7B_OUTPROJ_SMS", min(128, (HIDDEN // TileM) * (HIDDEN // 1024))
)
GATE_LOW_SMS = env_int("QWEN1P7B_GATE_LOW_SMS", 64)
UP_LOW_SMS = env_int("QWEN1P7B_UP_LOW_SMS", 64)
SILU_SMS = env_int("QWEN1P7B_SILU_SMS", 4)
DOWN_LOW_SMS = env_int(
    "QWEN1P7B_DOWN_LOW_SMS", min(128, (HIDDEN // TileM) * (MLP_LOW // 1024))
)
DOWN_HIGH_SMS = env_int(
    "QWEN1P7B_DOWN_HIGH_SMS", min(128, (HIDDEN // TileM) * (MLP_HIGH // 1024))
)
LOGITS_SPLIT_M = env_int("QWEN1P7B_LOGITS_SPLIT_M", 6)


def maybe_no_prefetch(name: str, sched):
    if "all" in PREFETCH_OFF or name in PREFETCH_OFF:
        sched.no_prefetch()
    return sched


def maybe_no_prefetch_list(name: str, sched):
    if "all" in PREFETCH_OFF or name in PREFETCH_OFF:
        for item in sched:
            if hasattr(item, "no_prefetch"):
                item.no_prefetch()
    return sched


def stage_enabled(stop_after: str, stage_name: str) -> bool:
    requested_idx = DEBUG_STAGE_ORDER.index(stop_after)
    stage_idx = DEBUG_STAGE_ORDER.index(stage_name)
    return stage_idx <= requested_idx


def bind_unused_late_barriers_to_zero(dae):
    for group in dae.resource_groups.values():
        for name, bar_info in group.bars.items():
            if bar_info["late_bind"] and bar_info["count"] is None:
                group.bindBarrier(name, 0)


def bind_late_barriers_with_default(dae, *insts, unresolved_count=None):
    bar_counts = dae.collect_barrier_release_counts(*insts)
    for group in dae.resource_groups.values():
        for name, bar_info in group.bars.items():
            if not bar_info["late_bind"] or bar_info["count"] is not None:
                continue

            matched_counts = {
                bar_counts[bar_id]
                for bar_id in group.bar_instances.get(name, [])
                if bar_id in bar_counts
            }
            if len(matched_counts) == 1:
                group.bindBarrier(name, matched_counts.pop())
                continue
            if len(matched_counts) == 0 and unresolved_count is not None:
                group.bindBarrier(name, unresolved_count)
                continue
            if len(matched_counts) > 1:
                raise ValueError(
                    f"Barrier {group.name}.{name} observed inconsistent release counts: {sorted(matched_counts)}"
                )
            raise ValueError(f"Could not infer release count for barrier {group.name}.{name}")

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

layerg.addTma("loadRMSLayer", [matRMSHidden] * num_layers, lambda t: t.wgmma_load(N, TileK * LinearAtom.n_batch, Major.K))
layerg.addTma("reduceHiddenLayer", [matHidden] * num_layers, lambda t: t.wgmma("reduce", N, TileM, Major.MN))
layerg.addTma("loadSiluLayer", [matSiLUOut] * num_layers, lambda t: t.wgmma_load(N, TileK * LinearAtom.n_batch, Major.K))
layerg.addTma("storeSiluLayer", [matSiLUOut] * num_layers, lambda t: t.wgmma_store(N, TileM, Major.MN))
layerg.addTma("loadAttnOLayer", [attnO] * num_layers, lambda t: t.wgmma_load(N, TileK * LinearAtom.n_batch, Major.K))
layerg.addTma("storeInterm", [matInterm] * num_layers, lambda t: t.wgmma_store(N, TileM, Major.MN))
layerg.addTma("storeGateOut", [matGateOut] * num_layers, lambda t: t.wgmma_store(N, TileM, Major.MN))
layerg.addTma("loadRMSInputW", matRMSInputW[1:], lambda t: t.tensor1d("load", HIDDEN))
layerg.addTma("loadRMSPostAttnW", matRMSPostAttnW, lambda t: t.tensor1d("load", HIDDEN))
layerg.addTma("loadQwenSideInput", matQwenSideInputs, lambda t: t.tensor1d("load", 3 * HEAD_DIM))
layerg.addTma("loadOutWs", matOutWs, lambda t: t.wgmma_load_tiled(TileM, TileK))
layerg.addTma("loadDown", matDowns, lambda t: t.wgmma_load_tiled(TileM, TileK))
layerg.addTma("loadUp", matUps, lambda t: t.wgmma_load_tiled(TileM, TileK))
layerg.addTma("loadGate", matGates, lambda t: t.wgmma_load_tiled(TileM, TileK))

tma_builder_MN = partial(build_tma_wgmma_mn, iK=-4)
cord_func_MN = partial(cord_func_MN_major, iK=-4)
tma_builder_K = partial(build_tma_wgmma_k, iN=-4)
cord_func_K = partial(cord_func_K_major, iN=-4)

layerg.addTma("loadQW", matqWs, lambda t: t.wgmma_load_tiled(TileM, TileK))
layerg.addTma("loadKW", matkWs, lambda t: t.wgmma_load_tiled(TileM, TileK))
layerg.addTma("loadVW", matvWs, lambda t: t.wgmma_load_tiled(TileM, TileK))
layerg.addTma("storeQ", attnQs, lambda t: t.wgmma("reduce", N, TileM, Major.MN))
layerg.addTma("storeK", attnKs, lambda t: t._build("reduce", 64, N, tma_store_attn_kv_seq_major, cord_id))
layerg.addTma("storeV", attnVs, lambda t: t._build("reduce", 64, N, tma_store_attn_kv_seq_major, cord_id))
layerg.addTma(
    "storeKCurrent",
    attnKs,
    lambda t: t.batched_rowmajor_2d("store", 1, HEAD_DIM),
)
matQ_attn_views = [attnQ.view(N, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM) for attnQ in attnQs]
matK_attn_views = [attnK.view(MAX_SEQ_LEN, N, NUM_KV_HEAD, HEAD_DIM) for attnK in attnKs]
matV_attn_views = [attnV.view(MAX_SEQ_LEN, N, NUM_KV_HEAD, HEAD_DIM) for attnV in attnVs]
matO_attn_view = attnO.view(N, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)

layerg.addTma("loadQ", matQ_attn_views, lambda t: t._build("load", HEAD_DIM, 64, tma_gqa_load_q, cord_gqa_load_q))
layerg.addTma("loadK", matK_attn_views, lambda t: t._build("load", HEAD_DIM, KVBlockSize, tma_builder_K, cord_func_K))
layerg.addTma("loadV", matV_attn_views, lambda t: t._build("load", HEAD_DIM, KVBlockSize, tma_builder_MN, cord_func_MN))

dae.build_groups()


def schedule_single_token(token_offset: int, token_pos: int):
    debug_stop_after = ctx.parsed_args.debug_stop_after
    need_token_restore = (len(input_token_id_and_pos) + num_generates) > 1
    loadEmbed1D = TmaLoad1D(matEmbed, bytes=HIDDEN * 2)
    storeHidden1D = TmaStore1D(matHidden, bytes=HIDDEN * 2)
    loadHidden1D = TmaLoad1D(matHidden, bytes=HIDDEN * 2)
    storeRMSHidden1D = TmaStore1D(matRMSHidden, bytes=HIDDEN * 2)

    embed_rms = SchedRMSShared(
        num_token=BATCH,
        epsilon=eps,
        hidden_size=HIDDEN,
        # The CLI supplies one decode token and broadcasts it to the logical
        # request batch. Ignore the per-SM row offset after CC0 redirects the
        # embedding load; otherwise request r reads token_id + r.
        tmas=(
            TmaLoad1D(matRMSInputW[0]),
            StaticCordAdapter(loadEmbed1D),
            storeRMSHidden1D,
        ),
        embedding=CC0(matTokens[0], token_offset, hidden_size=HIDDEN),
    ).bar("output", layerg["bar_pre_attn_rms"])
    copy_hidden = SchedCopy(
        size=HIDDEN * matHidden.element_size(),
        tmas=(
            StaticCordAdapter(loadEmbed1D),
            ToLinearCordAdapter(storeHidden1D, HIDDEN * matHidden.element_size()),
        ),
        before_copy=CC0(matTokens[0], token_offset, hidden_size=HIDDEN),
    )

    pre_attn_rms = SchedRMSShared(
        num_token=BATCH,
        epsilon=eps,
        hidden_size=HIDDEN,
        tmas=(layerg["loadRMSInputW"].cord(0), loadHidden1D, storeRMSHidden1D),
    ).bar("input", layerg["bar_layer"]).bar("output", layerg.next("bar_pre_attn_rms"))
    post_attn_rms = SchedRMSShared(
        num_token=BATCH,
        epsilon=eps,
        hidden_size=HIDDEN,
        tmas=(layerg["loadRMSPostAttnW"].cord(0), loadHidden1D, storeRMSHidden1D),
    ).bar("input", layerg["bar_out_mlp"]).bar("output", layerg["bar_post_attn_rms"])

    QProj = maybe_no_prefetch("q_proj", SchedGemv(
        LinearAtom,
        MNK=(QW, N, HIDDEN),
        tmas=(layerg["loadQW"], layerg["loadRMSLayer"], layerg["storeQ"]),
    )).bar("load", layerg["bar_pre_attn_rms"]).bar("store", layerg["bar_q_proj"])

    KProj = maybe_no_prefetch("k_proj", SchedGemv(
        LinearAtom,
        MNK=(KW, N, HIDDEN),
        tmas=(
            layerg["loadKW"],
            layerg["loadRMSLayer"],
            ToSeqMajorAttnKVStoreCordAdapter(layerg["storeK"], token_pos),
        ),
    )).bar("load", layerg["bar_pre_attn_rms"]).bar("store", layerg["bar_qkv_attn"])
    VProj = maybe_no_prefetch("v_proj", SchedGemv(
        LinearAtom,
        MNK=(VW, N, HIDDEN),
        tmas=(
            layerg["loadVW"],
            layerg["loadRMSLayer"],
            ToSeqMajorAttnKVStoreCordAdapter(layerg["storeV"], token_pos),
        ),
    )).bar("load", layerg["bar_pre_attn_rms"]).bar("store", layerg["bar_qkv_attn"])
    current_k_store = ToSeqMajorCurrentKStoreCordAdapter(
        layerg["storeKCurrent"],
        token_pos,
        NUM_KV_HEAD,
        HEAD_DIM,
    )

    Gqa = SchedAttentionDecoding(
        reqs=BATCH,
        seq_len=token_pos + 1,
        KV_BLOCK_SIZE=KVBlockSize,
        NUM_KV_HEADS=NUM_KV_HEAD,
        matO=matO_attn_view,
        tmas=(
            layerg["loadQ"],
            ToSeqMajorAttnKVLoadCordAdapter(layerg["loadK"]),
            ToSeqMajorAttnKVLoadCordAdapter(layerg["loadV"]),
        ),
        side_input=layerg["loadQwenSideInput"],
        k_store=current_k_store,
        token_pos=token_pos,
        num_active_q=HEAD_GROUP_SIZE,
    ).bar("q", layerg["bar_q_proj"]).bar("k", layerg["bar_qkv_attn"]).bar("o", layerg["bar_attn_out"])

    OutProj = maybe_no_prefetch("out_proj", SchedGemv(
        LinearAtom,
        MNK=(HIDDEN, N, HIDDEN),
        tmas=(layerg["loadOutWs"], layerg["loadAttnOLayer"], layerg["reduceHiddenLayer"]),
    )).bar("load", layerg["bar_attn_out"]).bar("store", layerg["bar_out_mlp"])

    gate_proj_low = maybe_no_prefetch("gate_low", SchedGemv(
        LinearAtom,
        MNK=(MLP_LOW, N, HIDDEN),
        tmas=(layerg["loadGate"], layerg["loadRMSLayer"], layerg["storeGateOut"]),
    )).bar("load", layerg["bar_post_attn_rms"]).bar("store", layerg["bar_silu_in"])
    reg_gate, reg_up = 0, 1
    reg_store_gate = RegStore(reg_gate, matGateOut[:, 0:TileM])
    reg_store_up = RegStore(reg_up, matInterm[:, 0:TileM])
    gate_proj_high = maybe_no_prefetch("gate_high", SchedGemv(
        LinearAtom,
        MNK=((MLP_LOW, MLP_HIGH), N, HIDDEN),
        tmas=(layerg["loadGate"], layerg["loadRMSLayer"], reg_store_gate),
    ))
    up_proj_low = maybe_no_prefetch("up_low", SchedGemv(
        LinearAtom,
        MNK=(MLP_LOW, N, HIDDEN),
        tmas=(layerg["loadUp"], layerg["loadRMSLayer"], layerg["storeInterm"]),
    )).bar("load", layerg["bar_post_attn_rms"]).bar("store", layerg["bar_silu_in"])
    up_proj_high = maybe_no_prefetch("up_high", SchedGemv(
        LinearAtom,
        MNK=((MLP_LOW, MLP_HIGH), N, HIDDEN),
        tmas=(layerg["loadUp"], layerg["loadRMSLayer"], reg_store_up),
    ))

    silu1 = SchedSmemSiLUInterleaved(
        num_token=N,
        gate_glob=matGateOut[:, :MLP_LOW],
        up_glob=matInterm[:, :MLP_LOW],
        out_glob=matSiLUOut[:, :MLP_LOW],
    ).bar("input", layerg["bar_silu_in"]).bar("output", layerg["bar_silu_out1"])

    silu_high = SchedRegSiLUFused(
        num_token=N,
        store_tma=layerg["storeSiluLayer"],
        reg_gate=reg_gate,
        reg_up=reg_up,
        base_offset=MLP_LOW,
        stride=TileM,
    ).bar("output", layerg["bar_silu_out2"])

    down_proj_low = maybe_no_prefetch("down_low", SchedGemv(
        LinearAtom,
        MNK=(HIDDEN, N, MLP_LOW),
        tmas=(layerg["loadDown"], layerg["loadSiluLayer"], layerg["reduceHiddenLayer"]),
    )).bar("load", layerg["bar_silu_out1"]).bar("store", layerg["bar_layer"])
    down_proj_high = maybe_no_prefetch("down_high", SchedGemv(
        LinearAtom,
        MNK=(HIDDEN, N, (MLP_LOW, MLP_HIGH)),
        tmas=(layerg["loadDown"], layerg["loadSiluLayer"], layerg["reduceHiddenLayer"]),
    )).bar("load", layerg["bar_silu_out2"]).bar("store", layerg["bar_layer"])

    qwen_gemvs = layers_like(GemvLayer, dae, Gemv_M64N8)
    logits_proj = []
    for i in range(logits_epoch):
        proj = qwen_gemvs(f"logits_proj_{i}", (matLogitsW[i], matRMSHidden, matLogits[i]), reduce=False)
        sched = maybe_no_prefetch_list("logits", proj.schedule_(group=False).split_M(LOGITS_SPLIT_M))
        if i == 0:
            sched.bar("load", layerg.over("bar_pre_attn_rms"))
            if "all" not in PREFETCH_OFF and "logits" not in PREFETCH_OFF:
                sched[0].no_prefetch()
        if i == logits_epoch - 1:
            sched.bar("store", systemg["bar_logits"])
        logits_proj.append(sched.place(full_sms))

    argmax = SchedArgmax(
        num_token=BATCH,
        logits_slice=logits_slice,
        num_slice=logits_epoch,
        AtomPartial=ARGMAX_PARTIAL_bf16_1152_50688_132,
        AtomReduce=ARGMAX_REDUCE_bf16_1152_132,
        matLogits=matLogits,
        matOutVal=matArgmaxVal[:BATCH],
        matOutIdx=matArgmaxIdx[:BATCH],
        matFinalOut=matTokens[:BATCH, token_offset + 1],
    ).bar("load", systemg["bar_logits"]).bar("val", systemg["bar_argmax_val"]).bar("idx", systemg["bar_argmax_idx"]).bar("final", systemg["bar_token_finish"])

    restore_bars_low = None
    restore_bars_high = None
    if need_token_restore:
        sstart, send = systemg.range_bars()
        restore_bars_low = SchedCopy(
            tmas=wrap_static(TmaLoad1D(dae.bars_src[:sstart]), TmaStore1D(dae.bars[:sstart])),
        ).bar("load", layerg.over("bar_pre_attn_rms")).bar("store", systemg["bar_token_finish"])
        restore_bars_high = SchedCopy(
            tmas=wrap_static(TmaLoad1D(dae.bars_src[sstart:send]), TmaStore1D(dae.bars[sstart:send])),
        )

    embed_rms = embed_rms.place(rms_sms)
    copy_hidden = copy_hidden.place(BATCH, base_sm=64)
    pre_attn_rms = pre_attn_rms.place(rms_sms)
    post_attn_rms = post_attn_rms.place(rms_sms)
    QProj = QProj.place(QPROJ_SMS)
    KProj = KProj.place(KPROJ_SMS, base_sm=64)
    VProj = VProj.place(VPROJ_SMS, base_sm=96)
    Gqa = Gqa.place(BATCH * NUM_KV_HEAD)
    OutProj = OutProj.place(OUTPROJ_SMS)
    gate_proj_low = gate_proj_low.place(GATE_LOW_SMS)
    gate_proj_high = gate_proj_high.place(MLP_HIGH_SMS)
    up_proj_low = up_proj_low.place(UP_LOW_SMS, base_sm=64)
    up_proj_high = up_proj_high.place(MLP_HIGH_SMS)
    silu1 = silu1.place(SILU_SMS, base_sm=128)
    silu_high = silu_high.place(MLP_HIGH_SMS)
    down_proj_low = down_proj_low.place(DOWN_LOW_SMS)
    down_proj_high = down_proj_high.place(DOWN_HIGH_SMS)
    argmax = argmax.place(full_sms)
    if restore_bars_low is not None:
        restore_bars_low = restore_bars_low.place(1, base_sm=128)
        restore_bars_high = restore_bars_high.place(1, base_sm=128)

    if debug_stop_after == "full":
        dae.bind_late_barrier_counts(
            embed_rms,
            copy_hidden,
            *([restore_bars_high] if restore_bars_high is not None else []),
            QProj,
            KProj,
            VProj,
            Gqa,
            OutProj,
            post_attn_rms,
            gate_proj_low,
            gate_proj_high,
            up_proj_low,
            up_proj_high,
            silu1,
            silu_high,
            down_proj_low,
            down_proj_high,
            pre_attn_rms,
            logits_proj,
            argmax,
            *([restore_bars_low] if restore_bars_low is not None else []),
        )

        dae.i(
            embed_rms,
            copy_hidden,
            *([restore_bars_high] if restore_bars_high is not None else []),
        )

        dae.i(
            QProj,
            KProj,
            VProj,
            Gqa,
            OutProj,
            post_attn_rms,
            gate_proj_low,
            gate_proj_high,
            up_proj_low,
            up_proj_high,
            silu1,
            silu_high,
            down_proj_low,
            down_proj_high,
            pre_attn_rms,
            LoopM.toNext(dae.copy_mptrs(), num_layers, resource_group=layerg),
            LoopC.toNext(dae.copy_cptrs(), num_layers),
            logits_proj,
            argmax,
            *([restore_bars_low] if restore_bars_low is not None else []),
        )
        return

    final_rms_items = [
        QProj,
        KProj,
        VProj,
        Gqa,
        OutProj,
        post_attn_rms,
        gate_proj_low,
        gate_proj_high,
        up_proj_low,
        up_proj_high,
        silu1,
        silu_high,
        down_proj_low,
        down_proj_high,
        pre_attn_rms,
        LoopM.toNext(dae.copy_mptrs(), num_layers, resource_group=layerg),
        LoopC.toNext(dae.copy_cptrs(), num_layers),
    ]
    stage_items = [
        ("final_rms", final_rms_items),
        ("logits", [logits_proj]),
        ("argmax", [argmax]),
        ("restore", [restore_bars_low] if restore_bars_low is not None else []),
    ]
    active_stage_items = []
    for stage_name, items in stage_items:
        if stage_enabled(debug_stop_after, stage_name):
            active_stage_items.extend(items)

    startup_items = [embed_rms, copy_hidden]
    if restore_bars_high is not None and stage_enabled(debug_stop_after, "restore"):
        startup_items.append(restore_bars_high)

    bind_late_barriers_with_default(dae, *startup_items, *active_stage_items, unresolved_count=0)
    bind_unused_late_barriers_to_zero(dae)

    dae.i(*startup_items)
    dae.i(*active_stage_items)


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

dae.s()
if ctx.parsed_args.correctness and (
    ctx.parsed_args.debug_stop_after != "full"
    or ctx.parsed_args.debug_layer_start != 0
    or ctx.parsed_args.debug_num_layers is not None
):
    raise ValueError("Single-token correctness requires the full schedule and full layer count")

if ctx.parsed_args.dry_build:
    print(
        f"[dry-build] built qwen3-1.7b schedule with hidden={HIDDEN}, intermediate={INTERMIDIATE}, "
        f"head_dim={HEAD_DIM}, layers={num_layers}, batch={BATCH}, max_seq_len={MAX_SEQ_LEN}"
    )
    print(f"[dry-build] logits_epoch={logits_epoch}, logits_slice={logits_slice}, vocab_size={vocab_size}")
else:
    print(f"run vdcores with {cur_offset + 1} tokens...")
    if ctx.parsed_args.debug_stop_after != "full" or ctx.parsed_args.debug_num_layers is not None:
        print(
            f"[debug] stop_after={ctx.parsed_args.debug_stop_after}, "
            f"layer_start={ctx.parsed_args.debug_layer_start}, num_layers={num_layers}"
        )
dae_app(dae)
if ctx.parsed_args.correctness:
    run_correctness_check(ctx)
