from functools import partial

import torch
from dae.launcher import *
from dae.model import *
from dae.schedule import *
from dae.tma_utils import (
    StaticCordAdapter,
    ToSplitMCordAdapter,
    ToSeqMajorAttnKVLoadCordAdapter,
    ToSeqMajorAttnKVStoreCordAdapter,
    ToSeqMajorCurrentKStoreCordAdapter,
    pack_weight_tile_major,
    tma_store_attn_kv_seq_major,
)
from dae.util import dae_app
from cli import DEBUG_STAGE_ORDER, parse_args
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
matZero = torch.zeros(HIDDEN, dtype=matHidden.dtype, device=matHidden.device)
matQSnapshots = (
    [torch.zeros_like(attnQ) for attnQ in attnQs]
    if ctx.parsed_args.correctness
    else None
)

MLP_PREFIX = min(4096, INTERMIDIATE)
MLP_TAIL = INTERMIDIATE - MLP_PREFIX
if MLP_PREFIX not in (2048, 4096) or MLP_TAIL <= 0 or MLP_TAIL % 64:
    raise ValueError(
        "Qwen Blackwell schedule requires a 2048/4096 prefix and a positive "
        f"64-aligned tail, got intermediate_size={INTERMIDIATE}"
    )

LinearAtom = Gemv_M64N8IssuerOnly
TileM, _, TileK = LinearAtom.MNK
print(f"[weights] packing Qwen projections as M{TileM}K{TileK} tiles")
matqWs = [pack_weight_tile_major(weight.contiguous(), TileM, TileK) for weight in matqWs]
matkWs = [pack_weight_tile_major(weight.contiguous(), TileM, TileK) for weight in matkWs]
matvWs = [pack_weight_tile_major(weight.contiguous(), TileM, TileK) for weight in matvWs]
matOutWs = [pack_weight_tile_major(weight.contiguous(), TileM, TileK) for weight in matOutWs]
matUps = [pack_weight_tile_major(weight.contiguous(), TileM, TileK) for weight in matUps]
matGates = [pack_weight_tile_major(weight.contiguous(), TileM, TileK) for weight in matGates]
matDowns = [pack_weight_tile_major(weight.contiguous(), TileM, TileK) for weight in matDowns]
dae.set_streaming(matqWs, matkWs, matvWs, matOutWs, matUps, matGates, matDowns)

# A fold consumes at least four K256 activation tiles.  Keep register-backed
# gate/up tails at fold one because RegStore state is local to each SM.
Q_PROJ_SMS = min(128, (QW // 64) * (HIDDEN // 1024))
KV_PROJ_SMS = min(64, (KW // 64) * (HIDDEN // 1024))
OUT_PROJ_SMS = min(128, (HIDDEN // 64) * (HIDDEN // 1024))
MLP_TAIL_SMS = MLP_TAIL // 64
DOWN_LOW_SMS = min(128, (HIDDEN // 64) * (MLP_PREFIX // 1024))
DOWN_TAIL_SMS = min(128, (HIDDEN // 64) * (MLP_TAIL // 1024))


def stage_enabled(stage_name: str) -> bool:
    return DEBUG_STAGE_ORDER.index(stage_name) <= DEBUG_STAGE_ORDER.index(
        ctx.parsed_args.debug_stop_after
    )


def bind_debug_barrier_counts(*schedules):
    """Bind active schedule counts and zero barriers beyond a debug frontier."""
    counts = dae.collect_barrier_release_counts(*schedules)
    for group in dae.resource_groups.values():
        for name, bar_info in group.bars.items():
            if not bar_info["late_bind"] or bar_info["count"] is not None:
                continue
            matched = {
                counts[bar_id]
                for bar_id in group.bar_instances.get(name, [])
                if bar_id in counts
            }
            if len(matched) > 1:
                raise ValueError(
                    f"Barrier {group.name}.{name} has inconsistent counts: "
                    f"{sorted(matched)}"
                )
            group.bindBarrier(name, matched.pop() if matched else 0)
    dae._late_barriers_bound = True


class SchedClearQ(Schedule):
    """Clear the preceding layer's fold-reduced Q buffer after consumption."""

    def __init__(
        self,
        load_zero,
        store_q,
        tile_bytes: int,
        tile_m: int,
        num_clear_sms: int,
        wait_bar,
    ):
        super().__init__()
        self.load_zero = load_zero
        self.store_q = store_q
        self.tile_bytes = tile_bytes
        self.tile_m = tile_m
        self.num_clear_sms = num_clear_sms
        self.wait_bar = wait_bar

    def schedule(self, sm: int):
        if sm < 0:
            return []
        count = (
            HIDDEN // self.tile_m + self.num_clear_sms - 1 - sm
        ) // self.num_clear_sms
        if count <= 0:
            return []
        store = self.store_q.cord(sm)
        finish = None
        if self._bar("store") is not None:
            store = store.bar(self._bar("store")).group()
            finish = IssueBarrier(self._bar("store")).group()
        return [
            IssueBarrier(self.wait_bar).group(),
            Copy(count, size=self.tile_bytes),
            RepeatM.on(
                count,
                (self.load_zero.cord(0), 0),
                (store, [self.num_clear_sms * self.tile_m, 0]),
            ),
            finish,
        ]

    def bar_release_count(self, role: str):
        if role != "store":
            return 0
        return self._bar_release_if_present(role, HIDDEN // self.tile_m)


class SchedSnapshotQ(Schedule):
    """Correctness-only ordinary-TMA snapshot before a Q buffer is cleared."""

    def __init__(self, load_q, store_q, tile_bytes: int, wait_bar):
        super().__init__()
        self.load_q = load_q
        self.store_q = store_q
        self.tile_bytes = tile_bytes
        self.wait_bar = wait_bar

    def schedule(self, sm: int):
        if sm < 0:
            return []
        return [
            Copy(1, size=self.tile_bytes),
            self.load_q.cord(sm).bar(self.wait_bar).group(),
            self.store_q.cord(sm),
        ]


defaultg = dae.get_group()
layerg = dae.add_group("layer", num_layers)
systemg = dae.add_group("system", 1)

defaultg.addBarrier("bar_embedding", REQ)
systemg.addBarrier("bar_logits")
systemg.addBarrier("bar_argmax_idx")
systemg.addBarrier("bar_argmax_val")
systemg.addBarrier("bar_token_finish")

layerg.addBarrier("bar_layer")
layerg.addBarrier("bar_out_mlp")
layerg.addBarrier("bar_q_proj")
layerg.addBarrier("bar_qkv_attn")
layerg.addBarrier("bar_attn_out")
layerg.addBarrier("bar_q_clear")
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
if matQSnapshots is not None:
    layerg.addTma(
        "loadQSnapshot",
        attnQs,
        lambda t: t.wgmma_load(N, TileM, Major.MN),
    )
    layerg.addTma(
        "storeQSnapshot",
        matQSnapshots,
        lambda t: t.wgmma_store(N, TileM, Major.MN),
    )
q_clear_targets = attnQs[-1:] + attnQs[:-1]
layerg.addTma(
    "storeQClear",
    q_clear_targets,
    lambda t: t.wgmma_store(N, TileM, Major.MN),
)
layerg.addTma("storeK", attnKs, lambda t: t._build("reduce", 64, N, tma_store_attn_kv_seq_major, cord_id))
layerg.addTma("storeV", attnVs, lambda t: t._build("reduce", 64, N, tma_store_attn_kv_seq_major, cord_id))
layerg.addTma("storeKCurrent", attnKs, lambda t: t.batched_rowmajor_2d("store", 1, HEAD_DIM))
matQ_attn_views = [attnQ.view(N, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM) for attnQ in attnQs]
matK_attn_views = [attnK.view(MAX_SEQ_LEN, N, NUM_KV_HEAD, HEAD_DIM) for attnK in attnKs]
matV_attn_views = [attnV.view(MAX_SEQ_LEN, N, NUM_KV_HEAD, HEAD_DIM) for attnV in attnVs]
matO_attn_view = attnO.view(N, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)

layerg.addTma("loadQ", matQ_attn_views, lambda t: t._build("load", HEAD_DIM, 64, tma_gqa_load_q, cord_gqa_load_q))
layerg.addTma("loadK", matK_attn_views, lambda t: t._build("load", HEAD_DIM, KVBlockSize, tma_builder_K, cord_func_K))
layerg.addTma("loadV", matV_attn_views, lambda t: t._build("load", HEAD_DIM, KVBlockSize, tma_builder_MN, cord_func_MN))

dae.build_groups()


def schedule_single_token(token_offset: int, token_pos: int):
    loadHidden1D = TmaLoad1D(matHidden, bytes=HIDDEN * 2)
    storeRMSHidden1D = TmaStore1D(matRMSHidden, bytes=HIDDEN * 2)

    # Each logical request owns its token selector and destination row. GEMV
    # remains physically N=8, but embedding setup must not alias every request
    # onto row zero when sweeping smaller or larger logical batches.
    embed_rms = ListSchedule([
        SchedRMSShared(
            num_token=1,
            epsilon=eps,
            tmas=(
                TmaLoad1D(matRMSInputW[0]),
                StaticCordAdapter(TmaLoad1D(matEmbed, bytes=HIDDEN * 2)),
                TmaStore1D(matRMSHidden[req], bytes=HIDDEN * 2),
            ),
            embedding=CC0(matTokens[req], token_offset, hidden_size=HIDDEN),
        ).bar("output", layerg["bar_pre_attn_rms"]).place(1, base_sm=req)
        for req in range(REQ)
    ])
    copy_hidden = ListSchedule([
        SchedCopy(
            size=HIDDEN * matHidden.element_size(),
            tmas=(
                StaticCordAdapter(TmaLoad1D(matEmbed, bytes=HIDDEN * 2)),
                TmaStore1D(matHidden[req], bytes=HIDDEN * 2),
            ),
            before_copy=CC0(matTokens[req], token_offset, hidden_size=HIDDEN),
        ).place(1, base_sm=64 + req)
        for req in range(REQ)
    ])

    pre_attn_rms = SchedRMSShared(
        num_token=REQ,
        epsilon=eps,
        tmas=(layerg["loadRMSInputW"].cord(0), loadHidden1D, storeRMSHidden1D),
    ).bar("input", layerg["bar_layer"]).bar("output", layerg.next("bar_pre_attn_rms"))
    post_attn_rms = SchedRMSShared(
        num_token=REQ,
        epsilon=eps,
        tmas=(layerg["loadRMSPostAttnW"].cord(0), loadHidden1D, storeRMSHidden1D),
    ).bar("input", layerg["bar_out_mlp"]).bar("output", layerg["bar_post_attn_rms"])

    QProj = SchedGemv(
        LinearAtom,
        MNK=(QW, N, HIDDEN),
        tmas=(layerg["loadQW"], layerg["loadRMSLayer"], layerg["storeQ"]),
    ).bar("load", layerg["bar_pre_attn_rms"]).bar("store", layerg["bar_q_proj"])

    KProj = SchedGemv(
        LinearAtom,
        MNK=(KW, N, HIDDEN),
        tmas=(
            layerg["loadKW"],
            layerg["loadRMSLayer"],
            ToSeqMajorAttnKVStoreCordAdapter(layerg["storeK"], token_pos),
        ),
    ).bar("store", layerg["bar_qkv_attn"])
    VProj = SchedGemv(
        LinearAtom,
        MNK=(VW, N, HIDDEN),
        tmas=(
            layerg["loadVW"],
            layerg["loadRMSLayer"],
            ToSeqMajorAttnKVStoreCordAdapter(layerg["storeV"], token_pos),
        ),
    ).bar("store", layerg["bar_qkv_attn"])
    current_k_store = ToSeqMajorCurrentKStoreCordAdapter(
        layerg["storeKCurrent"], token_pos, NUM_KV_HEAD, HEAD_DIM
    )

    Gqa = SchedAttentionDecoding(
        reqs=REQ,
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

    clear_q = SchedClearQ(
        TmaLoad1D(
            matZero[:N * TileM],
            bytes=N * TileM * matZero.element_size(),
        ),
        ToSplitMCordAdapter(layerg["storeQClear"], 64, TileM),
        N * TileM * matZero.element_size(),
        TileM,
        64,
        layerg["bar_pre_attn_rms"],
    ).bar("store", layerg["bar_q_clear"])
    snapshot_q = []
    if matQSnapshots is not None:
        snapshot_q = [SchedSnapshotQ(
            ToSplitMCordAdapter(layerg["loadQSnapshot"], 64, TileM),
            ToSplitMCordAdapter(layerg["storeQSnapshot"], 64, TileM),
            N * TileM * matZero.element_size(),
            layerg["bar_q_proj"],
        )]

    OutProj = SchedGemv(
        LinearAtom,
        MNK=(HIDDEN, N, HIDDEN),
        tmas=(layerg["loadOutWs"], layerg["loadAttnOLayer"], layerg["reduceHiddenLayer"]),
    ).bar("load", layerg["bar_attn_out"]).bar("store", layerg["bar_out_mlp"])

    gate_proj_low = SchedGemv(
        LinearAtom,
        MNK=(MLP_PREFIX, N, HIDDEN),
        tmas=(layerg["loadGate"], layerg["loadRMSLayer"], layerg["storeGateOut"]),
    ).bar("load", layerg["bar_post_attn_rms"]).bar("store", layerg["bar_silu_in"])
    up_proj_low = SchedGemv(
        LinearAtom,
        MNK=(MLP_PREFIX, N, HIDDEN),
        tmas=(layerg["loadUp"], layerg["loadRMSLayer"], layerg["storeInterm"]),
    ).bar("load", layerg["bar_post_attn_rms"]).bar("store", layerg["bar_silu_in"])

    silu1 = SchedSmemSiLUInterleaved(
        num_token=N,
        gate_glob=matGateOut[:, :MLP_PREFIX],
        up_glob=matInterm[:, :MLP_PREFIX],
        out_glob=matSiLUOut[:, :MLP_PREFIX],
    ).bar("input", layerg["bar_silu_in"]).bar("output", layerg["bar_silu_out1"])

    reg_gate, reg_up = 0, 1
    regStoreGate = RegStore(reg_gate, matGateOut[:, 0:TileM])
    regStoreUp = RegStore(reg_up, matInterm[:, 0:TileM])

    gate_proj_fused = SchedGemv(
        LinearAtom,
        MNK=((MLP_PREFIX, MLP_TAIL), N, HIDDEN),
        tmas=(layerg["loadGate"], layerg["loadRMSLayer"], regStoreGate),
    )
    up_proj_fused = SchedGemv(
        LinearAtom,
        MNK=((MLP_PREFIX, MLP_TAIL), N, HIDDEN),
        tmas=(layerg["loadUp"], layerg["loadRMSLayer"], regStoreUp),
    )
    silu_fused = SchedRegSiLUFused(
        num_token=N,
        store_tma=layerg["storeSiluLayer"],
        reg_gate=reg_gate,
        reg_up=reg_up,
        base_offset=MLP_PREFIX,
        stride=TileM,
    ).bar("output", layerg["bar_silu_out2"])

    down_proj_low = SchedGemv(
        LinearAtom,
        MNK=(HIDDEN, N, MLP_PREFIX),
        tmas=(layerg["loadDown"], layerg["loadSiluLayer"], layerg["reduceHiddenLayer"]),
    ).bar("load", layerg["bar_silu_out1"])
    down_proj_high = SchedGemv(
        LinearAtom,
        MNK=(HIDDEN, N, (MLP_PREFIX, MLP_TAIL)),
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
        num_token=REQ,
        logits_slice=logits_slice,
        num_slice=logits_epoch,
        AtomPartial=ARGMAX_PARTIAL_bf16_1152_50688_132,
        AtomReduce=ARGMAX_REDUCE_bf16_1152_132,
        matLogits=matLogits,
        matOutVal=matArgmaxVal,
        matOutIdx=matArgmaxIdx,
        matFinalOut=matTokens[:REQ, token_offset + 1],
    ).bar("load", systemg["bar_logits"]).bar("val", systemg["bar_argmax_val"]).bar("idx", systemg["bar_argmax_idx"]).bar("final", systemg["bar_token_finish"])

    sstart, send = systemg.range_bars()
    restore_bars_low = SchedCopy(
        tmas=(StaticCordAdapter(TmaLoad1D(dae.bars_src[:sstart])), StaticCordAdapter(TmaStore1D(dae.bars[:sstart]))),
    ).bar("load", layerg.over("bar_pre_attn_rms")).bar("store", systemg["bar_token_finish"])
    restore_bars_high = SchedCopy(
        tmas=(StaticCordAdapter(TmaLoad1D(dae.bars_src[sstart:send])), StaticCordAdapter(TmaStore1D(dae.bars[sstart:send]))),
    )

    pre_attn_rms = pre_attn_rms.place(rms_sms)
    post_attn_rms = post_attn_rms.place(rms_sms)
    QProj = QProj.place(Q_PROJ_SMS)
    if Q_PROJ_SMS == 128:
        KProj = KProj.place(KV_PROJ_SMS, base_sm=64)
        VProj = VProj.place(KV_PROJ_SMS)
    else:
        KProj = KProj.place(KV_PROJ_SMS, base_sm=64)
        VProj = VProj.place(KV_PROJ_SMS, base_sm=64 + KV_PROJ_SMS)
    Gqa = Gqa.place(REQ * NUM_KV_HEAD)
    snapshot_q = [schedule.place(64) for schedule in snapshot_q]
    clear_q = clear_q.place(64, base_sm=64)
    OutProj = OutProj.place(OUT_PROJ_SMS)
    gate_proj_low = gate_proj_low.place(64)
    up_proj_low = up_proj_low.place(64, base_sm=64)
    silu1 = silu1.place(4, base_sm=128)
    gate_proj_fused = gate_proj_fused.place(MLP_TAIL_SMS)
    up_proj_fused = up_proj_fused.place(MLP_TAIL_SMS)
    silu_fused = silu_fused.place(MLP_TAIL_SMS)
    down_proj_low = down_proj_low.place(DOWN_LOW_SMS)
    down_proj_high = down_proj_high.place(DOWN_TAIL_SMS)
    argmax = argmax.place(full_sms)
    restore_bars_low = restore_bars_low.place(1, base_sm=128)
    restore_bars_high = restore_bars_high.place(1, base_sm=128)

    active_schedules = [
        embed_rms,
        copy_hidden,
        restore_bars_high,
        *([QProj, KProj, VProj] if stage_enabled("qkv") else []),
        *([Gqa, *snapshot_q, clear_q] if stage_enabled("attention") else []),
        *([OutProj] if stage_enabled("out") else []),
        *([post_attn_rms] if stage_enabled("post_attn_rms") else []),
        *([gate_proj_low, up_proj_low] if stage_enabled("mlp_prefix") else []),
        *([silu1] if stage_enabled("silu_prefix") else []),
        *(
            [gate_proj_fused, up_proj_fused, silu_fused]
            if stage_enabled("mlp_tail")
            else []
        ),
        *([down_proj_low, down_proj_high] if stage_enabled("down") else []),
        *([pre_attn_rms] if stage_enabled("final_rms") else []),
        *([logits_proj] if stage_enabled("logits") else []),
        *([argmax] if stage_enabled("argmax") else []),
        *([restore_bars_low] if stage_enabled("restore") else []),
    ]
    if ctx.parsed_args.debug_stop_after == "full":
        dae.bind_late_barrier_counts(*active_schedules)
    else:
        bind_debug_barrier_counts(*active_schedules)

    dae.i(
        embed_rms,
        copy_hidden,
        restore_bars_high,
    )

    dae.i(
        *([QProj, KProj, VProj] if stage_enabled("qkv") else []),
        *([Gqa, *snapshot_q, clear_q] if stage_enabled("attention") else []),
        *([OutProj] if stage_enabled("out") else []),
        *([post_attn_rms] if stage_enabled("post_attn_rms") else []),
        *([gate_proj_low, up_proj_low] if stage_enabled("mlp_prefix") else []),
        *([silu1] if stage_enabled("silu_prefix") else []),
        *(
            [gate_proj_fused, up_proj_fused, silu_fused]
            if stage_enabled("mlp_tail")
            else []
        ),
        *([down_proj_low, down_proj_high] if stage_enabled("down") else []),
        *([pre_attn_rms] if stage_enabled("final_rms") else []),
        *(
            [
                LoopM.toNext(dae.copy_mptrs(), num_layers, resource_group=layerg),
                LoopC.toNext(dae.copy_cptrs(), num_layers),
            ]
            if stage_enabled("final_rms")
            else []
        ),
        *([logits_proj] if stage_enabled("logits") else []),
        *([argmax] if stage_enabled("argmax") else []),
        *([restore_bars_low] if stage_enabled("restore") else []),
    )


seed_prefill_kv_cache(ctx)

cur_offset = len(prefill_token_id_and_pos) - 1
cur_pos = prefill_token_id_and_pos[-1][1] if prefill_token_id_and_pos else -1
for token_offset, (token, pos) in enumerate(input_token_id_and_pos, start=len(prefill_token_id_and_pos)):
    matTokens[:REQ, token_offset] = token
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
if ctx.parsed_args.debug_stop_after != "full" or ctx.parsed_args.debug_num_layers:
    print(
        "[debug] "
        f"stop_after={ctx.parsed_args.debug_stop_after}, layers={num_layers}"
    )
dae.s()
dae_app(dae)
if ctx.parsed_args.debug_stop_after != "full":
    for name, tensor in (
        ("q", attnQs[0][:REQ]),
        ("k", attnKs[0][:cur_pos + 1, :REQ].transpose(0, 1)),
        ("v", attnVs[0][:cur_pos + 1, :REQ].transpose(0, 1)),
        ("attn_o", attnO[:REQ]),
        ("hidden", matHidden[:REQ]),
        ("silu", matSiLUOut[:REQ]),
    ):
        values = tensor.float()
        print(
            f"[debug] {name}: finite={bool(torch.isfinite(values).all())} "
            f"abs_sum={values.abs().sum().item():.6f}"
        )
if ctx.parsed_args.correctness:
    run_correctness_check(ctx, matQSnapshots)
