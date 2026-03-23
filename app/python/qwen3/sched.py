import argparse
import math
import os
import sys
from functools import partial

import torch
import torch.nn.functional as F
from dae.launcher import *
from dae.model import *
from dae.schedule import *
from dae.tma_utils import (
    StaticCordAdapter,
    ToAttnVStoreCordAdapter,
)
from dae.util import dae_app
from reference import (
    check_tensor_threshold,
    input_batch1,
    permute_rope_activation,
    reference_pass,
)
from transformers import AutoConfig, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen3-8B"
DEFAULT_MAX_SEQ_LEN = 512
DEFAULT_INPUT_TOKEN = 51


def parse_args():
    raw_argv = sys.argv[1:]
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument("-N", "--num-generates", type=int, default=16)
    arg_parser.add_argument("--hf-cache-dir", default="/tmp/huggingface_cache")
    arg_parser.add_argument("--correctness", action="store_true")
    parsed_args, remaining_argv = arg_parser.parse_known_args()
    num_generates_explicit = any(arg == "-N" or arg.startswith("--num-generates") for arg in raw_argv)
    if not num_generates_explicit and any(arg in ("-b", "--bench") for arg in remaining_argv):
        parsed_args.num_generates = 1
    if parsed_args.correctness and not any(arg in ("-l", "--launch", "-b", "--bench") for arg in remaining_argv):
        remaining_argv = [*remaining_argv, "--launch"]
    sys.argv = [sys.argv[0], *remaining_argv]
    return parsed_args


def get_rope_theta(config):
    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict) and "rope_theta" in rope_parameters:
        return rope_parameters["rope_theta"]
    rope_scaling = getattr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict) and "rope_theta" in rope_scaling:
        return rope_scaling["rope_theta"]
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is not None:
        return rope_theta
    raise ValueError("Could not determine rope theta from config")


def build_interleaved_rope_rows(max_seq_len, head_dim, rope_theta, device, dtype):
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    rope = torch.empty(max_seq_len, head_dim, device=device, dtype=dtype)
    rope[:, 0::2] = freqs.cos().to(dtype=dtype)
    rope[:, 1::2] = freqs.sin().to(dtype=dtype)
    return rope


def permute_rope_weight(weight, num_heads, head_dim, hidden_size):
    return (
        weight.view(num_heads, 2, head_dim // 2, hidden_size)
        .transpose(1, 2)
        .reshape_as(weight)
        .contiguous()
    )


def permute_rope_head_weight(weight):
    head_dim = weight.shape[-1]
    return (
        weight.view(2, head_dim // 2)
        .transpose(0, 1)
        .reshape_as(weight)
        .contiguous()
    )


def apply_rms_affine_rope_heads(hidden_states, weight, rope_row, eps):
    hidden_states = hidden_states.float()
    variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    hidden_states = hidden_states * weight.float().view(1, -1)
    even = hidden_states[..., 0::2]
    odd = hidden_states[..., 1::2]
    cos = rope_row[0::2].float()
    sin = rope_row[1::2].float()
    return torch.stack(
        (even * cos - odd * sin, even * sin + odd * cos),
        dim=-1,
    ).flatten(-2).to(dtype=weight.dtype)


parsed_args = parse_args()

gpu = torch.device("cuda")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir=parsed_args.hf_cache_dir,
    dtype=torch.bfloat16,
    device_map="auto",
    token=os.environ["HF_TOKEN"],
)
config = AutoConfig.from_pretrained(
    MODEL_NAME,
    cache_dir=parsed_args.hf_cache_dir,
    token=os.environ["HF_TOKEN"],
)
layers = model.model.layers

REQ, N = 8, 8
KVBlockSize = 64
rms_sms = REQ
num_sms = 128
full_sms = 132
MAX_SEQ_LEN = min(config.max_position_embeddings, DEFAULT_MAX_SEQ_LEN)
dae = Launcher(full_sms, device=gpu)

input_token_id_and_pos = [(DEFAULT_INPUT_TOKEN, 0)]
num_generates = 0 if parsed_args.correctness else parsed_args.num_generates - 1

dtype = model.dtype
eps = config.rms_norm_eps
rope_theta = get_rope_theta(config)
HIDDEN = config.hidden_size
INTERMIDIATE = config.intermediate_size
HEAD_DIM = getattr(config, "head_dim", HIDDEN // config.num_attention_heads)
NUM_Q_HEAD = config.num_attention_heads
NUM_KV_HEAD = config.num_key_value_heads
HEAD_GROUP_SIZE = NUM_Q_HEAD // NUM_KV_HEAD
QW = HEAD_DIM * NUM_Q_HEAD
KW = HEAD_DIM * NUM_KV_HEAD
VW = HEAD_DIM * NUM_KV_HEAD
num_layers = len(layers)

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

matRope = build_interleaved_rope_rows(MAX_SEQ_LEN, HEAD_DIM, rope_theta, gpu, dtype)
matTokens = torch.zeros(N, MAX_SEQ_LEN, dtype=torch.int64, device=gpu)
matHidden = torch.rand(N, HIDDEN, dtype=dtype, device=gpu) - 0.5
matRMSHidden = torch.rand(N, HIDDEN, dtype=dtype, device=gpu) - 0.5

attnQs = [torch.zeros(REQ, HIDDEN, dtype=dtype, device=gpu) for _ in range(num_layers)]
attnKs = [torch.zeros(REQ, MAX_SEQ_LEN, KW, dtype=dtype, device=gpu) for _ in range(num_layers)]
attnVs = [torch.zeros(REQ, MAX_SEQ_LEN, VW, dtype=dtype, device=gpu) for _ in range(num_layers)]
attnO = torch.zeros(REQ, HIDDEN, dtype=dtype, device=gpu)
matInterm = torch.zeros(N, INTERMIDIATE, dtype=dtype, device=gpu)
matGateOut = torch.zeros(N, INTERMIDIATE, dtype=dtype, device=gpu)
matSiLUOut = torch.zeros(N, INTERMIDIATE, dtype=dtype, device=gpu)

matEmbed = model.model.embed_tokens.weight
matRMSInputW = [l.input_layernorm.weight for l in layers] + [model.model.norm.weight]
matRMSPostAttnW = [l.post_attention_layernorm.weight for l in layers]
matQNormWs = [permute_rope_head_weight(l.self_attn.q_norm.weight.detach()) for l in layers]
matKNormWs = [permute_rope_head_weight(l.self_attn.k_norm.weight.detach()) for l in layers]

matqWs = [
    permute_rope_weight(l.self_attn.q_proj.weight, NUM_Q_HEAD, HEAD_DIM, HIDDEN)
    for l in layers
]
matkWs = [
    permute_rope_weight(l.self_attn.k_proj.weight, NUM_KV_HEAD, HEAD_DIM, HIDDEN)
    for l in layers
]
matvWs = [l.self_attn.v_proj.weight for l in layers]
matOutWs = [l.self_attn.o_proj.weight for l in layers]
matUps = [l.mlp.up_proj.weight for l in layers]
matGates = [l.mlp.gate_proj.weight for l in layers]
matDowns = [l.mlp.down_proj.weight for l in layers]

vocab_size = model.lm_head.weight.shape[0]
logits_slice = 64 * full_sms * 6
logits_epoch = math.ceil(vocab_size / logits_slice)
matLogits = []
matLogitsW = []
matLmHeadW = model.lm_head.weight.detach()
matLmHeadW.resize_(logits_slice * logits_epoch, HIDDEN)
matLmHeadW[vocab_size:, :].zero_()

for i in range(logits_epoch):
    matLogitsW.append(matLmHeadW[i * logits_slice: (i + 1) * logits_slice])
    matLogits.append(torch.zeros(N, logits_slice, dtype=dtype, device=gpu))

matArgmaxIdx = torch.zeros(N, full_sms, dtype=torch.long, device=gpu)
matArgmaxVal = torch.zeros(N, full_sms, dtype=dtype, device=gpu)

dae.set_persistent(matTokens)
dae.set_streaming(matqWs, matkWs, matvWs, matOutWs, matUps, matGates, matDowns)

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
layerg.addTma("loadQNormW", matQNormWs, lambda t: t.tensor1d("load", HEAD_DIM))
layerg.addTma("loadKNormW", matKNormWs, lambda t: t.tensor1d("load", HEAD_DIM))
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
    rope_row_load = StaticCordAdapter(TmaLoad1D(matRope[token_pos], bytes=HEAD_DIM * 2))

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
        q_norm=layerg["loadQNormW"],
        k_norm=layerg["loadKNormW"],
        rope=rope_row_load,
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


cur_offset, cur_pos = 0, 0
for token_offset, (token, pos) in enumerate(input_token_id_and_pos):
    matTokens[0, token_offset] = token
    if token_offset > 0:
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


def run_correctness_check():
    silu_threshold = 10.0
    final_hidden_threshold = 15.0
    final_rms_threshold = 12.0

    print("[correctness] running single-token reference capture...")
    inputs = input_batch1(
        *(token for token, _ in input_token_id_and_pos),
        mat=matTokens[0],
        positions=[pos for _, pos in input_token_id_and_pos],
    )

    captured, output = reference_pass(model, inputs, rope_theta=rope_theta)
    rope_row = matRope[input_token_id_and_pos[0][1]]
    all_ok = True

    for i in range(min(2, num_layers)):
        layer = captured[i]
        dae_q_rope = apply_rms_affine_rope_heads(
            attnQs[i][0].view(NUM_Q_HEAD, HEAD_DIM),
            matQNormWs[i],
            rope_row,
            eps,
        ).reshape(-1)
        print(f"[correctness] Layer {i}:")
        checks = [
            check_tensor_threshold("v_proj", layer["v_proj"][0, 0], attnVs[i][0, 0], 5.0),
            check_tensor_threshold("q_proj_interleaved", layer["q_proj_interleaved"][0, 0], attnQs[i][0], 5.0),
            check_tensor_threshold("q_rope_interleaved", layer["q_rope_interleaved"][0, 0], dae_q_rope, 5.0),
            check_tensor_threshold("k_rope_interleaved", layer["k_rope_interleaved"][0, 0], attnKs[i][0, 0], 5.0),
        ]
        all_ok = all_ok and all(passed for passed, _ in checks)

    print(f"[correctness] Checking Layer {num_layers - 1}:")
    layer = captured[num_layers - 1]
    silu_ref = F.silu(layer["gate_proj"][0, 0]) * layer["up_proj"][0, 0]
    final_checks = [
        check_tensor_threshold("gate_proj_low", layer["gate_proj"][0, 0, :4096], matGateOut[0, :4096], 5.0),
        check_tensor_threshold("up_proj_low", layer["up_proj"][0, 0, :4096], matInterm[0, :4096], 5.0),
        check_tensor_threshold("silu", silu_ref, matSiLUOut[0], silu_threshold),
        check_tensor_threshold("final_hidden", layer["hidden_state_out"][0, 0], matHidden[0], final_hidden_threshold),
        check_tensor_threshold("final_rms", captured["final"]["final_rms"][0, 0], matRMSHidden[0], final_rms_threshold),
    ]

    for i in range(logits_epoch):
        start = i * logits_slice
        end = min((i + 1) * logits_slice, vocab_size)
        final_checks.append(
            check_tensor_threshold(
                f"logits_{i}",
                captured["final"]["lm_head"][0, 0, start:end],
                matLogits[i][0, : end - start],
                10.0,
            )
        )

    all_ok = all_ok and all(passed for passed, _ in final_checks)

    ref_idx = torch.argmax(captured["final"]["lm_head"], dim=-1)
    dae_idx = matTokens[0, 1].item()
    ref_token = ref_idx[0, 0].item()
    token_ok = ref_token == dae_idx
    print(f"[correctness] {'PASS' if token_ok else 'FAIL'} final_token: ref={ref_token}, dae={dae_idx}")
    all_ok = all_ok and token_ok

    if not all_ok:
        raise RuntimeError("Correctness check failed")
    print("[correctness] all checks passed")


if parsed_args.correctness:
    run_correctness_check()
