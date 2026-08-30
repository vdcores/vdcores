import argparse
import math
import os
import sys
from functools import partial
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from dae.launcher import *
from dae.model import *
from dae.schedule import *
from dae.util import dae_app
from dae import runtime as dae_runtime
from debug_utils import (
    DEBUG_STAGE_ORDER,
    bind_late_barriers_with_default,
    bind_unused_late_barriers_to_zero,
    print_barrier_counts,
    stage_enabled,
)
from reference import check_tensor_threshold, input_batch1, reference_pass
from transformers import AutoConfig, AutoModelForCausalLM


DEFAULT_MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"
DEFAULT_MAX_SEQ_LEN = 512
DEFAULT_VOCAB_SIZE = 128256


def build_rope_table(max_seq_len, batch, head_dim, rope_theta, positions, device, dtype):
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    if len(positions) > batch:
        raise ValueError("RoPE position lanes exceed the physical decode batch")
    # The schedule addresses this table by absolute token position. Do not
    # pre-slice it by each request's starting position or that offset is applied
    # twice (once here and once by the TMA/raw-address coordinate).
    pos_range = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos_range, inv_freq)
    table = torch.empty(max_seq_len, batch, head_dim, dtype=dtype, device=device)
    table[:, :, 0::2] = freqs.cos().to(dtype=dtype).unsqueeze(1)
    table[:, :, 1::2] = freqs.sin().to(dtype=dtype).unsqueeze(1)
    return table


def permute_rope_weight(weight, head_dim, hidden, num_heads):
    return (
        weight.view(num_heads, 2, head_dim // 2, hidden)
        .transpose(1, 2)
        .reshape_as(weight)
        .contiguous()
    )


def permute_rope_activation(activation, head_dim, num_heads):
    return (
        activation.view(num_heads, 2, head_dim // 2)
        .transpose(1, 2)
        .reshape_as(activation)
        .contiguous()
    )


def apply_interleaved_rope_activation(activation, head_dim, num_heads, rope_row):
    states = activation.view(*activation.shape[:-1], num_heads, head_dim).float()
    cosine = rope_row[0::2].float()
    sine = rope_row[1::2].float()
    even = states[..., 0::2]
    odd = states[..., 1::2]
    rotated = torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine),
        dim=-1,
    ).flatten(-2)
    return rotated.reshape_as(activation).to(dtype=activation.dtype)


def get_rope_theta(config):
    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict) and "rope_theta" in rope_parameters:
        return rope_parameters["rope_theta"]
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is not None:
        return rope_theta
    raise ValueError("Could not determine rope_theta from config")


def detect_runtime_gaps(hidden_size, head_dim):
    gaps = []
    for fn in (
        lambda: select_attention_decode_instruction(head_dim),
        lambda: select_rms_smem_instruction(hidden_size),
        lambda: ensure_cc0_supported_hidden_size(hidden_size),
    ):
        try:
            fn()
        except NotImplementedError as exc:
            gaps.append(str(exc))
    return gaps


def build_synthetic_inputs(config, gpu, dtype, num_layers, hidden, intermediate, qw, kw, vw):
    def randn(*shape):
        return torch.rand(*shape, dtype=dtype, device=gpu) - 0.5

    mat_embed = randn(DEFAULT_VOCAB_SIZE, hidden)
    mat_rms_input_w = [randn(hidden) for _ in range(num_layers)] + [randn(hidden)]
    mat_rms_post_attn_w = [randn(hidden) for _ in range(num_layers)]
    mat_qws = [randn(qw, hidden) for _ in range(num_layers)]
    mat_kws = [randn(kw, hidden) for _ in range(num_layers)]
    mat_vws = [randn(vw, hidden) for _ in range(num_layers)]
    mat_out_ws = [randn(hidden, hidden) for _ in range(num_layers)]
    mat_ups = [randn(intermediate, hidden) for _ in range(num_layers)]
    mat_gates = [randn(intermediate, hidden) for _ in range(num_layers)]
    mat_downs = [randn(hidden, intermediate) for _ in range(num_layers)]
    mat_lm_head = randn(DEFAULT_VOCAB_SIZE, hidden)
    return {
        "embed": mat_embed,
        "rms_input_w": mat_rms_input_w,
        "rms_post_attn_w": mat_rms_post_attn_w,
        "qws": mat_qws,
        "kws": mat_kws,
        "vws": mat_vws,
        "out_ws": mat_out_ws,
        "ups": mat_ups,
        "gates": mat_gates,
        "downs": mat_downs,
        "lm_head": mat_lm_head,
    }
def parse_args():
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument("-N", "--num-generates", type=int, default=16)
    arg_parser.add_argument("--hf-cache-dir", default="/tmp/huggingface_cache")
    arg_parser.add_argument("--correctness", action="store_true")
    arg_parser.add_argument("--dry-build", action="store_true")
    arg_parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    arg_parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    arg_parser.add_argument("--debug-num-layers", type=int, default=None)
    arg_parser.add_argument("--debug-stop-after", choices=DEBUG_STAGE_ORDER, default="full")
    arg_parser.add_argument("--debug-print-barriers", action="store_true")
    parsed_args, remaining_argv = arg_parser.parse_known_args()
    if parsed_args.correctness and not any(arg in ("-l", "--launch", "-b", "--bench") for arg in remaining_argv):
        remaining_argv = [*remaining_argv, "--launch"]
    sys.argv = [sys.argv[0], *remaining_argv]
    return parsed_args


parsed_args = parse_args()

gpu = torch.device("cuda")
REQ, N = 8, 8
KVBlockSize = 64
rms_sms = REQ
num_sms = 128
full_sms = 132
dae = Launcher(full_sms, device=gpu)
# Full-model single-token correctness starts with an empty KV cache.  A decode
# at a nonzero absolute position requires seeding all prior K/V rows first;
# otherwise DAE attends zero-filled history that is absent from the reference.
# Token 791 has only a 0.03125 BF16 top-1 margin in this checkpoint, so valid
# reduction-order noise can flip it.  Use a single-token case with a stable
# 0.9375 reference margin for the end-to-end token check.
input_token_id_and_pos = [(29871, 0)]
num_generates = 0 if (parsed_args.correctness or parsed_args.dry_build) else parsed_args.num_generates - 1

if parsed_args.dry_build:
    config = SimpleNamespace(
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        rms_norm_eps=1e-5,
        rope_parameters={"rope_theta": 500000.0},
    )
    dtype = torch.bfloat16
    model = None
else:
    hf_token = os.environ.get("HF_TOKEN")
    auth_kwargs = {"token": hf_token} if hf_token else {}
    model = AutoModelForCausalLM.from_pretrained(
        parsed_args.model_name,
        cache_dir=parsed_args.hf_cache_dir,
        dtype=torch.bfloat16,
        device_map="auto",
        **auth_kwargs,
    )
    config = AutoConfig.from_pretrained(
        parsed_args.model_name,
        cache_dir=parsed_args.hf_cache_dir,
        **auth_kwargs,
    )
    dtype = model.dtype

eps = config.rms_norm_eps
rope_theta = get_rope_theta(config)
HIDDEN = config.hidden_size
INTERMIDIATE = config.intermediate_size
HEAD_DIM = HIDDEN // config.num_attention_heads
QW = HEAD_DIM * config.num_attention_heads
KW = HEAD_DIM * config.num_key_value_heads
VW = HEAD_DIM * config.num_key_value_heads
MAX_SEQ_LEN = parsed_args.max_seq_len
num_layers = config.num_hidden_layers if parsed_args.dry_build else len(model.model.layers)
if parsed_args.debug_num_layers is not None:
    if parsed_args.debug_num_layers <= 0:
        raise ValueError("--debug-num-layers must be positive")
    num_layers = min(num_layers, parsed_args.debug_num_layers)

runtime_gaps = detect_runtime_gaps(HIDDEN, HEAD_DIM)
if runtime_gaps and not parsed_args.dry_build:
    raise NotImplementedError(
        "The isolated llama3.2-1B path is configured, but these low-level runtime gaps still need discussion:\n- "
        + "\n- ".join(runtime_gaps)
    )

if parsed_args.correctness and (parsed_args.debug_stop_after != "full" or num_layers != config.num_hidden_layers):
    raise ValueError("Single-token correctness requires the full schedule and full layer count")

if parsed_args.dry_build:
    tensors = build_synthetic_inputs(config, gpu, dtype, num_layers, HIDDEN, INTERMIDIATE, QW, KW, VW)
    matEmbed = tensors["embed"]
    matRMSInputW = tensors["rms_input_w"]
    matRMSPostAttnW = tensors["rms_post_attn_w"]
    matqWs = [
        permute_rope_weight(w, HEAD_DIM, HIDDEN, QW // HEAD_DIM)
        for w in tensors["qws"]
    ]
    matkWs = [
        permute_rope_weight(w, HEAD_DIM, HIDDEN, KW // HEAD_DIM)
        for w in tensors["kws"]
    ]
    matvWs = tensors["vws"]
    matOutWs = tensors["out_ws"]
    matUps = tensors["ups"]
    matGates = tensors["gates"]
    matDowns = tensors["downs"]
    matLmHeadW = tensors["lm_head"]
else:
    layers = model.model.layers[:num_layers]
    matEmbed = model.model.embed_tokens.weight
    matRMSInputW = [l.input_layernorm.weight for l in layers] + [model.model.norm.weight]
    matRMSPostAttnW = [l.post_attention_layernorm.weight for l in layers]
    matqWs = [
        permute_rope_weight(l.self_attn.q_proj.weight, HEAD_DIM, HIDDEN, QW // HEAD_DIM)
        for l in layers
    ]
    matkWs = [
        permute_rope_weight(l.self_attn.k_proj.weight, HEAD_DIM, HIDDEN, KW // HEAD_DIM)
        for l in layers
    ]
    matvWs = [l.self_attn.v_proj.weight for l in layers]
    matOutWs = [l.self_attn.o_proj.weight for l in layers]
    matUps = [l.mlp.up_proj.weight for l in layers]
    matGates = [l.mlp.gate_proj.weight for l in layers]
    matDowns = [l.mlp.down_proj.weight for l in layers]
    matLmHeadW = model.lm_head.weight.detach()

matRope = build_rope_table(
    MAX_SEQ_LEN,
    N,
    HEAD_DIM,
    rope_theta,
    [pos for _, pos in input_token_id_and_pos],
    gpu,
    torch.bfloat16,
)
matRopeFused = matRope[:, 0, :].contiguous()
matTokens = torch.zeros(N, MAX_SEQ_LEN, dtype=torch.int64, device=gpu)
matHidden = torch.rand(N, HIDDEN, dtype=dtype, device=gpu) - 0.5
matRMSHidden = torch.rand(N, HIDDEN, dtype=dtype, device=gpu) - 0.5

attnQs = [torch.zeros(REQ, QW, dtype=dtype, device=gpu) for _ in range(num_layers)]
attnKs = [torch.zeros(REQ, MAX_SEQ_LEN, KW, dtype=dtype, device=gpu) for _ in range(num_layers)]
attnVs = [torch.zeros(REQ, MAX_SEQ_LEN, VW, dtype=dtype, device=gpu) for _ in range(num_layers)]
attnO = torch.zeros(REQ, HIDDEN, dtype=dtype, device=gpu)
matSiLUOut = torch.zeros(N, INTERMIDIATE, dtype=dtype, device=gpu)

logits_fold = 8
logits_slice = 8192 * logits_fold
vocab_size = matLmHeadW.shape[0]
logits_epoch = math.ceil(vocab_size / logits_slice)
matLmHeadPadded = torch.zeros(
    logits_slice * logits_epoch, HIDDEN, dtype=dtype, device=gpu
)
matLmHeadPadded[:vocab_size].copy_(matLmHeadW)
matLmHeadW = matLmHeadPadded

matLogits = []
matLogitsW = []
matArgmaxIdx = torch.zeros(N, 128, dtype=torch.long, device=gpu)
matArgmaxVal = torch.zeros(N, 128, dtype=dtype, device=gpu)
matArgmaxOut = torch.zeros(N, dtype=torch.long, device=gpu)

for i in range(logits_epoch):
    matLogitsW.append(matLmHeadW[i * logits_slice : (i + 1) * logits_slice])
    matLogits.append(torch.zeros(N, logits_slice, dtype=dtype, device=gpu))

QKVAtom = Gemv_M64N8IssuerOnly
RopeAtom = Gemv_M64N8_ROPE_128
LinearAtom = Gemv_M64N8IssuerOnly
OutAtom = Gemv_M128N8
QKVTileM, _, QKVTileK = QKVAtom.MNK
LinearTileM, _, LinearTileK = LinearAtom.MNK
OutTileM, _, OutTileK = OutAtom.MNK

matqWs = [pack_weight_tile_major(weight, QKVTileM, QKVTileK) for weight in matqWs]
matkWs = [pack_weight_tile_major(weight, QKVTileM, QKVTileK) for weight in matkWs]
matvWs = [pack_weight_tile_major(weight, QKVTileM, QKVTileK) for weight in matvWs]
matOutWs = [pack_weight_tile_major(weight, OutTileM, OutTileK) for weight in matOutWs]
matUps = [pack_weight_tile_major(weight, QKVTileM, QKVTileK) for weight in matUps]
matGates = [pack_weight_tile_major(weight, QKVTileM, QKVTileK) for weight in matGates]
matDowns = [pack_weight_tile_major(weight, LinearTileM, LinearTileK) for weight in matDowns]

dae.set_persistent(matTokens)
dae.set_streaming(matqWs, matkWs, matvWs, matOutWs, matUps, matGates, matDowns)

layerg = dae.add_group("layer", num_layers)
systemg = dae.add_group("system", 1)

systemg.addBarrier("bar_logits")
systemg.addBarrier("bar_argmax_idx")
systemg.addBarrier("bar_argmax_val")
systemg.addBarrier("bar_token_finish")

layerg.addBarrier("bar_layer")
layerg.addBarrier("bar_out_mlp")
layerg.addBarrier("bar_q_proj")
layerg.addBarrier("bar_qkv_attn")
layerg.addBarrier("bar_attn_out")
layerg.addBarrier("bar_silu_out2")
layerg.addBarrier("bar_pre_attn_rms")
layerg.addBarrier("bar_post_attn_rms")

TileM, _, TileK = Gemv_M64N8.MNK
layerg.addTma("loadRMSLayer", [matRMSHidden] * num_layers, lambda t: t.wgmma_load(N, TileK * Gemv_M64N8.n_batch, Major.K))
layerg.addTma("reduceHiddenLayer", [matHidden] * num_layers, lambda t: t.wgmma("reduce", N, LinearTileM, Major.MN))
layerg.addTma("reduceHiddenOutLayer", [matHidden] * num_layers, lambda t: t.wgmma("reduce", N, OutTileM, Major.MN))
layerg.addTma("loadSiluLayer", [matSiLUOut] * num_layers, lambda t: t.wgmma_load(N, TileK * Gemv_M64N8.n_batch, Major.K))
layerg.addTma("storeSiluLayer", [matSiLUOut] * num_layers, lambda t: t.wgmma_store(N, TileM, Major.MN))
layerg.addTma("loadAttnOLayer", [attnO] * num_layers, lambda t: t.wgmma_load(N, OutTileK * OutAtom.n_batch, Major.K))
layerg.addTma("loadRMSInputW", matRMSInputW[1:], lambda t: t.tensor1d("load", HIDDEN))
layerg.addTma("loadRMSPostAttnW", matRMSPostAttnW, lambda t: t.tensor1d("load", HIDDEN))
layerg.addTma("loadOutWs", matOutWs, lambda t: t.wgmma_load_tiled(OutTileM, OutTileK))
layerg.addTma("loadDown", matDowns, lambda t: t.wgmma_load_tiled(LinearTileM, LinearTileK))
layerg.addTma("loadUp", matUps, lambda t: t.wgmma_load_tiled(QKVTileM, QKVTileK))
layerg.addTma("loadGate", matGates, lambda t: t.wgmma_load_tiled(QKVTileM, QKVTileK))

tma_builder_MN = partial(build_tma_wgmma_mn, iK=-3)
cord_func_MN = partial(cord_func_MN_major, iK=-3)
tma_builder_K = partial(build_tma_wgmma_k, iN=-3)
cord_func_K = partial(cord_func_K_major, iN=-3)

layerg.addTma("loadQW", matqWs, lambda t: t.wgmma_load_tiled(QKVTileM, QKVTileK))
layerg.addTma("loadKW", matkWs, lambda t: t.wgmma_load_tiled(QKVTileM, QKVTileK))
layerg.addTma("loadVW", matvWs, lambda t: t.wgmma_load_tiled(QKVTileM, QKVTileK))
layerg.addTma("storeQ", attnQs, lambda t: t.wgmma("reduce", N, TileM, Major.MN))
layerg.addTma("storeK", attnKs, lambda t: t._build("reduce", 64, N, tma_store_attn_kv, cord_id))
layerg.addTma("storeV", attnVs, lambda t: t._build("reduce", 64, N, tma_store_attn_kv, cord_id))

NUM_KV_HEAD = config.num_key_value_heads
HEAD_GROUP_SIZE = config.num_attention_heads // config.num_key_value_heads
matQ_attn_views = [attnQ.view(N, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM) for attnQ in attnQs]
matK_attn_views = [attnK.view(N, MAX_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM) for attnK in attnKs]
matV_attn_views = [attnV.view(N, MAX_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM) for attnV in attnVs]
matO_attn_view = attnO.view(N, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)

layerg.addTma("loadQ", matQ_attn_views, lambda t: t._build("load", HEAD_DIM, 64, tma_gqa_load_q, cord_gqa_load_q))
layerg.addTma("loadK", matK_attn_views, lambda t: t._build("load", HEAD_DIM, KVBlockSize, tma_builder_K, cord_func_K))
layerg.addTma("loadV", matV_attn_views, lambda t: t._build("load", HEAD_DIM, KVBlockSize, tma_builder_MN, cord_func_MN))

dae.build_groups()


def schedule_single_token(token_offset: int, token_pos: int):
    need_token_restore = (len(input_token_id_and_pos) + num_generates) > 1
    loadEmbed1D = TmaLoad1D(matEmbed, bytes=HIDDEN * 2)
    storeHidden1D = TmaStore1D(matHidden, bytes=HIDDEN * 2)
    loadHidden1D = TmaLoad1D(matHidden, bytes=HIDDEN * 2)
    storeRMSHidden1D = TmaStore1D(matRMSHidden, bytes=HIDDEN * 2)

    embed_rms = SchedRMSShared(
        num_token=N,
        epsilon=eps,
        tmas=(TmaLoad1D(matRMSInputW[0]), loadEmbed1D, storeRMSHidden1D),
        hidden_size=HIDDEN,
        embedding=CC0(matTokens[0], token_offset, hidden_size=HIDDEN),
    ).bar("output", layerg["bar_pre_attn_rms"])

    copy_hidden = SchedCopy(
        size=HIDDEN * matHidden.element_size(),
        tmas=(
            StaticCordAdapter(loadEmbed1D),
            ToLinearCordAdapter(storeHidden1D, HIDDEN * 2),
        ),
        before_copy=CC0(matTokens[0], token_offset, hidden_size=HIDDEN),
    )

    pre_attn_rms = SchedRMSShared(
        num_token=N,
        epsilon=eps,
        hidden_size=HIDDEN,
        tmas=(layerg["loadRMSInputW"].cord(0), loadHidden1D, storeRMSHidden1D),
    ).bar("input", layerg["bar_layer"]).bar("output", layerg.next("bar_pre_attn_rms"))
    post_attn_rms = SchedRMSShared(
        num_token=N,
        epsilon=eps,
        hidden_size=HIDDEN,
        tmas=(layerg["loadRMSPostAttnW"].cord(0), loadHidden1D, storeRMSHidden1D),
    ).bar("input", layerg["bar_out_mlp"]).bar("output", layerg["bar_post_attn_rms"])

    QProj = SchedGemvRope(
        MNK=(QW, N, HIDDEN),
        tmas=(layerg["loadQW"], layerg["loadRMSLayer"], layerg["storeQ"]),
        rope_table=RawAddress(matRopeFused, dae_runtime.config.num_slots),
        hist_seq_len=token_pos,
        Atom=RopeAtom,
        rope_dim=HEAD_DIM,
    ).bar("load", layerg["bar_pre_attn_rms"]).bar("store", layerg["bar_q_proj"])
    QRope = []

    KProj = SchedGemvRope(
        MNK=(KW, N, HIDDEN),
        tmas=(
            layerg["loadKW"],
            layerg["loadRMSLayer"],
            ToAttnVStoreCordAdapter(layerg["storeK"], token_pos),
        ),
        rope_table=RawAddress(matRopeFused, dae_runtime.config.num_slots),
        hist_seq_len=token_pos,
        Atom=RopeAtom,
        rope_dim=HEAD_DIM,
    ).bar("load", layerg["bar_pre_attn_rms"]).bar("store", layerg["bar_qkv_attn"])
    KRope = []
    VProj = SchedGemv(
        QKVAtom,
        MNK=(VW, N, HIDDEN),
        tmas=(
            layerg["loadVW"],
            layerg["loadRMSLayer"],
            ToAttnVStoreCordAdapter(layerg["storeV"], token_pos),
        ),
    ).bar("load", layerg["bar_pre_attn_rms"]).bar("store", layerg["bar_qkv_attn"])

    GemvFactory = layers_like(GemvLayer, dae, Gemv_M64N8)
    Gqa = SchedAttentionDecoding(
        reqs=N,
        seq_len=token_pos + 1,
        KV_BLOCK_SIZE=KVBlockSize,
        NUM_KV_HEADS=NUM_KV_HEAD,
        matO=matO_attn_view,
        tmas=(layerg["loadQ"], layerg["loadK"], layerg["loadV"]),
        num_active_q=4,
    ).bar("o", layerg["bar_attn_out"]).bar("q", layerg["bar_q_proj"]).bar("k", layerg["bar_qkv_attn"])

    OutProj = SchedGemv(
        OutAtom,
        MNK=(HIDDEN, N, HIDDEN),
        tmas=(layerg["loadOutWs"], layerg["loadAttnOLayer"], layerg["reduceHiddenOutLayer"]),
    ).bar("load", layerg["bar_attn_out"]).bar("store", layerg["bar_out_mlp"])

    regGate, regUp = 0, 1
    regStoreGate = RegStore(regGate, size=N * TileM * matSiLUOut.element_size())
    regStoreUp = RegStore(regUp, size=N * TileM * matSiLUOut.element_size())

    gate_proj = SchedGemv(
        LinearAtom,
        MNK=(INTERMIDIATE, N, HIDDEN),
        tmas=(layerg["loadGate"], layerg["loadRMSLayer"], regStoreGate),
    ).bar("load", layerg["bar_post_attn_rms"])
    up_proj = SchedGemv(
        LinearAtom,
        MNK=(INTERMIDIATE, N, HIDDEN),
        tmas=(layerg["loadUp"], layerg["loadRMSLayer"], regStoreUp),
    ).bar("load", layerg["bar_post_attn_rms"])
    silu_fused = SchedRegSiLUFused(
        num_token=N,
        store_tma=layerg["storeSiluLayer"],
        reg_gate=regGate,
        reg_up=regUp,
        base_offset=0,
        stride=TileM,
    ).bar("output", layerg["bar_silu_out2"])
    down_proj = SchedGemv(
        LinearAtom,
        MNK=(HIDDEN, N, INTERMIDIATE),
        tmas=(layerg["loadDown"], layerg["loadSiluLayer"], layerg["reduceHiddenLayer"]),
    ).bar("load", layerg["bar_silu_out2"]).bar("store", layerg["bar_layer"])

    LogitsProj = []
    for i in range(logits_epoch):
        proj = GemvFactory(f"logits_proj_{i}", (matLogitsW[i], matRMSHidden, matLogits[i]), reduce=False)
        sched = proj.schedule_(group=False).split_M(logits_fold)
        if i == 0:
            sched.bar("load", layerg.over("bar_pre_attn_rms"))
            sched[0].no_prefetch()
        if i == logits_epoch - 1:
            sched.bar("store", systemg["bar_logits"])
        LogitsProj.append(sched.place(num_sms))

    Argmax = SchedArgmax(
        num_token=N,
        logits_slice=logits_slice,
        num_slice=logits_epoch,
        AtomPartial=ARGMAX_PARTIAL_bf16_1024_65536_128,
        AtomReduce=ARGMAX_REDUCE_bf16_1024_128,
        matLogits=matLogits,
        matOutVal=matArgmaxVal,
        matOutIdx=matArgmaxIdx,
        matFinalOut=matTokens[:, token_offset + 1],
    ).bar("load", systemg["bar_logits"]).bar("val", systemg["bar_argmax_val"]).bar("idx", systemg["bar_argmax_idx"]).bar("final", systemg["bar_token_finish"])

    sstart, send = systemg.range_bars()
    restore_bars_low = SchedCopy(
        tmas=wrap_static(TmaLoad1D(dae.bars_src[:sstart]), TmaStore1D(dae.bars[:sstart]))
    ).bar("load", layerg.over("bar_pre_attn_rms")).bar("store", systemg["bar_token_finish"])
    restore_bars_high = SchedCopy(
        tmas=wrap_static(TmaLoad1D(dae.bars_src[sstart:send]), TmaStore1D(dae.bars[sstart:send]))
    )

    embed_rms = embed_rms.place(rms_sms)
    copy_hidden = copy_hidden.place(N, base_sm=64)
    pre_attn_rms = pre_attn_rms.place(rms_sms)
    post_attn_rms = post_attn_rms.place(rms_sms)
    QProj = QProj.place(64)
    QRope = []
    KProj = KProj.place(16, base_sm=64)
    KRope = []
    VProj = VProj.place(16, base_sm=80)
    Gqa = Gqa.place(N * NUM_KV_HEAD)
    OutProj = OutProj.place(64)
    gate_proj = gate_proj.place(128)
    up_proj = up_proj.place(128)
    silu_fused = silu_fused.place(128)
    down_proj = down_proj.place(128)
    Argmax = Argmax.place(128)
    restore_bars_low = restore_bars_low.place(1, base_sm=128)
    restore_bars_high = restore_bars_high.place(1, base_sm=128)

    stage_items = [
        ("embed", []),
        ("q_proj", [QProj]),
        ("q_rope", [QRope]),
        ("k_proj", [KProj]),
        ("k_rope", [KRope]),
        ("v_proj", [VProj]),
        ("attn", [Gqa]),
        ("out", [OutProj]),
        ("post_attn_rms", [post_attn_rms]),
        ("gate", [gate_proj]),
        ("up", [up_proj]),
        ("silu", [silu_fused]),
        ("down", [down_proj]),
        ("final_rms", [pre_attn_rms]),
        ("logits", [LogitsProj]),
        ("argmax", [Argmax]),
        ("restore", [restore_bars_low] if need_token_restore else []),
    ]

    active_stage_items = []
    for stage_name, items in stage_items:
        if stage_enabled(parsed_args.debug_stop_after, stage_name):
            active_stage_items.extend(items)

    bound_items = [
        embed_rms,
        copy_hidden,
        restore_bars_high,
        *active_stage_items,
    ]

    bind_late_barriers_with_default(dae, *bound_items, unresolved_count=0)
    bind_unused_late_barriers_to_zero(dae)
    if parsed_args.debug_print_barriers:
        print_barrier_counts(dae)

    if parsed_args.dry_build:
        return

    dae.i(embed_rms, copy_hidden, restore_bars_high)
    dae.i(
        *([QProj] if stage_enabled(parsed_args.debug_stop_after, "q_proj") else []),
        *([QRope] if stage_enabled(parsed_args.debug_stop_after, "q_rope") else []),
        *([KProj] if stage_enabled(parsed_args.debug_stop_after, "k_proj") else []),
        *([KRope] if stage_enabled(parsed_args.debug_stop_after, "k_rope") else []),
        *([VProj] if stage_enabled(parsed_args.debug_stop_after, "v_proj") else []),
        *([Gqa] if stage_enabled(parsed_args.debug_stop_after, "attn") else []),
        *([OutProj] if stage_enabled(parsed_args.debug_stop_after, "out") else []),
        *([post_attn_rms] if stage_enabled(parsed_args.debug_stop_after, "post_attn_rms") else []),
        *([gate_proj] if stage_enabled(parsed_args.debug_stop_after, "gate") else []),
        *([up_proj] if stage_enabled(parsed_args.debug_stop_after, "up") else []),
        *([silu_fused] if stage_enabled(parsed_args.debug_stop_after, "silu") else []),
        *([down_proj] if stage_enabled(parsed_args.debug_stop_after, "down") else []),
        *([pre_attn_rms] if stage_enabled(parsed_args.debug_stop_after, "final_rms") else []),
        *(
            [
                LoopM.toNext(dae.copy_mptrs(), num_layers, resource_group=layerg),
                LoopC.toNext(dae.copy_cptrs(), num_layers),
            ]
            if stage_enabled(parsed_args.debug_stop_after, "final_rms")
            else []
        ),
        *([LogitsProj] if stage_enabled(parsed_args.debug_stop_after, "logits") else []),
        *([Argmax] if stage_enabled(parsed_args.debug_stop_after, "argmax") else []),
        *([restore_bars_low] if stage_enabled(parsed_args.debug_stop_after, "restore") and need_token_restore else []),
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

if parsed_args.dry_build:
    print(
        f"[dry-build] built llama3.2-1B schedule with hidden={HIDDEN}, intermediate={INTERMIDIATE}, "
        f"head_dim={HEAD_DIM}, layers={num_layers}, max_seq_len={MAX_SEQ_LEN}"
    )
    if runtime_gaps:
        print("[dry-build] unresolved runtime gaps:")
        for gap in runtime_gaps:
            print(f"  - {gap}")
    print(f"[dry-build] logits_epoch={logits_epoch}, logits_slice={logits_slice}, vocab_size={vocab_size}")
else:
    print(f"run vdcores with {cur_offset + 1} tokens...")
    if parsed_args.debug_stop_after != "full" or parsed_args.debug_num_layers is not None:
        print(
            f"[debug] stop_after={parsed_args.debug_stop_after}, "
            f"num_layers={num_layers}"
        )
    dae.s()
    dae_app(dae)


def run_correctness_check():
    if parsed_args.dry_build:
        raise RuntimeError("Correctness check is unavailable in --dry-build mode")

    inputs = input_batch1(
        *(e[0] for e in input_token_id_and_pos),
        mat=matTokens[0],
        positions=[e[1] for e in input_token_id_and_pos],
    )
    captured, _ = reference_pass(model, inputs)
    all_ok = True
    decode_pos = input_token_id_and_pos[0][1]
    rope_row = matRope[decode_pos, 0]

    for i in range(min(2, num_layers)):
        layer = captured[i]
        q_ref = apply_interleaved_rope_activation(
            permute_rope_activation(
                layer["q_proj"][0, 0], HEAD_DIM, QW // HEAD_DIM
            ),
            HEAD_DIM,
            QW // HEAD_DIM,
            rope_row,
        )
        k_ref = apply_interleaved_rope_activation(
            permute_rope_activation(
                layer["k_proj"][0, 0], HEAD_DIM, KW // HEAD_DIM
            ),
            HEAD_DIM,
            KW // HEAD_DIM,
            rope_row,
        )
        checks = [
            check_tensor_threshold(
                f"layer{i}.v_proj", layer["v_proj"][0, 0],
                attnVs[i][0, decode_pos], 5.0
            ),
            check_tensor_threshold(f"layer{i}.q_rope", q_ref, attnQs[i][0], 5.0),
            check_tensor_threshold(
                f"layer{i}.k_rope", k_ref, attnKs[i][0, decode_pos], 5.0
            ),
        ]
        all_ok = all_ok and all(passed for passed, _ in checks)

    layer = captured[num_layers - 1]
    silu_ref = F.silu(layer["gate_proj"][0, 0]) * layer["up_proj"][0, 0]
    final_checks = [
        check_tensor_threshold("silu", silu_ref, matSiLUOut[0, :], 5.0),
        check_tensor_threshold("final_hidden", layer["hidden_state_out"][0, 0], matHidden[0], 5.0),
        check_tensor_threshold("final_rms", captured["final"]["final_rms"][0, 0], matRMSHidden[0], 5.0),
        check_tensor_threshold("logits_low", captured["final"]["lm_head"][0, 0, :logits_slice], matLogits[0][0, :logits_slice], 10.0),
    ]
    if logits_epoch > 1:
        final_checks.append(
            check_tensor_threshold(
                "logits_high",
                captured["final"]["lm_head"][0, 0, logits_slice:vocab_size],
                matLogits[1][0, : vocab_size - logits_slice],
                10.0,
            )
        )
    all_ok = all_ok and all(passed for passed, _ in final_checks)

    ref_idx = torch.argmax(captured["final"]["lm_head"], dim=-1)
    dae_idx = matTokens[0, 1].item()
    print(
        f"[correctness] argmax reference={ref_idx[0, 0].item()} "
        f"dae={dae_idx} materialized={torch.argmax(torch.cat(matLogits, dim=1)[0, :vocab_size]).item()}"
    )
    all_ok = all_ok and ref_idx[0, 0].item() == dae_idx
    if not all_ok:
        raise RuntimeError("Correctness check failed")


if parsed_args.correctness:
    run_correctness_check()
