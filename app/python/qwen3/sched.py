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
from dae.util import dae_app
from reference import (
    check_tensor_threshold,
    input_batch1,
    mean_relative_diff_pct,
    reference_pass,
)
from transformers import AutoConfig, AutoModelForCausalLM


DEFAULT_MODEL_NAME = "Qwen/Qwen3-8B"
DEFAULT_MAX_SEQ_LEN = 512
ATTN_REQS = 8
DEBUG_STAGE_ORDER = (
    "embed",
    "q_proj",
    "k_proj",
    "v_proj",
    "attn",
    "out",
    "post_attn_rms",
    "gate",
    "up",
    "silu",
    "down",
    "final_rms",
    "logits",
    "argmax",
    "full",
)
def parse_args():
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    arg_parser.add_argument("--hf-cache-dir", default="/tmp/huggingface_cache")
    arg_parser.add_argument("--correctness", action="store_true")
    arg_parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    arg_parser.add_argument("--debug-num-layers", type=int, default=None)
    arg_parser.add_argument("--debug-stop-after", choices=DEBUG_STAGE_ORDER, default="full")
    arg_parser.add_argument("--debug-skip-attn-out-bar", action="store_true")
    arg_parser.add_argument("--debug-disable-attn", action="store_true")
    arg_parser.add_argument("--debug-disable-qkv", action="store_true")
    parsed_args, remaining_argv = arg_parser.parse_known_args()
    if parsed_args.correctness and not any(arg in ("-l", "--launch", "-b", "--bench") for arg in remaining_argv):
        remaining_argv = [*remaining_argv, "--launch"]
    sys.argv = [sys.argv[0], *remaining_argv]
    return parsed_args


def get_rope_theta(config):
    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict) and "rope_theta" in rope_parameters:
        return rope_parameters["rope_theta"]
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is not None:
        return rope_theta
    raise ValueError("Could not determine rope_theta from config")


def build_interleaved_rope_table(max_seq_len, head_dim, rope_theta, start_pos, device, dtype):
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    pos_range = torch.arange(start_pos, start_pos + max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos_range, inv_freq)
    table = torch.empty(max_seq_len, head_dim, dtype=dtype, device=device)
    table[:, 0::2] = freqs.cos().to(dtype=dtype)
    table[:, 1::2] = freqs.sin().to(dtype=dtype)
    return table.contiguous()


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rope_cos_sin(position, head_dim, rope_theta, device):
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    freqs = position * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def apply_qwen_rotate_half_rope(x, position, head_dim, rope_theta):
    cos, sin = rope_cos_sin(position, head_dim, rope_theta, x.device)
    return (x.float() * cos) + (rotate_half(x.float()) * sin)


def apply_interleaved_rope(x, position, head_dim, rope_theta):
    freqs = position * (
        1.0
        / (rope_theta ** (torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim))
    )
    cos = freqs.cos()
    sin = freqs.sin()
    even = x.float()[..., 0::2]
    odd = x.float()[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def plain_rms_per_head(x, eps):
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps)


def build_mlp_chunks(intermediate_size):
    chunks = []
    base = 0
    remaining = intermediate_size
    while remaining > 0:
        size = min(4096, remaining)
        if size % 1024 != 0:
            raise ValueError(
                f"Unsupported Qwen intermediate_size={intermediate_size}: "
                f"chunk tail {size} is not a multiple of 1024 for Gemv_M64N8"
            )
        chunks.append((base, size, size // 64))
        base += size
        remaining -= size
    return tuple(chunks)


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


parsed_args = parse_args()

gpu = torch.device("cuda")
REQ, N = ATTN_REQS, 8
KVBlockSize = 64
rms_sms = N
num_sms = 128
full_sms = 132
dae = Launcher(full_sms, device=gpu)

input_token_id_and_pos = [(791, 0)]
token_offset = 0
token_id, token_pos = input_token_id_and_pos[0]

model = AutoModelForCausalLM.from_pretrained(
    parsed_args.model_name,
    cache_dir=parsed_args.hf_cache_dir,
    dtype=torch.bfloat16,
    device_map="auto",
    token=os.environ["HF_TOKEN"],
)
config = AutoConfig.from_pretrained(
    parsed_args.model_name,
    cache_dir=parsed_args.hf_cache_dir,
    token=os.environ["HF_TOKEN"],
)

dtype = model.dtype
eps = config.rms_norm_eps
rope_theta = get_rope_theta(config)
HIDDEN = config.hidden_size
INTERMIDIATE = config.intermediate_size
HEAD_DIM = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
QW = HEAD_DIM * config.num_attention_heads
KW = HEAD_DIM * config.num_key_value_heads
VW = HEAD_DIM * config.num_key_value_heads
MAX_SEQ_LEN = parsed_args.max_seq_len
NUM_KV_HEAD = config.num_key_value_heads
HEAD_GROUP_SIZE = config.num_attention_heads // config.num_key_value_heads
MLP_CHUNKS = build_mlp_chunks(INTERMIDIATE)

layers = model.model.layers
num_layers = len(layers)
if parsed_args.debug_num_layers is not None:
    if parsed_args.debug_num_layers <= 0:
        raise ValueError("--debug-num-layers must be positive")
    num_layers = min(num_layers, parsed_args.debug_num_layers)
    layers = layers[:num_layers]

if parsed_args.correctness and num_layers != len(model.model.layers):
    raise ValueError("Single-token correctness requires the full Qwen3 layer stack")

assert QW == HIDDEN, "Q projection must map to hidden size"

defaultg = dae.get_group()
layerg = dae.add_group("layer", num_layers)
systemg = dae.add_group("system", 1)

systemg.addBarrier("bar_logits")
systemg.addBarrier("bar_argmax_idx")
systemg.addBarrier("bar_argmax_val")

layerg.addBarrier("bar_layer")
layerg.addBarrier("bar_out_mlp")
layerg.addBarrier("bar_q_proj")
layerg.addBarrier("bar_qkv_attn")
layerg.addBarrier("bar_attn_out")
layerg.addBarrier("bar_pre_attn_rms")
layerg.addBarrier("bar_post_attn_rms")
layerg.addBarrier("bar_silu_in")
layerg.addBarrier("bar_silu_out")

matRope = build_interleaved_rope_table(
    MAX_SEQ_LEN,
    HEAD_DIM,
    rope_theta,
    token_pos,
    gpu,
    torch.bfloat16,
)
matTokens = torch.zeros(N, MAX_SEQ_LEN, dtype=torch.int64, device=gpu)
matHidden = torch.rand(N, HIDDEN, dtype=dtype, device=gpu) - 0.5
matRMSHidden = torch.rand(N, HIDDEN, dtype=dtype, device=gpu) - 0.5

attnQs = [torch.zeros(N, QW, dtype=dtype, device=gpu) for _ in range(num_layers)]
attnKs = [torch.zeros(N, MAX_SEQ_LEN, KW, dtype=dtype, device=gpu) for _ in range(num_layers)]
attnVs = [torch.zeros(N, MAX_SEQ_LEN, VW, dtype=dtype, device=gpu) for _ in range(num_layers)]
attnO = torch.zeros(N, HIDDEN, dtype=dtype, device=gpu)
matInterm = torch.zeros(N, INTERMIDIATE, dtype=dtype, device=gpu)
matGateOut = torch.zeros(N, INTERMIDIATE, dtype=dtype, device=gpu)
matSiLUOut = torch.zeros(N, INTERMIDIATE, dtype=dtype, device=gpu)

matEmbed = model.model.embed_tokens.weight
matRMSInputW = [l.input_layernorm.weight for l in layers] + [model.model.norm.weight]
matRMSPostAttnW = [l.post_attention_layernorm.weight for l in layers]

matqWs = [l.self_attn.q_proj.weight for l in layers]
matkWs = [l.self_attn.k_proj.weight for l in layers]
matvWs = [l.self_attn.v_proj.weight for l in layers]
matOutWs = [l.self_attn.o_proj.weight for l in layers]
matUps = [l.mlp.up_proj.weight for l in layers]
matGates = [l.mlp.gate_proj.weight for l in layers]
matDowns = [l.mlp.down_proj.weight for l in layers]
matQNormWs = [l.self_attn.q_norm.weight for l in layers]
matKNormWs = [l.self_attn.k_norm.weight for l in layers]

logits_slice = 64 * full_sms * 6
vocab_size = model.lm_head.weight.shape[0]
logits_epoch = math.ceil(vocab_size / logits_slice)
matLogits = []
matLogitsW = []
matLmHeadW = model.lm_head.weight.detach()
matLmHeadW.resize_(logits_slice * logits_epoch, HIDDEN)
matLmHeadW[vocab_size:, :].zero_()

matArgmaxIdx = torch.zeros(N, full_sms, dtype=torch.long, device=gpu)
matArgmaxVal = torch.zeros(N, full_sms, dtype=dtype, device=gpu)

for i in range(logits_epoch):
    matLogitsW.append(matLmHeadW[i * logits_slice : (i + 1) * logits_slice])
    matLogits.append(torch.zeros(N, logits_slice, dtype=dtype, device=gpu))

dae.set_persistent(matTokens)
dae.set_streaming(matqWs, matkWs, matvWs, matOutWs, matUps, matGates, matDowns)

TileM, _, TileK = Gemv_M64N8.MNK
layerg.addTma("loadRMSLayer", [matRMSHidden] * num_layers, lambda t: t.wgmma_load(N, TileK * Gemv_M64N8.n_batch, Major.K))
layerg.addTma("reduceHiddenLayer", [matHidden] * num_layers, lambda t: t.wgmma("reduce", N, TileM, Major.MN))
layerg.addTma("loadSiluLayer", [matSiLUOut] * num_layers, lambda t: t.wgmma_load(N, TileK * Gemv_M64N8.n_batch, Major.K))
layerg.addTma("loadAttnOLayer", [attnO] * num_layers, lambda t: t.wgmma_load(N, TileK * Gemv_M64N8.n_batch, Major.K))
layerg.addTma("storeInterm", [matInterm] * num_layers, lambda t: t.wgmma_store(N, TileM, Major.MN))
layerg.addTma("storeGateOut", [matGateOut] * num_layers, lambda t: t.wgmma_store(N, TileM, Major.MN))

layerg.addTma("loadRMSInputW", matRMSInputW[1:], lambda t: t.tensor1d("load", HIDDEN))
layerg.addTma("loadRMSPostAttnW", matRMSPostAttnW, lambda t: t.tensor1d("load", HIDDEN))
layerg.addTma("loadOutWs", matOutWs, lambda t: t.wgmma_load(TileM, TileK, Major.K))
layerg.addTma("loadDown", matDowns, lambda t: t.wgmma_load(TileM, TileK, Major.K))
layerg.addTma("loadUp", matUps, lambda t: t.wgmma_load(TileM, TileK, Major.K))
layerg.addTma("loadGate", matGates, lambda t: t.wgmma_load(TileM, TileK, Major.K))

tma_builder_MN = partial(build_tma_wgmma_mn, iK=-3)
cord_func_MN = partial(cord_func_MN_major, iK=-3)
tma_builder_K = partial(build_tma_wgmma_k, iN=-3)
cord_func_K = partial(cord_func_K_major, iN=-3)

layerg.addTma("loadQW", matqWs, lambda t: t.wgmma_load(TileM, TileK, Major.K))
layerg.addTma("loadKW", matkWs, lambda t: t.wgmma_load(TileM, TileK, Major.K))
layerg.addTma("loadVW", matvWs, lambda t: t.wgmma_load(TileM, TileK, Major.K))
layerg.addTma("storeQ", attnQs, lambda t: t.wgmma("reduce", N, TileM, Major.MN))
layerg.addTma("storeK", attnKs, lambda t: t._build("reduce", 64, N, tma_store_attn_kv, cord_id))
layerg.addTma("storeV", attnVs, lambda t: t._build("reduce", 64, N, tma_store_attn_kv, cord_id))

matQ_attn_views = [attnQ.view(N, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM) for attnQ in attnQs]
matK_attn_views = [attnK.view(N, MAX_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM) for attnK in attnKs]
matV_attn_views = [attnV.view(N, MAX_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM) for attnV in attnVs]
matO_attn_view = attnO.view(N, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)

layerg.addTma("loadQ", matQ_attn_views, lambda t: t._build("load", HEAD_DIM, 64, tma_gqa_load_q, cord_gqa_load_q))
layerg.addTma("loadK", matK_attn_views, lambda t: t._build("load", HEAD_DIM, KVBlockSize, tma_builder_K, cord_func_K))
layerg.addTma("loadV", matV_attn_views, lambda t: t._build("load", HEAD_DIM, KVBlockSize, tma_builder_MN, cord_func_MN))

dae.build_groups()

loadEmbed1D = TmaLoad1D(matEmbed, bytes=HIDDEN * 2)
storeHidden1D = TmaStore1D(matHidden, bytes=HIDDEN * 2)
loadHidden1D = TmaLoad1D(matHidden, bytes=HIDDEN * 2)
storeRMSHidden1D = TmaStore1D(matRMSHidden, bytes=HIDDEN * 2)
loadRope1D = TmaLoad1D(matRope[token_pos], bytes=HEAD_DIM * matRope.element_size())

embed_rms = SchedRMSShared(
    num_token=N,
    epsilon=eps,
    hidden_size=HIDDEN,
    tmas=(TmaLoad1D(matRMSInputW[0]), loadEmbed1D, storeRMSHidden1D),
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
).bar("load", layerg["bar_pre_attn_rms"]).bar("store", layerg["bar_qkv_attn"])

VProj = SchedGemv(
    Gemv_M64N8,
    MNK=(VW, N, HIDDEN),
    tmas=(
        layerg["loadVW"],
        layerg["loadRMSLayer"],
        ToAttnVStoreCordAdapter(layerg["storeV"], token_pos),
    ),
).bar("load", layerg["bar_pre_attn_rms"]).bar("store", layerg["bar_qkv_attn"])

Gqa = SchedAttentionDecoding(
    reqs=REQ,
    seq_len=token_pos + 1,
    KV_BLOCK_SIZE=KVBlockSize,
    NUM_KV_HEADS=NUM_KV_HEAD,
    matO=matO_attn_view,
    tmas=(layerg["loadQ"], layerg["loadK"], layerg["loadV"]),
    need_norm=True,
    need_rope=True,
    rope_table=loadRope1D,
).bar("q", layerg["bar_q_proj"]).bar("k", layerg["bar_qkv_attn"]).bar("o", layerg["bar_attn_out"])

OutProj = SchedGemv(
    Gemv_M64N8,
    MNK=(HIDDEN, N, HIDDEN),
    tmas=(layerg["loadOutWs"], layerg["loadAttnOLayer"], layerg["reduceHiddenLayer"]),
)
if not parsed_args.debug_skip_attn_out_bar:
    OutProj.bar("load", layerg["bar_attn_out"])
OutProj.bar("store", layerg["bar_out_mlp"])

gate_projs = []
up_projs = []
for base, size, sms in MLP_CHUNKS:
    gate_projs.append(
        SchedGemv(
            Gemv_M64N8,
            MNK=((base, size), N, HIDDEN),
            tmas=(layerg["loadGate"], layerg["loadRMSLayer"], layerg["storeGateOut"]),
        ).place(sms).bar("load", layerg["bar_post_attn_rms"]).bar("store", layerg["bar_silu_in"])
    )
    up_projs.append(
        SchedGemv(
            Gemv_M64N8,
            MNK=((base, size), N, HIDDEN),
            tmas=(layerg["loadUp"], layerg["loadRMSLayer"], layerg["storeInterm"]),
        ).place(sms).bar("load", layerg["bar_post_attn_rms"]).bar("store", layerg["bar_silu_in"])
    )

silu_split = 6144
silu_low = SchedSmemSiLUInterleaved(
    num_token=N,
    gate_glob=matGateOut[:, :silu_split],
    up_glob=matInterm[:, :silu_split],
    out_glob=matSiLUOut[:, :silu_split],
)
silu_high = SchedSmemSiLUInterleaved(
    num_token=N,
    gate_glob=matGateOut[:, silu_split:],
    up_glob=matInterm[:, silu_split:],
    out_glob=matSiLUOut[:, silu_split:],
)
silu = ListSchedule([silu_low, silu_high], lead_bars={"input"}, tail_bars={"output"})
silu.bar("input", layerg["bar_silu_in"]).bar("output", layerg["bar_silu_out"])

down_projs = []
for base, size, _ in MLP_CHUNKS:
    down_projs.append(
        SchedGemv(
            Gemv_M64N8,
            MNK=(HIDDEN, N, (base, size)),
            tmas=(layerg["loadDown"], layerg["loadSiluLayer"], layerg["reduceHiddenLayer"]),
        ).place(64).bar("load", layerg["bar_silu_out"]).bar("store", layerg["bar_layer"])
    )

GemvFactory = layers_like(GemvLayer, dae, Gemv_M64N8)
LogitsProj = []
for i in range(logits_epoch):
    proj = GemvFactory(f"logits_proj_{i}", (matLogitsW[i], matRMSHidden, matLogits[i]), reduce=False)
    sched = proj.schedule_(group=False).split_M(6)
    if i == 0:
        sched.bar("load", layerg.over("bar_pre_attn_rms"))
        sched[0].no_prefetch()
    if i == logits_epoch - 1:
        sched.bar("store", systemg["bar_logits"])
    LogitsProj.append(sched.place(full_sms))

Argmax = SchedArgmax(
    num_token=N,
    logits_slice=logits_slice,
    num_slice=logits_epoch,
    AtomPartial=ARGMAX_PARTIAL_bf16_1152_50688_132,
    AtomReduce=ARGMAX_REDUCE_bf16_1152_132,
    matLogits=matLogits,
    matOutVal=matArgmaxVal,
    matOutIdx=matArgmaxIdx,
    matFinalOut=matTokens[:, token_offset + 1],
).bar("load", systemg["bar_logits"]).bar("val", systemg["bar_argmax_val"]).bar("idx", systemg["bar_argmax_idx"])

embed_rms = embed_rms.place(rms_sms)
copy_hidden = copy_hidden.place(N, base_sm=64)
pre_attn_rms = pre_attn_rms.place(rms_sms)
post_attn_rms = post_attn_rms.place(rms_sms)
QProj = QProj.place(128)
KProj = KProj.place(64, base_sm=64)
VProj = VProj.place(64)
Gqa = Gqa.place(REQ * NUM_KV_HEAD)
OutProj = OutProj.place(64)
silu = silu.place(4, base_sm=128)
Argmax = Argmax.place(full_sms)

matTokens[0, token_offset] = token_id

stage_items = [
    ("q_proj", [] if parsed_args.debug_disable_qkv else [QProj]),
    ("k_proj", [] if parsed_args.debug_disable_qkv else [KProj]),
    ("v_proj", [] if parsed_args.debug_disable_qkv else [VProj]),
    ("attn", [] if parsed_args.debug_disable_attn else [Gqa]),
    ("out", [OutProj]),
    ("post_attn_rms", [post_attn_rms]),
    ("gate", [gate_projs]),
    ("up", [up_projs]),
    ("silu", [silu]),
    ("down", [down_projs]),
    ("final_rms", [pre_attn_rms]),
    ("logits", [LogitsProj]),
    ("argmax", [Argmax]),
]

active_stage_items = []
for stage_name, items in stage_items:
    if stage_enabled(parsed_args.debug_stop_after, stage_name):
        active_stage_items.extend(items)

bound_items = [
    embed_rms,
    copy_hidden,
    *active_stage_items,
]
if parsed_args.debug_stop_after == "full":
    dae.bind_late_barrier_counts(*bound_items)
else:
    bind_late_barriers_with_default(dae, *bound_items, unresolved_count=0)
    bind_unused_late_barriers_to_zero(dae)

dae.i(
    *([embed_rms, copy_hidden] if stage_enabled(parsed_args.debug_stop_after, "embed") else []),
)
dae.i(
    *([QProj] if stage_enabled(parsed_args.debug_stop_after, "q_proj") else []),
    *([KProj] if stage_enabled(parsed_args.debug_stop_after, "k_proj") else []),
    *([VProj] if stage_enabled(parsed_args.debug_stop_after, "v_proj") else []),
    *([Gqa] if stage_enabled(parsed_args.debug_stop_after, "attn") else []),
    *([OutProj] if stage_enabled(parsed_args.debug_stop_after, "out") else []),
    *([post_attn_rms] if stage_enabled(parsed_args.debug_stop_after, "post_attn_rms") else []),
    *([gate_projs] if stage_enabled(parsed_args.debug_stop_after, "gate") else []),
    *([up_projs] if stage_enabled(parsed_args.debug_stop_after, "up") else []),
    *([silu] if stage_enabled(parsed_args.debug_stop_after, "silu") else []),
    *([down_projs] if stage_enabled(parsed_args.debug_stop_after, "down") else []),
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
)

print(
    f"run vdcores qwen3 single-token with {num_layers} layers..."
    f" stop_after={parsed_args.debug_stop_after}"
)
dae.s()
dae_app(dae)


def run_correctness_check():
    inputs = input_batch1(
        input_token_id_and_pos[0][0],
        mat=matTokens[0],
        positions=[input_token_id_and_pos[0][1]],
    )
    captured, _ = reference_pass(model, inputs)
    all_ok = True
    projection_ok = True
    warnings = []

    for i in range(min(2, num_layers)):
        layer = captured[i]
        print(f"[correctness] Layer {i}:")
        checks = [
            check_tensor_threshold("q_proj", layer["q_proj"][0, 0], attnQs[i][0], 5.0),
            check_tensor_threshold("k_proj", layer["k_proj"][0, 0], attnKs[i][0, token_pos], 5.0),
            check_tensor_threshold("v_proj", layer["v_proj"][0, 0], attnVs[i][0, token_pos], 5.0),
        ]
        future_k_max = 0.0
        if token_pos + 1 < MAX_SEQ_LEN:
            future_k_max = attnKs[i][0, token_pos + 1 :].abs().max().item()
        print(
            f"[correctness] k_cache[{i}] current-token diff uses raw k_proj; "
            f"future-position max={future_k_max:.6f}"
        )
        if future_k_max > 1e-5:
            print("[correctness] FAIL k_cache_future_positions: K cache store touched positions beyond the current token")
            all_ok = False
        projection_ok = projection_ok and all(passed for passed, _ in checks)
        all_ok = all_ok and projection_ok

    last_layer = captured[num_layers - 1]
    print(f"[correctness] Layer {num_layers - 1}:")
    attn_ok, _ = check_tensor_threshold("attn_context", last_layer["attn_context"][0, 0], attnO[0], 5.0)
    final_checks = [
        check_tensor_threshold("gate_proj", last_layer["gate_proj"][0, 0], matGateOut[0], 5.0),
        check_tensor_threshold("up_proj", last_layer["up_proj"][0, 0], matInterm[0], 5.0),
        check_tensor_threshold("silu", F.silu(last_layer["gate_proj"][0, 0]) * last_layer["up_proj"][0, 0], matSiLUOut[0], 5.0),
        check_tensor_threshold("final_hidden", last_layer["hidden_state_out"][0, 0], matHidden[0], 5.0),
        check_tensor_threshold("final_rms", captured["final"]["final_rms"][0, 0], matRMSHidden[0], 5.0),
        check_tensor_threshold("logits_0", captured["final"]["lm_head"][0, 0, :logits_slice], matLogits[0][0, :logits_slice], 10.0),
    ]
    if logits_epoch > 1:
        for i in range(1, logits_epoch):
            start = i * logits_slice
            end = min(vocab_size, start + logits_slice)
            final_checks.append(
                check_tensor_threshold(
                    f"logits_{i}",
                    captured["final"]["lm_head"][0, 0, start:end],
                    matLogits[i][0, : end - start],
                    10.0,
                )
            )
    all_ok = all_ok and attn_ok and all(passed for passed, _ in final_checks)

    q_norm_plain = plain_rms_per_head(last_layer["q_proj"][0, 0].view(config.num_attention_heads, HEAD_DIM), eps)
    q_norm_ref = last_layer["q_norm"][0, 0].float()
    k_norm_plain = plain_rms_per_head(last_layer["k_proj"][0, 0].view(config.num_key_value_heads, HEAD_DIM), eps)
    k_norm_ref = last_layer["k_norm"][0, 0].float()
    q_affine_diff = mean_relative_diff_pct(q_norm_ref, q_norm_plain, ref=q_norm_ref)
    k_affine_diff = mean_relative_diff_pct(k_norm_ref, k_norm_plain, ref=k_norm_ref)
    print(f"[correctness] q_norm affine delta: {q_affine_diff:.3f}%")
    print(f"[correctness] k_norm affine delta: {k_affine_diff:.3f}%")
    if q_affine_diff > 0.5 or k_affine_diff > 0.5:
        warnings.append("qk_norm_affine")
        print("[correctness] WARN qk_norm_affine: HF Q/K norm weights are not identity-scale")

    sanity_inputs = input_batch1(input_token_id_and_pos[0][0], positions=[7])
    sanity_captured, _ = reference_pass(model, sanity_inputs)
    sanity_layer = sanity_captured[0]
    q_norm_nonzero = sanity_layer["q_norm"][0, 0]
    k_norm_nonzero = sanity_layer["k_norm"][0, 0]
    q_hf = apply_qwen_rotate_half_rope(q_norm_nonzero, 7, HEAD_DIM, rope_theta)
    q_inter = apply_interleaved_rope(q_norm_nonzero, 7, HEAD_DIM, rope_theta)
    k_hf = apply_qwen_rotate_half_rope(k_norm_nonzero, 7, HEAD_DIM, rope_theta)
    k_inter = apply_interleaved_rope(k_norm_nonzero, 7, HEAD_DIM, rope_theta)
    q_rope_diff = mean_relative_diff_pct(q_hf, q_inter, ref=q_hf)
    k_rope_diff = mean_relative_diff_pct(k_hf, k_inter, ref=k_hf)
    print(f"[correctness] nonzero-pos rope delta q={q_rope_diff:.3f}% k={k_rope_diff:.3f}%")
    if q_rope_diff > 1.0 or k_rope_diff > 1.0:
        warnings.append("rope_rotation")
        print("[correctness] WARN rope_rotation: current interleaved rotation diverges from HF rotate_half semantics")

    ref_idx = torch.argmax(captured["final"]["lm_head"], dim=-1)
    dae_idx = matTokens[0, 1].item()
    token_ok = ref_idx[0, 0].item() == dae_idx
    print(
        f"[correctness] {'PASS' if token_ok else 'FAIL'} final_token: "
        f"ref={ref_idx[0, 0].item()}, dae={dae_idx}"
    )
    all_ok = all_ok and token_ok

    if projection_ok and not attn_ok:
        print("[correctness] classification: projections match; fused attention path is the likely mismatch source")
    if not projection_ok:
        print("[correctness] classification: mismatch appears before fused attention, likely in Q/K/V scheduling or stores")

    if not all_ok:
        raise RuntimeError("Correctness check failed")
    if warnings:
        print(
            "[correctness] single-token position-0 path passed; "
            f"supplemental warnings remain: {', '.join(warnings)}"
        )
    print("[correctness] all checks passed")


if parsed_args.correctness:
    run_correctness_check()
