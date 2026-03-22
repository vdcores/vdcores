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
from reference import check_tensor_threshold, input_batch1, reference_pass
from transformers import AutoConfig, AutoModelForCausalLM


DEFAULT_MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"
DEFAULT_MAX_SEQ_LEN = 512
DEFAULT_VOCAB_SIZE = 128256
DEBUG_STAGE_ORDER = (
    "embed",
    "q_proj",
    "q_rope",
    "k_proj",
    "k_rope",
    "v_proj",
    "attn",
    "out",
    "post_attn_rms",
    "gate_low",
    "gate_high",
    "up_low",
    "up_high",
    "silu_split",
    "gate_fused",
    "up_fused",
    "silu_fused",
    "down_low",
    "down_high",
    "final_rms",
    "logits",
    "argmax",
    "restore",
    "full",
)


def build_rope_table(max_seq_len, batch, head_dim, rope_theta, positions, device, dtype):
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    table = torch.ones(max_seq_len, batch, head_dim, dtype=dtype, device=device)
    for i, pos in enumerate(positions):
        pos_range = torch.arange(pos, max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(pos_range, inv_freq)
        table[: max_seq_len - pos, i, 0::2] = freqs.cos().to(dtype=dtype)
        table[: max_seq_len - pos, i, 1::2] = freqs.sin().to(dtype=dtype)
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


def stage_enabled(stage_name: str):
    requested_idx = DEBUG_STAGE_ORDER.index(parsed_args.debug_stop_after)
    stage_idx = DEBUG_STAGE_ORDER.index(stage_name)
    return stage_idx <= requested_idx


def bind_unused_late_barriers_to_zero():
    for group in dae.resource_groups.values():
        for name, bar_info in group.bars.items():
            if bar_info["late_bind"] and bar_info["count"] is None:
                group.bindBarrier(name, 0)


def print_barrier_counts():
    print("[debug] barrier counts:")
    for group_name, group in dae.resource_groups.items():
        for name, bar_info in group.bars.items():
            if bar_info["count"] is None:
                continue
            print(f"[debug]   {group_name}.{name} = {bar_info['count']}")


def bind_late_barriers_with_default(*insts, unresolved_count=None):
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
                raise ValueError(f"Barrier {group.name}.{name} observed inconsistent release counts: {sorted(matched_counts)}")
            raise ValueError(f"Could not infer release count for barrier {group.name}.{name}")


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
    arg_parser.add_argument("--debug-q-sms", type=int, default=None)
    arg_parser.add_argument("--debug-q-store-mode", choices=("auto", "reduce", "store"), default="auto")
    arg_parser.add_argument("--debug-k-sms", type=int, default=None)
    arg_parser.add_argument("--debug-v-sms", type=int, default=None)
    arg_parser.add_argument("--debug-out-sms", type=int, default=None)
    arg_parser.add_argument("--debug-down-low-sms", type=int, default=None)
    arg_parser.add_argument("--debug-down-high-sms", type=int, default=None)
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
input_token_id_and_pos = [(791, 0)]
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

matZero = torch.zeros(max(2048, INTERMIDIATE - 6144), dtype=dtype, device=gpu)
matRope = build_rope_table(
    MAX_SEQ_LEN,
    N,
    HEAD_DIM,
    rope_theta,
    [pos for _, pos in input_token_id_and_pos],
    gpu,
    torch.bfloat16,
)
matTokens = torch.zeros(N, MAX_SEQ_LEN, dtype=torch.int64, device=gpu)
matHidden = torch.rand(N, HIDDEN, dtype=dtype, device=gpu) - 0.5
matRMSHidden = torch.rand(N, HIDDEN, dtype=dtype, device=gpu) - 0.5

attnQs = [torch.zeros(REQ, QW, dtype=dtype, device=gpu) for _ in range(num_layers)]
attnKs = [torch.zeros(REQ, MAX_SEQ_LEN, KW, dtype=dtype, device=gpu) for _ in range(num_layers)]
attnVs = [torch.zeros(REQ, MAX_SEQ_LEN, VW, dtype=dtype, device=gpu) for _ in range(num_layers)]
attnO = torch.zeros(REQ, HIDDEN, dtype=dtype, device=gpu)
matInterm = torch.zeros(N, INTERMIDIATE, dtype=dtype, device=gpu)
matGateOut = torch.zeros(N, INTERMIDIATE, dtype=dtype, device=gpu)
matSiLUOut = torch.zeros(N, INTERMIDIATE, dtype=dtype, device=gpu)

logits_fold = 8
logits_slice = 8192 * logits_fold
vocab_size = matLmHeadW.shape[0]
logits_epoch = math.ceil(vocab_size / logits_slice)
matLmHeadW.resize_(logits_slice * logits_epoch, HIDDEN)
if logits_slice * logits_epoch > vocab_size:
    matLmHeadW[vocab_size:,].zero_()

matLogits = []
matLogitsW = []
matArgmaxIdx = torch.zeros(N, 128, dtype=torch.long, device=gpu)
matArgmaxVal = torch.zeros(N, 128, dtype=dtype, device=gpu)
matArgmaxOut = torch.zeros(N, dtype=torch.long, device=gpu)

for i in range(logits_epoch):
    matLogitsW.append(matLmHeadW[i * logits_slice : (i + 1) * logits_slice])
    matLogits.append(torch.zeros(N, logits_slice, dtype=dtype, device=gpu))

dae.set_persistent(matTokens)
dae.set_streaming(matqWs, matkWs, matvWs, matOutWs, matUps, matGates, matDowns)

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
layerg.addBarrier("bar_rms_layer", REQ)
layerg.addBarrier("bar_rms_mlp", REQ)
layerg.addBarrier("bar_silu_in")
layerg.addBarrier("bar_silu_out1")
layerg.addBarrier("bar_silu_out2")
layerg.addBarrier("bar_pre_attn_rms")
layerg.addBarrier("bar_post_attn_rms")

TileM, _, TileK = Gemv_M64N8.MNK
defaultg.addTma("loadRope", [matRope], lambda t: t._build("load", TileM, N, tma_load_tbl, cord_load_tbl))

layerg.addTma("loadRMSLayer", [matRMSHidden] * num_layers, lambda t: t.wgmma_load(N, TileK * Gemv_M64N8.n_batch, Major.K))
layerg.addTma("reduceHiddenLayer", [matHidden] * num_layers, lambda t: t.wgmma("reduce", N, TileM, Major.MN))
layerg.addTma("loadSiluLayer", [matSiLUOut] * num_layers, lambda t: t.wgmma_load(N, TileK * Gemv_M64N8.n_batch, Major.K))
layerg.addTma("storeSiluLayer", [matSiLUOut] * num_layers, lambda t: t.wgmma_store(N, TileM, Major.MN))
layerg.addTma("loadAttnOLayer", [attnO] * num_layers, lambda t: t.wgmma_load(N, TileK * Gemv_M64N8.n_batch, Major.K))
layerg.addTma("storeInterm", [matInterm] * num_layers, lambda t: t.wgmma_store(N, TileM, Major.MN))
layerg.addTma("storeGateOut", [matGateOut] * num_layers, lambda t: t.wgmma_store(N, TileM, Major.MN))
layerg.addTma("reduceInterm", [matInterm] * num_layers, lambda t: t.wgmma("reduce", N, TileM, Major.MN))
layerg.addTma("reduceGateOut", [matGateOut] * num_layers, lambda t: t.wgmma("reduce", N, TileM, Major.MN))

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
layerg.addTma("storeQPlain", attnQs, lambda t: t.wgmma_store(N, TileM, Major.MN))
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

    clear_size = INTERMIDIATE - 6144
    clear_interm = SchedCopy(
        size=clear_size * matInterm.element_size(),
        tmas=wrap_static(
            TmaLoad1D(matZero[:clear_size]),
            TmaStore1D(matInterm[0, 4096 : 4096 + clear_size]),
        ),
    )
    clear_gateout = SchedCopy(
        size=clear_size * matGateOut.element_size(),
        tmas=wrap_static(
            TmaLoad1D(matZero[:clear_size]),
            TmaStore1D(matGateOut[0, 4096 : 4096 + clear_size]),
        ),
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

    regStoreQ = RegStore(0, size=N * TileM * matQ_attn_views[0].element_size())
    regLoadQ = RegLoad(0)
    q_sms = parsed_args.debug_q_sms or 64
    k_sms = parsed_args.debug_k_sms or 16
    v_sms = parsed_args.debug_v_sms or 16
    out_sms = parsed_args.debug_out_sms or 64
    down_low_sms = parsed_args.debug_down_low_sms or 96
    down_high_sms = parsed_args.debug_down_high_sms or 64
    q_store_mode = parsed_args.debug_q_store_mode
    if q_store_mode == "auto":
        q_store_mode = "store" if q_sms == (QW // TileM) else "reduce"
    q_store_tma = layerg["storeQPlain"] if q_store_mode == "store" else layerg["storeQ"]
    QProj = SchedGemv(
        Gemv_M64N8,
        MNK=(QW, N, HIDDEN),
        tmas=(layerg["loadQW"], layerg["loadRMSLayer"], regStoreQ),
    ).bar("load", layerg["bar_pre_attn_rms"])
    QRope = SchedRope(
        ROPE_INTERLEAVE_512,
        tmas=(
            ToRopeTableCordAdapter(defaultg["loadRope"], token_pos, tile_repeats=max(1, HEAD_DIM // 64)),
            regLoadQ,
            ToSplitMCordAdapter(q_store_tma, QW // TileM, TileM),
        ),
    ).bar("store", layerg["bar_q_proj"])

    regStoreK = RegStore(0, size=N * TileM * matK_attn_views[0].element_size())
    regLoadK = RegLoad(0)
    KProj = SchedGemv(
        Gemv_M64N8,
        MNK=(KW, N, HIDDEN),
        tmas=(layerg["loadKW"], layerg["loadRMSLayer"], regStoreK),
    ).bar("load", layerg["bar_pre_attn_rms"])
    KRope = SchedRope(
        ROPE_INTERLEAVE_512,
        tmas=(
            ToRopeTableCordAdapter(defaultg["loadRope"], token_pos, tile_repeats=max(1, HEAD_DIM // 64)),
            regLoadK,
            ToAttnKVStoreCordAdapter(layerg["storeK"], KW // TileM, TileM, token_pos),
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
    ).bar("load", layerg["bar_pre_attn_rms"]).bar("store", layerg["bar_qkv_attn"])

    GemvFactory = layers_like(GemvLayer, dae, Gemv_M64N8)
    Gqa = SchedAttentionDecoding(
        reqs=N,
        seq_len=token_pos + 1,
        KV_BLOCK_SIZE=KVBlockSize,
        NUM_KV_HEADS=NUM_KV_HEAD,
        matO=matO_attn_view,
        tmas=(layerg["loadQ"], layerg["loadK"], layerg["loadV"]),
    ).bar("q", layerg["bar_q_proj"]).bar("k", layerg["bar_qkv_attn"]).bar("o", layerg["bar_attn_out"])

    OutProj = SchedGemv(
        Gemv_M64N8,
        MNK=(HIDDEN, N, HIDDEN),
        tmas=(layerg["loadOutWs"], layerg["loadAttnOLayer"], layerg["reduceHiddenLayer"]),
    ).bar("load", layerg["bar_attn_out"]).bar("store", layerg["bar_out_mlp"])

    regGate, regUp = 0, 1
    regStoreGate = RegStore(regGate, matGateOut[:, 0:TileM])
    regStoreUp = RegStore(regUp, matInterm[:, 0:TileM])

    gate_proj_low = SchedGemv(
        Gemv_M64N8,
        MNK=(4096, N, HIDDEN),
        tmas=(layerg["loadGate"], layerg["loadRMSLayer"], layerg["storeGateOut"]),
    ).bar("load", layerg["bar_post_attn_rms"])
    gate_proj_high = SchedGemv(
        Gemv_M64N8,
        MNK=((4096, 2048), N, HIDDEN),
        tmas=(layerg["loadGate"], layerg["loadRMSLayer"], layerg["reduceGateOut"]),
    ).bar("store", layerg["bar_silu_in"])
    up_proj_low = SchedGemv(
        Gemv_M64N8,
        MNK=(4096, N, HIDDEN),
        tmas=(layerg["loadUp"], layerg["loadRMSLayer"], layerg["storeInterm"]),
    ).bar("load", layerg["bar_post_attn_rms"])
    up_proj_high = SchedGemv(
        Gemv_M64N8,
        MNK=((4096, 2048), N, HIDDEN),
        tmas=(layerg["loadUp"], layerg["loadRMSLayer"], layerg["reduceInterm"]),
    ).bar("store", layerg["bar_silu_in"])

    mlp_split = 6144
    mlp_tail = INTERMIDIATE - mlp_split
    silu1 = SchedSmemSiLUInterleaved(
        num_token=N,
        gate_glob=matGateOut[:, :mlp_split],
        up_glob=matInterm[:, :mlp_split],
        out_glob=matSiLUOut[:, :mlp_split],
    ).bar("input", layerg["bar_silu_in"]).bar("output", layerg["bar_silu_out1"])
    gate_proj_fused = SchedGemv(
        Gemv_M64N8,
        MNK=((mlp_split, mlp_tail), N, HIDDEN),
        tmas=(layerg["loadGate"], layerg["loadRMSLayer"], regStoreGate),
    )
    up_proj_fused = SchedGemv(
        Gemv_M64N8,
        MNK=((mlp_split, mlp_tail), N, HIDDEN),
        tmas=(layerg["loadUp"], layerg["loadRMSLayer"], regStoreUp),
    )
    silu_fused = SchedRegSiLUFused(
        num_token=N,
        store_tma=layerg["storeSiluLayer"],
        reg_gate=regGate,
        reg_up=regUp,
        base_offset=mlp_split,
        stride=TileM,
    ).bar("output", layerg["bar_silu_out2"])
    down_proj_low = SchedGemv(
        Gemv_M64N8,
        MNK=(HIDDEN, N, 6144),
        tmas=(layerg["loadDown"], layerg["loadSiluLayer"], layerg["reduceHiddenLayer"]),
    )
    down_proj_high = SchedGemv(
        Gemv_M64N8,
        MNK=(HIDDEN, N, (mlp_split, mlp_tail)),
        tmas=(layerg["loadDown"], layerg["loadSiluLayer"], layerg["reduceHiddenLayer"]),
    ).bar("load", layerg["bar_silu_out2"]).bar("store", layerg["bar_layer"])
    down_proj_low.bar("load", layerg["bar_silu_out1"])

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
    clear_interm = clear_interm.place(1, base_sm=128)
    clear_gateout = clear_gateout.place(1, base_sm=129)
    pre_attn_rms = pre_attn_rms.place(rms_sms)
    post_attn_rms = post_attn_rms.place(rms_sms)
    QProj = QProj.place(q_sms)
    QRope = QRope.place(q_sms)
    KProj = KProj.place(k_sms, base_sm=64)
    KRope = KRope.place(k_sms, base_sm=64)
    VProj = VProj.place(v_sms, base_sm=64 + k_sms)
    Gqa = Gqa.place(N * NUM_KV_HEAD)
    OutProj = OutProj.place(out_sms)
    gate_proj_low = gate_proj_low.place(64)
    gate_proj_high = gate_proj_high.place(64)
    up_proj_low = up_proj_low.place(64, base_sm=64)
    up_proj_high = up_proj_high.place(64, base_sm=64)
    silu1 = silu1.place(4, base_sm=128)
    gate_proj_fused = gate_proj_fused.place(32)
    up_proj_fused = up_proj_fused.place(32)
    silu_fused = silu_fused.place(32)
    down_proj_low = down_proj_low.place(down_low_sms)
    down_proj_high = down_proj_high.place(down_high_sms)
    Argmax = Argmax.place(128)
    restore_bars_low = restore_bars_low.place(1, base_sm=128)
    restore_bars_high = restore_bars_high.place(1, base_sm=128)

    stage_items = [
        ("embed", [clear_interm, clear_gateout]),
        ("q_proj", [QProj]),
        ("q_rope", [QRope]),
        ("k_proj", [KProj]),
        ("k_rope", [KRope]),
        ("v_proj", [VProj]),
        ("attn", [Gqa]),
        ("out", [OutProj]),
        ("post_attn_rms", [post_attn_rms]),
        ("gate_low", [gate_proj_low]),
        ("gate_high", [gate_proj_high]),
        ("up_low", [up_proj_low]),
        ("up_high", [up_proj_high]),
        ("silu_split", [silu1]),
        ("gate_fused", [gate_proj_fused]),
        ("up_fused", [up_proj_fused]),
        ("silu_fused", [silu_fused]),
        ("down_low", [down_proj_low]),
        ("down_high", [down_proj_high]),
        ("final_rms", [pre_attn_rms]),
        ("logits", [LogitsProj]),
        ("argmax", [Argmax]),
        ("restore", [restore_bars_low] if need_token_restore else []),
    ]

    active_stage_items = []
    for stage_name, items in stage_items:
        if stage_enabled(stage_name):
            active_stage_items.extend(items)

    bound_items = [
        embed_rms,
        copy_hidden,
        restore_bars_high,
        *active_stage_items,
    ]

    if parsed_args.debug_stop_after != "full":
        bind_late_barriers_with_default(*bound_items, unresolved_count=0)
        bind_unused_late_barriers_to_zero()
    else:
        dae.bind_late_barrier_counts(
            *bound_items,
        )
    if parsed_args.debug_print_barriers:
        print_barrier_counts()

    if parsed_args.dry_build:
        return

    dae.i(embed_rms, copy_hidden, restore_bars_high)
    dae.i(
        *([clear_interm, clear_gateout] if stage_enabled("embed") else []),
        *([QProj] if stage_enabled("q_proj") else []),
        *([QRope] if stage_enabled("q_rope") else []),
        *([KProj] if stage_enabled("k_proj") else []),
        *([KRope] if stage_enabled("k_rope") else []),
        *([VProj] if stage_enabled("v_proj") else []),
        *([Gqa] if stage_enabled("attn") else []),
        *([OutProj] if stage_enabled("out") else []),
        *([post_attn_rms] if stage_enabled("post_attn_rms") else []),
        *([gate_proj_low] if stage_enabled("gate_low") else []),
        *([gate_proj_high] if stage_enabled("gate_high") else []),
        *([up_proj_low] if stage_enabled("up_low") else []),
        *([up_proj_high] if stage_enabled("up_high") else []),
        *([silu1] if stage_enabled("silu_split") else []),
        *([gate_proj_fused] if stage_enabled("gate_fused") else []),
        *([up_proj_fused] if stage_enabled("up_fused") else []),
        *([silu_fused] if stage_enabled("silu_fused") else []),
        *([down_proj_low] if stage_enabled("down_low") else []),
        *([down_proj_high] if stage_enabled("down_high") else []),
        *([pre_attn_rms] if stage_enabled("final_rms") else []),
        *(
            [
                LoopM.toNext(dae.copy_mptrs(), num_layers, resource_group=layerg),
                LoopC.toNext(dae.copy_cptrs(), num_layers),
            ]
            if stage_enabled("final_rms")
            else []
        ),
        *([LogitsProj] if stage_enabled("logits") else []),
        *([Argmax] if stage_enabled("argmax") else []),
        *([restore_bars_low] if stage_enabled("restore") and need_token_restore else []),
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
            f"num_layers={num_layers}, q_sms={parsed_args.debug_q_sms or 64}, "
            f"q_store_mode={parsed_args.debug_q_store_mode}, "
            f"k_sms={parsed_args.debug_k_sms or 16}, v_sms={parsed_args.debug_v_sms or 16}, "
            f"out_sms={parsed_args.debug_out_sms or 64}, "
            f"down_low_sms={parsed_args.debug_down_low_sms or 96}, "
            f"down_high_sms={parsed_args.debug_down_high_sms or 64}"
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

    for i in range(min(2, num_layers)):
        layer = captured[i]
        checks = [
            check_tensor_threshold("v_proj", layer["v_proj"][0, 0], attnVs[i][0, 0], 5.0),
            check_tensor_threshold(
                "q_proj",
                permute_rope_activation(layer["q_proj"][0, 0], HEAD_DIM, QW // HEAD_DIM),
                attnQs[i][0],
                5.0,
            ),
            check_tensor_threshold(
                "k_proj",
                permute_rope_activation(layer["k_proj"][0, 0], HEAD_DIM, KW // HEAD_DIM),
                attnKs[i][0, 0],
                5.0,
            ),
        ]
        all_ok = all_ok and all(passed for passed, _ in checks)

    layer = captured[num_layers - 1]
    silu_ref = F.silu(layer["gate_proj"][0, 0]) * layer["up_proj"][0, 0]
    final_checks = [
        check_tensor_threshold("gate_proj_split", layer["gate_proj"][0, 0, :6144], matGateOut[0, :6144], 5.0),
        check_tensor_threshold("up_proj_split", layer["up_proj"][0, 0, :6144], matInterm[0, :6144], 5.0),
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
    all_ok = all_ok and ref_idx[0, 0].item() == dae_idx
    if not all_ok:
        raise RuntimeError("Correctness check failed")


if parsed_args.correctness:
    run_correctness_check()
