import os
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from dae.launcher import Launcher
from reference import input_batch1, permute_rope_activation
from transformers import AutoConfig, AutoModelForCausalLM

from cli import (
    DEFAULT_DECODE_INPUT_TOKEN,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_PREFILL_TOKEN,
    MODEL_NAME,
)


DEFAULT_VOCAB_SIZE = 151936
DEFAULT_NUM_LAYERS = 28


def env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def env_text(name: str, default: str = "") -> str:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip()


def env_int_optional(name: str) -> int | None:
    raw = env_text(name)
    return None if raw == "" else int(raw)


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None else float(raw)


def hf_auth_kwargs() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN")
    if token:
        return {"token": token}
    return {}


def build_synthetic_inputs(
    *,
    vocab_size: int,
    hidden_size: int,
    intermediate_size: int,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    num_layers: int,
    device,
    dtype,
):
    def empty(*shape):
        return torch.empty(*shape, dtype=dtype, device=device)

    qw = head_dim * num_q_heads
    kw = head_dim * num_kv_heads
    vw = head_dim * num_kv_heads

    return {
        "embed": empty(vocab_size, hidden_size),
        "rms_input_w": [empty(hidden_size) for _ in range(num_layers)] + [empty(hidden_size)],
        "rms_post_attn_w": [empty(hidden_size) for _ in range(num_layers)],
        "q_norm_w": [empty(head_dim) for _ in range(num_layers)],
        "k_norm_w": [empty(head_dim) for _ in range(num_layers)],
        "qws": [empty(qw, hidden_size) for _ in range(num_layers)],
        "kws": [empty(kw, hidden_size) for _ in range(num_layers)],
        "vws": [empty(vw, hidden_size) for _ in range(num_layers)],
        "out_ws": [empty(hidden_size, hidden_size) for _ in range(num_layers)],
        "ups": [empty(intermediate_size, hidden_size) for _ in range(num_layers)],
        "gates": [empty(intermediate_size, hidden_size) for _ in range(num_layers)],
        "downs": [empty(hidden_size, intermediate_size) for _ in range(num_layers)],
        "lm_head": empty(vocab_size, hidden_size),
    }


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


@dataclass
class QwenScheduleContext:
    parsed_args: object
    gpu: torch.device
    model: object
    config: object
    layers: list
    BATCH: int
    REQ: int
    N: int
    KVBlockSize: int
    rms_sms: int
    num_sms: int
    full_sms: int
    MAX_SEQ_LEN: int
    dtype: torch.dtype
    eps: float
    rope_theta: float
    HIDDEN: int
    INTERMIDIATE: int
    HEAD_DIM: int
    NUM_Q_HEAD: int
    NUM_KV_HEAD: int
    HEAD_GROUP_SIZE: int
    QW: int
    KW: int
    VW: int
    num_layers: int
    dae: Launcher
    prefill_token_id_and_pos: list
    input_token_id_and_pos: list
    num_generates: int
    matRope: torch.Tensor
    matTokens: torch.Tensor
    matHidden: torch.Tensor
    matRMSHidden: torch.Tensor
    attnQs: list
    attnKs: list
    attnVs: list
    attnO: torch.Tensor
    matInterm: torch.Tensor
    matGateOut: torch.Tensor
    matSiLUOut: torch.Tensor
    matEmbed: torch.Tensor
    matRMSInputW: list
    matRMSPostAttnW: list
    matQNormWs: list
    matKNormWs: list
    matQwenSideInputs: list
    matqWs: list
    matkWs: list
    matvWs: list
    matOutWs: list
    matUps: list
    matGates: list
    matDowns: list
    vocab_size: int
    logits_slice: int
    logits_epoch: int
    matLogits: list
    matLogitsW: list
    matArgmaxIdx: torch.Tensor
    matArgmaxVal: torch.Tensor


def build_runtime_context(parsed_args):
    gpu = torch.device("cuda")

    if parsed_args.dry_build:
        config = SimpleNamespace(
            max_position_embeddings=DEFAULT_MAX_SEQ_LEN,
            hidden_size=2048,
            intermediate_size=6144,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            rms_norm_eps=1.0e-6,
            rope_scaling={"rope_theta": 1000000.0},
            vocab_size=DEFAULT_VOCAB_SIZE,
        )
        model = None
        dtype = torch.bfloat16
        layers = [None] * DEFAULT_NUM_LAYERS
    else:
        auth_kwargs = hf_auth_kwargs()
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
        layers = list(model.model.layers)
        dtype = model.dtype

    if parsed_args.debug_layer_start < 0 or parsed_args.debug_layer_start >= len(layers):
        raise ValueError("--debug-layer-start must select an existing layer")
    layer_end = None
    if parsed_args.debug_num_layers is not None:
        if parsed_args.debug_num_layers <= 0:
            raise ValueError("--debug-num-layers must be positive")
        layer_end = parsed_args.debug_layer_start + parsed_args.debug_num_layers
    layers = layers[parsed_args.debug_layer_start:layer_end]

    if not 1 <= parsed_args.batch_size <= 8:
        raise ValueError("--batch-size must be in [1, 8]")
    if parsed_args.max_seq_len <= 0:
        raise ValueError("--max-seq-len must be positive")
    BATCH = parsed_args.batch_size
    REQ, N = 8, 8
    KVBlockSize = 64
    rms_sms = BATCH
    num_sms = 128
    full_sms = 132
    MAX_SEQ_LEN = min(config.max_position_embeddings, parsed_args.max_seq_len)
    if not 0 <= parsed_args.prefill_length < MAX_SEQ_LEN:
        raise ValueError("--prefill-length must be in [0, max-seq-len)")
    dae = Launcher(full_sms, device=gpu)

    prefill_token_id_and_pos = [
        (DEFAULT_PREFILL_TOKEN, pos) for pos in range(parsed_args.prefill_length)
    ]
    input_token_id_and_pos = [
        (DEFAULT_DECODE_INPUT_TOKEN, parsed_args.prefill_length)
    ]
    num_generates = 0 if (parsed_args.correctness or parsed_args.dry_build) else parsed_args.num_generates - 1

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

    matRope = build_interleaved_rope_rows(MAX_SEQ_LEN, HEAD_DIM, rope_theta, gpu, dtype)
    matTokens = torch.zeros(N, MAX_SEQ_LEN, dtype=torch.int64, device=gpu)
    matHidden = torch.rand(N, HIDDEN, dtype=dtype, device=gpu) - 0.5
    matRMSHidden = torch.rand(N, HIDDEN, dtype=dtype, device=gpu) - 0.5

    attnQs = [torch.zeros(REQ, HIDDEN, dtype=dtype, device=gpu) for _ in range(num_layers)]
    # Keep request and head adjacent so the existing rank-4/rank-5 attention
    # TMA descriptors can collapse them without crossing the sequence stride.
    attnKs = [torch.zeros(MAX_SEQ_LEN, REQ, KW, dtype=dtype, device=gpu) for _ in range(num_layers)]
    attnVs = [torch.zeros(MAX_SEQ_LEN, REQ, VW, dtype=dtype, device=gpu) for _ in range(num_layers)]
    attnO = torch.zeros(REQ, HIDDEN, dtype=dtype, device=gpu)
    matInterm = torch.zeros(N, INTERMIDIATE, dtype=dtype, device=gpu)
    matGateOut = torch.zeros(N, INTERMIDIATE, dtype=dtype, device=gpu)
    matSiLUOut = torch.zeros(N, INTERMIDIATE, dtype=dtype, device=gpu)

    if parsed_args.dry_build:
        synthetic = build_synthetic_inputs(
            vocab_size=getattr(config, "vocab_size", DEFAULT_VOCAB_SIZE),
            hidden_size=HIDDEN,
            intermediate_size=INTERMIDIATE,
            head_dim=HEAD_DIM,
            num_q_heads=NUM_Q_HEAD,
            num_kv_heads=NUM_KV_HEAD,
            num_layers=num_layers,
            device=gpu,
            dtype=dtype,
        )
        matEmbed = synthetic["embed"]
        matRMSInputW = synthetic["rms_input_w"]
        matRMSPostAttnW = synthetic["rms_post_attn_w"]
        matQNormWs = synthetic["q_norm_w"]
        matKNormWs = synthetic["k_norm_w"]
        matqWs = synthetic["qws"]
        matkWs = synthetic["kws"]
        matvWs = synthetic["vws"]
        matOutWs = synthetic["out_ws"]
        matUps = synthetic["ups"]
        matGates = synthetic["gates"]
        matDowns = synthetic["downs"]
        matLmHeadW = synthetic["lm_head"]
    else:
        matEmbed = model.model.embed_tokens.weight
        matRMSInputW = [layer.input_layernorm.weight for layer in layers] + [model.model.norm.weight]
        matRMSPostAttnW = [layer.post_attention_layernorm.weight for layer in layers]
        matQNormWs = [permute_rope_head_weight(layer.self_attn.q_norm.weight.detach()) for layer in layers]
        matKNormWs = [permute_rope_head_weight(layer.self_attn.k_norm.weight.detach()) for layer in layers]
        matqWs = [
            permute_rope_weight(layer.self_attn.q_proj.weight, NUM_Q_HEAD, HEAD_DIM, HIDDEN)
            for layer in layers
        ]
        matkWs = [
            permute_rope_weight(layer.self_attn.k_proj.weight, NUM_KV_HEAD, HEAD_DIM, HIDDEN)
            for layer in layers
        ]
        matvWs = [layer.self_attn.v_proj.weight for layer in layers]
        matOutWs = [layer.self_attn.o_proj.weight for layer in layers]
        matUps = [layer.mlp.up_proj.weight for layer in layers]
        matGates = [layer.mlp.gate_proj.weight for layer in layers]
        matDowns = [layer.mlp.down_proj.weight for layer in layers]
        matLmHeadW = model.lm_head.weight.detach()

    matQwenSideInputsTensor = torch.empty(num_layers, MAX_SEQ_LEN, 3 * HEAD_DIM, dtype=dtype, device=gpu)
    matQwenSideInputs = []
    for layer_idx, (q_norm_w, k_norm_w) in enumerate(zip(matQNormWs, matKNormWs)):
        packed = matQwenSideInputsTensor[layer_idx]
        packed[:, 0:HEAD_DIM] = q_norm_w.view(1, HEAD_DIM)
        packed[:, HEAD_DIM:2 * HEAD_DIM] = k_norm_w.view(1, HEAD_DIM)
        packed[:, 2 * HEAD_DIM:3 * HEAD_DIM] = matRope
        matQwenSideInputs.append(packed)

    vocab_size = matLmHeadW.shape[0]
    logits_slice = 64 * full_sms * 6
    logits_epoch = (vocab_size + logits_slice - 1) // logits_slice
    matLogits = []
    matLogitsW = []
    padded_lm_head = torch.zeros(
        logits_slice * logits_epoch,
        HIDDEN,
        dtype=dtype,
        device=gpu,
    )
    padded_lm_head[:vocab_size].copy_(matLmHeadW)
    matLmHeadW = padded_lm_head

    for i in range(logits_epoch):
        matLogitsW.append(matLmHeadW[i * logits_slice : (i + 1) * logits_slice])
        matLogits.append(torch.zeros(N, logits_slice, dtype=dtype, device=gpu))

    matArgmaxIdx = torch.zeros(N, full_sms, dtype=torch.long, device=gpu)
    matArgmaxVal = torch.zeros(N, full_sms, dtype=dtype, device=gpu)

    cache_target = env_text(
        "QWEN1P7B_CACHE_WINDOW_TARGET",
        "tokens" if env_flag("QWEN1P7B_ENABLE_CACHE_HINTS", False) else "none",
    ).lower()
    if cache_target not in {"", "none", "off"}:
        cache_targets = {
            "tokens": matTokens,
            "embed": matEmbed,
            "rms_input_w0": matRMSInputW[0],
            "qwen_side_inputs_all": matQwenSideInputsTensor,
            "attn_k_l0": attnKs[0],
            "attn_v_l0": attnVs[0],
            "q_proj_l0": matqWs[0],
            "k_proj_l0": matkWs[0],
            "v_proj_l0": matvWs[0],
            "out_proj_l0": matOutWs[0],
            "up_proj_l0": matUps[0],
            "gate_proj_l0": matGates[0],
            "down_proj_l0": matDowns[0],
            "lm_head": matLmHeadW,
        }
        if cache_target not in cache_targets:
            raise ValueError(
                "Unsupported QWEN1P7B_CACHE_WINDOW_TARGET="
                f"{cache_target!r}; expected one of {sorted(cache_targets)} or none"
            )

        cache_mode = env_text("QWEN1P7B_CACHE_WINDOW_MODE", "persisting").lower()
        cache_num_bytes = env_int_optional("QWEN1P7B_CACHE_WINDOW_BYTES")
        if cache_mode == "persisting":
            dae.set_cache_window(
                cache_targets[cache_target],
                hit_ratio=env_float("QWEN1P7B_CACHE_WINDOW_HIT_RATIO", 1.0),
                hit_policy=2,
                miss_policy=0,
                num_bytes=cache_num_bytes,
            )
        elif cache_mode == "streaming":
            dae.set_cache_window(
                cache_targets[cache_target],
                hit_ratio=env_float("QWEN1P7B_CACHE_WINDOW_HIT_RATIO", 0.0),
                hit_policy=0,
                miss_policy=1,
                num_bytes=cache_num_bytes,
            )
        else:
            raise ValueError(
                "Unsupported QWEN1P7B_CACHE_WINDOW_MODE="
                f"{cache_mode!r} (expected persisting or streaming)"
            )

    return QwenScheduleContext(
        parsed_args=parsed_args,
        gpu=gpu,
        model=model,
        config=config,
        layers=layers,
        BATCH=BATCH,
        REQ=REQ,
        N=N,
        KVBlockSize=KVBlockSize,
        rms_sms=rms_sms,
        num_sms=num_sms,
        full_sms=full_sms,
        MAX_SEQ_LEN=MAX_SEQ_LEN,
        dtype=dtype,
        eps=eps,
        rope_theta=rope_theta,
        HIDDEN=HIDDEN,
        INTERMIDIATE=INTERMIDIATE,
        HEAD_DIM=HEAD_DIM,
        NUM_Q_HEAD=NUM_Q_HEAD,
        NUM_KV_HEAD=NUM_KV_HEAD,
        HEAD_GROUP_SIZE=HEAD_GROUP_SIZE,
        QW=QW,
        KW=KW,
        VW=VW,
        num_layers=num_layers,
        dae=dae,
        prefill_token_id_and_pos=prefill_token_id_and_pos,
        input_token_id_and_pos=input_token_id_and_pos,
        num_generates=num_generates,
        matRope=matRope,
        matTokens=matTokens,
        matHidden=matHidden,
        matRMSHidden=matRMSHidden,
        attnQs=attnQs,
        attnKs=attnKs,
        attnVs=attnVs,
        attnO=attnO,
        matInterm=matInterm,
        matGateOut=matGateOut,
        matSiLUOut=matSiLUOut,
        matEmbed=matEmbed,
        matRMSInputW=matRMSInputW,
        matRMSPostAttnW=matRMSPostAttnW,
        matQNormWs=matQNormWs,
        matKNormWs=matKNormWs,
        matQwenSideInputs=matQwenSideInputs,
        matqWs=matqWs,
        matkWs=matkWs,
        matvWs=matvWs,
        matOutWs=matOutWs,
        matUps=matUps,
        matGates=matGates,
        matDowns=matDowns,
        vocab_size=vocab_size,
        logits_slice=logits_slice,
        logits_epoch=logits_epoch,
        matLogits=matLogits,
        matLogitsW=matLogitsW,
        matArgmaxIdx=matArgmaxIdx,
        matArgmaxVal=matArgmaxVal,
    )


def seed_prefill_kv_cache(ctx: QwenScheduleContext):
    if ctx.model is None:
        return None

    for layer_k, layer_v in zip(ctx.attnKs, ctx.attnVs):
        layer_k.zero_()
        layer_v.zero_()

    if not ctx.prefill_token_id_and_pos:
        return None

    prefill_tokens = [token for token, _ in ctx.prefill_token_id_and_pos]
    prefill_positions = [pos for _, pos in ctx.prefill_token_id_and_pos]
    inputs = input_batch1(
        *prefill_tokens,
        mat=ctx.matTokens[0],
        positions=prefill_positions,
    )
    with torch.no_grad():
        output = ctx.model(**inputs, use_cache=True)

    pkv = output.past_key_values
    prefill_len = len(prefill_tokens)
    for layer_idx in range(ctx.num_layers):
        layer_cache = pkv.layers[layer_idx]
        k_cache = layer_cache.keys[0].permute(1, 0, 2).reshape(prefill_len, ctx.KW)
        v_cache = layer_cache.values[0].permute(1, 0, 2).reshape(prefill_len, ctx.VW)
        k_cache = permute_rope_activation(
            k_cache, ctx.NUM_KV_HEAD, ctx.HEAD_DIM
        )
        ctx.attnKs[layer_idx][:prefill_len].copy_(
            k_cache[:, None, :].expand(-1, ctx.REQ, -1)
        )
        ctx.attnVs[layer_idx][:prefill_len].copy_(
            v_cache[:, None, :].expand(-1, ctx.REQ, -1)
        )

    return output
