import os
from dataclasses import dataclass

import torch
from dae.launcher import Launcher
from reference import input_batch1, permute_rope_activation
from transformers import AutoConfig, AutoModelForCausalLM

from cli import (
    DEFAULT_DECODE_INPUT_TOKEN,
    DEFAULT_MAX_SEQ_LEN,
    MODEL_NAME,
)


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

    prefill_token_id_and_pos = []
    input_token_id_and_pos = [(DEFAULT_DECODE_INPUT_TOKEN, 0)]
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
    matRMSInputW = [layer.input_layernorm.weight for layer in layers] + [model.model.norm.weight]
    matRMSPostAttnW = [layer.post_attention_layernorm.weight for layer in layers]
    matQNormWs = [permute_rope_head_weight(layer.self_attn.q_norm.weight.detach()) for layer in layers]
    matKNormWs = [permute_rope_head_weight(layer.self_attn.k_norm.weight.detach()) for layer in layers]
    matQwenSideInputs = []
    for q_norm_w, k_norm_w in zip(matQNormWs, matKNormWs):
        packed = torch.empty(MAX_SEQ_LEN, 3 * HEAD_DIM, dtype=dtype, device=gpu)
        packed[:, 0:HEAD_DIM] = q_norm_w.view(1, HEAD_DIM)
        packed[:, HEAD_DIM:2 * HEAD_DIM] = k_norm_w.view(1, HEAD_DIM)
        packed[:, 2 * HEAD_DIM:3 * HEAD_DIM] = matRope
        matQwenSideInputs.append(packed)

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

    vocab_size = model.lm_head.weight.shape[0]
    logits_slice = 64 * full_sms * 6
    logits_epoch = (vocab_size + logits_slice - 1) // logits_slice
    matLogits = []
    matLogitsW = []
    matLmHeadW = model.lm_head.weight.detach()
    matLmHeadW.resize_(logits_slice * logits_epoch, HIDDEN)
    matLmHeadW[vocab_size:, :].zero_()

    for i in range(logits_epoch):
        matLogitsW.append(matLmHeadW[i * logits_slice : (i + 1) * logits_slice])
        matLogits.append(torch.zeros(N, logits_slice, dtype=dtype, device=gpu))

    matArgmaxIdx = torch.zeros(N, full_sms, dtype=torch.long, device=gpu)
    matArgmaxVal = torch.zeros(N, full_sms, dtype=dtype, device=gpu)

    dae.set_persistent(matTokens)
    dae.set_streaming(matqWs, matkWs, matvWs, matOutWs, matUps, matGates, matDowns)

    return QwenScheduleContext(
        parsed_args=parsed_args,
        gpu=gpu,
        model=model,
        config=config,
        layers=layers,
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
        ctx.attnKs[layer_idx][0, :prefill_len].copy_(
            permute_rope_activation(k_cache, ctx.NUM_KV_HEAD, ctx.HEAD_DIM)
        )
        ctx.attnVs[layer_idx][0, :prefill_len].copy_(v_cache)

    return output
