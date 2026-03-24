import os
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from dae.launcher import Launcher
from reference import input_batch1
from transformers import AutoConfig, AutoModelForCausalLM

from cli import DEFAULT_DECODE_INPUT_TOKEN, DEFAULT_MAX_SEQ_LEN, MODEL_NAME


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
    use_local_generated_weights: bool
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
    MOE_INTERMEDIATE: int
    TOP_K: int
    EXPERT_BUFFER_COUNT: int
    NUM_EXPERTS: int
    HEAD_DIM: int
    NUM_Q_HEAD: int
    NUM_KV_HEAD: int
    HEAD_GROUP_SIZE: int
    QW: int
    KW: int
    VW: int
    num_layers: int
    dae: Launcher
    input_token_id_and_pos: list
    num_generates: int
    matRope: torch.Tensor
    matTokens: torch.Tensor
    matHidden: torch.Tensor
    matRMSHidden: torch.Tensor
    attnQs: torch.Tensor
    attnKs: torch.Tensor
    attnVs: torch.Tensor
    attnO: torch.Tensor
    matRouterLogits: torch.Tensor
    matRouterTopKIdx: torch.Tensor
    matRouterTopKWeight: torch.Tensor
    matExpertAct: list
    matExpertActScaled: list
    matEmbed: torch.Tensor
    matRMSInputW0: torch.Tensor
    matRMSInputWLoop: torch.Tensor
    matRMSPostAttnW: torch.Tensor
    matQNormWs: list
    matKNormWs: list
    matQwenSideInputs: torch.Tensor
    matqWs: torch.Tensor
    matkWs: torch.Tensor
    matvWs: torch.Tensor
    matOutWs: torch.Tensor
    matRouterWs: torch.Tensor
    matExpertGateWs: torch.Tensor
    matExpertUpWs: torch.Tensor
    matExpertDownWs: torch.Tensor
    vocab_size: int
    logits_slice: int
    logits_epoch: int
    matLogits: list
    matLogitsW: list
    matLmHeadWFull: torch.Tensor
    matArgmaxIdx: torch.Tensor
    matArgmaxVal: torch.Tensor


def randn(*shape, dtype, device):
    return torch.rand(*shape, dtype=dtype, device=device) - 0.5


def build_runtime_context(parsed_args):
    gpu = torch.device("cuda")
    if parsed_args.local_generated_weights:
        config = AutoConfig.from_pretrained(
            MODEL_NAME,
            cache_dir=parsed_args.hf_cache_dir,
            token=os.environ["HF_TOKEN"],
        )
        model = None
        num_layers = min(parsed_args.synthetic_num_layers, config.num_hidden_layers)
        layers = [SimpleNamespace() for _ in range(num_layers)]
        dtype = torch.bfloat16
    else:
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
        num_layers = len(layers)
        dtype = model.dtype

    REQ, N = 8, 8
    KVBlockSize = 64
    rms_sms = REQ
    num_sms = 128
    full_sms = 132
    MAX_SEQ_LEN = min(config.max_position_embeddings, DEFAULT_MAX_SEQ_LEN)
    dae = Launcher(full_sms, device=gpu)

    input_token_id_and_pos = [(DEFAULT_DECODE_INPUT_TOKEN, 0)]
    num_generates = 0

    eps = config.rms_norm_eps
    rope_theta = get_rope_theta(config)
    HIDDEN = config.hidden_size
    INTERMIDIATE = config.intermediate_size
    MOE_INTERMEDIATE = config.moe_intermediate_size
    TOP_K = config.num_experts_per_tok
    if parsed_args.fixed_top_k is not None:
        if parsed_args.fixed_top_k <= 0:
            raise ValueError("--fixed-top-k must be positive")
        TOP_K = min(TOP_K, parsed_args.fixed_top_k)
    if parsed_args.correctness and parsed_args.fixed_top_k is not None:
        raise ValueError("--fixed-top-k changes model semantics and is incompatible with --correctness")
    NUM_EXPERTS = getattr(config, "num_experts", getattr(config, "num_local_experts"))
    expert_buffer_count = TOP_K if parsed_args.expert_buffers is None else parsed_args.expert_buffers
    if expert_buffer_count <= 0:
        raise ValueError("--expert-buffers must be positive")
    expert_buffer_count = min(expert_buffer_count, TOP_K)
    HEAD_DIM = getattr(config, "head_dim", HIDDEN // config.num_attention_heads)
    NUM_Q_HEAD = config.num_attention_heads
    NUM_KV_HEAD = config.num_key_value_heads
    HEAD_GROUP_SIZE = NUM_Q_HEAD // NUM_KV_HEAD
    QW = HEAD_DIM * NUM_Q_HEAD
    KW = HEAD_DIM * NUM_KV_HEAD
    VW = HEAD_DIM * NUM_KV_HEAD

    matRope = build_interleaved_rope_rows(MAX_SEQ_LEN, HEAD_DIM, rope_theta, gpu, dtype)
    matTokens = torch.zeros(N, MAX_SEQ_LEN, dtype=torch.int64, device=gpu)
    matHidden = torch.zeros(N, HIDDEN, dtype=dtype, device=gpu)
    matRMSHidden = torch.zeros(N, HIDDEN, dtype=dtype, device=gpu)

    attnQs = torch.zeros(REQ, QW, dtype=dtype, device=gpu)
    attnKs = torch.zeros(REQ, MAX_SEQ_LEN, KW, dtype=dtype, device=gpu)
    attnVs = torch.zeros(REQ, MAX_SEQ_LEN, VW, dtype=dtype, device=gpu)
    attnO = torch.zeros(REQ, QW, dtype=dtype, device=gpu)

    matRouterLogits = torch.zeros(N, NUM_EXPERTS, dtype=dtype, device=gpu)
    matRouterTopKIdx = torch.zeros(1, TOP_K, dtype=torch.int32, device=gpu)
    matRouterTopKWeight = torch.zeros(TOP_K, 32, dtype=dtype, device=gpu)
    matExpertAct = [torch.zeros(N, MOE_INTERMEDIATE, dtype=dtype, device=gpu) for _ in range(expert_buffer_count)]
    matExpertActScaled = [torch.zeros(N, MOE_INTERMEDIATE, dtype=dtype, device=gpu) for _ in range(expert_buffer_count)]

    if parsed_args.local_generated_weights:
        matEmbed = randn(config.vocab_size, HIDDEN, dtype=dtype, device=gpu)
        matRMSInputW0 = randn(HIDDEN, dtype=dtype, device=gpu)
        matRMSInputWLoop = randn(num_layers, HIDDEN, dtype=dtype, device=gpu)
        matRMSPostAttnW = randn(num_layers, HIDDEN, dtype=dtype, device=gpu)
        matQNormWs = [randn(HEAD_DIM, dtype=dtype, device=gpu) for _ in range(num_layers)]
        matKNormWs = [randn(HEAD_DIM, dtype=dtype, device=gpu) for _ in range(num_layers)]
    else:
        matEmbed = model.model.embed_tokens.weight
        matRMSInputW0 = layers[0].input_layernorm.weight
        matRMSInputWLoop = torch.stack(
            [layer.input_layernorm.weight for layer in layers[1:]] + [model.model.norm.weight],
            dim=0,
        ).contiguous()
        matRMSPostAttnW = torch.stack(
            [layer.post_attention_layernorm.weight for layer in layers],
            dim=0,
        ).contiguous()
        matQNormWs = [permute_rope_head_weight(layer.self_attn.q_norm.weight.detach()) for layer in layers]
        matKNormWs = [permute_rope_head_weight(layer.self_attn.k_norm.weight.detach()) for layer in layers]

    matQwenSideInputs = torch.empty(num_layers, 3 * HEAD_DIM, dtype=dtype, device=gpu)
    for i, (q_norm_w, k_norm_w) in enumerate(zip(matQNormWs, matKNormWs)):
        matQwenSideInputs[i, 0:HEAD_DIM] = q_norm_w
        matQwenSideInputs[i, HEAD_DIM:2 * HEAD_DIM] = k_norm_w
        matQwenSideInputs[i, 2 * HEAD_DIM:3 * HEAD_DIM] = matRope[0]

    if parsed_args.local_generated_weights:
        matqWs = randn(num_layers, QW, HIDDEN, dtype=dtype, device=gpu)
        matkWs = randn(num_layers, KW, HIDDEN, dtype=dtype, device=gpu)
        matvWs = randn(num_layers, VW, HIDDEN, dtype=dtype, device=gpu)
        matOutWs = randn(num_layers, HIDDEN, QW, dtype=dtype, device=gpu)
        matRouterWs = randn(num_layers, NUM_EXPERTS, HIDDEN, dtype=dtype, device=gpu)
        matExpertGateWs = randn(num_layers, NUM_EXPERTS, MOE_INTERMEDIATE, HIDDEN, dtype=dtype, device=gpu)
        matExpertUpWs = randn(num_layers, NUM_EXPERTS, MOE_INTERMEDIATE, HIDDEN, dtype=dtype, device=gpu)
        matExpertDownWs = randn(num_layers, NUM_EXPERTS, HIDDEN, MOE_INTERMEDIATE, dtype=dtype, device=gpu)
        matLmHeadW = randn(config.vocab_size, HIDDEN, dtype=dtype, device=gpu)
    else:
        matqWs = torch.stack(
            [permute_rope_weight(layer.self_attn.q_proj.weight, NUM_Q_HEAD, HEAD_DIM, HIDDEN) for layer in layers],
            dim=0,
        ).contiguous()
        matkWs = torch.stack(
            [permute_rope_weight(layer.self_attn.k_proj.weight, NUM_KV_HEAD, HEAD_DIM, HIDDEN) for layer in layers],
            dim=0,
        ).contiguous()
        matvWs = torch.stack([layer.self_attn.v_proj.weight for layer in layers], dim=0).contiguous()
        matOutWs = torch.stack([layer.self_attn.o_proj.weight for layer in layers], dim=0).contiguous()
        matRouterWs = torch.stack([layer.mlp.gate.weight for layer in layers], dim=0).contiguous()
        matExpertGateWs = torch.stack(
            [layer.mlp.experts.gate_up_proj[:, :MOE_INTERMEDIATE, :] for layer in layers],
            dim=0,
        ).contiguous()
        matExpertUpWs = torch.stack(
            [layer.mlp.experts.gate_up_proj[:, MOE_INTERMEDIATE:, :] for layer in layers],
            dim=0,
        ).contiguous()
        matExpertDownWs = torch.stack(
            [layer.mlp.experts.down_proj for layer in layers],
            dim=0,
        ).contiguous()
        matLmHeadW = model.lm_head.weight.detach()

    vocab_size = matLmHeadW.shape[0]
    logits_slice = 64 * full_sms * 6
    logits_epoch = (vocab_size + logits_slice - 1) // logits_slice
    matLogits = []
    matLogitsW = []
    matLmHeadW.resize_(logits_slice * logits_epoch, HIDDEN)
    matLmHeadW[vocab_size:, :].zero_()
    for i in range(logits_epoch):
        matLogitsW.append(matLmHeadW[i * logits_slice : (i + 1) * logits_slice])
        matLogits.append(torch.zeros(N, logits_slice, dtype=dtype, device=gpu))

    matArgmaxIdx = torch.zeros(N, full_sms, dtype=torch.long, device=gpu)
    matArgmaxVal = torch.zeros(N, full_sms, dtype=dtype, device=gpu)

    dae.set_persistent(matTokens)
    if not parsed_args.local_generated_weights:
        dae.set_streaming(
            matqWs,
            matkWs,
            matvWs,
            matOutWs,
            matRouterWs,
            matExpertGateWs,
            matExpertUpWs,
            matExpertDownWs,
        )

    return QwenScheduleContext(
        parsed_args=parsed_args,
        gpu=gpu,
        use_local_generated_weights=parsed_args.local_generated_weights,
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
        MOE_INTERMEDIATE=MOE_INTERMEDIATE,
        TOP_K=TOP_K,
        EXPERT_BUFFER_COUNT=expert_buffer_count,
        NUM_EXPERTS=NUM_EXPERTS,
        HEAD_DIM=HEAD_DIM,
        NUM_Q_HEAD=NUM_Q_HEAD,
        NUM_KV_HEAD=NUM_KV_HEAD,
        HEAD_GROUP_SIZE=HEAD_GROUP_SIZE,
        QW=QW,
        KW=KW,
        VW=VW,
        num_layers=num_layers,
        dae=dae,
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
        matRouterLogits=matRouterLogits,
        matRouterTopKIdx=matRouterTopKIdx,
        matRouterTopKWeight=matRouterTopKWeight,
        matExpertAct=matExpertAct,
        matExpertActScaled=matExpertActScaled,
        matEmbed=matEmbed,
        matRMSInputW0=matRMSInputW0,
        matRMSInputWLoop=matRMSInputWLoop,
        matRMSPostAttnW=matRMSPostAttnW,
        matQNormWs=matQNormWs,
        matKNormWs=matKNormWs,
        matQwenSideInputs=matQwenSideInputs,
        matqWs=matqWs,
        matkWs=matkWs,
        matvWs=matvWs,
        matOutWs=matOutWs,
        matRouterWs=matRouterWs,
        matExpertGateWs=matExpertGateWs,
        matExpertUpWs=matExpertUpWs,
        matExpertDownWs=matExpertDownWs,
        vocab_size=vocab_size,
        logits_slice=logits_slice,
        logits_epoch=logits_epoch,
        matLogits=matLogits,
        matLogitsW=matLogitsW,
        matLmHeadWFull=matLmHeadW[:vocab_size],
        matArgmaxIdx=matArgmaxIdx,
        matArgmaxVal=matArgmaxVal,
    )


def seed_prefill_kv_cache(ctx: QwenScheduleContext):
    return None
