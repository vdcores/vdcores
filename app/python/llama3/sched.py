import torch
import torch.nn.functional as F
import argparse
import sys
from functools import partial
from dae.launcher import *
from dae.schedule import *
from dae.model import *
from dae.util import dae_app
from dae import runtime as dae_runtime
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
try:
  from transformers.cache_utils import StaticKVCache
except ImportError:
  from transformers.cache_utils import StaticCache as StaticKVCache
from reference import input_batch1, reference_pass, check_tensor_threshold
import os
import statistics
import time

DEFAULT_MAX_DECODE_STEPS = 128
CONTROL_FLOW_TOKENS_PER_LAUNCH = 1

arg_parser = argparse.ArgumentParser(add_help=False)
arg_parser.add_argument("-N", "--num-generates", type=int, default=None)
arg_parser.add_argument("--max-decode-steps", type=int, default=DEFAULT_MAX_DECODE_STEPS)
arg_parser.add_argument("--hf-cache-dir", default="/tmp/huggingface_cache")
arg_parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
arg_parser.add_argument("--correctness", action="store_true")
arg_parser.add_argument("--prompt", default=None)
arg_parser.add_argument("--message", action="append", default=None)
arg_parser.add_argument("--control-flow", dest="control_flow", action="store_true", default=True)
arg_parser.add_argument("--no-control-flow", dest="control_flow", action="store_false")
parsed_args, remaining_argv = arg_parser.parse_known_args()

positional_prompt = None
if remaining_argv and not remaining_argv[0].startswith("-"):
  positional_prompt = remaining_argv[0]
  remaining_argv = remaining_argv[1:]
if positional_prompt is not None:
  if parsed_args.prompt is not None:
    raise ValueError("Use either positional prompt or --prompt, not both")
  parsed_args.prompt = positional_prompt
if parsed_args.prompt is not None and parsed_args.message:
  raise ValueError("Use prompt text or --message, not both")

def dae_execution_requested(argv):
  return any(
    arg in ("-l", "--launch", "-b", "--bench")
    or arg.startswith("--bench=")
    for arg in argv
  )

def dae_work_requested(argv):
  return any(
    arg in ("-l", "--launch", "-b", "--bench", "-i", "--instdump", "-w", "--write-compute-ops")
    or arg.startswith("--bench=")
    or arg.startswith("--instdump=")
    or arg.startswith("--write-compute-ops=")
    for arg in argv
  )

has_user_prompt = parsed_args.prompt is not None or bool(parsed_args.message)
if parsed_args.correctness and not dae_execution_requested(remaining_argv):
  remaining_argv = [*remaining_argv, "--launch"]
elif has_user_prompt and not dae_work_requested(remaining_argv):
  remaining_argv = [*remaining_argv, "--launch"]
will_execute = dae_execution_requested(remaining_argv)
benchmark_mode = any(
  arg in ("-b", "--bench") or arg.startswith("--bench=")
  for arg in remaining_argv
)
sys.argv = [sys.argv[0], *remaining_argv]

def dae_execution_iterations(argv):
  for idx, arg in enumerate(argv):
    if arg == "-l" or arg == "--launch":
      return 1
    if arg.startswith("--bench="):
      return int(arg.split("=", 1)[1])
    if arg == "-b" or arg == "--bench":
      if idx + 1 < len(argv) and not argv[idx + 1].startswith("-"):
        return int(argv[idx + 1])
      return 1
  return 0

###################################
# load model
###################################

model_name = parsed_args.model
cache_dir = parsed_args.hf_cache_dir
hf_token = os.environ.get("HF_TOKEN")
auth_kwargs = {"token": hf_token} if hf_token else {}

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    dtype=torch.bfloat16,
    device_map="auto",
    **auth_kwargs,
)
config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir, **auth_kwargs)
eps = config.rms_norm_eps # 1e-6
rope_theta = config.rope_parameters["rope_theta"]

layers = model.model.layers
tokenizer = AutoTokenizer.from_pretrained(
  model_name,
  cache_dir=cache_dir,
  **auth_kwargs,
)

def normalize_token_ids(tokens, *, add_special_tokens=False):
  if isinstance(tokens, str):
    tokens = tokenizer(tokens, add_special_tokens=add_special_tokens)["input_ids"]
  elif hasattr(tokens, "input_ids"):
    tokens = tokens.input_ids
  elif isinstance(tokens, dict):
    tokens = tokens["input_ids"]
  elif torch.is_tensor(tokens):
    tokens = tokens.detach().cpu().tolist()
  if tokens and isinstance(tokens[0], list):
    tokens = tokens[0]
  return [int(token_id) for token_id in tokens]

def prompt_token_ids():
  if parsed_args.message:
    messages = [{"role": "user", "content": message} for message in parsed_args.message]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
      tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
      )
    else:
      tokens = tokenizer("\n".join(parsed_args.message), add_special_tokens=True)["input_ids"]
    tokens = normalize_token_ids(tokens)
    if not tokens:
      raise ValueError("--message produced no tokens")
    print(f"[prompt] tokenized {len(parsed_args.message)} user message(s) to {len(tokens)} tokens")
    return tokens

  if parsed_args.prompt is None:
    return [791]
  tokens = normalize_token_ids(tokenizer(parsed_args.prompt, add_special_tokens=True))
  if not tokens:
    raise ValueError("--prompt produced no tokens")
  print(f"[prompt] tokenized user prompt to {len(tokens)} tokens")
  return tokens

def eos_token_ids():
  ids = set()
  for value in (tokenizer.eos_token_id, getattr(config, "eos_token_id", None)):
    if value is None:
      continue
    if isinstance(value, (list, tuple, set)):
      ids.update(int(token_id) for token_id in value)
    else:
      ids.add(int(value))
  return ids

def trim_after_eos(token_ids):
  eos_ids = eos_token_ids()
  if not eos_ids:
    return token_ids
  for idx, token_id in enumerate(token_ids):
    if int(token_id) in eos_ids:
      return token_ids[:idx + 1]
  return token_ids

###################################
# basic parameter of DAE
###################################

gpu = torch.device("cuda")


REQ, N = 8, 8
MAX_SEQ_LEN = 512
KVBlockSize = 128

rms_sms = REQ
num_sms = 128
blackwell_sms = 152
blackwell_aux_sms = blackwell_sms - num_sms
# Shard-local MLP readiness is the qualified Blackwell default.  Keep the
# coarse frontier as an A/B fallback without changing any compute opcode.
fine_mlp_barriers = os.environ.get("VDCORES_FINE_MLP_BARRIERS", "1") == "1"
packed_silu_shards = (
  fine_mlp_barriers
  and os.environ.get("VDCORES_PACKED_SILU_SHARDS", "1") == "1"
)
stage_profile = os.environ.get("VDCORES_STAGE_PROFILE", "0") == "1"
track_profile = os.environ.get("VDCORES_TRACK_PROFILE", "0") == "1"
interleave_down_high = (
  fine_mlp_barriers
  and os.environ.get("VDCORES_INTERLEAVE_DOWN_HIGH", "1") == "1"
)
fused_qk_rope = os.environ.get("VDCORES_FUSED_QK_ROPE", "1") == "1"
qkv_head_barriers = os.environ.get("VDCORES_QKV_HEAD_BARRIERS", "1") == "1"
q_fold1_aux_tail = (
  qkv_head_barriers
  and os.environ.get("VDCORES_Q_FOLD1_AUX_TAIL", "1") == "1"
)
v_k_tail = (
  q_fold1_aux_tail
  and os.environ.get("VDCORES_V_K_TAIL", "1") == "1"
)
v_k_tail_heads = (3, 6, 7) if v_k_tail else ()
if qkv_head_barriers and not fused_qk_rope:
  raise ValueError("per-head QKV barriers require fused Q/K RoPE tasks")
phased_attn_out = os.environ.get("VDCORES_PHASED_ATTN_OUT", "1") == "1"
attn_out_head_groups = ((0,), (1,), tuple(range(2, 8)))
device_sms = torch.cuda.get_device_properties(gpu).multi_processor_count
full_sms = int(os.environ.get("VDCORES_LLAMA_SMS", str(blackwell_sms)))
if full_sms > device_sms:
  raise RuntimeError(f"Llama schedule requested {full_sms} SMs, but the device has {device_sms}")
if full_sms != blackwell_sms:
  raise RuntimeError(
    f"the tuned Llama-3.1-8B schedule requires exactly {blackwell_sms} SMs, got {full_sms}"
  )
aux_sms = int(os.environ.get("VDCORES_LLAMA_AUX_SMS", str(blackwell_aux_sms)))
if aux_sms != blackwell_aux_sms:
  raise RuntimeError(
    f"the tuned schedule requires {blackwell_aux_sms} auxiliary SMs, got {aux_sms}"
  )
dae = Launcher(full_sms, device=gpu)


prompt_tokens = prompt_token_ids()
prefill_token_id_and_pos = [(token, pos) for pos, token in enumerate(prompt_tokens[:-1])]
input_token_id_and_pos = [(prompt_tokens[-1], len(prefill_token_id_and_pos))]
initial_decode_pos = input_token_id_and_pos[0][1]
tokens_until_kv_block_end = KVBlockSize - (initial_decode_pos % KVBlockSize)
def decode_steps_up_to_limit(limit: int):
  if limit <= 0:
    raise ValueError("--max-decode-steps must be positive")
  available_decode_tokens = MAX_SEQ_LEN - initial_decode_pos
  if available_decode_tokens <= 0:
    raise ValueError(
      f"prompt leaves no decode room: initial_decode_pos={initial_decode_pos}, MAX_SEQ_LEN={MAX_SEQ_LEN}"
    )
  target = min(limit, available_decode_tokens)
  return target

if parsed_args.num_generates is not None:
  if parsed_args.control_flow:
    total_decode_tokens = decode_steps_up_to_limit(parsed_args.num_generates)
  else:
    total_decode_tokens = parsed_args.num_generates
elif parsed_args.correctness and not parsed_args.control_flow:
  total_decode_tokens = 1
elif parsed_args.control_flow:
  total_decode_tokens = decode_steps_up_to_limit(parsed_args.max_decode_steps)
else:
  total_decode_tokens = tokens_until_kv_block_end
if total_decode_tokens <= 0:
  raise ValueError("decode step count must be positive")
num_generates = total_decode_tokens - len(input_token_id_and_pos)
final_decode_pos = input_token_id_and_pos[0][1] + total_decode_tokens - 1
if final_decode_pos >= MAX_SEQ_LEN:
  raise ValueError(
    f"decode position {final_decode_pos} exceeds MAX_SEQ_LEN={MAX_SEQ_LEN}"
  )
if parsed_args.control_flow and len(input_token_id_and_pos) != 1:
  raise ValueError("--control-flow currently supports exactly one initial token")
print(
  "[decode] "
  f"prefill_tokens={len(prefill_token_id_and_pos)}, "
  f"decode_steps={total_decode_tokens}, "
  f"final_position={final_decode_pos}, "
  f"KVBlockSize={KVBlockSize}"
)

dtype = model.dtype
HIDDEN = config.hidden_size
INTERMIDIATE = config.intermediate_size
HEAD_DIM = HIDDEN // config.num_attention_heads
QW = HEAD_DIM * config.num_attention_heads
KW = HEAD_DIM * config.num_key_value_heads
VW = HEAD_DIM * config.num_key_value_heads
num_layers = len(layers)


###################################
# Define groups, barriers and TMA for scheduling
###################################

defaultg = dae.get_group()
layerg = dae.add_group("layer", num_layers)
systemg = dae.add_group("system", 1)

defaultg.addBarrier('bar_embedding', N)

systemg.addBarrier('bar_argmax_partial')
systemg.addBarrier('bar_token_finish') # argmax plus restore-barrier copy after placement

layerg.addBarrier('bar_layer')
layerg.addBarrier('bar_out_mlp')
if not qkv_head_barriers:
  layerg.addBarrier('bar_q_proj')
  layerg.addBarrier('bar_qkv_attn')
if phased_attn_out:
  for group_id, heads in enumerate(attn_out_head_groups):
    layerg.addBarrier(f'bar_attn_out_group{group_id}', N * len(heads))
else:
  layerg.addBarrier('bar_attn_out')
  layerg.addBarrier('bar_q_clear')
if not fine_mlp_barriers:
  layerg.addBarrier('bar_silu_in')
  layerg.addBarrier('bar_silu_out1')
layerg.addBarrier('bar_silu_out2')
layerg.addBarrier('bar_pre_attn_rms')
layerg.addBarrier('bar_post_attn_rms')
if qkv_head_barriers:
  for head in range(8):
    layerg.addBarrier(f'bar_q_proj_head{head}')
    layerg.addBarrier(f'bar_qkv_attn_head{head}')
if fine_mlp_barriers:
  for shard_id in range(3):
    layerg.addBarrier(f'bar_silu_in{shard_id}')
    layerg.addBarrier(f'bar_silu_out1_{shard_id}')

###################################
# Define tensors
###################################

matZero = torch.zeros(4096, dtype=dtype, device=gpu)

_positions = torch.arange(MAX_SEQ_LEN).unsqueeze(0).to(gpu) # [1, seq]
_cos, _sin = model.model.rotary_emb(torch.zeros(1, device=gpu), _positions) # tensor only device matters here
matRope = torch.ones(MAX_SEQ_LEN, N, HEAD_DIM, dtype=torch.bfloat16, device=gpu)
matRope[:, :, 0::2] = _cos[0, :, :HEAD_DIM//2].unsqueeze(1) # llama duplicate it to full dim
matRope[:, :, 1::2] = _sin[0, :, :HEAD_DIM//2].unsqueeze(1)
matRopeFused = matRope[:, 0, :].contiguous()

# Keep one extra slot for the token produced after decoding position
# MAX_SEQ_LEN - 1; KV/RoPE storage itself remains capped at MAX_SEQ_LEN.
matTokens = torch.zeros(N, MAX_SEQ_LEN + 1, dtype=torch.int64, device=gpu)
matHidden = torch.rand(N, HIDDEN, dtype=dtype, device=gpu) - 0.5
matRMSHidden = torch.rand(N, HIDDEN, dtype=dtype, device=gpu) - 0.5

# TODO(zhiyuang): use single Q across layer for multitoken
attnQs = [torch.zeros(REQ, HIDDEN, dtype=dtype, device=gpu) for _ in range(num_layers)]
attnKs = [torch.zeros(REQ, MAX_SEQ_LEN, KW, dtype=dtype, device=gpu) for _ in range(num_layers)]
attnVs = [torch.zeros(REQ, MAX_SEQ_LEN, VW, dtype=dtype, device=gpu) for _ in range(num_layers)]
attnO = torch.zeros(REQ, HIDDEN, dtype=dtype, device=gpu)
matInterm = torch.zeros(N, INTERMIDIATE, dtype=dtype, device=gpu)
matGateOut = torch.zeros(N, INTERMIDIATE, dtype=dtype, device=gpu)
matSiLUOut = torch.zeros(N, INTERMIDIATE, dtype=dtype, device=gpu)

# embedding table
matEmbed = model.model.embed_tokens.weight

# RMS
# reorder the RMS weights, append the post-attn rms to the last
matRMSInputW = [l.input_layernorm.weight for l in layers] + [model.model.norm.weight]
matRMSPostAttnW = [l.post_attention_layernorm.weight for l in layers]

# QKV proj
def permute_rope_weight(weight, num_heads):
  return weight.view(num_heads, 2, HEAD_DIM // 2, HIDDEN).transpose(1, 2).reshape_as(weight).contiguous()
def permute_rope_activation(activation, num_heads):
  return (
    activation.view(*activation.shape[:-1], num_heads, 2, HEAD_DIM // 2)
    .transpose(-2, -1)
    .reshape_as(activation)
    .contiguous()
  )

def apply_interleaved_rope_activation(activation, num_heads, rope_row):
  states = activation.view(*activation.shape[:-1], num_heads, HEAD_DIM).float()
  cos = rope_row[0::2].float()
  sin = rope_row[1::2].float()
  even = states[..., 0::2]
  odd = states[..., 1::2]
  rotated = torch.stack(
    (even * cos - odd * sin, even * sin + odd * cos),
    dim=-1,
  ).flatten(-2)
  return rotated.reshape_as(activation).to(dtype=activation.dtype)

matqWs = [permute_rope_weight(l.self_attn.q_proj.weight, QW // HEAD_DIM) for l in layers]
matkWs = [permute_rope_weight(l.self_attn.k_proj.weight, KW // HEAD_DIM) for l in layers]
matvWs = [l.self_attn.v_proj.weight for l in layers]

# Attn out proj
matOutWs = [l.self_attn.o_proj.weight for l in layers]

matUps = [l.mlp.up_proj.weight for l in layers]
matGates = [l.mlp.gate_proj.weight for l in layers]
matDowns = [l.mlp.down_proj.weight for l in layers]

logits_fold = 8
logits_slice = 8192 * logits_fold
logits_epoch = 2

matLogitsW = []
matLmHeadW = model.lm_head.weight.detach()
vocab_size = matLmHeadW.shape[0]

# Pad to two 65,536-row epochs for the fixed 128-SM logits reduction.
matLmHeadW.resize_(logits_slice * logits_epoch, 4096)
matLmHeadW[vocab_size:,].zero_() # zero padding

matArgmaxPartial = torch.empty(N, 256, 16, dtype=torch.uint8, device=gpu)

for i in range(logits_epoch):
  matLogitsW.append(matLmHeadW[i * logits_slice: (i+1) * logits_slice])

# tensor cache policy
dae.set_persistent(matTokens)

def seed_prefill_kv_cache():
  for layer_k, layer_v in zip(attnKs, attnVs):
    layer_k.zero_()
    layer_v.zero_()

  if not prefill_token_id_and_pos:
    return None

  prefill_tokens = [token for token, _ in prefill_token_id_and_pos]
  prefill_positions = [pos for _, pos in prefill_token_id_and_pos]
  expected_positions = list(range(len(prefill_tokens)))
  if prefill_positions != expected_positions:
    raise ValueError(
      "PyTorch StaticKVCache prefill currently expects contiguous prompt positions "
      f"{expected_positions}, got {prefill_positions}"
    )

  inputs = input_batch1(
    *prefill_tokens,
    mat=matTokens[0],
    positions=prefill_positions,
  )
  cache_position = torch.arange(len(prefill_tokens), dtype=torch.long, device=gpu)
  static_cache = StaticKVCache(config=config, max_cache_len=MAX_SEQ_LEN)
  with torch.no_grad():
    output = model(
      **inputs,
      use_cache=True,
      past_key_values=static_cache,
      cache_position=cache_position,
    )

  cache = output.past_key_values
  prefill_len = len(prefill_tokens)
  for layer_idx in range(num_layers):
    layer_cache = cache.layers[layer_idx]
    k_cache = layer_cache.keys[0, :, :prefill_len, :].permute(1, 0, 2).reshape(prefill_len, KW)
    v_cache = layer_cache.values[0, :, :prefill_len, :].permute(1, 0, 2).reshape(prefill_len, VW)
    k_cache = permute_rope_activation(k_cache, KW // HEAD_DIM)
    attnKs[layer_idx][:, :prefill_len].copy_(k_cache.unsqueeze(0).expand(REQ, -1, -1))
    attnVs[layer_idx][:, :prefill_len].copy_(v_cache.unsqueeze(0).expand(REQ, -1, -1))

  print(f"[prefill] seeded {prefill_len} prompt tokens with {type(static_cache).__name__}")
  return output

###################################
# Register Tensor for TMA
###################################

QKVAtom = Gemv_M64N8IssuerOnly
LinearAtom = Gemv_M64N8IssuerOnly
OutAtom = Gemv_M128N8
LogitsAtom = Gemv_M128N8Argmax4
QKVTileM, _, QKVTileK = QKVAtom.MNK
LinearTileM, _, LinearTileK = LinearAtom.MNK
OutTileM, _, OutTileK = OutAtom.MNK
LogitsTileM, _, LogitsTileK = LogitsAtom.MNK

print(
  f"[weights] packing QKV/down as M{LinearTileM}K{LinearTileK} and "
  f"output as M{OutTileM}K{OutTileK} tiles"
)
matqWs = [pack_weight_tile_major(weight, QKVTileM, QKVTileK) for weight in matqWs]
matkWs = [pack_weight_tile_major(weight, QKVTileM, QKVTileK) for weight in matkWs]
matvWs = [pack_weight_tile_major(weight, QKVTileM, QKVTileK) for weight in matvWs]
matOutWs = [pack_weight_tile_major(weight, OutTileM, OutTileK) for weight in matOutWs]
matUps = [pack_weight_tile_major(weight, QKVTileM, QKVTileK) for weight in matUps]
matGates = [pack_weight_tile_major(weight, QKVTileM, QKVTileK) for weight in matGates]
matDowns = [pack_weight_tile_major(weight, LinearTileM, LinearTileK) for weight in matDowns]
matLogitsW = [
  pack_weight_tile_major(weight.contiguous(), LogitsTileM, LogitsTileK)
  for weight in matLogitsW
]
dae.set_streaming(
  matqWs,
  matkWs,
  matvWs,
  matOutWs,
  matUps,
  matGates,
  matDowns,
  matLogitsW,
)


def weight_load_tma(tma_tensor, tile_m, tile_k):
  return tma_tensor.wgmma_load_tiled(tile_m, tile_k)


def linear_output_tma(mode, tile_m=LinearTileM):
  return lambda t: t.wgmma(mode, N, tile_m, Major.MN)


defaultg.addTma(
  "loadLogitsB",
  [matRMSHidden],
  lambda t: t.wgmma_load(N, LogitsTileK * LogitsAtom.n_batch, Major.K),
)
for logits_idx in range(logits_epoch):
  defaultg.addTma(
    f"loadLogitsW{logits_idx}",
    [matLogitsW[logits_idx]],
    lambda t: t.wgmma_load_tiled(LogitsTileM, LogitsTileK),
  )


defaultg.addTma("loadRope", [matRope], lambda t: t._build("load", QKVTileM, N, tma_load_tbl, cord_load_tbl))

# load tmas for the same matrix for "grouped" instructions
layerg.addTma("loadRMSLayer", [matRMSHidden] * num_layers, lambda t: t.wgmma_load(N, QKVTileK * QKVAtom.n_batch, Major.K))
layerg.addTma("reduceHiddenLayer", [matHidden] * num_layers, linear_output_tma("reduce"))
layerg.addTma("reduceHiddenOutLayer", [matHidden] * num_layers, linear_output_tma("reduce", OutTileM))
layerg.addTma("loadSiluLayer", [matSiLUOut] * num_layers, lambda t: t.wgmma_load(N, LinearTileK * LinearAtom.n_batch, Major.K))
layerg.addTma("storeSiluLayer", [matSiLUOut] * num_layers, lambda t: t.wgmma_store(N, QKVTileM, Major.MN))
layerg.addTma("loadAttnOLayer", [attnO] * num_layers, lambda t: t.wgmma_load(N, OutTileK * OutAtom.n_batch, Major.K))

layerg.addTma("storeInterm", [matInterm] * num_layers, lambda t: t.wgmma_store(N, QKVTileM, Major.MN))
layerg.addTma("storeGateOut", [matGateOut] * num_layers, lambda t: t.wgmma_store(N, QKVTileM, Major.MN))

# RMS, skip the first one which is used for embedding fusion
layerg.addTma("loadRMSInputW", matRMSInputW[1:], lambda t: t.tensor1d("load", HIDDEN))
layerg.addTma("loadRMSPostAttnW", matRMSPostAttnW, lambda t: t.tensor1d("load", HIDDEN))

layerg.addTma("loadOutWs", matOutWs, lambda t: weight_load_tma(t, OutTileM, OutTileK))
layerg.addTma("loadDown", matDowns, lambda t: weight_load_tma(t, LinearTileM, LinearTileK))
layerg.addTma("loadUp", matUps, lambda t: weight_load_tma(t, QKVTileM, QKVTileK))
layerg.addTma("loadGate", matGates, lambda t: weight_load_tma(t, QKVTileM, QKVTileK))

tma_builder_MN = partial(build_tma_wgmma_mn, iK = -3)
cord_func_MN = partial(cord_func_MN_major, iK=-3)
cord_func_MN_cord2 = partial(cord_func_MN_major_cord2, iK=-3)

tma_builder_K = partial(build_tma_wgmma_k, iN = -3)
cord_func_K = partial(cord_func_K_major, iN=-3)

layerg.addTma("loadQW", matqWs, lambda t: weight_load_tma(t, QKVTileM, QKVTileK))
layerg.addTma("loadKW", matkWs, lambda t: weight_load_tma(t, QKVTileM, QKVTileK))
layerg.addTma("loadVW", matvWs, lambda t: weight_load_tma(t, QKVTileM, QKVTileK))
layerg.addTma("storeQ", attnQs, lambda t: t.wgmma("reduce", N, QKVTileM, Major.MN))
q_clear_targets = attnQs[-1:] + attnQs[:-1]
layerg.addTma("storeQClear", q_clear_targets, lambda t: t.wgmma_store(N, QKVTileM, Major.MN))
layerg.addTma("storeK", attnKs, lambda t: t._build("reduce", 64, N, tma_store_attn_kv, cord_id))
layerg.addTma("storeV", attnVs, lambda t: t._build("reduce", 64, N, tma_store_attn_kv, cord_id))

HEAD_DIM = ATTENTION_M64N64K16_F16_F32_64_64_hdim.HEAD_DIM
NUM_KV_HEAD = KW // HEAD_DIM
HEAD_GROUP_SIZE = QW // KW
matQ_attn_views = [attnQ.view(N, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM) for attnQ in attnQs]
matK_attn_views = [attnK.view(N, MAX_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM) for attnK in attnKs]
matV_attn_views = [attnV.view(N, MAX_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM) for attnV in attnVs]
matO_attn_view = attnO.view(N, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)

layerg.addTma('loadQ', matQ_attn_views, lambda t: t._build("load", HEAD_DIM, 8, tma_gqa_load_q, cord_gqa_load_q))
layerg.addTma('loadK', matK_attn_views, lambda t: t._build("load", HEAD_DIM, KVBlockSize, tma_builder_K, cord_func_K))
layerg.addTma('loadV', matV_attn_views, lambda t: t._build("load", HEAD_DIM, KVBlockSize, tma_builder_MN, cord_func_MN))

###################################
# Finish building resources
###################################

dae.build_groups()
tma_shift, bar_shift = layerg.get_shift()

###################################
# Start of Schedule
###################################

TOKEN_LOOP_REG = 1
TOKEN_BASE_REG = 2
KV_BLOCK_COUNT_REG = 3

stage_profile_events = {}


def stage_profile_marker(name: str, active_sms=None):
  if not stage_profile:
    return None
  if name not in stage_profile_events:
    event_id = 2 + len(stage_profile_events)
    if event_id >= dae_runtime.config.num_profile_events:
      raise RuntimeError("Llama stage profile exhausted the runtime event buffer")
    if active_sms is None:
      active_sms = tuple(range(full_sms))
    else:
      active_sms = tuple(active_sms)
    stage_profile_events[name] = (event_id, active_sms)
  return ProfileEvent(stage_profile_events[name][0])


def stage_profile_schedule_parts(prefix: str, schedules):
  if not stage_profile:
    return schedules
  profiled = []
  for part_id, schedule in enumerate(schedules):
    profiled.extend([
      schedule,
      stage_profile_marker(
        f"{prefix}{part_id}",
        getattr(
          schedule,
          "profile_sms",
          range(schedule.base_sm, schedule.base_sm + schedule.num_sms),
        ),
      ),
    ])
  return profiled


class CounterOffsetCordAdapter(CordAdapter):
  def __init__(self, inner, offsets):
    super().__init__(inner)
    self.offsets = offsets

  def cord(self, *cords):
    inst = self.inner.cord(*cords)
    for counter_reg, delta in self.offsets:
      inst = CounterOffsetMemoryInstruction(counter_reg, inst, delta)
    return inst


class SchedLmHeadEpoch1TailOffload(Schedule):
  """Move logical LM tasks 96--103 onto physical SMs 128--135."""

  def __init__(self, inner):
    super().__init__()
    self.inner = inner
    self.profile_sms = (*range(96), *range(104, 136))

  def _on_place(self):
    if self.num_sms != full_sms or self.inner.num_sms != num_sms:
      raise ValueError("LM tail offload requires the 128-on-152 placement")

  def schedule(self, sm: int):
    if 0 <= sm < 96 or 104 <= sm < 128:
      return self.inner.schedule(sm)
    if 128 <= sm < 136:
      return self.inner.schedule(sm - 32)
    return []

  def collect_barrier_release_counts(self):
    return self.inner.collect_barrier_release_counts()


class SchedClearQ(Schedule):
  def __init__(self, load_zero, store_q, tile_bytes: int, tile_m: int,
               num_clear_sms: int, wait_bars):
    super().__init__()
    self.load_zero = load_zero
    self.store_q = store_q
    self.tile_bytes = tile_bytes
    self.tile_m = tile_m
    self.num_clear_sms = num_clear_sms
    self.wait_bars = (
      list(wait_bars) if isinstance(wait_bars, (list, tuple))
      else [wait_bars]
    )

  def schedule(self, sm: int):
    if sm < 0:
      return []
    count = (HIDDEN // self.tile_m + self.num_clear_sms - 1 - sm) // self.num_clear_sms
    if count <= 0:
      return []
    store = self.store_q.cord(sm)
    if self._bar("store") is not None:
      store = store.bar(self._bar("store")).group()
    finish = (
      IssueBarrier(self._bar("store")).group()
      if self._bar("store") is not None else None
    )
    return [
      [IssueBarrier(wait_bar).group() for wait_bar in self.wait_bars],
      Copy(count, size=self.tile_bytes),
      RepeatM.on(
        count,
        (self.load_zero.cord(0), 0),
        (
          store,
          [self.num_clear_sms * self.tile_m, 0],
        ),
      ),
      finish,
    ]

  def bar_release_count(self, role: str):
    if role != "store":
      return 0
    return self._bar_release_if_present(role, HIDDEN // self.tile_m)


def _control_flow_offsets(token_delta, base_delta=None):
  offsets = [(TOKEN_LOOP_REG, token_delta)]
  if base_delta is not None:
    offsets.append((TOKEN_BASE_REG, base_delta))
  return offsets


def maybe_counter_offset(inst, delta, enabled: bool, base_delta=None):
  if not enabled:
    return inst
  for counter_reg, counter_delta in _control_flow_offsets(delta, base_delta):
    inst = CounterOffsetMemoryInstruction(counter_reg, inst, counter_delta)
  return inst


def maybe_counter_adapter(adapter, delta, enabled: bool, base_delta=None):
  if not enabled:
    return adapter
  return CounterOffsetCordAdapter(adapter, _control_flow_offsets(delta, base_delta))


def schedule_single_token(
    token_offset: int,
    token_pos: int,
    *,
    control_flow_tokens: int | None = None,
    token_loop_ptrs=None,
):
  control_flow = control_flow_tokens is not None
  base_counter_delta = 1 if control_flow else None
  # RMS
  # group is not working on RMS tmas, as it uses TMA1D
  # TODO(zhiyuang): dup the matEmbed for now
  loadEmbed1D = TmaLoad1D(matEmbed, bytes=HIDDEN * 2)
  storeHidden1D = TmaStore1D(matHidden, bytes=HIDDEN * 2)
  loadHidden1D = TmaLoad1D(matHidden, bytes=HIDDEN * 2)
  storeRMSHidden1D = TmaStore1D(matRMSHidden, bytes=HIDDEN * 2)

  embed_rms = SchedRMSShared(
    num_token=N, epsilon=eps,
    tmas=(TmaLoad1D(matRMSInputW[0]), loadEmbed1D, storeRMSHidden1D),
    embedding=maybe_counter_offset(
      CC0(matTokens[0], token_offset, hidden_size=HIDDEN),
      matTokens.element_size(),
      control_flow,
      base_counter_delta * matTokens.element_size() if control_flow else None,
    )
  ).bar("output", layerg['bar_pre_attn_rms'])
  # copy the HIDDEN from embedding
  copy_hidden = SchedCopy(
    size = HIDDEN * matHidden.element_size(),
    tmas = (
      StaticCordAdapter(loadEmbed1D),
      ToLinearCordAdapter(storeHidden1D, HIDDEN * 2),
    ),
    before_copy = maybe_counter_offset(
      CC0(matTokens[0], token_offset, hidden_size=HIDDEN),
      matTokens.element_size(),
      control_flow,
      base_counter_delta * matTokens.element_size() if control_flow else None,
    ),
  )

  pre_attn_rms = SchedRMSShared(
    num_token=N, epsilon=eps,
    tmas=(layerg['loadRMSInputW'].cord(0), loadHidden1D, storeRMSHidden1D)
  ).bar("input", layerg['bar_layer']).bar("output", layerg.next('bar_pre_attn_rms'))
  post_attn_rms = SchedRMSShared(
    num_token=N, epsilon=eps,
    tmas=(layerg['loadRMSPostAttnW'].cord(0), loadHidden1D, storeRMSHidden1D)
  ).bar("input", layerg['bar_out_mlp']).bar("output", layerg['bar_post_attn_rms'])

  # QKV projection and rotary handoff
  if fused_qk_rope:
    rope_counter_offsets = (
      [(TOKEN_BASE_REG, base_counter_delta * HEAD_DIM * matRopeFused.element_size())]
      if control_flow else []
    )
    if qkv_head_barriers:
      QProj = [
        SchedGemvRope(
          MNK=((head * 512, 512), N, (fold * 2048, 2048)),
          tmas=(layerg['loadQW'], layerg['loadRMSLayer'], layerg['storeQ']),
          rope_table=RawAddress(matRopeFused, dae_runtime.config.num_slots),
          hist_seq_len=token_pos,
          rope_counter_offsets=rope_counter_offsets,
        ).bar("load", layerg['bar_pre_attn_rms']).bar(
          "store", layerg[f'bar_q_proj_head{head}']
        ).place(
          8,
          base_sm=(
            num_sms + (head - 5) * 8
            if q_fold1_aux_tail and fold == 1 and head >= 5
            else fold * 64 + head * 8
          ),
        )
        for fold in range(2)
        for head in range(NUM_KV_HEAD)
      ]
    else:
      QProj = SchedGemvRope(
        MNK=(QW, N, HIDDEN),
        tmas=(layerg['loadQW'], layerg['loadRMSLayer'], layerg['storeQ']),
        rope_table=RawAddress(matRopeFused, dae_runtime.config.num_slots),
        hist_seq_len=token_pos,
        rope_counter_offsets=rope_counter_offsets,
      ).bar("load", layerg['bar_pre_attn_rms']).bar("store", layerg['bar_q_proj'])
    QRope = []
    k_store = maybe_counter_adapter(
      ToAttnVStoreCordAdapter(layerg['storeK'], token_pos),
      [0, 1, 0],
      control_flow,
      [0, base_counter_delta, 0] if control_flow else None,
    )
    if qkv_head_barriers:
      KProj = [
        SchedGemvRope(
          MNK=((head * HEAD_DIM, HEAD_DIM), N, HIDDEN),
          tmas=(layerg['loadKW'], layerg['loadRMSLayer'], k_store),
          rope_table=RawAddress(matRopeFused, dae_runtime.config.num_slots),
          hist_seq_len=token_pos,
          rope_counter_offsets=rope_counter_offsets,
        ).bar("store", layerg[f'bar_qkv_attn_head{head}']).place(
          8, base_sm=64 + head * 8
        )
        for head in range(NUM_KV_HEAD)
      ]
      if q_fold1_aux_tail:
        # K normally inherits the next-layer RMS acquisition from its
        # colocated Q fold.  Preserve that dependency when heads 5--7 move
        # their Q owner to the auxiliary CTAs.
        for head in range(5, NUM_KV_HEAD):
          KProj[head].bar("load", layerg['bar_pre_attn_rms'])
    else:
      KProj = SchedGemvRope(
        MNK=(KW, N, HIDDEN),
        tmas=(layerg['loadKW'], layerg['loadRMSLayer'], k_store),
        rope_table=RawAddress(matRopeFused, dae_runtime.config.num_slots),
        hist_seq_len=token_pos,
        rope_counter_offsets=rope_counter_offsets,
      ).bar("store", layerg['bar_qkv_attn'])
    KRope = []
  else:
    regStoreQ = RegStore(0, size=N * QKVTileM * matQ_attn_views[0].element_size())
    regLoadQ = RegLoad(0)
    QProj = SchedGemv(QKVAtom,
      MNK=(QW, N, HIDDEN),
      tmas=(layerg['loadQW'], layerg['loadRMSLayer'], regStoreQ),
    ).bar("load", layerg['bar_pre_attn_rms'])
    QRope = SchedRope(ROPE_INTERLEAVE_512,
      tmas=(
        maybe_counter_adapter(
          ToRopeTableCordAdapter(defaultg['loadRope'], token_pos),
          [0, 0, 0, 1],
          control_flow,
          [0, 0, 0, base_counter_delta] if control_flow else None,
        ),
        regLoadQ,
        ToSplitMCordAdapter(layerg['storeQ'], 128//2, QKVTileM),
      ),
    ).bar("store", layerg['bar_q_proj'])
    regStoreK = RegStore(0, size=N * QKVTileM * matK_attn_views[0].element_size())
    regLoadK = RegLoad(0)
    KProj = SchedGemv(QKVAtom,
      MNK=(KW, N, HIDDEN),
      tmas=(layerg['loadKW'], layerg['loadRMSLayer'], regStoreK),
    )
    KRope = SchedRope(ROPE_INTERLEAVE_512,
      tmas=(
        maybe_counter_adapter(
          ToRopeTableCordAdapter(defaultg['loadRope'], token_pos),
          [0, 0, 0, 1],
          control_flow,
          [0, 0, 0, base_counter_delta] if control_flow else None,
        ),
        regLoadK,
        maybe_counter_adapter(
          ToAttnKVStoreCordAdapter(layerg['storeK'], 64//4, QKVTileM, token_pos),
          [0, 1, 0],
          control_flow,
          [0, base_counter_delta, 0] if control_flow else None,
        ),
      ),
    ).bar("store", layerg['bar_qkv_attn'])
  v_store = maybe_counter_adapter(
    ToAttnVStoreCordAdapter(layerg['storeV'], token_pos),
    [0, 1, 0],
    control_flow,
    [0, base_counter_delta, 0] if control_flow else None,
  )
  if qkv_head_barriers:
    VProj = [
      SchedGemv(
        QKVAtom,
        MNK=((head * HEAD_DIM, HEAD_DIM), N, HIDDEN),
        tmas=(layerg['loadVW'], layerg['loadRMSLayer'], v_store),
      ).bar("store", layerg[f'bar_qkv_attn_head{head}']).place(
        8,
        base_sm=(
          104 + v_k_tail_heads.index(head) * 8
          if head in v_k_tail_heads else head * 8
        ),
      )
      for head in range(NUM_KV_HEAD)
    ]
    for head in v_k_tail_heads:
      VProj[head].bar("load", layerg['bar_pre_attn_rms'])
  else:
    VProj = SchedGemv(QKVAtom,
      MNK=(VW, N, HIDDEN),
      tmas=(layerg['loadVW'], layerg['loadRMSLayer'], v_store),
    ).bar("store", layerg['bar_qkv_attn'])

  Gqa = SchedAttentionDecoding(
    reqs = N, seq_len = token_pos + 1,
    KV_BLOCK_SIZE = KVBlockSize,
    NUM_KV_HEADS = NUM_KV_HEAD,
    matO = matO_attn_view,
    tmas = (layerg['loadQ'], layerg['loadK'], layerg['loadV']),
    seq_len_counter_reg=TOKEN_LOOP_REG if control_flow else None,
    num_kv_block_counter_reg=KV_BLOCK_COUNT_REG if control_flow else None,
    outer_seq_len_counter_reg=TOKEN_BASE_REG if control_flow else None,
    outer_seq_len_counter_stride=base_counter_delta or 0,
    num_active_q=HEAD_GROUP_SIZE,
    swapped_qk_pv=True,
    q_head_bars=(
      [layerg[f'bar_q_proj_head{head}'] for head in range(NUM_KV_HEAD)]
      if qkv_head_barriers else None
    ),
    kv_head_bars=(
      [layerg[f'bar_qkv_attn_head{head}'] for head in range(NUM_KV_HEAD)]
      if qkv_head_barriers else None
    ),
    o_head_bars=(
      [
        layerg[f'bar_attn_out_group{group_id}']
        for head in range(NUM_KV_HEAD)
        for group_id, heads in enumerate(attn_out_head_groups)
        if head in heads
      ]
      if phased_attn_out else None
    ),
    head_major=qkv_head_barriers,
    max_loop_count=min(
      control_flow_tokens or 1,
      KVBlockSize - (token_pos % KVBlockSize),
    ),
  )
  if not phased_attn_out:
    Gqa.bar("o", layerg['bar_attn_out'])
  if not qkv_head_barriers:
    Gqa.bar("q", layerg['bar_q_proj']).bar("k", layerg['bar_qkv_attn'])

  # accumulate to matHidden, which auto applies the residual add
  # Six M128 tiles use eight K512 folds and the other 26 use four K1024
  # folds: exactly 152 independent tasks, one for every Blackwell SM.
  out_tmas = (
    layerg['loadOutWs'], layerg['loadAttnOLayer'], layerg['reduceHiddenOutLayer']
  )
  if phased_attn_out:
    activation_bars = [
      layerg[f'bar_attn_out_group{group_id}']
      for head in range(NUM_KV_HEAD)
      for group_id, heads in enumerate(attn_out_head_groups)
      if head in heads
    ]
    def make_phased_out(mnk):
      return SchedGemvPhasedActivation(
        OutAtom, MNK=mnk, fold=1,
        tmas=out_tmas, activation_bars=activation_bars,
      ).bar("store", layerg['bar_out_mlp'])

    # Six M128 rows use eight K512 folds and the other 26 use four K1024
    # folds: 48 + 104 tasks, exactly one per physical SM. K<2048 stays on the
    # 76 CTAs outside attention; late-K uses the complementary physical set.
    out_proj_placement_specs = [
      *[
        (make_phased_out(((0, 768), N, (head * 512, 512))),
         6, 64 + head * 6)
        for head in range(4)
      ],
      (make_phased_out(((768, HIDDEN - 768), N, (0, 1024))), 26, 88),
      (make_phased_out(((768, HIDDEN - 768), N, (1024, 1024))), 26, 114),
      *[
        (make_phased_out(((0, 768), N, (2048 + head * 512, 512))),
         6, head * 6)
        for head in range(4)
      ],
      (make_phased_out(((768, 2560), N, (2048, 1024))), 20, 24),
      (make_phased_out(((768, 2560), N, (3072, 1024))), 20, 44),
      (make_phased_out(((3328, 768), N, (2048, 1024))), 6, 140),
      (make_phased_out(((3328, 768), N, (3072, 1024))), 6, 146),
    ]
  else:
    out_proj_fold8 = SchedGemv(
      OutAtom, MNK=((0, 768), N, HIDDEN), fold=8, tmas=out_tmas,
    ).bar("load", layerg['bar_attn_out']).bar("store", layerg['bar_out_mlp'])
    out_proj_fold4 = SchedGemv(
      OutAtom, MNK=((768, HIDDEN - 768), N, HIDDEN), fold=4, tmas=out_tmas,
    ).bar("load", layerg['bar_attn_out']).bar("store", layerg['bar_out_mlp'])
    out_proj_placement_specs = None

  # Gate Up + SiLU
  reg_gate = 0
  reg_store_gate = RegStore(reg_gate, matGateOut[:, :QKVTileM])
  legacy_silu_in = layerg['bar_silu_in'] if not fine_mlp_barriers else None
  legacy_silu_out1 = layerg['bar_silu_out1'] if not fine_mlp_barriers else None

  gate_proj_prefix = SchedGemv(QKVAtom,
    MNK=(6144, N, HIDDEN),
    tmas=(layerg['loadGate'], layerg['loadRMSLayer'], layerg['storeGateOut']),
  ).bar("load", layerg['bar_post_attn_rms']).bar("store", legacy_silu_in)
  up_proj_main = SchedGemv(QKVAtom,
    MNK=(2048, N, HIDDEN),
    tmas=(layerg['loadUp'], layerg['loadRMSLayer'], layerg['storeInterm']),
  ).bar("load", layerg['bar_post_attn_rms']).bar("store", legacy_silu_in)
  up_proj_aux0 = SchedGemv(QKVAtom,
    MNK=((2048, 1536), N, HIDDEN),
    tmas=(layerg['loadUp'], layerg['loadRMSLayer'], layerg['storeInterm']),
  ).bar("load", layerg['bar_post_attn_rms']).bar("store", legacy_silu_in)
  up_proj_aux1 = SchedGemv(QKVAtom,
    MNK=((3584, 1536), N, HIDDEN),
    tmas=(layerg['loadUp'], layerg['loadRMSLayer'], layerg['storeInterm']),
  ).bar("store", legacy_silu_in)
  up_proj_aux2 = SchedGemv(QKVAtom,
    MNK=((5120, 1024), N, HIDDEN),
    tmas=(layerg['loadUp'], layerg['loadRMSLayer'], layerg['storeInterm']),
  ).bar("store", legacy_silu_in)

  mlp_split = 6144
  silu1 = SchedSmemSiLUInterleaved(
    num_token=N,
    gate_glob=matGateOut[:, :mlp_split],
    up_glob=matInterm[:, :mlp_split],
    out_glob=matSiLUOut[:, :mlp_split],
    shards_per_token=3,
  ).bar("input", legacy_silu_in).bar("output", legacy_silu_out1)
  gate_proj_tail = SchedGemv(QKVAtom,
    MNK=((mlp_split, INTERMIDIATE - mlp_split), N, HIDDEN),
    tmas=(layerg['loadGate'], layerg['loadRMSLayer'], reg_store_gate),
  )
  up_silu_tail = SchedGemvUpSiLU(
    MNK=((mlp_split, INTERMIDIATE - mlp_split), N, HIDDEN),
    tmas=(
      layerg['loadUp'],
      layerg['loadRMSLayer'],
      layerg['storeSiluLayer'],
    ),
    gate_reg=reg_gate,
  ).bar("store", layerg['bar_silu_out2'])
  # The two low-K schedules contribute 192 fold-3 tasks and the two high-K
  # schedules contribute 256 fold-4 tasks.  Their placement uses all 152 SMs;
  # the final high-K tail is moved separately below based on absolute-path
  # readiness rather than uniform task count.
  down_proj_low0 = SchedGemv(LinearAtom,
    MNK=((0, 3072), N, 6144),
    fold=3,
    tmas=(layerg['loadDown'], layerg['loadSiluLayer'], layerg['reduceHiddenLayer'])
  ).bar("load", legacy_silu_out1).bar("store", layerg['bar_layer'])
  down_proj_low1 = SchedGemv(LinearAtom,
    MNK=((3072, 1024), N, 6144),
    fold=3,
    tmas=(layerg['loadDown'], layerg['loadSiluLayer'], layerg['reduceHiddenLayer'])
  ).bar("load", legacy_silu_out1).bar("store", layerg['bar_layer'])
  def make_down_proj_high(m_range):
    return SchedGemv(LinearAtom,
      MNK=(m_range, N, (6144, 8192)),
      fold=4,
      tmas=(layerg['loadDown'], layerg['loadSiluLayer'], layerg['reduceHiddenLayer'])
    ).bar("load", layerg['bar_silu_out2']).bar("store", layerg['bar_layer'])

  if fine_mlp_barriers:
    silu_in_bars = [layerg[f'bar_silu_in{i}'] for i in range(3)]
    silu_out_bars = [layerg[f'bar_silu_out1_{i}'] for i in range(3)]

    gate_prefix_parts = [
      SchedGemv(
        QKVAtom,
        MNK=((i * 2048, 2048), N, HIDDEN),
        tmas=(layerg['loadGate'], layerg['loadRMSLayer'], layerg['storeGateOut']),
      ).bar("load", layerg['bar_post_attn_rms']).bar("store", silu_in_bars[i])
      for i in range(3)
    ]
    up_prefix_parts = [
      SchedGemv(
        QKVAtom,
        MNK=(m_range, N, HIDDEN),
        tmas=(layerg['loadUp'], layerg['loadRMSLayer'], layerg['storeInterm']),
      ).bar("load", layerg['bar_post_attn_rms']).bar("store", silu_in_bars[shard])
      for m_range, shard in (
        ((0, 2048), 0),
        ((2048, 1536), 1),
        ((3584, 512), 1),
        ((4096, 1024), 2),
        ((5120, 1024), 2),
      )
    ]

    if packed_silu_shards:
      silu1 = []
      for shard_id in range(3):
        shard = SchedSmemSiLUInterleaved(
          num_token=N,
          gate_glob=matGateOut[:, :mlp_split],
          up_glob=matInterm[:, :mlp_split],
          out_glob=matSiLUOut[:, :mlp_split],
          shards_per_token=3,
          fixed_shard_id=shard_id,
        ).bar(f"input{shard_id}", silu_in_bars[shard_id]).bar(
          f"output{shard_id}", silu_out_bars[shard_id]
        )
        silu1.append(shard)
    else:
      silu1 = SchedSmemSiLUInterleaved(
        num_token=N,
        gate_glob=matGateOut[:, :mlp_split],
        up_glob=matInterm[:, :mlp_split],
        out_glob=matSiLUOut[:, :mlp_split],
        shards_per_token=3,
      )
      for shard_id in range(3):
        silu1.bar(f"input{shard_id}", silu_in_bars[shard_id])
        silu1.bar(f"output{shard_id}", silu_out_bars[shard_id])

    down_low_parts = []
    for shard_id in range(3):
      k_range = (shard_id * 2048, 2048)
      down_low_parts.extend([
        SchedGemv(
          LinearAtom,
          MNK=((0, 3072), N, k_range),
          fold=1,
          tmas=(
            layerg['loadDown'],
            layerg['loadSiluLayer'],
            layerg['reduceHiddenLayer'],
          ),
        ).bar("load", silu_out_bars[shard_id]).bar("store", layerg['bar_layer']),
        SchedGemv(
          LinearAtom,
          MNK=((3072, 1024), N, k_range),
          fold=1,
          tmas=(
            layerg['loadDown'],
            layerg['loadSiluLayer'],
            layerg['reduceHiddenLayer'],
          ),
        ).bar("load", silu_out_bars[shard_id]).bar("store", layerg['bar_layer']),
      ])

  # after all layers, logits projection
  LogitsProj = []
  for i in range(logits_epoch):
    sched = SchedGemvMGroupArgmax(
      LogitsAtom,
      MNK=(logits_slice, N, HIDDEN),
      tmas=(
        defaultg[f"loadLogitsW{i}"],
        defaultg["loadLogitsB"],
      ),
      mat_out_partial=matArgmaxPartial,
      vocabulary_base=i * logits_slice,
      partial_base=i * num_sms,
    )
    if i == 0:
      sched.bar("load", layerg.over('bar_pre_attn_rms'))
    sched.bar("partial", systemg['bar_argmax_partial'])
    sched = sched.place(num_sms)
    if i == 1:
      # Epoch 0 leaves SM96--103 behind the rest of the grid.  Do not put the
      # next epoch's partial-barrier tail on those CTAs when eight auxiliary
      # CTAs are already idle; task coordinates and release counts stay exact.
      sched = SchedLmHeadEpoch1TailOffload(sched).place(full_sms)
    LogitsProj.append(sched)

  # The LM-head epilogue keeps logits in TMEM/registers and emits only one
  # compact maximum per task/token.  Eight reducer SMs consume the 256 records.
  Argmax = SchedArgmaxReduceGlobal(
    num_token=N,
    AtomReduce=ARGMAX_REDUCE_GLOBAL_bf16_256,
    mat_out_partial=matArgmaxPartial,
    mat_final_out=matTokens[:, token_offset+1],
    final_counter_offsets=_control_flow_offsets(
      matTokens.stride(1) * matTokens.element_size(),
      base_counter_delta * matTokens.stride(1) * matTokens.element_size() if control_flow else None,
    ) if control_flow else None,
  ).bar("partial", systemg['bar_argmax_partial']).bar("final", systemg['bar_token_finish'])

  sstart, send = systemg.range_bars()

  # restore barrier
  restore_bars_low = SchedCopy(
    tmas = wrap_static(TmaLoad1D(dae.bars_src[:sstart]), TmaStore1D(dae.bars[:sstart]))
  ).bar("load", layerg.over('bar_pre_attn_rms')).bar("store", systemg['bar_token_finish'])
  restore_bars_high = SchedCopy(
    tmas = wrap_static(TmaLoad1D(dae.bars_src[sstart:send]), TmaStore1D(dae.bars[sstart:send]))
  )

  embed_rms = embed_rms.place(rms_sms)
  copy_hidden = copy_hidden.place(N, base_sm=64)
  pre_attn_rms = pre_attn_rms.place(rms_sms)
  post_attn_rms = post_attn_rms.place(rms_sms)
  QProj = QProj if qkv_head_barriers else QProj.place(128)
  QRope = [] if fused_qk_rope else QRope.place(128)
  KProj = KProj if qkv_head_barriers else KProj.place(64, base_sm=64)
  KRope = [] if fused_qk_rope else KRope.place(64, base_sm=64)
  VProj = VProj if qkv_head_barriers else VProj.place(64)
  Gqa = Gqa.place(N * NUM_KV_HEAD)
  if out_proj_placement_specs is not None:
    OutProj = [
      part.place(part_sms, base_sm=base_sm)
      for part, part_sms, base_sm in out_proj_placement_specs
    ]
  else:
    OutProj = [
      out_proj_fold8.place(48),
      out_proj_fold4.place(104, base_sm=48),
    ]
  if fine_mlp_barriers:
    gate_prefix_parts = [
      part.place(32, base_sm=shard_id * 32)
      for shard_id, part in enumerate(gate_prefix_parts)
    ]
    up_prefix_parts = [
      part.place(part_sms, base_sm=base_sm)
      for part, (part_sms, base_sm) in zip(
        up_prefix_parts,
        ((32, 96), (24, 128), (8, 128), (16, 136), (16, 136)),
      )
    ]
    mlp_prefix_schedules = [*gate_prefix_parts, *up_prefix_parts]
    if packed_silu_shards:
      # Keep shard-0/1 consumers on the eight CTAs that finish the shard-1
      # projection tail, and shard 2 on its late producer/consumer group.
      # This avoids making early down-projection owners wait on an unrelated
      # late shard while preserving all 24 token-shard tasks and barriers.
      silu1 = [
        silu1[0].place(N, base_sm=num_sms),
        silu1[1].place(N, base_sm=num_sms),
        silu1[2].place(N, base_sm=num_sms + 16),
      ]
    else:
      silu1 = silu1.place(N * 3, base_sm=num_sms)
    down_low_parts = [
      part.place(part_sms, base_sm=base_sm)
      for part, (part_sms, base_sm) in zip(
        down_low_parts,
        ((48, 0), (16, 104), (48, 48),
         (16, 120), (48, 96), (16, 136)),
      )
    ]
    down_low_schedules = down_low_parts
  else:
    gate_proj_prefix = gate_proj_prefix.place(96)
    up_proj_main = up_proj_main.place(32, base_sm=96)
    up_proj_aux0 = up_proj_aux0.place(24, base_sm=num_sms)
    up_proj_aux1 = up_proj_aux1.place(24, base_sm=num_sms)
    up_proj_aux2 = up_proj_aux2.place(16, base_sm=num_sms + 8)
    mlp_prefix_schedules = [
      gate_proj_prefix,
      up_proj_main,
      up_proj_aux0,
      up_proj_aux1,
      up_proj_aux2,
    ]
    silu1 = silu1.place(N * 3, base_sm=num_sms)
    down_low_schedules = [down_proj_low0, down_proj_low1]
  gate_proj_tail = gate_proj_tail.place(128)
  up_silu_tail = up_silu_tail.place(128)
  if not fine_mlp_barriers:
    down_proj_low0 = down_proj_low0.place(144)
    down_proj_low1 = down_proj_low1.place(48, base_sm=104)
    down_low_schedules = [down_proj_low0, down_proj_low1]
  if interleave_down_high:
    # Run a disjoint high-K output range on the SMs that would otherwise wait
    # for MLP shard 2.  Together these ranges retain exactly the original 152
    # reduction tasks and the same output-barrier release count.
    down_proj_high_early = [
      make_down_proj_high((0, 768)).place(48, base_sm=96),
    ]
    down_proj_high_rest = [
      make_down_proj_high((768, 1536)).place(96),
      make_down_proj_high((2304, 128)).place(8, base_sm=144),
    ]
    down_low_early_schedules = down_low_schedules[:4]
    down_low_late_schedules = down_low_schedules[4:]
  else:
    down_proj_high_early = []
    down_proj_high_rest = [make_down_proj_high((0, 2432)).place(152)]
    down_low_early_schedules = down_low_schedules
    down_low_late_schedules = []
  # SM96--103 reach the layer boundary about four microseconds later than
  # SM128--135.  Move the final two M tiles (four K folds each) to that
  # auxiliary cohort; the tile set and all 448 reduction contributors stay
  # unchanged, and no new readiness frontier is introduced.
  down_proj_high1 = [
    make_down_proj_high((2432, 1536)).place(96),
    make_down_proj_high((3968, 128)).place(8, base_sm=128),
  ]
  Argmax = Argmax.place(N)
  restore_bars_low = restore_bars_low.place(1, base_sm=128)
  restore_bars_high = restore_bars_high.place(1, base_sm=128)
  # Layer L clears L-1 only after L's input-RMS frontier.  Issue the unchanged
  # stores behind current Q/K/V so SM88--151 can overlap them with attention.
  clear_q = SchedClearQ(
    TmaLoad1D(matZero[:N * QKVTileM], bytes=N * QKVTileM * matZero.element_size()),
    ToSplitMCordAdapter(layerg['storeQClear'], 64, QKVTileM),
    N * QKVTileM * matZero.element_size(),
    QKVTileM,
    64,
    layerg['bar_pre_attn_rms'],
  )
  if not phased_attn_out:
    clear_q.bar("store", layerg['bar_q_clear'])
  clear_q = clear_q.place(64, base_sm=88)

  q_projection_profile_sms = (
    (*range(104), *range(num_sms, full_sms))
    if q_fold1_aux_tail else range(128)
  )
  v_projection_profile_sms = (
    tuple(
      104 + v_k_tail_heads.index(head) * 8 + task
      if head in v_k_tail_heads else head * 8 + task
      for head in range(NUM_KV_HEAD)
      for task in range(8)
    )
    if v_k_tail_heads else range(64)
  )

  dae.bind_late_barrier_counts(
    embed_rms,
    copy_hidden,
    restore_bars_high,
    QProj,
    QRope,
    KProj,
    KRope,
    VProj,
    Gqa,
    OutProj,
    post_attn_rms,
    mlp_prefix_schedules,
    silu1,
    gate_proj_tail,
    up_silu_tail,
    down_low_early_schedules,
    down_proj_high_early,
    down_low_late_schedules,
    down_proj_high_rest,
    down_proj_high1,
    pre_attn_rms,
    clear_q,
    LogitsProj,
    Argmax,
    restore_bars_low,
  )

  # build first rms with embedding
  dae.i(
    embed_rms,
    copy_hidden,
    restore_bars_high,
  )

  # Start a new schedule to mark the loop target.
  dae.i(
    stage_profile_marker("layer_start"),
    QProj,
    stage_profile_marker("q_proj", q_projection_profile_sms),
    QRope,
    stage_profile_marker("q_rope", q_projection_profile_sms),
    KProj,
    stage_profile_marker("k_proj", range(64, 128)),
    KRope,
    stage_profile_marker("k_rope", range(64, 128)),
    VProj,
    stage_profile_marker("v_proj", v_projection_profile_sms),

    Gqa,
    stage_profile_marker("attention", range(N * NUM_KV_HEAD)),
    clear_q,
    stage_profile_marker("clear_q", range(88, full_sms)),
    stage_profile_schedule_parts("out_proj_part", OutProj),
    stage_profile_marker("out_proj", range(full_sms)),

    # RMS
    post_attn_rms,
    stage_profile_marker("post_attn_rms", range(rms_sms)),
    
    # MLP
    stage_profile_schedule_parts("mlp_prefix_part", mlp_prefix_schedules),
    stage_profile_marker("mlp_prefix", range(full_sms)),
    silu1,
    stage_profile_marker("silu_prefix", range(num_sms, full_sms)),
    gate_proj_tail,
    stage_profile_marker("gate_tail", range(128)),
    up_silu_tail,
    stage_profile_marker("silu_tail", range(128)),

    stage_profile_schedule_parts("down_low_early_part", down_low_early_schedules),
    stage_profile_schedule_parts("down_high_early_part", down_proj_high_early),
    stage_profile_schedule_parts("down_low_late_part", down_low_late_schedules),
    stage_profile_schedule_parts("down_high_rest_part", down_proj_high_rest),
    stage_profile_marker("down_high0", range(full_sms)),
    down_proj_high1,
    stage_profile_marker("down_high1", range(104)),

    # rms for next layer
    pre_attn_rms,
    stage_profile_marker("next_rms", range(rms_sms)),

    # All 152 SMs need the layer loop.
    LoopM.toNext(dae.copy_mptrs(), num_layers, resource_group = layerg),
    LoopC.toNext(dae.copy_cptrs(), num_layers),
    stage_profile_marker("layers_done"),

    # # logits
    stage_profile_schedule_parts("lm_head_epoch", LogitsProj),
    stage_profile_marker(
      "lm_head",
      getattr(LogitsProj[-1], "profile_sms", range(num_sms)),
    ),

    # argmax and cleanup
    Argmax,
    stage_profile_marker("argmax", range(N)),

    restore_bars_low,
    stage_profile_marker("restore", range(128, 129)),
  )

  if control_flow:
    if token_loop_ptrs is None:
      raise ValueError("control-flow scheduling requires token_loop_ptrs")
    token_cptrs, token_mptrs = token_loop_ptrs
    dae.i(
      IssueBarrier(systemg['bar_token_finish']),
      LoopM.toNext(
        token_mptrs,
        control_flow_tokens,
        reg=TOKEN_LOOP_REG,
      ),
      LoopC.toNext(
        token_cptrs,
        control_flow_tokens,
        reg=TOKEN_LOOP_REG,
      ),
    )

###################################
# finish schedule and ready to run
###################################

seed_prefill_kv_cache()

decode_offset = len(prefill_token_id_and_pos)
cur_offset = decode_offset - 1
cur_pos = prefill_token_id_and_pos[-1][1] if prefill_token_id_and_pos else -1
if parsed_args.control_flow:
  token, pos = input_token_id_and_pos[0]
  matTokens[0, decode_offset] = token
  token_loop_ptrs = (dae.copy_cptrs(), dae.copy_mptrs())
  schedule_single_token(
    decode_offset,
    pos,
    control_flow_tokens=CONTROL_FLOW_TOKENS_PER_LAUNCH,
    token_loop_ptrs=token_loop_ptrs,
  )
  cur_offset = decode_offset + total_decode_tokens - 1
  cur_pos = pos + total_decode_tokens - 1
else:
  for token_offset, (token, pos) in enumerate(input_token_id_and_pos, start=decode_offset):
    matTokens[0, token_offset] = token
    if token_offset > decode_offset:
      dae.i(IssueBarrier(systemg['bar_token_finish']))
    schedule_single_token(token_offset, pos)
    cur_offset, cur_pos = token_offset, pos

  for i in range(num_generates):
    cur_offset += 1
    cur_pos += 1
    dae.i(IssueBarrier(systemg['bar_token_finish']))
    schedule_single_token(cur_offset, cur_pos)

print(
  f"run VDCores with {cur_offset+1} tokens... "
  f"fine_mlp_barriers={int(fine_mlp_barriers)}, "
  f"packed_silu_shards={int(packed_silu_shards)}, "
  f"interleave_down_high={int(interleave_down_high)}, "
  f"fused_qk_rope={int(fused_qk_rope)}, "
  f"qkv_head_barriers={int(qkv_head_barriers)}, "
  f"q_fold1_aux_tail={int(q_fold1_aux_tail)}, "
  f"v_k_tail={int(v_k_tail)}, "
  f"phased_attn_out={int(phased_attn_out)}, "
  f"stage_profile={int(stage_profile)}, "
  f"track_profile={int(track_profile)}"
)
dae.s()
if will_execute:
  dae.prepare_launch()
  if track_profile:
    dae.profile.zero_()


def control_flow_launch_plan():
  base_seq_len = initial_decode_pos + 1
  base_num_kv_blocks = (base_seq_len + KVBlockSize - 1) // KVBlockSize
  plan = []
  for token_offset in range(total_decode_tokens):
    seq_len = base_seq_len + token_offset
    num_kv_blocks = (seq_len + KVBlockSize - 1) // KVBlockSize
    plan.append((token_offset, num_kv_blocks - base_num_kv_blocks))
  return plan


def reset_decode_state():
  for attn_q in attnQs:
    attn_q.zero_()
  for attn_k, attn_v in zip(attnKs, attnVs):
    attn_k[:, initial_decode_pos:final_decode_pos + 1].zero_()
    attn_v[:, initial_decode_pos:final_decode_pos + 1].zero_()
  torch.cuda.synchronize()


def launch_control_flow_sequence(launch_plan):
  reset_decode_state()
  loop_counter_values = []
  for token_base, kv_block_delta in launch_plan:
    counters = dae.loop_counters.copy()
    counters[TOKEN_BASE_REG] = token_base
    counters[KV_BLOCK_COUNT_REG] = kv_block_delta
    loop_counter_values.append(counters)
  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)
  wall_start_ns = time.perf_counter_ns()
  start_event.record()
  dae.launch_sequence(loop_counter_values, synchronize=False, reset_bars=True)
  end_event.record()
  end_event.synchronize()
  wall_time_ns = float(time.perf_counter_ns() - wall_start_ns)
  sequence_time_ns = float(start_event.elapsed_time(end_event) * 1e6)
  return sequence_time_ns, wall_time_ns


perf_summary = None
if will_execute and parsed_args.control_flow:
  launch_plan = control_flow_launch_plan()
  iterations = dae_execution_iterations(remaining_argv)
  warmup_sequences = int(os.environ.get("DAE_BENCH_WARMUP", "3")) if benchmark_mode else 0
  for _ in range(max(0, warmup_sequences)):
    launch_control_flow_sequence(launch_plan)

  print(
    f"[{'bench' if benchmark_mode else 'launch'}] "
    f"VDCores with {dae.num_sms} SMs, chunks={len(launch_plan)}..."
  )
  measurements = [launch_control_flow_sequence(launch_plan) for _ in range(iterations)]
  kernel_times = [measurement[0] for measurement in measurements]
  wall_times = [measurement[1] for measurement in measurements]
  kernel_time_ns = statistics.median(kernel_times)
  wall_time_ns = statistics.median(wall_times)
  if benchmark_mode:
    print(
      f"Benchmark Results on {dae.num_sms} SMs and {iterations} iterations:\n"
      f"Min end-to-end time (ns): {min(wall_times):.2f}\n"
      f"Median end-to-end time (ns): {wall_time_ns:.2f}\n"
      f"Average end-to-end time (ns): {statistics.mean(wall_times):.2f}\n"
      f"Max end-to-end time (ns): {max(wall_times):.2f}"
    )
  tbt_ns = wall_time_ns / total_decode_tokens
  perf_summary = (
    "[perf] "
    f"kernel_time_ms={kernel_time_ns / 1e6:.3f}, "
    f"end_to_end_ms={wall_time_ns / 1e6:.3f}, "
    f"decode_tokens={total_decode_tokens}, "
    f"TBT_ms={tbt_ns / 1e6:.3f}, "
    f"tokens_per_s={1e9 / tbt_ns:.2f}"
  )
elif will_execute:
  dae_app(dae)
  profile_data = dae.profile[:, 0:2].cpu().numpy()
  kernel_time_ns = float(profile_data[:, 1].max() - profile_data[:, 0].min())
  tbt_ns = kernel_time_ns / total_decode_tokens
  perf_summary = (
    "[perf] "
    f"kernel_time_ms={kernel_time_ns / 1e6:.3f}, "
    f"decode_tokens={total_decode_tokens}, "
    f"TBT_ms={tbt_ns / 1e6:.3f}, "
    f"tokens_per_s={1e9 / tbt_ns:.2f}"
  )
else:
  dae_app(dae)

if will_execute and stage_profile:
  profile = dae.profile.cpu().numpy()
  detailed_profile_events = {
    name.strip()
    for name in os.environ.get("VDCORES_STAGE_PROFILE_DETAIL", "").split(",")
    if name.strip()
  }
  ordered_profile_events = sorted(
    stage_profile_events.items(), key=lambda item: item[1][0]
  )
  previous_name = None
  previous_event_id = None
  layer_start_event = stage_profile_events["layer_start"][0]
  earliest_layer_start = profile[:, layer_start_event].astype("int64").min()
  for name, (event_id, active_sms) in ordered_profile_events:
    if previous_event_id is None:
      previous_name = name
      previous_event_id = event_id
      continue
    active = list(active_sms)
    duration_us = (
      profile[active, event_id].astype("int64")
      - profile[active, previous_event_id].astype("int64")
    ) / 1.0e3
    frontier_us = (
      profile[active, event_id].astype("int64")
      - profile[active, layer_start_event].astype("int64")
    ) / 1.0e3
    slow_order = duration_us.argsort()[-min(6, len(active)):][::-1]
    slow_sms = ",".join(
      f"{active[idx]}:{duration_us[idx]:.3f}" for idx in slow_order
    )
    frontier_order = frontier_us.argsort()[-min(6, len(active)):][::-1]
    tail_sms = ",".join(
      f"{active[idx]}:{frontier_us[idx]:.3f}" for idx in frontier_order
    )
    print(
      "[stage-profile] "
      f"{previous_name}->{name} active={len(active)} "
      f"duration_us[p10={float(torch.quantile(torch.tensor(duration_us), 0.1)):.3f},"
      f"p50={float(torch.median(torch.tensor(duration_us))):.3f},"
      f"p90={float(torch.quantile(torch.tensor(duration_us), 0.9)):.3f},"
      f"max={duration_us.max():.3f}] "
      f"frontier_us[p10={float(torch.quantile(torch.tensor(frontier_us), 0.1)):.3f},"
      f"p50={float(torch.median(torch.tensor(frontier_us))):.3f},"
      f"p90={float(torch.quantile(torch.tensor(frontier_us), 0.9)):.3f},"
      f"max={frontier_us.max():.3f}] slow_sms={slow_sms} tail_sms={tail_sms}"
    )
    if name in detailed_profile_events:
      frontier_detail = ",".join(
        f"{sm}:{frontier_us[idx]:.3f}"
        for idx, sm in enumerate(active)
      )
      absolute_us = (
        profile[active, event_id].astype("int64") - earliest_layer_start
      ) / 1.0e3
      absolute_detail = ",".join(
        f"{sm}:{absolute_us[idx]:.3f}"
        for idx, sm in enumerate(active)
      )
      print(
        f"[stage-profile-detail] {name} "
        f"frontier_us={frontier_detail} absolute_us={absolute_detail}"
      )
    previous_name = name
    previous_event_id = event_id

if will_execute and track_profile:
  profile = dae.profile.cpu().numpy().astype("uint64")
  track_magic = 0x4454524B50524631
  if not (profile[:, 127] == track_magic).all():
    raise RuntimeError(
      "VDCORES_TRACK_PROFILE=1 requires a runtime built with make track_profile=1"
    )

  track_slots = {
    "compute_m2c_wait_us": 96,
    "alloc_slot_stall_us": 99,
    "alloc_issue_barrier_us": 102,
    "ldu0_queue_wait_us": 105,
    "ldu0_dependency_wait_us": 107,
    "ldu1_queue_wait_us": 110,
    "ldu1_dependency_wait_us": 112,
    "store_queue_wait_us": 115,
    "store_service_us": 117,
    "store_barrier_service_us": 118,
  }
  for name, event_id in track_slots.items():
    values_us = profile[:, event_id].astype("float64") / 1.0e3
    values = torch.from_numpy(values_us)
    tail_order = values_us.argsort()[-min(6, len(values_us)):][::-1]
    tail_sms = ",".join(
      f"{sm}:{values_us[sm]:.3f}" for sm in tail_order
    )
    print(
      "[track-profile] "
      f"{name}[p10={float(torch.quantile(values, 0.1)):.3f},"
      f"p50={float(torch.median(values)):.3f},"
      f"p90={float(torch.quantile(values, 0.9)):.3f},"
      f"max={values_us.max():.3f}] tail_sms={tail_sms}"
    )

  count_slots = {
    "compute_m2c_calls": 97,
    "compute_m2c_contended": 98,
    "alloc_slot_stall_events": 100,
    "alloc_slot_retries": 101,
    "alloc_issue_barrier_contended": 103,
    "alloc_instructions": 104,
    "ldu0_queue_wait_calls": 106,
    "ldu0_dependency_contended": 108,
    "ldu0_commands": 109,
    "ldu1_queue_wait_calls": 111,
    "ldu1_dependency_contended": 113,
    "ldu1_commands": 114,
    "store_queue_wait_calls": 116,
    "store_commands": 119,
    "store_barrier_commands": 120,
  }
  for name, event_id in count_slots.items():
    values = profile[:, event_id].astype("int64")
    print(
      "[track-profile-count] "
      f"{name}[min={values.min()},p50={int(torch.median(torch.from_numpy(values)))},"
      f"max={values.max()}]"
    )

  compute_span_ns = (
    profile[:, 1].astype("int64") - profile[:, 0].astype("int64")
  ).clip(min=1)
  compute_wait_share = (
    profile[:, 96].astype("float64") / compute_span_ns.astype("float64")
  ) * 100.0
  print(
    "[track-profile-derived] "
    f"compute_m2c_wait_share_pct[p10={float(torch.quantile(torch.from_numpy(compute_wait_share), 0.1)):.2f},"
    f"p50={float(torch.median(torch.from_numpy(compute_wait_share))):.2f},"
    f"p90={float(torch.quantile(torch.from_numpy(compute_wait_share), 0.9)):.2f},"
    f"max={compute_wait_share.max():.2f}]"
  )

def print_generated_text():
  generated_token_ids = matTokens[
    0,
    decode_offset + 1:decode_offset + total_decode_tokens + 1,
  ].detach().cpu().tolist()
  generated_for_text = trim_after_eos(generated_token_ids)
  generated_text = tokenizer.decode(generated_for_text, skip_special_tokens=True)

  print(f"[output] generated_tokens={len(generated_token_ids)}")
  print(f"[output] generated_text: {generated_text}")


def check_logits_against_reference(_reference_logits):
  print("[correctness] logits materialization skipped by fused LM-head argmax")
  return []


def run_correctness_check():
  if parsed_args.control_flow:
    print(
      "[correctness] running control-flow greedy reference "
      f"after {len(prefill_token_id_and_pos)} prefill tokens for {total_decode_tokens} decode steps..."
    )
    tokens = [token for token, _ in prefill_token_id_and_pos] + [input_token_id_and_pos[0][0]]
    base_pos = input_token_id_and_pos[0][1]
    generated = []
    with torch.no_grad():
      for step in range(total_decode_tokens):
        positions = list(range(len(tokens)))
        inputs = input_batch1(*tokens, positions=positions)
        output = model(**inputs, use_cache=False)
        next_token = torch.argmax(output.logits[0, -1]).item()
        generated.append(next_token)
        tokens.append(next_token)

    dae_generated = matTokens[0, decode_offset + 1:decode_offset + total_decode_tokens + 1].detach().cpu().tolist()
    token_ok = dae_generated == generated
    print(f"[correctness] {'PASS' if token_ok else 'FAIL'} generated_tokens: ref={generated}, dae={dae_generated}")
    if not token_ok:
      print("[correctness] control-flow diagnostics for final repeated body:")
      tail_start = max(0, decode_offset + total_decode_tokens - 80)
      tail_end = decode_offset + total_decode_tokens + 1
      token_tail = matTokens[0, tail_start:tail_end].detach().cpu().tolist()
      print(
        "[correctness] dae token tail: "
        f"{list(zip(range(tail_start, tail_end), token_tail))}"
      )
      inputs = input_batch1(*tokens[:-1], positions=list(range(len(tokens) - 1)))
      captured, _ = reference_pass(model, inputs)
      final_idx = decode_offset + total_decode_tokens - 1
      final_pos = base_pos + total_decode_tokens - 1
      final_rope_row = matRope[final_pos, 0]
      for i in range(min(2, num_layers)):
        layer = captured[i]
        kv_start = max(0, final_pos - 8)
        kv_end = min(MAX_SEQ_LEN, final_pos + 9)
        v_row_sums = attnVs[i][0, kv_start:kv_end].float().abs().sum(dim=1)
        k_row_sums = attnKs[i][0, kv_start:kv_end].float().abs().sum(dim=1)
        print(
          f"[correctness] layer {i} KV row abs sums near final: "
          f"V={list(zip(range(kv_start, kv_end), v_row_sums.detach().cpu().tolist()))}, "
          f"K={list(zip(range(kv_start, kv_end), k_row_sums.detach().cpu().tolist()))}"
        )
        k_ref = apply_interleaved_rope_activation(
          permute_rope_activation(layer['k_proj'][0, final_idx], KW // HEAD_DIM),
          KW // HEAD_DIM,
          final_rope_row,
        )
        print(f"[correctness] Layer {i}, token {final_idx}:")
        check_tensor_threshold("v_proj", layer['v_proj'][0, final_idx], attnVs[i][0, final_pos], 5.0)
        print("[correctness] skip q_rope snapshot: control-flow clear_q zeros the reusable Q buffer after attention consumes it")
        check_tensor_threshold("k_rope", k_ref, attnKs[i][0, final_pos], 5.0)
      check_tensor_threshold("final_hidden", captured[num_layers-1]['hidden_state_out'][0, final_idx], matHidden[0], 5.0)
      check_tensor_threshold("final_rms", captured['final']['final_rms'][0, final_idx], matRMSHidden[0], 5.0)
      check_logits_against_reference(captured['final']['lm_head'][0, final_idx])
      raise RuntimeError("Control-flow correctness check failed")
    print("[correctness] control-flow token check passed")
    return

  print(
    "[correctness] running single-token reference capture "
    f"after {len(prefill_token_id_and_pos)} prefill tokens..."
  )
  inputs = input_batch1(
    *(e[0] for e in prefill_token_id_and_pos),
    *(e[0] for e in input_token_id_and_pos),
    mat=matTokens[0],
    positions=[e[1] for e in prefill_token_id_and_pos] + [e[1] for e in input_token_id_and_pos],
  )

  captured, output = reference_pass(model, inputs)
  all_ok = True
  decode_index = len(prefill_token_id_and_pos)
  decode_pos = input_token_id_and_pos[0][1]
  decode_rope_row = matRope[decode_pos, 0]

  for i in range(min(2, num_layers)):
    layer = captured[i]
    k_ref = apply_interleaved_rope_activation(
      permute_rope_activation(layer['k_proj'][0, decode_index], KW // HEAD_DIM),
      KW // HEAD_DIM,
      decode_rope_row,
    )
    print(f"[correctness] Layer {i}:")
    checks = [
      check_tensor_threshold("v_proj", layer['v_proj'][0, decode_index], attnVs[i][0, decode_pos], 5.0),
      check_tensor_threshold("k_rope", k_ref, attnKs[i][0, decode_pos], 5.0),
    ]
    print("[correctness] skip q_rope snapshot: clear_q zeros the reusable Q buffer after attention consumes it")
    all_ok = all_ok and all(passed for passed, _ in checks)

  print(f"[correctness] Checking Layer {num_layers-1}:")
  layer = captured[num_layers-1]
  silu_ref = F.silu(layer['gate_proj'][0, decode_index]) * layer['up_proj'][0, decode_index]
  final_checks = [
    check_tensor_threshold("gate_proj_high", layer['gate_proj'][0, decode_index, :6144], matGateOut[0, :6144], 5.0),
    check_tensor_threshold("up_proj_high", layer['up_proj'][0, decode_index, :6144], matInterm[0, :6144], 5.0),
    check_tensor_threshold("silu", silu_ref, matSiLUOut[0, :], 5.0),
    check_tensor_threshold("final_hidden", layer['hidden_state_out'][0, decode_index], matHidden[0], 5.0),
    check_tensor_threshold("final_rms", captured['final']['final_rms'][0, decode_index], matRMSHidden[0], 5.0),
    *check_logits_against_reference(captured['final']['lm_head'][0, decode_index]),
  ]
  all_ok = all_ok and all(passed for passed, _ in final_checks)

  ref_idx = torch.argmax(captured['final']['lm_head'], dim=-1)
  dae_idx = matTokens[0, decode_index + 1].item()
  ref_token = ref_idx[0, decode_index].item()
  token_ok = ref_token == dae_idx
  print(f"[correctness] {'PASS' if token_ok else 'FAIL'} final_token: ref={ref_token}, dae={dae_idx}")
  all_ok = all_ok and token_ok

  if not all_ok:
    raise RuntimeError("Correctness check failed")
  print("[correctness] all checks passed")


if parsed_args.correctness:
  run_correctness_check()

if will_execute:
  print_generated_text()
  print(perf_summary)

# print("output tokens: ", matTokens[0, :cur_offset+2])
