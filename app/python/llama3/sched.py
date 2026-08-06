import torch
import torch.nn.functional as F
import argparse
import sys
from functools import partial
from dae.launcher import *
from dae.schedule import *
from dae.model import *
from dae.util import dae_app
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
layerg.addBarrier('bar_q_proj')
layerg.addBarrier('bar_qkv_attn')
layerg.addBarrier('bar_attn_out')
layerg.addBarrier('bar_q_clear')
layerg.addBarrier('bar_rms_layer', REQ)
layerg.addBarrier('bar_rms_mlp', REQ)
layerg.addBarrier('bar_silu_in', 0 if fine_mlp_barriers else None)
layerg.addBarrier('bar_silu_out1', 0 if fine_mlp_barriers else None)
layerg.addBarrier('bar_silu_out2')
layerg.addBarrier('bar_pre_attn_rms')
layerg.addBarrier('bar_post_attn_rms')
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

QKVAtom = Gemv_M64N8
LinearAtom = Gemv_M64N8
LogitsAtom = Gemv_M128N8Argmax4
QKVTileM, _, QKVTileK = QKVAtom.MNK
LinearTileM, _, LinearTileK = LinearAtom.MNK
LogitsTileM, _, LogitsTileK = LogitsAtom.MNK

print(f"[weights] packing projections as contiguous M{LinearTileM}K{LinearTileK} tiles")
matqWs = [pack_weight_tile_major(weight, QKVTileM, QKVTileK) for weight in matqWs]
matkWs = [pack_weight_tile_major(weight, QKVTileM, QKVTileK) for weight in matkWs]
matvWs = [pack_weight_tile_major(weight, QKVTileM, QKVTileK) for weight in matvWs]
matOutWs = [pack_weight_tile_major(weight, LinearTileM, LinearTileK) for weight in matOutWs]
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


def linear_output_tma(mode):
  return lambda t: t.wgmma(mode, N, LinearTileM, Major.MN)


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
layerg.addTma("loadSiluLayer", [matSiLUOut] * num_layers, lambda t: t.wgmma_load(N, LinearTileK * LinearAtom.n_batch, Major.K))
layerg.addTma("storeSiluLayer", [matSiLUOut] * num_layers, lambda t: t.wgmma_store(N, QKVTileM, Major.MN))
layerg.addTma("loadAttnOLayer", [attnO] * num_layers, lambda t: t.wgmma_load(N, LinearTileK * LinearAtom.n_batch, Major.K))

layerg.addTma("storeInterm", [matInterm] * num_layers, lambda t: t.wgmma_store(N, QKVTileM, Major.MN))
layerg.addTma("storeGateOut", [matGateOut] * num_layers, lambda t: t.wgmma_store(N, QKVTileM, Major.MN))

# RMS, skip the first one which is used for embedding fusion
layerg.addTma("loadRMSInputW", matRMSInputW[1:], lambda t: t.tensor1d("load", HIDDEN))
layerg.addTma("loadRMSPostAttnW", matRMSPostAttnW, lambda t: t.tensor1d("load", HIDDEN))

layerg.addTma("loadOutWs", matOutWs, lambda t: weight_load_tma(t, LinearTileM, LinearTileK))
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
layerg.addTma("storeQClear", attnQs, lambda t: t.wgmma_store(N, QKVTileM, Major.MN))
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


class CounterOffsetCordAdapter(CordAdapter):
  def __init__(self, inner, offsets):
    super().__init__(inner)
    self.offsets = offsets

  def cord(self, *cords):
    inst = self.inner.cord(*cords)
    for counter_reg, delta in self.offsets:
      inst = CounterOffsetMemoryInstruction(counter_reg, inst, delta)
    return inst


class SchedClearQ(Schedule):
  def __init__(self, load_zero, store_q, tile_bytes: int, tile_m: int, num_clear_sms: int, wait_bar: int):
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
    count = (HIDDEN // self.tile_m + self.num_clear_sms - 1 - sm) // self.num_clear_sms
    if count <= 0:
      return []
    return [
      IssueBarrier(self.wait_bar).group(),
      Copy(count, size=self.tile_bytes),
      RepeatM.on(
        count,
        (self.load_zero.cord(0), 0),
        (
          self.store_q.cord(sm).bar(self._bar("store")).group(),
          [self.num_clear_sms * self.tile_m, 0],
        ),
      ),
      IssueBarrier(self._bar("store")).group(),
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

  # QKV Projection
  # TODO(zhiyuang): add the ROPE for Q and K
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
  VProj = SchedGemv(QKVAtom,
    MNK=(VW, N, HIDDEN),
    tmas=(
      layerg['loadVW'],
      layerg['loadRMSLayer'],
      maybe_counter_adapter(
        ToAttnVStoreCordAdapter(layerg['storeV'], token_pos),
        [0, 1, 0],
        control_flow,
        [0, base_counter_delta, 0] if control_flow else None,
      ),
    ),
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
    max_loop_count=min(
      control_flow_tokens or 1,
      KVBlockSize - (token_pos % KVBlockSize),
    ),
  ).bar("q", layerg['bar_q_proj']).bar("k", layerg['bar_qkv_attn']).bar("o", layerg['bar_attn_out'])

  # accumulate to matHidden, which auto applies the residual add
  # Twelve M tiles use four K folds and the other 52 use two folds: exactly
  # 152 independent tasks, one for every Blackwell SM.
  out_proj_fold4 = SchedGemv(LinearAtom,
    MNK=((0, 768), N, HIDDEN),
    fold=4,
    tmas=(layerg['loadOutWs'], layerg['loadAttnOLayer'], layerg['reduceHiddenLayer'])
  ).bar("load", layerg['bar_attn_out']).bar("store", layerg['bar_out_mlp'])
  out_proj_fold2 = SchedGemv(LinearAtom,
    MNK=((768, HIDDEN - 768), N, HIDDEN),
    fold=2,
    tmas=(layerg['loadOutWs'], layerg['loadAttnOLayer'], layerg['reduceHiddenLayer'])
  ).bar("load", layerg['bar_attn_out']).bar("store", layerg['bar_out_mlp'])

  # Gate Up + SiLU
  reg_gate, reg_up = 0, 1
  reg_store_gate = RegStore(reg_gate, matGateOut[:, :QKVTileM])
  reg_store_up = RegStore(reg_up, matInterm[:, :QKVTileM])

  gate_proj_prefix = SchedGemv(QKVAtom,
    MNK=(6144, N, HIDDEN),
    tmas=(layerg['loadGate'], layerg['loadRMSLayer'], layerg['storeGateOut']),
  ).bar("load", layerg['bar_post_attn_rms']).bar("store", layerg['bar_silu_in'])
  up_proj_main = SchedGemv(QKVAtom,
    MNK=(2048, N, HIDDEN),
    tmas=(layerg['loadUp'], layerg['loadRMSLayer'], layerg['storeInterm']),
  ).bar("load", layerg['bar_post_attn_rms']).bar("store", layerg['bar_silu_in'])
  up_proj_aux0 = SchedGemv(QKVAtom,
    MNK=((2048, 1536), N, HIDDEN),
    tmas=(layerg['loadUp'], layerg['loadRMSLayer'], layerg['storeInterm']),
  ).bar("load", layerg['bar_post_attn_rms']).bar("store", layerg['bar_silu_in'])
  up_proj_aux1 = SchedGemv(QKVAtom,
    MNK=((3584, 1536), N, HIDDEN),
    tmas=(layerg['loadUp'], layerg['loadRMSLayer'], layerg['storeInterm']),
  ).bar("store", layerg['bar_silu_in'])
  up_proj_aux2 = SchedGemv(QKVAtom,
    MNK=((5120, 1024), N, HIDDEN),
    tmas=(layerg['loadUp'], layerg['loadRMSLayer'], layerg['storeInterm']),
  ).bar("store", layerg['bar_silu_in'])

  mlp_split = 6144
  silu1 = SchedSmemSiLUInterleaved(
    num_token=N,
    gate_glob=matGateOut[:, :mlp_split],
    up_glob=matInterm[:, :mlp_split],
    out_glob=matSiLUOut[:, :mlp_split],
    shards_per_token=3,
  ).bar("input", layerg['bar_silu_in']).bar("output", layerg['bar_silu_out1'])
  gate_proj_tail = SchedGemv(QKVAtom,
    MNK=((mlp_split, INTERMIDIATE - mlp_split), N, HIDDEN),
    tmas=(layerg['loadGate'], layerg['loadRMSLayer'], reg_store_gate),
  )
  up_proj_tail = SchedGemv(QKVAtom,
    MNK=((mlp_split, INTERMIDIATE - mlp_split), N, HIDDEN),
    tmas=(layerg['loadUp'], layerg['loadRMSLayer'], reg_store_up),
  )
  silu_tail = SchedRegSiLUFused(
    num_token=N,
    store_tma=layerg['storeSiluLayer'],
    reg_gate=reg_gate,
    reg_up=reg_up,
    base_offset=mlp_split,
    stride=QKVTileM,
  ).bar("output", layerg['bar_silu_out2'])
  # Balance the down projection over all 152 SMs.  The two low-K schedules
  # contribute 192 fold-3 tasks and the two high-K schedules contribute 256
  # fold-4 tasks.  Placement gives 144 SMs three K2048 tasks and eight SMs two
  # tasks, instead of leaving the 104-SM rectangular core with a 28-tile tail.
  down_proj_low0 = SchedGemv(LinearAtom,
    MNK=((0, 3072), N, 6144),
    fold=3,
    tmas=(layerg['loadDown'], layerg['loadSiluLayer'], layerg['reduceHiddenLayer'])
  ).bar("load", layerg['bar_silu_out1']).bar("store", layerg['bar_layer'])
  down_proj_low1 = SchedGemv(LinearAtom,
    MNK=((3072, 1024), N, 6144),
    fold=3,
    tmas=(layerg['loadDown'], layerg['loadSiluLayer'], layerg['reduceHiddenLayer'])
  ).bar("load", layerg['bar_silu_out1']).bar("store", layerg['bar_layer'])
  down_proj_high0 = SchedGemv(LinearAtom,
    MNK=((0, 2432), N, (6144, 8192)),
    fold=4,
    tmas=(layerg['loadDown'], layerg['loadSiluLayer'], layerg['reduceHiddenLayer'])
  ).bar("load", layerg['bar_silu_out2']).bar("store", layerg['bar_layer'])
  down_proj_high1 = SchedGemv(LinearAtom,
    MNK=((2432, 1664), N, (6144, 8192)),
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
    LogitsProj.append(sched.place(num_sms))

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
  QProj = QProj.place(128)
  QRope = QRope.place(128)
  KProj = KProj.place(64, base_sm=64)
  KRope = KRope.place(64, base_sm=64)
  VProj = VProj.place(64)
  Gqa = Gqa.place(N * NUM_KV_HEAD)
  OutProj = [
    out_proj_fold4.place(48),
    out_proj_fold2.place(104, base_sm=48),
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
  up_proj_tail = up_proj_tail.place(128)
  silu_tail = silu_tail.place(128)
  if not fine_mlp_barriers:
    down_proj_low0 = down_proj_low0.place(144)
    down_proj_low1 = down_proj_low1.place(48, base_sm=104)
    down_low_schedules = [down_proj_low0, down_proj_low1]
  down_proj_high0 = down_proj_high0.place(152)
  down_proj_high1 = down_proj_high1.place(104)
  Argmax = Argmax.place(N)
  restore_bars_low = restore_bars_low.place(1, base_sm=128)
  restore_bars_high = restore_bars_high.place(1, base_sm=128)
  clear_q = SchedClearQ(
    TmaLoad1D(matZero[:N * QKVTileM], bytes=N * QKVTileM * matZero.element_size()),
    ToSplitMCordAdapter(layerg['storeQClear'], 64, QKVTileM),
    N * QKVTileM * matZero.element_size(),
    QKVTileM,
    aux_sms,
    layerg['bar_attn_out'],
  ).bar("store", layerg['bar_q_clear']).place(aux_sms, base_sm=num_sms)

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
    up_proj_tail,
    silu_tail,
    down_low_schedules,
    down_proj_high0,
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
    QProj,
    QRope,
    KProj,
    KRope,
    VProj,

    Gqa,
    OutProj,

    # RMS
    post_attn_rms,
    
    # MLP
    mlp_prefix_schedules,
    silu1,
    gate_proj_tail,
    up_proj_tail,
    silu_tail,

    down_low_schedules,
    down_proj_high0,
    down_proj_high1,

    # rms for next layer
    pre_attn_rms,

    clear_q,

    # All 152 SMs need the layer loop.
    LoopM.toNext(dae.copy_mptrs(), num_layers, resource_group = layerg),
    LoopC.toNext(dae.copy_cptrs(), num_layers),

    # # logits
    LogitsProj,

    # argmax and cleanup
    Argmax,

    restore_bars_low,
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
  f"fine_mlp_barriers={int(fine_mlp_barriers)}"
)
dae.s()
if will_execute:
  dae.prepare_launch()


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
