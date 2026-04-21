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
import math

arg_parser = argparse.ArgumentParser(add_help=False)
arg_parser.add_argument("-N", "--num-generates", type=int, default=None)
arg_parser.add_argument("--hf-cache-dir", default="/tmp/huggingface_cache")
arg_parser.add_argument("--correctness", action="store_true")
arg_parser.add_argument("--prompt", default=None)
arg_parser.add_argument("--message", action="append", default=None)
arg_parser.add_argument("--control-flow", dest="control_flow", action="store_true", default=True)
arg_parser.add_argument("--no-control-flow", dest="control_flow", action="store_false")
parsed_args, remaining_argv = arg_parser.parse_known_args()
if parsed_args.prompt is not None and parsed_args.message:
  raise ValueError("Use either --prompt or --message, not both")
if parsed_args.correctness and not any(arg in ("-l", "--launch", "-b", "--bench") for arg in remaining_argv):
  remaining_argv = [*remaining_argv, "--launch"]
will_execute = any(arg in ("-l", "--launch", "-b", "--bench") for arg in remaining_argv)
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

model_name = 'meta-llama/Llama-3.1-8B-Instruct'
cache_dir = parsed_args.hf_cache_dir

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    dtype=torch.bfloat16,
    device_map="auto",
    token=os.environ['HF_TOKEN']
)
config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir, token=os.environ['HF_TOKEN'])
eps = config.rms_norm_eps # 1e-6
rope_theta = config.rope_parameters["rope_theta"]

layers = model.model.layers
tokenizer = AutoTokenizer.from_pretrained(
  model_name,
  cache_dir=cache_dir,
  token=os.environ['HF_TOKEN'],
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
KVBlockSize = 64

rms_sms = REQ
num_sms = 128
full_sms = 132
dae = Launcher(132, device=gpu)


prompt_tokens = prompt_token_ids()
prefill_token_id_and_pos = [(token, pos) for pos, token in enumerate(prompt_tokens[:-1])]
input_token_id_and_pos = [(prompt_tokens[-1], len(prefill_token_id_and_pos))]
initial_decode_pos = input_token_id_and_pos[0][1]
tokens_until_kv_block_end = KVBlockSize - (initial_decode_pos % KVBlockSize)
if parsed_args.num_generates is not None:
  total_decode_tokens = parsed_args.num_generates
elif parsed_args.correctness and not parsed_args.control_flow:
  total_decode_tokens = 1
else:
  total_decode_tokens = tokens_until_kv_block_end
if total_decode_tokens <= 0:
  raise ValueError("--num-generates must be positive")
num_generates = total_decode_tokens - len(input_token_id_and_pos)
final_decode_pos = input_token_id_and_pos[0][1] + total_decode_tokens - 1
if final_decode_pos >= MAX_SEQ_LEN:
  raise ValueError(
    f"decode position {final_decode_pos} exceeds MAX_SEQ_LEN={MAX_SEQ_LEN}"
  )
if parsed_args.control_flow:
  if len(input_token_id_and_pos) != 1:
    raise ValueError("--control-flow currently supports exactly one initial token")
  if total_decode_tokens <= 0:
    raise ValueError("--control-flow requires at least one token")
  remaining_after_initial_block = max(0, total_decode_tokens - tokens_until_kv_block_end)
  if remaining_after_initial_block % KVBlockSize != 0:
    raise ValueError(
      "--control-flow full-block decode currently requires the appended region "
      f"to be a multiple of KVBlockSize: initial_pos={initial_decode_pos}, "
      f"total_tokens={total_decode_tokens}, KVBlockSize={KVBlockSize}"
    )
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

systemg.addBarrier('bar_logits')
systemg.addBarrier('bar_argmax_idx')
systemg.addBarrier('bar_argmax_val')
systemg.addBarrier('bar_token_finish') # argmax plus restore-barrier copy after placement

layerg.addBarrier('bar_layer')
layerg.addBarrier('bar_out_mlp')
layerg.addBarrier('bar_q_proj')
layerg.addBarrier('bar_qkv_attn')
layerg.addBarrier('bar_attn_out')
layerg.addBarrier('bar_rms_layer', REQ)
layerg.addBarrier('bar_rms_mlp', REQ)
layerg.addBarrier('bar_silu_in')
layerg.addBarrier('bar_silu_out1')
layerg.addBarrier('bar_silu_out2')
layerg.addBarrier('bar_pre_attn_rms')
layerg.addBarrier('bar_post_attn_rms')

###################################
# Define tensors
###################################

# TODO(zhiyuang: replace with a zero op
matZero = torch.zeros(4096, dtype=dtype, device=gpu)

_positions = torch.arange(MAX_SEQ_LEN).unsqueeze(0).to(gpu) # [1, seq]
_cos, _sin = model.model.rotary_emb(torch.zeros(1, device=gpu), _positions) # tensor only device matters here
matRope = torch.ones(MAX_SEQ_LEN, N, HEAD_DIM, dtype=torch.bfloat16, device=gpu)
matRope[:, :, 0::2] = _cos[0, :, :HEAD_DIM//2].unsqueeze(1) # llama duplicate it to full dim
matRope[:, :, 1::2] = _sin[0, :, :HEAD_DIM//2].unsqueeze(1)

matTokens = torch.zeros(N, MAX_SEQ_LEN, dtype=torch.int64, device=gpu)
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

matLogits = []
matLogitsW = []
matLmHeadW = model.lm_head.weight.detach()
vocab_size = matLmHeadW.shape[0]

# TODO(zhiyuang): tune for qwen
matLmHeadW.resize_(logits_slice * logits_epoch, 4096) # pad to 64*full_sms*6, to simplify the scheduling
matLmHeadW[vocab_size:,].zero_() # zero padding

matArgmaxIdx = torch.zeros(N, 128, dtype=torch.long, device=gpu)
matArgmaxVal = torch.zeros(N, 128, dtype=dtype, device=gpu)
matArgmaxOut = torch.zeros(N, dtype=torch.long, device=gpu)

for i in range(logits_epoch):
  matLogitsW.append(matLmHeadW[i * logits_slice: (i+1) * logits_slice])
  matLogits.append(torch.zeros(N, logits_slice, dtype=dtype, device=gpu))

# tensor cache policy
dae.set_persistent(matTokens)
dae.set_streaming(matqWs, matkWs, matvWs, matOutWs, matUps, matGates, matDowns)

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

TileM, _, TileK = Gemv_M64N8.MNK
defaultg.addTma("loadRope", [matRope], lambda t: t._build("load", TileM, N, tma_load_tbl, cord_load_tbl))

# load tmas for the same matrix for "grouped" instructions
layerg.addTma("loadRMSLayer", [matRMSHidden] * num_layers, lambda t: t.wgmma_load(N, TileK * Gemv_M64N8.n_batch, Major.K))
layerg.addTma("reduceHiddenLayer", [matHidden] * num_layers, lambda t: t.wgmma("reduce", N, TileM, Major.MN))
layerg.addTma("loadSiluLayer", [matSiLUOut] * num_layers, lambda t: t.wgmma_load(N, TileK * Gemv_M64N8.n_batch, Major.K))
layerg.addTma("storeSiluLayer", [matSiLUOut] * num_layers, lambda t: t.wgmma_store(N, TileM, Major.MN))
layerg.addTma("loadAttnOLayer", [attnO] * num_layers, lambda t: t.wgmma_load(N, TileK * Gemv_M64N8.n_batch, Major.K))

layerg.addTma("storeInterm", [matInterm] * num_layers, lambda t: t.wgmma_store(N, TileM, Major.MN))
layerg.addTma("storeGateOut", [matGateOut] * num_layers, lambda t: t.wgmma_store(N, TileM, Major.MN))
layerg.addTma("reduceInterm", [matInterm] * num_layers, lambda t: t.wgmma("reduce", N, TileM, Major.MN))
layerg.addTma("reduceGateOut", [matGateOut] * num_layers, lambda t: t.wgmma("reduce", N, TileM, Major.MN))

# RMS, skip the first one which is used for embedding fusion
layerg.addTma("loadRMSInputW", matRMSInputW[1:], lambda t: t.tensor1d("load", HIDDEN))
layerg.addTma("loadRMSPostAttnW", matRMSPostAttnW, lambda t: t.tensor1d("load", HIDDEN))

layerg.addTma("loadOutWs", matOutWs, lambda t: t.wgmma_load(TileM, TileK, Major.K))
layerg.addTma("loadDown", matDowns, lambda t: t.wgmma_load(TileM, TileK, Major.K))
layerg.addTma("loadUp", matUps, lambda t: t.wgmma_load(TileM, TileK, Major.K))
layerg.addTma("loadGate", matGates, lambda t: t.wgmma_load(TileM, TileK, Major.K))

tma_builder_MN = partial(build_tma_wgmma_mn, iK = -3)
cord_func_MN = partial(cord_func_MN_major, iK=-3)
cord_func_MN_cord2 = partial(cord_func_MN_major_cord2, iK=-3)

tma_builder_K = partial(build_tma_wgmma_k, iN = -3)
cord_func_K = partial(cord_func_K_major, iN=-3)

TileM, _, TileK = Gemv_M64N8_ROPE_128.MNK
layerg.addTma("loadQW", matqWs, lambda t: t.wgmma_load(TileM, TileK, Major.K))
layerg.addTma("loadKW", matkWs, lambda t: t.wgmma_load(TileM, TileK, Major.K))
layerg.addTma("loadVW", matvWs, lambda t: t.wgmma_load(TileM, TileK, Major.K))
layerg.addTma("storeQ", attnQs, lambda t: t.wgmma("reduce", N, TileM, Major.MN))
layerg.addTma("storeQClear", attnQs, lambda t: t.wgmma_store(N, TileM, Major.MN))
layerg.addTma("storeK", attnKs, lambda t: t._build("reduce", 64, N, tma_store_attn_kv, cord_id))
layerg.addTma("storeV", attnVs, lambda t: t._build("reduce", 64, N, tma_store_attn_kv, cord_id))

HEAD_DIM = ATTENTION_M64N64K16_F16_F32_64_64_hdim.HEAD_DIM
NUM_KV_HEAD = KW // HEAD_DIM
HEAD_GROUP_SIZE = QW // KW
matQ_attn_views = [attnQ.view(N, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM) for attnQ in attnQs]
matK_attn_views = [attnK.view(N, MAX_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM) for attnK in attnKs]
matV_attn_views = [attnV.view(N, MAX_SEQ_LEN, NUM_KV_HEAD, HEAD_DIM) for attnV in attnVs]
matO_attn_view = attnO.view(N, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)

layerg.addTma('loadQ', matQ_attn_views, lambda t: t._build("load", HEAD_DIM, 64, tma_gqa_load_q, cord_gqa_load_q))
layerg.addTma('loadK', matK_attn_views, lambda t: t._build("load", HEAD_DIM, KVBlockSize, tma_builder_K, cord_func_K))
layerg.addTma('loadV', matV_attn_views, lambda t: t._build("load", HEAD_DIM, KVBlockSize, tma_builder_MN, cord_func_MN))


## testing matrix
matTestHidden = [torch.zeros(N, HIDDEN, dtype=dtype, device=gpu) for _ in range(num_layers)]
layerg.addTma("testReduce", matTestHidden, lambda t: t.wgmma("reduce", N, TileM, Major.MN))

###################################
# Finish building resources
###################################

dae.build_groups()
tma_shift, bar_shift = layerg.get_shift()

###################################
# Start of Schedule
###################################

TOKEN_LOOP_REG = 1
KV_BLOCK_LOOP_REG = 2


class CounterOffsetCordAdapter(CordAdapter):
  def __init__(self, inner, offsets):
    super().__init__(inner)
    self.offsets = offsets

  def cord(self, *cords):
    inst = self.inner.cord(*cords)
    for counter_reg, delta in self.offsets:
      inst = CounterOffsetMemoryInstruction(counter_reg, inst, delta)
    return inst


def _control_flow_offsets(token_delta, block_delta=None):
  offsets = [(TOKEN_LOOP_REG, token_delta)]
  if block_delta is not None:
    offsets.append((KV_BLOCK_LOOP_REG, block_delta))
  return offsets


def maybe_counter_offset(inst, delta, enabled: bool, block_delta=None):
  if not enabled:
    return inst
  for counter_reg, counter_delta in _control_flow_offsets(delta, block_delta):
    inst = CounterOffsetMemoryInstruction(counter_reg, inst, counter_delta)
  return inst


def maybe_counter_adapter(adapter, delta, enabled: bool, block_delta=None):
  if not enabled:
    return adapter
  return CounterOffsetCordAdapter(adapter, _control_flow_offsets(delta, block_delta))


class SchedClearQ(Schedule):
  def __init__(self, load_zero, store_q, tile_bytes: int, tile_m: int, num_clear_sms: int):
    super().__init__()
    self.load_zero = load_zero
    self.store_q = store_q
    self.tile_bytes = tile_bytes
    self.tile_m = tile_m
    self.num_clear_sms = num_clear_sms

  def schedule(self, sm: int):
    if sm < 0:
      return []
    count = (HIDDEN // self.tile_m + self.num_clear_sms - 1 - sm) // self.num_clear_sms
    if count <= 0:
      return []
    return [
      Copy(count, size=self.tile_bytes),
      RepeatM.on(
        count,
        (self.load_zero.cord(0), 0),
        (self.store_q.cord(sm).group(), [self.num_clear_sms * self.tile_m, 0]),
      ),
    ]


def schedule_single_token(
    token_offset: int,
    token_pos: int,
    *,
    control_flow_tokens: int | None = None,
    token_loop_ptrs=None,
    block_loop_tokens: int | None = None,
    block_loop_ptrs=None,
):
  control_flow = control_flow_tokens is not None
  block_control_flow = block_loop_tokens is not None
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
      KVBlockSize * matTokens.element_size() if block_control_flow else None,
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
      KVBlockSize * matTokens.element_size() if block_control_flow else None,
    ),
  )

  # TODO(zhiyuang): finish a set of clear functions
  clear_interm = SchedCopy(
    size = 2048 * matInterm.element_size(),
    tmas = wrap_static(
      TmaLoad1D(matZero[:2048]),
      TmaStore1D(matInterm[0,4096:4096+2048]),
    ),
  )
  clear_gateout = SchedCopy(
    size = 2048 * matGateOut.element_size(),
    tmas = wrap_static(
      TmaLoad1D(matZero[:2048]),
      TmaStore1D(matGateOut[0, 4096:4096+2048]),
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
  regStoreQ = RegStore(0, size=N * TileM * matQ_attn_views[0].element_size())
  regLoadQ = RegLoad(0)
  QProj = SchedGemv(Gemv_M64N8,
    MNK=(QW, N, HIDDEN),
    tmas=(layerg['loadQW'], layerg['loadRMSLayer'], regStoreQ),
  ).bar("load", layerg['bar_pre_attn_rms'])
  QRope = SchedRope(ROPE_INTERLEAVE_512,
    tmas=(
      maybe_counter_adapter(
        ToRopeTableCordAdapter(defaultg['loadRope'], token_pos),
        [0, 0, 0, 1],
        control_flow,
        [0, 0, 0, KVBlockSize] if block_control_flow else None,
      ),
      regLoadQ,
      ToSplitMCordAdapter(layerg['storeQ'], 128//2, TileM),
    ),
  ).bar("store", layerg['bar_q_proj'])
  regStoreK = RegStore(0, size=N * TileM * matK_attn_views[0].element_size())
  regLoadK = RegLoad(0)
  KProj = SchedGemv(Gemv_M64N8,
    MNK=(KW, N, HIDDEN),
    tmas=(layerg['loadKW'], layerg['loadRMSLayer'], regStoreK),
  )
  KRope = SchedRope(ROPE_INTERLEAVE_512,
    tmas=(
      maybe_counter_adapter(
        ToRopeTableCordAdapter(defaultg['loadRope'], token_pos),
        [0, 0, 0, 1],
        control_flow,
        [0, 0, 0, KVBlockSize] if block_control_flow else None,
      ),
      regLoadK,
      maybe_counter_adapter(
        ToAttnKVStoreCordAdapter(layerg['storeK'], 64//4, TileM, token_pos),
        [0, 1, 0],
        control_flow,
        [0, KVBlockSize, 0] if block_control_flow else None,
      ),
    ),
  ).bar("store", layerg['bar_qkv_attn'])
  VProj = SchedGemv(Gemv_M64N8,
    MNK=(VW, N, HIDDEN),
    tmas=(
      layerg['loadVW'],
      layerg['loadRMSLayer'],
      maybe_counter_adapter(
        ToAttnVStoreCordAdapter(layerg['storeV'], token_pos),
        [0, 1, 0],
        control_flow,
        [0, KVBlockSize, 0] if block_control_flow else None,
      ),
    ),
  ).bar("store", layerg['bar_qkv_attn'])

  QWen8BGemvs = layers_like(GemvLayer, dae, Gemv_M64N8)
  Gqa = SchedAttentionDecoding(
    reqs = N, seq_len = token_pos + 1,
    KV_BLOCK_SIZE = KVBlockSize,
    NUM_KV_HEADS = NUM_KV_HEAD,
    matO = matO_attn_view,
    tmas = (layerg['loadQ'], layerg['loadK'], layerg['loadV']),
    seq_len_counter_reg=TOKEN_LOOP_REG if control_flow else None,
    num_kv_block_counter_reg=KV_BLOCK_LOOP_REG if block_control_flow else None,
    max_loop_count=control_flow_tokens or 1,
  ).bar("q", layerg['bar_q_proj']).bar("k", layerg['bar_qkv_attn']).bar("o", layerg['bar_attn_out'])

  # accumulate to matHidden, which auto applies the residual add
  OutProj = SchedGemv(Gemv_M64N8,
    MNK=(HIDDEN, N, HIDDEN),
    tmas=(layerg['loadOutWs'], layerg['loadAttnOLayer'], layerg['reduceHiddenLayer'])
  ).bar("load", layerg['bar_attn_out']).bar("store", layerg['bar_out_mlp'])

  # Gate Up + SiLU
  regGate, regUp = 0, 1
  regStoreGate = RegStore(regGate, matGateOut[:,0:TileM])
  regStoreUp = RegStore(regUp, matInterm[:,0:TileM])

  gate_proj_low = SchedGemv(Gemv_M64N8,
    MNK=(4096, N, HIDDEN),
    tmas=(layerg['loadGate'], layerg['loadRMSLayer'], layerg['storeGateOut']),
  ).bar("load", layerg['bar_post_attn_rms'])
  gate_proj_high = SchedGemv(Gemv_M64N8,
    MNK=((4096, 2048), N, HIDDEN),
    tmas=(layerg['loadGate'], layerg['loadRMSLayer'], layerg['reduceGateOut']),
  ).bar("store", layerg['bar_silu_in'])
  up_proj_low = SchedGemv(Gemv_M64N8,
    MNK=(4096, N, HIDDEN),
    tmas=(layerg['loadUp'], layerg['loadRMSLayer'], layerg['storeInterm']),
  ).bar("load", layerg['bar_post_attn_rms'])
  up_proj_high = SchedGemv(Gemv_M64N8,
    MNK=((4096, 2048), N, HIDDEN),
    tmas=(layerg['loadUp'], layerg['loadRMSLayer'], layerg['reduceInterm']),
  ).bar("store", layerg['bar_silu_in'])

  mlp_split = 6144
  silu1 = SchedSmemSiLUInterleaved(
    num_token=N,
    gate_glob=matGateOut[:, :mlp_split],
    up_glob=matInterm[:, :mlp_split],
    out_glob=matSiLUOut[:, :mlp_split],
  ).bar("input", layerg['bar_silu_in']).bar("output", layerg['bar_silu_out1'])
  gate_proj_fused = SchedGemv(Gemv_M64N8,
    MNK=((6144,8192), N, HIDDEN),
    tmas=(layerg['loadGate'], layerg['loadRMSLayer'], regStoreGate))
  up_proj_fused = SchedGemv(Gemv_M64N8,
    MNK=((6144,8192), N, HIDDEN),
    tmas=(layerg['loadUp'], layerg['loadRMSLayer'], regStoreUp))
  silu_fused = SchedRegSiLUFused(
    num_token=N,
    store_tma=layerg['storeSiluLayer'],
    reg_gate=regGate,
    reg_up=regUp,
    base_offset=mlp_split,
    stride=TileM,
  ).bar("output", layerg['bar_silu_out2'])
  down_proj_low = SchedGemv(Gemv_M64N8,
    MNK=(HIDDEN, N, 6144),
    tmas=(layerg['loadDown'], layerg['loadSiluLayer'], layerg['reduceHiddenLayer']))
  down_proj_high = SchedGemv(Gemv_M64N8,
    MNK=(HIDDEN, N, (6144, 8192)),
    tmas=(layerg['loadDown'], layerg['loadSiluLayer'], layerg['reduceHiddenLayer'])
  ).bar("load", layerg['bar_silu_out2']).bar("store", layerg['bar_layer'])
  down_proj_low.bar("load", layerg['bar_silu_out1'])

  # after all layers, logits projection
  LogitsProj = []
  for i in range(logits_epoch):
    proj = QWen8BGemvs(f"logits_proj_{i}", (matLogitsW[i], matRMSHidden, matLogits[i]), reduce=False)
    sched = proj.schedule_(group=False).split_M(logits_fold)
    if i == 0:
      sched.bar("load", layerg.over('bar_pre_attn_rms'))
      sched[0].no_prefetch()
    if i == logits_epoch - 1:
      sched.bar("store", systemg['bar_logits'])
    LogitsProj.append(sched.place(num_sms))

  # argmax
  Argmax = SchedArgmax(
    num_token=N,
    logits_slice=logits_slice,
    num_slice=logits_epoch,
    AtomPartial=ARGMAX_PARTIAL_bf16_1024_65536_128,
    AtomReduce=ARGMAX_REDUCE_bf16_1024_128,
    matLogits=matLogits,
    matOutVal=matArgmaxVal,
    matOutIdx=matArgmaxIdx,
    matFinalOut=matTokens[:, token_offset+1],
    final_counter_offsets=_control_flow_offsets(
      matTokens.stride(1) * matTokens.element_size(),
      KVBlockSize * matTokens.stride(1) * matTokens.element_size() if block_control_flow else None,
    ) if control_flow else None,
  ).bar("load", systemg['bar_logits']).bar("val", systemg['bar_argmax_val']).bar("idx", systemg['bar_argmax_idx']).bar("final", systemg['bar_token_finish'])

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
  clear_interm = clear_interm.place(1, base_sm=128)
  clear_gateout = clear_gateout.place(1, base_sm=129)
  clear_q = SchedClearQ(
    TmaLoad1D(matZero[:N * TileM], bytes=N * TileM * matZero.element_size()),
    ToSplitMCordAdapter(layerg['storeQClear'], 64, TileM),
    N * TileM * matZero.element_size(),
    TileM,
    full_sms - num_sms,
  ).place(full_sms - num_sms, base_sm=num_sms)
  pre_attn_rms = pre_attn_rms.place(rms_sms)
  post_attn_rms = post_attn_rms.place(rms_sms)
  QProj = QProj.place(128)
  QRope = QRope.place(128)
  KProj = KProj.place(64, base_sm=64)
  KRope = KRope.place(64, base_sm=64)
  VProj = VProj.place(64)
  Gqa = Gqa.place(N * NUM_KV_HEAD)
  OutProj = OutProj.place(num_sms)
  gate_proj_low = gate_proj_low.place(64)
  gate_proj_high = gate_proj_high.place(64)
  up_proj_low = up_proj_low.place(64, base_sm=64)
  up_proj_high = up_proj_high.place(64, base_sm=64)
  silu1 = silu1.place(4, base_sm=128)
  gate_proj_fused = gate_proj_fused.place(128)
  up_proj_fused = up_proj_fused.place(128)
  silu_fused = silu_fused.place(128)
  down_proj_low = down_proj_low.place(128)
  down_proj_high = down_proj_high.place(128)
  Argmax = Argmax.place(128)
  restore_bars_low = restore_bars_low.place(1, base_sm=128)
  restore_bars_high = restore_bars_high.place(1, base_sm=128)

  dae.bind_late_barrier_counts(
    embed_rms,
    copy_hidden,
    restore_bars_high,
    clear_interm,
    clear_gateout,
    clear_q if control_flow else [],
    QProj,
    QRope,
    KProj,
    KRope,
    VProj,
    Gqa,
    OutProj,
    post_attn_rms,
    gate_proj_low,
    gate_proj_high,
    up_proj_low,
    up_proj_high,
    silu1,
    gate_proj_fused,
    up_proj_fused,
    silu_fused,
    down_proj_low,
    down_proj_high,
    pre_attn_rms,
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

  # start a new scheudule to mark the loop target
  dae.i(
    clear_interm,
    clear_gateout,
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
    gate_proj_low,
    gate_proj_high,
    up_proj_low,
    up_proj_high,
    silu1,
    gate_proj_fused,
    up_proj_fused,
    silu_fused,
    down_proj_low,
    down_proj_high,

    # rms for next layer
    pre_attn_rms,

    # clear the per-layer Q buffer after its consumers are finished; the next
    # token will not revisit this layer until the rest of the layers complete.
    clear_q if control_flow else [],

    # # all 132 SM need loop
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
      LoopM.toNext(token_mptrs, control_flow_tokens, reg=TOKEN_LOOP_REG),
      LoopC.toNext(token_cptrs, control_flow_tokens, reg=TOKEN_LOOP_REG),
    )
    if block_control_flow:
      if block_loop_ptrs is None:
        raise ValueError("block control-flow scheduling requires block_loop_ptrs")
      block_cptrs, block_mptrs = block_loop_ptrs
      dae.i(
        LoopM.toNext(block_mptrs, block_loop_tokens, reg=KV_BLOCK_LOOP_REG),
        LoopC.toNext(block_cptrs, block_loop_tokens, reg=KV_BLOCK_LOOP_REG),
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
  initial_block_tokens = min(total_decode_tokens, tokens_until_kv_block_end)
  schedule_single_token(
    decode_offset,
    pos,
    control_flow_tokens=initial_block_tokens,
    token_loop_ptrs=token_loop_ptrs,
  )
  cur_offset = decode_offset + initial_block_tokens - 1
  cur_pos = pos + initial_block_tokens - 1

  remaining_full_block_tokens = total_decode_tokens - initial_block_tokens
  if remaining_full_block_tokens:
    if remaining_full_block_tokens % KVBlockSize != 0:
      raise ValueError(
        "control-flow full-block decode requires remaining tokens after the initial block "
        f"to be a multiple of KVBlockSize: remaining={remaining_full_block_tokens}, KVBlockSize={KVBlockSize}"
      )
    block_loop_tokens = remaining_full_block_tokens // KVBlockSize
    full_block_offset = decode_offset + initial_block_tokens
    full_block_pos = pos + initial_block_tokens
    block_loop_ptrs = (dae.copy_cptrs(), dae.copy_mptrs())
    inner_loop_ptrs = block_loop_ptrs
    schedule_single_token(
      full_block_offset,
      full_block_pos,
      control_flow_tokens=KVBlockSize,
      token_loop_ptrs=inner_loop_ptrs,
      block_loop_tokens=block_loop_tokens,
      block_loop_ptrs=block_loop_ptrs,
    )
    cur_offset += remaining_full_block_tokens
    cur_pos += remaining_full_block_tokens
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

print(f"run vdcors with {cur_offset+1} tokens...")
dae.s()
dae_app(dae)
if will_execute:
  profile_data = dae.profile[:, 0:2].cpu().numpy()
  vdcores_time_ns = float(profile_data[:, 1].max() - profile_data[:, 0].min())
  tbt_ns = vdcores_time_ns / total_decode_tokens
  perf_summary = (
    "[perf] "
    f"vdcores_time_ms={vdcores_time_ns / 1e6:.3f}, "
    f"decode_tokens={total_decode_tokens}, "
    f"TBT_ms={tbt_ns / 1e6:.3f}, "
    f"tokens_per_s={1e9 / tbt_ns:.2f}"
  )
else:
  perf_summary = None

def print_generated_text():
  generated_token_ids = matTokens[
    0,
    decode_offset + 1:decode_offset + total_decode_tokens + 1,
  ].detach().cpu().tolist()
  generated_for_text = trim_after_eos(generated_token_ids)
  generated_text = tokenizer.decode(generated_for_text, skip_special_tokens=True)

  print(f"[output] generated_tokens={len(generated_token_ids)}")
  print(f"[output] generated_text: {generated_text}")


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
      inputs = input_batch1(*tokens[:-1], positions=list(range(len(tokens) - 1)))
      captured, _ = reference_pass(model, inputs)
      final_idx = decode_offset + total_decode_tokens - 1
      final_pos = base_pos + total_decode_tokens - 1
      final_rope_row = matRope[final_pos, 0]
      for i in range(min(2, num_layers)):
        layer = captured[i]
        q_ref = apply_interleaved_rope_activation(
          permute_rope_activation(layer['q_proj'][0, final_idx], QW // HEAD_DIM),
          QW // HEAD_DIM,
          final_rope_row,
        )
        k_ref = apply_interleaved_rope_activation(
          permute_rope_activation(layer['k_proj'][0, final_idx], KW // HEAD_DIM),
          KW // HEAD_DIM,
          final_rope_row,
        )
        print(f"[correctness] Layer {i}, token {final_idx}:")
        check_tensor_threshold("v_proj", layer['v_proj'][0, final_idx], attnVs[i][0, final_pos], 5.0)
        check_tensor_threshold("q_rope", q_ref, attnQs[i][0], 5.0)
        check_tensor_threshold("k_rope", k_ref, attnKs[i][0, final_pos], 5.0)
      check_tensor_threshold("final_hidden", captured[num_layers-1]['hidden_state_out'][0, final_idx], matHidden[0], 5.0)
      check_tensor_threshold("final_rms", captured['final']['final_rms'][0, final_idx], matRMSHidden[0], 5.0)
      check_tensor_threshold("logits_low", captured['final']['lm_head'][0, final_idx, :logits_slice], matLogits[0][0, :logits_slice], 10.0)
      check_tensor_threshold("logits_high", captured['final']['lm_head'][0, final_idx, logits_slice:vocab_size], matLogits[1][0, :vocab_size - logits_slice], 10.0)
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
    q_ref = apply_interleaved_rope_activation(
      permute_rope_activation(layer['q_proj'][0, decode_index], QW // HEAD_DIM),
      QW // HEAD_DIM,
      decode_rope_row,
    )
    k_ref = apply_interleaved_rope_activation(
      permute_rope_activation(layer['k_proj'][0, decode_index], KW // HEAD_DIM),
      KW // HEAD_DIM,
      decode_rope_row,
    )
    print(f"[correctness] Layer {i}:")
    checks = [
      check_tensor_threshold("v_proj", layer['v_proj'][0, decode_index], attnVs[i][0, decode_pos], 5.0),
      check_tensor_threshold("q_rope", q_ref, attnQs[i][0], 5.0),
      check_tensor_threshold("k_rope", k_ref, attnKs[i][0, decode_pos], 5.0),
    ]
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
    check_tensor_threshold("logits_low", captured['final']['lm_head'][0, decode_index, :logits_slice], matLogits[0][0, :logits_slice], 10.0),
    check_tensor_threshold("logits_high", captured['final']['lm_head'][0, decode_index, logits_slice:vocab_size], matLogits[1][0, :vocab_size - logits_slice], 10.0),
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
