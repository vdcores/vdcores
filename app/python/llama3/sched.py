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
from reference import input_batch1, reference_pass, check_tensor_threshold
import os
import math

arg_parser = argparse.ArgumentParser(add_help=False)
arg_parser.add_argument("-N", "--num-generates", type=int, default=16)
arg_parser.add_argument("--hf-cache-dir", default="/tmp/huggingface_cache")
arg_parser.add_argument("--correctness", action="store_true")
parsed_args, remaining_argv = arg_parser.parse_known_args()
if parsed_args.correctness and not any(arg in ("-l", "--launch", "-b", "--bench") for arg in remaining_argv):
  remaining_argv = [*remaining_argv, "--launch"]
sys.argv = [sys.argv[0], *remaining_argv]

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


input_token_id_and_pos = [
  (791, 0),
]
num_generates = 0 if parsed_args.correctness else parsed_args.num_generates - 1

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
for i, (_, pos) in enumerate(input_token_id_and_pos):
  _sub_cos = _cos[0, pos:MAX_SEQ_LEN, :HEAD_DIM//2] # llama duplicate it to full dim
  _sub_sin = _sin[0, pos:MAX_SEQ_LEN, :HEAD_DIM//2]
  # interleave cos and sin
  matRope[0:MAX_SEQ_LEN-pos, i, 0::2] = _sub_cos
  matRope[0:MAX_SEQ_LEN-pos, i, 1::2] = _sub_sin

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
  return activation.view(num_heads, 2, HEAD_DIM // 2).transpose(1, 2).reshape_as(activation).contiguous()

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

def schedule_single_token(token_offset: int, token_pos: int):
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
    embedding=CC0(matTokens[0], token_offset)
  ).bar("output", layerg['bar_pre_attn_rms'])
  # copy the HIDDEN from embedding
  copy_hidden = SchedCopy(
    size = HIDDEN * matHidden.element_size(),
    tmas = (loadEmbed1D, storeHidden1D),
    cords = (None, sm_cord_1d(HIDDEN * 2)),
    before_copy = CC0(matTokens[0], token_offset),
  )

  # TODO(zhiyuang): finish a set of clear functions
  clear_interm = SchedCopy(
    size = 2048 * matInterm.element_size(),
    tmas = (TmaLoad1D(matZero[:2048]), TmaStore1D(matInterm[0,4096:4096+2048])),
  )
  clear_gateout = SchedCopy(
    size = 2048 * matGateOut.element_size(),
    tmas = (TmaLoad1D(matZero[:2048]), TmaStore1D(matGateOut[0, 4096:4096+2048])),
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
    tmas=(defaultg['loadRope'], regLoadQ, layerg['storeQ']),
    cords=(sm_cord_rope(token_pos), None, sm_cord_splitM(128//2, TileM))
  ).bar("store", layerg['bar_q_proj'])
  regStoreK = RegStore(0, size=N * TileM * matK_attn_views[0].element_size())
  regLoadK = RegLoad(0)
  KProj = SchedGemv(Gemv_M64N8,
    MNK=(KW, N, HIDDEN),
    tmas=(layerg['loadKW'], layerg['loadRMSLayer'], regStoreK),
  )
  KRope = SchedRope(ROPE_INTERLEAVE_512,
    tmas=(defaultg['loadRope'], regLoadK, layerg['storeK']),
    cords=(sm_cord_rope(token_pos), None, sm_cord_store_attn_kv(64//4, TileM, token_pos)),
  ).bar("store", layerg['bar_qkv_attn'])
  VProj = SchedGemv(Gemv_M64N8,
    MNK=(VW, N, HIDDEN),
    tmas=(layerg['loadVW'], layerg['loadRMSLayer'], layerg['storeV']),
    cordconv=(None, None, conv_m2cord_attn_store_v(token_pos)),
  ).bar("store", layerg['bar_qkv_attn'])

  QWen8BGemvs = layers_like(GemvLayer, dae, Gemv_M64N8)
  Gqa = SchedAttentionDecoding(
    reqs = N, seq_len = token_pos + 1,
    KV_BLOCK_SIZE = KVBlockSize,
    NUM_KV_HEADS = NUM_KV_HEAD,
    matO = matO_attn_view,
    tmas = (layerg['loadQ'], layerg['loadK'], layerg['loadV']),
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
  ).bar("load", systemg['bar_logits']).bar("val", systemg['bar_argmax_val']).bar("idx", systemg['bar_argmax_idx']).bar("final", systemg['bar_token_finish'])

  sstart, send = systemg.range_bars()

  # restore barrier
  restore_bars_low = SchedCopy(
    tmas = (TmaLoad1D(dae.bars_src[:sstart]), TmaStore1D(dae.bars[:sstart]))
  ).bar("load", layerg.over('bar_pre_attn_rms')).bar("store", systemg['bar_token_finish'])
  restore_bars_high = SchedCopy(
    tmas = (TmaLoad1D(dae.bars_src[sstart:send]), TmaStore1D(dae.bars[sstart:send]))
  )

  embed_rms = embed_rms.place(rms_sms)
  copy_hidden = copy_hidden.place(N, base_sm=64)
  clear_interm = clear_interm.place(1, base_sm=128)
  clear_gateout = clear_gateout.place(1, base_sm=129)
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

    # # all 132 SM need loop
    LoopM.toNext(dae.copy_mptrs(), num_layers, resource_group = layerg),
    LoopC.toNext(dae.copy_cptrs(), num_layers),

    # # logits
    LogitsProj,

    # argmax and cleanup
    Argmax,

    restore_bars_low,
  )

###################################
# finish schedule and ready to run
###################################

cur_offset, cur_pos = 0, 0
for token_offset, (token, pos) in enumerate(input_token_id_and_pos):
  matTokens[0, token_offset] = token
  if token_offset > 0:
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


def run_correctness_check():
  print("[correctness] running single-token reference capture...")
  inputs = input_batch1(
    *(e[0] for e in input_token_id_and_pos),
    mat=matTokens[0],
    positions=[e[1] for e in input_token_id_and_pos],
  )

  captured, output = reference_pass(model, inputs)
  all_ok = True

  for i in range(min(2, num_layers)):
    layer = captured[i]
    print(f"[correctness] Layer {i}:")
    checks = [
      check_tensor_threshold("v_proj", layer['v_proj'][0, 0], attnVs[i][0, 0], 5.0),
      check_tensor_threshold("q_proj", permute_rope_activation(layer['q_proj'][0, 0], QW // HEAD_DIM), attnQs[i][0], 5.0),
      check_tensor_threshold("k_proj", permute_rope_activation(layer['k_proj'][0, 0], KW // HEAD_DIM), attnKs[i][0, 0], 5.0),
    ]
    all_ok = all_ok and all(passed for passed, _ in checks)

  print(f"[correctness] Checking Layer {num_layers-1}:")
  layer = captured[num_layers-1]
  silu_ref = F.silu(layer['gate_proj'][0, 0]) * layer['up_proj'][0, 0]
  final_checks = [
    check_tensor_threshold("gate_proj_high", layer['gate_proj'][0, 0, :6144], matGateOut[0, :6144], 5.0),
    check_tensor_threshold("up_proj_high", layer['up_proj'][0, 0, :6144], matInterm[0, :6144], 5.0),
    check_tensor_threshold("silu", silu_ref, matSiLUOut[0, :], 5.0),
    check_tensor_threshold("final_hidden", layer['hidden_state_out'][0, 0], matHidden[0], 5.0),
    check_tensor_threshold("final_rms", captured['final']['final_rms'][0, 0], matRMSHidden[0], 5.0),
    check_tensor_threshold("logits_low", captured['final']['lm_head'][0, 0, :logits_slice], matLogits[0][0, :logits_slice], 10.0),
    check_tensor_threshold("logits_high", captured['final']['lm_head'][0, 0, logits_slice:vocab_size], matLogits[1][0, :vocab_size - logits_slice], 10.0),
  ]
  all_ok = all_ok and all(passed for passed, _ in final_checks)

  ref_idx = torch.argmax(captured['final']['lm_head'], dim=-1)
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

# print("output tokens: ", matTokens[0, :cur_offset+2])
