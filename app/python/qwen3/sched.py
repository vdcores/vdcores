import torch
import torch.nn.functional as F
from functools import partial
from dae.launcher import *
from dae.schedule import *
from dae.model import *
from dae.util import dae_app
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os

###################################
# load model
###################################

# model_name = 'Qwen/Qwen3-8B'
model_name = 'meta-llama/Llama-3.1-8B-Instruct'

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir="/tmp/huggingface_cache",
    dtype=torch.bfloat16,
    device_map="auto",
    token=os.environ['HF_TOKEN']
)
config = AutoConfig.from_pretrained(model_name)
eps = config.rms_norm_eps # 1e-6

layers = model.model.layers

# for name, param in model.named_parameters():
#     print(name, tuple(param.shape))

###################################
# basic parameter of DAE
###################################

gpu = torch.device("cuda")
torch.manual_seed(0)

dtype = torch.bfloat16
N, HIDDEN, INTERMIDIATE = 8, 4096, 4096 * 3
QW, KW, VW = 128 * 32, 128 * 8, 128 * 8
num_layers = len(layers)
# TODO(zhiyuang): error when bar=8
rms_sms = 8

assert QW == HIDDEN, "QW projects to hidden size"

REQ = N
MAX_SEQ_LEN = 4096

num_sms = 128
full_sms = 132
dae = Launcher(full_sms, device=gpu)

QSize, KVBlockSize = 16, 64
HEAD_DIM = 128
hist_seq_len = 16
batch_seq_len = 7

###################################
# Define groups, barriers and TMA for scheduling
###################################

defaultg = dae.get_group()
layerg = dae.add_group("layer", num_layers)

defaultg.addBarrier('bar_embedding', N)
defaultg.addBarrier('bar_rms_final', rms_sms)
defaultg.addBarrier('bar_logits', full_sms)
defaultg.addBarrier('bar_argmax_idx', full_sms)
defaultg.addBarrier('bar_argmax_val', full_sms)
defaultg.addBarrier('bar_argmax_out', N)

layerg.addBarrier('bar_layer', num_sms)
layerg.addBarrier('bar_out_mlp', num_sms)
layerg.addBarrier('bar_q_proj', num_sms)
layerg.addBarrier('bar_qkv_attn', num_sms)
layerg.addBarrier('bar_attn_out', REQ * 8)
layerg.addBarrier('bar_rms_layer', REQ)
layerg.addBarrier('bar_rms_mlp', REQ)
layerg.addBarrier('bar_silu_in', num_sms)
layerg.addBarrier('bar_silu_out1', N)
layerg.addBarrier('bar_silu_out2', num_sms)
layerg.addBarrier('bar_pre_attn_rms', rms_sms)
layerg.addBarrier('bar_post_attn_rms', rms_sms)

###################################
# Define tensors
###################################

# TODO(zhiyuang: replace with a hugging face load

matRope = torch.ones(MAX_SEQ_LEN, N, HEAD_DIM, dtype=torch.bfloat16, device=gpu)
matTokens = torch.zeros(MAX_SEQ_LEN, dtype=torch.int32, device=gpu)
matHidden = torch.rand(N, HIDDEN, dtype=dtype, device=gpu) - 0.5
matRMSHidden = torch.rand(N, HIDDEN, dtype=dtype, device=gpu) - 0.5

attnQs = [torch.zeros(REQ, HIDDEN, dtype=dtype, device=gpu) for _ in range(num_layers)]
attnKs = [torch.zeros(REQ, KVBlockSize, KW, dtype=dtype, device=gpu) for _ in range(num_layers)]
attnVs = [torch.zeros(REQ, KVBlockSize, VW, dtype=dtype, device=gpu) for _ in range(num_layers)]
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
matqWs = [l.self_attn.q_proj.weight for l in layers]
matkWs = [l.self_attn.k_proj.weight for l in layers]
matvWs = [l.self_attn.v_proj.weight for l in layers]

# Attn out proj
matOutWs = [l.self_attn.o_proj.weight for l in layers]

matUps = [l.mlp.up_proj.weight for l in layers]
matGates = [l.mlp.gate_proj.weight for l in layers]
matDowns = [l.mlp.down_proj.weight for l in layers]

logits_slice = 64*full_sms * 6
logits_epoch = 3

matLogits = []
matLogitsW = []
matLmHeadW = model.lm_head.weight.detach()
matLmHeadW.resize_(152064, 4096) # pad to 64*full_sms*6, to simplify the scheduling
matLmHeadW[151936:,].zero_() # zero padding

matArgmaxIdx = torch.zeros(N, full_sms, dtype=torch.long, device=gpu)
matArgmaxVal = torch.zeros(N, full_sms, dtype=dtype, device=gpu)
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
layerg.addTma("storeK", attnKs, lambda t: t._build("reduce", 64, N, tma_builder_MN, cord_func_MN_cord2))
layerg.addTma("storeV", attnVs, lambda t: t._build("reduce", 64, N, tma_builder_MN, cord_func_MN_cord2))

HEAD_DIM = ATTENTION_M64N64K16_F16_F32_64_64_hdim.HEAD_DIM
NUM_KV_HEAD = KW // HEAD_DIM
HEAD_GROUP_SIZE = QW // KW
matQ_attn_views = [attnQ.view(N, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM) for attnQ in attnQs]
matK_attn_views = [attnK.view(N, KVBlockSize, NUM_KV_HEAD, HEAD_DIM) for attnK in attnKs]
matV_attn_views = [attnV.view(N, KVBlockSize, NUM_KV_HEAD, HEAD_DIM) for attnV in attnVs]
matO_attn_view = attnO.view(N, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM)

layerg.addTma('loadQ', matQ_attn_views, lambda t: t._build("load", HEAD_DIM, 64, tma_gqa_load_q, cord_gqa_load_q))
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
  embedding=CC0(matTokens, 0)
).place(rms_sms)
# copy the HIDDEN from embedding
copy_hidden = SchedCopy(
  size = HIDDEN * matHidden.element_size(),
  tmas = (loadEmbed1D, storeHidden1D),
  cords = (None, sm_cord_1d(HIDDEN * 2)),
  before_copy = CC0(matTokens, 0),
).place(N, base_sm=64)

pre_attn_rms = SchedRMSShared(
  num_token=N, epsilon=eps,
  tmas=(layerg['loadRMSInputW'].cord(0), loadHidden1D, storeRMSHidden1D)
).place(rms_sms)
post_attn_rms = SchedRMSShared(
  num_token=N, epsilon=eps,
  tmas=(layerg['loadRMSPostAttnW'].cord(0), loadHidden1D, storeRMSHidden1D)
).place(rms_sms)

# QKV Projection
# TODO(zhiyuang): add the ROPE for Q and K
regStoreQ = RegStore(0, size=N * TileM * matQ_attn_views[0].element_size())
regLoadQ = RegLoad(0)
QProj = SchedGemv(Gemv_M64N8,
  MNK=(QW, N, HIDDEN),
  tmas=(layerg['loadQW'], layerg['loadRMSLayer'], regStoreQ),
).place(128)
QRope = SchedRope(ROPE_INTERLEAVE_512,
  tmas=(defaultg['loadRope'], regLoadQ, layerg['storeQ']),
  cords=(sm_cord_rope(batch_seq_len), None, sm_cord_splitM(128, TileM))
).place(128)
regStoreK = RegStore(0, size=N * TileM * matK_attn_views[0].element_size())
regLoadK = RegLoad(0)
KProj = SchedGemv(Gemv_M64N8,
  MNK=(KW, N, HIDDEN),
  tmas=(layerg['loadKW'], layerg['loadRMSLayer'], regStoreK),
).place(64, base_sm=64)
KRope = SchedRope(ROPE_INTERLEAVE_512,
  tmas=(defaultg['loadRope'], regLoadK, layerg['storeK']),
  cords=(sm_cord_rope(batch_seq_len), None, sm_cord_splitM(64, TileM)),
).place(64, base_sm=64)
VProj = SchedGemv(Gemv_M64N8,
  MNK=(VW, N, HIDDEN),
  tmas=(layerg['loadVW'], layerg['loadRMSLayer'], layerg['storeV']),
).place(64)

QWen8BGemvs = layers_like(GemvLayer, dae, Gemv_M64N8)
Gqa = SchedAttentionDecoding(
  reqs = N, seq_len = 1,
  KV_BLOCK_SIZE = KVBlockSize,
  NUM_KV_HEADS = NUM_KV_HEAD,
  matO = matO_attn_view,
  tmas = (layerg['loadQ'], layerg['loadK'], layerg['loadV']),
).place(N * NUM_KV_HEAD)

# accumulate to matHidden, which auto applies the residual add
OutProj = SchedGemv(Gemv_M64N8,
  MNK=(HIDDEN, N, HIDDEN),
  tmas=(layerg['loadOutWs'], layerg['loadAttnOLayer'], layerg['reduceHiddenLayer'])).place(num_sms)

# Gate Up + SiLU
regGate, regUp = 0, 1
regStoreGate = RegStore(regGate, matGateOut[:,0:TileM])
regStoreUp = RegStore(regUp, matInterm[:,0:TileM])

# after all layers, logits projection
LogitsProj = []
for i in range(logits_epoch):
  proj = QWen8BGemvs(f"logits_proj_{i}", (matLogitsW[i], matRMSHidden, matLogits[i]), reduce=False)
  sched = proj.schedule_(group=False).split_M(6).place(full_sms)
  # TODO(zhiyuang): check the "no-prefetch" here
  if i == 0:
    sched.bar("load", layerg.over('bar_pre_attn_rms'))
    sched[0].no_prefetch()
  if i == logits_epoch - 1:
    sched.bar("store", defaultg['bar_logits'])
  LogitsProj.append(sched)

# argmax
Argmax = SchedArgmax(N, 152064, logits_slice, matLogits, matArgmaxVal, matArgmaxIdx, matArgmaxOut)

def silu1(sm: int):
  sm -= 128
  if sm < 0:
    return []
  insts = []
  start_token_id = sm * (N // 4)
  end_token_id = (sm + 1) * (N // 4)
  for i in range(start_token_id, end_token_id):
    insts.extend([
      SILU_MUL_SHARED_BF16_K_4096_INTER(1),
      TmaStore1D(matSiLUOut[i,:4096]).bar(layerg['bar_silu_out1']).group(),
      TmaLoad1D(matGateOut[i,:4096]).bar(layerg['bar_silu_in']).group() if i == start_token_id else TmaLoad1D(matGateOut[i,:4096]),
      TmaLoad1D(matInterm[i,:4096]),
    ])
  return insts

def silu_fused(sm : int):
  if sm >= 128:
    return []

  return [
    SILU_MUL_SHARED_BF16_K_64_SW128(N),

    layerg['storeSiluLayer'].cord(0, 4096 + sm * TileM).bar(layerg['bar_silu_out2']).group(),
    RegLoad(regGate), # Load the gate
    RegLoad(regUp), # load the up
  ]

# nloop = num_layers
# default_tmas, default_bars = defaultg.get_shift()
# print("nloop:", nloop, "default_bars", default_bars, "default_tmas", default_tmas, "bar_shift:", bar_shift, "tma_shift:", tma_shift)

# build first rms with embedding
dae.i(
  embed_rms
    .bar("output", layerg['bar_pre_attn_rms']),
  copy_hidden
)

# start a new scheudule to mark the loop target
dae.s(
  QProj
    .bar("load", layerg['bar_pre_attn_rms']),
  QRope
    .bar("store", layerg['bar_q_proj']),
  KProj,
  KRope
    .bar("store", layerg['bar_qkv_attn']),
  VProj
    .bar("store", layerg['bar_qkv_attn']),

  Gqa
    .bar("q", layerg['bar_q_proj'])
    .bar("k", layerg['bar_qkv_attn'])
    .bar("o", layerg['bar_attn_out']),
  OutProj
    .bar("load", layerg['bar_attn_out'])
    .bar("store", layerg['bar_out_mlp']),

  # # RMS
  post_attn_rms
    .bar("input", layerg['bar_out_mlp'])
    .bar("output", layerg['bar_post_attn_rms']),
  
  # # MLP
  SchedGemv(Gemv_M64N8,
    MNK=(4096, N, HIDDEN),
    tmas=(layerg['loadGate'], layerg['loadRMSLayer'], layerg['storeGateOut']),)
    .place(64)
    .bar("store", layerg['bar_silu_in'])
    .bar("load", layerg['bar_post_attn_rms']),
  SchedGemv(Gemv_M64N8,
    MNK=(4096, N, HIDDEN),
    tmas=(layerg['loadUp'], layerg['loadRMSLayer'], layerg['storeInterm']),
    ).place(64, base_sm=64)
    .bar("store", layerg['bar_silu_in'])
    .bar("load", layerg['bar_post_attn_rms']),
  silu1,
  SchedGemv(Gemv_M64N8,
    MNK=((4096,8192), N, HIDDEN),
    tmas=(layerg['loadGate'], layerg['loadRMSLayer'], regStoreGate)).place(128),
  SchedGemv(Gemv_M64N8,
    MNK=((4096,8192), N, HIDDEN),
    tmas=(layerg['loadUp'], layerg['loadRMSLayer'], regStoreUp)).place(128),
  silu_fused,
  SchedGemv(Gemv_M64N8,
    MNK=(HIDDEN, N, 4096),
    tmas=(layerg['loadDown'], layerg['loadSiluLayer'], layerg['reduceHiddenLayer']))
    .place(128)
    .bar("load", layerg['bar_silu_out1']),
  SchedGemv(Gemv_M64N8,
    MNK=(HIDDEN, N, (4096, 8192)),
    tmas=(layerg['loadDown'], layerg['loadSiluLayer'], layerg['reduceHiddenLayer']))
    .place(128)
    .bar("load", layerg['bar_silu_out2'])
    .bar("store", layerg['bar_layer']),

  # rms for next layer
  pre_attn_rms
    .bar("input", layerg['bar_layer'])
    .bar("output", layerg.next('bar_pre_attn_rms')),

  # # all 132 SM need loop
  LoopM.toNext(dae.copy_mptrs(), num_layers, resource_group = layerg),
  LoopC.toNext(dae.copy_cptrs(), num_layers),

  # logits
  LogitsProj,

  # argmax
  Argmax
    .ld_barrier(defaultg['bar_logits'])
    .val_barrier(defaultg['bar_argmax_val'])
    .idx_barrier(defaultg['bar_argmax_idx'])
    .final_barrier(defaultg['bar_argmax_out'])
)

###################################
# finish schedule and ready to run
###################################

matTokens[0] = 1 # dummy token to avoid empty seq len
dae_app(dae)

###################################
# verifications
###################################

def ref_gateup_silu():
  gate = matHidden @ matGate.T
  interm = matHidden @ matUp.T
  sinterm = F.silu(gate) * interm
  out = sinterm @ matDown.T
  return interm, gate, out, sinterm

def diff_gateup_silu():
  ref_interm, ref_gate, ref_out, ref_silu = ref_gateup_silu()
  dae_interm = matInterm
  dae_gate = matGateOut
  dae_silu = matSiLUOut
  dae_out = matOut

  # gate and interm can be compared if replace regstore with global store
  # tensor_diff("gate", ref_gate, dae_gate)
  # tensor_diff("interm", ref_interm, dae_interm)
  tensor_diff("silu", ref_silu, dae_silu)
  tensor_diff("out", ref_out, dae_out)

def logits_diff():
  # concate logitsW
  ref_logitsW = torch.cat(matLogitsW, dim=0)
  ref_logits = F.linear(matRMSHidden, ref_logitsW)
  dae_logits = torch.cat(matLogits, dim=1)
  tensor_diff("logits", ref_logits, dae_logits)

def final_rms_diff():
  var = matOut.pow(2).mean(dim=-1, keepdim=True)
  X = matOut * torch.rsqrt(var + eps)
  dae_final = matRMSHidden
  tensor_diff("final_rms", X[:,0], dae_final[:,0])

# test correctness
if False:
  # pre_attn_rms.diff()
  # QProj.diff()
  # KProj.diff()
  # VProj.diff()
  # gqa.diff()
  # OutProj.diff()
  # post_attn_rms.diff()
  diff_gateup_silu()
  # final_rms.diff()
  final_rms_diff()
  # logits_diff()

# (base) ubuntu@192-222-51-156:~/Mirage_GH/dae$ python app/python/qwen3/sched.py -b
# [bench] DAE with 128 SMs...
# Benchmark Results on 128 SMs and 1 iterations:
# Average duration (ns): 120157.25
# Average execution time (ns): 123456.00
