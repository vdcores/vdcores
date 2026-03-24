import sys

from dae.launcher import *
from dae.model import *
from dae.schedule import *
from dae.util import dae_app

from cli import parse_args
from runtime_context import build_runtime_context


ctx = build_runtime_context(parse_args())
dae = ctx.dae
N = ctx.N
HIDDEN = ctx.HIDDEN
eps = ctx.eps
rms_sms = ctx.rms_sms
full_sms = ctx.full_sms
logits_slice = ctx.logits_slice
logits_epoch = ctx.logits_epoch

matHidden = ctx.matHidden
matRMSHidden = ctx.matRMSHidden
matRMSInputWLoop = ctx.matRMSInputWLoop
matLogits = ctx.matLogits
matLogitsW = ctx.matLogitsW
matArgmaxIdx = ctx.matArgmaxIdx
matArgmaxVal = ctx.matArgmaxVal
matTokens = ctx.matTokens

layerg = dae.add_group("layerbars", 1)
systemg = dae.add_group("system", 1)
layerg.addBarrier("bar_pre_attn_rms")
for name in ("bar_logits", "bar_argmax_idx", "bar_argmax_val", "bar_token_finish"):
    systemg.addBarrier(name)
dae.build_groups()

final_rms_w = matRMSInputWLoop[0] if matRMSInputWLoop.dim() == 2 else matRMSInputWLoop

pre_attn_rms = SchedRMSShared(
    num_token=N,
    epsilon=eps,
    hidden_size=HIDDEN,
    tmas=(TmaLoad1D(final_rms_w), TmaLoad1D(matHidden), TmaStore1D(matRMSHidden)),
).bar("output", layerg["bar_pre_attn_rms"]).place(rms_sms)

qwen_gemvs = layers_like(GemvLayer, dae, Gemv_M64N8)
logits_proj = []
for i in range(logits_epoch):
    proj = qwen_gemvs(f"logits_proj_{i}", (matLogitsW[i], matRMSHidden, matLogits[i]), reduce=False)
    sched = proj.schedule_(group=False).split_M(6)
    if i == 0:
        sched.bar("load", layerg["bar_pre_attn_rms"])
    if i == logits_epoch - 1:
        sched.bar("store", systemg["bar_logits"])
    logits_proj.append(sched.place(full_sms))

argmax = SchedArgmax(
    num_token=N,
    logits_slice=logits_slice,
    num_slice=logits_epoch,
    AtomPartial=ARGMAX_PARTIAL_bf16_1152_50688_132,
    AtomReduce=ARGMAX_REDUCE_bf16_1152_132,
    matLogits=matLogits,
    matOutVal=matArgmaxVal,
    matOutIdx=matArgmaxIdx,
    matFinalOut=matTokens[:, 1],
).bar("load", systemg["bar_logits"]).bar("val", systemg["bar_argmax_val"]).bar("idx", systemg["bar_argmax_idx"]).bar("final", systemg["bar_token_finish"]).place(full_sms)

dae.bind_late_barrier_counts(pre_attn_rms, logits_proj, argmax)
dae.i(pre_attn_rms, logits_proj, argmax, IssueBarrier(systemg["bar_token_finish"]))
dae.s()
dae_app(dae)
