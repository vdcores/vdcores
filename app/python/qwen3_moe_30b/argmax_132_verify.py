import torch

from dae.launcher import *
from dae.schedule import SchedArgmax
from dae.util import dae_app, tensor_diff


torch.manual_seed(0)
gpu = torch.device("cuda")

NUM_TOKEN = 8
NUM_SMS = 132
LOGITS_SLICE = 64 * NUM_SMS * 6
LOGITS_EPOCH = 3
DTYPE = torch.bfloat16

matLogits = [torch.rand(NUM_TOKEN, LOGITS_SLICE, dtype=DTYPE, device=gpu) - 0.5 for _ in range(LOGITS_EPOCH)]
matArgOutVal = torch.zeros(NUM_TOKEN, NUM_SMS, dtype=DTYPE, device=gpu)
matArgOutIdx = torch.zeros(NUM_TOKEN, NUM_SMS, dtype=torch.long, device=gpu)
matArgOut = torch.zeros(NUM_TOKEN, dtype=torch.long, device=gpu)

dae = Launcher(NUM_SMS, device=gpu)
task_argmax = SchedArgmax(
    num_token=NUM_TOKEN,
    logits_slice=LOGITS_SLICE,
    num_slice=LOGITS_EPOCH,
    AtomPartial=ARGMAX_PARTIAL_bf16_1152_50688_132,
    AtomReduce=ARGMAX_REDUCE_bf16_1152_132,
    matLogits=matLogits,
    matOutVal=matArgOutVal,
    matOutIdx=matArgOutIdx,
    matFinalOut=matArgOut,
).place(NUM_SMS)

dae.i(task_argmax)
dae.s()
dae_app(dae)

matIn = torch.cat(matLogits, dim=-1)
ref_idx = torch.argmax(matIn, dim=-1)
ref_val = torch.gather(matIn, 1, ref_idx.unsqueeze(1)).squeeze(1)
dae_val = torch.gather(matIn, 1, matArgOut.unsqueeze(1)).squeeze(1)
tensor_diff("argmax_132", ref_val, dae_val)
