import torch
import copy
from math import sqrt
import torch.nn.functional as F
from dae.launcher import *
from dae.util import *
from dae.schedule import *

torch.manual_seed(2333)
gpu = torch.device("cuda")


NUM_TOKEN = 8
VOCAB_SIZE = 8192 * 16 # for schedule

num_sms = 128

assert NUM_TOKEN <= num_sms

dae = Launcher(num_sms, device=gpu)

# align with sched.py
logits_fold = 8
logits_slice = 8192 * logits_fold
logits_epoch = 2

dtype = torch.bfloat16
matLogits = []
for i in range(logits_epoch):
  matLogits.append(torch.rand(NUM_TOKEN, logits_slice, dtype=dtype, device=gpu))

matArgOutVal = torch.zeros(NUM_TOKEN, num_sms, dtype=dtype, device=gpu)
matArgOutIdx = torch.zeros(NUM_TOKEN, num_sms, dtype=torch.long, device=gpu)
matArgOut = torch.zeros(NUM_TOKEN, dtype=torch.long, device=gpu)

val_bar = dae.new_bar(num_sms)
idx_bar = dae.new_bar(num_sms)

task_argmax = SchedArgmax(
    num_token=NUM_TOKEN,
    logits_slice=logits_slice,
    num_slice=logits_epoch,
    AtomPartial=ARGMAX_PARTIAL_bf16_1024_65536_128,
    AtomReduce=ARGMAX_REDUCE_bf16_1024_128,
    matLogits=matLogits,
    matOutVal=matArgOutVal,
    matOutIdx=matArgOutIdx,
    matFinalOut=matArgOut,
).place(num_sms)

dae.i(
    task_argmax,   

    TerminateC(),
    TerminateM(),
)

print("Launching Attention DAE...")

dae_app(dae)

matIn = torch.cat(matLogits, dim=-1)
def ref():
    return torch.argmax(matIn, dim=-1)

refO = ref()
daeO = matArgOut

# compare max values of all rows instead of indice as value might duplicate
refV = torch.gather(matIn, 1, refO.unsqueeze(1)).squeeze(1)
daeV = torch.gather(matIn, 1, daeO.unsqueeze(1)).squeeze(1)
tensor_diff("DAE", refV, daeV)

# total_bytes = matArgIn.nbytes * 2
# dae.bench(total_bytes=total_bytes)
