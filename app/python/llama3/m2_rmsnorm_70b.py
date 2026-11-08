# M2 Part A: RMSNorm at Llama 3.1 70B's hidden size (8192), checked against PyTorch.
#
#   python m2_rmsnorm_70b.py -l                    correctness
#   python m2_rmsnorm_70b.py -b 100                correctness + timing
#   HID_DIM=4096 python m2_rmsnorm_70b.py -b 100   8B baseline to compare against
import os
import torch
from dae.launcher import *
from dae.util import dae_app, tensor_diff
from dae.schedule import SchedRMSShared

gpu = torch.device("cuda")
torch.manual_seed(0)

HID_DIM = int(os.environ.get("HID_DIM", "8192"))  # 8192 = 70B, 4096 = 8B
NUM_TOKEN = 8   # same as REQ in sched_70b.py
num_sms = 8     # one token per SM, same split the 70B app uses
EPS = 1e-5      # Llama 3.1 rms_norm_eps

dae = Launcher(num_sms, device=gpu)

# norm weights near 1, like trained weights; input centered on 0
matWeights = (0.75 + 0.5 * torch.rand(HID_DIM, device=gpu)).to(torch.bfloat16)
matIn = (torch.rand(NUM_TOKEN, HID_DIM, device=gpu) - 0.5).to(torch.bfloat16)
# NaN = "never written", so a skipped token can't pass as zeros
matOut = torch.full_like(matIn, float("nan"))

loadWeights = TmaTensor(dae, matWeights).tensor1d("load", HID_DIM)

rms = SchedRMSShared(
    num_token=NUM_TOKEN,
    epsilon=EPS,
    tmas=(
        loadWeights.cord(0),
        TmaLoad1D(matIn, bytes=HID_DIM * 2),
        TmaStore1D(matOut, bytes=HID_DIM * 2),
    ),
    hidden_size=HID_DIM,
).place(num_sms)

dae.s(rms)
dae_app(dae)


def ref():
    x = matIn.float()
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    return (x * matWeights.float()).to(torch.bfloat16)


assert not matOut.isnan().any(), "some outputs never written (run with -l or -b)"
tensor_diff(f"RMSNorm K={HID_DIM}", ref(), matOut)
