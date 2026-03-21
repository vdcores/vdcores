import torch
import copy
from math import sqrt
import torch.nn.functional as F
from dae.launcher import *
from dae.util import *
from dae.schedule import SchedRMS, SchedRMSShared

gpu = torch.device("cuda")

NUM_TOKEN = 8
HID_DIM = 4096

num_sms = 8
assert NUM_TOKEN % num_sms == 0

dae = Launcher(num_sms, device=gpu)

matWeights = torch.rand((HID_DIM), dtype=torch.bfloat16, device=gpu) - 0.5
matIn = torch.full((NUM_TOKEN, HID_DIM), 1.0, dtype=torch.bfloat16, device=gpu)
matOut = torch.zeros_like(matIn)

loadWeights = TmaTensor(dae, matWeights).tensor1d("load", HID_DIM)
    
def task_rms_norm(sm: int):
    start_token_id = sm * (NUM_TOKEN // num_sms)
    insts = [
        RMS_NORM_F16_K_4096(NUM_TOKEN // num_sms, 1.0),
        RawAddress(matWeights, 23),
        RawAddress(matIn[start_token_id], 24),
        RawAddress(matOut[start_token_id], 25),
    ]
    return insts

def task_rms_norm_smem(sm: int):
    start_token_id = sm * (NUM_TOKEN // num_sms)
    end_token_id = (sm + 1) * (NUM_TOKEN // num_sms)
    insts = [
        RMS_NORM_F16_K_4096_SMEM(NUM_TOKEN // num_sms, 1.0),
        # TmaLoad1D(matWeights),
        loadWeights.cord(0),
        TmaLoad1D(matIn[start_token_id:end_token_id]),
        TmaStore1D(matOut[start_token_id:end_token_id]),
    ]
    return insts

rms = SchedRMSShared(
    num_token=NUM_TOKEN,
    epsilon=1.0,
    tmas = (
        loadWeights.cord(0),
        TmaLoad1D(matIn, bytes=HID_DIM * 2),
        TmaStore1D(matOut, bytes=HID_DIM * 2),
    ),
).place(num_sms)

dae.s(
    # task_rms_norm_smem,
    rms
)

print("Launching Attention DAE...")

dae.launch()

def ref():
    var = matIn.pow(2).mean(dim=-1, keepdim=True)
    X = matIn * torch.rsqrt(var + 1.0)
    X = X * matWeights
    return X

refO = ref()
daeO = matOut

tensor_diff("DAE", refO, daeO)

dae_app(dae)
