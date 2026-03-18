import torch
from dae.launcher import *
from dae.util import *

gpu = torch.device("cuda")

num_sms = 1
num_loads = 64
load_bytes = 1024 * 8 # 16 KB per load

vec = torch.rand(num_sms, num_loads, load_bytes // 4, dtype=torch.float32, device=gpu)
out = torch.zeros_like(vec, device=gpu)

dae = Launcher(num_sms, device=gpu)

def tasks(sm: int):
    insts = []
    for i in range(num_loads):
        insts += [TmaLoad1D(vec[sm,i,...])]
        insts += [TmaStore1D(out[sm,i,...])]
    return insts

dae.i(
    tasks,
    TerminateM(),

    # Compute instructions
    Dummy(num_loads * 2),
    TerminateC()
)

dae_app(dae)

print("Verifying results...")

tensor_diff("TMA 1D Load/Store", vec, out)