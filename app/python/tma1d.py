import torch
from dae.launcher import *

gpu = torch.device("cuda")

num_sms = 132
num_loads = 1024
load_bytes = 1024 * 8 # 16 KB per load

vec = torch.rand(num_sms, num_loads, load_bytes // 4, dtype=torch.float32, device=gpu)

dae = Launcher(num_sms, device=gpu)

def tasks(sm: int):
    insts = []
    for i in range(num_loads):
        insts += [TmaLoad1D(vec[sm,i,...])]
    return insts

# TODO(zhiyuang): repeat is slower
def repeat_tasks(sm: int):
    return [
        RepeatM(num_loads, delta_addr=load_bytes),
        TmaLoad1D(vec[sm,0,...]).jump()
    ]

def repeat_func_tasks(sm: int):
    return RepeatM.on(num_loads,
        [TmaLoad1D(vec[sm,0,...]), load_bytes],
    )

dae.i(
    # tasks,
    repeat_func_tasks,
    TerminateM(),

    # Compute instructions
    Dummy(num_loads),
    TerminateC()
)

dae.launch()

dae.bench(1, total_bytes = num_loads * load_bytes * num_sms)
