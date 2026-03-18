import torch
from dae.launcher import *
from dae.util import *

gpu = torch.device("cuda")

num_sms = 132
num_loads = 1024
load_bytes = 1024 * 8 # 16 KB per load

vec = torch.rand(num_sms, num_loads, load_bytes // 4, dtype=torch.float32, device=gpu)
out = torch.zeros(num_sms, num_loads, load_bytes // 4, dtype=torch.float32, device=gpu)

dae = Launcher(num_sms, device=gpu)

def repeat_func_tasks(sm: int):
    return RepeatM.on(num_loads,
        [TmaLoad1D(vec[sm,0,...]), load_bytes],
        [TmaStore1D(out[sm,0,...]), load_bytes],
    )


dae.i(
    Copy(num_loads, load_bytes),
    repeat_func_tasks,

    TerminateM(),
    TerminateC()
)

dae_app(dae)
tensor_diff("copy", out, vec)
