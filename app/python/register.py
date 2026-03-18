import torch
from dae.launcher import *
from dae.util import *

gpu = torch.device("cuda")

num_sms = 132
load_bytes = 8192

mat = torch.zeros(num_sms, load_bytes // 2, dtype=torch.bfloat16, device=gpu)

dae = Launcher(num_sms, device=gpu)

def tasks(sm: int):
    return [
        Dummy(2),

        RegStore(0, mat[sm,...]),
        RegLoad(0)
    ]


dae.i(
    tasks,

    TerminateM(),
    TerminateC()
)

dae_app(dae)
