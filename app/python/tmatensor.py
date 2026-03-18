import torch
from dae.launcher import *

gpu = torch.device("cuda")

num_sms = 132
M, N = 1024, 1024
tileM, tileN = 64, 256

num_loads = M // tileM * N // tileN

vec = torch.rand(num_sms, N, M, dtype=torch.float16, device=gpu)
dae = Launcher(num_sms, device=gpu)

load = TmaTensor(dae, vec).wgmma_load(tileN, tileM, Major.MN)

def sm_task(sm: int):
    insts = []
    for m in range(0, M, tileM):
        for n in range(0, N, tileN):
            insts += [load.cord(sm, n, m)]
    return insts

def sm_repeat_task(sm: int):
    insts = []
    for m in range(0, M, tileM):
        insts += [
            RepeatM.on(N // tileN,
                [load.cord(sm, 0, m), load.cord2tma(0, tileN, 0)]
            ),
        ]
    return insts

dae.i(
    sm_task,
    TerminateM(),

    # Compute instructions
    Dummy(num_loads),
    TerminateC()
)

load_bytes = load.size
print(f"Testing TMA MN Major Load: num_loads={num_loads} size={tileN}x{tileM} load_bytes={load_bytes/1024}KB")
dae.bench(total_bytes = vec.nbytes)
