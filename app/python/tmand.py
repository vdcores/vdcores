import torch
from dae.launcher import *
from dae.util import *

gpu = torch.device("cuda")

TileM, TileK = 64, 128
M, K = 4096, 4096*6

num_sms = 128
dae = Launcher(num_sms, device=gpu)

matA = torch.rand(M, K, dtype=torch.float16, device=gpu) - 0.5
loadA = TmaTensor(dae, matA).wgmma_load(TileM, TileK, Major.K)

# matA = torch.rand(K, M, dtype=torch.float16, device=gpu) - 0.5
# loadA = TmaTensor(dae, matA).wgmma_load(TileK, TileM, Major.MN)

num_loads = K // TileK 
def sm_task(sm: int):
    m = TileM * sm
    return [
        Dummy(num_loads),
        RepeatM.on(K // TileK,
            (loadA.cord(m, 0), loadA.cord2tma(0, TileK))
        ),
    ]

dae.i(
    sm_task,

    TerminateC(),
    TerminateM(),
)

print("load size", loadA.size)

dae_app(dae, total_bytes = matA.nbytes)

cycles = dae.profile[:,3].cpu().numpy()
events = dae.profile[:,2].cpu().numpy()
print(f"average cycles: {cycles.mean()}")
print(f"average stall: {events.mean()}")