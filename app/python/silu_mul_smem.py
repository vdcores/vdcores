import torch
import copy
from math import sqrt
import torch.nn.functional as F
from dae.launcher import *
from dae.util import *
from dae.schedule import *

torch.manual_seed(0)
gpu = torch.device("cuda")

NUM_TOKEN = 8
active_token = 8
INTERM_DIM = 8192

num_sms = 128

dae = Launcher(132, device=gpu)

matGateProjOut = torch.rand(NUM_TOKEN, INTERM_DIM, dtype=torch.bfloat16, device=gpu) - 0.5
matUpProjOut = torch.rand(NUM_TOKEN, INTERM_DIM, dtype=torch.bfloat16, device=gpu) - 0.5
matOut = torch.zeros(NUM_TOKEN, INTERM_DIM, dtype=torch.bfloat16, device=gpu)

loadGate = TmaTensor(dae, matGateProjOut).wgmma_load(NUM_TOKEN, INTERM_DIM // num_sms, Major.MN)
loadUp = TmaTensor(dae, matUpProjOut).wgmma_load(NUM_TOKEN, INTERM_DIM // num_sms, Major.MN)
storeOut = TmaTensor(dae, matOut).wgmma_store(NUM_TOKEN, INTERM_DIM // num_sms, Major.MN)

def silu_and_mul(sm: int):
    if sm >= num_sms:
        return []

    return [
        SILU_MUL_SHARED_BF16_K_64_SW128(active_token),
        storeOut.cord(0, sm * (INTERM_DIM // num_sms)),
        loadGate.cord(0, sm * (INTERM_DIM // num_sms)),
        loadUp.cord(0, sm * (INTERM_DIM // num_sms)),
    ]

dae.i(
    silu_and_mul,

    TerminateC(),
    TerminateM(),
)

print("Launching Attention DAE...")


def ref():
    return F.silu(matGateProjOut) * matUpProjOut
    

dae_app(dae)

refO = ref()[:active_token]
daeO = matOut[:active_token]

tensor_diff("DAE", refO, daeO)

# find which column start to differ
# for i in range(INTERM_DIM):
#     if not torch.allclose(refO[:, i], daeO[:, i], atol=1e-3):
#         print(f"Difference starts at column {i}")
#         print(refO[:, 8191])
#         print(daeO[:, 8191])
#         break