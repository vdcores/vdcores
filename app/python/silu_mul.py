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
INTERM_DIM = 4096

num_sms = 4

dae = Launcher(132, device=gpu)

matGateProjOut = torch.rand(NUM_TOKEN, INTERM_DIM, dtype=torch.bfloat16, device=gpu) - 0.5
matUpProjOut = torch.rand(NUM_TOKEN, INTERM_DIM, dtype=torch.bfloat16, device=gpu) - 0.5
matOut = torch.zeros(NUM_TOKEN, INTERM_DIM, dtype=torch.bfloat16, device=gpu)

def silu_and_mul(sm: int):
    if sm >= num_sms:
        return []

    start_token_id = sm * (NUM_TOKEN // num_sms)
    end_token_id = (sm + 1) * (NUM_TOKEN // num_sms)
    return [
        SILU_MUL_SHARED_BF16_K_4096_INTER(NUM_TOKEN // num_sms),
        TmaStore1D(matOut[start_token_id:end_token_id]),
        TmaLoad1D(matGateProjOut[start_token_id:end_token_id]),
        TmaLoad1D(matUpProjOut[start_token_id:end_token_id]),
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

refO = ref()
daeO = matOut

tensor_diff("DAE", refO, daeO)

# find which column start to differ
# for i in range(INTERM_DIM):
#     if not torch.allclose(refO[:, i], daeO[:, i], atol=1e-3):
#         print(f"Difference starts at column {i}")
#         print(refO[:, 8191])
#         print(daeO[:, 8191])
#         break