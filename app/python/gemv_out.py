import torch
from dae.launcher import *
from dae.util import *
from dae.schedule import *
from dae.model import *

gpu = torch.device("cuda")

Atom = Gemv_M64N8

M, N, K = 4096, Atom.MNK[1], 4096
num_sms = 128

matA = torch.rand(M, K, dtype=torch.bfloat16, device=gpu) - 0.5
matB = torch.rand(N, K, dtype=torch.bfloat16, device=gpu) - 0.5
matC = torch.zeros(N, M, dtype=torch.bfloat16, device=gpu)

dae = Launcher(num_sms, device=gpu)

layer = GemvLayer(dae, Atom, "out_proj", (matA, matB, matC))


dae.s(layer.schedule().place(num_sms))

dae_app(dae)

layer.diff()
