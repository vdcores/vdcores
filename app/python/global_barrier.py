import torch
from dae.launcher import *
from dae.util import *
from dae.schedule import *
from dae.model import *

gpu = torch.device("cuda")

Atom = Gemv_M64N16

M, N, K = 4096, Atom.MNK[1], 4096
num_sms = 128

matW1 = torch.rand(M, K, dtype=torch.bfloat16, device=gpu) - 0.5
matIn = torch.rand(N, K, dtype=torch.bfloat16, device=gpu) - 0.5
matInterm = torch.zeros(N, M, dtype=torch.bfloat16, device=gpu)
matW2 = torch.rand(M, K, dtype=torch.bfloat16, device=gpu) - 0.5
matOut = torch.zeros(N, M, dtype=torch.bfloat16, device=gpu)

dae = Launcher(num_sms, device=gpu)

bar_id = dae.new_bar(num_sms)

l1 = GemvLayer(dae, Atom, "proj1", (matW1, matIn, matInterm))
l2 = GemvLayer(dae, Atom, "proj2", (matW2, matInterm, matOut))

dae.s(
    l1.schedule().bar("store", bar_id),
    l2.schedule().bar("load", bar_id)
    # l1.schedule(),
    # l2.schedule(),
)

dae_app(dae)

l1.diff()
l2.diff()
