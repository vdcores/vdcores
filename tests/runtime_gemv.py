"""Check runtime queue/input-slot lifetime with exact BF16 GEMV products.

After make pyext DAE_COMPUTE_OPS_FILE=tests/llama_runtime.ops:
    na run python tests/runtime_gemv.py
"""

import torch

from dae.instructions import Gemv_M64N8, Gemv_M64N8B2, TmaTensor
from dae.launcher import Launcher
from dae.schedule import SchedGemv
from dae.tma_utils import Major


def check(atom, sms, k):
    m, n = sms * 64, 8
    a = torch.empty((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.empty((n, k), dtype=torch.bfloat16, device="cuda")
    out = torch.empty((n, m), dtype=torch.bfloat16, device="cuda")
    launcher = Launcher(sms)
    load_a = TmaTensor(launcher, a).wgmma_load(64, 256, Major.K)
    load_b = TmaTensor(launcher, b).wgmma_load(n, 256 * atom.n_batch, Major.K)
    store = TmaTensor(launcher, out).wgmma_store(n, 64, Major.MN)
    completion = launcher.new_bar(sms)
    schedule = SchedGemv(atom, (m, n, k), (load_a, load_b, store), fold=1).place(sms)
    launcher.s(schedule.bar("store", completion))
    for _ in range(3):
        # Products are multiples of 1/256 and their FP32 sums are exact at
        # these sizes, so this catches lifetime errors without loose tolerances.
        a.copy_(torch.randint(-8, 9, a.shape, device="cuda").float() / 16)
        b.copy_(torch.randint(-8, 9, b.shape, device="cuda").float() / 16)
        out.fill_(float("nan"))
        reference = (b.float() @ a.float().t()).to(torch.bfloat16)
        launcher.launch()
        torch.testing.assert_close(out, reference, rtol=0, atol=0)
        assert launcher.bars.view(torch.uint32)[completion].item() == 0
    print(f"PASS: {atom.__name__}, {sms} SMs, K={k}", flush=True)


if __name__ == "__main__":
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    for case in ((Gemv_M64N8B2, 1, 512), (Gemv_M64N8, 1, 1024),
                 (Gemv_M64N8, 1, 14336), (Gemv_M64N8B2, 128, 4096),
                 (Gemv_M64N8, 128, 4096)):
        check(*case)
