import torch
from dae.launcher import *
from dae.util import *

# this test simulates a multi-stage data-dependent flow
# try to figure out if prefetch is useful in this case

gpu = torch.device("cuda")

Up192M, N, Up192K = Gemv_M192N16.MNK
UpTileM, N, UpTileK = Gemv_M128N16.MNK
DownTileM, N, DownTileK = Gemv_M64N16.MNK

HIDDEN, INTERMIDIATE = 4096, 4096 * 6
n_batch = 4

num_up_sms = 128
num_down_sms = 128
num_sms = max(num_up_sms, num_down_sms)

matHidden = torch.rand(N, HIDDEN, dtype=torch.float16, device=gpu) - 0.5
matUp = torch.rand(INTERMIDIATE, HIDDEN, dtype=torch.float16, device=gpu) - 0.5
matInterm = torch.zeros(N, INTERMIDIATE, dtype=torch.float16, device=gpu)
matDown = torch.rand(HIDDEN, INTERMIDIATE // 2, dtype=torch.float16, device=gpu) - 0.5
matOut = torch.zeros(N, HIDDEN, dtype=torch.float16, device=gpu)

blockDown = torch.rand(128, INTERMIDIATE // 2 // DownTileK // 2, DownTileK * DownTileM, dtype=torch.float16, device=gpu)
blockUp1 = torch.rand(128, HIDDEN // UpTileK, UpTileK * UpTileM, dtype=torch.float16, device=gpu)
blockUp2 = torch.rand(128, HIDDEN // DownTileK, DownTileK * DownTileM, dtype=torch.float16, device=gpu)
# for any K-major layout, if N == 8, it will be 128B (64 elem) aligned chucked
blockHidden = torch.rand(HIDDEN, N, dtype=torch.float16, device=gpu)
blockInterm = torch.rand(INTERMIDIATE // 2, N, dtype=torch.float16, device=gpu)
blockOut = torch.rand(HIDDEN, N, dtype=torch.float16, device=gpu)
print(blockDown.shape)

dae = Launcher(num_sms, device=gpu)

loadHidden = TmaTensor(dae, matHidden).wgmma_load(N, UpTileK * n_batch, Major.K)
loadUp = TmaTensor(dae, matUp).wgmma_load(UpTileM, UpTileK, Major.K)
storeInterm = TmaTensor(dae, matInterm).wgmma_store(N, UpTileM, Major.MN)

loadHidden2 = TmaTensor(dae, matHidden).wgmma_load(N, DownTileK * n_batch, Major.K)
loadUp2 = TmaTensor(dae, matUp).wgmma_load(DownTileM, DownTileK, Major.K)
storeInterm2 = TmaTensor(dae, matInterm).wgmma_store(N, DownTileM, Major.MN)

n_batch_192 = 8
loadHidden192 = TmaTensor(dae, matHidden).wgmma_load(N, Up192M * n_batch_192, Major.K)
loadUp192 = TmaTensor(dae, matUp).wgmma_load(Up192M, Up192K, Major.K)
storeInterm192 = TmaTensor(dae, matInterm).wgmma_store(N, Up192M, Major.MN)

# schedule for block tensors
loadHiddenBlock = TmaTensor(dae, blockHidden).wgmma_load(N, DownTileK * n_batch, Major.K)
loadUpBlock = TmaTensor(dae, blockUp1).wgmma_load(UpTileM, UpTileK, Major.K)
storeIntermBlock = TmaTensor(dae, blockInterm).wgmma_store(N, UpTileM, Major.MN)

loadInterm = TmaTensor(dae, matInterm).wgmma_load(N, DownTileK * n_batch, Major.K)
loadDown = TmaTensor(dae, matDown).wgmma_load(DownTileM, DownTileK, Major.K)
reduceOut = TmaTensor(dae, matOut).wgmma("reduce", N, DownTileM, Major.MN)

def up_single_epoch(sm: int):
    assert Up192M * 128 == INTERMIDIATE
    m = sm * Up192M
    return [
        Gemv_M192N16(HIDDEN // Up192K),
        RepeatM.on(HIDDEN // Up192K // n_batch_192,
            [loadHidden192.cord(0, 0).port(1), loadHidden192.cord2tma(0, n_batch_192 * Up192M)],
            *[
                [TmaLoad1D(blockUp192[sm, i, ...]), n_batch_192 * Up192M * Up192K * 2]
                for i in range(n_batch_192)
            ]
        ),
        storeInterm192.cord(0, m).bar(),
    ]

def up(sm: int):
    # wave 1
    m = sm * UpTileM
    insts = [
        Gemv_M128N16(HIDDEN // UpTileK),
        [
            RepeatM.on(HIDDEN // UpTileK // n_batch,
                # (TmaLoad1D(blockHidden[0:n_batch * UpTileK,:]), n_batch * UpTileK * N * 2),
                (loadHidden.cord(0, 0).port(1), loadHidden.cord2tma(0, n_batch * UpTileK)),
                # *[
                #     [TmaLoad1D(blockUp1[sm, i, ...]), n_batch * UpTileM * UpTileK * 2]
                #     for i in range(n_batch)
                # ]
                *[
                    [loadUp.cord(m, UpTileK * i), loadUp.cord2tma(0, n_batch * UpTileK)]
                    for i in range(n_batch)
                ]
            ),
            # TODO(zhiyuang): bar forces wait here. correct?
            storeInterm.cord(0, m),
        ]
    ]
    # wave 2
    m = UpTileM * num_up_sms + sm * DownTileM
    insts += [
        Gemv_M64N16(HIDDEN // DownTileK),
        [
            RepeatM.on(HIDDEN // DownTileK // n_batch,
                # (TmaLoad1D(blockHidden[0:n_batch * DownTileK,:]), n_batch * DownTileK * N * 2),
                [loadHidden2.cord(0, 0).port(1), loadHidden2.cord2tma(0, n_batch * DownTileK)],
                *[
                    [TmaLoad1D(blockUp2[sm, i, ...]), n_batch * DownTileM * DownTileK * 2]
                    for i in range(n_batch)
                ]
            ),
            storeInterm2.cord(0, m).bar(),
        ]
    ]
    return insts

bar_up = storeInterm2

nstage_down = 2
def down(sm: int):
    assert sm <= num_down_sms

    stage = sm // 64
    sm = sm % 64

    K = INTERMIDIATE // 2 // nstage_down
    k_start = stage * K
    
    m = sm * DownTileM

    return [
        Gemv_M64N16(K // DownTileK),
        
        # rest of the body
        RepeatM.on(K // DownTileK // n_batch,
            # (TmaLoad1D(blockInterm[k_start:k_start + n_batch * DownTileK,:]), n_batch * DownTileK * N * 2),
            [loadInterm.cord(0, k_start + n_batch * DownTileK).port(1), loadInterm.cord2tma(0, n_batch * DownTileK)],
            *[
                (TmaLoad1D(blockDown[sm, i, ...]), n_batch * DownTileM * DownTileK * 2)
                for i in range(n_batch)
            ]
            # *[
            #     [loadDown.cord(m, k_start + (i + n_batch) * DownTileK), loadDown.cord2tma(0, n_batch * DownTileK)]
            #     for i in range(n_batch)
            # ]
        ),
        reduceOut.cord(0, m).bar(),
    ]
    
dae.i(
    up,
    down,
    GlobalBarrier(num_down_sms, reduceOut),

    TerminateC(),
    TerminateM(),
)

print(f"MLP GEMV on [H={HIDDEN}, INTERM={INTERMIDIATE}, N={N}], up SMs={num_up_sms}, down SMs={num_down_sms}:")
print("theory load speed:", (matHidden.nbytes / 8 + matUp.nbytes + matInterm.nbytes / 8 + matDown.nbytes) / 1024 ** 3 / 3700 * 1e6, "us")
print("theory load speed (no L2):", (matHidden.nbytes * num_up_sms + matUp.nbytes + matInterm.nbytes * num_down_sms + matDown.nbytes) / 1024 ** 3 / 3700 * 1e6, "us")
print()

def gemv_ref():
    interm = matHidden @ matUp.T
    out = interm[:, :INTERMIDIATE // 2] @ matDown.T
    return interm, out

interm, out = gemv_ref()

dae_app(dae)

tensor_diff("interm", interm, matInterm)
tensor_diff("out", out, matOut)
