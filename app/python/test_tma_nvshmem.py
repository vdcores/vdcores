# import torch
# import dae.nvshmem as nvshmem
# from dae import runtime
# from dae.tma_utils import build_tma_wgmma_kmajor

# runtime_ctx = nvshmem.init(symmetric_size="512M")
# gpu = torch.device("cuda")
# dtype = torch.bfloat16

# K = 512
# MAX_TOKENS_PER_EXPERT = 256

# plain = torch.zeros(MAX_TOKENS_PER_EXPERT, K, dtype=dtype, device=gpu)
# print("plain tensor:", flush=True)
# try:
#     rank, desc = build_tma_wgmma_kmajor(plain, K, MAX_TOKENS_PER_EXPERT)
#     print("  OK, rank =", rank, flush=True)
# except Exception as exc:
#     print("  FAILED:", exc, flush=True)

# sym = nvshmem.zeros(MAX_TOKENS_PER_EXPERT * K * 4, dtype=dtype)
# sym_view = sym[: MAX_TOKENS_PER_EXPERT * K].view(MAX_TOKENS_PER_EXPERT, K)
# print("nvshmem tensor:", flush=True)
# try:
#     rank, desc = build_tma_wgmma_kmajor(sym_view, K, MAX_TOKENS_PER_EXPERT)
#     print("  OK, rank =", rank, flush=True)
# except Exception as exc:
#     print("  FAILED:", exc, flush=True)

# nvshmem.finalize()

import torch
import dae.nvshmem as nvshmem
from dae import runtime
from math import prod

nvshmem.init(symmetric_size="512M")
gpu = torch.device("cuda")
dtype = torch.bfloat16

K = 512
MAX_TOKENS_PER_EXPERT = 256

mat = torch.zeros(MAX_TOKENS_PER_EXPERT, K, dtype=dtype, device=gpu)

tileK = K
tileN = MAX_TOKENS_PER_EXPERT

N, Kd = mat.shape[-2], mat.shape[-1]
elsize = mat.element_size()
blockK = 128 // elsize

print(f"N={N} K={Kd} elsize={elsize} blockK={blockK} tileK={tileK} tileN={tileN}", flush=True)
print(f"tileK % blockK == 0: {tileK % blockK == 0}", flush=True)

global_dims = [blockK, N, Kd // blockK]
global_strides = [Kd * elsize, blockK * elsize]
box_dims = [blockK, tileN, tileK // blockK]
box_strides = [1, 1, 1]

print(f"global_dims={global_dims}", flush=True)
print(f"global_strides={global_strides}", flush=True)
print(f"box_dims={box_dims}", flush=True)
print(f"box_strides={box_strides}", flush=True)

try:
    desc = runtime.build_tma_desc(
        mat, global_dims, global_strides, box_dims, box_strides, 128, 0
    )
    print("OK", flush=True)
except Exception as exc:
    print("FAILED:", exc, flush=True)

nvshmem.finalize()