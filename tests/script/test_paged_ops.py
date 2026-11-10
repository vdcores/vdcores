import struct

import torch

from dae.instructions import Dummy, PagedLoad1D, PagedStore1D, TmaLoad1D, Copy
from dae.launcher import Launcher
from dae.paged import PagedCacheConfig


# Model Params
LAYERS = 8
HEADS = 4
HEAD_DIM = 128

# Session Management
SESSIONS = 2
MAX_SEQ_LEN = 64

# Paged KV Cache Management
TOKENS_PER_PAGE = 16
NUM_PAGES = (SESSIONS * MAX_SEQ_LEN) // TOKENS_PER_PAGE



page_pool = torch.rand(
    (NUM_PAGES, TOKENS_PER_PAGE, LAYERS, HEADS, 2, HEAD_DIM),
    dtype = torch.bfloat16,
    device = "cuda"
)

page_table = torch.tensor(
    [page_pool[i].data_ptr() for i in range(NUM_PAGES)],
    dtype = torch.int64, 
    device = "cuda"
)

context_table = torch.tensor(
    [[0, 1, 2, 3], [4, 5, 6, 7]], 
    dtype = torch.int64, 
    device = "cuda"
)

cache = PagedCacheConfig(page_table=page_table, context_table=context_table,
                            tokens_per_page=TOKENS_PER_PAGE, kv_seq_len=MAX_SEQ_LEN,
                            num_req=SESSIONS, layers=LAYERS,
                            num_kv_heads=HEADS, head_dim=HEAD_DIM)

launcher = Launcher(1, "cuda")

# tQ = torch.full(
#     (LAYERS, HEADS, 2, HEAD_DIM),
#     fill_value = 2, 
#     dtype = torch.bfloat16,
#     device = "cuda"
# )

tK = torch.full(
    (LAYERS, HEADS, 2, HEAD_DIM), 
    fill_value = 3, 
    dtype = torch.bfloat16,
    device = "cuda"
)

launcher.s(TmaLoad1D(tK), PagedStore1D(0, 0, cache), Copy(1, cache.bytes_per_token()))
launcher.launch() 


assert torch.equal(page_pool[0, 0], tK)