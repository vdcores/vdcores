import struct

import torch

from dae.instructions import Dummy, PagedLoad1D
from dae.launcher import Launcher
from dae.paged import PagedCacheConfig


NUM_REQ = 2
PAGES_PER_SESSION = 3
TOKENS_PER_PAGE = 16
KV_SEQ_LEN = PAGES_PER_SESSION * TOKENS_PER_PAGE
NUM_PAGES = NUM_REQ * PAGES_PER_SESSION
LAYERS = 8
NUM_KV_HEAD = 4
HEAD_DIM = 128
KEY_INDEX, VALUE_INDEX = 0, 1
NUM_KV_VECTORS = 2
DTYPE = torch.bfloat16
BF16_ELEMENT_BYTES = 2
DEVICE = "cuda"
NUM_SMS = 1

# Scrambled physical pages expose assumptions about contiguous logical pages.
CONTEXT_MAPPING = ((5, 0, 3), (1, 4, 2))

# Check the first and last token of each page.
BOUNDARY_TOKENS = []
for page in range(PAGES_PER_SESSION):
    first_token = page * TOKENS_PER_PAGE
    last_token = first_token + TOKENS_PER_PAGE - 1
    BOUNDARY_TOKENS.extend((first_token, last_token))

EXPECTED_TOKEN_BYTES = (
    LAYERS * NUM_KV_HEAD * NUM_KV_VECTORS * HEAD_DIM * BF16_ELEMENT_BYTES
)
# Native byte order, two 8-byte address integers, six 4-byte dimensions,
# no padding. This test assumes the current 64-bit native struct layout.
DESCRIPTOR_FORMAT = "=2Q6I"


def main():
    # Contiguous layout: [page, token, layer, head, K/V, dim].
    pool = torch.empty(
        (NUM_PAGES, TOKENS_PER_PAGE, LAYERS, NUM_KV_HEAD, NUM_KV_VECTORS, HEAD_DIM),
        dtype=DTYPE, device=DEVICE,
    )
    page_table = torch.tensor([pool[i].data_ptr() for i in range(NUM_PAGES)],
                              dtype=torch.int64, device=DEVICE)

    context_table = torch.tensor(CONTEXT_MAPPING, dtype=torch.int64, device=DEVICE)
    
    cache = PagedCacheConfig(page_table=page_table, context_table=context_table,
                             tokens_per_page=TOKENS_PER_PAGE, kv_seq_len=KV_SEQ_LEN,
                             num_req=NUM_REQ, layers=LAYERS,
                             num_kv_heads=NUM_KV_HEAD, head_dim=HEAD_DIM)
    
    # Verify C++ packed the input pointers and dimensions in the expected order.
    # This layout inspection is test-only; production packing stays in C++.
    descriptor = bytes(cache.descriptor.cpu().tolist())
    expected_fields = (
        page_table.data_ptr(), context_table.data_ptr(),
        TOKENS_PER_PAGE, KV_SEQ_LEN, NUM_REQ, LAYERS, NUM_KV_HEAD, HEAD_DIM,
    )
    assert struct.unpack(DESCRIPTOR_FORMAT, descriptor) == expected_fields
    
    assert cache.bytes_per_token() == EXPECTED_TOKEN_BYTES
    
    # Test verifies device printed address at boundaries matches what we expect 
    launcher = Launcher(NUM_SMS, DEVICE)
    for session in range(NUM_REQ):
        for token in BOUNDARY_TOKENS:
            page_index, page_offset = divmod(token, TOKENS_PER_PAGE)
            physical = int(context_table[session, page_index].item())
            for layer in range(LAYERS):
                for head in range(NUM_KV_HEAD):
                    key = pool[physical, page_offset, layer, head, KEY_INDEX].data_ptr()
                    value = pool[physical, page_offset, layer, head, VALUE_INDEX].data_ptr()
                    print(f"EXPECT session={session} token={token} layer={layer} "
                          f"head={head} K=0x{key:x} V=0x{value:x}", flush=True)
            launcher.i(PagedLoad1D(token, session, cache), Dummy(1))
    launcher.s()
    launcher.launch()
    
    # Tests two failure cases: session with logical page unmapped; page table having null pointer to pool
    # Checks device code still completes schedule (rather than wait indef) and prints error reason. 
    failure_token = BOUNDARY_TOKENS[0]
    failure_page_index = failure_token // TOKENS_PER_PAGE
    unmapped_session, null_page_session = 0, 1
    null_page_id = int(context_table[null_page_session, failure_page_index].item())
    context_table[unmapped_session, failure_page_index] = -1
    page_table[null_page_id] = 0
    failures = Launcher(NUM_SMS, DEVICE)
    failure_loads = (
        PagedLoad1D(failure_token, unmapped_session, cache),
        PagedLoad1D(failure_token, null_page_session, cache),
    )
    failures.s(*failure_loads, Dummy(len(failure_loads)))
    failures.launch()
    torch.cuda.synchronize()
    print("Native descriptor and paged/Dummy schedules completed.")


if __name__ == "__main__":
    main()
