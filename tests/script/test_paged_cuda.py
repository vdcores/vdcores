import struct
import random
import torch

from dae.instructions import Dummy, Copy, TmaStore1D, TmaLoad1D, PagedLoad1D, PagedStore1D
from dae.launcher import Launcher
from dae.paged import PagedCacheConfig


# TODO: Remove all of these constants later
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

# Contiguous layout: [page, token, layer, head, K/V, dim].
class PagedMemoryTestHarness:
    """
    Initializes memory pool, page table, context table, and cache to test loads and stores. 
    Currently uses the constants defined above, but likely to instead use parameters in the future. 
    """
    def __init__(self):        
        self.pool = torch.full(
            (NUM_PAGES, TOKENS_PER_PAGE, LAYERS, NUM_KV_HEAD, NUM_KV_VECTORS, HEAD_DIM), -1, 
            dtype=DTYPE, device=DEVICE,
        )
        
        self.permutation = list(range(NUM_PAGES))
        random.Random(42).shuffle(self.permutation)
        
        page_table = torch.tensor([self.pool[i].data_ptr() for i in self.permutation], dtype=torch.int64, device=DEVICE)

        self.context_table = torch.tensor(((5, 0, 3), (1, 4, 2)), dtype=torch.int64, device=DEVICE)
        
        self.cache = PagedCacheConfig(page_table=page_table, context_table=self.context_table,
                                tokens_per_page=TOKENS_PER_PAGE, kv_seq_len=KV_SEQ_LEN,
                                num_req=NUM_REQ, layers=LAYERS,
                                num_kv_heads=NUM_KV_HEAD, head_dim=HEAD_DIM)

    """
    Tests PagedLoad1D by storing a dummy vector, load to SMEM by PagedLoad, and checking output. 
    """
    def load_test(self, token_id: int, session_id: int):
        launcher = Launcher(1, "cuda")
        
        expected = torch.rand((LAYERS, NUM_KV_HEAD, 2, HEAD_DIM), dtype=DTYPE, device=DEVICE)
        
        actual = torch.zeros((LAYERS, NUM_KV_HEAD, 2, HEAD_DIM), dtype = DTYPE, device= DEVICE)
        
        logical_page = token_id // TOKENS_PER_PAGE
        page_offset = token_id % TOKENS_PER_PAGE
        
        page_id = self.context_table[session_id][logical_page].item()
        
        self.pool[self.permutation[page_id]][page_offset] = expected
        
        launcher.s(PagedLoad1D(session_id = session_id, token_id = token_id, cache = self.cache), Copy(1, EXPECTED_TOKEN_BYTES), TmaStore1D(dst = actual))
        launcher.launch()
        
        torch.cuda.synchronize()
        
        assert torch.equal(expected, actual), (
            f"PagedLoad1D failed for session={session_id}, "
            f"token={token_id}, "
            f"logical_page={logical_page}, "
            f"physical_page={page_id}, "
            f"pool_index={self.permutation[page_id]}"
        )
        
        print(f'Passed PagedLoad1D test (token, session) = ({token_id}, {session_id})')
        
    """
    Test for PagedStore1D. First issues a load using TmaLoad1D (assumed to be reliable), issues a copy into working space slots, 
    then issues a PagedStore into global memory given the token_id and session_id. Check confirms that the store is as expected. 
    All loads and stores operate on single-token KV cache. 
    """
    def store_test(self, token_id: int, session_id: int):
        launcher = Launcher(1, "cuda")
        
        expected = torch.rand((LAYERS, NUM_KV_HEAD, 2, HEAD_DIM), dtype= DTYPE, device =DEVICE)
        
        launcher.s(
            TmaLoad1D(src = expected), 
            Copy(1, EXPECTED_TOKEN_BYTES), 
            PagedStore1D(token_id = token_id, session_id = session_id, cache = self.cache)
        )
        
        launcher.launch()
        
        torch.cuda.synchronize() 
        
        page_id = self.permutation[self.context_table[session_id][token_id // TOKENS_PER_PAGE].item()]
        page_offset = token_id % TOKENS_PER_PAGE
        
        actual = self.pool[page_id][page_offset]
        
        assert torch.equal(expected, actual), (
            f"PagedStore1D failed for session={session_id}, "
            f"token={token_id}, "
            f"physical_page={page_id}, "
            f"pool_index={self.permutation[page_id]}"
        )
        
        print(f'Passed PagedStore1D test (token, session) = ({token_id}, {session_id})')
    
    """
    Verifies that PagedStore1D only changes the intended portion of the pool. 
    """
    def store_test_pool(self, token_id: int, session_id: int): 
        launcher = Launcher(1, "cuda")
        
        expected_pool = self.pool.clone()
        
        tK = torch.rand((LAYERS, NUM_KV_HEAD, 2, HEAD_DIM), dtype= DTYPE, device =DEVICE)

        page_id = self.permutation[self.context_table[session_id][token_id // TOKENS_PER_PAGE].item()]
        page_offset = token_id % TOKENS_PER_PAGE
        
        expected_pool[page_id][page_offset] = tK 

        launcher.s(
            TmaLoad1D(src = tK), 
            Copy(1, EXPECTED_TOKEN_BYTES), 
            PagedStore1D(token_id = token_id, session_id = session_id, cache = self.cache)
        )
        
        launcher.launch() 
        
        torch.cuda.synchronize() 
        
        assert torch.equal(expected_pool, self.pool), (
            f"PagedStore1D failed (pool-wide) for session={session_id}, "
            f"token={token_id}, "
            f"physical_page={page_id}, "
            f"pool_index={self.permutation[page_id]}"
        )
        
        print(f'PagedStore1D passed (pool-wide) for (token, session) = ({token_id}, {session_id})')
        
        

def main():
    harness = PagedMemoryTestHarness()
    
    for session in range(NUM_REQ):
        for token in range(TOKENS_PER_PAGE * PAGES_PER_SESSION):
            harness.load_test(token_id = token, session_id=session)
    
    for session in range(NUM_REQ):
        for token in range(TOKENS_PER_PAGE * PAGES_PER_SESSION):
            harness.store_test(token_id = token, session_id = session)

    for session in range(NUM_REQ):
        for token in range(TOKENS_PER_PAGE * PAGES_PER_SESSION):
            harness.store_test_pool(token_id = token, session_id = session)


if __name__ == "__main__":
    main()
