import torch
import numpy as np 
# from dae.instructions import paged_load
# from dae.launcher import * 

cpu = torch.device("cpu")
torch.manual_seed(0)

# page pool layout
    # page -> layers
        # layer -> some number of kv pairs for each layer 

NUM_REQ = 8 
# max sequence length for a request
KV_SEQ_LEN = 512 
LAYERS = 8

# we'll have 4 KV pairs for each layer 
NUM_KV_HEAD = 4
# size of each Key/Value tensor 
HEAD_DIM = 128 
TOKENS_PER_PAGE = 16 

# add layer size later, this is size of each token KV cache in bytes
PER_TOKEN_SIZE = LAYERS * NUM_KV_HEAD * 2 * HEAD_DIM * 2 
max_pages = (KV_SEQ_LEN * NUM_REQ) // TOKENS_PER_PAGE

avail_pages = [0, 5, 1, 3]
mem_pool = torch.rand(
    (max_pages, TOKENS_PER_PAGE, LAYERS, NUM_KV_HEAD, 2, HEAD_DIM),
    dtype = torch.bfloat16,
    device = cpu
)

# assumes we are servicing request 0 
def paged_store(token_id: int, vec : torch.tensor, page_table: torch.tensor, context_table : torch.tensor):
    
    page_index, page_offset = token_id // TOKENS_PER_PAGE, token_id % TOKENS_PER_PAGE
    page_id = context_table[0][page_index]
    
    if page_id < 0:
        context_table[0][page_index] = page_id = avail_pages.pop() 

    mem_pool[page_id][page_offset] = vec 
    page_table[page_id] = mem_pool[page_id].data_ptr()
    
    
def paged_load(token_id : int, page_table, context_table):
    page_addr = page_table[context_table[0][token_id // TOKENS_PER_PAGE]]
    token_addr = page_addr + (token_id % TOKENS_PER_PAGE) * PER_TOKEN_SIZE
    
    print(f'TOKEN {token_id}')
    print("------------------")
    
    key_addr = []
    value_addr = [] 
    
    layer_size = LAYERS * NUM_KV_HEAD * 2 * HEAD_DIM * 2 
    head_size = 2 * HEAD_DIM * 2
    
    for layer in range(0, LAYERS): 
        for head in range(0, NUM_KV_HEAD): 
            key = token_addr + layer * layer_size + head * head_size
            val = key + HEAD_DIM * 2 #value is adjacent to key
            
            key_addr.append(hex(key))
            value_addr.append(hex(val))
    
    print(f'\t key blocks   = {key_addr}')
    print(f'\t value blocks = {value_addr}')

# init the tables to max sizes ( i think 32-bit pointers )

# pre-config tables 
page_table = torch.full(
    (max_pages,), 
    fill_value = 0, 
    dtype = torch.int64,
    device = cpu
)

context_table = torch.full(
    (NUM_REQ, KV_SEQ_LEN // TOKENS_PER_PAGE),
    fill_value = -1, 
    dtype = torch.int64, 
    device = cpu
)

SEQ_LEN = 20 

for i in range(0, SEQ_LEN):
    # creating and storing 32 dummy key value blocks for a layer 
    tK = torch.rand(
        (LAYERS, NUM_KV_HEAD, 2, HEAD_DIM),
        dtype = torch.bfloat16, 
        device = cpu
    )
    
    paged_store(i, tK, page_table, context_table)

for i in range(0, SEQ_LEN):
    paged_load(i, page_table, context_table)
    print("")

