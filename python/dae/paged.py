import operator

from . import runtime

class PagedCacheConfig:
    def __init__(self, *, page_table, context_table, tokens_per_page,
                 kv_seq_len, num_req, layers, num_kv_heads, head_dim):
        
        self.page_table = page_table
        self.context_table = context_table
        self.tokens_per_page = tokens_per_page
        
        self.layers = layers 
        self.num_kv_heads = num_kv_heads 
        self.head_dim = head_dim 
        
        self.kv_seq_len = kv_seq_len
        self.num_req = num_req
        
        self.descriptor = runtime.build_paged_config(
            page_table, context_table, self.tokens_per_page, self.kv_seq_len,
            self.num_req, self.layers, self.num_kv_heads, self.head_dim,
        )
    
    
    def bytes_per_token(self):
        # Assumes BF16 type
        return self.layers * self.num_kv_heads * 2 * self.head_dim * 2


__all__ = ["PagedCacheConfig"]
