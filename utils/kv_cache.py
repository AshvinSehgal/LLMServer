import torch
from collections import deque
import math
class ContiguousKVCache:
    def __init__(self, num_layers, num_kv_heads, max_seq_length, head_dim, device, dtype):
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive: {num_layers}")
        self.num_layers = num_layers
        if num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be positive: {num_kv_heads}")
        self.num_kv_heads = num_kv_heads
        if max_seq_length <= 0:
            raise ValueError(f"max_seq_length must be positive: {max_seq_length}")
        self.max_seq_length = max_seq_length
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive: {head_dim}")
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        # Per-request cache layout:
        # key_cache[layer_index, kv_head_index, token_position, head_dimension]
        # value_cache[layer_index, kv_head_index, token_position, head_dimension]
        cache_shape = (num_layers, num_kv_heads, max_seq_length, head_dim)
        self.key_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
        self.value_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
    
    def append(self, layer_index, start_pos, new_keys, new_values):
        if layer_index < 0 or layer_index >= self.num_layers:
            raise IndexError(f"Invalid layer index: {layer_index}, num_layers: {self.num_layers}")
        if start_pos < 0:
            raise ValueError(f"Start position cannot be negative: {start_pos}")
        if new_keys.ndim != 3 or new_values.ndim != 3:
            raise ValueError(f"Keys and values must have 3 dimensions: keys: {new_keys.ndim}, values: {new_values.ndim}")
        if new_keys.shape != new_values.shape:
            raise ValueError(f"Keys and value shapes must match: keys: {tuple(new_keys.shape)}, values: {tuple(new_values.shape)}")
        if new_keys.shape[0] != self.num_kv_heads:
            raise ValueError(f"Invalid number of KV heads: {new_keys.shape[0]}, expected: {self.num_kv_heads}")
        if new_keys.shape[2] != self.head_dim:
            raise ValueError(f"Invalid head dimension: {new_keys.shape[2]}, expected: {self.head_dim}")
        if new_keys.device != self.key_cache.device or new_values.device != self.value_cache.device:
            raise ValueError(f"Cache and input tensors must be on the same device: cache: {self.key_cache.device}, keys: {new_keys.device}, values: {new_values.device}")
        if new_keys.dtype != self.key_cache.dtype or new_values.dtype != self.value_cache.dtype:
            raise ValueError(f"Cache and input tensors must use the same dtype: cache: {self.key_cache.dtype}, keys: {new_keys.dtype}, values: {new_values.dtype}")
        num_new_tokens = new_keys.shape[1]
        if num_new_tokens <= 0:
            raise ValueError(f"Cannot append zero tokens: {num_new_tokens}")
        end_pos = start_pos + num_new_tokens
        if end_pos > self.max_seq_length:
            raise KVCacheExhaustedError(f"KV cache capacity exceeded: end_pos: {end_pos}, max_seq_length: {self.max_seq_length}")
        self.key_cache[layer_index,:,start_pos:end_pos,:].copy_(new_keys)
        self.value_cache[layer_index,:,start_pos:end_pos,:].copy_(new_values)
    
    def get(self, layer_index, end_pos):
        if layer_index < 0 or layer_index >= self.num_layers:
            raise IndexError(f"Invalid layer index: {layer_index}, num_layers: {self.num_layers}")
        if end_pos <= 0:
            raise ValueError(f"end_pos must be positive: {end_pos}")
        if end_pos > self.max_seq_length:
            raise ValueError(f"end_pos exceeds KV cache capacity: end_pos: {end_pos}, max_seq_length: {self.max_seq_length}")
        keys = self.key_cache[layer_index, :, :end_pos, :]
        values = self.value_cache[layer_index, :, :end_pos, :]
        return keys, values

class PagedKVCache:
    def __init__(self, num_layers, num_blocks, block_size, num_kv_heads, head_dim, device, dtype):
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive: {num_layers}")
        self.num_layers = num_layers
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive: {num_blocks}")
        self.num_blocks = num_blocks
        if block_size <= 0:
            raise ValueError(f"block_size must be positive: {block_size}")
        self.block_size = block_size
        if num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be positive: {num_kv_heads}")
        self.num_kv_heads = num_kv_heads
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive: {head_dim}")
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        # Cache layout:
        # key_cache[num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        # value_cache[num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        cache_shape = (num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        self.key_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
        self.value_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
        self.free_blocks = deque(range(self.num_blocks))
        self.block_references = [0] * self.num_blocks
        self.num_free_blocks = self.num_blocks
        self.allocated_blocks = set()
        self.num_allocated_blocks = 0
        self.cacheable_blocks = set()
        self.block_hashes = {}

    def add_cacheable_block(self, block_id):
        if block_id not in self.allocated_blocks:
            raise KVCacheStateError(f"Block {block_id} cannot be cached, it isn't currently allocated")
        self.cacheable_blocks.add(block_id)
    
    def retain_block(self, block_id):
        if block_id < 0 or block_id >= self.num_blocks:
            raise ValueError(f"Block id {block_id} is invalid, num_blocks is {self.num_blocks}")
        if block_id not in self.allocated_blocks:
            raise KVCacheStateError(f"Block {block_id} cannot be retained, it is not allocated")
        self.block_references[block_id] += 1
    
    def free_block(self, block_id):
        if block_id < 0 or block_id >= self.num_blocks:
            raise ValueError(f"Block id {block_id} is invalid, num_blocks is {self.num_blocks}")
        if block_id not in self.allocated_blocks:
            raise KVCacheStateError(f"Block {block_id} cannot be freed, it is not allocated")
        self.block_references[block_id] -= 1
        if self.block_references[block_id] == 0:
            self.allocated_blocks.discard(block_id)
            self.num_allocated_blocks -= 1
            self.free_blocks.append(block_id)
            self.cacheable_blocks.discard(block_id)
            if block_id in self.block_hashes:
                del self.block_hashes[block_id]
            self.num_free_blocks += 1
    
    def free_block_table(self, block_table):
        unique_blocks = set()
        for block_id in block_table:
            if block_id not in self.allocated_blocks:
                raise KVCacheStateError(f"Block {block_id} cannot be freed, it is not allocated")
            if block_id in unique_blocks:
                raise ValueError(f"Duplicate block id {block_id} found")
            unique_blocks.add(block_id)
        for block_id in block_table:
            self.free_block(block_id)
        block_table.clear()
    
    def allocate_block(self):
        if not self.free_blocks:
            raise KVCacheExhaustedError(f"KV Cache is out of blocks: num_blocks: {self.num_blocks}, num_free_blocks: {self.num_free_blocks}")
        free_cache_block = self.free_blocks.popleft()
        self.num_free_blocks -= 1
        self.allocated_blocks.add(free_cache_block)
        self.num_allocated_blocks += 1
        self.block_references[free_cache_block] = 1
        return free_cache_block
    
    def get_token_location(self, block_table, token_pos):
        if token_pos < 0:
            raise ValueError(f"token_pos must be non-negative: {token_pos}")
        logical_block = token_pos // self.block_size
        if logical_block >= len(block_table):
            raise IndexError(f"Block {logical_block} is out of bounds for the block table")
        block_offset = token_pos % self.block_size
        physical_block = block_table[logical_block]
        if physical_block not in self.allocated_blocks:
            raise KVCacheStateError(f"Block {logical_block} is not an allocated block")
        return physical_block, block_offset
    
    def ensure_token_block(self, block_table, token_pos):
        if token_pos < 0:
            raise ValueError(f"token_pos must be non-negative: {token_pos}")
        logical_block = token_pos // self.block_size
        block_offset = token_pos % self.block_size          
        if logical_block < len(block_table):
            return self.get_token_location(block_table, token_pos)
        if logical_block == len(block_table):
            if block_offset == 0:
                block_id = self.allocate_block()
                block_table.append(block_id)
                return block_id, 0
            else:
                raise KVCacheStateError(f"Block offset must be 0")
        else:
            raise KVCacheStateError(f"Block {logical_block} is out of bounds for the block table")
    
    def append(self, layer_index, block_table, start_pos, new_keys, new_values):
        if layer_index < 0 or layer_index >= self.num_layers:
            raise IndexError(f"Invalid layer index: {layer_index}, num_layers: {self.num_layers}")
        if start_pos < 0:
            raise ValueError(f"Start position cannot be negative: {start_pos}")
        if new_keys.ndim != 3 or new_values.ndim != 3:
            raise ValueError(f"Keys and values must have 3 dimensions: keys: {new_keys.ndim}, values: {new_values.ndim}")
        if new_keys.shape != new_values.shape:
            raise ValueError(f"Keys and value shapes must match: keys: {tuple(new_keys.shape)}, values: {tuple(new_values.shape)}")
        if new_keys.shape[0] != self.num_kv_heads:
            raise ValueError(f"Invalid number of KV heads: {new_keys.shape[0]}, expected: {self.num_kv_heads}")
        if new_keys.shape[2] != self.head_dim:
            raise ValueError(f"Invalid head dimension: {new_keys.shape[2]}, expected: {self.head_dim}")
        if new_keys.device != self.key_cache.device or new_values.device != self.value_cache.device:
            raise ValueError(f"Cache and input tensors must be on the same device: cache: {self.key_cache.device}, keys: {new_keys.device}, values: {new_values.device}")
        if new_keys.dtype != self.key_cache.dtype or new_values.dtype != self.value_cache.dtype:
            raise ValueError(f"Cache and input tensors must use the same dtype: cache: {self.key_cache.dtype}, keys: {new_keys.dtype}, values: {new_values.dtype}")
        num_new_tokens = new_keys.shape[1]
        if num_new_tokens <= 0:
            raise ValueError(f"Cannot append zero tokens: {num_new_tokens}")
        for new_token_index in range(num_new_tokens):
            token_pos = start_pos + new_token_index
            physical_block, block_offset = self.ensure_token_block(block_table, token_pos)
            self.key_cache[layer_index, physical_block, block_offset, :, :] = new_keys[:, new_token_index, :]
            self.value_cache[layer_index, physical_block, block_offset, :, :] = new_values[:, new_token_index, :]
        
    def get(self, layer_index, block_table, end_pos):
        if layer_index < 0 or layer_index >= self.num_layers:
            raise IndexError(f"Invalid layer index: {layer_index}, num_layers: {self.num_layers}")
        if end_pos <= 0:
            raise ValueError(f"end_pos must be positive: {end_pos}")
        num_reqd_blocks = math.ceil(end_pos / self.block_size)
        if num_reqd_blocks > len(block_table):
            raise IndexError(f"Number of blocks required {num_reqd_blocks} is more than the block table has")
        keys = []
        values = []
        for i in range(num_reqd_blocks):
            physical_block = block_table[i]
            if physical_block not in self.allocated_blocks:
                raise KVCacheStateError(f"Block {physical_block} is not an allocated block")
            tokens_from_block = min(self.block_size, end_pos - (i * self.block_size))
            key_piece = self.key_cache[layer_index, physical_block, :tokens_from_block].permute(1, 0, 2)
            value_piece = self.value_cache[layer_index, physical_block, :tokens_from_block].permute(1, 0, 2)
            keys.append(key_piece)
            values.append(value_piece)
        return torch.cat(keys, dim=1), torch.cat(values, dim=1)
            

class KVCacheManager:
    def __init__(self, num_layers, num_kv_heads, head_dim, device, dtype):
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive: {num_layers}")
        self.num_layers = num_layers
        if num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be positive: {num_kv_heads}")
        self.num_kv_heads = num_kv_heads
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive: {head_dim}")
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.caches = {}
        
    def allocate(self, request_id, max_seq_length):
        if max_seq_length <= 0:
            raise ValueError(f"max_seq_length must be positive: {max_seq_length}")
        if request_id in self.caches:
            raise ValueError(f"Cache for this request already exists: {request_id}")
        kv_cache = ContiguousKVCache(self.num_layers, self.num_kv_heads, max_seq_length, self.head_dim, self.device, self.dtype)
        self.caches[request_id] = kv_cache
        return kv_cache
    
    def append(self, request_id, layer_index, start_pos, new_keys, new_values):
        if request_id not in self.caches:
            raise KeyError(f"Request does not have a KV cache: {request_id}")
        self.caches[request_id].append(layer_index, start_pos, new_keys, new_values)
    
    def get(self, request_id, layer_index, end_pos):
        if request_id not in self.caches:
            raise KeyError(f"Request does not have a KV cache: {request_id}")
        return self.caches[request_id].get(layer_index, end_pos)
    
    def has_cache(self, request_id):
        return request_id in self.caches
    
    def free(self, request_id):
        if request_id not in self.caches:
            raise KeyError(f"Request does not have a KV cache: {request_id}")
        del self.caches[request_id]

class KVCacheExhaustedError(BufferError):
    pass

class KVCacheStateError(RuntimeError):
    pass
