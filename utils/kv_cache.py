import torch

class ContiguousKVCache:
    def __init__(self, num_layers, num_kv_heads, max_seq_length, head_dim, device, dtype):
        if num_layers <= 0:
            raise ValueError("num_layers is <= 0")
        self.num_layers = num_layers
        if num_kv_heads <= 0:
            raise ValueError("num_kv_heads is <= 0")
        self.num_kv_heads = num_kv_heads
        if max_seq_length <= 0:
            raise ValueError("max_seq_length is <= 0")
        self.max_seq_length = max_seq_length
        if head_dim <= 0:
            raise ValueError("head_dim is <= 0")
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
            raise ValueError("Invalid layer index")
        if start_pos < 0:
            raise ValueError("Start position cannot be negative")
        if new_keys.ndim != 3 or new_values.ndim != 3:
            raise ValueError("Keys and values must have 3 dimensions")
        if new_keys.shape != new_values.shape:
            raise ValueError("Keys and value shapes must match")
        if new_keys.shape[0] != self.num_kv_heads:
            raise ValueError("Invalid number of KV heads")
        if new_keys.shape[2] != self.head_dim:
            raise ValueError("Invalid head dimension")
        if new_keys.device != self.key_cache.device or new_values.device != self.value_cache.device:
            raise ValueError("Cache and input tensors must be on the same device")
        if new_keys.dtype != self.key_cache.dtype or new_values.dtype != self.value_cache.dtype:
            raise ValueError("Cache and input tensors must use the same dtype")
        num_new_tokens = new_keys.shape[1]
        if num_new_tokens <= 0:
            raise ValueError("Cannot append zero tokens")
        end_pos = start_pos + num_new_tokens
        if end_pos > self.max_seq_length:
            raise ValueError("KV cache capacity exceeded")
        self.key_cache[layer_index,:,start_pos:end_pos,:].copy_(new_keys)
        self.value_cache[layer_index,:,start_pos:end_pos,:].copy_(new_values)
    
    def get(self, layer_index, end_pos):
        if layer_index < 0 or layer_index >= self.num_layers:
            raise ValueError("Invalid layer index")
        if end_pos <= 0:
            raise ValueError("end_pos must be positive")
        if end_pos > self.max_seq_length:
            raise ValueError("end_pos exceeds KV cache capacity")
        keys = self.key_cache[layer_index, :, :end_pos, :]
        values = self.value_cache[layer_index, :, :end_pos, :]
        return keys, values


class KVCacheManager:
    def __init__(self, num_layers, num_kv_heads, head_dim, device, dtype):
        if num_layers <= 0:
            raise ValueError("num_layers is <= 0")
        self.num_layers = num_layers
        if num_kv_heads <= 0:
            raise ValueError("num_kv_heads is <= 0")
        self.num_kv_heads = num_kv_heads
        if head_dim <= 0:
            raise ValueError("head_dim is <= 0")
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.caches = {}
        
    def allocate(self, request_id, max_seq_length):
        if max_seq_length <= 0:
            raise ValueError("max_seq_length is <= 0")
        if request_id in self.caches:
            raise ValueError("Cache for this request already exists")
        kv_cache = ContiguousKVCache(self.num_layers, self.num_kv_heads, max_seq_length, self.head_dim, self.device, self.dtype)
        self.caches[request_id] = kv_cache
        return kv_cache
    
    def append(self, request_id, layer_index, start_pos, new_keys, new_values):
        if request_id not in self.caches:
            raise ValueError("Request does not have a KV cache")
        self.caches[request_id].append(layer_index, start_pos, new_keys, new_values)
    
    def get(self, request_id, layer_index, end_pos):
        if request_id not in self.caches:
            raise ValueError("Request does not have a KV cache")
        return self.caches[request_id].get(layer_index, end_pos)
    
    def has_cache(self, request_id):
        return request_id in self.caches
    
    def free(self, request_id):
        if request_id not in self.caches:
            raise ValueError("Request does not have a KV cache")
        del self.caches[request_id]