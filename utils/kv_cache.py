import numpy as np
import torch
from collections import deque, OrderedDict
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
        self.vectorized_prefill_write = torch.device(device).type == "cuda"
        self.slot_memo = {}
        self.free_blocks = deque(range(self.num_blocks))
        self.block_references = [0] * self.num_blocks
        self.num_free_blocks = self.num_blocks
        self.allocated_blocks = set()
        self.num_allocated_blocks = 0
        self.cacheable_blocks = set()
        self.block_hashes = {}
        self.hash_to_block_id = {}
        self.idle_lru = OrderedDict()
        self._batch_block_ids_key = None
        self._batch_block_ids = None
        self._ragged_block_ids_key = None
        self._ragged_block_ids = None
        self._ragged_mask_key = None
        self._ragged_mask = None
        self.contiguous_read_fast_path = True

    def add_cacheable_block(self, block_id):
        if block_id not in self.allocated_blocks:
            raise KVCacheStateError(f"Block {block_id} cannot be cached, it isn't currently allocated")
        self.cacheable_blocks.add(block_id)
    
    def retain_block(self, block_id):
        if block_id < 0 or block_id >= self.num_blocks:
            raise ValueError(f"Block id {block_id} is invalid, num_blocks is {self.num_blocks}")
        if block_id not in self.allocated_blocks:
            raise KVCacheStateError(f"Block {block_id} cannot be retained, it is not allocated")
        if self.block_references[block_id] == 0:
            self.idle_lru.pop(block_id, None)
        self.block_references[block_id] += 1
    
    def share_block(self, digest, new_request_block_table):
        block_id = self.hash_to_block_id.get(digest, None)
        if block_id is None or block_id not in self.allocated_blocks or block_id not in self.cacheable_blocks:
            return None
        block_id_digest = self.block_hashes.get(block_id, None)
        if block_id_digest is None or block_id_digest != digest:
            return None
        if block_id in new_request_block_table:
            raise ValueError("One request must not list the same physical block table twice")
        self.retain_block(block_id)
        new_request_block_table.append(block_id)
        return block_id

    def free_block(self, block_id):
        if block_id < 0 or block_id >= self.num_blocks:
            raise ValueError(f"Block id {block_id} is invalid, num_blocks is {self.num_blocks}")
        if block_id not in self.allocated_blocks:
            raise KVCacheStateError(f"Block {block_id} cannot be freed, it is not allocated")
        if self.block_references[block_id] == 0:
            raise KVCacheStateError(f"Block {block_id} cannot be freed, it has no active references")
        self.block_references[block_id] -= 1
        if self.block_references[block_id] > 0:
            return
        digest = self.block_hashes.get(block_id)
        if block_id in self.cacheable_blocks and digest is not None and self.hash_to_block_id.get(digest) == block_id:
            self.idle_lru[block_id] = digest
            return
        self.discard_block(block_id)

    def discard_block(self, block_id):
        self.idle_lru.pop(block_id, None)
        digest = self.block_hashes.pop(block_id, None)
        if digest is not None and self.hash_to_block_id.get(digest) == block_id:
            del self.hash_to_block_id[digest]
        self.allocated_blocks.discard(block_id)
        self.num_allocated_blocks -= 1
        self.free_blocks.append(block_id)
        self.cacheable_blocks.discard(block_id)
        self.num_free_blocks += 1
    
    def free_block_table(self, block_table):
        unique_blocks = set()
        for block_id in block_table:
            if block_id not in self.allocated_blocks:
                raise KVCacheStateError(f"Block {block_id} cannot be freed, it is not allocated")
            if self.block_references[block_id] == 0:
                raise KVCacheStateError(f"Block {block_id} cannot be freed, it has no active references")
            if block_id in unique_blocks:
                raise ValueError(f"Duplicate block id {block_id} found")
            unique_blocks.add(block_id)
        for block_id in block_table:
            self.free_block(block_id)
        block_table.clear()
    
    def allocate_block(self):
        if not self.free_blocks:
            if not self.idle_lru:
                raise KVCacheExhaustedError(f"KV Cache is out of blocks: num_blocks: {self.num_blocks}, num_free_blocks: {self.num_free_blocks}")
            oldest_block_id = next(iter(self.idle_lru))
            self.discard_block(oldest_block_id)
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
        if self.vectorized_prefill_write and num_new_tokens > 1:
            # One scatter per tensor instead of two small copies per touched block: with 16-token blocks a
            # 1024-token prompt otherwise launches thousands of tiny kernels per forward pass.
            slots = self.prefill_slots(block_table, start_pos, num_new_tokens)
            self.key_cache[layer_index].view(-1, self.num_kv_heads, self.head_dim).index_copy_(0, slots, new_keys.permute(1, 0, 2).contiguous())
            self.value_cache[layer_index].view(-1, self.num_kv_heads, self.head_dim).index_copy_(0, slots, new_values.permute(1, 0, 2).contiguous())
            return
        new_token_index = 0
        while new_token_index < num_new_tokens:
            token_pos = start_pos + new_token_index
            physical_block, block_offset = self.ensure_token_block(block_table, token_pos)
            tokens_in_block = min(self.block_size - block_offset, num_new_tokens - new_token_index)
            source_slice = slice(new_token_index, new_token_index + tokens_in_block)
            target_slice = slice(block_offset, block_offset + tokens_in_block)
            self.key_cache[layer_index, physical_block, target_slice].copy_(new_keys[:, source_slice].permute(1, 0, 2))
            self.value_cache[layer_index, physical_block, target_slice].copy_(new_values[:, source_slice].permute(1, 0, 2))
            new_token_index += tokens_in_block

    def prefill_slots(self, block_table, start_pos, num_tokens):
        """Device tensor of flat cache slots for tokens [start_pos, start_pos + num_tokens), allocating blocks as needed.

        The result depends only on (block table contents, start, count), so it is memoized: every layer of a forward
        pass asks for the same range.
        """
        last_block = (start_pos + num_tokens - 1) // self.block_size
        if last_block >= len(block_table):
            for logical_block in range(len(block_table), last_block + 1):
                self.ensure_token_block(block_table, max(start_pos, logical_block * self.block_size))
        key = (tuple(block_table), start_pos, num_tokens)
        slots = self.slot_memo.get(key)
        if slots is None:
            if len(self.slot_memo) >= 256:
                self.slot_memo.clear()
            positions = np.arange(start_pos, start_pos + num_tokens)
            blocks = np.asarray(block_table, dtype=np.int64)[positions // self.block_size]
            slots = torch.from_numpy(blocks * self.block_size + positions % self.block_size).to(self.key_cache.device)
            self.slot_memo[key] = slots
        return slots

    def slot_for(self, block_table, token_pos):
        """Flat cache slot (block * block_size + offset) for a token, allocating its block if needed."""
        physical_block, block_offset = self.ensure_token_block(block_table, token_pos)
        return physical_block * self.block_size + block_offset

    def write_slots(self, layer_index, slot_mapping, keys, values):
        """Write one new token per row. keys/values: [rows, num_kv_heads, head_dim]; slot_mapping: [rows] int64 flat slots.

        Uses device-side indices only (no Python shapes), so it can be captured in a CUDA graph. Rows that share a slot
        must carry identical data (padded graph rows duplicate row 0).
        """
        flat_keys = self.key_cache[layer_index].view(-1, self.num_kv_heads, self.head_dim)
        flat_values = self.value_cache[layer_index].view(-1, self.num_kv_heads, self.head_dim)
        flat_keys.index_copy_(0, slot_mapping, keys)
        flat_values.index_copy_(0, slot_mapping, values)

    def get(self, layer_index, block_table, end_pos):
        if layer_index < 0 or layer_index >= self.num_layers:
            raise IndexError(f"Invalid layer index: {layer_index}, num_layers: {self.num_layers}")
        if end_pos <= 0:
            raise ValueError(f"end_pos must be positive: {end_pos}")
        num_reqd_blocks = math.ceil(end_pos / self.block_size)
        if num_reqd_blocks > len(block_table):
            raise IndexError(f"Number of blocks required {num_reqd_blocks} is more than the block table has")
        physical_blocks = block_table[:num_reqd_blocks]
        if self.contiguous_read_fast_path and all(block_id in self.allocated_blocks and block_id == physical_blocks[0] + index for index, block_id in enumerate(physical_blocks)):
            first_block = physical_blocks[0]
            last_block = first_block + num_reqd_blocks
            keys = self.key_cache[layer_index, first_block:last_block].reshape(-1, self.num_kv_heads, self.head_dim)[:end_pos].permute(1, 0, 2)
            values = self.value_cache[layer_index, first_block:last_block].reshape(-1, self.num_kv_heads, self.head_dim)[:end_pos].permute(1, 0, 2)
            return keys, values
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

    def get_batch(self, layer_index, block_tables, end_pos):
        if layer_index < 0 or layer_index >= self.num_layers:
            raise IndexError(f"Invalid layer index: {layer_index}, num_layers: {self.num_layers}")
        if end_pos <= 0:
            raise ValueError(f"end_pos must be positive: {end_pos}")
        if not block_tables:
            raise ValueError("At least one block table is required")
        num_required_blocks = math.ceil(end_pos / self.block_size)
        for block_table in block_tables:
            if len(block_table) < num_required_blocks:
                raise IndexError(f"Number of blocks required {num_required_blocks} is more than the block table has")
            for block_id in block_table[:num_required_blocks]:
                if block_id not in self.allocated_blocks:
                    raise KVCacheStateError(f"Block {block_id} is not an allocated block")
        block_ids_key = tuple(tuple(table[:num_required_blocks]) for table in block_tables)
        if block_ids_key != self._batch_block_ids_key:
            self._batch_block_ids = torch.tensor(block_ids_key, device=self.device, dtype=torch.long).flatten()
            self._batch_block_ids_key = block_ids_key
        shape = (len(block_tables), num_required_blocks * self.block_size, self.num_kv_heads, self.head_dim)
        keys = self.key_cache[layer_index].index_select(0, self._batch_block_ids).reshape(shape)[:, :end_pos].permute(0, 2, 1, 3).contiguous()
        values = self.value_cache[layer_index].index_select(0, self._batch_block_ids).reshape(shape)[:, :end_pos].permute(0, 2, 1, 3).contiguous()
        return keys, values

    def get_batch_ragged(self, layer_index, block_tables, end_positions):
        if layer_index < 0 or layer_index >= self.num_layers:
            raise IndexError(f"Invalid layer index: {layer_index}, num_layers: {self.num_layers}")
        if not block_tables or len(block_tables) != len(end_positions):
            raise ValueError(f"Block tables and end positions must have the same nonzero length: tables: {len(block_tables)}, positions: {len(end_positions)}")
        if any(end_pos <= 0 for end_pos in end_positions):
            raise ValueError(f"End positions must be positive: {end_positions}")
        max_end_pos = max(end_positions)
        max_blocks = math.ceil(max_end_pos / self.block_size)
        pad_block_id = block_tables[0][0] if block_tables[0] else None
        padded_tables = []
        for block_table, end_pos in zip(block_tables, end_positions):
            required_blocks = math.ceil(end_pos / self.block_size)
            if len(block_table) < required_blocks:
                raise IndexError(f"Number of blocks required {required_blocks} is more than the block table has")
            used_blocks = block_table[:required_blocks]
            for block_id in used_blocks:
                if block_id not in self.allocated_blocks:
                    raise KVCacheStateError(f"Block {block_id} is not an allocated block")
            padded_tables.append(tuple(used_blocks + [pad_block_id] * (max_blocks - required_blocks)))
        block_ids_key = tuple(padded_tables)
        if block_ids_key != self._ragged_block_ids_key:
            self._ragged_block_ids = torch.tensor(block_ids_key, device=self.device, dtype=torch.long).flatten()
            self._ragged_block_ids_key = block_ids_key
        mask_key = tuple(end_positions)
        if mask_key != self._ragged_mask_key:
            self._ragged_mask = (torch.arange(max_end_pos, device=self.device).unsqueeze(0) < torch.tensor(end_positions, device=self.device).unsqueeze(1))[:, None, None, None, :]
            self._ragged_mask_key = mask_key
        shape = (len(block_tables), max_blocks * self.block_size, self.num_kv_heads, self.head_dim)
        keys = self.key_cache[layer_index].index_select(0, self._ragged_block_ids).reshape(shape)[:, :max_end_pos].permute(0, 2, 1, 3).contiguous()
        values = self.value_cache[layer_index].index_select(0, self._ragged_block_ids).reshape(shape)[:, :max_end_pos].permute(0, 2, 1, 3).contiguous()
        return keys, values, self._ragged_mask
            

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
