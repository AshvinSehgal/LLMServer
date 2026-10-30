import gc
import weakref
import torch
import pytest
from utils.kv_cache import ContiguousKVCache, PagedKVCache, KVCacheManager, KVCacheExhaustedError, KVCacheStateError

def test_contiguous_kv_cache():
    cache = ContiguousKVCache(num_layers=3, num_kv_heads=2, max_seq_length=8, head_dim=4, device=torch.device("cpu"), dtype=torch.float32)
    assert cache.num_layers == 3
    assert cache.num_kv_heads == 2
    assert cache.max_seq_length == 8
    assert cache.head_dim == 4
    assert cache.device == torch.device("cpu")
    assert cache.dtype == torch.float32
    assert cache.key_cache.shape == (3, 2, 8, 4)
    assert cache.value_cache.shape == (3, 2, 8, 4)
    assert cache.key_cache.device == torch.device("cpu")
    assert cache.value_cache.device == torch.device("cpu")
    assert cache.key_cache.dtype == torch.float32
    assert cache.value_cache.dtype == torch.float32
    assert torch.count_nonzero(cache.key_cache).item() == 0
    assert torch.count_nonzero(cache.value_cache).item() == 0

def test_contiguous_kv_cache_layout():
    cache = ContiguousKVCache(num_layers=2, num_kv_heads=3, max_seq_length=5, head_dim=4, device=torch.device("cpu"), dtype=torch.float32)
    assert cache.key_cache.shape == (2, 3, 5, 4)
    assert cache.value_cache.shape == (2, 3, 5, 4)
    assert cache.key_cache[1].shape == (3, 5, 4)
    assert cache.value_cache[1].shape == (3, 5, 4)
    assert cache.key_cache[1, :, 2, :].shape == (3, 4)
    assert cache.value_cache[1, :, 2, :].shape == (3, 4)
    assert cache.key_cache[1, :, :3, :].shape == (3, 3, 4)
    assert cache.value_cache[1, :, :3, :].shape == (3, 3, 4)
    assert cache.key_cache.is_contiguous()
    assert cache.value_cache.is_contiguous()
    cache.key_cache[1, 2, 3, 0] = 7
    cache.value_cache[0, 1, 4, 2] = 9
    assert cache.key_cache[1, 2, 3, 0].item() == 7
    assert cache.value_cache[0, 1, 4, 2].item() == 9
    assert torch.count_nonzero(cache.key_cache).item() == 1
    assert torch.count_nonzero(cache.value_cache).item() == 1

def test_paged_kv_cache():
    cache = PagedKVCache(num_layers=3, num_blocks=8, block_size=5, num_kv_heads=2, head_dim=4, device=torch.device("cpu"), dtype=torch.float64)
    assert cache.num_layers == 3
    assert cache.num_blocks == 8
    assert cache.block_size == 5
    assert cache.num_kv_heads == 2
    assert cache.head_dim == 4
    assert cache.device == torch.device("cpu")
    assert cache.dtype == torch.float64
    assert cache.key_cache.shape == (3, 8, 5, 2, 4)
    assert cache.value_cache.shape == (3, 8, 5, 2, 4)
    assert cache.key_cache.device == torch.device("cpu")
    assert cache.value_cache.device == torch.device("cpu")
    assert cache.key_cache.dtype == torch.float64
    assert cache.value_cache.dtype == torch.float64
    assert cache.key_cache.is_contiguous()
    assert cache.value_cache.is_contiguous()
    assert torch.count_nonzero(cache.key_cache).item() == 0
    assert torch.count_nonzero(cache.value_cache).item() == 0

def test_paged_kv_cache_layout():
    cache = PagedKVCache(num_layers=2, num_blocks=4, block_size=5, num_kv_heads=3, head_dim=4, device=torch.device("cpu"), dtype=torch.float32)
    assert cache.key_cache[1].shape == (4, 5, 3, 4)
    assert cache.value_cache[1].shape == (4, 5, 3, 4)
    assert cache.key_cache[1, 2].shape == (5, 3, 4)
    assert cache.value_cache[1, 2].shape == (5, 3, 4)
    assert cache.key_cache[1, 2, 3].shape == (3, 4)
    assert cache.value_cache[1, 2, 3].shape == (3, 4)

def test_paged_kv_cache_physical_blocks_are_isolated():
    cache = PagedKVCache(num_layers=2, num_blocks=4, block_size=5, num_kv_heads=3, head_dim=4, device=torch.device("cpu"), dtype=torch.float32)
    key_data_ptr = cache.key_cache.data_ptr()
    value_data_ptr = cache.value_cache.data_ptr()
    cache.key_cache[1, 2, 3, 0, 1] = 7
    cache.value_cache[0, 3, 4, 2, 3] = 9
    assert cache.key_cache[1, 2, 3, 0, 1].item() == 7
    assert cache.value_cache[0, 3, 4, 2, 3].item() == 9
    assert torch.count_nonzero(cache.key_cache).item() == 1
    assert torch.count_nonzero(cache.value_cache).item() == 1
    assert torch.count_nonzero(cache.key_cache[1, 1]).item() == 0
    assert torch.count_nonzero(cache.value_cache[0, 2]).item() == 0
    assert cache.key_cache.data_ptr() == key_data_ptr
    assert cache.value_cache.data_ptr() == value_data_ptr

def test_paged_kv_cache_starts_with_every_block_free():
    cache = PagedKVCache(num_layers=2, num_blocks=4, block_size=5, num_kv_heads=3, head_dim=4, device=torch.device("cpu"), dtype=torch.float32)
    assert list(cache.free_blocks) == [0, 1, 2, 3]
    assert cache.block_references == [0, 0, 0, 0]
    assert cache.allocated_blocks == set()
    assert cache.num_free_blocks == 4
    assert cache.num_allocated_blocks == 0
    assert cache.cacheable_blocks == set()
    assert cache.block_hashes == {}

def test_paged_kv_cache_marks_allocated_block_cacheable_until_last_reference_is_freed():
    cache = PagedKVCache(num_layers=1, num_blocks=1, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    block_id = cache.allocate_block()
    cache.add_cacheable_block(block_id)
    cache.block_hashes[block_id] = "prompt-hash"
    assert cache.cacheable_blocks == {block_id}
    assert cache.block_hashes == {block_id: "prompt-hash"}
    cache.retain_block(block_id)
    cache.free_block(block_id)
    assert cache.cacheable_blocks == {block_id}
    assert cache.block_hashes == {block_id: "prompt-hash"}
    cache.free_block(block_id)
    assert cache.cacheable_blocks == set()
    assert cache.block_hashes == {}
    assert cache.allocate_block() == block_id
    assert cache.cacheable_blocks == set()
    assert cache.block_hashes == {}

def test_paged_kv_cache_can_free_partial_block_without_hash():
    cache = PagedKVCache(num_layers=1, num_blocks=1, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    block_id = cache.allocate_block()
    cache.free_block(block_id)
    assert cache.block_hashes == {}
    assert cache.num_free_blocks == 1

def test_paged_kv_cache_rejects_marking_unallocated_block_cacheable():
    cache = PagedKVCache(num_layers=1, num_blocks=2, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(KVCacheStateError):
        cache.add_cacheable_block(0)
    block_id = cache.allocate_block()
    cache.free_block(block_id)
    with pytest.raises(KVCacheStateError):
        cache.add_cacheable_block(block_id)
    assert cache.cacheable_blocks == set()

def test_paged_kv_cache_allocates_blocks_in_order_without_reallocating_storage():
    cache = PagedKVCache(num_layers=2, num_blocks=4, block_size=5, num_kv_heads=3, head_dim=4, device=torch.device("cpu"), dtype=torch.float32)
    key_data_ptr = cache.key_cache.data_ptr()
    value_data_ptr = cache.value_cache.data_ptr()
    allocated_blocks = [cache.allocate_block() for _ in range(3)]
    assert allocated_blocks == [0, 1, 2]
    assert cache.block_references == [1, 1, 1, 0]
    assert list(cache.free_blocks) == [3]
    assert cache.allocated_blocks == {0, 1, 2}
    assert cache.num_free_blocks == len(cache.free_blocks) == 1
    assert cache.num_allocated_blocks == len(cache.allocated_blocks) == 3
    assert cache.key_cache.data_ptr() == key_data_ptr
    assert cache.value_cache.data_ptr() == value_data_ptr

def test_paged_kv_cache_rejects_allocation_when_exhausted():
    cache = PagedKVCache(num_layers=2, num_blocks=2, block_size=5, num_kv_heads=3, head_dim=4, device=torch.device("cpu"), dtype=torch.float32)
    assert cache.allocate_block() == 0
    assert cache.allocate_block() == 1
    with pytest.raises(KVCacheExhaustedError):
        cache.allocate_block()
    assert cache.num_free_blocks == len(cache.free_blocks) == 0
    assert cache.num_allocated_blocks == len(cache.allocated_blocks) == 2

def test_paged_kv_cache_frees_and_reuses_block():
    cache = PagedKVCache(num_layers=2, num_blocks=1, block_size=5, num_kv_heads=3, head_dim=4, device=torch.device("cpu"), dtype=torch.float32)
    block_id = cache.allocate_block()
    cache.free_block(block_id)
    assert cache.block_references == [0]
    assert list(cache.free_blocks) == [0]
    assert cache.allocated_blocks == set()
    assert cache.num_free_blocks == len(cache.free_blocks) == 1
    assert cache.num_allocated_blocks == len(cache.allocated_blocks) == 0
    assert cache.allocate_block() == block_id
    assert cache.block_references == [1]

def test_paged_kv_cache_shared_block_stays_allocated_until_last_request_finishes():
    cache = PagedKVCache(num_layers=1, num_blocks=2, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    first_block_table = []
    keys = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    values = keys + 100
    cache.append(layer_index=0, block_table=first_block_table, start_pos=0, new_keys=keys, new_values=values)
    block_id = first_block_table[0]
    cache.retain_block(block_id)
    second_block_table = [block_id]
    assert cache.block_references == [2, 0]
    assert cache.allocated_blocks == {block_id}
    assert cache.num_allocated_blocks == 1
    assert cache.num_free_blocks == 1
    cache.free_block_table(first_block_table)
    assert first_block_table == []
    assert second_block_table == [block_id]
    assert cache.block_references == [1, 0]
    assert cache.allocated_blocks == {block_id}
    assert list(cache.free_blocks) == [1]
    cached_keys, cached_values = cache.get(layer_index=0, block_table=second_block_table, end_pos=4)
    assert torch.equal(cached_keys, keys)
    assert torch.equal(cached_values, values)
    cache.free_block_table(second_block_table)
    assert second_block_table == []
    assert cache.block_references == [0, 0]
    assert cache.allocated_blocks == set()
    assert list(cache.free_blocks) == [1, 0]
    assert cache.num_allocated_blocks == 0
    assert cache.num_free_blocks == 2

def test_paged_kv_cache_share_block_reuses_values_and_releases_on_last_reference():
    cache = PagedKVCache(num_layers=1, num_blocks=2, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    first_block_table = []
    second_block_table = []
    keys = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    values = keys + 100
    cache.append(layer_index=0, block_table=first_block_table, start_pos=0, new_keys=keys, new_values=values)
    block_id = first_block_table[0]
    cache.add_cacheable_block(block_id)
    cache.block_hashes[block_id] = "shared-prefix"
    cache.hash_to_block_id["shared-prefix"] = block_id
    assert cache.share_block("shared-prefix", second_block_table) == block_id
    assert second_block_table == first_block_table == [block_id]
    assert cache.block_references[block_id] == 2
    assert cache.num_allocated_blocks == 1
    cached_keys, cached_values = cache.get(layer_index=0, block_table=second_block_table, end_pos=4)
    assert torch.equal(cached_keys, keys)
    assert torch.equal(cached_values, values)
    cache.free_block_table(first_block_table)
    assert cache.block_references[block_id] == 1
    assert cache.hash_to_block_id == {"shared-prefix": block_id}
    third_block_table = []
    assert cache.share_block("shared-prefix", third_block_table) == block_id
    assert cache.block_references[block_id] == 2
    cache.free_block_table(third_block_table)
    cache.free_block_table(second_block_table)
    assert cache.block_references[block_id] == 0
    assert cache.allocated_blocks == set()
    assert cache.cacheable_blocks == set()
    assert cache.block_hashes == {}
    assert cache.hash_to_block_id == {}

def test_paged_kv_cache_share_block_misses_without_valid_cache_entry():
    cache = PagedKVCache(num_layers=1, num_blocks=2, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    second_block_table = []
    assert cache.share_block("missing", second_block_table) is None
    block_id = cache.allocate_block()
    cache.hash_to_block_id["prefix"] = block_id
    assert cache.share_block("prefix", second_block_table) is None
    cache.add_cacheable_block(block_id)
    cache.block_hashes[block_id] = "different-prefix"
    assert cache.share_block("prefix", second_block_table) is None
    assert second_block_table == []
    assert cache.block_references[block_id] == 1

def test_paged_kv_cache_share_block_handles_missing_hash_metadata():
    cache = PagedKVCache(num_layers=1, num_blocks=1, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    block_id = cache.allocate_block()
    cache.add_cacheable_block(block_id)
    cache.hash_to_block_id["prefix"] = block_id
    second_block_table = []
    assert cache.share_block("prefix", second_block_table) is None
    assert second_block_table == []
    assert cache.block_references[block_id] == 1

def test_paged_kv_cache_share_block_rejects_duplicate_entry_without_retaining():
    cache = PagedKVCache(num_layers=1, num_blocks=1, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    block_id = cache.allocate_block()
    cache.add_cacheable_block(block_id)
    cache.block_hashes[block_id] = "prefix"
    cache.hash_to_block_id["prefix"] = block_id
    block_table = [block_id]
    with pytest.raises(ValueError):
        cache.share_block("prefix", block_table)
    assert block_table == [block_id]
    assert cache.block_references[block_id] == 1

def test_paged_kv_cache_frees_only_matching_hash_lookup():
    cache = PagedKVCache(num_layers=1, num_blocks=2, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    first_block_id = cache.allocate_block()
    second_block_id = cache.allocate_block()
    cache.add_cacheable_block(first_block_id)
    cache.add_cacheable_block(second_block_id)
    cache.block_hashes[first_block_id] = "prefix"
    cache.block_hashes[second_block_id] = "prefix"
    cache.hash_to_block_id["prefix"] = first_block_id
    cache.free_block(second_block_id)
    assert cache.hash_to_block_id == {"prefix": first_block_id}
    cache.free_block(first_block_id)
    assert cache.hash_to_block_id == {}

def test_paged_kv_cache_shared_prefix_and_private_block_release_independently():
    cache = PagedKVCache(num_layers=1, num_blocks=3, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    first_block_table = [cache.allocate_block(), cache.allocate_block()]
    cache.retain_block(first_block_table[0])
    second_block_table = [first_block_table[0], cache.allocate_block()]
    assert cache.block_references == [2, 1, 1]
    cache.free_block_table(first_block_table)
    assert cache.block_references == [1, 0, 1]
    assert cache.allocated_blocks == {0, 2}
    assert list(cache.free_blocks) == [1]
    cache.free_block_table(second_block_table)
    assert cache.block_references == [0, 0, 0]
    assert cache.allocated_blocks == set()
    assert cache.num_free_blocks == 3

@pytest.mark.parametrize("block_id", [-1, 2])
def test_paged_kv_cache_rejects_invalid_block_id_on_retain(block_id):
    cache = PagedKVCache(num_layers=1, num_blocks=2, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(ValueError):
        cache.retain_block(block_id)
    assert cache.block_references == [0, 0]
    assert cache.num_allocated_blocks == 0

def test_paged_kv_cache_rejects_retaining_unallocated_or_freed_block():
    cache = PagedKVCache(num_layers=1, num_blocks=2, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(KVCacheStateError):
        cache.retain_block(0)
    block_id = cache.allocate_block()
    cache.free_block(block_id)
    with pytest.raises(KVCacheStateError):
        cache.retain_block(block_id)
    assert cache.block_references == [0, 0]
    assert cache.allocated_blocks == set()
    assert list(cache.free_blocks) == [1, 0]

@pytest.mark.parametrize("block_id", [-1, 3])
def test_paged_kv_cache_rejects_invalid_block_id_on_free(block_id):
    cache = PagedKVCache(num_layers=2, num_blocks=3, block_size=5, num_kv_heads=3, head_dim=4, device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(ValueError):
        cache.free_block(block_id)

def test_paged_kv_cache_rejects_unallocated_and_double_free():
    cache = PagedKVCache(num_layers=2, num_blocks=3, block_size=5, num_kv_heads=3, head_dim=4, device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(KVCacheStateError):
        cache.free_block(0)
    block_id = cache.allocate_block()
    cache.free_block(block_id)
    with pytest.raises(KVCacheStateError):
        cache.free_block(block_id)
    assert list(cache.free_blocks) == [1, 2, 0]
    assert cache.num_free_blocks == len(cache.free_blocks) == 3
    assert cache.num_allocated_blocks == len(cache.allocated_blocks) == 0

def test_paged_kv_cache_frees_request_block_table_and_reuses_blocks():
    cache = PagedKVCache(num_layers=2, num_blocks=3, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    block_table = []
    cache.append(layer_index=0, block_table=block_table, start_pos=0, new_keys=torch.ones(2, 5, 3), new_values=torch.ones(2, 5, 3))
    assert block_table == [0, 1]
    cache.free_block_table(block_table)
    assert block_table == []
    assert list(cache.free_blocks) == [2, 0, 1]
    assert cache.allocated_blocks == set()
    assert cache.num_free_blocks == len(cache.free_blocks) == 3
    assert cache.num_allocated_blocks == len(cache.allocated_blocks) == 0
    assert [cache.allocate_block() for _ in range(3)] == [2, 0, 1]

def test_paged_kv_cache_can_allocate_after_request_releases_exhausted_cache():
    cache = PagedKVCache(num_layers=1, num_blocks=1, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    first_block_table = [cache.allocate_block()]
    with pytest.raises(KVCacheExhaustedError):
        cache.allocate_block()
    cache.free_block_table(first_block_table)
    second_block_table = [cache.allocate_block()]
    assert first_block_table == []
    assert second_block_table == [0]
    assert cache.allocated_blocks == {0}

def test_paged_kv_cache_frees_only_selected_request_blocks():
    cache = PagedKVCache(num_layers=1, num_blocks=4, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    first_block_table = [cache.allocate_block(), cache.allocate_block()]
    second_block_table = [cache.allocate_block()]
    cache.free_block_table(first_block_table)
    assert first_block_table == []
    assert second_block_table == [2]
    assert cache.allocated_blocks == {2}
    assert list(cache.free_blocks) == [3, 0, 1]
    assert cache.num_free_blocks == 3
    assert cache.num_allocated_blocks == 1

def test_paged_kv_cache_repeated_request_cleanup_is_safe():
    cache = PagedKVCache(num_layers=1, num_blocks=2, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    block_table = [cache.allocate_block()]
    cache.free_block_table(block_table)
    cache.free_block_table(block_table)
    assert block_table == []
    assert list(cache.free_blocks) == [1, 0]
    assert cache.allocated_blocks == set()
    assert cache.num_free_blocks == 2
    assert cache.num_allocated_blocks == 0

def test_paged_kv_cache_rejects_stale_request_block_table_without_partial_free():
    cache = PagedKVCache(num_layers=1, num_blocks=3, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    first_block = cache.allocate_block()
    stale_block = cache.allocate_block()
    cache.free_block(stale_block)
    block_table = [first_block, stale_block]
    with pytest.raises(KVCacheStateError):
        cache.free_block_table(block_table)
    assert block_table == [first_block, stale_block]
    assert cache.allocated_blocks == {first_block}
    assert list(cache.free_blocks) == [2, stale_block]
    assert cache.num_free_blocks == 2
    assert cache.num_allocated_blocks == 1

def test_paged_kv_cache_rejects_duplicate_request_block_without_partial_free():
    cache = PagedKVCache(num_layers=1, num_blocks=2, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    block_id = cache.allocate_block()
    block_table = [block_id, block_id]
    with pytest.raises(ValueError):
        cache.free_block_table(block_table)
    assert block_table == [block_id, block_id]
    assert cache.allocated_blocks == {block_id}
    assert list(cache.free_blocks) == [1]
    assert cache.num_free_blocks == 1
    assert cache.num_allocated_blocks == 1

@pytest.mark.parametrize("token_pos,expected_location", [(0, (0, 0)), (3, (0, 3)), (4, (2, 0)), (5, (2, 1)), (7, (2, 3))])
def test_paged_kv_cache_gets_token_location_across_block_boundaries(token_pos, expected_location):
    cache = PagedKVCache(num_layers=2, num_blocks=4, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    first_block = cache.allocate_block()
    cache.allocate_block()
    second_block = cache.allocate_block()
    block_table = [first_block, second_block]
    assert block_table == [0, 2]
    assert cache.get_token_location(block_table, token_pos) == expected_location

def test_paged_kv_cache_token_locations_are_request_specific():
    cache = PagedKVCache(num_layers=2, num_blocks=4, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    first_request_block_table = [cache.allocate_block()]
    second_request_block_table = [cache.allocate_block()]
    assert cache.get_token_location(first_request_block_table, 3) == (0, 3)
    assert cache.get_token_location(second_request_block_table, 3) == (1, 3)

def test_paged_kv_cache_token_location_does_not_mutate_allocator_state():
    cache = PagedKVCache(num_layers=2, num_blocks=4, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    block_table = [cache.allocate_block(), cache.allocate_block()]
    free_blocks = list(cache.free_blocks)
    allocated_blocks = cache.allocated_blocks.copy()
    num_free_blocks = cache.num_free_blocks
    num_allocated_blocks = cache.num_allocated_blocks
    key_data_ptr = cache.key_cache.data_ptr()
    value_data_ptr = cache.value_cache.data_ptr()
    assert cache.get_token_location(block_table, 6) == (1, 2)
    assert list(cache.free_blocks) == free_blocks
    assert cache.allocated_blocks == allocated_blocks
    assert cache.num_free_blocks == num_free_blocks
    assert cache.num_allocated_blocks == num_allocated_blocks
    assert cache.key_cache.data_ptr() == key_data_ptr
    assert cache.value_cache.data_ptr() == value_data_ptr

def test_paged_kv_cache_rejects_negative_token_position():
    cache = PagedKVCache(num_layers=2, num_blocks=4, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    block_table = [cache.allocate_block()]
    with pytest.raises(ValueError):
        cache.get_token_location(block_table, -1)

@pytest.mark.parametrize("block_table,token_pos", [([], 0), ([0], 4)])
def test_paged_kv_cache_rejects_token_without_logical_block(block_table, token_pos):
    cache = PagedKVCache(num_layers=2, num_blocks=4, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    if block_table:
        assert cache.allocate_block() == block_table[0]
    with pytest.raises(IndexError):
        cache.get_token_location(block_table, token_pos)

def test_paged_kv_cache_rejects_unallocated_physical_block_in_table():
    cache = PagedKVCache(num_layers=2, num_blocks=4, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(KVCacheStateError):
        cache.get_token_location([2], 0)

def test_paged_kv_cache_rejects_freed_physical_block_in_table():
    cache = PagedKVCache(num_layers=2, num_blocks=4, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    block_id = cache.allocate_block()
    block_table = [block_id]
    cache.free_block(block_id)
    with pytest.raises(KVCacheStateError):
        cache.get_token_location(block_table, 0)

def test_paged_kv_cache_ensures_first_token_block():
    cache = PagedKVCache(num_layers=2, num_blocks=4, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    block_table = []
    assert cache.ensure_token_block(block_table, 0) == (0, 0)
    assert block_table == [0]
    assert list(cache.free_blocks) == [1, 2, 3]
    assert cache.allocated_blocks == {0}
    assert cache.num_free_blocks == len(cache.free_blocks) == 3
    assert cache.num_allocated_blocks == len(cache.allocated_blocks) == 1

def test_paged_kv_cache_reuses_block_for_tokens_inside_it():
    cache = PagedKVCache(num_layers=2, num_blocks=4, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    block_table = []
    assert cache.ensure_token_block(block_table, 0) == (0, 0)
    assert cache.ensure_token_block(block_table, 1) == (0, 1)
    assert cache.ensure_token_block(block_table, 3) == (0, 3)
    assert block_table == [0]
    assert cache.allocated_blocks == {0}
    assert cache.num_allocated_blocks == 1

def test_paged_kv_cache_allocates_once_at_each_block_boundary():
    cache = PagedKVCache(num_layers=2, num_blocks=4, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    block_table = []
    assert cache.ensure_token_block(block_table, 0) == (0, 0)
    assert cache.ensure_token_block(block_table, 4) == (1, 0)
    assert cache.ensure_token_block(block_table, 4) == (1, 0)
    assert cache.ensure_token_block(block_table, 6) == (1, 2)
    assert block_table == [0, 1]
    assert cache.allocated_blocks == {0, 1}
    assert cache.num_allocated_blocks == 2

def test_paged_kv_cache_ensures_request_blocks_independently():
    cache = PagedKVCache(num_layers=2, num_blocks=4, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    first_request_block_table = []
    second_request_block_table = []
    assert cache.ensure_token_block(first_request_block_table, 0) == (0, 0)
    assert cache.ensure_token_block(second_request_block_table, 0) == (1, 0)
    assert cache.ensure_token_block(first_request_block_table, 4) == (2, 0)
    assert first_request_block_table == [0, 2]
    assert second_request_block_table == [1]
    assert cache.allocated_blocks == {0, 1, 2}

def test_paged_kv_cache_ensure_token_block_does_not_reallocate_storage():
    cache = PagedKVCache(num_layers=2, num_blocks=4, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    block_table = []
    key_data_ptr = cache.key_cache.data_ptr()
    value_data_ptr = cache.value_cache.data_ptr()
    cache.ensure_token_block(block_table, 0)
    cache.ensure_token_block(block_table, 4)
    assert cache.key_cache.data_ptr() == key_data_ptr
    assert cache.value_cache.data_ptr() == value_data_ptr

def test_paged_kv_cache_rejects_negative_token_position_when_ensuring_block():
    cache = PagedKVCache(num_layers=2, num_blocks=4, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    block_table = []
    with pytest.raises(ValueError):
        cache.ensure_token_block(block_table, -1)
    assert block_table == []
    assert cache.num_allocated_blocks == 0

def test_paged_kv_cache_rejects_non_boundary_token_without_block():
    cache = PagedKVCache(num_layers=2, num_blocks=4, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    block_table = []
    with pytest.raises(KVCacheStateError):
        cache.ensure_token_block(block_table, 2)
    assert block_table == []
    assert cache.num_allocated_blocks == 0

def test_paged_kv_cache_rejects_skipped_logical_block():
    cache = PagedKVCache(num_layers=2, num_blocks=4, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    block_table = []
    with pytest.raises(KVCacheStateError):
        cache.ensure_token_block(block_table, 8)
    assert block_table == []
    assert cache.num_allocated_blocks == 0

def test_paged_kv_cache_exhaustion_does_not_mutate_request_block_table():
    cache = PagedKVCache(num_layers=2, num_blocks=1, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    first_request_block_table = []
    second_request_block_table = []
    assert cache.ensure_token_block(first_request_block_table, 0) == (0, 0)
    with pytest.raises(KVCacheExhaustedError):
        cache.ensure_token_block(second_request_block_table, 0)
    assert first_request_block_table == [0]
    assert second_request_block_table == []
    assert list(cache.free_blocks) == []
    assert cache.allocated_blocks == {0}
    assert cache.num_free_blocks == 0
    assert cache.num_allocated_blocks == 1

def test_paged_kv_cache_appends_tokens_inside_one_block():
    cache = PagedKVCache(num_layers=2, num_blocks=4, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    block_table = []
    new_keys = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)
    new_values = new_keys + 100
    cache.append(layer_index=0, block_table=block_table, start_pos=0, new_keys=new_keys, new_values=new_values)
    assert block_table == [0]
    assert torch.equal(cache.key_cache[0, 0, :3], new_keys.permute(1, 0, 2))
    assert torch.equal(cache.value_cache[0, 0, :3], new_values.permute(1, 0, 2))
    assert torch.count_nonzero(cache.key_cache[0, 0, 3:]).item() == 0
    assert torch.count_nonzero(cache.value_cache[0, 0, 3:]).item() == 0
    assert torch.count_nonzero(cache.key_cache[1]).item() == 0
    assert torch.count_nonzero(cache.value_cache[1]).item() == 0

def test_paged_kv_cache_appends_across_block_boundary():
    cache = PagedKVCache(num_layers=2, num_blocks=4, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    block_table = []
    new_keys = torch.arange(36, dtype=torch.float32).reshape(2, 6, 3)
    new_values = new_keys + 100
    cache.append(layer_index=0, block_table=block_table, start_pos=0, new_keys=new_keys, new_values=new_values)
    assert block_table == [0, 1]
    assert torch.equal(cache.key_cache[0, 0], new_keys[:, :4].permute(1, 0, 2))
    assert torch.equal(cache.value_cache[0, 0], new_values[:, :4].permute(1, 0, 2))
    assert torch.equal(cache.key_cache[0, 1, :2], new_keys[:, 4:].permute(1, 0, 2))
    assert torch.equal(cache.value_cache[0, 1, :2], new_values[:, 4:].permute(1, 0, 2))
    assert torch.count_nonzero(cache.key_cache[0, 1, 2:]).item() == 0
    assert torch.count_nonzero(cache.value_cache[0, 1, 2:]).item() == 0

def test_paged_kv_cache_appends_decode_token_to_existing_block_table():
    cache = PagedKVCache(num_layers=2, num_blocks=4, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    block_table = []
    prefill_keys = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    prefill_values = prefill_keys + 100
    cache.append(layer_index=0, block_table=block_table, start_pos=0, new_keys=prefill_keys, new_values=prefill_values)
    decode_keys = torch.full((2, 1, 3), 200, dtype=torch.float32)
    decode_values = torch.full((2, 1, 3), 300, dtype=torch.float32)
    cache.append(layer_index=0, block_table=block_table, start_pos=4, new_keys=decode_keys, new_values=decode_values)
    assert block_table == [0, 1]
    assert torch.equal(cache.key_cache[0, 1, 0], decode_keys[:, 0])
    assert torch.equal(cache.value_cache[0, 1, 0], decode_values[:, 0])

def test_paged_kv_cache_layers_share_block_table_but_store_separate_values():
    cache = PagedKVCache(num_layers=2, num_blocks=4, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    block_table = []
    first_layer_keys = torch.ones(2, 5, 3)
    first_layer_values = torch.full((2, 5, 3), 2.0)
    second_layer_keys = torch.full((2, 5, 3), 3.0)
    second_layer_values = torch.full((2, 5, 3), 4.0)
    cache.append(layer_index=0, block_table=block_table, start_pos=0, new_keys=first_layer_keys, new_values=first_layer_values)
    cache.append(layer_index=1, block_table=block_table, start_pos=0, new_keys=second_layer_keys, new_values=second_layer_values)
    assert block_table == [0, 1]
    assert cache.num_allocated_blocks == 2
    assert torch.equal(cache.key_cache[0, 0], first_layer_keys[:, :4].permute(1, 0, 2))
    assert torch.equal(cache.value_cache[0, 1, 0], first_layer_values[:, 4])
    assert torch.equal(cache.key_cache[1, 0], second_layer_keys[:, :4].permute(1, 0, 2))
    assert torch.equal(cache.value_cache[1, 1, 0], second_layer_values[:, 4])

def test_paged_kv_cache_appends_through_non_contiguous_physical_blocks():
    cache = PagedKVCache(num_layers=1, num_blocks=5, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    unrelated_first_block = cache.allocate_block()
    block_table = []
    first_keys = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    first_values = first_keys + 100
    cache.append(layer_index=0, block_table=block_table, start_pos=0, new_keys=first_keys, new_values=first_values)
    unrelated_middle_block = cache.allocate_block()
    second_keys = torch.full((2, 1, 3), 200, dtype=torch.float32)
    second_values = torch.full((2, 1, 3), 300, dtype=torch.float32)
    cache.append(layer_index=0, block_table=block_table, start_pos=4, new_keys=second_keys, new_values=second_values)
    assert unrelated_first_block == 0
    assert unrelated_middle_block == 2
    assert block_table == [1, 3]
    assert torch.equal(cache.key_cache[0, 1], first_keys.permute(1, 0, 2))
    assert torch.equal(cache.value_cache[0, 3, 0], second_values[:, 0])
    assert torch.count_nonzero(cache.key_cache[0, 0]).item() == 0
    assert torch.count_nonzero(cache.key_cache[0, 2]).item() == 0

def test_paged_kv_cache_append_does_not_reallocate_storage():
    cache = PagedKVCache(num_layers=2, num_blocks=4, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    block_table = []
    key_data_ptr = cache.key_cache.data_ptr()
    value_data_ptr = cache.value_cache.data_ptr()
    cache.append(layer_index=0, block_table=block_table, start_pos=0, new_keys=torch.ones(2, 5, 3), new_values=torch.ones(2, 5, 3))
    assert cache.key_cache.data_ptr() == key_data_ptr
    assert cache.value_cache.data_ptr() == value_data_ptr

@pytest.mark.parametrize("layer_index,start_pos,error_type", [(-1, 0, IndexError), (2, 0, IndexError), (0, -1, ValueError)])
def test_paged_kv_cache_rejects_invalid_append_location(layer_index, start_pos, error_type):
    cache = PagedKVCache(num_layers=2, num_blocks=4, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(error_type):
        cache.append(layer_index=layer_index, block_table=[], start_pos=start_pos, new_keys=torch.ones(2, 1, 3), new_values=torch.ones(2, 1, 3))

def test_paged_kv_cache_rejects_invalid_append_shapes():
    cache = PagedKVCache(num_layers=2, num_blocks=4, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(ValueError):
        cache.append(layer_index=0, block_table=[], start_pos=0, new_keys=torch.ones(2, 3), new_values=torch.ones(2, 3))
    with pytest.raises(ValueError):
        cache.append(layer_index=0, block_table=[], start_pos=0, new_keys=torch.ones(2, 1, 3), new_values=torch.ones(2, 2, 3))
    with pytest.raises(ValueError):
        cache.append(layer_index=0, block_table=[], start_pos=0, new_keys=torch.ones(3, 1, 3), new_values=torch.ones(3, 1, 3))
    with pytest.raises(ValueError):
        cache.append(layer_index=0, block_table=[], start_pos=0, new_keys=torch.ones(2, 1, 4), new_values=torch.ones(2, 1, 4))

def test_paged_kv_cache_rejects_append_dtype_mismatch():
    cache = PagedKVCache(num_layers=2, num_blocks=4, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(ValueError):
        cache.append(layer_index=0, block_table=[], start_pos=0, new_keys=torch.ones(2, 1, 3, dtype=torch.float64), new_values=torch.ones(2, 1, 3, dtype=torch.float64))

def test_paged_kv_cache_rejects_append_with_zero_tokens():
    cache = PagedKVCache(num_layers=2, num_blocks=4, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    block_table = []
    with pytest.raises(ValueError):
        cache.append(layer_index=0, block_table=block_table, start_pos=0, new_keys=torch.empty(2, 0, 3), new_values=torch.empty(2, 0, 3))
    assert block_table == []
    assert cache.num_allocated_blocks == 0

@pytest.mark.parametrize("end_pos", [1, 4, 6, 8])
def test_paged_kv_cache_gets_requested_prefix_across_blocks(end_pos):
    cache = PagedKVCache(num_layers=2, num_blocks=4, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    block_table = []
    keys = torch.arange(48, dtype=torch.float32).reshape(2, 8, 3)
    values = keys + 100
    cache.append(layer_index=0, block_table=block_table, start_pos=0, new_keys=keys, new_values=values)
    cached_keys, cached_values = cache.get(layer_index=0, block_table=block_table, end_pos=end_pos)
    assert cached_keys.shape == (2, end_pos, 3)
    assert cached_values.shape == (2, end_pos, 3)
    assert torch.equal(cached_keys, keys[:, :end_pos])
    assert torch.equal(cached_values, values[:, :end_pos])

def test_paged_kv_cache_gets_non_contiguous_physical_blocks_in_logical_order():
    cache = PagedKVCache(num_layers=1, num_blocks=5, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    cache.allocate_block()
    block_table = []
    first_keys = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    first_values = first_keys + 100
    cache.append(layer_index=0, block_table=block_table, start_pos=0, new_keys=first_keys, new_values=first_values)
    cache.allocate_block()
    second_keys = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3) + 200
    second_values = second_keys + 100
    cache.append(layer_index=0, block_table=block_table, start_pos=4, new_keys=second_keys, new_values=second_values)
    cached_keys, cached_values = cache.get(layer_index=0, block_table=block_table, end_pos=6)
    assert block_table == [1, 3]
    assert torch.equal(cached_keys, torch.cat((first_keys, second_keys), dim=1))
    assert torch.equal(cached_values, torch.cat((first_values, second_values), dim=1))

def test_paged_kv_cache_gets_only_requested_layer_and_request():
    cache = PagedKVCache(num_layers=2, num_blocks=4, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    first_block_table = []
    second_block_table = []
    first_keys = torch.ones(2, 3, 3)
    first_values = torch.full((2, 3, 3), 2.0)
    second_keys = torch.full((2, 3, 3), 3.0)
    second_values = torch.full((2, 3, 3), 4.0)
    cache.append(layer_index=0, block_table=first_block_table, start_pos=0, new_keys=first_keys, new_values=first_values)
    cache.append(layer_index=1, block_table=first_block_table, start_pos=0, new_keys=second_keys, new_values=second_values)
    cache.append(layer_index=1, block_table=second_block_table, start_pos=0, new_keys=first_keys + 10, new_values=first_values + 10)
    cached_keys, cached_values = cache.get(layer_index=1, block_table=first_block_table, end_pos=3)
    assert torch.equal(cached_keys, second_keys)
    assert torch.equal(cached_values, second_values)
    assert first_block_table == [0]
    assert second_block_table == [1]

def test_paged_kv_cache_get_does_not_mutate_allocator_or_storage():
    cache = PagedKVCache(num_layers=1, num_blocks=3, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    block_table = []
    cache.append(layer_index=0, block_table=block_table, start_pos=0, new_keys=torch.ones(2, 5, 3), new_values=torch.ones(2, 5, 3))
    key_data_ptr = cache.key_cache.data_ptr()
    value_data_ptr = cache.value_cache.data_ptr()
    cache.get(layer_index=0, block_table=block_table, end_pos=5)
    assert block_table == [0, 1]
    assert list(cache.free_blocks) == [2]
    assert cache.allocated_blocks == {0, 1}
    assert cache.num_free_blocks == 1
    assert cache.num_allocated_blocks == 2
    assert cache.key_cache.data_ptr() == key_data_ptr
    assert cache.value_cache.data_ptr() == value_data_ptr

@pytest.mark.parametrize("layer_index,end_pos,error_type", [(-1, 1, IndexError), (2, 1, IndexError), (0, 0, ValueError), (0, -1, ValueError), (0, 5, IndexError)])
def test_paged_kv_cache_rejects_invalid_get_range(layer_index, end_pos, error_type):
    cache = PagedKVCache(num_layers=2, num_blocks=3, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    block_table = []
    cache.append(layer_index=0, block_table=block_table, start_pos=0, new_keys=torch.ones(2, 4, 3), new_values=torch.ones(2, 4, 3))
    with pytest.raises(error_type):
        cache.get(layer_index=layer_index, block_table=block_table, end_pos=end_pos)

def test_paged_kv_cache_rejects_unallocated_block_during_get():
    cache = PagedKVCache(num_layers=1, num_blocks=2, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    block_table = [cache.allocate_block()]
    cache.free_block(block_table[0])
    with pytest.raises(KVCacheStateError):
        cache.get(layer_index=0, block_table=block_table, end_pos=1)

@pytest.mark.parametrize("num_layers,num_blocks,block_size,num_kv_heads,head_dim", [(0, 8, 4, 2, 3), (2, 0, 4, 2, 3), (2, 8, 0, 2, 3), (2, 8, 4, 0, 3), (2, 8, 4, 2, 0)])
def test_paged_kv_cache_rejects_invalid_dimensions(num_layers, num_blocks, block_size, num_kv_heads, head_dim):
    with pytest.raises(ValueError):
        PagedKVCache(num_layers=num_layers, num_blocks=num_blocks, block_size=block_size, num_kv_heads=num_kv_heads, head_dim=head_dim, device=torch.device("cpu"), dtype=torch.float32)

def test_contiguous_kv_cache_appends_prefill_and_decode_tokens():
    cache = ContiguousKVCache(num_layers=2, num_kv_heads=2, max_seq_length=5, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    prefill_keys = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)
    prefill_values = prefill_keys + 100
    cache.append(layer_index=0, start_pos=0, new_keys=prefill_keys, new_values=prefill_values)
    assert torch.equal(cache.key_cache[0, :, :3, :], prefill_keys)
    assert torch.equal(cache.value_cache[0, :, :3, :], prefill_values)
    assert torch.count_nonzero(cache.key_cache[0, :, 3:, :]).item() == 0
    assert torch.count_nonzero(cache.value_cache[0, :, 3:, :]).item() == 0
    assert torch.count_nonzero(cache.key_cache[1]).item() == 0
    assert torch.count_nonzero(cache.value_cache[1]).item() == 0
    decode_keys = torch.full((2, 1, 3), 200, dtype=torch.float32)
    decode_values = torch.full((2, 1, 3), 300, dtype=torch.float32)
    cache.append(layer_index=0, start_pos=3, new_keys=decode_keys, new_values=decode_values)
    assert torch.equal(cache.key_cache[0, :, 3:4, :], decode_keys)
    assert torch.equal(cache.value_cache[0, :, 3:4, :], decode_values)
    assert torch.count_nonzero(cache.key_cache[0, :, 4:, :]).item() == 0
    assert torch.count_nonzero(cache.value_cache[0, :, 4:, :]).item() == 0

def test_contiguous_kv_cache_allows_write_ending_at_capacity():
    cache = ContiguousKVCache(num_layers=1, num_kv_heads=2, max_seq_length=5, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    keys = torch.ones(2, 2, 3)
    values = torch.full((2, 2, 3), 2.0)
    cache.append(layer_index=0, start_pos=3, new_keys=keys, new_values=values)
    assert torch.equal(cache.key_cache[0, :, 3:5, :], keys)
    assert torch.equal(cache.value_cache[0, :, 3:5, :], values)

def test_contiguous_kv_cache_gets_prefill_and_decode_prefixes():
    cache = ContiguousKVCache(num_layers=2, num_kv_heads=2, max_seq_length=5, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    prefill_keys = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)
    prefill_values = prefill_keys + 100
    cache.append(layer_index=0, start_pos=0, new_keys=prefill_keys, new_values=prefill_values)
    cached_keys, cached_values = cache.get(layer_index=0, end_pos=3)
    assert cached_keys.shape == (2, 3, 3)
    assert cached_values.shape == (2, 3, 3)
    assert torch.equal(cached_keys, prefill_keys)
    assert torch.equal(cached_values, prefill_values)
    decode_keys = torch.full((2, 1, 3), 200, dtype=torch.float32)
    decode_values = torch.full((2, 1, 3), 300, dtype=torch.float32)
    cache.append(layer_index=0, start_pos=3, new_keys=decode_keys, new_values=decode_values)
    cached_keys, cached_values = cache.get(layer_index=0, end_pos=4)
    assert cached_keys.shape == (2, 4, 3)
    assert cached_values.shape == (2, 4, 3)
    assert torch.equal(cached_keys, torch.cat([prefill_keys, decode_keys], dim=1))
    assert torch.equal(cached_values, torch.cat([prefill_values, decode_values], dim=1))

def test_contiguous_kv_cache_get_is_layer_specific_and_returns_views():
    cache = ContiguousKVCache(num_layers=2, num_kv_heads=2, max_seq_length=5, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    first_layer_keys = torch.ones(2, 2, 3)
    first_layer_values = torch.full((2, 2, 3), 2.0)
    second_layer_keys = torch.full((2, 2, 3), 3.0)
    second_layer_values = torch.full((2, 2, 3), 4.0)
    cache.append(layer_index=0, start_pos=0, new_keys=first_layer_keys, new_values=first_layer_values)
    cache.append(layer_index=1, start_pos=0, new_keys=second_layer_keys, new_values=second_layer_values)
    cached_keys, cached_values = cache.get(layer_index=1, end_pos=2)
    assert torch.equal(cached_keys, second_layer_keys)
    assert torch.equal(cached_values, second_layer_values)
    cached_keys[0, 0, 0] = 9
    cached_values[0, 0, 0] = 10
    assert cache.key_cache[1, 0, 0, 0].item() == 9
    assert cache.value_cache[1, 0, 0, 0].item() == 10
    assert cache.key_cache[0, 0, 0, 0].item() == 1
    assert cache.value_cache[0, 0, 0, 0].item() == 2

def test_contiguous_kv_cache_get_allows_full_capacity():
    cache = ContiguousKVCache(num_layers=1, num_kv_heads=2, max_seq_length=5, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    keys = torch.ones(2, 5, 3)
    values = torch.full((2, 5, 3), 2.0)
    cache.append(layer_index=0, start_pos=0, new_keys=keys, new_values=values)
    cached_keys, cached_values = cache.get(layer_index=0, end_pos=5)
    assert torch.equal(cached_keys, keys)
    assert torch.equal(cached_values, values)

@pytest.mark.parametrize("layer_index,end_pos,error_type", [(-1, 1, IndexError), (2, 1, IndexError), (0, -1, ValueError), (0, 0, ValueError), (0, 6, ValueError)])
def test_contiguous_kv_cache_rejects_invalid_get_range(layer_index, end_pos, error_type):
    cache = ContiguousKVCache(num_layers=2, num_kv_heads=2, max_seq_length=5, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(error_type):
        cache.get(layer_index=layer_index, end_pos=end_pos)

@pytest.mark.parametrize("layer_index,start_pos,error_type", [(-1, 0, IndexError), (2, 0, IndexError), (0, -1, ValueError)])
def test_contiguous_kv_cache_rejects_invalid_append_location(layer_index, start_pos, error_type):
    cache = ContiguousKVCache(num_layers=2, num_kv_heads=2, max_seq_length=5, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(error_type):
        cache.append(layer_index=layer_index, start_pos=start_pos, new_keys=torch.ones(2, 1, 3), new_values=torch.ones(2, 1, 3))

def test_contiguous_kv_cache_rejects_capacity_overflow():
    cache = ContiguousKVCache(num_layers=1, num_kv_heads=2, max_seq_length=5, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(KVCacheExhaustedError):
        cache.append(layer_index=0, start_pos=4, new_keys=torch.ones(2, 2, 3), new_values=torch.ones(2, 2, 3))

def test_contiguous_kv_cache_rejects_invalid_append_shapes():
    cache = ContiguousKVCache(num_layers=1, num_kv_heads=2, max_seq_length=5, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(ValueError):
        cache.append(layer_index=0, start_pos=0, new_keys=torch.ones(1, 2, 1, 3), new_values=torch.ones(2, 1, 3))
    with pytest.raises(ValueError):
        cache.append(layer_index=0, start_pos=0, new_keys=torch.ones(2, 1, 3), new_values=torch.ones(2, 2, 3))
    with pytest.raises(ValueError):
        cache.append(layer_index=0, start_pos=0, new_keys=torch.ones(3, 1, 3), new_values=torch.ones(3, 1, 3))
    with pytest.raises(ValueError):
        cache.append(layer_index=0, start_pos=0, new_keys=torch.ones(2, 1, 4), new_values=torch.ones(2, 1, 4))
    with pytest.raises(ValueError):
        cache.append(layer_index=0, start_pos=0, new_keys=torch.empty(2, 0, 3), new_values=torch.empty(2, 0, 3))

def test_contiguous_kv_cache_rejects_append_dtype_and_device_mismatch():
    cache = ContiguousKVCache(num_layers=1, num_kv_heads=2, max_seq_length=5, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(ValueError):
        cache.append(layer_index=0, start_pos=0, new_keys=torch.ones(2, 1, 3, dtype=torch.float64), new_values=torch.ones(2, 1, 3, dtype=torch.float64))
    with pytest.raises(ValueError):
        cache.append(layer_index=0, start_pos=0, new_keys=torch.ones(2, 1, 3, device="meta"), new_values=torch.ones(2, 1, 3, device="meta"))

def test_kv_cache_manager_allocates_isolated_request_caches():
    manager = KVCacheManager(num_layers=3, num_kv_heads=2, head_dim=4, device=torch.device("cpu"), dtype=torch.float32)
    first_cache = manager.allocate("req-1", max_seq_length=8)
    second_cache = manager.allocate("req-2", max_seq_length=12)
    assert manager.caches == {"req-1": first_cache, "req-2": second_cache}
    assert first_cache.max_seq_length == 8
    assert second_cache.max_seq_length == 12
    assert first_cache.key_cache.data_ptr() != second_cache.key_cache.data_ptr()
    assert first_cache.value_cache.data_ptr() != second_cache.value_cache.data_ptr()
    first_cache.key_cache[0, 0, 0, 0] = 1
    first_cache.value_cache[0, 0, 0, 0] = 2
    assert second_cache.key_cache[0, 0, 0, 0].item() == 0
    assert second_cache.value_cache[0, 0, 0, 0].item() == 0

def test_kv_cache_manager_rejects_duplicate_request_id():
    manager = KVCacheManager(num_layers=3, num_kv_heads=2, head_dim=4, device=torch.device("cpu"), dtype=torch.float32)
    manager.allocate("req-1", max_seq_length=8)
    with pytest.raises(ValueError):
        manager.allocate("req-1", max_seq_length=8)

def test_kv_cache_manager_appends_to_requested_cache():
    manager = KVCacheManager(num_layers=2, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    first_cache = manager.allocate("req-1", max_seq_length=5)
    second_cache = manager.allocate("req-2", max_seq_length=5)
    keys = torch.ones(2, 2, 3)
    values = torch.full((2, 2, 3), 2.0)
    manager.append("req-1", layer_index=1, start_pos=0, new_keys=keys, new_values=values)
    assert torch.equal(first_cache.key_cache[1, :, :2, :], keys)
    assert torch.equal(first_cache.value_cache[1, :, :2, :], values)
    assert torch.count_nonzero(second_cache.key_cache).item() == 0
    assert torch.count_nonzero(second_cache.value_cache).item() == 0

def test_kv_cache_manager_rejects_append_for_unknown_request():
    manager = KVCacheManager(num_layers=2, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(KeyError):
        manager.append("unknown", layer_index=0, start_pos=0, new_keys=torch.ones(2, 1, 3), new_values=torch.ones(2, 1, 3))

def test_kv_cache_manager_gets_requested_cache():
    manager = KVCacheManager(num_layers=2, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    manager.allocate("req-1", max_seq_length=5)
    manager.allocate("req-2", max_seq_length=5)
    first_keys = torch.ones(2, 2, 3)
    first_values = torch.full((2, 2, 3), 2.0)
    second_keys = torch.full((2, 2, 3), 3.0)
    second_values = torch.full((2, 2, 3), 4.0)
    manager.append("req-1", layer_index=1, start_pos=0, new_keys=first_keys, new_values=first_values)
    manager.append("req-2", layer_index=1, start_pos=0, new_keys=second_keys, new_values=second_values)
    cached_keys, cached_values = manager.get("req-2", layer_index=1, end_pos=2)
    assert torch.equal(cached_keys, second_keys)
    assert torch.equal(cached_values, second_values)

def test_kv_cache_manager_rejects_get_for_unknown_request():
    manager = KVCacheManager(num_layers=2, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(KeyError):
        manager.get("unknown", layer_index=0, end_pos=1)

def test_kv_cache_manager_frees_only_requested_cache():
    manager = KVCacheManager(num_layers=2, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    first_cache = manager.allocate("req-1", max_seq_length=5)
    second_cache = manager.allocate("req-2", max_seq_length=5)
    second_cache.key_cache[0, 0, 0, 0] = 7
    second_cache.value_cache[0, 0, 0, 0] = 8
    assert manager.has_cache("req-1")
    assert manager.has_cache("req-2")
    manager.free("req-1")
    assert not manager.has_cache("req-1")
    assert manager.has_cache("req-2")
    assert manager.caches == {"req-2": second_cache}
    assert second_cache.key_cache[0, 0, 0, 0].item() == 7
    assert second_cache.value_cache[0, 0, 0, 0].item() == 8
    assert first_cache is not second_cache

def test_kv_cache_manager_rejects_unknown_and_double_free():
    manager = KVCacheManager(num_layers=2, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(KeyError):
        manager.free("unknown")
    manager.allocate("req-1", max_seq_length=5)
    manager.free("req-1")
    with pytest.raises(KeyError):
        manager.free("req-1")

def test_kv_cache_manager_allows_request_id_reuse_after_free():
    manager = KVCacheManager(num_layers=2, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    first_cache = manager.allocate("req-1", max_seq_length=5)
    manager.free("req-1")
    second_cache = manager.allocate("req-1", max_seq_length=7)
    assert second_cache is not first_cache
    assert second_cache.max_seq_length == 7
    assert manager.caches == {"req-1": second_cache}

def test_kv_cache_manager_releases_cache_ownership():
    manager = KVCacheManager(num_layers=2, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    cache = manager.allocate("req-1", max_seq_length=5)
    cache_reference = weakref.ref(cache)
    manager.free("req-1")
    del cache
    gc.collect()
    assert cache_reference() is None

def test_kv_cache_manager_repeated_allocation_and_free():
    manager = KVCacheManager(num_layers=2, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    cache_references = []
    for index in range(100):
        cache = manager.allocate(f"req-{index}", max_seq_length=5)
        cache_references.append(weakref.ref(cache))
        manager.free(f"req-{index}")
        del cache
    gc.collect()
    assert manager.caches == {}
    assert all(cache_reference() is None for cache_reference in cache_references)

@pytest.mark.parametrize("num_layers,num_kv_heads,max_seq_length,head_dim", [(0, 2, 8, 4), (3, 0, 8, 4), (3, 2, 0, 4), (3, 2, 8, 0)])
def test_contiguous_kv_cache_rejects_invalid_dimensions(num_layers, num_kv_heads, max_seq_length, head_dim):
    with pytest.raises(ValueError):
        ContiguousKVCache(num_layers=num_layers, num_kv_heads=num_kv_heads, max_seq_length=max_seq_length, head_dim=head_dim, device=torch.device("cpu"), dtype=torch.float32)

@pytest.mark.parametrize("num_layers,num_kv_heads,head_dim", [(0, 2, 4), (3, 0, 4), (3, 2, 0)])
def test_kv_cache_manager_rejects_invalid_dimensions(num_layers, num_kv_heads, head_dim):
    with pytest.raises(ValueError):
        KVCacheManager(num_layers=num_layers, num_kv_heads=num_kv_heads, head_dim=head_dim, device=torch.device("cpu"), dtype=torch.float32)

def test_kv_cache_manager_rejects_invalid_max_sequence_length():
    manager = KVCacheManager(num_layers=3, num_kv_heads=2, head_dim=4, device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(ValueError):
        manager.allocate("req-1", max_seq_length=0)
