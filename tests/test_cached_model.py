import torch
import pytest
from utils.batch import PrefillBatch, DecodeBatch
from utils.kv_cache import PagedKVCache
from utils.model import Model
from utils.request import Request

@pytest.mark.parametrize("decode_attention", ["direct", "gather_sdpa"])
def test_cached_model_decode_matches_full_forward(decode_attention):
    torch.manual_seed(42)
    model = Model(vocab_size=32, hidden_size=8, intermediate_size=16, n_hidden_layers=2, head_dim=2, rope_theta=10000, n_query_heads=4, n_kv_heads=2, bias=False, attention_scaling=2 ** -0.5, rms_norm_eps=1e-6, decode_attention=decode_attention).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    cache = PagedKVCache(num_layers=2, num_blocks=2, block_size=3, num_kv_heads=2, head_dim=2, device=torch.device("cpu"), dtype=torch.float32)
    block_table = []
    with torch.inference_mode():
        full_logits = model(input_ids)
        prefill_logits = model(input_ids[:, :3], kv_cache_manager=cache, request_ids=["req-1"], block_tables=[block_table], start_pos=[0], input_lens=[3])
        decode_logits = model(input_ids[:, 3:], pos_ids=torch.tensor([[3]]), causal_mask=torch.zeros(1, 1, 1, 4), kv_cache_manager=cache, request_ids=["req-1"], block_tables=[block_table], start_pos=[3], input_lens=[1])
    torch.testing.assert_close(prefill_logits, full_logits[:, :3], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(decode_logits, full_logits[:, 3:], rtol=1e-5, atol=1e-5)
    for layer_index in range(2):
        cached_keys, cached_values = cache.get(layer_index=layer_index, block_table=block_table, end_pos=4)
        assert cached_keys.shape == (2, 4, 2)
        assert cached_values.shape == (2, 4, 2)
    assert block_table == [0, 1]

def test_multi_request_cached_model_matches_individual_full_forwards():
    torch.manual_seed(42)
    model = Model(vocab_size=32, hidden_size=8, intermediate_size=16, n_hidden_layers=2, head_dim=2, rope_theta=10000, n_query_heads=4, n_kv_heads=2, bias=False, attention_scaling=2 ** -0.5, rms_norm_eps=1e-6).eval()
    first_request = Request("req-1", "", [1, 2], 2, None)
    second_request = Request("req-2", "", [3, 4, 6, 7], 2, None)
    requests = [first_request, second_request]
    for request in requests:
        request.start_prefill()
    prefill_batch = PrefillBatch(requests, pad_token_id=0, device=torch.device("cpu"), dtype=torch.float32)
    cache = PagedKVCache(num_layers=2, num_blocks=6, block_size=2, num_kv_heads=2, head_dim=2, device=torch.device("cpu"), dtype=torch.float32)
    with torch.inference_mode():
        expected_first_prefill = model(torch.tensor([[1, 2]]))[:, -1]
        expected_second_prefill = model(torch.tensor([[3, 4, 6, 7]]))[:, -1]
        prefill_logits = model(prefill_batch.input_ids, prefill_batch.pos_ids, prefill_batch.attention_mask, kv_cache_manager=cache, request_ids=prefill_batch.request_ids, block_tables=prefill_batch.block_tables, start_pos=prefill_batch.start_pos, input_lens=prefill_batch.input_lens)
    torch.testing.assert_close(prefill_logits[0, prefill_batch.last_token_indices[0]], expected_first_prefill[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(prefill_logits[1, prefill_batch.last_token_indices[1]], expected_second_prefill[0], rtol=1e-5, atol=1e-5)
    first_request.advance_prefill(2)
    second_request.advance_prefill(4)
    first_request.complete_prefill()
    second_request.complete_prefill()
    first_request.append_generated_token(5)
    second_request.append_generated_token(8)
    decode_batch = DecodeBatch(requests, device=torch.device("cpu"), dtype=torch.float32)
    with torch.inference_mode():
        expected_first_decode = model(torch.tensor([[1, 2, 5]]))[:, -1]
        expected_second_decode = model(torch.tensor([[3, 4, 6, 7, 8]]))[:, -1]
        decode_logits = model(decode_batch.input_ids, decode_batch.pos_ids, decode_batch.attention_mask, kv_cache_manager=cache, request_ids=decode_batch.request_ids, block_tables=decode_batch.block_tables, start_pos=decode_batch.start_pos, input_lens=decode_batch.input_lens)
    torch.testing.assert_close(decode_logits[0, 0], expected_first_decode[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(decode_logits[1, 0], expected_second_decode[0], rtol=1e-5, atol=1e-5)
    for layer_index in range(2):
        first_keys, first_values = cache.get(layer_index=layer_index, block_table=first_request.block_table, end_pos=3)
        second_keys, second_values = cache.get(layer_index=layer_index, block_table=second_request.block_table, end_pos=5)
        assert first_keys.shape == (2, 3, 2)
        assert first_values.shape == (2, 3, 2)
        assert second_keys.shape == (2, 5, 2)
        assert second_values.shape == (2, 5, 2)
    assert first_request.block_table == [0, 3]
    assert second_request.block_table == [1, 2, 4]

def test_chunked_prefill_matches_full_forward_for_different_start_positions():
    torch.manual_seed(42)
    model = Model(vocab_size=32, hidden_size=8, intermediate_size=16, n_hidden_layers=2, head_dim=2, rope_theta=10000, n_query_heads=4, n_kv_heads=2, bias=False, attention_scaling=2 ** -0.5, rms_norm_eps=1e-6).eval()
    first_request = Request("req-1", "", [1, 2, 3, 4, 5], 2, None)
    second_request = Request("req-2", "", [6, 7, 8, 9], 2, None)
    requests = [first_request, second_request]
    for request in requests:
        request.start_prefill()
    cache = PagedKVCache(num_layers=2, num_blocks=6, block_size=2, num_kv_heads=2, head_dim=2, device=torch.device("cpu"), dtype=torch.float32)
    first_batch = PrefillBatch(requests, pad_token_id=0, device=torch.device("cpu"), dtype=torch.float32, chunk_lengths=[4, 2])
    with torch.inference_mode():
        expected_first = model(torch.tensor([[1, 2, 3, 4, 5]]))[:, -1]
        expected_second = model(torch.tensor([[6, 7, 8, 9]]))[:, -1]
        model(first_batch.input_ids, first_batch.pos_ids, first_batch.attention_mask, kv_cache_manager=cache, request_ids=first_batch.request_ids, block_tables=first_batch.block_tables, start_pos=first_batch.start_pos, input_lens=first_batch.input_lens)
        first_request.advance_prefill(4)
        second_request.advance_prefill(2)
        second_batch = PrefillBatch(requests, pad_token_id=0, device=torch.device("cpu"), dtype=torch.float32, chunk_lengths=[1, 2])
        actual = model(second_batch.input_ids, second_batch.pos_ids, second_batch.attention_mask, kv_cache_manager=cache, request_ids=second_batch.request_ids, block_tables=second_batch.block_tables, start_pos=second_batch.start_pos, input_lens=second_batch.input_lens)
    torch.testing.assert_close(actual[0, second_batch.last_token_indices[0]], expected_first[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual[1, second_batch.last_token_indices[1]], expected_second[0], rtol=1e-5, atol=1e-5)
    assert first_request.block_table == [0, 1, 3]
    assert second_request.block_table == [2, 4]

def test_shared_prefix_prefill_matches_full_forward_without_overwriting_first_request():
    torch.manual_seed(42)
    model = Model(vocab_size=32, hidden_size=8, intermediate_size=16, n_hidden_layers=2, head_dim=2, rope_theta=10000, n_query_heads=4, n_kv_heads=2, bias=False, attention_scaling=2 ** -0.5, rms_norm_eps=1e-6).eval()
    cache = PagedKVCache(num_layers=2, num_blocks=4, block_size=2, num_kv_heads=2, head_dim=2, device=torch.device("cpu"), dtype=torch.float32)
    first_request = Request("req-1", "", [1, 2, 3, 4, 5, 6], 2, None)
    second_request = Request("req-2", "", [1, 2, 3, 4, 7, 8], 2, None)
    first_request.start_prefill()
    first_batch = PrefillBatch([first_request], pad_token_id=0, device=torch.device("cpu"), dtype=torch.float32)
    with torch.inference_mode():
        model(first_batch.input_ids, first_batch.pos_ids, first_batch.attention_mask, kv_cache_manager=cache, request_ids=first_batch.request_ids, block_tables=first_batch.block_tables, start_pos=first_batch.start_pos, input_lens=first_batch.input_lens)
        expected_second = model(torch.tensor([second_request.prompt_token_ids]))[:, -1]
    original_keys, original_values = cache.get(layer_index=0, block_table=first_request.block_table, end_pos=6)
    original_keys = original_keys.clone()
    original_values = original_values.clone()
    for block_id, digest in zip(first_request.block_table[:2], ["block-0", "block-1"]):
        cache.add_cacheable_block(block_id)
        cache.block_hashes[block_id] = digest
        cache.hash_to_block_id[digest] = block_id
        assert cache.share_block(digest, second_request.block_table) == block_id
        second_request.num_computed_tokens += 2
    second_request.start_prefill()
    second_batch = PrefillBatch([second_request], pad_token_id=0, device=torch.device("cpu"), dtype=torch.float32)
    with torch.inference_mode():
        actual_second = model(second_batch.input_ids, second_batch.pos_ids, second_batch.attention_mask, kv_cache_manager=cache, request_ids=second_batch.request_ids, block_tables=second_batch.block_tables, start_pos=second_batch.start_pos, input_lens=second_batch.input_lens)
    torch.testing.assert_close(actual_second[0, second_batch.last_token_indices[0]], expected_second[0], rtol=1e-5, atol=1e-5)
    current_keys, current_values = cache.get(layer_index=0, block_table=first_request.block_table, end_pos=6)
    torch.testing.assert_close(current_keys, original_keys, rtol=0, atol=0)
    torch.testing.assert_close(current_values, original_values, rtol=0, atol=0)
    assert first_request.block_table == [0, 1, 2]
    assert second_request.block_table == [0, 1, 3]

@pytest.mark.parametrize("decode_attention", ["direct", "gather_sdpa"])
def test_shared_prefix_requests_decode_different_continuations_without_overwriting_each_other(decode_attention):
    torch.manual_seed(42)
    model = Model(vocab_size=32, hidden_size=8, intermediate_size=16, n_hidden_layers=2, head_dim=2, rope_theta=10000, n_query_heads=4, n_kv_heads=2, bias=False, attention_scaling=2 ** -0.5, rms_norm_eps=1e-6, decode_attention=decode_attention).eval()
    cache = PagedKVCache(num_layers=2, num_blocks=3, block_size=2, num_kv_heads=2, head_dim=2, device=torch.device("cpu"), dtype=torch.float32)
    first_request = Request("req-1", "", [1, 2, 3], 2, None)
    second_request = Request("req-2", "", [1, 2, 4], 2, None)
    first_request.start_prefill()
    first_batch = PrefillBatch([first_request], pad_token_id=0, device=torch.device("cpu"), dtype=torch.float32)
    with torch.inference_mode():
        model(first_batch.input_ids, first_batch.pos_ids, first_batch.attention_mask, kv_cache_manager=cache, request_ids=first_batch.request_ids, block_tables=first_batch.block_tables, start_pos=first_batch.start_pos, input_lens=first_batch.input_lens)
    first_request.advance_prefill(3)
    first_request.complete_prefill()
    shared_block = first_request.block_table[0]
    cache.add_cacheable_block(shared_block)
    cache.block_hashes[shared_block] = "shared-prefix"
    cache.hash_to_block_id["shared-prefix"] = shared_block
    assert cache.share_block("shared-prefix", second_request.block_table) == shared_block
    second_request.num_computed_tokens = 2
    second_request.start_prefill()
    second_batch = PrefillBatch([second_request], pad_token_id=0, device=torch.device("cpu"), dtype=torch.float32)
    with torch.inference_mode():
        model(second_batch.input_ids, second_batch.pos_ids, second_batch.attention_mask, kv_cache_manager=cache, request_ids=second_batch.request_ids, block_tables=second_batch.block_tables, start_pos=second_batch.start_pos, input_lens=second_batch.input_lens)
    second_request.advance_prefill(1)
    second_request.complete_prefill()
    original_shared = [(cache.key_cache[layer_index, shared_block].clone(), cache.value_cache[layer_index, shared_block].clone()) for layer_index in range(2)]
    first_request.append_generated_token(5)
    second_request.append_generated_token(6)
    decode_batch = DecodeBatch([first_request, second_request], device=torch.device("cpu"), dtype=torch.float32)
    with torch.inference_mode():
        actual = model(decode_batch.input_ids, decode_batch.pos_ids, decode_batch.attention_mask, kv_cache_manager=cache, request_ids=decode_batch.request_ids, block_tables=decode_batch.block_tables, start_pos=decode_batch.start_pos, input_lens=decode_batch.input_lens)
        expected_first = model(torch.tensor([[1, 2, 3, 5]]))[:, -1]
        expected_second = model(torch.tensor([[1, 2, 4, 6]]))[:, -1]
    torch.testing.assert_close(actual[0, 0], expected_first[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual[1, 0], expected_second[0], rtol=1e-5, atol=1e-5)
    assert first_request.block_table[0] == second_request.block_table[0] == shared_block
    assert first_request.block_table[1] != second_request.block_table[1]
    for layer_index, (keys, values) in enumerate(original_shared):
        torch.testing.assert_close(cache.key_cache[layer_index, shared_block], keys, rtol=0, atol=0)
        torch.testing.assert_close(cache.value_cache[layer_index, shared_block], values, rtol=0, atol=0)

def test_gather_sdpa_decode_matches_full_forward_across_block_boundaries():
    torch.manual_seed(42)
    model = Model(vocab_size=32, hidden_size=8, intermediate_size=16, n_hidden_layers=2, head_dim=2, rope_theta=10000, n_query_heads=4, n_kv_heads=2, bias=False, attention_scaling=2 ** -0.5, rms_norm_eps=1e-6, decode_attention="gather_sdpa").eval()
    for sequence_length in (2, 4, 7, 10):
        input_ids = torch.arange(1, sequence_length + 1).unsqueeze(0)
        cache = PagedKVCache(num_layers=2, num_blocks=6, block_size=2, num_kv_heads=2, head_dim=2, device=torch.device("cpu"), dtype=torch.float32)
        block_table = []
        with torch.inference_mode():
            expected = model(input_ids)[:, -1]
            model(input_ids[:, :-1], kv_cache_manager=cache, request_ids=["req-1"], block_tables=[block_table], start_pos=[0], input_lens=[sequence_length - 1])
            actual = model(input_ids[:, -1:], pos_ids=torch.tensor([[sequence_length - 1]]), causal_mask=torch.zeros(1, 1, 1, sequence_length), kv_cache_manager=cache, request_ids=["req-1"], block_tables=[block_table], start_pos=[sequence_length - 1], input_lens=[1])[:, -1]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
