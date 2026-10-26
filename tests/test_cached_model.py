import torch
from utils.batch import PrefillBatch, DecodeBatch
from utils.kv_cache import KVCacheManager
from utils.model import Model
from utils.request import Request

def test_cached_model_decode_matches_full_forward():
    torch.manual_seed(42)
    model = Model(vocab_size=32, hidden_size=8, intermediate_size=16, n_hidden_layers=2, head_dim=2, rope_theta=10000, n_query_heads=4, n_kv_heads=2, bias=False, attention_scaling=2 ** -0.5, rms_norm_eps=1e-6).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    manager = KVCacheManager(num_layers=2, num_kv_heads=2, head_dim=2, device=torch.device("cpu"), dtype=torch.float32)
    manager.allocate("req-1", max_seq_length=4)
    with torch.inference_mode():
        full_logits = model(input_ids)
        prefill_logits = model(input_ids[:, :3], kv_cache_manager=manager, request_ids=["req-1"], start_pos=[0], input_lens=[3])
        decode_logits = model(input_ids[:, 3:], pos_ids=torch.tensor([[3]]), causal_mask=torch.zeros(1, 1, 1, 4), kv_cache_manager=manager, request_ids=["req-1"], start_pos=[3], input_lens=[1])
    torch.testing.assert_close(prefill_logits, full_logits[:, :3], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(decode_logits, full_logits[:, 3:], rtol=1e-5, atol=1e-5)
    for layer_index in range(2):
        cached_keys, cached_values = manager.get("req-1", layer_index=layer_index, end_pos=4)
        assert cached_keys.shape == (2, 4, 2)
        assert cached_values.shape == (2, 4, 2)

def test_multi_request_cached_model_matches_individual_full_forwards():
    torch.manual_seed(42)
    model = Model(vocab_size=32, hidden_size=8, intermediate_size=16, n_hidden_layers=2, head_dim=2, rope_theta=10000, n_query_heads=4, n_kv_heads=2, bias=False, attention_scaling=2 ** -0.5, rms_norm_eps=1e-6).eval()
    first_request = Request("req-1", "", [1, 2], 2, None)
    second_request = Request("req-2", "", [3, 4, 6, 7], 2, None)
    requests = [first_request, second_request]
    for request in requests:
        request.start_prefill()
    prefill_batch = PrefillBatch(requests, pad_token_id=0, device=torch.device("cpu"), dtype=torch.float32)
    manager = KVCacheManager(num_layers=2, num_kv_heads=2, head_dim=2, device=torch.device("cpu"), dtype=torch.float32)
    manager.allocate("req-1", max_seq_length=3)
    manager.allocate("req-2", max_seq_length=5)
    with torch.inference_mode():
        expected_first_prefill = model(torch.tensor([[1, 2]]))[:, -1]
        expected_second_prefill = model(torch.tensor([[3, 4, 6, 7]]))[:, -1]
        prefill_logits = model(prefill_batch.input_ids, prefill_batch.pos_ids, prefill_batch.attention_mask, kv_cache_manager=manager, request_ids=prefill_batch.request_ids, start_pos=prefill_batch.start_pos, input_lens=prefill_batch.input_lens)
    torch.testing.assert_close(prefill_logits[0, prefill_batch.last_token_indices[0]], expected_first_prefill[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(prefill_logits[1, prefill_batch.last_token_indices[1]], expected_second_prefill[0], rtol=1e-5, atol=1e-5)
    first_request.complete_prefill()
    second_request.complete_prefill()
    first_request.append_generated_token(5)
    second_request.append_generated_token(8)
    decode_batch = DecodeBatch(requests, device=torch.device("cpu"), dtype=torch.float32)
    with torch.inference_mode():
        expected_first_decode = model(torch.tensor([[1, 2, 5]]))[:, -1]
        expected_second_decode = model(torch.tensor([[3, 4, 6, 7, 8]]))[:, -1]
        decode_logits = model(decode_batch.input_ids, decode_batch.pos_ids, decode_batch.attention_mask, kv_cache_manager=manager, request_ids=decode_batch.request_ids, start_pos=decode_batch.start_pos, input_lens=decode_batch.input_lens)
    torch.testing.assert_close(decode_logits[0, 0], expected_first_decode[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(decode_logits[1, 0], expected_second_decode[0], rtol=1e-5, atol=1e-5)
    for layer_index in range(2):
        first_keys, first_values = manager.get("req-1", layer_index=layer_index, end_pos=3)
        second_keys, second_values = manager.get("req-2", layer_index=layer_index, end_pos=5)
        assert first_keys.shape == (2, 3, 2)
        assert first_values.shape == (2, 3, 2)
        assert second_keys.shape == (2, 5, 2)
        assert second_values.shape == (2, 5, 2)
