import torch
from utils.kv_cache import KVCacheManager
from utils.model import Model

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
