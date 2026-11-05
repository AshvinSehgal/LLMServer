import torch
from unittest.mock import patch
from transformers import AutoModelForCausalLM
from utils.attention import SelfAttention
from utils.kv_cache import PagedKVCache
from utils.layers import RotaryEmbedding


def test_self_attention():
    model_name = "Qwen/Qwen3-1.7B"
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager").to(device).eval()
    config = model.config
    qwen_attention = model.model.layers[0].self_attn
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dtype = qwen_attention.q_proj.weight.dtype
    my_attention = SelfAttention(
        hidden_size=config.hidden_size,
        n_query_heads=config.num_attention_heads,
        n_kv_heads=config.num_key_value_heads,
        head_dim=head_dim,
        bias=qwen_attention.q_proj.bias is not None,
        attention_scaling=qwen_attention.scaling,
        rms_norm_eps=config.rms_norm_eps,
    ).to(device=device, dtype=dtype).eval()
    my_attention.load_state_dict(qwen_attention.state_dict())
    batch_size = 2
    sequence_length = 4
    torch.manual_seed(42)
    hidden_states = torch.randn(batch_size, sequence_length, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(sequence_length, device=device).unsqueeze(0).expand(batch_size, -1)
    with torch.inference_mode():
        position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
    causal_mask = torch.full((sequence_length, sequence_length), float("-inf"), device=device, dtype=dtype)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    with torch.inference_mode():
        expected_output, _ = qwen_attention(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
            past_key_values=None,
        )
        actual_output = my_attention(hidden_states, position_embeddings, causal_mask)
    print("Expected shape:", expected_output.shape)
    print("Actual shape:  ", actual_output.shape)
    print("Maximum difference:", (actual_output - expected_output).abs().max().item())
    assert actual_output.shape == expected_output.shape
    torch.testing.assert_close(actual_output, expected_output, rtol=1e-5, atol=1e-5)

def test_cached_self_attention_matches_full_attention():
    torch.manual_seed(42)
    attention = SelfAttention(hidden_size=8, n_query_heads=4, n_kv_heads=2, head_dim=2, bias=False, attention_scaling=2 ** -0.5, rms_norm_eps=1e-6).eval()
    rotary_embedding = RotaryEmbedding(head_dim=2, rope_theta=10000)
    full_hidden_states = torch.randn(1, 4, 8)
    full_pos_ids = torch.arange(4).unsqueeze(0)
    full_pos_embeddings = rotary_embedding(full_hidden_states, full_pos_ids)
    full_mask = torch.full((1, 1, 4, 4), torch.finfo(torch.float32).min).triu(diagonal=1)
    cache = PagedKVCache(num_layers=1, num_blocks=2, block_size=3, num_kv_heads=2, head_dim=2, device=torch.device("cpu"), dtype=torch.float32)
    block_table = []
    with torch.inference_mode():
        full_output = attention(full_hidden_states, full_pos_embeddings, full_mask)
        prefill_hidden_states = full_hidden_states[:, :3]
        prefill_pos_embeddings = rotary_embedding(prefill_hidden_states, full_pos_ids[:, :3])
        prefill_output = attention(prefill_hidden_states, prefill_pos_embeddings, full_mask[:, :, :3, :3], kv_cache=cache, layer_index=0, request_ids=["req-1"], block_tables=[block_table], start_pos=[0], input_lens=[3])
        decode_hidden_states = full_hidden_states[:, 3:]
        decode_pos_embeddings = rotary_embedding(decode_hidden_states, full_pos_ids[:, 3:])
        with patch.object(cache, "get", side_effect=AssertionError("Decode must not gather the KV cache")):
            decode_output = attention(decode_hidden_states, decode_pos_embeddings, torch.zeros(1, 1, 1, 4), kv_cache=cache, layer_index=0, request_ids=["req-1"], block_tables=[block_table], start_pos=[3], input_lens=[1])
    torch.testing.assert_close(prefill_output, full_output[:, :3], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(decode_output, full_output[:, 3:], rtol=1e-5, atol=1e-5)
    cached_keys, cached_values = cache.get(layer_index=0, block_table=block_table, end_pos=4)
    assert block_table == [0, 1]
    assert cached_keys.shape == (2, 4, 2)
    assert cached_values.shape == (2, 4, 2)

def test_direct_paged_read_matches_gather_with_noncontiguous_blocks():
    torch.manual_seed(42)
    attention = SelfAttention(hidden_size=8, n_query_heads=4, n_kv_heads=2, head_dim=2, bias=False, attention_scaling=2 ** -0.5, rms_norm_eps=1e-6).eval()
    cache = PagedKVCache(num_layers=1, num_blocks=4, block_size=3, num_kv_heads=2, head_dim=2, device=torch.device("cpu"), dtype=torch.float32)
    block_table = [cache.allocate_block()]
    cache.allocate_block()
    block_table.append(cache.allocate_block())
    keys = torch.randn(2, 5, 2)
    values = torch.randn(2, 5, 2)
    query = torch.randn(1, 4, 1, 2)
    cache.append(0, block_table, 0, keys, values)
    gathered_keys, gathered_values = cache.get(0, block_table, 5)
    gathered_keys = gathered_keys.repeat_interleave(2, dim=0).unsqueeze(0)
    gathered_values = gathered_values.repeat_interleave(2, dim=0).unsqueeze(0)
    weights = torch.softmax(torch.matmul(query, gathered_keys.mT) * attention.attention_scaling, dim=-1, dtype=torch.float32)
    expected = torch.matmul(weights, gathered_values)
    with patch.object(cache, "get", side_effect=AssertionError("Direct paged read must not gather the KV cache")):
        actual = attention.direct_paged_read(query, cache, 0, block_table, 5, 2)
    assert block_table == [0, 2]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

def test_sdpa_matches_eager_attention_for_full_and_cached_prefill():
    torch.manual_seed(42)
    eager = SelfAttention(hidden_size=8, n_query_heads=4, n_kv_heads=2, head_dim=2, bias=False, attention_scaling=2 ** -0.5, rms_norm_eps=1e-6).eval()
    sdpa = SelfAttention(hidden_size=8, n_query_heads=4, n_kv_heads=2, head_dim=2, bias=False, attention_scaling=2 ** -0.5, rms_norm_eps=1e-6, use_sdpa=True).eval()
    sdpa.load_state_dict(eager.state_dict())
    hidden_states = torch.randn(1, 4, 8)
    positions = torch.arange(4).unsqueeze(0)
    embeddings = RotaryEmbedding(head_dim=2, rope_theta=10000)(hidden_states, positions)
    mask = torch.full((1, 1, 4, 4), torch.finfo(torch.float32).min).triu(diagonal=1)
    with torch.inference_mode():
        expected_full = eager(hidden_states, embeddings, mask)
        actual_full = sdpa(hidden_states, embeddings, mask)
    torch.testing.assert_close(actual_full, expected_full, rtol=1e-5, atol=1e-5)
    eager_cache = PagedKVCache(num_layers=1, num_blocks=2, block_size=2, num_kv_heads=2, head_dim=2, device=torch.device("cpu"), dtype=torch.float32)
    sdpa_cache = PagedKVCache(num_layers=1, num_blocks=2, block_size=2, num_kv_heads=2, head_dim=2, device=torch.device("cpu"), dtype=torch.float32)
    with torch.inference_mode():
        expected_prefill = eager(hidden_states, embeddings, mask, kv_cache=eager_cache, layer_index=0, request_ids=["req-1"], block_tables=[[]], start_pos=[0], input_lens=[4])
        actual_prefill = sdpa(hidden_states, embeddings, mask, kv_cache=sdpa_cache, layer_index=0, request_ids=["req-1"], block_tables=[[]], start_pos=[0], input_lens=[4])
    torch.testing.assert_close(actual_prefill, expected_prefill, rtol=1e-5, atol=1e-5)
