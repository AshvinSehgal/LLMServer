import torch
from transformers import AutoModelForCausalLM
from utils.attention import SelfAttention
from utils.kv_cache import KVCacheManager
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
    manager = KVCacheManager(num_layers=1, num_kv_heads=2, head_dim=2, device=torch.device("cpu"), dtype=torch.float32)
    manager.allocate("req-1", max_seq_length=4)
    with torch.inference_mode():
        full_output = attention(full_hidden_states, full_pos_embeddings, full_mask)
        prefill_hidden_states = full_hidden_states[:, :3]
        prefill_pos_embeddings = rotary_embedding(prefill_hidden_states, full_pos_ids[:, :3])
        prefill_output = attention(prefill_hidden_states, prefill_pos_embeddings, full_mask[:, :, :3, :3], kv_cache_manager=manager, layer_index=0, request_ids=["req-1"], start_pos=[0], input_lens=[3])
        decode_hidden_states = full_hidden_states[:, 3:]
        decode_pos_embeddings = rotary_embedding(decode_hidden_states, full_pos_ids[:, 3:])
        decode_output = attention(decode_hidden_states, decode_pos_embeddings, torch.zeros(1, 1, 1, 4), kv_cache_manager=manager, layer_index=0, request_ids=["req-1"], start_pos=[3], input_lens=[1])
    torch.testing.assert_close(prefill_output, full_output[:, :3], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(decode_output, full_output[:, 3:], rtol=1e-5, atol=1e-5)
    cached_keys, cached_values = manager.get("req-1", layer_index=0, end_pos=4)
    assert cached_keys.shape == (2, 4, 2)
    assert cached_values.shape == (2, 4, 2)
