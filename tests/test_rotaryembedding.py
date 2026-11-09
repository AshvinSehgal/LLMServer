import torch
from transformers import AutoModelForCausalLM
from utils.layers import RotaryEmbedding, apply_rotary_embedding
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

def test_rotary_embedding():
    model_name="Qwen/Qwen3-1.7B"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=True)
    head_dim = getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)
    rope_theta = getattr(model.config, "rope_theta", None)
    if rope_theta is None:
        rope_theta = model.config.rope_parameters["rope_theta"]
    dtype = next(model.parameters()).dtype
    my_rope = RotaryEmbedding(head_dim=head_dim, rope_theta=rope_theta).to(device=device, dtype=dtype)
    reference_rope = model.model.rotary_emb
    pos_ids = torch.arange(4, device=device).unsqueeze(0)
    hidden_states = torch.randn(1, 4, model.config.hidden_size, device=device, dtype=dtype)
    with torch.inference_mode():
        expected_cosine, expected_sine = reference_rope(hidden_states, pos_ids)
        actual_cosine, actual_sine = my_rope(hidden_states, pos_ids)
    torch.testing.assert_close(actual_cosine, expected_cosine, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_sine, expected_sine, rtol=1e-5, atol=1e-5)
    print(actual_cosine[0, 0])
    print(actual_sine[0, 0])
    query = torch.randn(1, model.config.num_attention_heads, 4, head_dim, device=device)
    key = torch.randn(1, model.config.num_key_value_heads, 4, head_dim, device=device)
    with torch.inference_mode():
        expected_query, expected_key = apply_rotary_pos_emb(query, key, expected_cosine, expected_sine)
        actual_query, actual_key = apply_rotary_embedding(query, key, actual_cosine, actual_sine)
    torch.testing.assert_close(actual_query, expected_query, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_key, expected_key, rtol=1e-5, atol=1e-5)

def test_rotary_embedding_reuses_inverse_frequencies():
    rope = RotaryEmbedding(head_dim=8, rope_theta=10000)
    hidden_states = torch.zeros(1, 3, 8)
    first_cosine, first_sine = rope(hidden_states, torch.tensor([[0, 1, 2]]))
    cached_frequencies = rope._inverse_frequencies
    second_cosine, second_sine = rope(hidden_states, torch.tensor([[0, 1, 2]]))
    assert rope._inverse_frequencies is cached_frequencies
    torch.testing.assert_close(first_cosine, second_cosine)
    torch.testing.assert_close(first_sine, second_sine)
