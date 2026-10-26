import torch
from transformers import AutoModelForCausalLM
from utils.model import DecoderLayer


def test_decoder_layer():
    model_name = "Qwen/Qwen3-1.7B"
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager").to(device).eval()
    config = model.config
    qwen_layer = model.model.layers[0]
    qwen_attention = qwen_layer.self_attn
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dtype = qwen_attention.q_proj.weight.dtype
    my_decoder_layer = DecoderLayer(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        n_query_heads=config.num_attention_heads,
        n_kv_heads=config.num_key_value_heads,
        head_dim=head_dim,
        bias=qwen_attention.q_proj.bias is not None,
        attention_scaling=qwen_attention.scaling,
        rms_norm_eps=config.rms_norm_eps,
    ).to(device=device, dtype=dtype).eval()
    my_decoder_layer.load_state_dict(qwen_layer.state_dict())
    batch_size = 2
    sequence_length = 4
    torch.manual_seed(42)
    hidden_states = torch.randn(batch_size, sequence_length, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(sequence_length, device=device).unsqueeze(0).expand(batch_size, -1)
    with torch.inference_mode():
        position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
    causal_mask = torch.full((sequence_length, sequence_length), float("-inf"), device=device, dtype=dtype)
    causal_mask = torch.triu(causal_mask, diagonal=1).unsqueeze(0).unsqueeze(0)
    with torch.inference_mode():
        expected_output = qwen_layer(
            hidden_states=hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        actual_output = my_decoder_layer(hidden_states, position_embeddings, causal_mask)
    print("Expected shape:", expected_output.shape)
    print("Actual shape:  ", actual_output.shape)
    print("Maximum difference:", (actual_output - expected_output).abs().max().item())
    assert actual_output.shape == expected_output.shape
    torch.testing.assert_close(actual_output, expected_output, rtol=1e-5, atol=1e-5)
