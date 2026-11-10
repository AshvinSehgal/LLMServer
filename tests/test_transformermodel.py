import torch
from unittest.mock import patch
from transformers import AutoModelForCausalLM, Qwen3Config
from utils.model_loader import model_loader, resolve_decode_attention

def test_model_loader_selects_mps_decode_attention_by_default():
    assert resolve_decode_attention(torch.device("mps"), None) == "gather_sdpa"
    assert resolve_decode_attention(torch.device("cpu"), None) == "direct"
    assert resolve_decode_attention(torch.device("mps"), "direct") == "direct"
    assert resolve_decode_attention(torch.device("cpu"), "gather_sdpa") == "gather_sdpa"

def test_model_loader_assigns_weights_without_a_second_cpu_copy():
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=4, rope_theta=10000, rms_norm_eps=1e-6)
    reference = AutoModelForCausalLM.from_config(config).eval()
    with patch("utils.model_loader.AutoModelForCausalLM.from_pretrained", return_value=reference), patch("utils.model_loader.CheckpointReader.open", return_value=None):  # transformers fallback path
        custom = model_loader("tiny-qwen", device=torch.device("cpu"))
    assert custom.embed_tokens.weight.data_ptr() == reference.model.embed_tokens.weight.data_ptr()
    assert custom.rotary_embedding.dim_indices.device.type == "cpu"
    assert all(parameter.device.type != "meta" for parameter in custom.parameters())
    input_ids = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        expected = reference(input_ids=input_ids, use_cache=False).logits
        actual = custom(input_ids)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

def test_model_loader_fuses_projections_after_loading_weights():
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=4, rope_theta=10000, rms_norm_eps=1e-6)
    reference = AutoModelForCausalLM.from_config(config).eval()
    with patch("utils.model_loader.AutoModelForCausalLM.from_pretrained", return_value=reference), patch("utils.model_loader.CheckpointReader.open", return_value=None):  # transformers fallback path
        custom = model_loader("tiny-qwen", device=torch.device("cpu"), fuse_projections=True)
    assert all(layer.self_attn.qkv_proj is not None and layer.mlp.gate_up_proj is not None for layer in custom.layers)
    assert all(parameter.device.type != "meta" for parameter in custom.parameters())
    input_ids = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        expected = reference(input_ids=input_ids, use_cache=False).logits
        actual = custom(input_ids)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

def get_hook(outputs, name):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            outputs[name] = tuple(value.detach().clone() for value in output)
        else:
            outputs[name] = output.detach().clone()
    return hook

def get_tensor(output):
    if isinstance(output, tuple):
        return output[0]
    return output

def report_difference(name, actual, expected):
    actual = get_tensor(actual)
    expected = get_tensor(expected)
    assert actual.shape == expected.shape
    difference = (actual - expected).abs().float()
    print(name, "maximum difference:", difference.max().item(), "mean difference:", difference.mean().item())

def get_tolerance(dtype):
    if dtype == torch.float32:
        return 1e-4, 1e-4
    elif dtype == torch.float16:
        return 1e-2, 1e-2
    elif dtype == torch.bfloat16:
        return 5e-2, 5e-1
    return 1e-5, 1e-5

def test_transformer_model():
    model_name = "Qwen/Qwen3-1.7B"
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    my_model = model_loader(model_name=model_name, device=device)
    dtype = next(my_model.parameters()).dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager").to(device=device, dtype=dtype).eval()
    config = model.config
    if config.tie_word_embeddings:
        assert my_model.embed_tokens.weight.data_ptr() == my_model.lm_head.weight.data_ptr()
    input_ids = torch.tensor([[785, 6722, 315, 9625, 374]], device=device)
    expected_intermediates = {}
    actual_intermediates = {}
    hook_handles = [
        model.model.embed_tokens.register_forward_hook(get_hook(expected_intermediates, "embedding")),
        my_model.embed_tokens.register_forward_hook(get_hook(actual_intermediates, "embedding")),
        model.model.rotary_emb.register_forward_hook(get_hook(expected_intermediates, "rope")),
        my_model.rotary_embedding.register_forward_hook(get_hook(actual_intermediates, "rope")),
        model.model.norm.register_forward_hook(get_hook(expected_intermediates, "final_norm")),
        my_model.norm.register_forward_hook(get_hook(actual_intermediates, "final_norm")),
        model.lm_head.register_forward_hook(get_hook(expected_intermediates, "logits")),
        my_model.lm_head.register_forward_hook(get_hook(actual_intermediates, "logits")),
    ]
    for index, (expected_layer, actual_layer) in enumerate(zip(model.model.layers, my_model.layers)):
        hook_handles.append(expected_layer.self_attn.register_forward_hook(get_hook(expected_intermediates, f"attention_{index}")))
        hook_handles.append(actual_layer.self_attn.register_forward_hook(get_hook(actual_intermediates, f"attention_{index}")))
        hook_handles.append(expected_layer.mlp.register_forward_hook(get_hook(expected_intermediates, f"mlp_{index}")))
        hook_handles.append(actual_layer.mlp.register_forward_hook(get_hook(actual_intermediates, f"mlp_{index}")))
        hook_handles.append(expected_layer.register_forward_hook(get_hook(expected_intermediates, f"layer_{index}")))
        hook_handles.append(actual_layer.register_forward_hook(get_hook(actual_intermediates, f"layer_{index}")))
    try:
        with torch.inference_mode():
            expected_output = model(input_ids=input_ids, use_cache=False).logits
            actual_output = my_model(input_ids)
    finally:
        for handle in hook_handles:
            handle.remove()
    report_difference("embedding", actual_intermediates["embedding"], expected_intermediates["embedding"])
    report_difference("rope_cosine", actual_intermediates["rope"][0], expected_intermediates["rope"][0])
    report_difference("rope_sine", actual_intermediates["rope"][1], expected_intermediates["rope"][1])
    for index in range(config.num_hidden_layers):
        report_difference(f"attention_{index}", actual_intermediates[f"attention_{index}"], expected_intermediates[f"attention_{index}"])
        report_difference(f"mlp_{index}", actual_intermediates[f"mlp_{index}"], expected_intermediates[f"mlp_{index}"])
        report_difference(f"layer_{index}", actual_intermediates[f"layer_{index}"], expected_intermediates[f"layer_{index}"])
    report_difference("final_norm", actual_intermediates["final_norm"], expected_intermediates["final_norm"])
    report_difference("logits", actual_intermediates["logits"], expected_intermediates["logits"])
    print("Expected shape:", expected_output.shape)
    print("Actual shape:  ", actual_output.shape)
    print("Maximum difference:", (actual_output - expected_output).abs().max().item())
    assert actual_output.shape == expected_output.shape
    assert actual_output.shape == (input_ids.shape[0], input_ids.shape[1], config.vocab_size)
    assert torch.equal(actual_output[:, -1, :].argmax(dim=-1), expected_output[:, -1, :].argmax(dim=-1))
    rtol, atol = get_tolerance(dtype)
    torch.testing.assert_close(actual_output, expected_output, rtol=rtol, atol=atol)


def test_streaming_loader_materializes_every_parameter_and_matches_reference(tmp_path):
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=4, rope_theta=10000, rms_norm_eps=1e-6)
    reference = AutoModelForCausalLM.from_config(config).eval()
    reference.save_pretrained(tmp_path)
    custom = model_loader(str(tmp_path), device=torch.device("cpu"), dtype=torch.float32)
    assert all(parameter.device.type != "meta" for parameter in custom.parameters())
    assert all(layer.self_attn.qkv_proj is not None for layer in custom.layers)
    input_ids = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        torch.testing.assert_close(custom(input_ids), reference(input_ids=input_ids, use_cache=False).logits, rtol=1e-4, atol=1e-4)
