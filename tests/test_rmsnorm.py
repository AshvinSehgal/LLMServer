import torch
from transformers import AutoModelForCausalLM
from utils.layers import RMSNorm

def test_rmsnorm():
    model_name="Qwen/Qwen3-1.7B"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=True)
    reference_norm = model.model.layers[0].input_layernorm
    my_norm = RMSNorm(normalization_size=model.config.hidden_size, eps=model.config.rms_norm_eps
                        ).to(device=reference_norm.weight.device, dtype=reference_norm.weight.dtype)
    with torch.no_grad():
        my_norm.weight.copy_(reference_norm.weight)
    test_input = torch.randn(2, 5, model.config.hidden_size,
        device=reference_norm.weight.device, dtype=reference_norm.weight.dtype)
    with torch.inference_mode():
        expected = reference_norm(test_input)
        actual = my_norm(test_input)
    print("Expected shape:", expected.shape)
    print("Actual shape:", actual.shape)
    print("Largest difference:", (expected - actual).abs().max().item())
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    print("RMSNorm matches Qwen3")