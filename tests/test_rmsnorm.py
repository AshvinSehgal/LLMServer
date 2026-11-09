import torch
import pytest
from transformers import AutoModelForCausalLM
from utils.layers import RMSNorm

@pytest.mark.parametrize("device", ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_native_rmsnorm_matches_manual_path(device, dtype):
    torch.manual_seed(42)
    norm = RMSNorm(128, 1e-6).to(device=device, dtype=dtype).eval()
    hidden_states = torch.randn(2, 5, 128, device=device, dtype=dtype)
    with torch.inference_mode():
        native = norm(hidden_states)
        norm.use_native = False
        manual = norm(hidden_states)
    torch.testing.assert_close(native, manual, rtol=1e-3 if dtype == torch.float16 else 1e-5, atol=2e-3 if dtype == torch.float16 else 1e-5)

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
