import torch
from transformers import AutoModelForCausalLM
from utils.layers import MLP


def test_mlp():
    model_name = "Qwen/Qwen3-1.7B"
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    qwen_mlp = model.model.layers[0].mlp
    batch_size = 2
    sequence_length = 4
    dtype = qwen_mlp.gate_proj.weight.dtype
    my_mlp = MLP(model.config.hidden_size, model.config.intermediate_size, qwen_mlp.gate_proj.bias is not None).to(device=device, dtype=dtype).eval()
    my_mlp.load_state_dict(qwen_mlp.state_dict())
    torch.manual_seed(42)
    hidden_states = torch.randn(batch_size, sequence_length, model.config.hidden_size, device=device, dtype=dtype)
    with torch.inference_mode():
        expected_output = qwen_mlp(hidden_states)
        actual_output = my_mlp(hidden_states)
    print("Input shape:   ", hidden_states.shape)
    print("Expected shape:", expected_output.shape)
    print("Actual shape:  ", actual_output.shape)
    print("Maximum difference:", (actual_output - expected_output).abs().max().item())
    assert actual_output.shape == hidden_states.shape
    assert actual_output.shape == expected_output.shape
    torch.testing.assert_close(actual_output, expected_output, rtol=1e-5, atol=1e-5)
