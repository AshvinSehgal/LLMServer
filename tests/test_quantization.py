import pytest
import torch
import torch.nn as nn
from unittest.mock import patch
from transformers import AutoModelForCausalLM, Qwen3Config
from utils.layers import Int8Linear
from utils.model_loader import model_loader


def tiny_reference():
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8, rope_theta=10000, rms_norm_eps=1e-6)
    return AutoModelForCausalLM.from_config(config).eval()


@pytest.mark.parametrize("bias", [False, True])
def test_int8_linear_matches_float_linear_within_quantization_error(bias):
    torch.manual_seed(0)
    linear = nn.Linear(48, 20, bias=bias)
    quantized = Int8Linear.from_linear(linear)
    assert quantized.weight.dtype == torch.int8
    hidden_states = torch.randn(2, 3, 48)
    with torch.inference_mode():
        expected = linear(hidden_states)
        actual = quantized(hidden_states)
    assert actual.shape == expected.shape
    # Per-channel int8 error is at most half a quantization step per weight.
    bound = (quantized.scale.float().sum() * 0.5 * hidden_states.abs().amax()).item()
    assert (actual - expected).abs().max().item() < bound
    torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.05)


def test_int8_linear_handles_all_zero_rows_and_chunked_quantization():
    linear = nn.Linear(8, 5, bias=False)
    with torch.no_grad():
        linear.weight[2].zero_()
    with patch.object(Int8Linear, "QUANTIZE_CHUNK_ROWS", 2):
        chunked = Int8Linear.from_linear(linear)
    whole = Int8Linear.from_linear(linear)
    assert torch.equal(chunked.weight, whole.weight)
    assert torch.equal(chunked.scale, whole.scale)
    assert torch.isfinite(chunked(torch.randn(1, 8))).all()
    assert chunked(torch.ones(1, 8))[0, 2] == 0


def test_model_loader_quantizes_every_linear_layer_to_int8():
    reference = tiny_reference()
    with patch("utils.model_loader.AutoModelForCausalLM.from_pretrained", return_value=reference):
        custom = model_loader("tiny-qwen", device=torch.device("cpu"), fuse_projections=False, quantize="int8")
    assert not any(isinstance(module, nn.Linear) for module in custom.modules())
    assert sum(isinstance(module, Int8Linear) for module in custom.modules()) == 2 * 7 + 1  # q, k, v, o, gate, up, down per layer, plus lm_head
    assert next(custom.parameters()).dtype == torch.float32
    input_ids = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        expected = reference(input_ids=input_ids, use_cache=False).logits
        actual = custom(input_ids)
    assert actual.shape == expected.shape
    assert (actual - expected).abs().max().item() < 0.1 * expected.abs().max().item()


def test_model_loader_quantizes_fused_projections():
    reference = tiny_reference()
    with patch("utils.model_loader.AutoModelForCausalLM.from_pretrained", return_value=reference):
        custom = model_loader("tiny-qwen", device=torch.device("cpu"), fuse_projections=True, quantize="int8")
    assert all(isinstance(layer.self_attn.qkv_proj, Int8Linear) and isinstance(layer.mlp.gate_up_proj, Int8Linear) for layer in custom.layers)
    assert isinstance(custom.lm_head, Int8Linear)
    with torch.inference_mode():
        assert custom(torch.tensor([[1, 2, 3]])).shape == (1, 3, 64)


def test_model_loader_rejects_unknown_quantization_mode():
    with pytest.raises(ValueError, match="quantization"):
        model_loader("tiny-qwen", device=torch.device("cpu"), quantize="int3")


def test_benchmark_reference_quantization_follows_custom_by_default():
    from types import SimpleNamespace
    from benchmark import reference_quantize_for
    assert reference_quantize_for(SimpleNamespace(quantize="int8", reference_quantize="match")) == "int8"
    assert reference_quantize_for(SimpleNamespace(quantize=None, reference_quantize="match")) is None
    assert reference_quantize_for(SimpleNamespace(quantize="int8", reference_quantize="none")) is None
    assert reference_quantize_for(SimpleNamespace(quantize=None, reference_quantize="int8")) == "int8"


def test_quantize_linear_layers_converts_a_huggingface_model_in_place():
    from utils.layers import quantize_linear_layers
    reference = tiny_reference()
    quantize_linear_layers(reference)
    assert not any(isinstance(module, nn.Linear) for module in reference.modules())
    with torch.inference_mode():
        assert reference(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False).logits.shape == (1, 3, 64)


def test_int8_linear_accepts_non_contiguous_input():
    linear = nn.Linear(16, 8, bias=False)
    quantized = Int8Linear.from_linear(linear)
    hidden_states = torch.randn(3, 5, 16)[:, -2:, :]
    assert not hidden_states.is_contiguous()
    with torch.inference_mode():
        torch.testing.assert_close(quantized(hidden_states), linear(hidden_states), rtol=0.05, atol=0.05)


from utils.layers import Int4Linear, quantize_linear_layers

mps_only = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS int4 kernel required")


@mps_only
def test_int4_linear_matches_group_dequantized_weights_and_falls_back_for_larger_inputs():
    torch.manual_seed(0)
    linear = nn.Linear(128, 64, bias=False).to("mps", torch.float16)
    quantized = Int4Linear.from_linear(linear, group_size=32)
    assert isinstance(quantized, Int4Linear)
    with torch.inference_mode():
        for rows in (1, 2, 3, 40):
            hidden_states = torch.randn(rows, 128, device="mps", dtype=torch.float16)
            expected = linear(hidden_states).float()
            actual = quantized(hidden_states).float()
            tolerance = 0.5 if rows <= Int4Linear.MAX_INT4_ROWS else 0.1
            assert (actual - expected).abs().max().item() < tolerance * expected.abs().max().item()
        # The int4 kernel must agree with an explicit dequantization of the stored nibbles.
        identity = torch.eye(128, device="mps", dtype=torch.float16)
        previous = Int4Linear.MAX_INT4_ROWS
        Int4Linear.MAX_INT4_ROWS = 10**9
        try:
            kernel_weight = quantized(identity).float().t()
        finally:
            Int4Linear.MAX_INT4_ROWS = previous
    grouped = linear.weight.float().reshape(64, 4, 32)
    step = (grouped.amax(-1) - grouped.amin(-1)) / 15
    assert ((kernel_weight.reshape(64, 4, 32) - grouped).abs() <= step[..., None] * 0.6).all()


@mps_only
def test_int4_mode_keeps_lm_head_int8_and_handles_unsupported_shapes():
    reference = tiny_reference().to("mps", torch.float16)
    reference.model.layers[0].mlp.down_proj = nn.Linear(64, 30, bias=False).to("mps", torch.float16)  # 30 % 8 != 0
    quantize_linear_layers(reference, "int4")
    assert type(reference.lm_head) is Int8Linear
    assert type(reference.model.layers[0].mlp.down_proj) is Int8Linear
    assert any(isinstance(module, Int4Linear) for module in reference.modules())


def test_int4_falls_back_to_int8_off_mps():
    linear = nn.Linear(64, 16, bias=False)
    assert type(Int4Linear.from_linear(linear)) is Int8Linear
    with pytest.raises(ValueError, match="quantization"):
        quantize_linear_layers(linear, "int2")


def test_model_loader_fuses_projections_by_default_and_uses_sdpa_prefill():
    reference = tiny_reference()
    with patch("utils.model_loader.AutoModelForCausalLM.from_pretrained", return_value=reference):
        custom = model_loader("tiny-qwen", device=torch.device("cpu"))
    assert all(layer.self_attn.qkv_proj is not None and layer.mlp.gate_up_proj is not None for layer in custom.layers)
    assert all(layer.self_attn.use_sdpa for layer in custom.layers)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        torch.testing.assert_close(custom(input_ids), reference(input_ids=input_ids, use_cache=False).logits, rtol=1e-4, atol=1e-4)


@mps_only
@pytest.mark.parametrize("rows", [1, 7, 8, 9, 10, 11, 16])
def test_int8_linear_row_padding_does_not_change_results(rows):
    torch.manual_seed(0)
    linear = nn.Linear(64, 32, bias=True).to("mps", torch.float16)
    quantized = Int8Linear.from_linear(linear)
    hidden_states = torch.randn(rows, 64, device="mps", dtype=torch.float16)
    with torch.inference_mode():
        padded = quantized(hidden_states)
        previous = Int8Linear.PAD_ROWS_FROM
        Int8Linear.PAD_ROWS_FROM = 10**9
        try:
            plain = quantized(hidden_states)
        finally:
            Int8Linear.PAD_ROWS_FROM = previous
    assert padded.shape == (rows, 32)
    torch.testing.assert_close(padded, plain, rtol=2e-3, atol=2e-3)


@mps_only
@pytest.mark.parametrize("rows", [1, 9, 10, 17, 27, 28])
def test_vocabulary_projection_row_chunking_matches_single_matmul(rows):
    from utils.model import Model
    torch.manual_seed(0)
    model = Model(vocab_size=96, hidden_size=32, intermediate_size=64, n_hidden_layers=1, head_dim=8, rope_theta=10000.0, n_query_heads=4, n_kv_heads=2, bias=False, attention_scaling=8 ** -0.5, rms_norm_eps=1e-6).to("mps", torch.float16).eval()
    hidden_states = torch.randn(rows, 32, device="mps", dtype=torch.float16)
    with torch.inference_mode():
        chunked = model.project_logits(hidden_states)
        expected = model.lm_head(hidden_states)
    assert chunked.shape == (rows, 96)
    torch.testing.assert_close(chunked, expected, rtol=2e-3, atol=2e-3)


def test_vocabulary_projection_ignores_row_chunking_off_mps():
    from utils.model import Model
    model = Model(vocab_size=96, hidden_size=32, intermediate_size=64, n_hidden_layers=1, head_dim=8, rope_theta=10000.0, n_query_heads=4, n_kv_heads=2, bias=False, attention_scaling=8 ** -0.5, rms_norm_eps=1e-6).eval()
    hidden_states = torch.randn(12, 32)
    with torch.inference_mode():
        assert torch.equal(model.project_logits(hidden_states), model.lm_head(hidden_states))


@mps_only
@pytest.mark.parametrize("rows", [1, 9, 16, 40])
def test_int8_linear_supports_dimensions_the_mps_kernel_rejects(rows):
    torch.manual_seed(0)
    linear = nn.Linear(48, 24, bias=True).to("mps", torch.float16)  # neither dimension is a multiple of 32
    quantized = Int8Linear.from_linear(linear)
    hidden_states = torch.randn(rows, 48, device="mps", dtype=torch.float16)
    with torch.inference_mode():
        torch.testing.assert_close(quantized(hidden_states), linear(hidden_states), rtol=0.05, atol=0.05)
