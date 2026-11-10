"""The engine must reproduce Hugging Face logits for every supported decoder family, and refuse the rest."""
import pytest
import torch
from transformers import GemmaConfig, LlamaConfig, MistralConfig, Phi3Config, Qwen2Config, Qwen3Config
from transformers import AutoModelForCausalLM

from utils.architecture import UnsupportedModelError
from utils.model_loader import model_loader

COMMON = dict(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
LLAMA3_ROPE = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 64}

SUPPORTED = {
    "qwen3": lambda: Qwen3Config(head_dim=16, **COMMON),
    "qwen2_bias": lambda: Qwen2Config(**COMMON),
    "llama": lambda: Llama(tie=False),
    "llama_tied": lambda: Llama(tie=True),
    "llama3_rope": lambda: Llama(tie=True, rope_scaling=LLAMA3_ROPE),
    "llama_linear_rope": lambda: Llama(tie=False, rope_scaling={"rope_type": "linear", "factor": 4.0}),
    "llama_mha": lambda: LlamaConfig(**{**COMMON, "num_key_value_heads": 4}),
    "llama_mlp_bias": lambda: LlamaConfig(mlp_bias=True, attention_bias=True, **COMMON),
    "mistral": lambda: MistralConfig(sliding_window=None, **COMMON),
}


def Llama(tie, rope_scaling=None):
    return LlamaConfig(tie_word_embeddings=tie, rope_scaling=rope_scaling, rope_theta=500000.0, **COMMON)


def save_random_model(config, path):
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config)
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.08)
    model.eval().save_pretrained(path)
    return model


@pytest.mark.parametrize("name", sorted(SUPPORTED))
def test_prefill_logits_match_huggingface(name, tmp_path):
    reference = save_random_model(SUPPORTED[name](), tmp_path)
    mine = model_loader(str(tmp_path), device=torch.device("cpu"), dtype=torch.float32, use_sdpa=True)
    ids = torch.randint(0, 128, (2, 11))
    with torch.no_grad():
        expected = reference(ids).logits
        actual = mine(ids)
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=2e-4, rtol=1e-3), (actual - expected).abs().max().item()
    assert mine.layers[0].self_attn.q_norm is None or name == "qwen3"


@pytest.mark.parametrize("name", ["qwen2_bias", "llama3_rope", "llama_mlp_bias"])
def test_quantized_and_unfused_models_still_load(name, tmp_path):
    save_random_model(SUPPORTED[name](), tmp_path)
    fused = model_loader(str(tmp_path), device=torch.device("cpu"), dtype=torch.float32, quantize="int8")
    unfused = model_loader(str(tmp_path), device=torch.device("cpu"), dtype=torch.float32, fuse_projections=False)
    ids = torch.randint(0, 128, (1, 7))
    with torch.no_grad():
        assert torch.allclose(fused(ids), unfused(ids), atol=0.15)


UNSUPPORTED = {
    "gemma": (lambda: GemmaConfig(head_dim=16, pad_token_id=0, **COMMON), "not supported"),
    "phi3": (lambda: Phi3Config(pad_token_id=0, **COMMON), "not supported"),
    "mistral_sliding": (lambda: MistralConfig(sliding_window=32, **COMMON), "sliding-window"),
    "llama_dynamic_rope": (lambda: Llama(tie=True, rope_scaling={"rope_type": "dynamic", "factor": 2.0}), "RoPE type"),
}


@pytest.mark.parametrize("name", sorted(UNSUPPORTED))
def test_unsupported_architectures_fail_with_a_clear_message(name, tmp_path):
    config_factory, expected = UNSUPPORTED[name]
    save_random_model(config_factory(), tmp_path)
    with pytest.raises(UnsupportedModelError, match=expected):
        model_loader(str(tmp_path), device=torch.device("cpu"), dtype=torch.float32)
