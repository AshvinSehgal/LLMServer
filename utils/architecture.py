"""Describes a Hugging Face decoder-only model in the terms the engine's Model understands.

The engine implements one architecture family: pre-norm decoder layers with grouped-query attention, rotary embeddings, a
gated MLP and RMSNorm. Llama, Mistral, Qwen2 and Qwen3 are all instances of it, differing only in a few switches (Q/K
normalization, which projections have biases, the activation, the rotary frequency schedule). Everything else is rejected
up front with a message that says what is different, instead of loading and producing garbage.
"""
from utils.layers import ACTIVATIONS

DECODER_LAYER_CHILDREN = {"input_layernorm", "self_attn", "post_attention_layernorm", "mlp"}
ATTENTION_CHILDREN = {"q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"}
MLP_CHILDREN = {"gate_proj", "up_proj", "down_proj", "act_fn"}
SCALED_ROPE_TYPES = ("linear", "llama3", "yarn")
# Families whose logits are verified against Hugging Face in tests/test_architectures.py. Others can share this layer structure yet
# differ in the forward pass (Gemma scales embeddings and uses (1 + weight) norms), so they are refused unless explicitly allowed.
VERIFIED_MODEL_TYPES = ("qwen3", "qwen2", "llama", "mistral")


class UnsupportedModelError(ValueError):
    pass


def rope_settings(config):
    """(theta, rope_type, rope_parameters dict)."""
    parameters = getattr(config, "rope_scaling", None) or getattr(config, "rope_parameters", None) or {}
    theta = getattr(config, "rope_theta", None)
    if theta is None:
        theta = parameters.get("rope_theta", 10000.0)
    rope_type = parameters.get("rope_type", parameters.get("type", "default"))
    return theta, rope_type, parameters


def rotary_frequencies(config):
    """Precomputed (inverse_frequencies, attention_factor) for scaled RoPE, or None for the plain theta schedule."""
    _, rope_type, _ = rope_settings(config)
    if rope_type in (None, "default"):
        return None
    if rope_type not in SCALED_ROPE_TYPES:
        raise UnsupportedModelError(f"RoPE type {rope_type!r} is not supported (supported: default, {', '.join(SCALED_ROPE_TYPES)}); dynamic and long-context schedules change with sequence length")
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    inverse_frequencies, attention_factor = ROPE_INIT_FUNCTIONS[rope_type](config, "cpu")
    return inverse_frequencies, attention_factor


def describe(model, allow_unverified=False):
    """Validates a loaded Hugging Face causal LM and returns keyword arguments for utils.model.Model."""
    config = model.config
    name = getattr(config, "model_type", type(model).__name__)

    def reject(reason):
        raise UnsupportedModelError(f"Model type {name!r} is not supported: {reason}")

    if name not in VERIFIED_MODEL_TYPES and not allow_unverified:
        reject(f"only {', '.join(VERIFIED_MODEL_TYPES)} are verified against Hugging Face. Other families can differ in subtle ways, so pass allow_unverified=True to try one and check its output yourself")
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if inner is None or layers is None or not hasattr(inner, "embed_tokens") or not hasattr(inner, "norm"):
        reject("expected model.model.{embed_tokens, layers, norm}")
    layer = layers[0]
    extra = {child for child, _ in layer.named_children()} - DECODER_LAYER_CHILDREN
    missing = DECODER_LAYER_CHILDREN - {child for child, _ in layer.named_children()}
    if extra or missing:
        reject(f"decoder layers must be input_layernorm/self_attn/post_attention_layernorm/mlp (extra: {sorted(extra)}, missing: {sorted(missing)}); extra norms, as in Gemma, or fused projections, as in Phi-3, need engine changes")
    attention, mlp = layer.self_attn, layer.mlp
    attention_children = {child for child, _ in attention.named_children()}
    if not {"q_proj", "k_proj", "v_proj", "o_proj"} <= attention_children or attention_children - ATTENTION_CHILDREN:
        reject(f"unexpected attention layout {sorted(attention_children)}")
    if {child for child, _ in mlp.named_children()} - MLP_CHILDREN or not {"gate_proj", "up_proj", "down_proj"} <= {child for child, _ in mlp.named_children()}:
        reject("the MLP must be a gated MLP with gate_proj/up_proj/down_proj")
    if ("q_norm" in attention_children) != ("k_norm" in attention_children):
        reject("q_norm and k_norm must come together")
    use_sliding = getattr(config, "use_sliding_window", None)
    if use_sliding is None:
        use_sliding = getattr(config, "sliding_window", None) is not None
    if use_sliding or "sliding_attention" in (getattr(config, "layer_types", None) or []):
        reject("sliding-window attention is not implemented")
    for attribute in ("attn_logit_softcapping", "final_logit_softcapping"):
        if getattr(config, attribute, None):
            reject(f"{attribute} is not implemented")
    if getattr(config, "partial_rotary_factor", 1.0) not in (1.0, None):
        reject("partial rotary embeddings are not implemented")
    if getattr(model, "lm_head", None) is None or model.lm_head.bias is not None:
        reject("expected a bias-free lm_head")
    hidden_act = getattr(config, "hidden_act", "silu")
    if hidden_act not in ACTIVATIONS:
        reject(f"activation {hidden_act!r} is not supported (supported: {sorted(ACTIVATIONS)})")

    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    theta, _, _ = rope_settings(config)
    kwargs = dict(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        n_hidden_layers=config.num_hidden_layers,
        head_dim=head_dim,
        rope_theta=theta,
        n_query_heads=config.num_attention_heads,
        n_kv_heads=config.num_key_value_heads,
        bias=False,
        qkv_bias=attention.q_proj.bias is not None,
        o_bias=attention.o_proj.bias is not None,
        mlp_bias=mlp.gate_proj.bias is not None,
        qk_norm="q_norm" in attention_children,
        hidden_act=hidden_act,
        attention_scaling=getattr(attention, "scaling", head_dim ** -0.5),
        rms_norm_eps=config.rms_norm_eps,
    )
    return kwargs, rotary_frequencies(config)
