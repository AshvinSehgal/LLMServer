import torch
from transformers import AutoModelForCausalLM
import gc
from utils.model import Model

def model_loader(model_name="Qwen/Qwen3-1.7B", device=None, dtype=None):
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager").eval()
    config = model.config
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    qwen_attention = model.model.layers[0].self_attn
    if dtype is None:
        dtype = qwen_attention.q_proj.weight.dtype
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is None:
        rope_theta = config.rope_parameters["rope_theta"]
    my_model = Model(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        n_hidden_layers=config.num_hidden_layers,
        head_dim=head_dim,
        rope_theta=rope_theta,
        n_query_heads=config.num_attention_heads,
        n_kv_heads=config.num_key_value_heads,
        bias=qwen_attention.q_proj.bias is not None,
        attention_scaling=qwen_attention.scaling,
        rms_norm_eps=config.rms_norm_eps,
    ).to(dtype=dtype).eval()
    reference_state_dict = model.state_dict()
    mapped_state_dict = {key.removeprefix("model."): value for key, value in reference_state_dict.items()}
    my_model.load_state_dict(mapped_state_dict, strict=True)
    my_model.config = model.config
    my_model.generation_config = model.generation_config
    if config.tie_word_embeddings:
        my_model.lm_head.weight = my_model.embed_tokens.weight
    del mapped_state_dict
    del reference_state_dict
    del qwen_attention
    del model
    gc.collect()
    my_model.to(device=device)
    return my_model
