import torch
from transformers import AutoModelForCausalLM
import gc
from utils.model import Model
from utils.layers import quantize_linear_layers

def resolve_decode_attention(device, decode_attention):
    if decode_attention is None:
        return "gather_sdpa" if torch.device(device).type in ("mps", "cuda") else "direct"
    if decode_attention not in ("direct", "gather_sdpa"):
        raise ValueError(f"Unknown decode attention path: {decode_attention}")
    return decode_attention

def model_loader(model_name="Qwen/Qwen3-1.7B", device=None, dtype=None, use_sdpa=True, decode_attention=None, fuse_projections=True, quantize=None):
    if quantize not in (None, "int8", "int4"):
        raise ValueError(f"Unknown quantization mode: {quantize}")
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    decode_attention = resolve_decode_attention(device, decode_attention)
    # Load straight into the target dtype: a float32 copy of a 4B model is 16 GB, which exhausts a 16 GB Mac before the engine even starts.
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", **({"dtype": dtype} if dtype is not None else {})).eval()
    config = model.config
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    qwen_attention = model.model.layers[0].self_attn
    if dtype is None:
        dtype = qwen_attention.q_proj.weight.dtype
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is None:
        rope_theta = config.rope_parameters["rope_theta"]
    with torch.device("meta"):
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
            use_sdpa=use_sdpa,
            decode_attention=decode_attention,
        ).eval()
    my_model.rotary_embedding.dim_indices = torch.arange(0, head_dim, 2, dtype=torch.long)
    my_model.name = model_name
    reference_state_dict = model.state_dict()
    mapped_state_dict = {key.removeprefix("model."): value for key, value in reference_state_dict.items()}
    my_model.load_state_dict(mapped_state_dict, strict=True, assign=True)
    my_model.config = model.config
    my_model.generation_config = model.generation_config
    if config.tie_word_embeddings:
        my_model.lm_head.weight = my_model.embed_tokens.weight
    del mapped_state_dict
    del reference_state_dict
    del qwen_attention
    del model
    gc.collect()
    # Cast and fuse on the CPU, then move to the device one decoder layer at a time, quantizing each as it arrives. Doing the
    # concatenations on the GPU at full model size corrupted weights on a 16 GB Mac (Qwen3-4B), and moving everything before
    # quantizing needed the unquantized model plus the quantized one in GPU memory at once.
    my_model.to(dtype=dtype)
    if fuse_projections:
        my_model.fuse_projections()
    for layer in my_model.layers:
        layer.to(device=device)
        if quantize is not None:
            quantize_linear_layers(layer, quantize)
        gc.collect()
    my_model.to(device=device)
    if quantize is not None:
        quantize_linear_layers(my_model, quantize)  # embeddings' tied vocabulary projection; decoder layers are already quantized
        gc.collect()
    if torch.device(device).type == "mps":
        torch.mps.empty_cache()
    return my_model
