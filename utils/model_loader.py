import torch
from transformers import AutoModelForCausalLM
import gc
from utils.model import Model
from utils.layers import quantize_linear_layers
from utils.architecture import describe, UnsupportedModelError

def resolve_decode_attention(device, decode_attention):
    if decode_attention is None:
        return "gather_sdpa" if torch.device(device).type in ("mps", "cuda") else "direct"
    if decode_attention not in ("direct", "gather_sdpa"):
        raise ValueError(f"Unknown decode attention path: {decode_attention}")
    return decode_attention

def model_loader(model_name="Qwen/Qwen3-1.7B", device=None, dtype=None, use_sdpa=True, decode_attention=None, fuse_projections=True, quantize=None, allow_unverified=False):
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
    model_kwargs, rotary = describe(model, allow_unverified=allow_unverified)
    head_dim = model_kwargs["head_dim"]
    if dtype is None:
        dtype = model.model.layers[0].self_attn.q_proj.weight.dtype
    with torch.device("meta"):
        my_model = Model(use_sdpa=use_sdpa, decode_attention=decode_attention, **model_kwargs).eval()
    my_model.rotary_embedding.dim_indices = torch.arange(0, head_dim, 2, dtype=torch.long)
    if rotary is not None:
        my_model.rotary_embedding.set_inverse_frequencies(*rotary)
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
