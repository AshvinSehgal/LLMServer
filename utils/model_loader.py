import torch
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
import gc
from utils.model import Model
from utils.layers import quantize_linear_layers
from utils.architecture import describe, UnsupportedModelError
from utils.weights import CheckpointReader

LARGE_MODEL_PARAMETERS = 6e9  # above this, the embedding table stays on the CPU (see model_loader)

def resolve_decode_attention(device, decode_attention):
    if decode_attention is None:
        return "gather_sdpa" if torch.device(device).type in ("mps", "cuda") else "direct"
    if decode_attention not in ("direct", "gather_sdpa"):
        raise ValueError(f"Unknown decode attention path: {decode_attention}")
    return decode_attention

def model_loader(model_name="Qwen/Qwen3-1.7B", device=None, dtype=None, use_sdpa=True, decode_attention=None, fuse_projections=True, quantize=None, allow_unverified=False, embedding_on_cpu=None):
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
    reader = CheckpointReader.open(model_name)
    if reader is None:
        return load_without_streaming(model_name, device, dtype, use_sdpa, decode_attention, fuse_projections, quantize, allow_unverified)
    config = AutoConfig.from_pretrained(model_name)
    with torch.device("meta"):
        reference = AutoModelForCausalLM.from_config(config, attn_implementation="eager")  # structure only, no memory
    model_kwargs, rotary = describe(reference, allow_unverified=allow_unverified)
    head_dim = model_kwargs["head_dim"]
    dtype = dtype if dtype is not None else torch.get_default_dtype()
    with torch.device("meta"):
        my_model = Model(use_sdpa=use_sdpa, decode_attention=decode_attention, **model_kwargs).eval()
    my_model.rotary_embedding.dim_indices = torch.arange(0, head_dim, 2, dtype=torch.long)
    if rotary is not None:
        my_model.rotary_embedding.set_inverse_frequencies(*rotary)
    my_model.name = model_name
    my_model.config = config
    my_model.generation_config = GenerationConfig.from_pretrained(model_name)
    if embedding_on_cpu is None:
        embedding_on_cpu = torch.device(device).type == "mps" and sum(parameter.numel() for parameter in my_model.parameters()) > LARGE_MODEL_PARAMETERS

    def take(module, prefix, drop=()):
        """Copies the checkpoint tensors under prefix into module's (meta) parameters, cast to dtype."""
        state = {key: tensor.to(dtype) for key, tensor in reader.with_prefix(prefix).items() if key not in drop}
        module.load_state_dict(state, strict=True, assign=True)

    take(my_model.embed_tokens, "model.embed_tokens.")
    take(my_model.norm, "model.norm.")
    if config.tie_word_embeddings or not reader.has("lm_head.weight"):
        my_model.lm_head.weight = my_model.embed_tokens.weight
    else:
        take(my_model.lm_head, "lm_head.")
    if embedding_on_cpu:
        # A lookup of a few rows does not need the GPU; keeping a 1+ GB table off it matters on a 16 GB machine.
        if my_model.lm_head.weight is my_model.embed_tokens.weight:
            my_model.lm_head.weight = torch.nn.Parameter(my_model.embed_tokens.weight.detach())  # separate Parameter so lm_head can move to the GPU alone
        my_model.embedding_on_cpu = True
    for index, layer in enumerate(my_model.layers):
        # Materialize, fuse and quantize one decoder layer at a time so only a single layer is ever held at full precision.
        take(layer, f"model.layers.{index}.")
        if fuse_projections:
            layer.self_attn.fuse_projections()
            layer.mlp.fuse_projections()
        layer.to(device=device)
        if quantize is not None:
            quantize_linear_layers(layer, quantize)
        gc.collect()
    reader.close()
    my_model.norm.to(device=device)
    my_model.lm_head.to(device=device)
    my_model.rotary_embedding.to(device=device)
    if not embedding_on_cpu:
        my_model.embed_tokens.to(device=device)
    if quantize is not None:
        quantize_linear_layers(my_model, quantize)  # the vocabulary projection; decoder layers are already quantized
    gc.collect()
    if torch.device(device).type == "mps":
        torch.mps.empty_cache()
    return my_model


def load_without_streaming(model_name, device, dtype, use_sdpa, decode_attention, fuse_projections, quantize, allow_unverified):
    """Fallback for checkpoints that are not safetensors: load everything through transformers, then convert."""
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
        quantize_linear_layers(my_model, quantize)
        gc.collect()
    if torch.device(device).type == "mps":
        torch.mps.empty_cache()
    return my_model
