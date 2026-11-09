import torch
import torch.nn as nn
import torch.nn.functional as F

class Int8Linear(nn.Module):
    """Weight-only int8 linear layer with one scale per output channel.

    Decoding is memory-bandwidth bound, so reading 1 byte per weight instead of 2 nearly
    halves the time of each matmul. Activations stay in the model dtype.
    """
    QUANTIZE_CHUNK_ROWS = 16384

    def __init__(self, weight, scale, bias=None):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.register_buffer("weight", weight)
        self.register_buffer("scale", scale)
        self.register_buffer("bias", bias)

    @classmethod
    def from_linear(cls, linear):
        source = linear.weight.detach()
        weight = torch.empty(source.shape, dtype=torch.int8, device=source.device)
        scale = torch.empty(source.shape[0], dtype=source.dtype, device=source.device)
        for start in range(0, source.shape[0], cls.QUANTIZE_CHUNK_ROWS):
            rows = source[start:start + cls.QUANTIZE_CHUNK_ROWS].float()
            row_scale = (rows.abs().amax(dim=1) / 127).clamp_min(torch.finfo(torch.float32).tiny)
            weight[start:start + rows.shape[0]] = (rows / row_scale[:, None]).round().clamp_(-127, 127).to(torch.int8)
            scale[start:start + rows.shape[0]] = row_scale.to(source.dtype)
        bias = None if linear.bias is None else linear.bias.detach()
        return cls(weight, scale, bias)

    # MPS's int8 kernel is a matrix-vector kernel whose cost climbs steeply from 8 rows; padding 8-10 rows to 16
    # lands on its tiled kernel, which is faster. Below 8 rows the natural size wins, and from 11 rows the extra
    # pad and slice kernels cost more than they save.
    PAD_ROWS_FROM = 8
    PAD_ROWS_BELOW = 11
    PAD_ROWS_TO = 16

    def forward(self, hidden_states):
        flat = hidden_states.reshape(-1, self.in_features).contiguous()
        # MPS's tiled int8 kernel (used from 16 rows) needs both dimensions to be multiples of 32.
        kernel_supported = flat.device.type != "mps" or (self.in_features % 32 == 0 and self.out_features % 32 == 0)
        if kernel_supported and flat.dtype in (torch.float16, torch.float32) and self.scale.dtype == flat.dtype:
            rows = flat.shape[0]
            if flat.device.type == "mps" and self.PAD_ROWS_FROM <= rows < self.PAD_ROWS_BELOW:
                output = torch._weight_int8pack_mm(F.pad(flat, (0, 0, 0, self.PAD_ROWS_TO - rows)), self.weight, self.scale)[:rows]
            else:
                output = torch._weight_int8pack_mm(flat, self.weight, self.scale)
        else:
            output = F.linear(flat, self.weight.to(flat.dtype) * self.scale.to(flat.dtype)[:, None])
        if self.bias is not None:
            output = output + self.bias
        return output.reshape(*hidden_states.shape[:-1], self.out_features)

class Int4Linear(Int8Linear):
    """Group-wise int4 weights for 1-2 row inputs, with an int8 copy for everything else.

    MPS's int4 kernel is a matrix-vector kernel: it beats int8 at one or two rows (single-stream decode)
    but is several times slower for larger batches and prefill, so those rows use the int8 weights.
    """
    MAX_INT4_ROWS = 2
    GROUP_SIZE = 32

    def __init__(self, weight, scale, packed, scale_zero, group_size, bias=None):
        super().__init__(weight, scale, bias)
        self.group_size = group_size
        self.register_buffer("packed", packed)
        self.register_buffer("scale_zero", scale_zero)

    @staticmethod
    def supports(linear, group_size):
        weight = linear.weight
        return weight.device.type == "mps" and weight.dtype in (torch.float16, torch.bfloat16) and weight.shape[1] % group_size == 0 and weight.shape[1] % 32 == 0 and weight.shape[0] % 8 == 0

    @classmethod
    def from_linear(cls, linear, group_size=None):
        group_size = group_size or cls.GROUP_SIZE
        base = Int8Linear.from_linear(linear)
        if not cls.supports(linear, group_size):
            return base
        source = linear.weight.detach()
        out_features, in_features = source.shape
        packed_rows = []
        scale_zero = torch.empty(in_features // group_size, out_features, 2, dtype=source.dtype, device=source.device)
        for start in range(0, out_features, cls.QUANTIZE_CHUNK_ROWS):
            rows = source[start:start + cls.QUANTIZE_CHUNK_ROWS].float().reshape(-1, in_features // group_size, group_size)
            minimum = rows.amin(dim=-1)
            step = ((rows.amax(dim=-1) - minimum) / 15).clamp_min(1e-6)
            levels = ((rows - minimum[..., None]) / step[..., None]).round().clamp_(0, 15).to(torch.uint8).reshape(-1, in_features)
            packed_rows.append(torch._convert_weight_to_int4pack((levels[:, 0::2] << 4) | levels[:, 1::2], 2))
            scale_zero[:, start:start + rows.shape[0], 0] = step.t().to(source.dtype)
            scale_zero[:, start:start + rows.shape[0], 1] = (minimum + 8 * step).t().to(source.dtype)
        return cls(base.weight, base.scale, torch.cat(packed_rows, dim=0), scale_zero, group_size, base.bias)

    def forward(self, hidden_states):
        flat = hidden_states.reshape(-1, self.in_features).contiguous()
        if flat.shape[0] > self.MAX_INT4_ROWS or flat.dtype != self.scale_zero.dtype:
            return super().forward(hidden_states)
        output = torch._weight_int4pack_mm(flat, self.packed, self.group_size, self.scale_zero)
        if self.bias is not None:
            output = output + self.bias
        return output.reshape(*hidden_states.shape[:-1], self.out_features)

def quantize_linear_layers(module, mode="int8"):
    """Recursively replace every nn.Linear under module with a weight-only quantized layer.

    "int4" uses group-wise int4 for decoder layers; the vocabulary projection stays int8 because
    its errors go straight into the logits.
    """
    if mode not in ("int8", "int4"):
        raise ValueError(f"Unknown quantization mode: {mode}")
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            use_int4 = mode == "int4" and name != "lm_head"
            setattr(module, name, Int4Linear.from_linear(child) if use_int4 else Int8Linear.from_linear(child))
        else:
            quantize_linear_layers(child, mode)

class RMSNorm(nn.Module):
    def __init__(self, normalization_size, eps=1e-6, use_native=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalization_size))
        self.eps = eps
        self.use_native = use_native
    
    def forward(self, hidden_states):
        if self.use_native:
            return F.rms_norm(hidden_states, (self.weight.shape[0],), self.weight, self.eps)
        original_dtype = hidden_states.dtype
        hidden_states_fp32 = hidden_states.to(torch.float32)
        variance = hidden_states_fp32.pow(2).mean(dim=-1, keepdim=True)
        normalized_states = hidden_states_fp32 * torch.rsqrt(variance + self.eps)
        normalized_states = normalized_states.to(original_dtype)
        return self.weight * normalized_states

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, rope_theta):
        super().__init__()
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        dim_indices = torch.arange(0, head_dim, 2, dtype=torch.long)
        self.register_buffer("dim_indices", dim_indices, persistent=False)
        self._inverse_frequencies = None
    
    def forward(self, hidden_states, pos_ids):
        if self._inverse_frequencies is None or self._inverse_frequencies.device != pos_ids.device:
            self._inverse_frequencies = 1.0 / (self.rope_theta ** (self.dim_indices.to(device=pos_ids.device, dtype=torch.float32) / self.head_dim))
        frequencies = pos_ids.to(torch.float32).unsqueeze(-1) * self._inverse_frequencies.view(1, 1, -1)
        angles = torch.cat([frequencies, frequencies], dim=-1)
        cosine = torch.cos(angles)
        sine = torch.sin(angles)
        return cosine.to(device=hidden_states.device, dtype=hidden_states.dtype), sine.to(device=hidden_states.device, dtype=hidden_states.dtype)

def rotate_half(hidden_states):
    half_size = hidden_states.shape[-1] // 2
    first_half = hidden_states[...,:half_size]
    second_half = hidden_states[...,half_size:]
    return torch.cat([-second_half, first_half], dim=-1)

def apply_rotary_embedding(query, key, cosine, sine):
    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    rotated_query = query * cosine + rotate_half(query) * sine
    rotated_key = key * cosine + rotate_half(key) * sine
    return rotated_query, rotated_key

class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, bias):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.gate_up_proj = None
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act_fn = nn.SiLU()

    def fuse_projections(self):
        if self.gate_up_proj is not None:
            return
        fused = nn.Linear(self.gate_proj.in_features, self.gate_proj.out_features + self.up_proj.out_features, bias=self.gate_proj.bias is not None, device="meta")
        fused.weight = nn.Parameter(torch.cat((self.gate_proj.weight, self.up_proj.weight), dim=0))
        if self.gate_proj.bias is not None:
            fused.bias = nn.Parameter(torch.cat((self.gate_proj.bias, self.up_proj.bias), dim=0))
        self.gate_up_proj = fused
        self.gate_proj = None
        self.up_proj = None
    
    def forward(self, hidden_states):
        if self.gate_up_proj is None:
            gate = self.gate_proj(hidden_states)
            up = self.up_proj(hidden_states)
        else:
            gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        intermediate = self.act_fn(gate) * up
        output = self.down_proj(intermediate)
        return output
