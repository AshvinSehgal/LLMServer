import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, normalization_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalization_size))
        self.eps = eps
    
    def forward(self, hidden_states):
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
    
    def forward(self, hidden_states, pos_ids):
        inverse_frequencies = 1.0 / (self.rope_theta ** (self.dim_indices.to(torch.float32) / self.head_dim))
        frequencies = pos_ids.to(torch.float32).unsqueeze(-1) * inverse_frequencies.view(1, 1, -1)
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
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act_fn = nn.SiLU()
    
    def forward(self, hidden_states):
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        intermediate = self.act_fn(gate) * up
        output = self.down_proj(intermediate)
        return output
