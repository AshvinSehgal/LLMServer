import torch
import torch.nn as nn
from utils.layers import RMSNorm, apply_rotary_embedding

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, n_query_heads, n_kv_heads, head_dim, bias, attention_scaling, rms_norm_eps):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, n_query_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(n_query_heads * head_dim, hidden_size, bias=bias)
        self.q_norm = RMSNorm(head_dim, rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, rms_norm_eps)
        self.bias = bias
        self.n_query_heads = n_query_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.attention_scaling = attention_scaling
    
    def forward(self, hidden_states, pos_embeddings, causal_mask, kv_cache_manager=None, layer_index=None, request_ids=None, start_pos=None, input_lens=None):
        Q = self.q_proj(hidden_states)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)
        batch_size, seq_len, _ = K.shape
        Q = Q.view(batch_size, seq_len, self.n_query_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        Q = self.q_norm(Q)
        K = self.k_norm(K)
        cosine, sine = pos_embeddings
        Q, K = apply_rotary_embedding(Q, K, cosine, sine)
        if self.n_query_heads % self.n_kv_heads != 0:
            raise ValueError(f"Number of query heads must be divisible by the number of KV heads: query heads: {self.n_query_heads}, KV heads: {self.n_kv_heads}")
        group_size = self.n_query_heads // self.n_kv_heads
        if kv_cache_manager is None:
            K = torch.repeat_interleave(K, repeats=group_size, dim=1)
            V = torch.repeat_interleave(V, repeats=group_size, dim=1)
            attention_weights = torch.matmul(Q, K.mT) * self.attention_scaling
            attention_weights += causal_mask
            attention_weights = torch.softmax(attention_weights, dim=-1, dtype=torch.float32).to(Q.dtype)
            attention_scores = torch.matmul(attention_weights, V)
        else:
            if layer_index is None or request_ids is None or start_pos is None or input_lens is None:
                raise ValueError(f"Cache metadata is required when using the KV cache: layer_index: {layer_index}, request_ids: {request_ids}, start_pos: {start_pos}, input_lens: {input_lens}")
            if len(request_ids) != batch_size or len(start_pos) != batch_size or len(input_lens) != batch_size:
                raise ValueError(f"Cache metadata must contain one entry per request: batch_size: {batch_size}, request_ids: {len(request_ids)}, start_pos: {len(start_pos)}, input_lens: {len(input_lens)}")
            attention_scores = torch.zeros_like(Q)
            for batch_index in range(batch_size):
                request_id = request_ids[batch_index]
                request_start_pos = start_pos[batch_index]
                request_input_len = input_lens[batch_index]
                if request_input_len <= 0 or request_input_len > seq_len:
                    raise ValueError(f"Invalid input length for cached attention: {request_input_len}, sequence length: {seq_len}")
                request_end_pos = request_start_pos + request_input_len
                request_q = Q[batch_index:batch_index + 1, :, :request_input_len, :]
                new_keys = K[batch_index, :, :request_input_len, :]
                new_values = V[batch_index, :, :request_input_len, :]
                kv_cache_manager.append(request_id, layer_index, request_start_pos, new_keys, new_values)
                cached_keys, cached_values = kv_cache_manager.get(request_id, layer_index, request_end_pos)
                cached_keys = cached_keys.unsqueeze(0)
                cached_values = cached_values.unsqueeze(0)
                cached_keys = torch.repeat_interleave(cached_keys, repeats=group_size, dim=1)
                cached_values = torch.repeat_interleave(cached_values, repeats=group_size, dim=1)
                request_attention_weights = torch.matmul(request_q, cached_keys.mT) * self.attention_scaling
                request_mask = causal_mask[batch_index:batch_index + 1, :, :request_input_len, :request_end_pos]
                request_attention_weights += request_mask
                request_attention_weights = torch.softmax(request_attention_weights, dim=-1, dtype=torch.float32).to(request_q.dtype)
                request_attention_scores = torch.matmul(request_attention_weights, cached_values)
                attention_scores[batch_index:batch_index + 1, :, :request_input_len, :] = request_attention_scores
        attention_scores = attention_scores.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        output = self.o_proj(attention_scores)
        return output
        
