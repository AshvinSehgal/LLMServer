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
    
    def forward(self, hidden_states, pos_embeddings, causal_mask):
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
            raise ValueError("Number of query heads must be divisible by the number of KV heads")
        group_size = self.n_query_heads // self.n_kv_heads
        K = torch.repeat_interleave(K, repeats=group_size, dim=1)
        V = torch.repeat_interleave(V, repeats=group_size, dim=1)
        attention_weights = torch.matmul(Q, K.mT) * self.attention_scaling
        attention_weights += causal_mask
        attention_weights = torch.softmax(attention_weights, dim=-1, dtype=torch.float32).to(Q.dtype)
        attention_scores = torch.matmul(attention_weights, V)
        attention_scores = attention_scores.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        output = self.o_proj(attention_scores)
        return output
        
