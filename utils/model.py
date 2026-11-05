import torch
import torch.nn as nn
from utils.layers import RMSNorm, RotaryEmbedding, MLP
from utils.attention import SelfAttention

class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, n_query_heads, n_kv_heads, head_dim, bias, attention_scaling, rms_norm_eps, use_sdpa=False, decode_attention="direct"):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, rms_norm_eps)
        self.self_attn = SelfAttention(hidden_size, n_query_heads, n_kv_heads, head_dim, bias, attention_scaling, rms_norm_eps, use_sdpa=use_sdpa, decode_attention=decode_attention)
        self.post_attention_layernorm = RMSNorm(hidden_size, rms_norm_eps)
        self.mlp = MLP(hidden_size, intermediate_size, bias)
    
    def forward(self, hidden_states, pos_embeddings, causal_mask, kv_cache_manager=None, layer_index=None, request_ids=None, block_tables=None, start_pos=None, input_lens=None):
        residual = hidden_states
        normalized_hidden_states = self.input_layernorm(residual)
        attention_scores = self.self_attn(
            normalized_hidden_states, 
            pos_embeddings, 
            causal_mask, 
            kv_cache=kv_cache_manager, 
            layer_index=layer_index, 
            request_ids=request_ids,
            block_tables=block_tables, 
            start_pos=start_pos, 
            input_lens=input_lens
            )
        hidden_states = residual + attention_scores
        residual = hidden_states
        normalized_hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(normalized_hidden_states)
        hidden_states = residual + mlp_output
        return hidden_states

class Model(nn.Module):
    def __init__(self, vocab_size, hidden_size, intermediate_size, n_hidden_layers, head_dim, rope_theta, n_query_heads, n_kv_heads, bias, attention_scaling, rms_norm_eps, use_sdpa=False, decode_attention="direct"):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.rotary_embedding = RotaryEmbedding(head_dim, rope_theta)
        self.layers = nn.ModuleList([
            DecoderLayer(hidden_size, intermediate_size, n_query_heads, n_kv_heads, head_dim, bias, attention_scaling, rms_norm_eps, use_sdpa=use_sdpa, decode_attention=decode_attention)
            for _ in range(n_hidden_layers)])
        self.norm = RMSNorm(hidden_size, rms_norm_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
    def forward(self, input_ids, pos_ids=None, causal_mask=None, kv_cache_manager=None, request_ids=None, block_tables=None, start_pos=None, input_lens=None):
        hidden_states = self.embed_tokens(input_ids)
        batch_size, seq_len = input_ids.shape
        if pos_ids is None:
            pos_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.int).unsqueeze(0)
        cosine, sine = self.rotary_embedding(hidden_states, pos_ids)
        pos_embeddings = (cosine, sine)
        if causal_mask is None:
            mask_value = torch.finfo(hidden_states.dtype).min
            causal_mask = torch.full((batch_size, 1, seq_len, seq_len), mask_value, device=hidden_states.device, dtype=hidden_states.dtype).triu(diagonal=1)
        for layer_index, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states, 
                pos_embeddings, 
                causal_mask,
                kv_cache_manager=kv_cache_manager,
                layer_index=layer_index,
                request_ids=request_ids,
                block_tables=block_tables,
                start_pos=start_pos,
                input_lens=input_lens
                )
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits
        
