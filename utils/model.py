import torch
import torch.nn as nn
from utils.layers import RMSNorm, RotaryEmbedding, MLP, quantize_linear_layers
from utils.attention import SelfAttention
from utils.fused_ops import add_rms_norm, fused_ops_usable

class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, n_query_heads, n_kv_heads, head_dim, bias, attention_scaling, rms_norm_eps, use_sdpa=False, decode_attention="direct"):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, rms_norm_eps)
        self.self_attn = SelfAttention(hidden_size, n_query_heads, n_kv_heads, head_dim, bias, attention_scaling, rms_norm_eps, use_sdpa=use_sdpa, decode_attention=decode_attention)
        self.post_attention_layernorm = RMSNorm(hidden_size, rms_norm_eps)
        self.mlp = MLP(hidden_size, intermediate_size, bias)
    
    def forward(self, hidden_states, pos_embeddings, causal_mask, kv_cache_manager=None, layer_index=None, request_ids=None, block_tables=None, start_pos=None, input_lens=None, packed=False):
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
            input_lens=input_lens,
            packed=packed
            )
        hidden_states = residual + attention_scores
        residual = hidden_states
        normalized_hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(normalized_hidden_states)
        hidden_states = residual + mlp_output
        return hidden_states

    def forward_decode_paged(self, hidden_states, pos_embeddings, kv_cache, layer_index, slot_mapping, block_tables, seq_lens):
        residual = hidden_states
        attention_output = self.self_attn.forward_decode_paged(self.input_layernorm(residual), pos_embeddings, kv_cache, layer_index, slot_mapping, block_tables, seq_lens)
        hidden_states = residual + attention_output
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))

class Model(nn.Module):
    def __init__(self, vocab_size, hidden_size, intermediate_size, n_hidden_layers, head_dim, rope_theta, n_query_heads, n_kv_heads, bias, attention_scaling, rms_norm_eps, use_sdpa=False, decode_attention="direct"):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.rotary_embedding = RotaryEmbedding(head_dim, rope_theta)
        self.layers = nn.ModuleList([
            DecoderLayer(hidden_size, intermediate_size, n_query_heads, n_kv_heads, head_dim, bias, attention_scaling, rms_norm_eps, use_sdpa=use_sdpa, decode_attention=decode_attention)
            for _ in range(n_hidden_layers)])
        self.norm = RMSNorm(hidden_size, rms_norm_eps)
        self.fused_decode_ops = False  # set by the engine when the fused CUDA kernels are usable
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def fuse_projections(self):
        for layer in self.layers:
            layer.self_attn.fuse_projections()
            layer.mlp.fuse_projections()
        
    def forward_decode_paged(self, input_ids, pos_ids, slot_mapping, block_tables, seq_lens, kv_cache_manager):
        """Decode one token per row against the paged cache using only tensor inputs (CUDA-graph friendly).

        input_ids, pos_ids: [rows, 1]; slot_mapping: [rows]; block_tables: [rows, max_blocks]; seq_lens: [rows].
        Returns logits [rows, vocab].
        """
        hidden_states = self.embed_tokens(input_ids.long())
        cosine, sine = self.rotary_embedding(hidden_states, pos_ids)
        half_dim = sine.shape[-1] // 2
        pos_embeddings = (cosine, sine, torch.cat([-sine[..., :half_dim], sine[..., half_dim:]], dim=-1))
        slot_mapping = slot_mapping.long()
        if self.fused_decode_ops and self.layers[0].mlp.gate_up_proj is not None:
            # Residual add + RMSNorm and SiLU*up each run as one kernel; the residual stream stays in `hidden_states`.
            eps = self.norm.eps
            hidden_states, normed = add_rms_norm(hidden_states, None, self.layers[0].input_layernorm.weight, eps)
            for layer_index, layer in enumerate(self.layers):
                attention_output = layer.self_attn.forward_decode_paged(normed, pos_embeddings, kv_cache_manager, layer_index, slot_mapping, block_tables, seq_lens)
                hidden_states, normed = add_rms_norm(hidden_states, attention_output, layer.post_attention_layernorm.weight, eps)
                mlp_output = layer.mlp.forward_fused(normed)
                next_weight = self.layers[layer_index + 1].input_layernorm.weight if layer_index + 1 < len(self.layers) else self.norm.weight
                hidden_states, normed = add_rms_norm(hidden_states, mlp_output, next_weight, eps)
            return self.project_logits(normed[:, 0])
        for layer_index, layer in enumerate(self.layers):
            hidden_states = layer.forward_decode_paged(hidden_states, pos_embeddings, kv_cache_manager, layer_index, slot_mapping, block_tables, seq_lens)
        return self.project_logits(self.norm(hidden_states[:, 0]))

    def quantize(self, mode="int8"):
        """Replace every Linear (including lm_head) with a weight-only quantized layer; embeddings and norms stay in the model dtype."""
        quantize_linear_layers(self, mode)

    def quantize_int8(self):
        self.quantize("int8")

    def forward(self, input_ids, pos_ids=None, causal_mask=None, kv_cache_manager=None, request_ids=None, block_tables=None, start_pos=None, input_lens=None, logit_rows=None, logit_positions=None, packed=False):
        hidden_states = self.embed_tokens(input_ids)
        batch_size, seq_len = input_ids.shape
        if pos_ids is None:
            pos_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.int).unsqueeze(0)
        cosine, sine = self.rotary_embedding(hidden_states, pos_ids)
        half_dim = sine.shape[-1] // 2
        pos_embeddings = (cosine, sine, torch.cat([-sine[..., :half_dim], sine[..., half_dim:]], dim=-1))
        if causal_mask is None and (kv_cache_manager is None or seq_len > 1):
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
                input_lens=input_lens,
                packed=packed
                )
        if logit_rows is not None:
            if logit_positions is None:
                raise ValueError("logit_positions are required when logit_rows are provided")
            if len(logit_rows) == 0:
                return hidden_states.new_empty((0, self.lm_head.out_features))
            if not packed and seq_len == 1 and len(logit_rows) == batch_size and all(row == index for index, row in enumerate(logit_rows)):
                hidden_states = hidden_states[:, 0]  # decode: every row, its only position
            else:
                rows = torch.as_tensor(logit_rows, device=hidden_states.device, dtype=torch.long)
                hidden_states = hidden_states[0, logit_positions[rows]] if packed else hidden_states[rows, logit_positions[rows]]
        hidden_states = self.norm(hidden_states)
        return self.project_logits(hidden_states)

    # MPS's fp16/fp32 matmul against the vocabulary matrix is about 3x slower from 10 rows up, so
    # 10-27 rows run faster as chunks of at most 9 rows.
    MAX_FAST_LOGIT_ROWS = 9
    MAX_CHUNKED_LOGIT_ROWS = 27

    def project_logits(self, hidden_states):
        rows = hidden_states.shape[0] if hidden_states.ndim == 2 else 0
        if isinstance(self.lm_head, nn.Linear) and hidden_states.device.type == "mps" and self.MAX_FAST_LOGIT_ROWS < rows <= self.MAX_CHUNKED_LOGIT_ROWS:
            chunk_rows = -(-rows // -(-rows // self.MAX_FAST_LOGIT_ROWS))
            return torch.cat([self.lm_head(chunk) for chunk in hidden_states.split(chunk_rows)])
        return self.lm_head(hidden_states)
        
