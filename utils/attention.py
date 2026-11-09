import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.layers import RMSNorm, apply_rotary_embedding

def joint_qk_norm_rotary(qk, weight, cosine, signed_sine, eps):
    """RMS-normalize and rotate stacked Q/K heads ([batch, seq, heads, head_dim]) in one pass."""
    qk = F.rms_norm(qk, (qk.shape[-1],), None, eps) * weight
    half = qk.shape[-1] // 2
    # Swapping the halves is cheaper as a slice-and-concatenate than as a gather; the sign lives in signed_sine.
    return torch.addcmul(qk * cosine.unsqueeze(2), torch.cat([qk[..., half:], qk[..., :half]], dim=-1), signed_sine.unsqueeze(2))

class SelfAttention(nn.Module):

    def __init__(self, hidden_size, n_query_heads, n_kv_heads, head_dim, bias, attention_scaling, rms_norm_eps, use_sdpa=False, decode_attention="direct"):
        super().__init__()
        if decode_attention not in ("direct", "gather_sdpa"):
            raise ValueError(f"Unknown decode attention path: {decode_attention}")
        self.q_proj = nn.Linear(hidden_size, n_query_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=bias)
        self.qkv_proj = None
        self.o_proj = nn.Linear(n_query_heads * head_dim, hidden_size, bias=bias)
        self.q_norm = RMSNorm(head_dim, rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, rms_norm_eps)
        self.bias = bias
        self.n_query_heads = n_query_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.attention_scaling = attention_scaling
        self.use_sdpa = use_sdpa
        self.decode_attention = decode_attention
        self.batched_decode = True
        self.grouped_single_decode = True
        self.joint_qk_rotary = True
        self._qk_norm_weight = None
        self._qk_norm_key = None

    def fuse_projections(self):
        if self.qkv_proj is not None:
            return
        projections = (self.q_proj, self.k_proj, self.v_proj)
        output_sizes = [projection.out_features for projection in projections]
        fused = nn.Linear(self.q_proj.in_features, sum(output_sizes), bias=self.bias, device="meta")
        fused.weight = nn.Parameter(torch.cat([projection.weight for projection in projections], dim=0))
        if self.bias:
            fused.bias = nn.Parameter(torch.cat([projection.bias for projection in projections], dim=0))
        self.qkv_proj = fused
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
    
    def qk_norm_weight(self, reference):
        key = (self.q_norm.weight.data_ptr(), self.k_norm.weight.data_ptr(), reference.dtype, reference.device)
        if key != self._qk_norm_key:
            self._qk_norm_weight = torch.cat([self.q_norm.weight.expand(self.n_query_heads, -1), self.k_norm.weight.expand(self.n_kv_heads, -1)], dim=0).to(reference.dtype)
            self._qk_norm_key = key
        return self._qk_norm_weight

    def direct_paged_read(self, request_q, kv_cache, layer_index, block_table, end_pos, group_size):
        query = request_q[0, :, 0, :].float()
        kv_head_indices = torch.arange(self.n_query_heads, device=query.device) // group_size
        running_max = torch.full((self.n_query_heads,), float("-inf"), device=query.device)
        denominator = torch.zeros(self.n_query_heads, device=query.device)
        numerator = torch.zeros_like(query)
        num_blocks = (end_pos + kv_cache.block_size - 1) // kv_cache.block_size
        for logical_block in range(num_blocks):
            token_pos = logical_block * kv_cache.block_size
            physical_block, _ = kv_cache.get_token_location(block_table, token_pos)
            num_tokens = min(kv_cache.block_size, end_pos - token_pos)
            keys = kv_cache.key_cache[layer_index, physical_block, :num_tokens].index_select(1, kv_head_indices).float()
            values = kv_cache.value_cache[layer_index, physical_block, :num_tokens].index_select(1, kv_head_indices).float()
            scores = torch.einsum("hd,thd->ht", query, keys) * self.attention_scaling
            block_max = scores.max(dim=-1).values
            new_max = torch.maximum(running_max, block_max)
            old_weight = torch.exp(running_max - new_max)
            weights = torch.exp(scores - new_max[:, None])
            denominator = denominator * old_weight + weights.sum(dim=-1)
            numerator = numerator * old_weight[:, None] + torch.einsum("ht,thd->hd", weights, values)
            running_max = new_max
        return (numerator / denominator[:, None]).to(request_q.dtype)[None, :, None, :]

    def gather_sdpa_read(self, request_q, kv_cache, layer_index, block_table, end_pos, group_size):
        cached_keys, cached_values = kv_cache.get(layer_index, block_table, end_pos)
        if self.grouped_single_decode:
            return self.grouped_sdpa_read(request_q, cached_keys.unsqueeze(0), cached_values.unsqueeze(0), group_size)
        cached_keys = cached_keys.repeat_interleave(group_size, dim=0).unsqueeze(0)
        cached_values = cached_values.repeat_interleave(group_size, dim=0).unsqueeze(0)
        return F.scaled_dot_product_attention(request_q, cached_keys, cached_values, dropout_p=0.0, scale=self.attention_scaling)

    def grouped_sdpa_read(self, query, keys, values, group_size, attention_mask=None):
        batch_size = query.shape[0]
        grouped_query = query.reshape(batch_size, self.n_kv_heads, group_size, query.shape[2], self.head_dim).contiguous()
        if attention_mask is not None:
            attention_mask = attention_mask.expand(batch_size, self.n_kv_heads, group_size, query.shape[2], keys.shape[2]).contiguous()
        output = F.scaled_dot_product_attention(grouped_query, keys.unsqueeze(2), values.unsqueeze(2), attn_mask=attention_mask, dropout_p=0.0, scale=self.attention_scaling)
        return output.reshape(batch_size, self.n_query_heads, query.shape[2], self.head_dim)

    def forward(self, hidden_states, pos_embeddings, causal_mask, kv_cache=None, layer_index=None, request_ids=None, block_tables=None, start_pos=None, input_lens=None, packed=False):
        batch_size, seq_len, _ = hidden_states.shape
        n_qk_heads = self.n_query_heads + self.n_kv_heads
        if self.qkv_proj is None:
            Q = self.q_proj(hidden_states)
            K = self.k_proj(hidden_states)
            V = self.v_proj(hidden_states)
            qk = None
        else:
            qkv = self.qkv_proj(hidden_states)
            qk = qkv[..., :n_qk_heads * self.head_dim].view(batch_size, seq_len, n_qk_heads, self.head_dim)
            V = qkv[..., n_qk_heads * self.head_dim:]
            Q = K = None
        V = V.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        cosine, sine = pos_embeddings[:2]
        if self.joint_qk_rotary and self.q_norm.use_native and self.k_norm.use_native:
            if qk is None:
                qk = torch.cat([Q.view(batch_size, seq_len, self.n_query_heads, self.head_dim), K.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)], dim=2)
            signed_sine = pos_embeddings[2] if len(pos_embeddings) > 2 else torch.cat([-sine[..., :self.head_dim // 2], sine[..., self.head_dim // 2:]], dim=-1)
            # One normalization and one rotation for Q and K together (about half the kernels of separate calls).
            qk = joint_qk_norm_rotary(qk, self.qk_norm_weight(qk), cosine, signed_sine, self.q_norm.eps)
            Q = qk[:, :, :self.n_query_heads].permute(0, 2, 1, 3)
            K = qk[:, :, self.n_query_heads:].permute(0, 2, 1, 3)
        else:
            if qk is not None:
                Q = qk[:, :, :self.n_query_heads]
                K = qk[:, :, self.n_query_heads:]
            Q = Q.view(batch_size, seq_len, self.n_query_heads, self.head_dim).permute(0, 2, 1, 3)
            K = K.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
            Q = self.q_norm(Q)
            K = self.k_norm(K)
            Q, K = apply_rotary_embedding(Q, K, cosine, sine)
        if self.n_query_heads % self.n_kv_heads != 0:
            raise ValueError(f"Number of query heads must be divisible by the number of KV heads: query heads: {self.n_query_heads}, KV heads: {self.n_kv_heads}")
        group_size = self.n_query_heads // self.n_kv_heads
        if kv_cache is None:
            K = torch.repeat_interleave(K, repeats=group_size, dim=1)
            V = torch.repeat_interleave(V, repeats=group_size, dim=1)
            if self.use_sdpa:
                attention_scores = F.scaled_dot_product_attention(Q, K, V, attn_mask=causal_mask, dropout_p=0.0, scale=self.attention_scaling)
            else:
                attention_weights = torch.matmul(Q, K.mT) * self.attention_scaling
                attention_weights += causal_mask
                attention_weights = torch.softmax(attention_weights, dim=-1, dtype=torch.float32).to(Q.dtype)
                attention_scores = torch.matmul(attention_weights, V)
        else:
            if layer_index is None or request_ids is None or block_tables is None or start_pos is None or input_lens is None:
                raise ValueError(f"Cache metadata is required when using the KV cache: layer_index: {layer_index}, request_ids: {request_ids}, block_tables: {block_tables}, start_pos: {start_pos}, input_lens: {input_lens}")
            expected_requests = len(input_lens) if packed else batch_size
            if len(request_ids) != expected_requests or len(block_tables) != expected_requests or len(start_pos) != expected_requests or len(input_lens) != expected_requests:
                raise ValueError(f"Cache metadata must contain one entry per request: batch_size: {expected_requests}, request_ids: {len(request_ids)}, block_tables: {len(block_tables)}, start_pos: {len(start_pos)}, input_lens: {len(input_lens)}")
            batched_decode = not packed and self.batched_decode and self.decode_attention == "gather_sdpa" and batch_size > 1 and seq_len == 1 and all(length == 1 for length in input_lens)
            if packed:
                if batch_size != 1 or any(start_pos) or sum(input_lens) != seq_len:
                    raise ValueError("Packed prefill requires one row of initial prompt segments")
                offset = 0
                for block_table, length in zip(block_tables, input_lens):
                    kv_cache.append(layer_index, block_table, 0, K[0, :, offset:offset + length], V[0, :, offset:offset + length])
                    offset += length
                repeated_keys = K.repeat_interleave(group_size, dim=1)
                repeated_values = V.repeat_interleave(group_size, dim=1)
                if self.use_sdpa:
                    attention_scores = F.scaled_dot_product_attention(Q, repeated_keys, repeated_values, attn_mask=causal_mask, dropout_p=0.0, scale=self.attention_scaling)
                else:
                    attention_weights = torch.matmul(Q, repeated_keys.mT) * self.attention_scaling
                    attention_weights += causal_mask
                    attention_weights = torch.softmax(attention_weights, dim=-1, dtype=torch.float32).to(Q.dtype)
                    attention_scores = torch.matmul(attention_weights, repeated_values)
            elif batched_decode:
                end_positions = [position + 1 for position in start_pos]
                for batch_index in range(batch_size):
                    kv_cache.append(layer_index, block_tables[batch_index], start_pos[batch_index], K[batch_index], V[batch_index])
                if len(set(end_positions)) == 1:
                    cached_keys, cached_values = kv_cache.get_batch(layer_index, block_tables, end_positions[0])
                    attention_mask = None
                else:
                    cached_keys, cached_values, attention_mask = kv_cache.get_batch_ragged(layer_index, block_tables, end_positions)
                attention_scores = self.grouped_sdpa_read(Q, cached_keys, cached_values, group_size, attention_mask)
            else:
                direct_output = batch_size == 1 and input_lens[0] == seq_len
                attention_scores = None if direct_output else torch.zeros_like(Q)
                for batch_index in range(batch_size):
                    request_block_table = block_tables[batch_index]
                    request_start_pos = start_pos[batch_index]
                    request_input_len = input_lens[batch_index]
                    if request_input_len <= 0 or request_input_len > seq_len:
                        raise ValueError(f"Invalid input length for cached attention: {request_input_len}, sequence length: {seq_len}")
                    request_end_pos = request_start_pos + request_input_len
                    request_q = Q[batch_index:batch_index + 1, :, :request_input_len, :]
                    new_keys = K[batch_index, :, :request_input_len, :]
                    new_values = V[batch_index, :, :request_input_len, :]
                    kv_cache.append(layer_index, request_block_table, request_start_pos, new_keys, new_values)
                    if request_input_len == 1:
                        if self.decode_attention == "gather_sdpa":
                            request_attention_scores = self.gather_sdpa_read(request_q, kv_cache, layer_index, request_block_table, request_end_pos, group_size)
                        else:
                            request_attention_scores = self.direct_paged_read(request_q, kv_cache, layer_index, request_block_table, request_end_pos, group_size)
                        if direct_output:
                            attention_scores = request_attention_scores
                        else:
                            attention_scores[batch_index:batch_index + 1, :, :request_input_len, :] = request_attention_scores
                    else:
                        if self.use_sdpa and request_start_pos == 0:
                            # First chunk: the keys are exactly this call's keys, so skip the cache read-back,
                            # the KV-head duplication and the explicit mask (padding rows are sliced away).
                            request_attention_scores = F.scaled_dot_product_attention(request_q, new_keys.unsqueeze(0), new_values.unsqueeze(0), dropout_p=0.0, scale=self.attention_scaling, is_causal=True, enable_gqa=True)
                        else:
                            cached_keys, cached_values = kv_cache.get(layer_index, request_block_table, request_end_pos)
                            cached_keys = cached_keys.unsqueeze(0)
                            cached_values = cached_values.unsqueeze(0)
                            cached_keys = torch.repeat_interleave(cached_keys, repeats=group_size, dim=1)
                            cached_values = torch.repeat_interleave(cached_values, repeats=group_size, dim=1)
                            request_mask = causal_mask[batch_index:batch_index + 1, :, :request_input_len, :request_end_pos]
                            if self.use_sdpa:
                                request_attention_scores = F.scaled_dot_product_attention(request_q, cached_keys, cached_values, attn_mask=request_mask, dropout_p=0.0, scale=self.attention_scaling)
                            else:
                                request_attention_weights = torch.matmul(request_q, cached_keys.mT) * self.attention_scaling
                                request_attention_weights += request_mask
                                request_attention_weights = torch.softmax(request_attention_weights, dim=-1, dtype=torch.float32).to(request_q.dtype)
                                request_attention_scores = torch.matmul(request_attention_weights, cached_values)
                        if direct_output:
                            attention_scores = request_attention_scores
                        else:
                            attention_scores[batch_index:batch_index + 1, :, :request_input_len, :] = request_attention_scores
        attention_scores = attention_scores.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        output = self.o_proj(attention_scores)
        return output
        
