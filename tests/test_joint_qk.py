import pytest
import torch
from utils.attention import SelfAttention
from utils.kv_cache import PagedKVCache
from utils.layers import RotaryEmbedding


def build_attention(fused, seed=0):
    torch.manual_seed(seed)
    attention = SelfAttention(hidden_size=32, n_query_heads=4, n_kv_heads=2, head_dim=8, bias=False, attention_scaling=8 ** -0.5, rms_norm_eps=1e-6, use_sdpa=True, decode_attention="gather_sdpa").eval()
    for norm in (attention.q_norm, attention.k_norm):
        torch.nn.init.normal_(norm.weight, mean=1.0, std=0.3)
    if fused:
        attention.fuse_projections()
    return attention


def embeddings(batch, seq_len, with_signed):
    rotary = RotaryEmbedding(8, 10000.0)
    hidden = torch.zeros(batch, seq_len, 32)
    cosine, sine = rotary(hidden, torch.arange(seq_len).unsqueeze(0).expand(batch, -1))
    if with_signed:
        return cosine, sine, torch.cat([-sine[..., :4], sine[..., 4:]], dim=-1)
    return cosine, sine


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("with_signed", [False, True])
@pytest.mark.parametrize("batch,seq_len", [(1, 1), (3, 1), (1, 9), (2, 9)])
def test_joint_qk_norm_rotary_matches_separate_path(fused, with_signed, batch, seq_len):
    attention = build_attention(fused)
    hidden = torch.randn(batch, seq_len, 32)
    mask = torch.full((batch, 1, seq_len, seq_len), torch.finfo(torch.float32).min).triu(diagonal=1)
    with torch.inference_mode():
        attention.joint_qk_rotary = True
        joint = attention(hidden, embeddings(batch, seq_len, with_signed), mask)
        attention.joint_qk_rotary = False
        separate = attention(hidden, embeddings(batch, seq_len, False), mask)
    torch.testing.assert_close(joint, separate, rtol=1e-5, atol=1e-5)


def test_joint_qk_norm_weight_cache_follows_weight_changes():
    attention = build_attention(False)
    reference = torch.zeros(1, 1, 6, 8)
    first = attention.qk_norm_weight(reference).clone()
    assert attention.qk_norm_weight(reference) is attention.qk_norm_weight(reference)
    attention.q_norm.weight = torch.nn.Parameter(torch.full((8,), 2.0))
    assert not torch.equal(attention.qk_norm_weight(reference), first)
    assert torch.equal(attention.qk_norm_weight(reference)[:4], torch.full((4, 8), 2.0))


def test_first_chunk_sdpa_prefill_matches_cached_read_path():
    attention = build_attention(True)
    cache = PagedKVCache(num_layers=1, num_blocks=8, block_size=4, num_kv_heads=2, head_dim=8, device=torch.device("cpu"), dtype=torch.float32)
    hidden = torch.randn(1, 6, 32)
    mask = torch.full((1, 1, 6, 6), torch.finfo(torch.float32).min).triu(diagonal=1)
    with torch.inference_mode():
        fast = attention(hidden, embeddings(1, 6, True), mask, kv_cache=cache, layer_index=0, request_ids=["a"], block_tables=[[]], start_pos=[0], input_lens=[6])
        attention.use_sdpa = False
        cache2 = PagedKVCache(num_layers=1, num_blocks=8, block_size=4, num_kv_heads=2, head_dim=8, device=torch.device("cpu"), dtype=torch.float32)
        slow = attention(hidden, embeddings(1, 6, True), mask, kv_cache=cache2, layer_index=0, request_ids=["a"], block_tables=[[]], start_pos=[0], input_lens=[6])
    torch.testing.assert_close(fast, slow, rtol=1e-4, atol=1e-5)
