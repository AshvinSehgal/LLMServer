import pytest
import torch
import torch.nn.functional as F

from test_paged_decode import FakeTokenizer, tiny_model
from utils import prefix_attention
from utils.engine import Engine
from utils.prefix_attention import attention_with_lse, chunk_attention, merge_attention


def full_masked(q, keys, values, group_size, start):
    """Reference: the chunk's queries (positions start..) over keys 0..start+L with one bottom-right-aligned causal mask."""
    length, context = q.shape[2], keys.shape[1]
    mask = torch.zeros(length, context).masked_fill(torch.arange(context).view(1, -1) > (start + torch.arange(length)).view(-1, 1), float("-inf"))
    expand = lambda tensor: tensor.repeat_interleave(group_size, dim=0).unsqueeze(0)
    return F.scaled_dot_product_attention(q, expand(keys), expand(values), attn_mask=mask.to(q.dtype), scale=q.shape[-1] ** -0.5)


@pytest.mark.parametrize("group_size", [1, 4])
@pytest.mark.parametrize("start,length", [(1, 1), (5, 3), (64, 16), (100, 37)])
def test_split_chunk_attention_equals_one_masked_call(group_size, start, length):
    torch.manual_seed(0)
    kv_heads, dim = 2, 16
    q = torch.randn(1, kv_heads * group_size, length, dim)
    keys, values = torch.randn(kv_heads, start + length, dim), torch.randn(kv_heads, start + length, dim)
    expected = full_masked(q, keys, values, group_size, start)
    actual = chunk_attention(q, keys[:, start:], values[:, start:], keys[:, :start], values[:, :start], group_size, dim ** -0.5)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_merge_of_two_halves_is_softmax_over_the_union():
    torch.manual_seed(1)
    q, k, v = torch.randn(1, 2, 4, 8), torch.randn(1, 2, 10, 8), torch.randn(1, 2, 10, 8)
    a = attention_with_lse(q, k[:, :, :6], v[:, :, :6], 8 ** -0.5, causal=False)
    b = attention_with_lse(q, k[:, :, 6:], v[:, :, 6:], 8 ** -0.5, causal=False)
    assert torch.allclose(merge_attention(a[0], a[1], b[0], b[1]), F.scaled_dot_product_attention(q, k, v), atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("group_size", [1, 4])
@pytest.mark.parametrize("start,length", [(4096, 512), (16384, 4096), (1000, 33)])
def test_flash_split_matches_masked_attention_on_cuda(dtype, group_size, start, length):
    torch.manual_seed(0)
    kv_heads, dim = 4, 128
    q = torch.randn(1, kv_heads * group_size, length, dim, device="cuda", dtype=dtype)
    keys, values = (torch.randn(kv_heads, start + length, dim, device="cuda", dtype=dtype) for _ in range(2))
    expand = lambda tensor: tensor.repeat_interleave(group_size, dim=0).unsqueeze(0)
    mask = torch.zeros(length, start + length, device="cuda", dtype=dtype).masked_fill(
        torch.arange(start + length, device="cuda").view(1, -1) > (start + torch.arange(length, device="cuda")).view(-1, 1), float("-inf"))
    expected = F.scaled_dot_product_attention(q, expand(keys), expand(values), attn_mask=mask, scale=dim ** -0.5)
    actual = chunk_attention(q, keys[:, start:], values[:, start:], keys[:, :start], values[:, :start], group_size, dim ** -0.5)
    assert not prefix_attention._FLASH_BROKEN
    assert (actual.float() - expected.float()).abs().max().item() < 0.02


def test_chunked_prefill_through_the_split_path_matches_unchunked(monkeypatch):
    monkeypatch.setattr(prefix_attention, "MIN_CONTEXT_FOR_SPLIT", 0)
    prompt = list(range(3, 30))
    calls = []
    original = prefix_attention.chunk_attention
    monkeypatch.setattr(prefix_attention, "chunk_attention", lambda *args: calls.append(1) or original(*args))
    outputs = []
    for chunk in (None, 5):
        engine = Engine(tiny_model(seed=0), FakeTokenizer({"a": prompt}), max_new_tokens=6, max_batch_size=1, max_num_batched_tokens=64, prefill_chunk_size=chunk,
                        kv_cache_num_blocks=32, kv_cache_block_size=4, stream_text=False)
        request = engine.add_request(None, prompt_token_ids=prompt)
        while engine.step():
            pass
        outputs.append(request.generated_token_ids)
    assert calls, "the chunked run never used the split path"
    assert outputs[0] == outputs[1]


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9, reason="Hopper required")
@pytest.mark.parametrize("group_size", [1, 4])
@pytest.mark.parametrize("length", [2048, 5000])
def test_cudnn_causal_attention_matches_flash_on_cuda(group_size, length):
    torch.manual_seed(0)
    kv_heads, dim = 4, 128
    q = torch.randn(1, kv_heads * group_size, length, dim, device="cuda", dtype=torch.bfloat16)
    keys, values = (torch.randn(kv_heads, length, dim, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    expected = F.scaled_dot_product_attention(q, keys.unsqueeze(0), values.unsqueeze(0), is_causal=True, enable_gqa=True, scale=dim ** -0.5)
    actual = prefix_attention.causal_attention(q, keys, values, group_size, dim ** -0.5)
    assert not prefix_attention._CUDNN_BROKEN
    assert (actual.float() - expected.float()).abs().max().item() < 0.02


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9, reason="Hopper required")
@pytest.mark.parametrize("causal", [False, True])
def test_cudnn_log_sum_exp_matches_the_reference_on_cuda(causal, monkeypatch):
    monkeypatch.setattr(prefix_attention, "USE_CUDNN_LSE", True)
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 4, 3000, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    output, lse = attention_with_lse(q, k, v, 128 ** -0.5, causal)
    assert not prefix_attention._CUDNN_LSE_BROKEN
    scores = torch.matmul(q.float(), k.float().mT) * 128 ** -0.5
    if causal:
        scores = scores.masked_fill(torch.ones(3000, 3000, dtype=torch.bool, device="cuda").triu(1), float("-inf"))
    assert (lse - torch.logsumexp(scores, dim=-1)).abs().max().item() < 0.02
    assert (output.float() - F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=128 ** -0.5).float()).abs().max().item() < 0.02


def test_warmup_does_nothing_without_cuda():
    if torch.cuda.is_available():
        pytest.skip("CPU behaviour")
    assert prefix_attention.warmup(torch.device("cpu"), torch.float32, 4, 16, 8192, 0.25) == 0


def test_engine_run_records_a_step_profile():
    import asyncio
    prompt = list(range(3, 20))
    engine = Engine(tiny_model(seed=0), FakeTokenizer({"a": prompt}), max_new_tokens=5, max_batch_size=2, max_num_batched_tokens=64,
                    kv_cache_num_blocks=32, kv_cache_block_size=4, stream_text=False)
    engine.add_request(None, prompt_token_ids=prompt)
    engine.add_request(None, prompt_token_ids=prompt[:9])
    asyncio.run(engine.run())
    profile = engine.step_profile
    assert profile["steps"] > 0 and profile["decode_steps"] > 0 and profile["prefill_steps"] > 0
    assert sum(count for count, _ in profile["decode_by_rows"].values()) == profile["decode_steps"]
