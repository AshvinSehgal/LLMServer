"""Paged decode path: pure-PyTorch checks that run anywhere, plus Triton/CUDA-graph checks that only run on CUDA."""
import copy
from unittest.mock import patch
import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen3Config
from utils.engine import Engine
from utils.kv_cache import PagedKVCache
from utils.model_loader import model_loader
from utils import paged_attention as pa

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
triton_only = pytest.mark.skipif(not (torch.cuda.is_available() and pa.TRITON_AVAILABLE), reason="CUDA with Triton required")


def random_cache(num_blocks, block_size, n_kv_heads, head_dim, batch, lengths, seed=0, dtype=torch.float32, device="cpu"):
    """Random K/V cache plus shuffled, non-contiguous block tables for the given context lengths."""
    generator = torch.Generator().manual_seed(seed)
    keys = torch.randn(num_blocks, block_size, n_kv_heads, head_dim, generator=generator).to(device, dtype)
    values = torch.randn(num_blocks, block_size, n_kv_heads, head_dim, generator=generator).to(device, dtype)
    max_blocks = max(-(-length // block_size) for length in lengths)
    order = torch.randperm(num_blocks, generator=generator).tolist()
    tables = torch.zeros(batch, max_blocks, dtype=torch.int32)
    cursor = 0
    for row, length in enumerate(lengths):
        needed = -(-length // block_size)
        tables[row, :needed] = torch.tensor(order[cursor:cursor + needed], dtype=torch.int32)
        cursor += needed
    return keys, values, tables.to(device), torch.tensor(lengths, dtype=torch.int32, device=device)


def dense_reference(query, keys, values, tables, lengths, scale):
    """Independent per-sequence softmax attention, one sequence at a time."""
    outputs = []
    group = query.shape[1] // keys.shape[2]
    for row, length in enumerate(lengths.tolist()):
        blocks = tables[row, :-(-length // keys.shape[1])].long()
        k = keys[blocks].reshape(-1, keys.shape[2], keys.shape[3])[:length].float()
        v = values[blocks].reshape(-1, values.shape[2], values.shape[3])[:length].float()
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        scores = torch.einsum("hd,thd->ht", query[row].float(), k) * scale
        outputs.append(torch.einsum("ht,thd->hd", torch.softmax(scores, dim=-1), v))
    return torch.stack(outputs)


@pytest.mark.parametrize("n_query_heads,n_kv_heads", [(4, 4), (4, 2), (8, 2), (16, 8)])
@pytest.mark.parametrize("block_size", [4, 16, 5])
def test_reference_attention_matches_dense_attention(n_query_heads, n_kv_heads, block_size):
    lengths = [1, 3, 17, 33]
    keys, values, tables, seq_lens = random_cache(40, block_size, n_kv_heads, 16, len(lengths), lengths)
    query = torch.randn(len(lengths), n_query_heads, 16)
    actual = pa.paged_decode_attention_reference(query, keys, values, tables, seq_lens, 0.25)
    torch.testing.assert_close(actual, dense_reference(query, keys, values, tables, seq_lens, 0.25), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("num_splits,tokens_per_split,tile", [(1, 64, 8), (4, 8, 4), (8, 4, 4), (8, 16, 16), (4, 5, 2)])
def test_split_algorithm_matches_reference(num_splits, tokens_per_split, tile):
    """The Triton kernels' partial/merge algorithm, including empty splits and partial tiles, equals plain attention."""
    lengths = [1, 4, 9, 20, 31]
    keys, values, tables, seq_lens = random_cache(30, 8, 2, 16, len(lengths), lengths, seed=3)
    query = torch.randn(len(lengths), 4, 16)
    expected = pa.paged_decode_attention_reference(query, keys, values, tables, seq_lens, 0.3)
    coverage = num_splits * tokens_per_split
    if max(lengths) > coverage:
        lengths_ok = [min(length, coverage) for length in lengths]
        seq_lens = torch.tensor(lengths_ok, dtype=torch.int32)
        expected = pa.paged_decode_attention_reference(query, keys, values, tables, seq_lens, 0.3)
    actual = pa.paged_decode_attention_split_reference(query, keys, values, tables, seq_lens, 0.3, num_splits, tokens_per_split, tile)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_default_split_sizing_covers_the_cache():
    for num_blocks, block_size, splits, tile in [(500, 500, 32, 64), (13, 16, 32, 16), (1, 1, 32, 64)]:
        per_split = pa.default_tokens_per_split(num_blocks, block_size, splits, tile)
        assert per_split % tile == 0 and per_split * splits >= num_blocks * block_size


def test_dispatcher_uses_reference_off_cuda():
    keys, values, tables, seq_lens = random_cache(8, 4, 2, 8, 2, [5, 9])
    query = torch.randn(2, 4, 8)
    torch.testing.assert_close(pa.paged_decode_attention(query, keys, values, tables, seq_lens, 0.5), pa.paged_decode_attention_reference(query, keys, values, tables, seq_lens, 0.5))


def test_write_slots_matches_append():
    cache_a = PagedKVCache(2, 6, 4, 2, 8, torch.device("cpu"), torch.float32)
    cache_b = PagedKVCache(2, 6, 4, 2, 8, torch.device("cpu"), torch.float32)
    table_a, table_b = [], []
    keys = torch.randn(2, 6, 8)
    values = torch.randn(2, 6, 8)
    for position in range(6):
        cache_a.append(1, table_a, position, keys[:, position:position + 1], values[:, position:position + 1])
        slot = cache_b.slot_for(table_b, position)
        cache_b.write_slots(1, torch.tensor([slot]), keys[None, :, position], values[None, :, position])
    assert table_a == table_b
    assert torch.equal(cache_a.key_cache, cache_b.key_cache) and torch.equal(cache_a.value_cache, cache_b.value_cache)


class FakeTokenizer:
    bos_token_id = 1
    pad_token_id = 0
    eos_token_id = None

    def __init__(self, prompts):
        self.prompts = prompts

    def __call__(self, prompt):
        return {"input_ids": self.prompts[prompt]}


def tiny_model(seed=0, layers=2):
    torch.manual_seed(seed)
    config = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2, head_dim=8, rope_theta=10000, rms_norm_eps=1e-6)
    reference = AutoModelForCausalLM.from_config(config).eval()
    reference.generation_config.bos_token_id = 1
    reference.generation_config.pad_token_id = 0
    reference.generation_config.eos_token_id = None
    with patch("utils.model_loader.AutoModelForCausalLM.from_pretrained", return_value=reference), patch("utils.model_loader.CheckpointReader.open", return_value=None):
        return model_loader("tiny-qwen", device=torch.device("cpu"), dtype=torch.float32)


def generate(model, prompts, max_new_tokens, **engine_options):
    engine = Engine(model, FakeTokenizer(prompts), max_new_tokens=max_new_tokens, max_batch_size=8, max_num_batched_tokens=64, kv_cache_num_blocks=64, kv_cache_block_size=4, stream_text=False, **engine_options)
    requests = [engine.add_request(prompt) for prompt in prompts]
    while engine.step():
        pass
    return [request.generated_token_ids for request in requests]


def test_paged_forward_matches_cached_forward_logits():
    model = tiny_model()
    cache = PagedKVCache(2, 16, 4, 2, 8, torch.device("cpu"), torch.float32)
    prompt = torch.tensor([[5, 9, 3, 7, 11]])
    table = []
    with torch.inference_mode():
        model(prompt, torch.arange(5).unsqueeze(0), None, kv_cache_manager=cache, request_ids=["a"], block_tables=[table], start_pos=[0], input_lens=[5])
        cache_gather = copy.deepcopy(cache)
        table_gather = list(table)
        expected = model(torch.tensor([[42]]), torch.tensor([[5]]), None, kv_cache_manager=cache_gather, request_ids=["a"], block_tables=[table_gather], start_pos=[5], input_lens=[1], logit_rows=[0], logit_positions=torch.zeros(1, dtype=torch.long))
        slot = cache.slot_for(table, 5)
        tables = torch.zeros(1, 4, dtype=torch.int32)
        tables[0, :len(table)] = torch.tensor(table, dtype=torch.int32)
        actual = model.forward_decode_paged(torch.tensor([[42]]), torch.tensor([[5]]), torch.tensor([slot]), tables, torch.tensor([6], dtype=torch.int32), cache)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(cache.key_cache, cache_gather.key_cache, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("prompts", [
    {"a": [5, 9, 3, 7, 11]},
    {"a": [5, 9, 3], "b": [12, 4, 8, 20, 21, 22, 23, 24, 25], "c": [30]},
])
def test_engine_paged_decode_generates_the_same_tokens(prompts):
    model = tiny_model()
    expected = generate(model, prompts, 12, paged_decode=False)
    actual = generate(model, prompts, 12, paged_decode=True)
    assert actual == expected
    assert all(len(tokens) == 12 for tokens in actual)


def test_engine_falls_back_when_context_exceeds_paged_limit():
    model = tiny_model()
    prompts = {"a": [5, 9, 3, 7, 11, 13, 14, 15]}
    expected = generate(model, prompts, 10, paged_decode=False)
    # A limit of 8 tokens is exceeded after the first decode steps, so later steps must use the gather path.
    assert generate(model, prompts, 10, paged_decode=True, paged_context_limit=8) == expected


def test_paged_decode_is_off_by_default_without_cuda():
    engine = Engine(tiny_model(), FakeTokenizer({"a": [5]}), max_new_tokens=2, max_batch_size=2, max_num_batched_tokens=16, kv_cache_num_blocks=8, kv_cache_block_size=4, stream_text=False)
    assert engine.paged_decode is False and engine.graph_runner is None


@triton_only
@pytest.mark.parametrize("n_query_heads,n_kv_heads,head_dim", [(16, 8, 128), (8, 8, 64), (32, 8, 128), (4, 1, 128)])
@pytest.mark.parametrize("block_size", [16, 32, 500])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_kernel_matches_reference(n_query_heads, n_kv_heads, head_dim, block_size, dtype):
    lengths = [1, 17, 300, 1025, 2049]
    num_blocks = sum(-(-length // block_size) for length in lengths) + 3
    keys, values, tables, seq_lens = random_cache(num_blocks, block_size, n_kv_heads, head_dim, len(lengths), lengths, seed=1, dtype=dtype, device="cuda")
    query = torch.randn(len(lengths), n_query_heads, head_dim, device="cuda", dtype=dtype)
    scale = head_dim ** -0.5
    expected = dense_reference(query, keys, values, tables, seq_lens, scale)
    for variant in ("dot", "broadcast"):
        for num_splits in (1, 4, 32):
            actual = pa.paged_decode_attention_triton(query, keys, values, tables, seq_lens, scale, num_splits=num_splits, variant=variant)
            torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)


@triton_only
def test_triton_kernel_with_fixed_split_size_and_idle_splits():
    lengths = [3, 700]
    keys, values, tables, seq_lens = random_cache(80, 16, 8, 128, 2, lengths, dtype=torch.float16, device="cuda")
    query = torch.randn(2, 16, 128, device="cuda", dtype=torch.float16)
    expected = dense_reference(query, keys, values, tables, seq_lens, 128 ** -0.5)
    actual = pa.paged_decode_attention_triton(query, keys, values, tables, seq_lens, 128 ** -0.5, num_splits=32, tokens_per_split=64)
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)


@cuda_only
@pytest.mark.parametrize("batch", [1, 3])
def test_cuda_graph_engine_generates_the_same_tokens(batch):
    model = tiny_model().to("cuda")
    prompts = {name: ids for name, ids in zip("abcd", ([5, 9, 3, 7, 11], [12, 4, 8], [30, 31, 32, 33, 34, 35, 36], [60]))}
    prompts = dict(list(prompts.items())[:batch])

    def run(**options):
        engine = Engine(model, FakeTokenizer(prompts), max_new_tokens=16, max_batch_size=8, max_num_batched_tokens=64, kv_cache_num_blocks=64, kv_cache_block_size=16, stream_text=False, **options)
        requests = [engine.add_request(prompt) for prompt in prompts]
        while engine.step():
            pass
        return [request.generated_token_ids for request in requests], engine

    expected, _ = run(paged_decode=False)
    eager, _ = run(paged_decode=True, cuda_graphs=False)
    graphed, engine = run(paged_decode=True, cuda_graphs=True)
    assert eager == expected
    assert graphed == expected
    assert engine.graph_runner is not None and engine.graph_runner.graphs


def test_engine_accepts_token_ids_and_can_ignore_eos():
    model = tiny_model()
    tokenizer = FakeTokenizer({})
    plain = Engine(model, tokenizer, max_new_tokens=6, max_batch_size=2, max_num_batched_tokens=32, kv_cache_num_blocks=16, kv_cache_block_size=4, stream_text=False)
    request = plain.add_request(None, prompt_token_ids=[5, 9, 3])
    while plain.step():
        pass
    assert len(request.generated_token_ids) == 6 and request.prompt_token_ids == [5, 9, 3]
    first = request.generated_token_ids[1]
    stopping = Engine(model, tokenizer, max_new_tokens=6, max_batch_size=2, max_num_batched_tokens=32, kv_cache_num_blocks=16, kv_cache_block_size=4, stream_text=False)
    stopping.eos_token_ids = {first}
    stopped = stopping.add_request(None, prompt_token_ids=[5, 9, 3])
    while stopping.step():
        pass
    assert len(stopped.generated_token_ids) == 2
    ignoring = Engine(model, tokenizer, max_new_tokens=6, max_batch_size=2, max_num_batched_tokens=32, kv_cache_num_blocks=16, kv_cache_block_size=4, stream_text=False, ignore_eos=True)
    ignoring.eos_token_ids = {first}
    ignored = ignoring.add_request(None, prompt_token_ids=[5, 9, 3])
    while ignoring.step():
        pass
    assert ignored.generated_token_ids == request.generated_token_ids
    with pytest.raises(ValueError):
        plain.add_request(None, prompt_token_ids=[])
