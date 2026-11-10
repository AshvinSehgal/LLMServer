import copy

import pytest

from test_paged_decode import FakeTokenizer, tiny_model
from utils.engine import Engine
from utils.request import RequestFinishReason


def run(model, prompts, max_new_tokens, **options):
    engine = Engine(model, FakeTokenizer(prompts), max_new_tokens=max_new_tokens, max_batch_size=4, max_num_batched_tokens=64, kv_cache_num_blocks=64, kv_cache_block_size=4, stream_text=False, **options)
    requests = [engine.add_request(prompt) for prompt in prompts]
    while engine.step():
        pass
    return engine, requests


PROMPTS = {"a": [5, 9, 3, 7, 11, 13, 14, 15, 16]}


@pytest.mark.parametrize("draft_seed", [0, 1])  # 0 = identical draft (every proposal accepted), 1 = unrelated draft (mostly rejected)
@pytest.mark.parametrize("k", [1, 3, 5])
@pytest.mark.parametrize("max_new_tokens", [1, 2, 7, 20])
def test_speculative_output_equals_plain_greedy_decoding(draft_seed, k, max_new_tokens):
    target = tiny_model(seed=0)
    draft = tiny_model(seed=draft_seed)
    _, expected = run(target, PROMPTS, max_new_tokens)
    engine, actual = run(target, PROMPTS, max_new_tokens, draft_model=draft, num_draft_tokens=k)
    assert actual[0].generated_token_ids == expected[0].generated_token_ids
    assert len(actual[0].generated_token_ids) == max_new_tokens
    assert actual[0].finish_reason == RequestFinishReason.LENGTH


def test_identical_draft_accepts_every_proposal():
    target = tiny_model(seed=0)
    engine, requests = run(target, PROMPTS, 24, draft_model=tiny_model(seed=0), num_draft_tokens=3)
    assert engine.speculator.rounds > 0
    assert engine.speculator.accepted == engine.speculator.proposed


def test_speculation_releases_draft_blocks_when_requests_finish():
    engine, _ = run(tiny_model(seed=0), PROMPTS, 12, draft_model=tiny_model(seed=1), num_draft_tokens=3)
    assert engine.speculator.tables == {} and engine.speculator.draft_cache.num_free_blocks == engine.speculator.draft_cache.num_blocks


def test_batches_with_several_requests_still_match_plain_decoding():
    prompts = {"a": [5, 9, 3, 7, 11], "b": [12, 4, 8, 20, 21, 22, 23], "c": [30, 31]}
    _, expected = run(tiny_model(seed=0), prompts, 10)
    engine, actual = run(tiny_model(seed=0), prompts, 10, draft_model=tiny_model(seed=1), num_draft_tokens=3)
    assert [r.generated_token_ids for r in actual] == [r.generated_token_ids for r in expected]


def test_vocabulary_mismatch_is_rejected():
    from transformers import AutoModelForCausalLM, Qwen3Config
    from unittest.mock import patch
    from utils.model_loader import model_loader
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    reference = AutoModelForCausalLM.from_config(config).eval()
    reference.generation_config.pad_token_id = 0
    import torch
    with patch("utils.model_loader.AutoModelForCausalLM.from_pretrained", return_value=reference), patch("utils.model_loader.CheckpointReader.open", return_value=None):
        small_vocab = model_loader("tiny", device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(ValueError, match="vocabulary"):
        run(tiny_model(seed=0), PROMPTS, 4, draft_model=small_vocab)
