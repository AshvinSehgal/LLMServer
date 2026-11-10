import json
from benchmark_cuda import build_prompts, compare_tokens, format_report, parse_ints


class Tokenizer:
    """Whitespace tokenizer: one token per word, ids are word hashes."""
    def __call__(self, text):
        return {"input_ids": [sum(map(ord, word)) + len(word) for word in text.split()]}


def test_prompts_have_exact_length_and_unique_first_block():
    prompts = build_prompts(Tokenizer(), 100, 5, "x")
    assert [len(prompt) for prompt in prompts] == [100] * 5
    assert len({tuple(prompt[:4]) for prompt in prompts}) == 5
    assert build_prompts(Tokenizer(), 100, 5, "x") == prompts  # deterministic, so every backend sees the same input
    assert build_prompts(Tokenizer(), 100, 5, "y") != prompts


def test_parse_ints_and_token_agreement():
    assert parse_ints("1, 8,32") == [1, 8, 32]
    assert compare_tokens([[1, 2], [3, 4]], [[1, 2], [3, 5]]) == 0.5
    assert compare_tokens(None, [[1]]) is None


def result(backend, ttft, decode, error=None, tokens=None):
    entry = {"prompt_tokens": 128, "batch": 8, "new_tokens": 16, "ttft_s": ttft, "decode_tok_s": decode, "e2e_tok_s": decode / 2, "tokens": tokens or [[1, 2]]}
    return {"backend": backend, "env": {"gpu": "A100", "torch": "2.x"}, "results": [entry] if not error else [], "error": error}


def test_report_handles_missing_backend_and_computes_speedups():
    report = format_report([result("hf", 0.4, 100.0), result("vllm", 0.2, 400.0, tokens=[[1, 3]]), result("custom", 0.1, 800.0), result("sglang", 0, 0, error="boom")])
    assert "custom vs hf" in report and "8.00x" in report      # decode 800 / 100
    assert "2.00x" in report                                    # decode 800 / 400 vs vLLM
    assert "4.00x" in report                                    # ttft 0.4 / 0.1
    assert "1.00" in report and "0.00" in report                # token agreement for custom and vllm
    only_custom = format_report([result("custom", 0.1, 800.0), result("vllm", 0, 0, error="No module named 'vllm'")])
    assert "NOTE: backend vllm failed" in only_custom and "n/a" in only_custom
    json.dumps(result("hf", 1, 1))
