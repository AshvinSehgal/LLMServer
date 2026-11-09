import time
import torch
import pytest
from transformers import GPT2Config, GPT2LMHeadModel, StoppingCriteriaList
from utils.attention import SelfAttention
from benchmark import FirstTokenTimer, print_stats, print_repeated_summary, parse_positive_values, make_matrix_prompt, summarize_matrix_runs, resolve_eos_token_ids, trim_after_eos, first_token_difference, print_token_diagnostics, print_prefill_logit_diagnostics, measure_custom_case, measure_custom_mixed_case, reference_dtype_for, run_matrix, run_mixed_prefill

def test_first_token_timer_records_hugging_face_generation():
    model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_positions=16, n_embd=16, n_layer=1, n_head=2, eos_token_id=31, pad_token_id=0)).eval()
    start = time.perf_counter()
    timer = FirstTokenTimer(torch.device("cpu"), start)
    with torch.inference_mode():
        output = model.generate(input_ids=torch.tensor([[1, 2]]), max_new_tokens=2, do_sample=False, stopping_criteria=StoppingCriteriaList([timer]))
    assert output.shape[1] > 2
    assert timer.first_token_seconds is not None
    assert 0 <= timer.first_token_seconds <= time.perf_counter() - start

def test_benchmark_prints_existing_stats(capsys):
    print_stats("Hello", "World", torch.device("cpu"), "test-model", 2, 3, 1.5, 0.5, 0.2)
    assert capsys.readouterr().out == "\nPrompt: Hello\nResponse: World\n\nDevice: cpu | Model: test-model\nPrompt tokens: 2 | Generated tokens: 3\nModel load: 1.50 s | Generation: 0.50 s\nTime to first token: 0.20 s\nGeneration throughput: 6.00 tokens/s\nDecode throughput: 6.67 tokens/s\n"

def test_benchmark_prints_repeated_medians(capsys):
    print_repeated_summary("Custom model", [([1, 2, 3], 0.5, 0.1), ([1, 2, 3], 0.7, 0.2), ([1, 2, 3], 0.6, 0.2)])
    assert capsys.readouterr().out == "Custom model median over 3 runs: Generation: 0.60 s | Time to first token: 0.20 s\nCustom model median decode throughput: 5.00 tokens/s\n"

def test_benchmark_matrix_helpers():
    class Tokenizer:
        def __call__(self, prompt):
            return {"input_ids": prompt.split()}
    assert parse_positive_values("1, 2", "values") == (1, 2)
    with pytest.raises(ValueError):
        parse_positive_values("1,0", "values")
    assert len(Tokenizer()(make_matrix_prompt(Tokenizer(), 20))["input_ids"]) >= 20
    first_token, generation_rate, decode_rate = summarize_matrix_runs([([[1, 2, 3]], 0.5, 0.1), ([[1, 2, 3]], 0.7, 0.2)], 1)
    assert first_token == pytest.approx(0.15)
    assert generation_rate == pytest.approx((6 + 3 / 0.7) / 2)
    assert decode_rate == pytest.approx(4.5)
    from types import SimpleNamespace
    assert reference_dtype_for(SimpleNamespace(custom_dtype="fp16", reference_dtype="match")) == torch.float16
    assert reference_dtype_for(SimpleNamespace(custom_dtype="fp16", reference_dtype="fp32")) == torch.float32

def test_benchmark_trims_padding_after_eos():
    from types import SimpleNamespace
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[2, 3]), config=SimpleNamespace(eos_token_id=4))
    tokenizer = SimpleNamespace(eos_token_id=5)
    assert resolve_eos_token_ids(model, tokenizer) == {2, 3}
    assert trim_after_eos([10, 3, 0, 0], {2, 3}) == [10, 3]
    assert trim_after_eos([10, 11], {2, 3}) == [10, 11]
    assert first_token_difference([10, 3], [10, 3]) is None
    assert first_token_difference([10, 3], [10, 11]) == (1, 3, 11)
    assert first_token_difference([10, 3], [10]) == (1, 3, None)

def test_benchmark_reports_first_difference_by_run_and_request(capsys):
    reference_runs = [([[10, 2], [20, 2]], 1.0, 0.1), ([[10, 2], [20, 3]], 1.0, 0.1)]
    custom_runs = [([[10, 2], [20]], 1.0, 0.1)]
    print_token_diagnostics(reference_runs, custom_runs)
    output = capsys.readouterr().out
    assert "HF run 2, request 2: reference count=2, generated count=2, first difference at generated token 2: reference=2, actual=3" in output
    assert "custom run 1, request 2: reference count=2, generated count=1, first difference at generated token 2: reference=2, actual=None" in output

def test_benchmark_reports_prefill_logit_argmax_and_margin(capsys):
    print_prefill_logit_diagnostics(torch.tensor([[0.0, 2.0, 1.0]]), torch.tensor([[0.0, 1.0, 2.0]]))
    output = capsys.readouterr().out
    assert "HF first token=1, custom first token=2" in output
    assert "max difference=1" in output

def test_benchmark_matrix_reports_cases(monkeypatch, capsys):
    from types import SimpleNamespace
    import benchmark
    class Tokenizer:
        def __call__(self, prompt):
            return {"input_ids": list(range(len(prompt.split())))}
    class ReferenceModel:
        generation_config = SimpleNamespace(eos_token_id=2)
        config = SimpleNamespace(eos_token_id=2)
        def to(self, device=None, dtype=None):
            return self
        def eval(self):
            return self
    class LoaderModel:
        def __init__(self):
            self.attention = SelfAttention(hidden_size=8, n_query_heads=4, n_kv_heads=2, head_dim=2, bias=False, attention_scaling=2 ** -0.5, rms_norm_eps=1e-6, decode_attention="gather_sdpa")
            self.layers = [SimpleNamespace(self_attn=self.attention)]
        def modules(self):
            return [self.attention]
    loaded_options = []
    def load_model(*args, **kwargs):
        loaded_options.append((kwargs["dtype"], kwargs["fuse_projections"]))
        return LoaderModel()
    monkeypatch.setattr(benchmark.AutoTokenizer, "from_pretrained", lambda name: Tokenizer())
    monkeypatch.setattr(benchmark.AutoModelForCausalLM, "from_pretrained", lambda name: ReferenceModel())
    monkeypatch.setattr(benchmark, "model_loader", load_model)
    monkeypatch.setattr(benchmark, "measure_hugging_face_case", lambda *args: ([[1, 2, 3]], 0.5, 0.1))
    monkeypatch.setattr(benchmark, "measure_custom_case", lambda *args: ([[1, 2, 3]], 0.4, 0.1, (0,)))
    args = SimpleNamespace(model="test", sdpa=False, custom_dtype="fp16", fuse_projections=True, decode_attention="auto", compare_decode_attention=False, compare_prefill_attention=False, compare_batched_decode=False, repeats=2, matrix_prompt_tokens="8", matrix_output_tokens="3", matrix_batch_sizes="1", matrix_block_sizes="4,8", matrix_stream_modes="off,on", matrix_prefix_modes="cold", matrix_token_details=False, matrix_diagnose_logits=False)
    run_matrix(args, torch.device("cpu"))
    output = capsys.readouterr().out
    assert loaded_options == [(torch.float16, True)]
    assert output.count("gather_sdpa") == 4
    assert output.count("| True\n") == 4

def test_benchmark_matrix_compares_prefill_decode_and_prefix_modes(monkeypatch, capsys):
    from types import SimpleNamespace
    import benchmark
    class Tokenizer:
        def __call__(self, prompt):
            return {"input_ids": list(range(len(prompt.split())))}
    class ReferenceModel:
        generation_config = SimpleNamespace(eos_token_id=2)
        config = SimpleNamespace(eos_token_id=2)
        def to(self, device=None, dtype=None):
            return self
        def eval(self):
            return self
    class LoaderModel:
        def __init__(self):
            self.attention = SelfAttention(hidden_size=8, n_query_heads=4, n_kv_heads=2, head_dim=2, bias=False, attention_scaling=2 ** -0.5, rms_norm_eps=1e-6, decode_attention="gather_sdpa")
            self.layers = [SimpleNamespace(self_attn=self.attention)]
        def modules(self):
            return [self.attention]
    observed = []
    def measure_custom(model, tokenizer, prompt, max_new_tokens, batch_size, block_size, stream_text, device, prefix_mode):
        observed.append((model.attention.use_sdpa, model.attention.batched_decode, prefix_mode))
        return [[1, 2, 3], [1, 2, 3]], 0.4, 0.1, (1, 1) if prefix_mode == "warm" else (0, 0)
    monkeypatch.setattr(benchmark.AutoTokenizer, "from_pretrained", lambda name: Tokenizer())
    monkeypatch.setattr(benchmark.AutoModelForCausalLM, "from_pretrained", lambda name: ReferenceModel())
    monkeypatch.setattr(benchmark, "model_loader", lambda *args, **kwargs: LoaderModel())
    monkeypatch.setattr(benchmark, "measure_hugging_face_case", lambda *args: ([[1, 2, 3], [1, 2, 3]], 0.5, 0.1))
    monkeypatch.setattr(benchmark, "measure_custom_case", measure_custom)
    args = SimpleNamespace(model="test", sdpa=False, decode_attention="auto", compare_decode_attention=False, compare_prefill_attention=True, compare_batched_decode=True, repeats=1, matrix_prompt_tokens="8", matrix_output_tokens="3", matrix_batch_sizes="2", matrix_block_sizes="4", matrix_stream_modes="off", matrix_prefix_modes="cold,warm", matrix_token_details=False, matrix_diagnose_logits=False)
    run_matrix(args, torch.device("cpu"))
    assert len(set(observed)) == 8
    assert capsys.readouterr().out.count("| True\n") == 8

@pytest.mark.parametrize("device", ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []))
def test_benchmark_warm_prefix_reuses_blocks_and_preserves_tokens(device):
    from types import SimpleNamespace
    from utils.model import Model
    class Tokenizer:
        bos_token_id = 1
        pad_token_id = 0
        eos_token_id = 31
        def __call__(self, prompt):
            return {"input_ids": [1, 2, 3, 4, 5]}
    torch.manual_seed(42)
    model = Model(vocab_size=32, hidden_size=8, intermediate_size=16, n_hidden_layers=2, head_dim=2, rope_theta=10000, n_query_heads=4, n_kv_heads=2, bias=False, attention_scaling=2 ** -0.5, rms_norm_eps=1e-6, decode_attention="gather_sdpa").to(device).eval()
    model.config = SimpleNamespace(num_hidden_layers=2, num_key_value_heads=2, num_attention_heads=4, hidden_size=8, head_dim=2, bos_token_id=1, pad_token_id=0, eos_token_id=31)
    model.generation_config = SimpleNamespace(bos_token_id=1, pad_token_id=0, eos_token_id=31)
    model.name = "tiny"
    cold = measure_custom_case(model, Tokenizer(), "prompt", 2, 1, 2, False, torch.device(device), "cold")
    warm = measure_custom_case(model, Tokenizer(), "prompt", 2, 1, 2, False, torch.device(device), "warm")
    assert cold[0] == warm[0]
    assert cold[3] == (0,)
    assert warm[3] == (2,)

@pytest.mark.parametrize("device", ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []))
def test_benchmark_mixed_prefill_bucketing_preserves_tokens(device):
    from types import SimpleNamespace
    from utils.model import Model
    class Tokenizer:
        bos_token_id = 1
        pad_token_id = 0
        eos_token_id = 31
        def __call__(self, prompt):
            return {"input_ids": [1, 2] if prompt == "short" else list(range(1, 14))}
    torch.manual_seed(42)
    model = Model(vocab_size=32, hidden_size=8, intermediate_size=16, n_hidden_layers=2, head_dim=2, rope_theta=10000, n_query_heads=4, n_kv_heads=2, bias=False, attention_scaling=2 ** -0.5, rms_norm_eps=1e-6, decode_attention="gather_sdpa").to(device).eval()
    model.config = SimpleNamespace(num_hidden_layers=2, num_key_value_heads=2, num_attention_heads=4, hidden_size=8, head_dim=2, bos_token_id=1, pad_token_id=0, eos_token_id=31)
    model.generation_config = SimpleNamespace(bos_token_id=1, pad_token_id=0, eos_token_id=31)
    model.name = "tiny"
    plain = measure_custom_mixed_case(model, Tokenizer(), ["short", "long"], 2, 4, False, torch.device(device))
    bucketed = measure_custom_mixed_case(model, Tokenizer(), ["short", "long"], 2, 4, True, torch.device(device))
    packed = measure_custom_mixed_case(model, Tokenizer(), ["short", "long"], 2, 4, False, torch.device(device), packed_prefill=True)
    assert plain[0] == bucketed[0] == packed[0]
    assert all(len(tokens) == 2 for tokens in plain[0])
