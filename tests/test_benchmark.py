import time
import torch
from transformers import GPT2Config, GPT2LMHeadModel, StoppingCriteriaList
from benchmark import FirstTokenTimer, print_stats, print_repeated_summary

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
