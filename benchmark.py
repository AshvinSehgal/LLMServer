"""Compare Hugging Face with the custom engine on one prompt or a workload matrix.

Example: python benchmark.py --matrix --repeats 3 --matrix-block-sizes 16,32 --matrix-stream-modes off,on
"""
import argparse
import gc
import itertools
import math
import statistics
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from utils.attention import SelfAttention
from utils.engine import Engine
from utils.layers import quantize_linear_layers
from utils.model_loader import model_loader

CUSTOM_DTYPES = {"source": None, "fp32": torch.float32, "fp16": torch.float16}


def load_reference_model(model_name, device, dtype=None, quantize=None):
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    if dtype is not None:
        model.to(dtype=dtype)
    if quantize is not None:
        # Same weights and kernel as the custom engine, so the comparison isolates engine overhead.
        quantize_linear_layers(model, quantize)
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
    return model


def reference_quantize_for(args):
    setting = getattr(args, "reference_quantize", "match")
    return getattr(args, "quantize", None) if setting == "match" else (None if setting == "none" else setting)


def reference_dtype_for(args):
    setting = getattr(args, "reference_dtype", "match")
    if setting == "match":
        setting = getattr(args, "custom_dtype", "source")
    return CUSTOM_DTYPES[setting]


def actual_model_dtype(model):
    return next(model.parameters()).dtype if hasattr(model, "parameters") else "unknown"


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class FirstTokenTimer(StoppingCriteria):
    def __init__(self, device, generation_start):
        self.device = device
        self.generation_start = generation_start
        self.first_token_seconds = None

    def __call__(self, input_ids, scores, **kwargs):
        if self.first_token_seconds is None:
            synchronize(self.device)
            self.first_token_seconds = time.perf_counter() - self.generation_start
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


def print_stats(prompt, response, device, model_name, prompt_count, generated_count, load_seconds, generation_seconds, first_token_seconds):
    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")
    print(f"\nDevice: {device} | Model: {model_name}")
    print(f"Prompt tokens: {prompt_count} | Generated tokens: {generated_count}")
    print(f"Model load: {load_seconds:.2f} s | Generation: {generation_seconds:.2f} s")
    print(f"Time to first token: {first_token_seconds:.2f} s")
    print(f"Generation throughput: {generated_count / generation_seconds:.2f} tokens/s")
    if generated_count > 1:
        print(f"Decode throughput: {(generated_count - 1) / (generation_seconds - first_token_seconds):.2f} tokens/s")


def print_repeated_summary(label, results):
    generation_times = [result[1] for result in results]
    first_token_times = [result[2] for result in results]
    decode_speeds = [(len(result[0]) - 1) / (result[1] - result[2]) for result in results if len(result[0]) > 1]
    print(f"{label} median over {len(results)} runs: Generation: {statistics.median(generation_times):.2f} s | Time to first token: {statistics.median(first_token_times):.2f} s")
    if decode_speeds:
        print(f"{label} median decode throughput: {statistics.median(decode_speeds):.2f} tokens/s")


def parse_positive_values(value, name):
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise ValueError(f"{name} must be comma-separated positive integers: {value}") from error
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"{name} must be comma-separated positive integers: {value}")
    return values


def make_matrix_prompt(tokenizer, target_tokens):
    phrase = "Explain what a KV cache does in one sentence. "
    prompt = phrase
    while len(tokenizer(prompt)["input_ids"]) < target_tokens:
        prompt += phrase
    return prompt


def resolve_eos_token_ids(model, tokenizer):
    eos_token_ids = model.generation_config.eos_token_id
    if eos_token_ids is None:
        eos_token_ids = model.config.eos_token_id
    if eos_token_ids is None:
        eos_token_ids = tokenizer.eos_token_id
    if eos_token_ids is None:
        return set()
    return {eos_token_ids} if isinstance(eos_token_ids, int) else set(eos_token_ids)


def trim_after_eos(token_ids, eos_token_ids):
    return token_ids[:next((index + 1 for index, token_id in enumerate(token_ids) if token_id in eos_token_ids), len(token_ids))]


def first_token_difference(expected, actual):
    for index in range(max(len(expected), len(actual))):
        expected_id = expected[index] if index < len(expected) else None
        actual_id = actual[index] if index < len(actual) else None
        if expected_id != actual_id:
            return index, expected_id, actual_id
    return None


def print_token_diagnostics(reference_runs, custom_runs):
    for label, runs in (("HF", reference_runs), ("custom", custom_runs)):
        for run_index, run in enumerate(runs, start=1):
            for request_index, actual in enumerate(run[0], start=1):
                expected = reference_runs[0][0][request_index - 1]
                difference = first_token_difference(expected, actual)
                if difference is None:
                    detail = "match"
                else:
                    index, expected_id, actual_id = difference
                    detail = f"first difference at generated token {index + 1}: reference={expected_id}, actual={actual_id}"
                print(f"  {label} run {run_index}, request {request_index}: reference count={len(expected)}, generated count={len(actual)}, {detail}", flush=True)


def print_prefill_logit_diagnostics(reference_logits, custom_logits):
    difference = (custom_logits.float() - reference_logits.float()).abs()
    for request_index in range(reference_logits.shape[0]):
        reference_row = reference_logits[request_index].float()
        custom_row = custom_logits[request_index].float()
        reference_top = torch.topk(reference_row, 2)
        custom_top = torch.topk(custom_row, 2)
        print(f"  prefill logits request {request_index + 1}: HF first token={reference_top.indices[0].item()}, custom first token={custom_top.indices[0].item()}, max difference={difference[request_index].max().item():.6g}, HF top-2 margin={(reference_top.values[0] - reference_top.values[1]).item():.6g}, custom top-2 margin={(custom_top.values[0] - custom_top.values[1]).item():.6g}", flush=True)


def summarize_matrix_runs(runs, batch_size):
    first_token = statistics.median(run[2] for run in runs)
    generation_rates = [sum(len(ids) for ids in run[0]) / run[1] for run in runs]
    decode_rates = [(sum(len(ids) for ids in run[0]) - batch_size) / (run[1] - run[2]) for run in runs if sum(len(ids) for ids in run[0]) > batch_size and run[1] > run[2]]
    return first_token, statistics.median(generation_rates), statistics.median(decode_rates) if decode_rates else 0.0


def measure_hugging_face_case(model, input_ids, max_new_tokens, device, eos_token_ids=None):
    synchronize(device)
    generation_start = time.perf_counter()
    timer = FirstTokenTimer(device, generation_start)
    with torch.inference_mode():
        output_ids = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, stopping_criteria=StoppingCriteriaList([timer]))
    synchronize(device)
    generation_seconds = time.perf_counter() - generation_start
    if timer.first_token_seconds is None:
        raise RuntimeError("Hugging Face generation did not report a first token")
    generated_ids = [row[input_ids.shape[1]:].tolist() for row in output_ids]
    if eos_token_ids is not None:
        generated_ids = [trim_after_eos(ids, eos_token_ids) for ids in generated_ids]
    return generated_ids, generation_seconds, timer.first_token_seconds


def measure_custom_case(model, tokenizer, prompt, max_new_tokens, batch_size, block_size, stream_text, device, prefix_mode="cold"):
    prompt_length = len(tokenizer(prompt)["input_ids"])
    num_blocks = (batch_size + 1) * math.ceil((prompt_length + max_new_tokens) / block_size) + batch_size
    engine = Engine(model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens, max_batch_size=batch_size, max_num_batched_tokens=max(128, prompt_length * batch_size), kv_cache_num_blocks=num_blocks, kv_cache_block_size=block_size, stream_text=stream_text)
    if prefix_mode == "warm":
        primer = engine.add_request(prompt)
        primer.max_new_tokens = 1
        while engine.step():
            pass
    elif prefix_mode != "cold":
        raise ValueError(f"Unknown prefix mode: {prefix_mode}")
    requests = [engine.add_request(prompt) for _ in range(batch_size)]
    reused_blocks = tuple(len(request.block_table) for request in requests)
    synchronize(device)
    generation_start = time.perf_counter()
    first_token_seconds = None
    while True:
        more_work = engine.step()
        if first_token_seconds is None and all(request.generated_token_ids for request in requests):
            synchronize(device)
            first_token_seconds = time.perf_counter() - generation_start
        if not more_work:
            break
    synchronize(device)
    generation_seconds = time.perf_counter() - generation_start
    if first_token_seconds is None:
        raise RuntimeError("Custom generation did not report a first token")
    return [request.generated_token_ids.copy() for request in requests], generation_seconds, first_token_seconds, reused_blocks


def measure_custom_mixed_case(model, tokenizer, prompts, max_new_tokens, block_size, prefill_bucketing, device, packed_prefill=False):
    prompt_lengths = [len(tokenizer(prompt)["input_ids"]) for prompt in prompts]
    num_blocks = sum(math.ceil((length + max_new_tokens) / block_size) + 1 for length in prompt_lengths)
    engine = Engine(model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens, max_batch_size=len(prompts), max_num_batched_tokens=max(128, sum(prompt_lengths)), kv_cache_num_blocks=num_blocks, kv_cache_block_size=block_size, stream_text=False, prefill_bucketing=prefill_bucketing, packed_prefill=packed_prefill)
    requests = [engine.add_request(prompt) for prompt in prompts]
    synchronize(device)
    start = time.perf_counter()
    first_token_seconds = None
    while True:
        more_work = engine.step()
        if first_token_seconds is None and all(request.generated_token_ids for request in requests):
            synchronize(device)
            first_token_seconds = time.perf_counter() - start
        if not more_work:
            break
    synchronize(device)
    return [request.generated_token_ids.copy() for request in requests], time.perf_counter() - start, first_token_seconds


def run_mixed_prefill(args, device):
    targets = parse_positive_values(args.mixed_prefill_tokens, "--mixed-prefill-tokens")
    if len(targets) < 2:
        raise ValueError("--mixed-prefill-tokens requires at least two prompt lengths")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [make_matrix_prompt(tokenizer, target) for target in targets]
    actual_lengths = [len(tokenizer(prompt)["input_ids"]) for prompt in prompts]
    reference_model = load_reference_model(args.model, device, reference_dtype_for(args), reference_quantize_for(args))
    print(f"HF weight dtype: {actual_model_dtype(reference_model)} | quantize: {reference_quantize_for(args)}", flush=True)
    eos_token_ids = resolve_eos_token_ids(reference_model, tokenizer)
    expected = []
    for prompt in prompts:
        input_ids = torch.tensor([tokenizer(prompt)["input_ids"]], device=device, dtype=torch.long)
        expected.append(measure_hugging_face_case(reference_model, input_ids, args.max_new_tokens, device, eos_token_ids)[0][0])
    del reference_model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    model = model_loader(args.model, device=device, dtype=CUSTOM_DTYPES[args.custom_dtype], use_sdpa=args.sdpa, decode_attention=None if args.decode_attention == "auto" else args.decode_attention, fuse_projections=args.fuse_projections, quantize=getattr(args, "quantize", None))
    print(f"Custom weight dtype: {actual_model_dtype(model)}", flush=True)
    print("prompt_lengths | prefill | TTFT | gen tok/s | decode tok/s | token match", flush=True)
    for label, bucketing, packed in (("padded", False, False), ("bucketed", True, False), ("packed", False, True)):
        measure_custom_mixed_case(model, tokenizer, prompts, args.max_new_tokens, args.mixed_prefill_block_size, bucketing, device, packed)
        runs = [measure_custom_mixed_case(model, tokenizer, prompts, args.max_new_tokens, args.mixed_prefill_block_size, bucketing, device, packed) for _ in range(args.repeats)]
        ttft, generation, decode = summarize_matrix_runs(runs, len(prompts))
        token_match = all(run[0] == expected for run in runs)
        print(f"{','.join(map(str, actual_lengths)):>14} | {label:>8} | {ttft:.3f}s | {generation:.2f} | {decode:.2f} | {token_match}", flush=True)
        if not token_match:
            print_token_diagnostics([(expected, 0, 0)], runs)


def run_matrix(args, device):
    prompt_targets = parse_positive_values(args.matrix_prompt_tokens, "--matrix-prompt-tokens")
    output_limits = parse_positive_values(args.matrix_output_tokens, "--matrix-output-tokens")
    batch_sizes = parse_positive_values(args.matrix_batch_sizes, "--matrix-batch-sizes")
    block_sizes = parse_positive_values(args.matrix_block_sizes, "--matrix-block-sizes")
    stream_modes = tuple(item.strip() for item in args.matrix_stream_modes.split(","))
    if not stream_modes or any(mode not in ("on", "off") for mode in stream_modes):
        raise ValueError(f"--matrix-stream-modes must contain only on and off: {args.matrix_stream_modes}")
    prefix_modes = tuple(item.strip() for item in args.matrix_prefix_modes.split(","))
    if not prefix_modes or any(mode not in ("cold", "warm") for mode in prefix_modes):
        raise ValueError(f"--matrix-prefix-modes must contain only cold and warm: {args.matrix_prefix_modes}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    cases = tuple(itertools.product(prompt_targets, output_limits, batch_sizes))
    print("Hugging Face Transformers matrix", flush=True)
    reference_model = load_reference_model(args.model, device, reference_dtype_for(args), reference_quantize_for(args))
    print(f"HF weight dtype: {actual_model_dtype(reference_model)} | quantize: {reference_quantize_for(args)}", flush=True)
    eos_token_ids = resolve_eos_token_ids(reference_model, tokenizer)
    prompts = {target: make_matrix_prompt(tokenizer, target) for target in prompt_targets}
    reference_results = {}
    reference_logits = {}
    for target, max_new_tokens, batch_size in cases:
        prompt_ids = tokenizer(prompts[target])["input_ids"]
        input_ids = torch.tensor([prompt_ids] * batch_size, device=device, dtype=torch.long)
        measure_hugging_face_case(reference_model, input_ids, max_new_tokens, device, eos_token_ids)
        runs = [measure_hugging_face_case(reference_model, input_ids, max_new_tokens, device, eos_token_ids) for _ in range(args.repeats)]
        reference_results[(target, max_new_tokens, batch_size)] = runs
        if args.matrix_diagnose_logits and (target, batch_size) not in reference_logits:
            with torch.inference_mode():
                reference_logits[(target, batch_size)] = reference_model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False).logits[:, -1].float().cpu()
        print(f"HF done: prompt={len(prompt_ids)} output_limit={max_new_tokens} batch={batch_size}", flush=True)
    del reference_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    custom_dtype = getattr(args, "custom_dtype", "source")
    fuse_projections = getattr(args, "fuse_projections", True)
    print(f"Custom model matrix (dtype={custom_dtype}, fused={fuse_projections}, quantize={getattr(args, 'quantize', None)})", flush=True)
    model = model_loader(args.model, device=device, dtype=CUSTOM_DTYPES[custom_dtype], use_sdpa=args.sdpa, decode_attention=None if args.decode_attention == "auto" else args.decode_attention, fuse_projections=fuse_projections, quantize=getattr(args, "quantize", None))
    print(f"Custom weight dtype: {actual_model_dtype(model)}", flush=True)
    if device.type == "mps":
        print(f"Custom model MPS allocated after load: {torch.mps.current_allocated_memory() / 2 ** 30:.2f} GiB", flush=True)
    decode_modes = ("direct", "gather_sdpa") if args.compare_decode_attention else (model.layers[0].self_attn.decode_attention,)
    prefill_modes = (False, True) if args.compare_prefill_attention else (args.sdpa,)
    batched_modes = (False, True) if args.compare_batched_decode else (True,)
    print("prompt | output_limit | batch | block | attention | prefill | batch_decode | prefix | reused_blocks | text | HF TTFT | custom TTFT | HF gen tok/s | custom gen tok/s | HF decode tok/s | custom decode tok/s | decode speedup | token match", flush=True)
    for target, max_new_tokens, batch_size in cases:
        reference_runs = reference_results[(target, max_new_tokens, batch_size)]
        reference_ttft, reference_generation, reference_decode = summarize_matrix_runs(reference_runs, batch_size)
        prompt = prompts[target]
        prompt_length = len(tokenizer(prompt)["input_ids"])
        for block_size, decode_mode, prefill_sdpa, batched_decode, prefix_mode, stream_mode in itertools.product(block_sizes, decode_modes, prefill_modes, batched_modes, prefix_modes, stream_modes):
            for module in model.modules():
                if isinstance(module, SelfAttention):
                    module.decode_attention = decode_mode
                    module.use_sdpa = prefill_sdpa
                    module.batched_decode = batched_decode
            stream_text = stream_mode == "on"
            measure_custom_case(model, tokenizer, prompt, max_new_tokens, batch_size, block_size, stream_text, device, prefix_mode)
            runs = [measure_custom_case(model, tokenizer, prompt, max_new_tokens, batch_size, block_size, stream_text, device, prefix_mode) for _ in range(args.repeats)]
            custom_ttft, custom_generation, custom_decode = summarize_matrix_runs(runs, batch_size)
            speedup = custom_decode / reference_decode if reference_decode else 0.0
            token_match = all(run[0] == reference_runs[0][0] for run in reference_runs + runs)
            reused_blocks = ",".join(str(count) for count in runs[0][3])
            prefill_label = "sdpa" if prefill_sdpa else "manual"
            print(f"{prompt_length:>6} | {max_new_tokens:>12} | {batch_size:>5} | {block_size:>5} | {decode_mode:>11} | {prefill_label:>7} | {str(batched_decode):>12} | {prefix_mode:>6} | {reused_blocks:>13} | {stream_mode:>4} | {reference_ttft:>7.3f}s | {custom_ttft:>11.3f}s | {reference_generation:>12.2f} | {custom_generation:>16.2f} | {reference_decode:>15.2f} | {custom_decode:>19.2f} | {speedup:>14.2f}x | {token_match}", flush=True)
            if not token_match or args.matrix_token_details:
                print_token_diagnostics(reference_runs, runs)
            if not token_match and args.matrix_diagnose_logits:
                input_ids = torch.tensor([tokenizer(prompt)["input_ids"]] * batch_size, device=device, dtype=torch.long)
                with torch.inference_mode():
                    custom_logits = model(input_ids)[:, -1].float().cpu()
                print_prefill_logit_diagnostics(reference_logits[(target, batch_size)], custom_logits)


def main():
    parser = argparse.ArgumentParser(description="Benchmark one prompt with Hugging Face Transformers and the custom LLM engine.")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--prompt", default="Explain what a KV cache does in one short sentence.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--sdpa", action=argparse.BooleanOptionalAction, default=True, help="Use scaled dot-product attention for custom-model prefill (default on; --no-sdpa for the manual path)")
    parser.add_argument("--custom-dtype", choices=CUSTOM_DTYPES, default="source", help="Custom model weight and KV-cache dtype; source uses the installed Transformers loader's dtype")
    parser.add_argument("--reference-dtype", choices=("match", *CUSTOM_DTYPES), default="match", help="Hugging Face weight dtype; match uses --custom-dtype for a fair comparison")
    parser.add_argument("--fuse-projections", action=argparse.BooleanOptionalAction, default=True, help="Combine Q/K/V and gate/up projections in the custom model (default on; --no-fuse-projections to disable)")
    parser.add_argument("--reference-quantize", choices=("match", "none", "int8", "int4"), default="match", help="Hugging Face reference quantization; match applies the same int8 layers as --quantize so the comparison is like-for-like")
    parser.add_argument("--quantize", choices=("int8", "int4"), default=None, help="Weight-only int8 quantization for the custom model (lossy; the Hugging Face reference stays unquantized)")
    parser.add_argument("--repeats", type=int, default=1, help="Measure multiple fresh requests per loaded model")
    parser.add_argument("--decode-attention", choices=("auto", "direct", "gather_sdpa"), default="auto", help="Decode attention path for the custom model")
    parser.add_argument("--compare-decode-attention", action="store_true", help="Measure both decode paths on one loaded custom model")
    parser.add_argument("--matrix", action="store_true", help="Compare prompt lengths, output lengths, batch sizes, block sizes, and streaming modes")
    parser.add_argument("--mixed-prefill-tokens", help="Compare bucketing off/on for simultaneous prompts of different token lengths, e.g. 16,128")
    parser.add_argument("--mixed-prefill-block-size", type=int, default=16)
    parser.add_argument("--matrix-prompt-tokens", default="16,128", help="Approximate prompt token counts for --matrix")
    parser.add_argument("--matrix-output-tokens", default="32,64", help="Maximum generated tokens for --matrix")
    parser.add_argument("--matrix-batch-sizes", default="1,2", help="Concurrent requests for --matrix")
    parser.add_argument("--matrix-block-sizes", default="16", help="KV block sizes for --matrix")
    parser.add_argument("--matrix-stream-modes", default="off", help="Text decoding modes for --matrix: off, on, or off,on")
    parser.add_argument("--matrix-prefix-modes", default="cold", help="Prefix cache modes for --matrix: cold, warm, or cold,warm")
    parser.add_argument("--compare-prefill-attention", action="store_true", help="Compare manual and SDPA prefill attention in --matrix")
    parser.add_argument("--compare-batched-decode", action="store_true", help="Compare per-request and batched equal-length decode in --matrix")
    parser.add_argument("--matrix-token-details", action="store_true", help="Print token counts and first differences for every matrix request and repeat")
    parser.add_argument("--matrix-diagnose-logits", action="store_true", help="On token mismatch, compare Hugging Face and custom first-token prefill logits")
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.mixed_prefill_block_size <= 0:
        parser.error("--mixed-prefill-block-size must be positive")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.mixed_prefill_tokens:
        try:
            run_mixed_prefill(args, device)
        except ValueError as error:
            parser.error(str(error))
        return
    if args.matrix:
        try:
            run_matrix(args, device)
        except ValueError as error:
            parser.error(str(error))
        return

    print("Hugging Face Transformers", flush=True)
    print("Loading tokenizer and model...", flush=True)
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    reference_model = load_reference_model(args.model, device, reference_dtype_for(args), reference_quantize_for(args))
    print(f"HF weight dtype: {actual_model_dtype(reference_model)} | quantize: {reference_quantize_for(args)}", flush=True)
    synchronize(device)
    load_seconds = time.perf_counter() - load_start
    prompt_token_ids = tokenizer(args.prompt)["input_ids"]
    if not args.prompt.strip():
        bos_token_id = next((token_id for token_id in (reference_model.generation_config.bos_token_id, reference_model.config.bos_token_id, tokenizer.bos_token_id) if token_id is not None), None)
        if bos_token_id is None:
            raise ValueError("Model bos_token_id not found for empty prompt")
        prompt_token_ids = [bos_token_id]
    input_ids = torch.tensor([prompt_token_ids], device=device, dtype=torch.long)

    print("Warming up...", flush=True)
    with torch.inference_mode():
        reference_model.generate(input_ids=input_ids, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
    synchronize(device)

    print(f"Loaded in {load_seconds:.2f} s on {device}. Generating...", flush=True)
    reference_results = []
    for _ in range(args.repeats):
        synchronize(device)
        generation_start = time.perf_counter()
        timer = FirstTokenTimer(device, generation_start)
        with torch.inference_mode():
            output_ids = reference_model.generate(input_ids=input_ids, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True, stopping_criteria=StoppingCriteriaList([timer]))
        synchronize(device)
        generation_seconds = time.perf_counter() - generation_start
        if timer.first_token_seconds is None:
            raise RuntimeError("Hugging Face generation did not report a first token")
        generated_ids = output_ids[0, input_ids.shape[1]:].tolist()
        reference_results.append((generated_ids, generation_seconds, timer.first_token_seconds))
    response = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print_stats(args.prompt, response, device, args.model, len(prompt_token_ids), len(generated_ids), load_seconds, generation_seconds, timer.first_token_seconds)
    if args.repeats > 1:
        print_repeated_summary("Hugging Face", reference_results)

    del reference_model
    del output_ids
    del input_ids
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    print("\nCustom model", flush=True)
    print("Loading tokenizer and model...", flush=True)
    load_start = time.perf_counter()
    model = model_loader(args.model, device=device, dtype=CUSTOM_DTYPES[args.custom_dtype], use_sdpa=args.sdpa, decode_attention=None if args.decode_attention == "auto" else args.decode_attention, fuse_projections=args.fuse_projections, quantize=getattr(args, "quantize", None))
    print(f"Custom weight dtype: {actual_model_dtype(model)}", flush=True)
    synchronize(device)
    load_seconds = time.perf_counter() - load_start
    if device.type == "mps":
        print(f"Custom model MPS allocated after load: {torch.mps.current_allocated_memory() / 2 ** 30:.2f} GiB", flush=True)
    prompt_length = len(prompt_token_ids)
    block_size = 16
    num_blocks = math.ceil((prompt_length + args.max_new_tokens) / block_size) + 1
    decode_modes = ("direct", "gather_sdpa") if args.compare_decode_attention else (model.layers[0].self_attn.decode_attention,)
    for decode_mode in decode_modes:
        for module in model.modules():
            if isinstance(module, SelfAttention):
                module.decode_attention = decode_mode
        label = f"Custom model ({decode_mode})" if args.compare_decode_attention else "Custom model"
        warmup_engine = Engine(model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens, max_batch_size=1, max_num_batched_tokens=128, kv_cache_num_blocks=num_blocks, kv_cache_block_size=block_size)
        print(f"Warming up {label}...", flush=True)
        warmup_engine.add_request(args.prompt)
        while warmup_engine.step():
            pass
        synchronize(device)
        del warmup_engine
        print(f"Loaded in {load_seconds:.2f} s on {device}. Generating {label}...", flush=True)
        custom_results = []
        for _ in range(args.repeats):
            engine = Engine(model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens, max_batch_size=1, max_num_batched_tokens=128, kv_cache_num_blocks=num_blocks, kv_cache_block_size=block_size)
            request = engine.add_request(args.prompt)
            synchronize(device)
            generation_start = time.perf_counter()
            first_token_seconds = None
            while True:
                more_work = engine.step()
                if first_token_seconds is None and request.generated_token_ids:
                    synchronize(device)
                    first_token_seconds = time.perf_counter() - generation_start
                if not more_work:
                    break
            generation_seconds = time.perf_counter() - generation_start
            custom_results.append((request.generated_token_ids.copy(), generation_seconds, first_token_seconds))
        generated_count = len(request.generated_token_ids)
        response = tokenizer.decode(request.generated_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print_stats(args.prompt, response, device, args.model, len(request.prompt_token_ids), generated_count, load_seconds, generation_seconds, first_token_seconds)
        if args.repeats > 1:
            print_repeated_summary(label, custom_results)
        print(f"{label} greedy token IDs match Hugging Face: {all(result[0] == reference_results[0][0] for result in reference_results + custom_results)}")


if __name__ == "__main__":
    main()
