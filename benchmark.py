"""Compare Hugging Face generation with the custom engine on one prompt."""
import argparse
import gc
import math
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from utils.engine import Engine
from utils.model_loader import model_loader


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


def main():
    parser = argparse.ArgumentParser(description="Benchmark one prompt with Hugging Face Transformers and the custom LLM engine.")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--prompt", default="Explain what a KV cache does in one short sentence.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--sdpa", action="store_true", help="Enable scaled dot-product attention for the custom model")
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Hugging Face Transformers", flush=True)
    print("Loading tokenizer and model...", flush=True)
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    reference_model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
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
    response = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print_stats(args.prompt, response, device, args.model, len(prompt_token_ids), len(generated_ids), load_seconds, generation_seconds, timer.first_token_seconds)

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
    model = model_loader(args.model, device=device, use_sdpa=args.sdpa)
    synchronize(device)
    load_seconds = time.perf_counter() - load_start
    prompt_length = len(prompt_token_ids)
    block_size = 16
    num_blocks = math.ceil((prompt_length + args.max_new_tokens) / block_size) + 1
    warmup_engine = Engine(model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens, max_batch_size=1, max_num_batched_tokens=128, kv_cache_num_blocks=num_blocks, kv_cache_block_size=block_size)
    print("Warming up...", flush=True)
    warmup_engine.add_request(args.prompt)
    while warmup_engine.step():
        pass
    synchronize(device)
    del warmup_engine
    engine = Engine(model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens, max_batch_size=1, max_num_batched_tokens=128, kv_cache_num_blocks=num_blocks, kv_cache_block_size=block_size)
    request = engine.add_request(args.prompt)

    print(f"Loaded in {load_seconds:.2f} s on {device}. Generating...", flush=True)
    synchronize(device)
    generation_start = time.perf_counter()
    first_token_seconds = None
    while True:
        more_work = engine.step()
        synchronize(device)
        if first_token_seconds is None and request.generated_token_ids:
            first_token_seconds = time.perf_counter() - generation_start
        if not more_work:
            break
    generation_seconds = time.perf_counter() - generation_start
    generated_count = len(request.generated_token_ids)
    response = tokenizer.decode(request.generated_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    print_stats(args.prompt, response, device, args.model, len(request.prompt_token_ids), generated_count, load_seconds, generation_seconds, first_token_seconds)


if __name__ == "__main__":
    main()
