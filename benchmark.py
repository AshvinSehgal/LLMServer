"""Run one prompt through the complete engine and measure generation speed."""

import argparse
import math
import time

import torch
from transformers import AutoTokenizer

from utils.engine import Engine
from utils.model_loader import model_loader


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def main():
    parser = argparse.ArgumentParser(description="Benchmark one prompt with the custom LLM engine.")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--prompt", default="Explain what a KV cache does in one short sentence.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")

    print("Loading tokenizer and model...", flush=True)
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = model_loader(args.model)
    device = next(model.parameters()).device
    synchronize(device)
    load_seconds = time.perf_counter() - load_start

    prompt_length = len(tokenizer(args.prompt)["input_ids"]) if args.prompt.strip() else 1
    block_size = 16
    num_blocks = math.ceil((prompt_length + args.max_new_tokens) / block_size) + 1
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

    print(f"\nPrompt: {args.prompt}")
    print(f"Response: {response}")
    print(f"\nDevice: {device} | Model: {args.model}")
    print(f"Prompt tokens: {len(request.prompt_token_ids)} | Generated tokens: {generated_count}")
    print(f"Model load: {load_seconds:.2f} s | Generation: {generation_seconds:.2f} s")
    print(f"Time to first token: {first_token_seconds:.2f} s")
    print(f"Generation throughput: {generated_count / generation_seconds:.2f} tokens/s")
    if generated_count > 1:
        print(f"Decode throughput: {(generated_count - 1) / (generation_seconds - first_token_seconds):.2f} tokens/s")


if __name__ == "__main__":
    main()
