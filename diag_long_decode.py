"""Where does decode time go for long contexts at small batch? Times every engine step and says which path it took.

    python diag_long_decode.py --model Qwen/Qwen3-8B --prompt-tokens 16384 --batch 4

Reports, per variant, the prefill time, the per-step decode time (median, 90th percentile, max) split by rows decoding, how many
decode steps fell off the paged fast path, and the attention slicing the engine chose. Variants change the attention kernel or
the slice length so a slow cell can be attributed to the kernel, the slicing or the scheduler.
"""
import argparse
import math
import os
import statistics
import time

import torch
from transformers import AutoTokenizer

from benchmark_cuda import build_prompts
from utils.engine import Engine
from utils.model_loader import model_loader


def run_variant(model, tokenizer, args, label, kernel=None, tokens_per_split=None, tag="x"):
    if kernel:
        os.environ["LLMSERVER_ATTENTION_KERNEL"] = kernel
    else:
        os.environ.pop("LLMSERVER_ATTENTION_KERNEL", None)
    prompts = build_prompts(tokenizer, args.prompt_tokens, args.batch, f"diag-{label}-{tag}")
    longest = args.prompt_tokens + args.new_tokens
    blocks = args.batch * math.ceil(longest / 16) + 8
    engine = Engine(model, tokenizer, max_new_tokens=args.new_tokens, max_batch_size=args.batch, max_num_batched_tokens=args.max_num_batched_tokens, kv_cache_num_blocks=blocks,
                    kv_cache_block_size=16, stream_text=False, max_completed_requests=0, ignore_eos=True, paged_context_limit=max(8192, longest))
    if tokens_per_split:
        engine.kv_cache.paged_tokens_per_split = tokens_per_split
        engine.kv_cache.paged_num_splits = min(64, -(-longest // tokens_per_split))
    fallbacks = []
    original = engine.run_paged_decode
    def timed(requests):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = original(requests)
        torch.cuda.synchronize()
        timed.log.append((len(requests), time.perf_counter() - start, result is None))
        return result
    timed.log = []
    engine.run_paged_decode = timed
    requests = [engine.add_request(None, prompt_token_ids=prompt) for prompt in prompts]
    torch.cuda.synchronize()
    begin = time.perf_counter()
    steps = []
    while engine.step():
        torch.cuda.synchronize()
        steps.append(time.perf_counter() - begin)
    total = steps[-1]
    first_decode = None
    print(f"{label}: total {total:.2f} s | slices {engine.kv_cache.paged_num_splits} x {engine.kv_cache.paged_tokens_per_split} tokens | paged limit {engine.paged_context_limit} | graph {engine.graph_runner is not None}")
    by_rows = {}
    for rows, seconds, fell in timed.log:
        by_rows.setdefault(rows, []).append((seconds, fell))
    for rows, entries in sorted(by_rows.items()):
        times = sorted(seconds * 1000 for seconds, _ in entries)
        print(f"    {len(entries):4d} decode calls with {rows} rows: median {statistics.median(times):6.2f} ms  p90 {times[int(0.9 * (len(times) - 1))]:6.2f}  max {times[-1]:6.2f} | off the paged path: {sum(fell for _, fell in entries)}")
    del engine
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--prompt-tokens", type=int, default=16384)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = model_loader(args.model, device=torch.device("cuda"), dtype=torch.bfloat16, quantize=None)
    print(f"{args.model} | {args.batch} requests x {args.prompt_tokens}-token prompts, {args.new_tokens} new tokens, max_num_batched_tokens {args.max_num_batched_tokens}")
    for repeat in range(2):  # the first pass includes graph capture and kernel compilation
        run_variant(model, tokenizer, args, f"default (pass {repeat})", tag=repeat)
    run_variant(model, tokenizer, args, "256-token slices", tokens_per_split=256, tag="a")
    run_variant(model, tokenizer, args, "512-token slices", tokens_per_split=512, tag="b")
    run_variant(model, tokenizer, args, "broadcast kernel", kernel="broadcast", tag="c")


if __name__ == "__main__":
    main()
