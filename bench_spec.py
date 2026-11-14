"""Speculative decoding vs plain decoding on this engine, batch 1, natural chat prompts (CUDA or any device).

    python bench_spec.py --model Qwen/Qwen3-8B --draft-model Qwen/Qwen3-0.6B --draft-tokens 3,5 --new-tokens 128

For every prompt it runs the target alone, then with each draft length k, and reports decode tokens/s, the draft acceptance
rate and whether the output is token-identical to plain decoding (greedy speculation is mathematically identical, but verifying
k+1 positions in one pass rounds differently from one-at-a-time decoding, so rare bf16 near-tie differences are possible).
"""
import argparse
import math
import time

import torch
from transformers import AutoTokenizer

from bench_prompts import QUESTIONS, chat_token_ids
from utils.engine import Engine
from utils.model_loader import model_loader


def run_once(model, tokenizer, ids, new_tokens, draft, k, device):
    blocks = math.ceil((len(ids) + new_tokens + 64) / 16) + 8
    engine = Engine(model=model, tokenizer=tokenizer, max_new_tokens=new_tokens, max_batch_size=1, max_num_batched_tokens=2048, kv_cache_num_blocks=blocks,
                    kv_cache_block_size=16, stream_text=False, max_completed_requests=0, ignore_eos=True, paged_context_limit=4096, draft_model=draft, num_draft_tokens=k)
    request = engine.add_request("", prompt_token_ids=ids)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    first = None
    while engine.step():
        if first is None and request.generated_token_ids:
            if device == "cuda":
                torch.cuda.synchronize()
            first = time.perf_counter() - start
    if device == "cuda":
        torch.cuda.synchronize()
    total = time.perf_counter() - start
    generated = request.generated_token_ids
    rate = (len(generated) - 1) / max(total - (first or 0.0), 1e-9)
    stats = None
    if engine.speculator is not None and engine.speculator.proposed:
        stats = (engine.speculator.accepted / engine.speculator.proposed, len(generated) / max(engine.speculator.rounds, 1))
    return generated, rate, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--draft-tokens", default="3,5")
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quantize", default=None, choices=(None, "int8"))
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    target = model_loader(args.model, device=torch.device(args.device), dtype=dtype, quantize=args.quantize)
    draft = model_loader(args.draft_model, device=torch.device(args.device), dtype=dtype, quantize=args.quantize)
    ks = [int(value) for value in args.draft_tokens.split(",")]
    prompts = [chat_token_ids(tokenizer, question) for question in QUESTIONS]
    for variant in [(None, 0)] + [(draft, k) for k in ks]:  # warm every code path once
        run_once(target, tokenizer, prompts[0], 24, variant[0], max(variant[1], 1), args.device)
    plain = [run_once(target, tokenizer, ids, args.new_tokens, None, 1, args.device) for ids in prompts]
    print(f"target {args.model} | draft {args.draft_model} | {args.new_tokens} new tokens, batch 1, {len(prompts)} chat prompts")
    plain_rate = len(prompts) * (args.new_tokens - 1) / sum((args.new_tokens - 1) / rate for _, rate, _ in plain)
    print(f"plain decoding: {plain_rate:.1f} tok/s (harmonic mean over prompts)")
    for k in ks:
        rows = [run_once(target, tokenizer, ids, args.new_tokens, draft, k, args.device) for ids in prompts]
        rate = len(prompts) * (args.new_tokens - 1) / sum((args.new_tokens - 1) / r for _, r, _ in rows)
        accepted = sum(stats[0] for _, _, stats in rows if stats) / max(1, sum(1 for _, _, stats in rows if stats))
        per_round = sum(stats[1] for _, _, stats in rows if stats) / max(1, sum(1 for _, _, stats in rows if stats))
        identical = sum(1 for (tokens, _, _), (plain_tokens, _, _) in zip(rows, plain) if tokens == plain_tokens)
        print(f"speculative k={k}: {rate:.1f} tok/s = {rate / plain_rate:.2f}x plain | acceptance {accepted:.2f} | {per_round:.2f} tokens per round | identical to plain {identical}/{len(prompts)}")
        for question, (_, row_rate, stats), (_, base_rate, _) in zip(QUESTIONS, rows, plain):
            print(f"    {question[:48]:<48} plain {base_rate:6.1f}  spec {row_rate:6.1f}  accept {stats[0] if stats else float('nan'):.2f}")


if __name__ == "__main__":
    main()
