"""Explain a greedy-output mismatch between this engine and Hugging Face: is it a bf16 near-tie or a real bug?

Replays the benchmark's batch-1 prompt, finds the first token where the engine's output differs from HF's, and prints the
top logits at that position from HF in bf16 and in float32 (the float32 gap is the ground truth). Then asks the engine
what it picks after the same prefix with a single prefill (no decode path) and by decoding up to that token.

    python agreement_check.py --model Qwen/Qwen3-1.7B --prompt-tokens 128 --output-tokens 128 [--tag tn-0]

Reading the result: if the two tokens' float32 logits are within about 0.1 of each other, the mismatch is a rounding
near-tie and either answer is right. If the gap is large and the engine picked the lower one, it is a bug.
"""
import argparse
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark_cuda import build_prompts
from utils.engine import Engine
from utils.model_loader import model_loader


def hf_generate(model, prompt, count, device):
    ids = torch.tensor([prompt], device=device)
    with torch.inference_mode():
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=count, min_new_tokens=count, do_sample=False,
                             temperature=None, top_p=None, top_k=None, pad_token_id=0)
    return out[0, len(prompt):].tolist()


def next_token_logits(model, tokens, device):
    with torch.inference_mode():
        return model(input_ids=torch.tensor([tokens], device=device)).logits[0, -1].float().cpu()


def engine_run(model, tokenizer, prompt, count, device):
    engine = Engine(model, tokenizer, max_new_tokens=count, max_batch_size=1, max_num_batched_tokens=2048,
                    kv_cache_num_blocks=math.ceil((len(prompt) + count) / 16) + 8, kv_cache_block_size=16, stream_text=False,
                    max_completed_requests=0, ignore_eos=True, paged_context_limit=8192)
    request = engine.add_request(None, prompt_token_ids=prompt)
    while engine.step():
        pass
    return request.generated_token_ids


def describe(logits, tokenizer, a, b):
    top = torch.topk(logits, 4)
    shown = ", ".join(f"{i}({tokenizer.decode([i])!r})={v:.3f}" for v, i in zip(top.values.tolist(), top.indices.tolist()))
    return f"top4: {shown} | logit[{a}]-logit[{b}] = {(logits[a] - logits[b]).item():+.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--tag", default="tn-0")
    args = parser.parse_args()
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt = build_prompts(tokenizer, args.prompt_tokens, 1, args.tag)[0]

    hf = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device).eval()
    reference = hf_generate(hf, prompt, args.output_tokens, device)
    custom_model = model_loader(args.model, device=torch.device(device), dtype=torch.bfloat16, quantize=None)
    mine = engine_run(custom_model, tokenizer, prompt, args.output_tokens, device)
    first = next((i for i, (a, b) in enumerate(zip(reference, mine)) if a != b), None)
    print(f"HF vs engine: {'identical' if first is None else f'first difference at generated token {first}'}")
    if first is None:
        return
    hf_token, my_token = reference[first], mine[first]
    prefix = prompt + reference[:first]
    print(f"HF chose {hf_token} ({tokenizer.decode([hf_token])!r}); engine chose {my_token} ({tokenizer.decode([my_token])!r})")
    print("HF bf16   :", describe(next_token_logits(hf, prefix, device), tokenizer, hf_token, my_token))
    hf = hf.float()
    print("HF float32:", describe(next_token_logits(hf, prefix, device), tokenizer, hf_token, my_token))
    del hf
    torch.cuda.empty_cache()
    prefill_only = engine_run(custom_model, tokenizer, prefix, 1, device)[0]
    print(f"engine, single prefill of the same prefix, picks {prefill_only} ({tokenizer.decode([prefill_only])!r})  [{'matches HF' if prefill_only == hf_token else 'matches the engine decode'}]")


if __name__ == "__main__":
    main()
