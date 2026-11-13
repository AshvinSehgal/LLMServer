"""Are greedy-output mismatches against Hugging Face rounding near-ties or errors?

For each batch size, generates greedily with HF (bf16) and with this engine from the benchmark's own prompts, finds the first
token where each differing request diverges, and re-scores that position with a float32 HF model (the ground truth). The
float32 logit gap between the two candidate tokens says how close the call was, and which side float32 prefers says who was
closer to right. Near-ties (small gap) are bf16 rounding noise; a large gap with the engine on the wrong side is a bug.

    python agreement_check.py --model Qwen/Qwen3-1.7B --prompt-tokens 128 --output-tokens 128 --batches 1,8,32
"""
import argparse
import math
import statistics

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark_cuda import build_prompts
from utils.engine import Engine
from utils.model_loader import model_loader


def hf_generate(model, prompts, count, device):
    ids = torch.tensor(prompts, device=device)
    with torch.inference_mode():
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=count, min_new_tokens=count, do_sample=False,
                             temperature=None, top_p=None, top_k=None, pad_token_id=0)
    return out[:, ids.shape[1]:].tolist()


def engine_generate(model, tokenizer, prompts, count):
    longest = max(len(prompt) for prompt in prompts) + count
    engine = Engine(model, tokenizer, max_new_tokens=count, max_batch_size=len(prompts), max_num_batched_tokens=16384,
                    kv_cache_num_blocks=len(prompts) * math.ceil(longest / 16) + 8, kv_cache_block_size=16, stream_text=False,
                    max_completed_requests=0, ignore_eos=True, paged_context_limit=max(8192, longest))
    requests = [engine.add_request(None, prompt_token_ids=prompt) for prompt in prompts]
    while engine.step():
        pass
    return [request.generated_token_ids for request in requests]


def next_token_logits(model, tokens, device):
    with torch.inference_mode():
        return model(input_ids=torch.tensor([tokens], device=device)).logits[0, -1].float().cpu()


def analyse(label, prompts, reference, other, truth, tokenizer, device):
    """Score each first divergence of `other` against `reference` with the float32 model."""
    wrong, gaps = 0, []
    differing = 0
    for prompt, expected, actual in zip(prompts, reference, other):
        first = next((i for i, (a, b) in enumerate(zip(expected, actual)) if a != b), None)
        if first is None:
            continue
        differing += 1
        logits = next_token_logits(truth, prompt + expected[:first], device)
        gap = (logits[expected[first]] - logits[actual[first]]).item()  # > 0: float32 prefers the reference's token
        gaps.append(abs(gap))
        wrong += gap > 0
    if not differing:
        print(f"  {label}: identical to HF for all {len(prompts)} requests")
        return
    quantiles = statistics.quantiles(gaps, n=10) if len(gaps) > 1 else gaps
    print(f"  {label}: {differing}/{len(prompts)} requests diverge; float32 prefers HF's token in {wrong} and this side's in {differing - wrong}; "
          f"|float32 logit gap| median {statistics.median(gaps):.3f}, 90th percentile {quantiles[-1]:.3f}, max {max(gaps):.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--batches", default="1,8,32")
    args = parser.parse_args()
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()
    prompt_sets = {}
    hf_outputs = {}
    for batch in [int(value) for value in args.batches.split(",")]:
        prompt_sets[batch] = build_prompts(tokenizer, args.prompt_tokens, batch, f"tn-{args.prompt_tokens}-{batch}-0")  # the benchmark's own prompts
        hf_outputs[batch] = hf_generate(hf, prompt_sets[batch], args.output_tokens, device)
    custom = model_loader(args.model, device=torch.device(device), dtype=torch.bfloat16, quantize=None)
    engine_outputs = {batch: engine_generate(custom, tokenizer, prompts, args.output_tokens) for batch, prompts in prompt_sets.items()}
    del custom
    hf = hf.float()  # ground truth
    torch.cuda.empty_cache()
    for batch, prompts in prompt_sets.items():
        print(f"batch {batch}, {args.prompt_tokens}-token prompts, {args.output_tokens} new tokens:")
        analyse("engine vs HF bf16", prompts, hf_outputs[batch], engine_outputs[batch], hf, tokenizer, device)


if __name__ == "__main__":
    main()
