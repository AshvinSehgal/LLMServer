"""Perplexity on WikiText-2 (test split) for Hugging Face bf16, this engine in bf16 and this engine quantized.

    python eval_quality.py --model Qwen/Qwen3-8B --windows 40 --window-tokens 1024 --device cuda

Same tokens, same windows, teacher-forced; lower is better. The engine's bf16 row should match Hugging Face's to within noise
(it checks the engine's own model code); the int8 row shows what weight quantization costs. Needs the `datasets` package.
"""
import argparse
import math

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.model_loader import model_loader


def load_tokens(tokenizer, windows, window_tokens):
    from datasets import load_dataset
    text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    count = min(windows, len(ids) // window_tokens)
    return ids[:count * window_tokens].view(count, window_tokens)


@torch.inference_mode()
def perplexity(logits_fn, windows, device, batch=2):
    total, count = 0.0, 0
    for start in range(0, len(windows), batch):
        ids = windows[start:start + batch].to(device)
        logits = logits_fn(ids)[:, :-1].float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="sum")
        total += loss.item()
        count += ids[:, 1:].numel()
    return math.exp(total / count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--windows", type=int, default=40)
    parser.add_argument("--window-tokens", type=int, default=1024)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variants", default="hf,engine,int8")
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    windows = load_tokens(tokenizer, args.windows, args.window_tokens)
    print(f"{args.model} | WikiText-2 test | {len(windows)} windows x {args.window_tokens} tokens | {args.dtype}")
    for variant in args.variants.split(","):
        if variant == "hf":
            model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
            value = perplexity(lambda ids: model(input_ids=ids).logits, windows, device)
        else:
            model = model_loader(args.model, device=device, dtype=dtype, quantize="int8" if variant == "int8" else None)
            value = perplexity(lambda ids: model(ids), windows, device)
        label = {"hf": f"Hugging Face {args.dtype}", "engine": f"engine {args.dtype}", "int8": "engine int8 (weight-only)"}[variant]
        print(f"  {label:<28} perplexity {value:.4f}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
