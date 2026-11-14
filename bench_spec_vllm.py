"""vLLM side of the speculative-decoding comparison (run with the vLLM virtualenv's python). Same prompts, batch 1, greedy.

    python bench_spec_vllm.py --model Qwen/Qwen3-8B --draft-model Qwen/Qwen3-0.6B --draft-tokens 3,5 --new-tokens 128

Draft-model speculation is not available in every vLLM version; an unsupported configuration is reported, not fatal.
"""
import argparse
import time

from transformers import AutoTokenizer

from bench_prompts import QUESTIONS, chat_token_ids


def measure(llm, prompts, new_tokens):
    from vllm import SamplingParams
    full = SamplingParams(temperature=0.0, max_tokens=new_tokens, min_tokens=new_tokens, ignore_eos=True)
    one = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    llm.generate([{"prompt_token_ids": prompts[0]}], full, use_tqdm=False)  # warm-up
    seconds = 0.0
    for ids in prompts:
        start = time.perf_counter()
        llm.generate([{"prompt_token_ids": ids}], one, use_tqdm=False)
        first = time.perf_counter() - start
        start = time.perf_counter()
        llm.generate([{"prompt_token_ids": ids}], full, use_tqdm=False)
        seconds += (time.perf_counter() - start) - first
    return len(prompts) * (new_tokens - 1) / seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--draft-tokens", default="3,5")
    parser.add_argument("--new-tokens", type=int, default=128)
    args = parser.parse_args()
    from vllm import LLM
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [chat_token_ids(tokenizer, question) for question in QUESTIONS]
    common = dict(model=args.model, dtype="bfloat16", enable_prefix_caching=False, gpu_memory_utilization=0.4, max_model_len=2048, max_num_seqs=1, seed=0)
    llm = LLM(**common)
    print(f"vLLM plain decoding: {measure(llm, prompts, args.new_tokens):.1f} tok/s", flush=True)
    del llm
    for k in [int(value) for value in args.draft_tokens.split(",")]:
        try:
            llm = LLM(speculative_config={"model": args.draft_model, "num_speculative_tokens": k}, **common)
            print(f"vLLM draft-model speculation k={k}: {measure(llm, prompts, args.new_tokens):.1f} tok/s", flush=True)
            del llm
        except Exception as failure:
            print(f"vLLM draft-model speculation k={k}: not available here ({type(failure).__name__}: {str(failure)[:200]})", flush=True)


if __name__ == "__main__":
    main()
