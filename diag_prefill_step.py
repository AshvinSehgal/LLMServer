"""Why does one prefill step cost ~24 ms (1.7B) to ~31 ms (8B) in the serving run, whatever the prompt chunk size?

    python diag_prefill_step.py --model Qwen/Qwen3-1.7B

Builds the engine the way server.py does, then runs single-request prefill steps (a short, a ~600-token and a ~1800-token prompt behind
the shared system prompt, as in serve_bench.py) and reports wall time per step with the GPU synchronised, split three ways:
the engine's own phases (scheduling, batch construction, model forward, sampling and bookkeeping), the Python functions that take the
time (cProfile) and the CUDA kernel count and GPU busy time (torch profiler). If the GPU is mostly idle, the step is host-bound.
"""
import argparse
import cProfile
import io
import math
import pstats
import statistics
import time

import torch
from transformers import AutoTokenizer

from bench_prompts import QUESTIONS
from benchmark_cuda import FILLER
from utils.engine import Engine
from utils.model_loader import model_loader

SYSTEM = ("You are a careful, concise assistant. Answer directly, show steps only when asked, and say so when you are unsure. "
          "Keep answers well organised and avoid repeating the question back to the user. " * 3)


def prompt_ids(tokenizer, kind, index):
    tag = f"[diag {kind}-{index}] "
    content = {"short": tag + QUESTIONS[index % len(QUESTIONS)],
               "medium": tag + "Summarize the following text in three sentences:\n" + FILLER * 6,
               "long": tag + "Summarize the following text in three sentences:\n" + FILLER * 18}[kind]
    text = tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def sync():
    torch.cuda.synchronize()


def one_prefill_step(engine, ids, hooks=None):
    """Adds the request, runs exactly its prefill step and returns (seconds, new tokens in the step)."""
    request = engine.add_request(None, prompt_token_ids=ids)
    sync()
    start = time.perf_counter()
    output = engine.schedule_step()
    after_schedule = time.perf_counter()
    engine.process_prefill_batch(output)
    sync()
    end = time.perf_counter()
    tokens = sum(output.prefill_token_counts)
    engine.cancel_request(request.request_id)
    engine.remove_finished_and_cancelled_requests()
    return end - start, after_schedule - start, tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = model_loader(args.model, device=torch.device("cuda"), dtype=torch.bfloat16)
    max_context, batch = 4096, 32
    engine = Engine(model, tokenizer, max_new_tokens=256, max_batch_size=batch, max_num_batched_tokens=args.max_num_batched_tokens,
                    kv_cache_num_blocks=math.ceil(max_context / 16) * batch, kv_cache_block_size=16, stream_text=True)
    print(f"{torch.cuda.get_device_name()} | {args.model} | budget {args.max_num_batched_tokens}")
    for index in range(4):  # warm-up: kernels, graphs, the shared system-prompt prefix in the cache
        one_prefill_step(engine, prompt_ids(tokenizer, "short", 100 + index))
    results = {}
    for kind in ("short", "medium", "long"):
        times, tokens = [], []
        for index in range(args.repeats):
            seconds, _, count = one_prefill_step(engine, prompt_ids(tokenizer, kind, index))
            times.append(seconds)
            tokens.append(count)
        results[kind] = (statistics.median(times), statistics.median(tokens))
        print(f"{kind:7s} prompt: prefill step median {statistics.median(times) * 1000:7.2f} ms  min {min(times) * 1000:7.2f}  new tokens in the step {statistics.median(tokens):.0f}")

    ids = prompt_ids(tokenizer, "short", 50)
    profiler = cProfile.Profile()
    profiler.enable()
    one_prefill_step(engine, ids)
    profiler.disable()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("tottime").print_stats(18)
    print("\nPython time of one short-prompt prefill step (cProfile, by own time):")
    print("\n".join(line for line in stream.getvalue().splitlines() if line.strip())[:3500])

    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        seconds, _, count = one_prefill_step(engine, prompt_ids(tokenizer, "short", 51))
    events = prof.key_averages()
    kernel_events = [event for event in events if event.device_time_total > 0 and event.device_type == torch.autograd.DeviceType.CUDA]
    launches = sum(event.count for event in kernel_events)
    gpu_busy = sum(event.device_time_total for event in kernel_events) / 1000
    print(f"\ntorch profiler, one short-prompt prefill step ({count} new tokens): wall {seconds * 1000:.1f} ms, {launches} kernel launches, GPU busy {gpu_busy:.1f} ms")
    print(events.table(sort_by="cpu_time_total", row_limit=14, max_name_column_width=48))


if __name__ == "__main__":
    main()
