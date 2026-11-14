"""Open-loop load test for any OpenAI-compatible chat server (this repo's server.py or `vllm serve`).

    python serve_bench.py --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-8B --rates 2,4,8,16 --requests 96

Requests arrive as a Poisson process at each rate (requests/s), whatever the server's backlog, which is how real traffic
behaves; a closed loop would hide queueing. The mix is a shared system prompt plus 70% short chat questions, 20% medium
(~600-token) and 10% long (~1800-token) documents, each with a unique first line, so prefix caching helps only through the shared
system prompt. Streamed responses give time to first token (TTFT) and time per output token (TPOT); the final chunk's usage
gives the token counts. Reports throughput and p50/p90/p99 latencies per rate.
"""
import argparse
import asyncio
import json
import random
import time

import httpx

from bench_prompts import QUESTIONS
from benchmark_cuda import FILLER

SYSTEM = ("You are a careful, concise assistant. Answer directly, show steps only when asked, and say so when you are unsure. "
          "Keep answers well organised and avoid repeating the question back to the user. " * 3)


def build_workload(count, seed):
    rng = random.Random(seed)
    jobs = []
    for index in range(count):
        kind = rng.choices(["short", "medium", "long"], weights=[70, 20, 10])[0]
        tag = f"[request {seed}-{index}] "
        if kind == "short":
            content = tag + rng.choice(QUESTIONS)
        elif kind == "medium":
            content = tag + "Summarize the following text in three sentences:\n" + FILLER * 6
        else:
            content = tag + "Summarize the following text in three sentences:\n" + FILLER * 18
        jobs.append({"kind": kind, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]})
    return jobs


async def one_request(client, url, model, job, max_tokens, results):
    payload = {"model": model, "messages": job["messages"], "max_tokens": max_tokens, "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True}}
    sent = time.perf_counter()
    first = None
    completion_tokens = None
    try:
        async with client.stream("POST", url + "/v1/chat/completions", json=payload, timeout=600) as response:
            if response.status_code != 200:
                results.append({"kind": job["kind"], "error": f"HTTP {response.status_code}"})
                return
            async for line in response.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if chunk.get("usage"):
                    completion_tokens = chunk["usage"].get("completion_tokens")
                choices = chunk.get("choices") or []
                if first is None and choices and choices[0].get("delta", {}).get("content"):
                    first = time.perf_counter()
    except Exception as failure:
        results.append({"kind": job["kind"], "error": f"{type(failure).__name__}: {failure}"})
        return
    done = time.perf_counter()
    if first is None or not completion_tokens:
        results.append({"kind": job["kind"], "error": "no tokens"})
        return
    results.append({"kind": job["kind"], "ttft": first - sent, "e2e": done - sent, "tokens": completion_tokens,
                    "tpot": (done - first) / (completion_tokens - 1) if completion_tokens > 1 else None})


def percentile(values, fraction):
    values = sorted(value for value in values if value is not None)
    return values[min(len(values) - 1, int(fraction * len(values)))] if values else float("nan")


async def run_rate(args, rate, seed):
    jobs = build_workload(args.requests, seed)
    results = []
    rng = random.Random(seed + 1)
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=1000)) as client:
        start = time.perf_counter()
        tasks = []
        at = 0.0
        for job in jobs:
            at += rng.expovariate(rate) if rate != float("inf") else 0.0
            delay = start + at - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(one_request(client, args.base_url, args.model, job, args.max_tokens, results)))
        await asyncio.gather(*tasks)
        wall = time.perf_counter() - start
    good = [r for r in results if "error" not in r]
    errors = len(results) - len(good)
    tokens = sum(r["tokens"] for r in good)
    return {"rate": rate, "requests": len(results), "errors": errors, "wall_s": wall, "req_per_s": len(good) / wall, "output_tok_per_s": tokens / wall,
            "ttft_p50": percentile([r["ttft"] for r in good], 0.5), "ttft_p90": percentile([r["ttft"] for r in good], 0.9), "ttft_p99": percentile([r["ttft"] for r in good], 0.99),
            "tpot_p50": percentile([r["tpot"] for r in good], 0.5), "tpot_p90": percentile([r["tpot"] for r in good], 0.9), "tpot_p99": percentile([r["tpot"] for r in good], 0.99),
            "e2e_p50": percentile([r["e2e"] for r in good], 0.5), "e2e_p99": percentile([r["e2e"] for r in good], 0.99)}


async def main_async(args):
    summaries = []
    async with httpx.AsyncClient() as client:  # warm-up: compile kernels, capture graphs, fill the shared-prefix cache
        for index in range(3):
            await one_request(client, args.base_url, args.model, build_workload(3, 999)[index], 32, [])
    for rate in [float("inf") if value == "inf" else float(value) for value in args.rates.split(",")]:
        summary = await run_rate(args, rate, args.seed)
        summaries.append(summary)
        print(f"rate {'burst' if rate == float('inf') else rate:>5} req/s | done {summary['requests'] - summary['errors']:>3}/{summary['requests']} | {summary['req_per_s']:6.2f} req/s {summary['output_tok_per_s']:8.1f} out tok/s | "
              f"TTFT p50 {summary['ttft_p50'] * 1000:7.0f} p90 {summary['ttft_p90'] * 1000:7.0f} p99 {summary['ttft_p99'] * 1000:7.0f} ms | "
              f"TPOT p50 {summary['tpot_p50'] * 1000:6.1f} p90 {summary['tpot_p90'] * 1000:6.1f} p99 {summary['tpot_p99'] * 1000:6.1f} ms | E2E p50 {summary['e2e_p50']:5.2f} p99 {summary['e2e_p99']:5.2f} s", flush=True)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(summaries, handle, indent=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--rates", default="2,4,8,16", help="comma-separated arrival rates in requests/s; 'inf' sends everything at once")
    parser.add_argument("--requests", type=int, default=96)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default=None)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
