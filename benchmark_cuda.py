"""Compare Hugging Face Transformers, this engine and vLLM on identical token-ID prompts.

Each backend runs in its own process (so only one model is ever resident on the GPU) and writes a JSON file; the driver
merges them into a table.

    python benchmark_cuda.py --run-all --model Qwen/Qwen3-1.7B --prompt-tokens 128,1024 --batch-sizes 1,8,32 --output-tokens 128

Timing is library-agnostic. For each (prompt length, batch size) configuration:
  * ttft_s   = time to generate 1 token for the whole batch (prefill latency)
  * total_s  = time to generate `output_tokens` tokens for the whole batch
  * decode tok/s = batch * (output_tokens - 1) / (total_s - ttft_s)
Every library is forced to generate exactly `output_tokens` tokens (EOS ignored) from the same token IDs, greedily.
Prompts differ from their first KV block on, so prefix caching cannot help any library.
"""
import argparse
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time

FILLER = ("The history of computing spans many decades of research, from early mechanical calculators to modern parallel processors. "
          "Engineers study how memory bandwidth, latency and arithmetic throughput shape the speed of numerical programs, and they build "
          "libraries that hide those details behind simple interfaces. A language model generates text one token at a time, reading its "
          "entire set of weights for every step, which makes the speed of generation depend mostly on how quickly memory can be read. ")


def build_prompts(tokenizer, prompt_tokens, count, tag):
    filler = tokenizer(FILLER * (prompt_tokens // 40 + 2))["input_ids"]
    prompts = []
    for index in range(count):
        head = tokenizer(f"Request {tag}-{index}: ")["input_ids"]
        prompts.append((head + filler)[:prompt_tokens])
    return prompts


def parse_ints(text):
    return [int(item) for item in text.split(",") if item.strip()]


def sync(torch_module):
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()
    elif getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        torch_module.mps.synchronize()


def pick_device(torch_module, requested):
    if requested != "auto":
        return requested
    if torch_module.cuda.is_available():
        return "cuda"
    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFBackend:
    name = "hf"

    def __init__(self, args, torch_module):
        from transformers import AutoModelForCausalLM
        self.torch = torch_module
        self.device = pick_device(torch_module, args.device)
        self.model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch_module, args.dtype)).to(self.device).eval()

    def prepare(self, max_batch, max_length):
        pass

    def run(self, prompts, new_tokens):
        torch = self.torch
        input_ids = torch.tensor(prompts, device=self.device)
        with torch.inference_mode():
            output = self.model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), max_new_tokens=new_tokens, min_new_tokens=new_tokens,
                                         do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=0)
        return output[:, input_ids.shape[1]:].tolist()

    def close(self):
        del self.model


class CustomBackend:
    """This engine. Variants: graph (paged Triton attention + CUDA graphs), eager (paged, no graphs), gather (previous path)."""

    def __init__(self, args, torch_module, variant="graph"):
        from transformers import AutoTokenizer
        from utils.model_loader import model_loader
        self.torch = torch_module
        self.variant = variant
        self.name = "custom" if variant == "graph" else f"custom-{variant}"
        self.device = pick_device(torch_module, args.device)
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.model = model_loader(args.model, device=torch_module.device(self.device), dtype=getattr(torch_module, args.dtype), quantize=args.custom_quantize)
        self.engine = None

    def prepare(self, max_batch, max_length):
        from utils.engine import Engine
        torch = self.torch
        block_size = self.args.block_size
        config = self.model.config
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        bytes_per_block = 2 * config.num_hidden_layers * block_size * config.num_key_value_heads * head_dim * torch.empty(0, dtype=self.model.embed_tokens.weight.dtype).element_size()
        wanted = max_batch * math.ceil(max_length / block_size) + 8
        if self.device == "cuda":
            free_bytes = torch.cuda.mem_get_info()[0]
            wanted = min(wanted, int(free_bytes * 0.8 // bytes_per_block))
        # The paged Triton path and CUDA graphs only exist on CUDA; elsewhere every variant uses the default decode path.
        on_cuda = self.device == "cuda"
        options = {"paged_decode": self.variant != "gather" if on_cuda else False, "cuda_graphs": self.variant == "graph" if on_cuda else False}
        self.engine = Engine(self.model, self.tokenizer, max_new_tokens=1, max_batch_size=max_batch, max_num_batched_tokens=self.args.max_num_batched_tokens,
                             kv_cache_num_blocks=wanted, kv_cache_block_size=block_size, stream_text=False, max_completed_requests=0,
                             ignore_eos=True, paged_context_limit=max(8192, max_length), **options)
        self.capacity_tokens = wanted * block_size

    def run(self, prompts, new_tokens):
        engine = self.engine
        engine.max_new_tokens = new_tokens
        if sum(len(prompt) + new_tokens for prompt in prompts) > self.capacity_tokens:
            raise MemoryError("KV cache too small for this configuration")
        requests = [engine.add_request(None, prompt_token_ids=prompt) for prompt in prompts]
        while engine.step():
            pass
        return [request.generated_token_ids for request in requests]

    def close(self):
        self.engine = None
        del self.model


class VLLMBackend:
    name = "vllm"

    def __init__(self, args, torch_module):
        self.args = args
        self.torch = torch_module
        self.llm = None
        # FlashInfer's sampler JIT-compiles with nvcc at startup and fails on the image's CUDA 12.1 toolkit; the stock sampler is equivalent for greedy decoding.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        import vllm  # fail early with a clear message if vLLM is not installed
        self.vllm = vllm

    def prepare(self, max_batch, max_length):
        self.llm = self.vllm.LLM(model=self.args.model, dtype=self.args.dtype, enable_prefix_caching=False, gpu_memory_utilization=self.args.vllm_gpu_utilization,
                                 max_model_len=max_length + 16, max_num_seqs=max(max_batch, 1), max_num_batched_tokens=self.args.max_num_batched_tokens, seed=0)

    def run(self, prompts, new_tokens):
        params = self.vllm.SamplingParams(temperature=0.0, max_tokens=new_tokens, min_tokens=new_tokens, ignore_eos=True)
        outputs = self.llm.generate([{"prompt_token_ids": prompt} for prompt in prompts], params, use_tqdm=False)
        return [list(output.outputs[0].token_ids) for output in outputs]

    def close(self):
        self.llm = None


def environment(torch_module):
    info = {"python": platform.python_version(), "torch": torch_module.__version__, "platform": platform.platform()}
    if torch_module.cuda.is_available():
        properties = torch_module.cuda.get_device_properties(0)
        info.update(gpu=properties.name, gpu_memory_gb=round(properties.total_memory / 2 ** 30, 1), compute_capability=f"{properties.major}.{properties.minor}", cuda=torch_module.version.cuda)
    for module_name in ("transformers", "triton", "vllm"):
        try:
            info[module_name] = __import__(module_name).__version__
        except Exception:
            info[module_name] = None
    return info


def median(values):
    return statistics.median(values) if values else float("nan")


def run_backend(args):
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from transformers import AutoTokenizer
    result = {"backend": args.backend, "env": environment(torch), "results": [], "error": None, "args": vars(args)}
    backend = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if args.backend == "hf":
            backend = HFBackend(args, torch)
        elif args.backend == "vllm":
            backend = VLLMBackend(args, torch)
        elif args.backend.startswith("custom"):
            backend = CustomBackend(args, torch, variant=args.backend.partition("-")[2] or "graph")
        else:
            raise ValueError(f"Unknown backend {args.backend}")
        prompt_lengths, batch_sizes, output_tokens = parse_ints(args.prompt_tokens), parse_ints(args.batch_sizes), args.output_tokens
        backend.prepare(max(batch_sizes), max(prompt_lengths) + output_tokens)
        for prompt_length in prompt_lengths:
            for batch in batch_sizes:
                entry = {"prompt_tokens": prompt_length, "batch": batch, "new_tokens": output_tokens}
                try:
                    backend.run(build_prompts(tokenizer, prompt_length, batch, "warm"), output_tokens)
                    ttft, total, tokens = [], [], None
                    for repeat in range(args.repeats):
                        prompts = build_prompts(tokenizer, prompt_length, batch, f"t1-{repeat}")
                        sync(torch)
                        start = time.perf_counter()
                        backend.run(prompts, 1)
                        sync(torch)
                        ttft.append(time.perf_counter() - start)
                    for repeat in range(args.repeats):
                        prompts = build_prompts(tokenizer, prompt_length, batch, f"tn-{repeat}")
                        sync(torch)
                        start = time.perf_counter()
                        generated = backend.run(prompts, output_tokens)
                        sync(torch)
                        total.append(time.perf_counter() - start)
                        if repeat == 0:
                            tokens = generated
                    ttft_s, total_s = median(ttft), median(total)
                    decode_s = max(total_s - ttft_s, 1e-9)
                    entry.update(ttft_s=ttft_s, total_s=total_s, decode_tok_s=batch * (output_tokens - 1) / decode_s, e2e_tok_s=batch * output_tokens / total_s,
                                 ms_per_token=1000 * decode_s / max(output_tokens - 1, 1), tokens=tokens, ttft_runs=ttft, total_runs=total)
                    if torch.cuda.is_available():
                        entry["peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
                except Exception as error:  # record per-configuration failures (e.g. out of memory) and continue
                    entry["error"] = f"{type(error).__name__}: {error}"
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                result["results"].append(entry)
                print(f"[{args.backend}] prompt={prompt_length} batch={batch}: " + (entry.get("error") or f"ttft={entry['ttft_s']:.3f}s decode={entry['decode_tok_s']:.1f} tok/s e2e={entry['e2e_tok_s']:.1f} tok/s"), flush=True)
        if args.backend.startswith("custom") and args.profile and torch.cuda.is_available():
            profile_custom(backend, tokenizer, args, torch)
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        print(f"[{args.backend}] FAILED: {result['error']}", flush=True)
    finally:
        if backend is not None:
            backend.close()
        gc.collect()
    with open(args.json, "w") as handle:
        json.dump(result, handle)
    return 0 if result["error"] is None else 1


def profile_custom(backend, tokenizer, args, torch):
    """Kernel-level profile of a few decode steps, with and without CUDA graphs, written to --profile-out."""
    from torch.profiler import profile, ProfilerActivity
    engine = backend.engine
    lines = []
    for batch in (1, max(parse_ints(args.batch_sizes))):
        for graphs in (False, True):
            runner = engine.graph_runner
            if graphs and runner is None:
                continue
            engine.graph_runner = runner if graphs else None
            prompts = build_prompts(tokenizer, min(parse_ints(args.prompt_tokens)), batch, f"prof-{batch}-{graphs}")
            engine.max_new_tokens = 64
            requests = [engine.add_request(None, prompt_token_ids=prompt) for prompt in prompts]
            for _ in range(3):
                engine.step()
            sync(torch)
            start = time.perf_counter()
            for _ in range(5):
                engine.step()
            sync(torch)
            step_ms = (time.perf_counter() - start) / 5 * 1000
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for _ in range(5):
                    engine.step()
                sync(torch)
            lines.append(f"===== batch={batch} cuda_graphs={graphs} decode step {step_ms:.2f} ms =====")
            lines.append(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=70))
            for request in requests:
                engine.cancel_request(request.request_id)
            engine.graph_runner = runner
    with open(args.profile_out, "w") as handle:
        handle.write("\n".join(lines))


def compare_tokens(reference, other):
    if not reference or not other:
        return None
    identical = sum(1 for a, b in zip(reference, other) if a == b)
    return identical / len(reference)


def format_report(results):
    by_backend = {item["backend"]: item for item in results}
    names = [name for name in ("hf", "vllm", "custom-gather", "custom-eager", "custom") if name in by_backend]
    configs = []
    for item in results:
        for entry in item["results"]:
            key = (entry["prompt_tokens"], entry["batch"])
            if key not in configs:
                configs.append(key)
    lookup = {(name, key): entry for name, item in by_backend.items() for entry in item["results"] for key in [(entry["prompt_tokens"], entry["batch"])]}
    out = []
    env = next((item["env"] for item in results if item["env"]), {})
    out.append(f"GPU: {env.get('gpu')} ({env.get('gpu_memory_gb')} GB, sm_{str(env.get('compute_capability', '')).replace('.', '')}) | torch {env.get('torch')} | CUDA {env.get('cuda')} | triton {env.get('triton')} | vllm {env.get('vllm')} | transformers {env.get('transformers')}")
    for item in results:
        if item["error"]:
            out.append(f"NOTE: backend {item['backend']} failed: {item['error']}")
    for title, field, fmt in (("Time to first token (s, lower is better)", "ttft_s", "{:.3f}"), ("Decode throughput (tokens/s across the batch, higher is better)", "decode_tok_s", "{:.1f}"), ("End-to-end throughput (tokens/s, higher is better)", "e2e_tok_s", "{:.1f}")):
        out.append(f"\n{title}")
        out.append("| prompt | batch | " + " | ".join(names) + " | custom vs hf | custom vs vllm |")
        out.append("|---|---|" + "---|" * (len(names) + 2))
        for key in configs:
            cells, values = [], {}
            for name in names:
                entry = lookup.get((name, key))
                value = entry.get(field) if entry and not entry.get("error") else None
                values[name] = value
                cells.append(fmt.format(value) if value is not None else ("err" if entry else "n/a"))

            def ratio(other):
                custom, baseline = values.get("custom"), values.get(other)
                if custom is None or baseline is None:
                    return "n/a"
                return f"{(baseline / custom if field == 'ttft_s' else custom / baseline):.2f}x"

            out.append(f"| {key[0]} | {key[1]} | " + " | ".join(cells) + f" | {ratio('hf')} | {ratio('vllm')} |")
    out.append("\n(Speedup columns: >1 means the custom engine is faster; for time-to-first-token they are other/custom.)")
    out.append("\nGreedy token agreement with Hugging Face (fraction of requests with identical output)")
    out.append("| prompt | batch | " + " | ".join(name for name in names if name != "hf") + " |")
    out.append("|---|---|" + "---|" * (len(names) - 1))
    for key in configs:
        reference = lookup.get(("hf", key), {}).get("tokens")
        cells = []
        for name in names:
            if name == "hf":
                continue
            agreement = compare_tokens(reference, lookup.get((name, key), {}).get("tokens"))
            cells.append(f"{agreement:.2f}" if agreement is not None else "n/a")
        out.append(f"| {key[0]} | {key[1]} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def run_all(args):
    backends = [name.strip() for name in args.backends.split(",") if name.strip()]
    if args.ablate and "custom" in backends:
        backends = [name for name in backends if name != "custom"] + ["custom-gather", "custom-eager", "custom"]
    os.makedirs(args.out_dir, exist_ok=True)
    forwarded = ["--model", args.model, "--prompt-tokens", args.prompt_tokens, "--batch-sizes", args.batch_sizes, "--output-tokens", str(args.output_tokens),
                 "--repeats", str(args.repeats), "--dtype", args.dtype, "--device", args.device, "--block-size", str(args.block_size),
                 "--max-num-batched-tokens", str(args.max_num_batched_tokens), "--vllm-gpu-utilization", str(args.vllm_gpu_utilization)]
    if args.custom_quantize:
        forwarded += ["--custom-quantize", args.custom_quantize]
    results = []
    for backend in backends:
        path = os.path.join(args.out_dir, f"{backend}.json")
        interpreter = args.vllm_python if backend == "vllm" and args.vllm_python else sys.executable
        command = [interpreter, os.path.abspath(__file__), "--backend", backend, "--json", path] + forwarded
        if args.profile and backend == "custom":
            command += ["--profile", "--profile-out", os.path.join(args.out_dir, "profile.txt")]
        print(f"\n=== running {backend} in a fresh process ===", flush=True)
        status = subprocess.run(command).returncode
        if os.path.exists(path):
            with open(path) as handle:
                results.append(json.load(handle))
        else:
            results.append({"backend": backend, "env": {}, "results": [], "error": f"process exited with status {status} before writing results"})
    report = format_report(results)
    with open(os.path.join(args.out_dir, "report.md"), "w") as handle:
        handle.write(report)
    print("\n" + report)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-all", action="store_true", help="Run every backend in its own process and print a combined report")
    parser.add_argument("--backends", default="hf,custom,vllm", help="Comma-separated: hf, custom, vllm")
    parser.add_argument("--ablate", action="store_true", help="Also run custom-gather (old path) and custom-eager (paged, no CUDA graph) to attribute the gains")
    parser.add_argument("--backend", help="Run a single backend (used by --run-all): hf, vllm, custom, custom-eager, custom-gather")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--prompt-tokens", default="128,1024")
    parser.add_argument("--batch-sizes", default="1,8,32")
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--vllm-gpu-utilization", type=float, default=0.85)
    parser.add_argument("--vllm-python", default=None, help="Python interpreter to use for the vllm backend (e.g. a virtualenv that has vLLM and its own pinned PyTorch)")
    parser.add_argument("--custom-quantize", default=None, choices=(None, "int8", "int4"))
    parser.add_argument("--profile", action="store_true", help="Write a kernel-level profile of the custom engine's decode steps")
    parser.add_argument("--profile-out", default="profile.txt")
    parser.add_argument("--json", default="result.json")
    parser.add_argument("--out-dir", default="bench_results")
    args = parser.parse_args()
    if args.run_all:
        return run_all(args)
    if not args.backend:
        parser.error("pass --run-all or --backend")
    return run_backend(args)


if __name__ == "__main__":
    sys.exit(main())
