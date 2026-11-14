"""Microbenchmark of the decode-shaped matrix multiplies (CUDA only): cuBLAS bf16 vs int8 weight-only kernels.

For each layer shape of a model and each row count (batch size) it times, inside a CUDA graph, cuBLAS bf16 (F.linear), the
dequantize-then-matmul fallback for int8, and the Triton w8a16 kernel (int8 weights) plus the same kernel on bf16 weights
over a sweep of tile sizes, warps and pipeline stages. Weights rotate across enough copies to exceed the GPU's L2 cache, so
the numbers are DRAM bandwidth, not cache hits.

    python bench_gemm.py [--hidden 4096 --intermediate 12288 --heads 32 --kv-heads 8 --head-dim 128 --vocab 151936]
"""
import argparse
import itertools

import torch
import torch.nn.functional as F

from utils.quant_kernels import w8a16_matmul


def time_graph(call, repeats=10, calls_per_graph=24):
    call(0)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for index in range(calls_per_graph):
            call(index)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / (repeats * calls_per_graph)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=12288)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--vocab", type=int, default=151936)
    parser.add_argument("--model", default=None, help="read the layer dimensions from this Hugging Face model's config")
    args = parser.parse_args()
    if args.model:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(args.model)
        args.hidden, args.intermediate, args.heads, args.kv_heads, args.vocab = config.hidden_size, config.intermediate_size, config.num_attention_heads, config.num_key_value_heads, config.vocab_size
        args.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    shapes = {"qkv": ((args.heads + 2 * args.kv_heads) * args.head_dim, args.hidden), "o_proj": (args.hidden, args.heads * args.head_dim),
              "gate_up": (2 * args.intermediate, args.hidden), "down": (args.hidden, args.intermediate), "lm_head": (args.vocab, args.hidden)}
    sweep = list(itertools.product((32, 64, 128), (128, 256), (3, 5), (4, 8)))
    print(f"{torch.cuda.get_device_name()} | bf16 activations | times are per call, weights rotate over >200 MB so DRAM bandwidth is measured")
    for name, (n, k) in shapes.items():
        copies = min(16, max(2, int(200e6 // (n * k)) + 1))
        w16 = [torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05 for _ in range(copies)]
        w8 = [torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8) for _ in range(copies)]
        scale = (torch.rand(n, device="cuda") * 0.01 + 0.001).to(torch.bfloat16)
        ones = torch.ones(n, device="cuda", dtype=torch.bfloat16)
        for rows in (1, 8, 32):
            x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
            reference = F.linear(x.float(), w8[0].float() * scale.float()[:, None])
            cublas = time_graph(lambda i: F.linear(x, w16[i % copies]))
            dequant = time_graph(lambda i: F.linear(x, w8[i % copies].to(torch.bfloat16) * scale[:, None]))
            rows_out = []
            for label, weights, scl, weight_bytes in (("int8", w8, scale, n * k), ("bf16", w16, ones, 2 * n * k)):
                best = None
                for block_n, block_k, stages, warps in sweep:
                    try:
                        ms = time_graph(lambda i: w8a16_matmul(x, weights[i % copies], scl, block_n, block_k, warps, stages))
                    except Exception:
                        continue
                    if best is None or ms < best[0]:
                        best = (ms, block_n, block_k, stages, warps)
                if best is None:
                    rows_out.append(f"    triton {label}: no configuration compiled")
                    continue
                ms, block_n, block_k, stages, warps = best
                error = (w8a16_matmul(x, weights[0], scl, block_n, block_k, warps, stages).float() - (reference if label == "int8" else F.linear(x.float(), weights[0].float()))).abs().max().item()
                rows_out.append(f"    triton {label}: {ms * 1000:7.1f} us  {weight_bytes / ms / 1e6:6.0f} GB/s  best of {len(sweep)}: block_n={block_n} block_k={block_k} stages={stages} warps={warps}  max err {error:.4f}")
            print(f"{name:8s} N={n:6d} K={k:5d} rows={rows:2d} | cuBLAS bf16 {cublas * 1000:7.1f} us ({2 * n * k / cublas / 1e6:5.0f} GB/s) | int8 dequant fallback {dequant * 1000:7.1f} us")
            print("\n".join(rows_out), flush=True)
        del w16, w8


if __name__ == "__main__":
    main()
