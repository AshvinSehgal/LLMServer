"""Which attention backend is fastest for long prefill on this GPU? Causal attention, one sequence, GQA expanded to full heads.

    python bench_prefill_attention.py --heads 32 --kv-heads 8 --head-dim 128

Times PyTorch's flash (FlashAttention-2), cuDNN (Hopper-tuned, off by default in PyTorch 2.5.1), memory-efficient and the cuDNN-free
default dispatch at several lengths, in TFLOP/s of useful (causal) work, and checks each against the flash result.
"""
import argparse
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def timed(call, repeats=5):
    call()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        call()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--lengths", default="1024,4096,16384,32000")
    parser.add_argument("--model", help="read the head counts and head_dim from this model's config")
    args = parser.parse_args()
    if args.model:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(args.model)
        args.heads, args.kv_heads = config.num_attention_heads, config.num_key_value_heads
        args.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    print(f"{torch.cuda.get_device_name()} | torch {torch.__version__} | {args.heads} heads, {args.kv_heads} kv heads, head_dim {args.head_dim}, bf16, causal")
    backends = {"flash": SDPBackend.FLASH_ATTENTION, "cudnn": SDPBackend.CUDNN_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION}
    for length in [int(v) for v in args.lengths.split(",")]:
        q = torch.randn(1, args.heads, length, args.head_dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, args.kv_heads, length, args.head_dim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, args.kv_heads, length, args.head_dim, device="cuda", dtype=torch.bfloat16)
        expand = lambda t: t.repeat_interleave(args.heads // args.kv_heads, dim=1)
        ke, ve = expand(k), expand(v)
        flops = 2 * 2 * args.heads * length * length * args.head_dim / 2  # QK^T and PV, half of the square is masked
        reference = None
        for name, backend in backends.items():
            try:
                with sdpa_kernel(backend):
                    call = lambda: F.scaled_dot_product_attention(q, ke, ve, is_causal=True)
                    seconds = timed(call)
                    result = call()
            except Exception as error:
                print(f"  {length:6d} {name:9s} unavailable: {str(error).splitlines()[0][:90]}")
                continue
            reference = result if reference is None else reference
            print(f"  {length:6d} {name:9s} {seconds * 1000:8.2f} ms  {flops / seconds / 1e12:6.0f} TFLOP/s  max diff vs first backend {(result.float() - reference.float()).abs().max().item():.4f}")
        del q, k, v, ke, ve
    chunk_shapes(args)


def chunk_shapes(args):
    """The second chunk of a long prompt: queries [chunk] against a prefix (no mask) and against themselves (causal), merged. The first call
    at each new shape is timed separately from the repeats: the v9.36 long-context run had a cold first run 4x slower than the second."""
    from utils import prefix_attention
    print("split chunk attention, one layer: first call at a new shape / repeat (ms), cudnn vs flash with log-sum-exp")
    for prefix, chunk in [(16384, 15616), (17152, 14848), (17920, 14080), (18688, 13312), (16384, 16383)]:
        q = torch.randn(1, args.heads, chunk, args.head_dim, device="cuda", dtype=torch.bfloat16)
        pk, pv = (torch.randn(1, args.heads, prefix, args.head_dim, device="cuda", dtype=torch.bfloat16) for _ in range(2))
        ck, cv = (torch.randn(1, args.heads, chunk, args.head_dim, device="cuda", dtype=torch.bfloat16) for _ in range(2))
        for name, use_cudnn in (("cudnn", True), ("flash", False)):
            prefix_attention.USE_CUDNN = use_cudnn
            def call():
                a = prefix_attention.attention_with_lse(q, pk, pv, args.head_dim ** -0.5, False)
                b = prefix_attention.attention_with_lse(q, ck, cv, args.head_dim ** -0.5, True)
                return prefix_attention.merge_attention(a[0], a[1], b[0], b[1])
            torch.cuda.synchronize()
            start = time.perf_counter()
            call()
            torch.cuda.synchronize()
            first = time.perf_counter() - start
            print(f"  prefix {prefix:6d} chunk {chunk:6d} {name:6s} first {first * 1000:8.1f}   repeat {timed(call) * 1000:8.2f}")
    prefix_attention.USE_CUDNN = True
    print(f"  cudnn fallback flags: plain broken={prefix_attention._CUDNN_BROKEN} lse broken={prefix_attention._CUDNN_LSE_BROKEN}")


if __name__ == "__main__":
    main()
