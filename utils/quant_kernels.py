"""Weight-only int8 matmul for CUDA decode.

Decoding reads every weight once per step, so int8 weights should nearly halve the step time of a memory-bound model. The
plain PyTorch fallback (`weight.to(dtype) * scale` then F.linear) materializes a full 16-bit copy of each weight on every call,
which is slower than not quantizing at all. This kernel reads the int8 bytes, converts them to the activation dtype inside the
tile loop, accumulates in fp32 on tensor cores, and applies the per-output-channel scale once at the end.

It serves the few activation rows decode has (<= MAX_ROWS); prefill's thousands of rows are compute bound and keep the fallback.
The same kernel also runs with 16-bit weights and a scale of ones, which `bench_gemm.py` uses to compare it with cuBLAS.

Narrow outputs (N=4096 gives only 128 programs on a 132-SM GPU, each streaming a long K) leave most of the GPU idle, so the K
range can be split across programs (split-K): each writes an fp32 partial result and the partials are summed in a fixed order, so
results stay deterministic (no atomics).
"""
import os

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    TRITON_AVAILABLE = False

MAX_ROWS = 64


if TRITON_AVAILABLE:
    @triton.jit
    def _w8a16_kernel(x_ptr, w_ptr, scale_ptr, out_ptr, M, N, K, K_PER_SPLIT, stride_xm, stride_wn, stride_om, stride_split,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        """out[s][M, N] = (x[M, K_s] @ w[N, K_s]^T) * scale[N] for the s-th K range; one program per (BLOCK_N columns, BLOCK_M rows, K range)."""
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_k = tl.program_id(2)
        k_start = pid_k * K_PER_SPLIT
        k_end = tl.minimum(K, k_start + K_PER_SPLIT)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(k_start, k_end, BLOCK_K):
            k_index = k + offs_k
            x = tl.load(x_ptr + offs_m[:, None] * stride_xm + k_index[None, :], mask=(offs_m[:, None] < M) & (k_index[None, :] < K), other=0.0)
            w = tl.load(w_ptr + offs_n[:, None] * stride_wn + k_index[None, :], mask=(offs_n[:, None] < N) & (k_index[None, :] < K), other=0)
            acc += tl.dot(x, tl.trans(w.to(x.dtype)))
        scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        result = acc * scale[None, :]
        tl.store(out_ptr + pid_k * stride_split + offs_m[:, None] * stride_om + offs_n[None, :], result.to(out_ptr.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def cdiv(a, b):
    return -(-a // b)


def block_rows(rows):
    return 16 if rows <= 16 else 32 if rows <= 32 else 64


TARGET_PROGRAMS = 264  # about two programs per SM on an H200 (132 SMs)
MAX_SPLIT_K = 8


def default_config(n, rows):
    """(block_n, block_k) from the H200 sweep: wide outputs want 64 columns per program, narrow ones 32 columns and longer K steps."""
    return (64, 128) if n >= 16384 else (32, 256)


def auto_split_k(n, k, rows, block_n, block_k):
    """Smallest power-of-two K split that gives about two programs per SM, keeping at least four K steps per program."""
    programs = cdiv(n, block_n) * cdiv(rows, block_rows(rows))
    split = 1
    while programs * split < TARGET_PROGRAMS and split < MAX_SPLIT_K and k // (split * 2) >= 4 * block_k:
        split *= 2
    return split


def w8a16_usable(x, weight):
    return (TRITON_AVAILABLE and x.is_cuda and x.dtype in (torch.bfloat16, torch.float16) and x.ndim == 2 and x.shape[0] <= MAX_ROWS
            and weight.is_cuda and weight.is_contiguous() and x.shape[1] == weight.shape[1] and x.shape[1] % 16 == 0
            and os.environ.get("LLMSERVER_W8A16", "1") != "0")


def w8a16_matmul(x, weight, scale, block_n=None, block_k=None, num_warps=4, num_stages=3, split_k=None):
    """x: [M, K] bf16/fp16 (contiguous); weight: [N, K] int8 (or 16-bit); scale: [N] in x's dtype. Returns [M, N] in x's dtype.

    block_n, block_k and split_k default to a shape-based choice; pass them to sweep (bench_gemm.py)."""
    rows, k = x.shape
    n = weight.shape[0]
    default_n, default_k = default_config(n, rows)
    block_n = block_n or default_n
    block_k = block_k or default_k
    bm = block_rows(rows)
    split = split_k or auto_split_k(n, k, rows, block_n, block_k)
    k_per_split = cdiv(cdiv(k, split), block_k) * block_k
    split = cdiv(k, k_per_split)  # drop empty trailing ranges
    if split == 1:
        out = torch.empty(rows, n, device=x.device, dtype=x.dtype)
        stride_split = 0
    else:
        out = torch.empty(split, rows, n, device=x.device, dtype=torch.float32)
        stride_split = rows * n
    _w8a16_kernel[(cdiv(n, block_n), cdiv(rows, bm), split)](
        x, weight, scale, out, rows, n, k, k_per_split, x.stride(0), weight.stride(0), n, stride_split,
        BLOCK_M=bm, BLOCK_N=block_n, BLOCK_K=block_k, num_warps=num_warps, num_stages=num_stages)
    return out if split == 1 else out.sum(dim=0).to(x.dtype)
