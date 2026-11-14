"""Weight-only int8 matmul for CUDA decode.

Decoding reads every weight once per step, so int8 weights should nearly halve the step time of a memory-bound model. The
plain PyTorch fallback (`weight.to(dtype) * scale` then F.linear) materializes a full 16-bit copy of each weight on every call,
which is slower than not quantizing at all. This kernel reads the int8 bytes, converts them to the activation dtype inside the
tile loop, accumulates in fp32 on tensor cores, and applies the per-output-channel scale once at the end.

It serves the few activation rows decode has (<= MAX_ROWS); prefill's thousands of rows are compute bound and keep the fallback.
The same kernel also runs with 16-bit weights and a scale of ones, which `bench_gemm.py` uses to compare it with cuBLAS.
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
    def _w8a16_kernel(x_ptr, w_ptr, scale_ptr, out_ptr, M, N, K, stride_xm, stride_wn, stride_om,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        """out[M, N] = (x[M, K] @ w[N, K]^T) * scale[N]; one program per (BLOCK_N output columns, BLOCK_M rows)."""
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            k_index = k + offs_k
            x = tl.load(x_ptr + offs_m[:, None] * stride_xm + k_index[None, :], mask=(offs_m[:, None] < M) & (k_index[None, :] < K), other=0.0)
            w = tl.load(w_ptr + offs_n[:, None] * stride_wn + k_index[None, :], mask=(offs_n[:, None] < N) & (k_index[None, :] < K), other=0)
            acc += tl.dot(x, tl.trans(w.to(x.dtype)))
        scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        result = acc * scale[None, :]
        tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :], result.to(out_ptr.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def block_rows(rows):
    return 16 if rows <= 16 else 32 if rows <= 32 else 64


def w8a16_usable(x, weight):
    return (TRITON_AVAILABLE and x.is_cuda and x.dtype in (torch.bfloat16, torch.float16) and x.ndim == 2 and x.shape[0] <= MAX_ROWS
            and weight.is_cuda and weight.is_contiguous() and x.shape[1] == weight.shape[1] and x.shape[1] % 16 == 0
            and os.environ.get("LLMSERVER_W8A16", "1") != "0")


def w8a16_matmul(x, weight, scale, block_n=32, block_k=128, num_warps=4, num_stages=4):
    """x: [M, K] bf16/fp16 (contiguous); weight: [N, K] int8 (or 16-bit); scale: [N] in x's dtype. Returns [M, N] in x's dtype."""
    rows, k = x.shape
    n = weight.shape[0]
    out = torch.empty(rows, n, device=x.device, dtype=x.dtype)
    bm = block_rows(rows)
    _w8a16_kernel[(triton.cdiv(n, block_n), triton.cdiv(rows, bm))](
        x, weight, scale, out, rows, n, k, x.stride(0), weight.stride(0), out.stride(0),
        BLOCK_M=bm, BLOCK_N=block_n, BLOCK_K=block_k, num_warps=num_warps, num_stages=num_stages)
    return out
