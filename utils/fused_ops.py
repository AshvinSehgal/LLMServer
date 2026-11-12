"""Fused elementwise kernels for the CUDA decode step.

PyTorch 2.5 decomposes rms_norm into pow/mean/rsqrt/mul launches and runs the residual add and SiLU*up as separate
kernels. At batch 1 each launch is ~1-2 microseconds of GPU time, so the roughly 300 tiny kernels per decode step cost
more than the attention. These kernels do each job in one pass over the activations.

Every function has a plain PyTorch twin (`*_reference`) that doubles as the CPU/MPS fallback and the test oracle. Both
round at the same points (the residual sum and SiLU are rounded to the activation dtype, as the unfused model does).
"""
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    TRITON_AVAILABLE = False


def add_rms_norm_reference(hidden, delta, weight, eps):
    """Returns (hidden + delta, rms_norm(hidden + delta) * weight). delta may be None (plain norm; first output is hidden)."""
    total = hidden if delta is None else hidden + delta
    return total, F.rms_norm(total, (weight.shape[0],), weight, eps)


def qkv_norm_rotary_write_reference(qkv, norm_weight, cosine, signed_sine, slot_mapping, key_cache, value_cache, n_query_heads, n_kv_heads, head_dim, eps):
    """Unfused twin of `qkv_norm_rotary_write`: returns Q [rows, n_query_heads, head_dim] and writes K, V into the cache slots.

    qkv: [rows, 1, (n_query_heads + 2 * n_kv_heads) * head_dim]; norm_weight: [n_query_heads + n_kv_heads, head_dim] or None;
    cosine/signed_sine: [rows, 1, head_dim]; key_cache/value_cache: one layer, [blocks, block_size, n_kv_heads, head_dim].
    """
    rows = qkv.shape[0]
    n_qk = n_query_heads + n_kv_heads
    qk = qkv[..., :n_qk * head_dim].reshape(rows, 1, n_qk, head_dim)
    values = qkv[..., n_qk * head_dim:].reshape(rows, n_kv_heads, head_dim)
    if norm_weight is not None:
        qk = F.rms_norm(qk, (head_dim,), None, eps) * norm_weight
    half = head_dim // 2
    qk = torch.addcmul(qk * cosine.unsqueeze(2), torch.cat([qk[..., half:], qk[..., :half]], dim=-1), signed_sine.unsqueeze(2))
    key_cache.view(-1, n_kv_heads, head_dim).index_copy_(0, slot_mapping, qk[:, 0, n_query_heads:])
    value_cache.view(-1, n_kv_heads, head_dim).index_copy_(0, slot_mapping, values)
    return qk[:, 0, :n_query_heads]


def qk_norm_rotary_reference(qkv, norm_weight, cosine, signed_sine, n_query_heads, n_kv_heads, head_dim, eps):
    """Q/K RMSNorm (optional) + rotary for a [batch, seq, (n_query_heads + 2 * n_kv_heads) * head_dim] projection; returns [batch, seq, n_query_heads + n_kv_heads, head_dim]."""
    batch, seq = qkv.shape[:2]
    n_qk = n_query_heads + n_kv_heads
    qk = qkv[..., :n_qk * head_dim].reshape(batch, seq, n_qk, head_dim)
    if norm_weight is not None:
        qk = F.rms_norm(qk, (head_dim,), None, eps) * norm_weight
    half = head_dim // 2
    return torch.addcmul(qk * cosine.unsqueeze(2), torch.cat([qk[..., half:], qk[..., :half]], dim=-1), signed_sine.unsqueeze(2))


def silu_mul_reference(gate_up):
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


if TRITON_AVAILABLE:
    @triton.jit
    def _add_rms_norm_kernel(x_ptr, delta_ptr, weight_ptr, sum_ptr, out_ptr, eps, N, BLOCK: tl.constexpr, HAS_DELTA: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0)
        if HAS_DELTA:
            delta = tl.load(delta_ptr + row * N + cols, mask=mask, other=0.0)
            x = (x.to(tl.float32) + delta.to(tl.float32)).to(x_ptr.dtype.element_ty)  # round the residual stream like the unfused add
            tl.store(sum_ptr + row * N + cols, x, mask=mask)
        xf = x.to(tl.float32)
        rstd = tl.rsqrt(tl.sum(xf * xf, axis=0) / N + eps)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + row * N + cols, (xf * rstd * weight).to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _silu_mul_kernel(gate_up_ptr, out_ptr, N, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < N
        gate = tl.load(gate_up_ptr + row * 2 * N + cols, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(gate_up_ptr + row * 2 * N + N + cols, mask=mask, other=0.0).to(tl.float32)
        silu = (gate * tl.sigmoid(gate)).to(out_ptr.dtype.element_ty).to(tl.float32)  # SiLU rounds to the activation dtype before the multiply
        tl.store(out_ptr + row * N + cols, (silu * up).to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _qkv_norm_rotary_write_kernel(qkv_ptr, weight_ptr, cos_ptr, sin_ptr, slot_ptr, q_ptr, key_cache_ptr, value_cache_ptr, eps,
                                      N_Q: tl.constexpr, N_KV: tl.constexpr, HEAD_DIM: tl.constexpr, HAS_NORM: tl.constexpr, WRITE_CACHE: tl.constexpr):
        """One program per (row, head): normalize + rotate Q/K heads (rounding where the unfused ops round), copy V, and store
        Q to its output and K/V straight into the cache slot."""
        row = tl.program_id(0)
        head = tl.program_id(1)
        HALF: tl.constexpr = HEAD_DIM // 2
        offs = tl.arange(0, HALF)
        src = qkv_ptr + row * (N_Q + 2 * N_KV) * HEAD_DIM + head * HEAD_DIM
        dtype = qkv_ptr.dtype.element_ty
        x1 = tl.load(src + offs)
        x2 = tl.load(src + HALF + offs)
        if head < N_Q + N_KV:
            if HAS_NORM:
                f1 = x1.to(tl.float32)
                f2 = x2.to(tl.float32)
                rstd = tl.rsqrt((tl.sum(f1 * f1, axis=0) + tl.sum(f2 * f2, axis=0)) / HEAD_DIM + eps)
                w1 = tl.load(weight_ptr + head * HEAD_DIM + offs).to(tl.float32)
                w2 = tl.load(weight_ptr + head * HEAD_DIM + HALF + offs).to(tl.float32)
                x1 = (((f1 * rstd).to(dtype)).to(tl.float32) * w1).to(dtype)  # rms_norm rounds to dtype, then the weight multiply rounds again
                x2 = (((f2 * rstd).to(dtype)).to(tl.float32) * w2).to(dtype)
            c1 = tl.load(cos_ptr + row * HEAD_DIM + offs).to(tl.float32)
            c2 = tl.load(cos_ptr + row * HEAD_DIM + HALF + offs).to(tl.float32)
            s1 = tl.load(sin_ptr + row * HEAD_DIM + offs).to(tl.float32)
            s2 = tl.load(sin_ptr + row * HEAD_DIM + HALF + offs).to(tl.float32)
            a1 = (x1.to(tl.float32) * c1).to(dtype).to(tl.float32)  # x * cos rounds, then addcmul adds the swapped * signed_sin in one fused step
            a2 = (x2.to(tl.float32) * c2).to(dtype).to(tl.float32)
            r1 = (a1 + x2.to(tl.float32) * s1).to(dtype)
            r2 = (a2 + x1.to(tl.float32) * s2).to(dtype)
            if WRITE_CACHE:
                if head < N_Q:
                    dst = q_ptr + row * N_Q * HEAD_DIM + head * HEAD_DIM
                else:
                    slot = tl.load(slot_ptr + row)
                    dst = key_cache_ptr + slot * (N_KV * HEAD_DIM) + (head - N_Q) * HEAD_DIM
            else:
                dst = q_ptr + row * (N_Q + N_KV) * HEAD_DIM + head * HEAD_DIM  # q_ptr is the [rows, N_Q + N_KV, HEAD_DIM] output
            tl.store(dst + offs, r1)
            tl.store(dst + HALF + offs, r2)
        elif WRITE_CACHE:  # without the cache write the grid has no V programs; this also keeps the dummy slot pointer from being typed
            slot = tl.load(slot_ptr + row)
            dst = value_cache_ptr + slot * (N_KV * HEAD_DIM) + (head - N_Q - N_KV) * HEAD_DIM
            tl.store(dst + offs, x1)
            tl.store(dst + HALF + offs, x2)


def next_power_of_two(value):
    return 1 << max(0, value - 1).bit_length()


def fused_ops_usable(device):
    return TRITON_AVAILABLE and torch.device(device).type == "cuda"


def add_rms_norm(hidden, delta, weight, eps):
    """(hidden + delta, rms_norm(hidden + delta) * weight) in one kernel on CUDA; hidden/delta must be contiguous."""
    if not (TRITON_AVAILABLE and hidden.is_cuda and hidden.is_contiguous() and (delta is None or delta.is_contiguous())):
        return add_rms_norm_reference(hidden, delta, weight, eps)
    width = hidden.shape[-1]
    rows = hidden.numel() // width
    out = torch.empty_like(hidden)
    total = hidden if delta is None else torch.empty_like(hidden)
    _add_rms_norm_kernel[(rows,)](hidden, hidden if delta is None else delta, weight, total, out, eps, width,
                                  BLOCK=next_power_of_two(width), HAS_DELTA=delta is not None, num_warps=8 if width >= 2048 else 4)
    return total, out


def silu_mul(gate_up):
    """silu(gate) * up where gate_up is [..., 2 * N] with gate first; one kernel on CUDA."""
    if not (TRITON_AVAILABLE and gate_up.is_cuda and gate_up.is_contiguous()):
        return silu_mul_reference(gate_up)
    width = gate_up.shape[-1] // 2
    rows = gate_up.numel() // (2 * width)
    out = torch.empty(*gate_up.shape[:-1], width, device=gate_up.device, dtype=gate_up.dtype)
    block = 1024
    _silu_mul_kernel[(rows, -(-width // block))](gate_up, out, width, BLOCK=block, num_warps=4)
    return out


def qkv_norm_rotary_write(qkv, norm_weight, cosine, signed_sine, slot_mapping, key_cache, value_cache, n_query_heads, n_kv_heads, head_dim, eps):
    """Q/K normalization + rotary and the KV-cache write for one decode token per row, in one kernel launch on CUDA.

    Replaces about nine small launches per layer (rms_norm, weight multiply, rotary muls/cat/addcmul, two index_copy_).
    Falls back to `qkv_norm_rotary_write_reference` when Triton/CUDA, contiguity, dtype or head size do not allow it.
    """
    usable = (TRITON_AVAILABLE and qkv.is_cuda and qkv.is_contiguous() and key_cache.is_contiguous() and value_cache.is_contiguous()
              and head_dim >= 2 and head_dim & (head_dim - 1) == 0 and cosine.dtype == qkv.dtype and signed_sine.dtype == qkv.dtype
              and key_cache.dtype == qkv.dtype and (norm_weight is None or norm_weight.dtype == qkv.dtype))
    if not usable:
        return qkv_norm_rotary_write_reference(qkv, norm_weight, cosine, signed_sine, slot_mapping, key_cache, value_cache, n_query_heads, n_kv_heads, head_dim, eps)
    rows = qkv.shape[0]
    q = torch.empty(rows, n_query_heads, head_dim, device=qkv.device, dtype=qkv.dtype)
    _qkv_norm_rotary_write_kernel[(rows, n_query_heads + 2 * n_kv_heads)](
        qkv, qkv if norm_weight is None else norm_weight.contiguous(), cosine.contiguous(), signed_sine.contiguous(), slot_mapping, q, key_cache, value_cache, eps,
        N_Q=n_query_heads, N_KV=n_kv_heads, HEAD_DIM=head_dim, HAS_NORM=norm_weight is not None, WRITE_CACHE=True, num_warps=1)
    return q


def qk_norm_rotary(qkv, norm_weight, cosine, signed_sine, n_query_heads, n_kv_heads, head_dim, eps):
    """Prefill/any-length twin of `qkv_norm_rotary_write` without the cache write: one kernel replaces rms_norm, the weight
    multiply, the rotary multiplies, the half-swap concatenate and the addcmul (about a quarter of prefill GPU time at batch 8)."""
    usable = (TRITON_AVAILABLE and qkv.is_cuda and qkv.is_contiguous() and head_dim >= 2 and head_dim & (head_dim - 1) == 0
              and cosine.dtype == qkv.dtype and signed_sine.dtype == qkv.dtype and (norm_weight is None or norm_weight.dtype == qkv.dtype))
    if not usable:
        return qk_norm_rotary_reference(qkv, norm_weight, cosine, signed_sine, n_query_heads, n_kv_heads, head_dim, eps)
    batch, seq = qkv.shape[:2]
    rows = batch * seq
    n_qk = n_query_heads + n_kv_heads
    out = torch.empty(batch, seq, n_qk, head_dim, device=qkv.device, dtype=qkv.dtype)
    cos = cosine.reshape(rows, head_dim)
    sin = signed_sine.reshape(rows, head_dim)
    _qkv_norm_rotary_write_kernel[(rows, n_qk)](
        qkv, qkv if norm_weight is None else norm_weight.contiguous(), cos, sin, cos, out, out, out, eps,
        N_Q=n_query_heads, N_KV=n_kv_heads, HEAD_DIM=head_dim, HAS_NORM=norm_weight is not None, WRITE_CACHE=False, num_warps=1)
    return out
