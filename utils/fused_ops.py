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
