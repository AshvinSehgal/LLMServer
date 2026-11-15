"""Attention for a prefill chunk that resumes after already-computed context (a long prompt split into chunks, or a cached prefix).

The chunk's queries must see the whole prefix and, causally, the chunk itself. A single call needs a [chunk, context] mask, which
rules out the flash kernel (PyTorch's `is_causal` aligns the mask to the top-left corner, wrong when the queries are the last rows),
so the call falls back to a slower masked kernel and a dense mask. The two parts are instead computed separately and merged:

    prefix part: every query attends to the whole prefix            (no mask)
    chunk part:  every query attends to the chunk's own keys        (square causal)

Each part returns its output and the log-sum-exp of its scores, and softmax over the union is the weighted sum
out = (out_a * exp(lse_a - m) + out_b * exp(lse_b - m)) / (exp(lse_a - m) + exp(lse_b - m)), m = max(lse_a, lse_b).
"""
import warnings

import torch
import torch.nn.functional as F

MIN_CONTEXT_FOR_SPLIT = 4096  # below this the masked kernel is fast enough, and the plain path stays bit-identical to before
MIN_LENGTH_FOR_CUDNN = 2048   # the cuDNN kernel is about 1.8x faster than flash on H200 from 4096 tokens up, and no better at 1024
USE_CUDNN = True
USE_CUDNN_LSE = False  # off: cuDNN builds a plan per (query, key) shape, about 1 s each on first sight, and split chunks rarely repeat a shape
LENGTH_BUCKET = 1024   # fresh prompts are padded to a multiple of this so cuDNN sees few distinct shapes
_FLASH_BROKEN = False
_CUDNN_BROKEN = False      # the plain cuDNN call failed
_CUDNN_LSE_BROKEN = False  # the log-sum-exp operator failed (kept separate: the plain call may still work)


def _cudnn_usable(q):
    return USE_CUDNN and not _CUDNN_BROKEN and q.is_cuda and q.dtype in (torch.bfloat16, torch.float16) and q.shape[-1] % 8 == 0 and q.shape[-1] <= 256 \
        and q.shape[2] >= MIN_LENGTH_FOR_CUDNN and torch.cuda.get_device_capability(q.device)[0] >= 9


def causal_attention(q, new_keys, new_values, group_size, scale):
    """Causal attention over a fresh prompt: q [1, H, L, D], new_keys/values [KVH, L, D] -> [1, H, L, D].
    cuDNN's kernel (no native GQA, so the KV heads are expanded) on Hopper for long prompts, PyTorch's default dispatch otherwise."""
    global _CUDNN_BROKEN
    if _cudnn_usable(q):
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            expand = lambda tensor: tensor.repeat_interleave(group_size, dim=0).unsqueeze(0)
            length = q.shape[2]
            padded = -(-length // LENGTH_BUCKET) * LENGTH_BUCKET
            pad = lambda tensor: F.pad(tensor, (0, 0, 0, padded - length))  # causal: real queries never see the zero rows after them
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                return F.scaled_dot_product_attention(pad(q), pad(expand(new_keys)), pad(expand(new_values)), dropout_p=0.0, scale=scale, is_causal=True)[:, :, :length]
        except Exception as error:
            _CUDNN_BROKEN = True
            warnings.warn(f"cuDNN attention unavailable ({error!r}); using the default kernels")
    return F.scaled_dot_product_attention(q, new_keys.unsqueeze(0), new_values.unsqueeze(0), dropout_p=0.0, scale=scale, is_causal=True, enable_gqa=True)


def attention_with_lse(q, k, v, scale, causal):
    """q [B, H, Lq, D], k and v [B, H, Lk, D] -> (output like q, log-sum-exp float32 [B, H, Lq])."""
    global _FLASH_BROKEN, _CUDNN_LSE_BROKEN
    if USE_CUDNN_LSE and _cudnn_usable(q) and not _CUDNN_LSE_BROKEN:
        try:  # PyTorch 2.5.1: (q, k, v, attn_bias, compute_log_sumexp, dropout_p, is_causal, return_debug_mask, *, scale); older builds lack attn_bias
            try:
                result = torch.ops.aten._scaled_dot_product_cudnn_attention(q, k, v, None, True, 0.0, causal, False, scale=scale)
            except RuntimeError:
                result = torch.ops.aten._scaled_dot_product_cudnn_attention(q, k, v, True, 0.0, causal, False, scale=scale)
            output, lse = result[:2]
            if lse.numel() != q.shape[0] * q.shape[1] * q.shape[2]:
                raise RuntimeError(f"unexpected log-sum-exp shape {tuple(lse.shape)}")
            return output, lse.reshape(q.shape[:3]).float()
        except Exception as error:
            _CUDNN_LSE_BROKEN = True
            warnings.warn(f"cuDNN attention with log-sum-exp unavailable ({error!r}); using flash")
    if q.is_cuda and q.dtype in (torch.bfloat16, torch.float16) and q.shape[-1] % 8 == 0 and q.shape[-1] <= 256 and not _FLASH_BROKEN:
        try:
            output, lse = torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, 0.0, causal, False, scale=scale)[:2]
            return output, lse[..., :q.shape[2]].float()
        except Exception as error:  # an operator signature that differs between PyTorch versions: use the reference path from now on
            _FLASH_BROKEN = True
            warnings.warn(f"flash attention with log-sum-exp unavailable ({error!r}); using the reference implementation")
    scores = torch.matmul(q.float(), k.float().mT) * scale
    if causal:
        scores = scores.masked_fill(torch.ones(scores.shape[-2:], dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    return torch.matmul(torch.exp(scores - lse.unsqueeze(-1)), v.float()).to(q.dtype), lse


def merge_attention(first, first_lse, second, second_lse):
    peak = torch.maximum(first_lse, second_lse)
    first_weight = torch.exp(first_lse - peak)
    second_weight = torch.exp(second_lse - peak)
    total = (first.float() * first_weight.unsqueeze(-1) + second.float() * second_weight.unsqueeze(-1)) / (first_weight + second_weight).unsqueeze(-1)
    return total.to(first.dtype)


def chunk_attention(q, new_keys, new_values, prefix_keys, prefix_values, group_size, scale):
    """q [1, H, L, D]; new_keys/values [KVH, L, D] (this chunk); prefix_keys/values [KVH, P, D] (P > 0) -> [1, H, L, D]."""
    expand = lambda tensor: tensor.repeat_interleave(group_size, dim=0).unsqueeze(0)
    prefix_out, prefix_lse = attention_with_lse(q, expand(prefix_keys), expand(prefix_values), scale, causal=False)
    chunk_out, chunk_lse = attention_with_lse(q, expand(new_keys), expand(new_values), scale, causal=True)
    return merge_attention(prefix_out, prefix_lse, chunk_out, chunk_lse)
