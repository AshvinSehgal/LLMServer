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

MIN_CONTEXT_FOR_SPLIT = 4096  # below this the masked kernel is fast enough, and the plain path stays bit-identical to before
_FLASH_BROKEN = False


def attention_with_lse(q, k, v, scale, causal):
    """q [B, H, Lq, D], k and v [B, H, Lk, D] -> (output like q, log-sum-exp float32 [B, H, Lq])."""
    global _FLASH_BROKEN
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
