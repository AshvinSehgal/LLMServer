"""Decode attention that reads the paged KV cache through on-device block tables.

Every function here takes only tensors (no Python lengths), so the whole decode step can be captured in a CUDA graph.

* paged_decode_attention_reference: plain PyTorch, any device. Gathers every block of every sequence, so it is slow
  and memory hungry; it is the correctness oracle and the fallback off CUDA.
* paged_decode_attention_split_reference: PyTorch re-implementation of the Triton kernel's exact split-KV algorithm,
  used to test the partial/combine math on machines without Triton.
* paged_decode_attention_triton: split-KV ("flash-decoding") Triton kernels. Each program handles one KV head's group of
  query heads for one slice of the context, then a second kernel merges the slices with a log-sum-exp reduction.
"""
import os
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:  # Triton is only installed with CUDA builds of PyTorch
    triton = None
    tl = None
    TRITON_AVAILABLE = False

DEFAULT_NUM_SPLITS = 32


def next_power_of_two(value):
    return 1 << max(0, (value - 1)).bit_length()


def choose_tile(group_pad):
    """Tokens processed per loop iteration; keeps the [group, tile, head_dim] product small enough for registers."""
    return 64 if group_pad <= 2 else 32 if group_pad <= 4 else 16


def default_tokens_per_split(num_blocks, block_size, num_splits, tile):
    """Smallest slice length (a multiple of tile) such that num_splits slices cover the whole cache."""
    per_split = -(-(num_blocks * block_size) // num_splits)
    return -(-per_split // tile) * tile


def paged_decode_attention_reference(query, key_cache, value_cache, block_tables, seq_lens, scale):
    """query [B, Hq, D]; key/value_cache [num_blocks, block_size, Hkv, D]; block_tables [B, max_blocks]; seq_lens [B]."""
    batch, n_query_heads, head_dim = query.shape
    _, block_size, n_kv_heads, _ = key_cache.shape
    group = n_query_heads // n_kv_heads
    max_tokens = block_tables.shape[1] * block_size
    tables = block_tables.long()
    keys = key_cache[tables].reshape(batch, max_tokens, n_kv_heads, head_dim)
    values = value_cache[tables].reshape(batch, max_tokens, n_kv_heads, head_dim)
    valid = torch.arange(max_tokens, device=query.device)[None, :] < seq_lens.long()[:, None]
    grouped_query = query.reshape(batch, n_kv_heads, group, head_dim).float()
    scores = torch.einsum("bhgd,bthd->bhgt", grouped_query, keys.float()) * scale
    scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    output = torch.einsum("bhgt,bthd->bhgd", weights, values.float())
    return output.reshape(batch, n_query_heads, head_dim).to(query.dtype)


def paged_decode_attention_split_reference(query, key_cache, value_cache, block_tables, seq_lens, scale, num_splits, tokens_per_split, tile):
    """The Triton kernels' algorithm in PyTorch: per-slice online softmax over tiles, then a log-sum-exp merge."""
    batch, n_query_heads, head_dim = query.shape
    _, block_size, n_kv_heads, _ = key_cache.shape
    group = n_query_heads // n_kv_heads
    output = torch.zeros(batch, n_query_heads, head_dim, dtype=torch.float32)
    for b in range(batch):
        length = int(seq_lens[b])
        for kv_head in range(n_kv_heads):
            q = query[b, kv_head * group:(kv_head + 1) * group].float()
            partial_max, partial_sum, partial_acc = [], [], []
            for split in range(num_splits):
                start = split * tokens_per_split
                end = min(start + tokens_per_split, length)
                m = torch.full((group,), float("-inf"))
                l = torch.zeros(group)
                acc = torch.zeros(group, head_dim)
                for t0 in range(start, end, tile):
                    tokens = torch.arange(t0, min(t0 + tile, end))
                    physical = block_tables[b, tokens // block_size].long()
                    k = key_cache[physical, tokens % block_size, kv_head].float()
                    v = value_cache[physical, tokens % block_size, kv_head].float()
                    scores = (q @ k.T) * scale
                    m_new = torch.maximum(m, scores.max(dim=1).values)
                    alpha = torch.exp(m - m_new)
                    p = torch.exp(scores - m_new[:, None])
                    l = l * alpha + p.sum(dim=1)
                    acc = acc * alpha[:, None] + p @ v
                    m = m_new
                partial_max.append(m)
                partial_sum.append(l)
                partial_acc.append(acc)
            m_all = torch.stack(partial_max).max(dim=0).values
            weights = torch.exp(torch.stack(partial_max) - m_all)
            denominator = (weights * torch.stack(partial_sum)).sum(dim=0)
            merged = (torch.stack(partial_acc) * weights[:, :, None]).sum(dim=0) / denominator[:, None]
            output[b, kv_head * group:(kv_head + 1) * group] = merged
    return output.to(query.dtype)


if TRITON_AVAILABLE:
    @triton.jit
    def _paged_decode_partial_kernel(
        q_ptr, k_ptr, v_ptr, block_tables_ptr, seq_lens_ptr,
        partial_acc_ptr, partial_max_ptr, partial_sum_ptr,
        scale,
        stride_qb, stride_qh,
        stride_kb, stride_kt, stride_kh,
        stride_bt,
        BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr,
        NUM_Q_HEADS: tl.constexpr, GROUP: tl.constexpr, GROUP_PAD: tl.constexpr,
        TILE: tl.constexpr, TOKENS_PER_SPLIT: tl.constexpr, NUM_SPLITS: tl.constexpr,
    ):
        b = tl.program_id(0)
        kv_head = tl.program_id(1)
        split = tl.program_id(2)
        seq_len = tl.load(seq_lens_ptr + b)
        start = split * TOKENS_PER_SPLIT
        end = tl.minimum(start + TOKENS_PER_SPLIT, seq_len)

        rows = tl.arange(0, GROUP_PAD)
        row_mask = rows < GROUP
        q_head = kv_head * GROUP + rows
        d = tl.arange(0, HEAD_DIM)
        q = tl.load(q_ptr + b * stride_qb + q_head[:, None] * stride_qh + d[None, :], mask=row_mask[:, None], other=0.0).to(tl.float32)

        m_i = tl.full([GROUP_PAD], float("-inf"), tl.float32)
        l_i = tl.zeros([GROUP_PAD], tl.float32)
        acc = tl.zeros([GROUP_PAD, HEAD_DIM], tl.float32)
        for t0 in range(start, end, TILE):
            tok = t0 + tl.arange(0, TILE)
            valid = tok < end
            physical = tl.load(block_tables_ptr + b * stride_bt + tok // BLOCK_SIZE, mask=valid, other=0).to(tl.int64)
            kv_offset = physical * stride_kb + (tok % BLOCK_SIZE) * stride_kt + kv_head * stride_kh
            k = tl.load(k_ptr + kv_offset[:, None] + d[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
            scores = tl.sum(q[:, None, :] * k[None, :, :], axis=2) * scale
            scores = tl.where(valid[None, :], scores, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            v = tl.load(v_ptr + kv_offset[:, None] + d[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
            acc = acc * alpha[:, None] + tl.sum(p[:, :, None] * v[None, :, :], axis=1)
            m_i = m_new

        slot = (b * NUM_Q_HEADS + q_head) * NUM_SPLITS + split
        tl.store(partial_acc_ptr + slot[:, None] * HEAD_DIM + d[None, :], acc, mask=row_mask[:, None])
        tl.store(partial_max_ptr + slot, m_i, mask=row_mask)
        tl.store(partial_sum_ptr + slot, l_i, mask=row_mask)

    @triton.jit
    def _paged_decode_partial_dot_kernel(
        q_ptr, k_ptr, v_ptr, block_tables_ptr, seq_lens_ptr,
        partial_acc_ptr, partial_max_ptr, partial_sum_ptr,
        scale,
        stride_qb, stride_qh,
        stride_kb, stride_kt, stride_kh,
        stride_bt,
        BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr,
        NUM_Q_HEADS: tl.constexpr, GROUP: tl.constexpr, GROUP_PAD: tl.constexpr,
        TILE: tl.constexpr, TOKENS_PER_SPLIT: tl.constexpr, NUM_SPLITS: tl.constexpr,
    ):
        """Same split-KV algorithm as _paged_decode_partial_kernel, but Q.K^T and P.V are tl.dot (tensor cores) on the
        cache's native dtype instead of fp32 broadcast-multiply-reduce, which spills registers at head_dim 128. The group
        of query heads is padded to the 16 rows tl.dot needs; decode is memory bound so the wasted rows are free."""
        b = tl.program_id(0)
        kv_head = tl.program_id(1)
        split = tl.program_id(2)
        seq_len = tl.load(seq_lens_ptr + b)
        start = split * TOKENS_PER_SPLIT
        end = tl.minimum(start + TOKENS_PER_SPLIT, seq_len)

        rows = tl.arange(0, GROUP_PAD)
        row_mask = rows < GROUP
        q_head = kv_head * GROUP + rows
        d = tl.arange(0, HEAD_DIM)
        q = tl.load(q_ptr + b * stride_qb + q_head[:, None] * stride_qh + d[None, :], mask=row_mask[:, None], other=0.0)

        m_i = tl.full([GROUP_PAD], float("-inf"), tl.float32)
        l_i = tl.zeros([GROUP_PAD], tl.float32)
        acc = tl.zeros([GROUP_PAD, HEAD_DIM], tl.float32)
        for t0 in range(start, end, TILE):
            tok = t0 + tl.arange(0, TILE)
            valid = tok < end
            physical = tl.load(block_tables_ptr + b * stride_bt + tok // BLOCK_SIZE, mask=valid, other=0).to(tl.int64)
            kv_offset = physical * stride_kb + (tok % BLOCK_SIZE) * stride_kt + kv_head * stride_kh
            k = tl.load(k_ptr + kv_offset[:, None] + d[None, :], mask=valid[:, None], other=0.0)
            scores = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
            scores = tl.where(valid[None, :], scores, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            v = tl.load(v_ptr + kv_offset[:, None] + d[None, :], mask=valid[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision="ieee")
            m_i = m_new

        slot = (b * NUM_Q_HEADS + q_head) * NUM_SPLITS + split
        tl.store(partial_acc_ptr + slot[:, None] * HEAD_DIM + d[None, :], acc, mask=row_mask[:, None])
        tl.store(partial_max_ptr + slot, m_i, mask=row_mask)
        tl.store(partial_sum_ptr + slot, l_i, mask=row_mask)

    @triton.jit
    def _paged_decode_reduce_kernel(
        partial_acc_ptr, partial_max_ptr, partial_sum_ptr, out_ptr,
        stride_ob, stride_oh,
        NUM_Q_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr, NUM_SPLITS: tl.constexpr,
    ):
        b = tl.program_id(0)
        h = tl.program_id(1)
        s = tl.arange(0, NUM_SPLITS)
        d = tl.arange(0, HEAD_DIM)
        slot = (b * NUM_Q_HEADS + h) * NUM_SPLITS + s
        m = tl.load(partial_max_ptr + slot)
        l = tl.load(partial_sum_ptr + slot)
        weights = tl.exp(m - tl.max(m, axis=0))
        denominator = tl.sum(weights * l, axis=0)
        acc = tl.load(partial_acc_ptr + slot[:, None] * HEAD_DIM + d[None, :])
        out = tl.sum(acc * weights[:, None], axis=0) / denominator
        tl.store(out_ptr + b * stride_ob + h * stride_oh + d, out.to(out_ptr.dtype.element_ty))


DOT_TILE = 64
DOT_MIN_GROUP_ROWS = 16


def attention_kernel_variant():
    """"dot" (tensor-core kernel, default) or "broadcast" (fp32 broadcast-reduce kernel); LLMSERVER_ATTENTION_KERNEL overrides."""
    return os.environ.get("LLMSERVER_ATTENTION_KERNEL", "dot")


def paged_decode_attention_triton(query, key_cache, value_cache, block_tables, seq_lens, scale, num_splits=DEFAULT_NUM_SPLITS, tokens_per_split=None,
                                  variant=None, tile=None, num_warps=4, num_stages=None):
    batch, n_query_heads, head_dim = query.shape
    num_blocks, block_size, n_kv_heads, _ = key_cache.shape
    if key_cache.stride() != value_cache.stride():
        raise ValueError("Key and value caches must share a layout")
    if key_cache.stride(3) != 1 or query.stride(2) != 1:
        raise ValueError("Head dimension must be contiguous")
    if head_dim & (head_dim - 1) or num_splits & (num_splits - 1):
        raise ValueError(f"head_dim and num_splits must be powers of two: {head_dim}, {num_splits}")
    group = n_query_heads // n_kv_heads
    variant = variant or attention_kernel_variant()
    if head_dim < 16:
        variant = "broadcast"  # tl.dot needs every dimension >= 16
    if variant == "dot":
        group_pad = max(DOT_MIN_GROUP_ROWS, next_power_of_two(group))
        tile = tile or DOT_TILE
        partial_kernel = _paged_decode_partial_dot_kernel
    else:
        group_pad = max(2, next_power_of_two(group))
        tile = tile or choose_tile(group_pad)
        partial_kernel = _paged_decode_partial_kernel
    if tokens_per_split is None:
        tokens_per_split = default_tokens_per_split(num_blocks, block_size, num_splits, tile)
    partial_acc = torch.empty(batch, n_query_heads, num_splits, head_dim, device=query.device, dtype=torch.float32)
    partial_max = torch.empty(batch, n_query_heads, num_splits, device=query.device, dtype=torch.float32)
    partial_sum = torch.empty_like(partial_max)
    output = torch.empty_like(query)
    launch_options = {} if num_stages is None else {"num_stages": num_stages}
    partial_kernel[(batch, n_kv_heads, num_splits)](
        query, key_cache, value_cache, block_tables, seq_lens,
        partial_acc, partial_max, partial_sum,
        scale,
        query.stride(0), query.stride(1),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        block_tables.stride(0),
        BLOCK_SIZE=block_size, HEAD_DIM=head_dim, NUM_Q_HEADS=n_query_heads, GROUP=group, GROUP_PAD=group_pad,
        TILE=tile, TOKENS_PER_SPLIT=tokens_per_split, NUM_SPLITS=num_splits,
        num_warps=num_warps, **launch_options,
    )
    _paged_decode_reduce_kernel[(batch, n_query_heads)](
        partial_acc, partial_max, partial_sum, output,
        output.stride(0), output.stride(1),
        NUM_Q_HEADS=n_query_heads, HEAD_DIM=head_dim, NUM_SPLITS=num_splits,
        num_warps=4,
    )
    return output


def triton_attention_usable(device):
    return TRITON_AVAILABLE and torch.device(device).type == "cuda" and os.environ.get("LLMSERVER_PAGED_ATTENTION", "") != "reference"


def paged_decode_attention(query, key_cache, value_cache, block_tables, seq_lens, scale, num_splits=DEFAULT_NUM_SPLITS, tokens_per_split=None):
    if triton_attention_usable(query.device):
        return paged_decode_attention_triton(query, key_cache, value_cache, block_tables, seq_lens, scale, num_splits, tokens_per_split)
    return paged_decode_attention_reference(query, key_cache, value_cache, block_tables, seq_lens, scale)
