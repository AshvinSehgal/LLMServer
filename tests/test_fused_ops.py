import pytest
import torch
import torch.nn.functional as F
from utils.fused_ops import add_rms_norm, add_rms_norm_reference, silu_mul, silu_mul_reference, TRITON_AVAILABLE

cuda_only = pytest.mark.skipif(not (torch.cuda.is_available() and TRITON_AVAILABLE), reason="CUDA + Triton required")


def test_reference_add_rms_norm_matches_unfused_ops():
    torch.manual_seed(0)
    hidden, delta, weight = torch.randn(3, 1, 64), torch.randn(3, 1, 64), torch.rand(64) + 0.5
    total, normed = add_rms_norm_reference(hidden, delta, weight, 1e-6)
    assert torch.equal(total, hidden + delta)
    assert torch.allclose(normed, F.rms_norm(hidden + delta, (64,), weight, 1e-6))
    same, plain = add_rms_norm_reference(hidden, None, weight, 1e-6)
    assert same is hidden and torch.allclose(plain, F.rms_norm(hidden, (64,), weight, 1e-6))


def test_silu_mul_reference_matches_unfused_ops():
    gate_up = torch.randn(2, 1, 32)
    gate, up = gate_up.chunk(2, dim=-1)
    assert torch.allclose(silu_mul_reference(gate_up), F.silu(gate) * up)


def test_fused_decode_dataflow_matches_unfused_model():
    from utils.model import Model
    torch.manual_seed(0)
    kwargs = dict(vocab_size=50, hidden_size=32, intermediate_size=64, n_hidden_layers=3, head_dim=8, rope_theta=10000.0, n_query_heads=4, n_kv_heads=2, bias=False, attention_scaling=8 ** -0.5, rms_norm_eps=1e-6)
    model = Model(**kwargs).eval()
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    model.fuse_projections()
    from utils.kv_cache import PagedKVCache
    cache = PagedKVCache(num_layers=3, num_blocks=8, block_size=4, num_kv_heads=2, head_dim=8, device="cpu", dtype=torch.float32)
    table = [cache.allocate_block(), cache.allocate_block()]
    ids, pos = torch.tensor([[7]]), torch.tensor([[5]])
    slot = torch.tensor([table[1] * 4 + 1])
    tables = torch.tensor([table + [0] * 2], dtype=torch.int32)
    lens = torch.tensor([6], dtype=torch.int32)
    outputs = []
    for fused in (False, True):
        cache.key_cache.zero_(); cache.value_cache.zero_()
        model.fused_decode_ops = fused
        with torch.no_grad():
            outputs.append(model.forward_decode_paged(ids, pos, slot, tables, lens, cache))
    assert torch.allclose(outputs[0], outputs[1], atol=1e-5)


@cuda_only
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows,width", [(1, 2048), (7, 2048), (32, 1024), (3, 96)])
def test_triton_add_rms_norm_matches_reference(dtype, rows, width):
    torch.manual_seed(0)
    hidden = torch.randn(rows, 1, width, device="cuda", dtype=dtype)
    delta = torch.randn_like(hidden)
    weight = (torch.rand(width, device="cuda") + 0.5).to(dtype)
    total, normed = add_rms_norm(hidden, delta, weight, 1e-6)
    ref_total, ref_normed = add_rms_norm_reference(hidden, delta, weight, 1e-6)
    assert torch.equal(total, ref_total)
    assert torch.allclose(normed.float(), ref_normed.float(), atol=2e-2, rtol=2e-2)
    _, plain = add_rms_norm(hidden, None, weight, 1e-6)
    assert torch.allclose(plain.float(), F.rms_norm(hidden, (width,), weight, 1e-6).float(), atol=2e-2, rtol=2e-2)


@cuda_only
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows,width", [(1, 6144), (16, 6144), (5, 1000)])
def test_triton_silu_mul_matches_reference(dtype, rows, width):
    torch.manual_seed(0)
    gate_up = torch.randn(rows, 1, 2 * width, device="cuda", dtype=dtype)
    assert torch.allclose(silu_mul(gate_up).float(), silu_mul_reference(gate_up).float(), atol=2e-2, rtol=2e-2)


def _qkv_case(dtype, qk_norm, device="cpu", rows=3, n_q=4, n_kv=2, head_dim=16):
    from utils.attention import SelfAttention
    from utils.kv_cache import PagedKVCache
    torch.manual_seed(1)
    attention = SelfAttention(32, n_q, n_kv, head_dim, bias=False, attention_scaling=head_dim ** -0.5, rms_norm_eps=1e-6, qk_norm=qk_norm).to(device=device, dtype=dtype).eval()
    for parameter in attention.parameters():
        torch.nn.init.normal_(parameter, std=0.3)
    attention.fuse_projections()
    hidden = torch.randn(rows, 1, 32, device=device, dtype=dtype)
    angle = torch.rand(rows, 1, head_dim // 2, device=device) * 3
    cosine = torch.cat([angle.cos()] * 2, dim=-1).to(dtype)
    sine = torch.cat([angle.sin()] * 2, dim=-1).to(dtype)
    signed = torch.cat([-sine[..., :head_dim // 2], sine[..., head_dim // 2:]], dim=-1)
    cache = PagedKVCache(num_layers=1, num_blocks=6, block_size=4, num_kv_heads=n_kv, head_dim=head_dim, device=device, dtype=dtype)
    slots = torch.tensor([21, 3, 10][:rows], device=device)
    return attention, hidden, (cosine, sine, signed), cache, slots


@pytest.mark.parametrize("qk_norm", [True, False])
def test_qkv_write_reference_matches_project_qkv_path(qk_norm):
    from utils.fused_ops import qkv_norm_rotary_write_reference
    attention, hidden, pos, cache, slots = _qkv_case(torch.float32, qk_norm)
    with torch.no_grad():
        Q, K, V = attention.project_qkv(hidden, pos)
        cache.write_slots(0, slots, K[:, :, 0, :], V[:, :, 0, :])
        expected_keys, expected_values = cache.key_cache.clone(), cache.value_cache.clone()
        cache.key_cache.zero_(); cache.value_cache.zero_()
        qkv = attention.qkv_proj(hidden)
        weight = attention.qk_norm_weight(qkv) if qk_norm else None
        query = qkv_norm_rotary_write_reference(qkv, weight, pos[0], pos[2], slots, cache.key_cache[0], cache.value_cache[0], 4, 2, 16, 1e-6)
    assert torch.allclose(query, Q[:, :, 0, :], atol=1e-6)
    assert torch.allclose(cache.key_cache, expected_keys, atol=1e-6) and torch.allclose(cache.value_cache, expected_values)


@cuda_only
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("qk_norm", [True, False])
def test_qkv_write_kernel_matches_reference(dtype, qk_norm):
    from utils.fused_ops import qkv_norm_rotary_write, qkv_norm_rotary_write_reference
    attention, hidden, pos, cache, slots = _qkv_case(dtype, qk_norm, device="cuda")
    with torch.no_grad():
        qkv = attention.qkv_proj(hidden)
        weight = attention.qk_norm_weight(qkv) if qk_norm else None
        expected = qkv_norm_rotary_write_reference(qkv, weight, pos[0], pos[2], slots, cache.key_cache[0], cache.value_cache[0], 4, 2, 16, 1e-6)
        expected_keys, expected_values = cache.key_cache.clone(), cache.value_cache.clone()
        cache.key_cache.zero_(); cache.value_cache.zero_()
        got = qkv_norm_rotary_write(qkv, weight, pos[0], pos[2], slots, cache.key_cache[0], cache.value_cache[0], 4, 2, 16, 1e-6)
    tolerance = 1e-2 if dtype == torch.bfloat16 else 2e-3
    assert torch.allclose(got.float(), expected.float(), atol=tolerance)
    assert torch.allclose(cache.key_cache.float(), expected_keys.float(), atol=tolerance)
    assert torch.equal(cache.value_cache, expected_values)
