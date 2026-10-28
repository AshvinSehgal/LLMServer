import asyncio
import torch
import pytest
from utils.kv_cache import PagedKVCache
from utils.request import Request, RequestFinishReason, RequestStateError, RequestStatus

def test_request():
    output_queue = asyncio.Queue()
    request = Request("req-1", "Hello", [9707], 20, output_queue)
    assert request.request_id == "req-1"
    assert request.block_table == []
    assert request.prompt == "Hello"
    assert request.prompt_token_ids == [9707]
    assert request.generated_token_ids == []
    assert request.max_new_tokens == 20
    assert request.output_queue is output_queue
    assert request.num_computed_tokens == 0
    assert request.status == RequestStatus.WAITING
    assert request.finish_reason is None
    assert request.created_at is not None
    assert request.finished_at is None

def test_request_lifecycle():
    request = Request("req-1", "Hello", [10, 11], 20, asyncio.Queue())
    request.start_prefill()
    assert request.status == RequestStatus.PREFILL
    request.advance_prefill(1)
    assert request.num_computed_tokens == 1
    assert request.status == RequestStatus.PREFILL
    request.advance_prefill(1)
    request.complete_prefill()
    assert request.status == RequestStatus.DECODING
    assert request.num_computed_tokens == 2
    request.append_generated_token(50)
    assert request.generated_token_ids == [50]
    request.complete_decode()
    assert request.num_computed_tokens == 3
    request.finish(RequestFinishReason.EOS)
    assert request.status == RequestStatus.FINISHED
    assert request.finish_reason == RequestFinishReason.EOS
    assert request.finished_at is not None
    with pytest.raises(RequestStateError):
        request.finish(RequestFinishReason.LENGTH)

def test_request_rejects_invalid_transitions():
    request = Request("req-1", "Hello", [9707], 20, asyncio.Queue())
    with pytest.raises(RequestStateError):
        request.complete_prefill()
    with pytest.raises(RequestStateError):
        request.append_generated_token(50)
    with pytest.raises(RequestStateError):
        request.complete_decode()
    request.start_prefill()
    with pytest.raises(RequestStateError):
        request.complete_prefill()
    with pytest.raises(ValueError):
        request.advance_prefill(0)
    with pytest.raises(ValueError):
        request.advance_prefill(2)

def test_request_cancellation():
    request = Request("req-1", "Hello", [9707], 20, asyncio.Queue())
    request.cancel()
    assert request.status == RequestStatus.CANCELLED
    assert request.finish_reason == RequestFinishReason.CANCELLED
    assert request.finished_at is not None

def test_request_block_tables_are_independent_and_ordered():
    first_request = Request("req-1", "First", [10, 11], 20, asyncio.Queue())
    second_request = Request("req-2", "Second", [20, 21], 20, asyncio.Queue())
    assert first_request.block_table == []
    assert second_request.block_table == []
    assert first_request.block_table is not second_request.block_table
    first_request.block_table.extend([7, 2, 9])
    second_request.block_table.append(4)
    assert first_request.block_table == [7, 2, 9]
    assert second_request.block_table == [4]

def test_request_block_tables_track_allocated_physical_blocks():
    cache = PagedKVCache(num_layers=2, num_blocks=4, block_size=4, num_kv_heads=2, head_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    first_request = Request("req-1", "First", [10, 11], 20, asyncio.Queue())
    second_request = Request("req-2", "Second", [20, 21], 20, asyncio.Queue())
    first_request.block_table.append(cache.allocate_block())
    second_request.block_table.append(cache.allocate_block())
    first_request.block_table.append(cache.allocate_block())
    assert first_request.block_table == [0, 2]
    assert second_request.block_table == [1]
    assert set(first_request.block_table).isdisjoint(second_request.block_table)
    assert set(first_request.block_table + second_request.block_table) <= cache.allocated_blocks
