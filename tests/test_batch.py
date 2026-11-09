import asyncio
import torch
import pytest
from utils.batch import DecodeBatch, PrefillBatch, PackedPrefillBatch
from utils.request import Request

def test_prefill_batch():
    requests = [Request("req-1", "A B", [10, 11], 20, asyncio.Queue()), Request("req-2", "C D E F", [20, 21, 22, 23], 20, asyncio.Queue())]
    batch = PrefillBatch(requests=requests, pad_token_id=99, device=torch.device("cpu"), dtype=torch.float32)
    mask_value = torch.finfo(torch.float32).min
    expected_attention_mask = torch.tensor([[[[0, mask_value, mask_value, mask_value], [0, 0, mask_value, mask_value], [0, 0, mask_value, mask_value], [0, 0, mask_value, mask_value]]], [[[0, mask_value, mask_value, mask_value], [0, 0, mask_value, mask_value], [0, 0, 0, mask_value], [0, 0, 0, 0]]]], dtype=torch.float32)
    assert batch.requests == requests
    assert batch.request_ids == ["req-1", "req-2"]
    assert batch.block_tables[0] is requests[0].block_table
    assert batch.block_tables[1] is requests[1].block_table
    assert batch.start_pos == [0, 0]
    assert batch.input_lens == [2, 4]
    assert batch.batch_size == 2
    assert batch.max_prompt_length == 4
    assert torch.equal(batch.prompt_lengths, torch.tensor([2, 4]))
    assert torch.equal(batch.last_token_indices, torch.tensor([1, 3]))
    assert torch.equal(batch.input_ids, torch.tensor([[10, 11, 99, 99], [20, 21, 22, 23]]))
    assert torch.equal(batch.padding_mask, torch.tensor([[True, True, False, False], [True, True, True, True]]))
    assert torch.equal(batch.pos_ids, torch.tensor([[0, 1, 0, 0], [0, 1, 2, 3]]))
    assert torch.equal(batch.attention_mask, expected_attention_mask)

def test_packed_prefill_batch_blocks_cross_request_attention():
    requests = [Request("req-1", "A B", [10, 11], 20, asyncio.Queue()), Request("req-2", "C D E", [20, 21, 22], 20, asyncio.Queue())]
    batch = PackedPrefillBatch(requests, torch.device("cpu"), torch.float32, [2, 3])
    mask_value = torch.finfo(torch.float32).min
    assert batch.packed
    assert batch.input_lens == [2, 3]
    assert torch.equal(batch.input_ids, torch.tensor([[10, 11, 20, 21, 22]]))
    assert torch.equal(batch.pos_ids, torch.tensor([[0, 1, 0, 1, 2]]))
    assert torch.equal(batch.last_token_indices, torch.tensor([1, 4]))
    assert torch.equal(batch.attention_mask[0, 0], torch.tensor([[0, mask_value, mask_value, mask_value, mask_value], [0, 0, mask_value, mask_value, mask_value], [mask_value, mask_value, 0, mask_value, mask_value], [mask_value, mask_value, 0, 0, mask_value], [mask_value, mask_value, 0, 0, 0]]))

def test_prefill_batch_rejects_empty_requests():
    with pytest.raises(ValueError):
        PrefillBatch(requests=[], pad_token_id=99, device=torch.device("cpu"), dtype=torch.float32)

def test_prefill_batch_uses_chunk_slices_and_absolute_positions():
    first_request = Request("req-1", "", [10, 11, 12, 13, 14], 20, asyncio.Queue())
    second_request = Request("req-2", "", [20, 21, 22], 20, asyncio.Queue())
    first_request.start_prefill()
    first_request.advance_prefill(3)
    second_request.start_prefill()
    second_request.advance_prefill(1)
    batch = PrefillBatch([first_request, second_request], pad_token_id=99, device=torch.device("cpu"), dtype=torch.float32, chunk_lengths=[2, 1])
    mask_value = torch.finfo(torch.float32).min
    assert batch.start_pos == [3, 1]
    assert batch.end_pos == [5, 2]
    assert batch.input_lens == [2, 1]
    assert torch.equal(batch.input_ids, torch.tensor([[13, 14], [21, 99]]))
    assert torch.equal(batch.pos_ids, torch.tensor([[3, 4], [1, 0]]))
    assert torch.equal(batch.last_token_indices, torch.tensor([1, 0]))
    assert torch.equal(batch.attention_mask[0, 0], torch.tensor([[0, 0, 0, 0, mask_value], [0, 0, 0, 0, 0]]))
    assert torch.equal(batch.attention_mask[1, 0], torch.tensor([[0, 0, mask_value, mask_value, mask_value], [0, 0, mask_value, mask_value, mask_value]]))

def test_prefill_batch_rejects_invalid_chunk_lengths():
    request = Request("req-1", "", [10, 11], 20, asyncio.Queue())
    with pytest.raises(ValueError):
        PrefillBatch([request], pad_token_id=99, device=torch.device("cpu"), dtype=torch.float32, chunk_lengths=[])
    with pytest.raises(ValueError):
        PrefillBatch([request], pad_token_id=99, device=torch.device("cpu"), dtype=torch.float32, chunk_lengths=[0])
    with pytest.raises(ValueError):
        PrefillBatch([request], pad_token_id=99, device=torch.device("cpu"), dtype=torch.float32, chunk_lengths=[3])

def test_decode_batch():
    first_request = Request("req-1", "A B", [10, 11], 20, asyncio.Queue())
    second_request = Request("req-2", "C D E F", [20, 21, 22, 23], 20, asyncio.Queue())
    first_request.start_prefill()
    first_request.advance_prefill(2)
    first_request.complete_prefill()
    first_request.append_generated_token(50)
    second_request.start_prefill()
    second_request.advance_prefill(4)
    second_request.complete_prefill()
    second_request.append_generated_token(60)
    second_request.complete_decode()
    second_request.append_generated_token(61)
    batch = DecodeBatch(requests=[first_request, second_request], device=torch.device("cpu"), dtype=torch.float32)
    assert batch.requests == [first_request, second_request]
    assert batch.request_ids == ["req-1", "req-2"]
    assert batch.block_tables[0] is first_request.block_table
    assert batch.block_tables[1] is second_request.block_table
    assert batch.start_pos == [2, 5]
    assert batch.input_lens == [1, 1]
    assert batch.batch_size == 2
    assert torch.equal(batch.last_token_indices, torch.tensor([0, 0]))
    assert torch.equal(batch.input_ids, torch.tensor([[50], [61]]))
    assert torch.equal(batch.pos_ids, torch.tensor([[2], [5]]))
    assert batch.attention_mask is None

def test_decode_batch_rejects_request_without_generated_tokens():
    request = Request("req-1", "A B", [10, 11], 20, asyncio.Queue())
    with pytest.raises(ValueError):
        DecodeBatch(requests=[request], device=torch.device("cpu"), dtype=torch.float32)
