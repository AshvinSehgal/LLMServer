import asyncio
import torch
import pytest
from utils.batch import DecodeBatch, PrefillBatch, RecomputeDecodeBatch
from utils.request import Request

def test_prefill_batch():
    requests = [Request("req-1", "A B", [10, 11], 20, asyncio.Queue()), Request("req-2", "C D E F", [20, 21, 22, 23], 20, asyncio.Queue())]
    batch = PrefillBatch(requests=requests, pad_token_id=99, device=torch.device("cpu"), dtype=torch.float32)
    mask_value = torch.finfo(torch.float32).min
    expected_attention_mask = torch.tensor([[[[0, mask_value, mask_value, mask_value], [0, 0, mask_value, mask_value], [0, 0, mask_value, mask_value], [0, 0, mask_value, mask_value]]], [[[0, mask_value, mask_value, mask_value], [0, 0, mask_value, mask_value], [0, 0, 0, mask_value], [0, 0, 0, 0]]]], dtype=torch.float32)
    assert batch.requests == requests
    assert batch.batch_size == 2
    assert batch.max_prompt_length == 4
    assert torch.equal(batch.prompt_lengths, torch.tensor([2, 4]))
    assert torch.equal(batch.last_token_indices, torch.tensor([1, 3]))
    assert torch.equal(batch.input_ids, torch.tensor([[10, 11, 99, 99], [20, 21, 22, 23]]))
    assert torch.equal(batch.padding_mask, torch.tensor([[True, True, False, False], [True, True, True, True]]))
    assert torch.equal(batch.pos_ids, torch.tensor([[0, 1, 0, 0], [0, 1, 2, 3]]))
    assert torch.equal(batch.attention_mask, expected_attention_mask)

def test_prefill_batch_rejects_empty_requests():
    with pytest.raises(ValueError):
        PrefillBatch(requests=[], pad_token_id=99, device=torch.device("cpu"), dtype=torch.float32)

def test_decode_batch():
    first_request = Request("req-1", "A B", [10, 11], 20, asyncio.Queue())
    second_request = Request("req-2", "C D E F", [20, 21, 22, 23], 20, asyncio.Queue())
    first_request.start_prefill()
    first_request.complete_prefill()
    first_request.append_generated_token(50)
    second_request.start_prefill()
    second_request.complete_prefill()
    second_request.append_generated_token(60)
    second_request.complete_decode()
    second_request.append_generated_token(61)
    batch = DecodeBatch(requests=[first_request, second_request], device=torch.device("cpu"), dtype=torch.float32)
    mask_value = torch.finfo(torch.float32).min
    expected_attention_mask = torch.tensor([[[[0, 0, 0, mask_value, mask_value, mask_value]]], [[[0, 0, 0, 0, 0, 0]]]], dtype=torch.float32)
    assert batch.requests == [first_request, second_request]
    assert batch.batch_size == 2
    assert torch.equal(batch.input_ids, torch.tensor([[50], [61]]))
    assert torch.equal(batch.pos_ids, torch.tensor([[2], [5]]))
    assert torch.equal(batch.padding_mask, torch.tensor([[True, True, True, False, False, False], [True, True, True, True, True, True]]))
    assert torch.equal(batch.attention_mask, expected_attention_mask)

def test_decode_batch_rejects_request_without_generated_tokens():
    request = Request("req-1", "A B", [10, 11], 20, asyncio.Queue())
    with pytest.raises(ValueError):
        DecodeBatch(requests=[request], device=torch.device("cpu"), dtype=torch.float32)

def test_recompute_decode_batch():
    first_request = Request("req-1", "A B", [10, 11], 20, asyncio.Queue())
    second_request = Request("req-2", "C D E F", [20, 21, 22, 23], 20, asyncio.Queue())
    first_request.start_prefill()
    first_request.complete_prefill()
    first_request.append_generated_token(50)
    second_request.start_prefill()
    second_request.complete_prefill()
    second_request.append_generated_token(60)
    second_request.complete_decode()
    second_request.append_generated_token(61)
    batch = RecomputeDecodeBatch(requests=[first_request, second_request], pad_token_id=99, device=torch.device("cpu"), dtype=torch.float32)
    mask_value = torch.finfo(torch.float32).min
    assert batch.requests == [first_request, second_request]
    assert batch.batch_size == 2
    assert batch.max_seq_length == 6
    assert torch.equal(batch.seq_lengths, torch.tensor([3, 6]))
    assert torch.equal(batch.last_token_indices, torch.tensor([2, 5]))
    assert torch.equal(batch.input_ids, torch.tensor([[10, 11, 50, 99, 99, 99], [20, 21, 22, 23, 60, 61]]))
    assert torch.equal(batch.padding_mask, torch.tensor([[True, True, True, False, False, False], [True, True, True, True, True, True]]))
    assert torch.equal(batch.pos_ids, torch.tensor([[0, 1, 2, 0, 0, 0], [0, 1, 2, 3, 4, 5]]))
    assert batch.attention_mask.shape == (2, 1, 6, 6)
    assert torch.equal(batch.attention_mask[0, 0, 2], torch.tensor([0, 0, 0, mask_value, mask_value, mask_value]))
    assert torch.equal(batch.attention_mask[0, 0, 5], torch.tensor([0, 0, 0, mask_value, mask_value, mask_value]))
    assert torch.equal(batch.attention_mask[1, 0, 2], torch.tensor([0, 0, 0, mask_value, mask_value, mask_value]))
    assert torch.equal(batch.attention_mask[1, 0, 5], torch.tensor([0, 0, 0, 0, 0, 0]))

def test_recompute_decode_batch_rejects_request_not_ready_for_decode():
    request = Request("req-1", "A B", [10, 11], 20, asyncio.Queue())
    with pytest.raises(ValueError):
        RecomputeDecodeBatch(requests=[request], pad_token_id=99, device=torch.device("cpu"), dtype=torch.float32)
