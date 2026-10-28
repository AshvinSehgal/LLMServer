import asyncio
import pytest
from utils.request import Request, RequestStatus
from utils.scheduler import Scheduler, SchedulerOutput

def create_request(request_id, prompt_token_ids=None):
    return Request(request_id, "Hello", prompt_token_ids or [9707], 20, asyncio.Queue())

def test_scheduler():
    scheduler = Scheduler(max_batch_size=2, max_num_batched_tokens=2)
    requests = [create_request(f"req-{index}") for index in range(3)]
    scheduler.waiting.extend(requests)
    first_output = scheduler.schedule()
    assert isinstance(first_output, SchedulerOutput)
    assert first_output.prefill_requests == requests[:2]
    assert first_output.prefill_token_counts == [1, 1]
    assert first_output.decode_requests == []
    assert first_output.num_scheduled_tokens == 2
    assert scheduler.running == requests[:2]
    assert list(scheduler.waiting) == requests[2:]
    assert requests[0].status == RequestStatus.PREFILL
    assert requests[1].status == RequestStatus.PREFILL
    assert requests[2].status == RequestStatus.WAITING
    for request in requests[:2]:
        request.advance_prefill(1)
        request.complete_prefill()
    second_output = scheduler.schedule()
    assert second_output.prefill_requests == []
    assert second_output.decode_requests == requests[:2]
    assert second_output.num_scheduled_tokens == 2
    assert scheduler.running == requests[:2]
    assert list(scheduler.waiting) == requests[2:]
    scheduler.running.remove(requests[0])
    third_output = scheduler.schedule()
    assert third_output.prefill_requests == [requests[2]]
    assert third_output.prefill_token_counts == [1]
    assert third_output.decode_requests == [requests[1]]
    assert third_output.num_scheduled_tokens == 2
    assert scheduler.running == [requests[1], requests[2]]
    assert list(scheduler.waiting) == []
    assert requests[2].status == RequestStatus.PREFILL

def test_scheduler_rejects_invalid_batch_size():
    with pytest.raises(ValueError):
        Scheduler(max_batch_size=0, max_num_batched_tokens=2)

def test_scheduler_token_budget_includes_decode_and_prefill_tokens():
    scheduler = Scheduler(max_batch_size=3, max_num_batched_tokens=5)
    decode_request = create_request("req-decode")
    decode_request.start_prefill()
    decode_request.advance_prefill(1)
    decode_request.complete_prefill()
    first_waiting_request = create_request("req-first", [10, 11, 12])
    second_waiting_request = create_request("req-second", [20, 21])
    scheduler.running.append(decode_request)
    scheduler.waiting.extend([first_waiting_request, second_waiting_request])
    output = scheduler.schedule()
    assert output.decode_requests == [decode_request]
    assert output.prefill_requests == [first_waiting_request, second_waiting_request]
    assert output.prefill_token_counts == [3, 1]
    assert output.num_scheduled_tokens == 5
    assert scheduler.running == [decode_request, first_waiting_request, second_waiting_request]
    assert list(scheduler.waiting) == []

def test_scheduler_counts_only_uncomputed_prompt_tokens():
    scheduler = Scheduler(max_batch_size=2, max_num_batched_tokens=4)
    decode_request = create_request("req-decode")
    decode_request.start_prefill()
    decode_request.advance_prefill(1)
    decode_request.complete_prefill()
    partial_request = create_request("req-partial", [10, 11, 12, 13, 14])
    partial_request.num_computed_tokens = 2
    scheduler.running.append(decode_request)
    scheduler.waiting.append(partial_request)
    output = scheduler.schedule()
    assert output.decode_requests == [decode_request]
    assert output.prefill_requests == [partial_request]
    assert output.prefill_token_counts == [3]
    assert output.num_scheduled_tokens == 4
    assert partial_request.status == RequestStatus.PREFILL
    assert scheduler.running == [decode_request, partial_request]
    assert list(scheduler.waiting) == []

def test_scheduler_rejects_invalid_token_budget():
    with pytest.raises(ValueError):
        Scheduler(max_batch_size=2, max_num_batched_tokens=0)
    with pytest.raises(ValueError):
        Scheduler(max_batch_size=3, max_num_batched_tokens=2)
    with pytest.raises(ValueError):
        Scheduler(max_batch_size=1, max_num_batched_tokens=2, prefill_chunk_size=0)

def test_scheduler_chunks_prompt_larger_than_token_budget():
    scheduler = Scheduler(max_batch_size=1, max_num_batched_tokens=3)
    request = create_request("req-large", [10, 11, 12, 13])
    scheduler.waiting.append(request)
    first_output = scheduler.schedule()
    assert first_output.prefill_requests == [request]
    assert first_output.prefill_token_counts == [3]
    assert first_output.num_scheduled_tokens == 3
    request.advance_prefill(3)
    second_output = scheduler.schedule()
    assert second_output.prefill_requests == [request]
    assert second_output.prefill_token_counts == [1]
    assert second_output.num_scheduled_tokens == 1

def test_scheduler_reschedules_running_prefill_without_using_another_slot():
    scheduler = Scheduler(max_batch_size=2, max_num_batched_tokens=4, prefill_chunk_size=2)
    first_request = create_request("req-first", [10, 11, 12, 13, 14])
    second_request = create_request("req-second", [20, 21, 22, 23, 24])
    scheduler.waiting.extend([first_request, second_request])
    first_output = scheduler.schedule()
    assert first_output.prefill_requests == [first_request, second_request]
    assert first_output.prefill_token_counts == [2, 2]
    first_request.advance_prefill(2)
    second_request.advance_prefill(2)
    second_output = scheduler.schedule()
    assert second_output.prefill_requests == [first_request, second_request]
    assert second_output.prefill_token_counts == [2, 2]
    assert second_output.num_scheduled_tokens == 4
    assert scheduler.running == [first_request, second_request]
