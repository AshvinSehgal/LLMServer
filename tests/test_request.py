import asyncio
import pytest
from utils.request import Request, RequestFinishReason, RequestStatus

def test_request():
    output_queue = asyncio.Queue()
    request = Request("req-1", "Hello", [9707], 20, output_queue)
    assert request.request_id == "req-1"
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
    with pytest.raises(ValueError):
        request.finish(RequestFinishReason.LENGTH)

def test_request_rejects_invalid_transitions():
    request = Request("req-1", "Hello", [9707], 20, asyncio.Queue())
    with pytest.raises(ValueError):
        request.complete_prefill()
    with pytest.raises(ValueError):
        request.append_generated_token(50)
    with pytest.raises(ValueError):
        request.complete_decode()

def test_request_cancellation():
    request = Request("req-1", "Hello", [9707], 20, asyncio.Queue())
    request.cancel()
    assert request.status == RequestStatus.CANCELLED
    assert request.finish_reason == RequestFinishReason.CANCELLED
    assert request.finished_at is not None
