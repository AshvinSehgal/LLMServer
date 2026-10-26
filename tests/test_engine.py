from types import SimpleNamespace
import asyncio
import torch
import torch.nn as nn
import pytest
from utils.engine import Engine
from utils.request import RequestFinishReason, RequestStatus

class Tokenizer:
    def __init__(self, bos_token_id, pad_token_id=4, eos_token_id=2):
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    def __call__(self, prompt):
        return {"input_ids": [10, 11]}

    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        decoded_tokens = {2: "", 50: "Hel", 51: "lo", 60: "World"}
        return "".join(decoded_tokens[token_id] for token_id in token_ids)

class Model(nn.Module):
    def __init__(self, token_ids_by_call, bos_token_id=1, pad_token_id=4, eos_token_id=2):
        super().__init__()
        self.dummy_parameter = nn.Parameter(torch.zeros(1))
        self.generation_config = SimpleNamespace(bos_token_id=bos_token_id, pad_token_id=pad_token_id, eos_token_id=eos_token_id)
        self.config = SimpleNamespace(bos_token_id=bos_token_id, pad_token_id=pad_token_id, eos_token_id=eos_token_id)
        self.token_ids_by_call = token_ids_by_call
        self.inference_mode_by_call = []

    def forward(self, input_ids, pos_ids, attention_mask):
        self.inference_mode_by_call.append(torch.is_inference_mode_enabled())
        token_ids = self.token_ids_by_call.pop(0)
        assert len(token_ids) == input_ids.shape[0]
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 100, device=input_ids.device)
        for row, token_id in enumerate(token_ids):
            logits[row, :, token_id] = 1
        return logits

def test_engine_add_request():
    model = Model(token_ids_by_call=[])
    engine = Engine(model=model, tokenizer=Tokenizer(bos_token_id=3), max_new_tokens=20, max_batch_size=2, max_num_batched_tokens=20)
    empty_request = engine.add_request("")
    normal_request = engine.add_request("Hello")
    assert empty_request.prompt_token_ids == [1]
    assert normal_request.prompt_token_ids == [10, 11]
    assert empty_request.request_id.startswith("req-")
    assert empty_request.request_id != normal_request.request_id
    assert empty_request.output_queue is not normal_request.output_queue
    assert list(engine.scheduler.waiting) == [empty_request, normal_request]

def test_engine_token_id_fallbacks():
    model = Model(token_ids_by_call=[], bos_token_id=None, pad_token_id=None, eos_token_id=None)
    model.config = SimpleNamespace(bos_token_id=2, pad_token_id=5, eos_token_id=6)
    engine = Engine(model=model, tokenizer=Tokenizer(bos_token_id=3, pad_token_id=7, eos_token_id=8), max_new_tokens=20, max_batch_size=2, max_num_batched_tokens=20)
    assert engine.bos_token_id == 2
    assert engine.pad_token_id == 5
    assert engine.eos_token_ids == {6}
    model.config = SimpleNamespace(bos_token_id=None, pad_token_id=None, eos_token_id=None)
    engine = Engine(model=model, tokenizer=Tokenizer(bos_token_id=3, pad_token_id=7, eos_token_id=8), max_new_tokens=20, max_batch_size=2, max_num_batched_tokens=20)
    assert engine.bos_token_id == 3
    assert engine.pad_token_id == 7
    assert engine.eos_token_ids == {8}

def test_engine_prefill_batch_and_inference_mode():
    model = Model(token_ids_by_call=[[50, 60]])
    engine = Engine(model=model, tokenizer=Tokenizer(bos_token_id=3), max_new_tokens=3, max_batch_size=2, max_num_batched_tokens=20)
    first_request = engine.add_request("First")
    second_request = engine.add_request("Second")
    assert engine.step()
    assert first_request.generated_token_ids == [50]
    assert second_request.generated_token_ids == [60]
    assert first_request.status == RequestStatus.DECODING
    assert second_request.status == RequestStatus.DECODING
    assert first_request.num_computed_tokens == 2
    assert second_request.num_computed_tokens == 2
    assert model.inference_mode_by_call == [True]

def test_engine_decode_and_length_stopping_streams_incremental_text():
    model = Model(token_ids_by_call=[[50], [51]])
    engine = Engine(model=model, tokenizer=Tokenizer(bos_token_id=3), max_new_tokens=2, max_batch_size=1, max_num_batched_tokens=20)
    request = engine.add_request("Hello")
    assert engine.step()
    assert not engine.step()
    assert request.generated_token_ids == [50, 51]
    assert request.num_computed_tokens == 3
    assert request.status == RequestStatus.FINISHED
    assert request.finish_reason == RequestFinishReason.LENGTH
    assert request not in engine.scheduler.running
    assert request.output_queue.get_nowait() == "Hel"
    assert request.output_queue.get_nowait() == "lo"
    assert request.output_queue.get_nowait() is None
    assert model.inference_mode_by_call == [True, True]

def test_engine_eos_stopping():
    model = Model(token_ids_by_call=[[2]], eos_token_id=[2, 3])
    engine = Engine(model=model, tokenizer=Tokenizer(bos_token_id=3), max_new_tokens=5, max_batch_size=1, max_num_batched_tokens=20)
    request = engine.add_request("Hello")
    assert not engine.step()
    assert request.generated_token_ids == [2]
    assert request.status == RequestStatus.FINISHED
    assert request.finish_reason == RequestFinishReason.EOS
    assert request not in engine.scheduler.running
    assert request.output_queue.get_nowait() is None

def test_engine_rejects_missing_bos_and_pad_token_ids():
    model = Model(token_ids_by_call=[], bos_token_id=None, pad_token_id=None)
    model.config = SimpleNamespace(bos_token_id=None, pad_token_id=None, eos_token_id=2)
    with pytest.raises(ValueError):
        Engine(model=model, tokenizer=Tokenizer(bos_token_id=None), max_new_tokens=20, max_batch_size=2, max_num_batched_tokens=20)
    with pytest.raises(ValueError):
        Engine(model=model, tokenizer=Tokenizer(bos_token_id=1, pad_token_id=None), max_new_tokens=20, max_batch_size=2, max_num_batched_tokens=20)

def test_empty_engine_step_returns_false():
    engine = Engine(model=Model(token_ids_by_call=[]), tokenizer=Tokenizer(bos_token_id=3), max_new_tokens=20, max_batch_size=2, max_num_batched_tokens=20)
    assert not engine.step()

@pytest.mark.asyncio
async def test_engine_run():
    model = Model(token_ids_by_call=[[50], [51]])
    engine = Engine(model=model, tokenizer=Tokenizer(bos_token_id=3), max_new_tokens=2, max_batch_size=1, max_num_batched_tokens=20)
    request = engine.add_request("Hello")
    await engine.run()
    assert request.generated_token_ids == [50, 51]
    assert request.status == RequestStatus.FINISHED
    assert request.finish_reason == RequestFinishReason.LENGTH
    assert engine.scheduler.running == []
    assert list(engine.scheduler.waiting) == []

@pytest.mark.asyncio
async def test_engine_run_accepts_request_between_steps():
    model = Model(token_ids_by_call=[[50], [51], [60], [51]])
    engine = Engine(model=model, tokenizer=Tokenizer(bos_token_id=3), max_new_tokens=2, max_batch_size=1, max_num_batched_tokens=20)
    first_request = engine.add_request("First")
    engine_task = asyncio.create_task(engine.run())
    while not first_request.generated_token_ids:
        await asyncio.sleep(0)
    second_request = engine.add_request("Second")
    await engine_task
    assert first_request.generated_token_ids == [50, 51]
    assert second_request.generated_token_ids == [60, 51]
    assert first_request.status == RequestStatus.FINISHED
    assert second_request.status == RequestStatus.FINISHED
    assert engine.scheduler.running == []
    assert list(engine.scheduler.waiting) == []

def test_engine_continuous_batching_replaces_finished_request():
    model = Model(token_ids_by_call=[[2, 50], [60], [51]])
    engine = Engine(model=model, tokenizer=Tokenizer(bos_token_id=3), max_new_tokens=3, max_batch_size=2, max_num_batched_tokens=20)
    short_request = engine.add_request("Short")
    long_request = engine.add_request("Long")
    waiting_request = engine.add_request("Waiting")
    assert engine.step()
    assert short_request.status == RequestStatus.FINISHED
    assert short_request.finish_reason == RequestFinishReason.EOS
    assert long_request.generated_token_ids == [50]
    assert engine.scheduler.running == [long_request]
    assert list(engine.scheduler.waiting) == [waiting_request]
    assert engine.step()
    assert long_request.generated_token_ids == [50, 51]
    assert waiting_request.generated_token_ids == [60]
    assert long_request.status == RequestStatus.DECODING
    assert waiting_request.status == RequestStatus.DECODING
    assert engine.scheduler.running == [long_request, waiting_request]
    assert list(engine.scheduler.waiting) == []

def test_engine_token_budget_limits_prefill_admission():
    model = Model(token_ids_by_call=[[50]])
    engine = Engine(model=model, tokenizer=Tokenizer(bos_token_id=3), max_new_tokens=3, max_batch_size=2, max_num_batched_tokens=2)
    first_request = engine.add_request("First")
    second_request = engine.add_request("Second")
    assert engine.step()
    assert first_request.generated_token_ids == [50]
    assert second_request.generated_token_ids == []
    assert engine.scheduler.running == [first_request]
    assert list(engine.scheduler.waiting) == [second_request]

def test_engine_rejects_prompt_larger_than_token_budget():
    engine = Engine(model=Model(token_ids_by_call=[]), tokenizer=Tokenizer(bos_token_id=3), max_new_tokens=3, max_batch_size=1, max_num_batched_tokens=1)
    with pytest.raises(ValueError):
        engine.add_request("Hello")

def test_engine_cancels_waiting_request():
    engine = Engine(model=Model(token_ids_by_call=[]), tokenizer=Tokenizer(bos_token_id=3), max_new_tokens=3, max_batch_size=1, max_num_batched_tokens=2)
    request = engine.add_request("Hello")
    released_requests = []
    engine.release_request_resources = released_requests.append
    assert engine.cancel_request(request.request_id)
    assert request.status == RequestStatus.CANCELLED
    assert request.finish_reason == RequestFinishReason.CANCELLED
    assert request.finished_at is not None
    assert list(engine.scheduler.waiting) == []
    assert engine.scheduler.running == []
    assert request.output_queue.get_nowait() is None
    assert released_requests == [request]

def test_engine_cancels_running_request_and_clears_queued_output():
    engine = Engine(model=Model(token_ids_by_call=[[50]]), tokenizer=Tokenizer(bos_token_id=3), max_new_tokens=3, max_batch_size=1, max_num_batched_tokens=2)
    request = engine.add_request("Hello")
    assert engine.step()
    assert request.generated_token_ids == [50]
    assert request in engine.scheduler.running
    assert engine.cancel_request(request.request_id)
    assert request.status == RequestStatus.CANCELLED
    assert request.finish_reason == RequestFinishReason.CANCELLED
    assert request not in engine.scheduler.running
    assert request.output_queue.get_nowait() is None
    with pytest.raises(asyncio.QueueEmpty):
        request.output_queue.get_nowait()

def test_engine_rejects_unknown_repeated_and_finished_cancellation():
    engine = Engine(model=Model(token_ids_by_call=[[2]]), tokenizer=Tokenizer(bos_token_id=3), max_new_tokens=3, max_batch_size=1, max_num_batched_tokens=2)
    assert not engine.cancel_request("unknown-request")
    cancelled_request = engine.add_request("Cancelled")
    assert engine.cancel_request(cancelled_request.request_id)
    assert not engine.cancel_request(cancelled_request.request_id)
    finished_request = engine.add_request("Finished")
    assert not engine.step()
    assert finished_request.status == RequestStatus.FINISHED
    assert not engine.cancel_request(finished_request.request_id)
    assert finished_request.status == RequestStatus.FINISHED
    assert finished_request.finish_reason == RequestFinishReason.EOS
