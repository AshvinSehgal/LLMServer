from types import SimpleNamespace
import asyncio
from datetime import timedelta
import torch
import torch.nn as nn
import pytest
from utils.engine import Engine
from utils.kv_cache import PagedKVCache, KVCacheExhaustedError
from utils.request import RequestFinishReason, RequestStatus

class Tokenizer:
    def __init__(self, bos_token_id, pad_token_id=4, eos_token_id=2, prompt_token_ids=None):
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.prompt_token_ids = prompt_token_ids or [10, 11]

    def __call__(self, prompt):
        if isinstance(self.prompt_token_ids, dict):
            return {"input_ids": self.prompt_token_ids[prompt]}
        return {"input_ids": self.prompt_token_ids}

    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        decoded_tokens = {2: "", 50: "Hel", 51: "lo", 60: "World"}
        return "".join(decoded_tokens[token_id] for token_id in token_ids)

class Model(nn.Module):
    def __init__(self, token_ids_by_call, bos_token_id=1, pad_token_id=4, eos_token_id=2):
        super().__init__()
        self.dummy_parameter = nn.Parameter(torch.zeros(1))
        self.generation_config = SimpleNamespace(bos_token_id=bos_token_id, pad_token_id=pad_token_id, eos_token_id=eos_token_id)
        self.config = SimpleNamespace(bos_token_id=bos_token_id, pad_token_id=pad_token_id, eos_token_id=eos_token_id, hidden_size=8, num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2, head_dim=2)
        self.token_ids_by_call = token_ids_by_call
        self.inference_mode_by_call = []
        self.calls = []

    def forward(self, input_ids, pos_ids, attention_mask, **kwargs):
        self.inference_mode_by_call.append(torch.is_inference_mode_enabled())
        self.calls.append({"input_shape": tuple(input_ids.shape), "input_ids": input_ids.tolist(), "pos_ids": pos_ids.tolist(), "request_ids": kwargs["request_ids"], "block_tables": kwargs["block_tables"], "start_pos": kwargs["start_pos"], "input_lens": kwargs["input_lens"], "kv_cache_manager": kwargs["kv_cache_manager"]})
        for block_table, start_pos, input_len in zip(kwargs["block_tables"], kwargs["start_pos"], kwargs["input_lens"]):
            for token_pos in range(start_pos, start_pos + input_len):
                kwargs["kv_cache_manager"].ensure_token_block(block_table, token_pos)
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
    assert engine.active_requests == {empty_request.request_id: empty_request, normal_request.request_id: normal_request}
    assert engine.completed_requests == {}

def test_engine_token_id_fallbacks():
    model = Model(token_ids_by_call=[], bos_token_id=None, pad_token_id=None, eos_token_id=None)
    model.config.bos_token_id = 2
    model.config.pad_token_id = 5
    model.config.eos_token_id = 6
    engine = Engine(model=model, tokenizer=Tokenizer(bos_token_id=3, pad_token_id=7, eos_token_id=8), max_new_tokens=20, max_batch_size=2, max_num_batched_tokens=20)
    assert engine.bos_token_id == 2
    assert engine.pad_token_id == 5
    assert engine.eos_token_ids == {6}
    model.config.bos_token_id = None
    model.config.pad_token_id = None
    model.config.eos_token_id = None
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
    assert model.calls[0]["input_shape"] == (2, 2)
    assert model.calls[0]["request_ids"] == [first_request.request_id, second_request.request_id]
    assert model.calls[0]["start_pos"] == [0, 0]
    assert model.calls[0]["input_lens"] == [2, 2]
    assert model.calls[0]["kv_cache_manager"] is engine.kv_cache
    assert model.calls[0]["block_tables"][0] is first_request.block_table
    assert model.calls[0]["block_tables"][1] is second_request.block_table
    assert first_request.block_table == [0]
    assert second_request.block_table == [1]
    assert engine.kv_cache.num_allocated_blocks == 2

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
    assert [call["input_shape"] for call in model.calls] == [(1, 2), (1, 1)]
    assert model.calls[1]["request_ids"] == [request.request_id]
    assert model.calls[1]["start_pos"] == [2]
    assert model.calls[1]["input_lens"] == [1]
    assert request.block_table == []
    assert engine.kv_cache.num_allocated_blocks == 0
    assert engine.kv_cache.num_free_blocks == engine.kv_cache.num_blocks

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
    assert request.block_table == []
    assert engine.kv_cache.num_allocated_blocks == 0

def test_engine_rejects_missing_bos_and_pad_token_ids():
    model = Model(token_ids_by_call=[], bos_token_id=None, pad_token_id=None)
    model.config.bos_token_id = None
    model.config.pad_token_id = None
    model.config.eos_token_id = 2
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

@pytest.mark.asyncio
async def test_engine_run_yields_decode_output_before_prefill():
    model = Model(token_ids_by_call=[[50], [51], [60], [51]])
    engine = Engine(model=model, tokenizer=Tokenizer(bos_token_id=3), max_new_tokens=2, max_batch_size=2, max_num_batched_tokens=4)
    decode_request = engine.add_request("Decode")
    assert engine.step()
    assert decode_request.output_queue.get_nowait() == "Hel"
    prefill_request = engine.add_request("Prefill")
    async def consume_decode_output():
        assert await decode_request.output_queue.get() == "lo"
        assert len(model.calls) == 2
        assert prefill_request.generated_token_ids == []
    consumer_task = asyncio.create_task(consume_decode_output())
    await asyncio.sleep(0)
    await asyncio.wait_for(asyncio.gather(consumer_task, engine.run()), timeout=5)
    assert [call["request_ids"] for call in model.calls] == [[decode_request.request_id], [decode_request.request_id], [prefill_request.request_id], [prefill_request.request_id]]

@pytest.mark.asyncio
async def test_engine_run_skips_prefill_cancelled_during_yield():
    model = Model(token_ids_by_call=[[50], [51]])
    engine = Engine(model=model, tokenizer=Tokenizer(bos_token_id=3), max_new_tokens=2, max_batch_size=2, max_num_batched_tokens=4)
    decode_request = engine.add_request("Decode")
    assert engine.step()
    assert decode_request.output_queue.get_nowait() == "Hel"
    prefill_request = engine.add_request("Prefill")
    async def cancel_after_decode():
        assert await decode_request.output_queue.get() == "lo"
        assert engine.cancel_request(prefill_request.request_id)
    cancellation_task = asyncio.create_task(cancel_after_decode())
    await asyncio.sleep(0)
    await asyncio.wait_for(asyncio.gather(cancellation_task, engine.run()), timeout=5)
    assert prefill_request.status == RequestStatus.CANCELLED
    assert prefill_request.generated_token_ids == []
    assert prefill_request.block_table == []
    assert [call["request_ids"] for call in model.calls] == [[decode_request.request_id], [decode_request.request_id]]
    assert engine.scheduler.running == []

def test_engine_continuous_batching_replaces_finished_request():
    model = Model(token_ids_by_call=[[2, 50], [51], [60]])
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

def test_engine_cache_capacity_error_does_not_admit_request():
    model = Model(token_ids_by_call=[[50]])
    engine = Engine(model=model, tokenizer=Tokenizer(bos_token_id=3), max_new_tokens=1, max_batch_size=1, max_num_batched_tokens=2)
    engine.kv_cache = PagedKVCache(num_layers=2, num_blocks=1, block_size=2, num_kv_heads=2, head_dim=2, device=torch.device("cpu"), dtype=torch.float32)
    occupied_block = engine.kv_cache.allocate_block()
    request = engine.add_request("Waiting")
    with pytest.raises(KVCacheExhaustedError):
        engine.step()
    assert request.status == RequestStatus.WAITING
    assert request.num_computed_tokens == 0
    assert request.block_table == []
    assert list(engine.scheduler.waiting) == [request]
    assert engine.scheduler.running == []
    assert model.calls == []
    engine.kv_cache.free_block(occupied_block)
    assert not engine.step()
    assert request.generated_token_ids == [50]
    assert request.status == RequestStatus.FINISHED

def test_engine_waits_for_cache_headroom_before_admitting_request():
    model = Model(token_ids_by_call=[[50], [51], [60]])
    tokenizer = Tokenizer(bos_token_id=3, prompt_token_ids={"First": [10, 11], "Second": [20, 21]})
    engine = Engine(model=model, tokenizer=tokenizer, max_new_tokens=2, max_batch_size=2, max_num_batched_tokens=4, kv_cache_num_blocks=3, kv_cache_block_size=2)
    first_request = engine.add_request("First")
    assert engine.step()
    second_request = engine.add_request("Second")
    assert engine.step()
    assert first_request.status == RequestStatus.FINISHED
    assert first_request.block_table == []
    assert second_request.status == RequestStatus.WAITING
    assert second_request.generated_token_ids == []
    assert list(engine.scheduler.waiting) == [second_request]
    assert engine.step()
    assert second_request.generated_token_ids == [60]
    assert second_request.status == RequestStatus.DECODING
    assert [call["request_ids"] for call in model.calls] == [[first_request.request_id], [first_request.request_id], [second_request.request_id]]

def test_engine_chunks_prompt_larger_than_token_budget():
    model = Model(token_ids_by_call=[[50], [50]])
    engine = Engine(model=model, tokenizer=Tokenizer(bos_token_id=3), max_new_tokens=1, max_batch_size=1, max_num_batched_tokens=1)
    request = engine.add_request("Hello")
    assert engine.step()
    assert request.num_computed_tokens == 1
    assert request.status == RequestStatus.PREFILL
    assert request.generated_token_ids == []
    assert request.output_queue.empty()
    assert not engine.step()
    assert request.num_computed_tokens == 2
    assert request.generated_token_ids == [50]
    assert request.status == RequestStatus.FINISHED
    assert [call["start_pos"] for call in model.calls] == [[0], [1]]

def test_engine_streams_only_after_final_prefill_chunk():
    model = Model(token_ids_by_call=[[60], [60], [50], [51]])
    tokenizer = Tokenizer(bos_token_id=3, prompt_token_ids=list(range(10, 18)))
    engine = Engine(model=model, tokenizer=tokenizer, max_new_tokens=2, max_batch_size=1, max_num_batched_tokens=3, prefill_chunk_size=3)
    request = engine.add_request("Long prompt")
    for expected_count in (3, 6):
        assert engine.step()
        assert request.num_computed_tokens == expected_count
        assert request.status == RequestStatus.PREFILL
        assert request.generated_token_ids == []
        assert request.output_queue.empty()
    assert engine.step()
    assert request.num_computed_tokens == 8
    assert request.status == RequestStatus.DECODING
    assert request.generated_token_ids == [50]
    assert request.output_queue.get_nowait() == "Hel"
    assert [call["input_ids"] for call in model.calls] == [[[10, 11, 12]], [[13, 14, 15]], [[16, 17]]]
    assert [call["pos_ids"] for call in model.calls] == [[[0, 1, 2]], [[3, 4, 5]], [[6, 7]]]
    assert [call["start_pos"] for call in model.calls] == [[0], [3], [6]]
    assert [call["input_lens"] for call in model.calls] == [[3], [3], [2]]
    assert not engine.step()
    assert request.generated_token_ids == [50, 51]
    assert request.status == RequestStatus.FINISHED
    assert request.output_queue.get_nowait() == "lo"
    assert request.output_queue.get_nowait() is None
    assert request.block_table == []

def test_engine_samples_only_finished_prefill_rows_in_mixed_batch():
    model = Model(token_ids_by_call=[[50, 60], [51], [60]])
    tokenizer = Tokenizer(bos_token_id=3, prompt_token_ids={"Short": [10, 11], "Long": [20, 21, 22, 23]})
    engine = Engine(model=model, tokenizer=tokenizer, max_new_tokens=2, max_batch_size=2, max_num_batched_tokens=4, prefill_chunk_size=2)
    short_request = engine.add_request("Short")
    long_request = engine.add_request("Long")
    assert engine.step()
    assert short_request.generated_token_ids == [50]
    assert short_request.status == RequestStatus.DECODING
    assert long_request.generated_token_ids == []
    assert long_request.status == RequestStatus.PREFILL
    assert long_request.num_computed_tokens == 2
    assert long_request.output_queue.empty()
    assert engine.step()
    assert short_request.generated_token_ids == [50, 51]
    assert long_request.generated_token_ids == [60]
    assert long_request.num_computed_tokens == 4
    assert model.calls[1]["request_ids"] == [short_request.request_id]
    assert model.calls[1]["start_pos"] == [2]
    assert model.calls[2]["request_ids"] == [long_request.request_id]
    assert model.calls[2]["start_pos"] == [2]

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
    assert request.block_table == []
    assert engine.kv_cache.num_allocated_blocks == 0
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

def test_engine_moves_terminal_requests_to_bounded_completed_history():
    engine = Engine(model=Model(token_ids_by_call=[[2], [2], [2]]), tokenizer=Tokenizer(bos_token_id=3), max_new_tokens=3, max_batch_size=1, max_num_batched_tokens=2, max_completed_requests=2)
    first_request = engine.add_request("First")
    assert not engine.step()
    second_request = engine.add_request("Second")
    assert not engine.step()
    third_request = engine.add_request("Third")
    assert not engine.step()
    assert engine.active_requests == {}
    assert list(engine.completed_requests) == [second_request.request_id, third_request.request_id]
    assert first_request.request_id not in engine.completed_requests
    completed_request = engine.completed_requests[third_request.request_id]
    assert completed_request.request_id == third_request.request_id
    assert completed_request.status == RequestStatus.FINISHED
    assert completed_request.finish_reason == RequestFinishReason.EOS
    assert completed_request.prompt_token_count == 2
    assert completed_request.generated_token_count == 1
    assert completed_request.output_text == ""
    assert not hasattr(completed_request, "prompt")
    assert not hasattr(completed_request, "output_queue")
    assert not hasattr(completed_request, "generated_token_ids")

def test_engine_prunes_completed_history_after_ttl():
    engine = Engine(model=Model(token_ids_by_call=[[2]]), tokenizer=Tokenizer(bos_token_id=3), max_new_tokens=3, max_batch_size=1, max_num_batched_tokens=2, completed_request_ttl_seconds=10)
    request = engine.add_request("Hello")
    assert not engine.step()
    completed_request = engine.completed_requests[request.request_id]
    engine.prune_completed_requests(now=completed_request.finished_at + timedelta(seconds=11))
    assert engine.completed_requests == {}

def test_engine_can_disable_completed_history():
    engine = Engine(model=Model(token_ids_by_call=[[2]]), tokenizer=Tokenizer(bos_token_id=3), max_new_tokens=3, max_batch_size=1, max_num_batched_tokens=2, max_completed_requests=0)
    request = engine.add_request("Hello")
    assert not engine.step()
    assert request.request_id not in engine.active_requests
    assert engine.completed_requests == {}
