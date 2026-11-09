from types import SimpleNamespace
from unittest.mock import Mock, patch
import pytest
import torch
from torch.profiler import ProfilerActivity, profile
from utils.layers import RMSNorm
from profile_engine import mps_profile_context, profile_model_operations, profiled_step


def test_profiled_step_runs_engine_phases_in_order():
    scheduler_output = object()
    engine = Mock()
    engine.schedule_step.return_value = scheduler_output
    engine.scheduler = SimpleNamespace(waiting=[], running=[object()])
    with patch("profile_engine.record_function"):
        assert profiled_step(engine)
    assert engine.method_calls[0].args == ()
    assert engine.method_calls[0][0] == "schedule_step"
    assert [call[0] for call in engine.method_calls] == ["schedule_step", "process_decode_batch", "remove_finished_and_cancelled_requests", "process_prefill_batch", "remove_finished_and_cancelled_requests"]
    engine.process_decode_batch.assert_called_once_with(scheduler_output)
    engine.process_prefill_batch.assert_called_once_with(scheduler_output)


def test_mps_profile_context_is_optional_and_rejects_other_devices():
    with mps_profile_context(torch.device("cpu"), False):
        pass
    with pytest.raises(ValueError, match="requires an MPS device"):
        mps_profile_context(torch.device("cpu"), True)


def test_mps_profile_context_uses_signposts_without_forcing_each_dispatch_to_finish():
    with patch("profile_engine.torch.mps.profiler.profile") as make_profile:
        assert mps_profile_context(torch.device("mps"), True) is make_profile.return_value
    make_profile.assert_called_once_with(mode="interval", wait_until_completed=False)


def test_profile_model_operations_labels_hot_paths_and_restores_methods():
    norm = RMSNorm(4, 1e-6)
    cache = SimpleNamespace(append=Mock(), get=Mock(), get_batch=Mock(), get_batch_ragged=Mock())
    engine = SimpleNamespace(model=torch.nn.Sequential(norm), kv_cache=cache)
    with profile(activities=[ProfilerActivity.CPU], acc_events=True) as profiler:
        with profile_model_operations(engine):
            norm(torch.ones(1, 4))
            cache.append()
            cache.get()
            cache.get_batch()
            cache.get_batch_ragged()
    names = {event.key for event in profiler.key_averages()}
    assert {"RMSNorm.forward", "PagedKVCache.append", "PagedKVCache.get", "PagedKVCache.get_batch", "PagedKVCache.get_batch_ragged"} <= names
    assert "forward" not in norm.__dict__
    cache.append.assert_called_once()
    cache.get.assert_called_once()
    cache.get_batch.assert_called_once()
    cache.get_batch_ragged.assert_called_once()
