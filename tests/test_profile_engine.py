from types import SimpleNamespace
from unittest.mock import Mock, patch
import pytest
import torch
from profile_engine import mps_profile_context, profiled_step


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
