from types import SimpleNamespace
from unittest.mock import Mock, patch
from profile_engine import profiled_step


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
