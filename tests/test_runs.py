import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from fiae.errors import ErrorCode, FIAEError
from fiae.events import EventType
from fiae.runs import RunManager, RunState, CancellationToken


@pytest.fixture()
def manager(tmp_path):
    return RunManager(tmp_path / "runs")


def test_run_directory_layout(manager, tmp_path):
    handle = manager.create_run({"target": "churn", "task": "auto"})
    for rel in (
        "run.json",
        "events.jsonl",
        "decisions.jsonl",
        "metrics",
        "reports",
        "artifacts",
        "index.sqlite",
    ):
        assert (handle.path / rel).exists(), f"missing {rel}"
    m = json.loads((handle.path / "run.json").read_text(encoding="utf-8"))
    assert m["completion_state"] == "CREATED"
    assert m["config_hash"]
    handle.bus.close()


def test_run_created_event_emitted(manager):
    handle = manager.create_run({"x": 1})
    handle.bus.close()
    lines = (handle.path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    types = [json.loads(line)["event_type"] for line in lines]
    assert EventType.RUN_CREATED in types


def test_normal_state_lifecycle(manager):
    handle = manager.create_run({"x": 1})
    handle.transition(RunState.PLANNING)
    handle.transition(RunState.RUNNING)
    handle.transition(RunState.COMPLETED, reason="completion contract passed")
    assert handle.state is RunState.COMPLETED
    handle.bus.close()


def test_created_cannot_become_completed_directly(manager):
    handle = manager.create_run({"x": 1})
    with pytest.raises(FIAEError) as exc:
        handle.transition(RunState.COMPLETED)
    assert exc.value.code is ErrorCode.REPRODUCIBILITY_MISMATCH
    handle.bus.close()


def test_terminal_states_are_absorbing(manager):
    handle = manager.create_run({"x": 1})
    handle.transition(RunState.PLANNING)
    handle.transition(RunState.FAILED)
    with pytest.raises(FIAEError):
        handle.transition(RunState.RUNNING)
    handle.bus.close()


def test_cancellation_token():
    tok = CancellationToken()
    assert not tok.cancelled
    tok.cancel()
    assert tok.cancelled
    with pytest.raises(FIAEError) as exc:
        tok.check()
    assert exc.value.code is ErrorCode.CANCELLED


def test_config_hash_stable_across_key_order(manager):
    h1 = manager.create_run({"a": 1, "b": 2})
    h1.bus.close()
    h2 = manager.create_run({"b": 2, "a": 1})
    h2.bus.close()
    m1 = json.loads((h1.path / "run.json").read_text(encoding="utf-8"))
    m2 = json.loads((h2.path / "run.json").read_text(encoding="utf-8"))
    assert m1["config_hash"] == m2["config_hash"]
