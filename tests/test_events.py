import json
import sqlite3
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fiae.events import EventBus, EventEnvelope, EventType


def make_bus(tmp_path, **kwargs):
    return EventBus(
        events_path=tmp_path / "events.jsonl",
        sqlite_path=tmp_path / "index.sqlite",
        **kwargs,
    )


def test_emit_writes_jsonl_and_sqlite(tmp_path):
    bus = make_bus(tmp_path)
    bus.emit(run_id="run_1", event_type=EventType.RUN_CREATED, component="test")
    bus.emit(
        EventEnvelope(run_id="run_1", event_type="TEST_EVENT", component="test", payload={"k": 1})
    )
    bus.close()

    lines = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert {e["event_type"] for e in lines} == {"RUN_CREATED", "TEST_EVENT"}
    for e in lines:
        assert e["event_id"]
        assert e["timestamp"]
        assert e["schema_version"] == 1

    con = sqlite3.connect(tmp_path / "index.sqlite")
    rows = con.execute("SELECT event_type, run_id, payload FROM events").fetchall()
    con.close()
    assert len(rows) == 2
    assert {r[0] for r in rows} == {"RUN_CREATED", "TEST_EVENT"}
    assert json.loads(next(r for r in rows if r[0] == "TEST_EVENT")[2]) == {"k": 1}


def test_durable_emit_is_synchronous(tmp_path):
    bus = make_bus(tmp_path)
    bus.emit(run_id="run_1", event_type="CRITICAL_DECISION", component="audit", durable=True)
    # No flush/close needed: durable events are already on disk.
    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    bus.close()


def test_no_events_dropped_under_load(tmp_path):
    bus = make_bus(tmp_path, max_queue=64)
    n = 2000
    def producer():
        for i in range(n):
            bus.emit(run_id="run_1", event_type=f"E{i}", component="load")
    t = threading.Thread(target=producer)
    t.start()
    t.join()
    bus.close()
    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == n


def test_subscriber_receives_every_event(tmp_path):
    bus = make_bus(tmp_path)
    seen = []
    bus.subscribe(lambda e: seen.append(e.event_type))
    bus.emit(run_id="r", event_type="A", component="x")
    bus.emit(run_id="r", event_type="B", component="x", durable=True)
    bus.close()
    assert "A" in seen and "B" in seen
