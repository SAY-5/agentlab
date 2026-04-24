import time
from pathlib import Path

from agentlab.core.types import AgentDef, Result, Run
from agentlab.store import Store


def test_insert_and_read_run(tmp_path: Path):
    db = tmp_path / "t.db"
    s = Store(db)
    run = Run(
        id="run_1",
        started_at=time.time(),
        finished_at=None,
        suite_name="smoke",
        suite_version="1",
        agents=[AgentDef(id="mock:m", provider="mock", model="m")],
        notes="hello",
    )
    s.insert_run(run)
    runs = s.list_runs()
    assert len(runs) == 1
    assert runs[0]["id"] == "run_1"

    fetched = s.get_run("run_1")
    assert fetched is not None
    assert fetched["agents"][0]["id"] == "mock:m"


def test_insert_and_read_results(tmp_path: Path):
    db = tmp_path / "t.db"
    s = Store(db)
    run = Run(
        id="run_2",
        started_at=time.time(),
        finished_at=None,
        suite_name="smoke",
        suite_version="1",
        agents=[AgentDef(id="mock:m", provider="mock", model="m")],
    )
    s.insert_run(run)
    r = Result(
        run_id="run_2",
        task_id="say_hello",
        agent_id="mock:m",
        trial_idx=0,
        started_at=time.time(),
        finished_at=time.time(),
        status="ok",
        score=0.75,
        tokens_in=20,
        tokens_out=5,
        trajectory=[{"kind": "model", "content": "hi"}],
    )
    s.insert_result(r)

    rows = s.list_results("run_2")
    assert len(rows) == 1
    assert rows[0]["score"] == 0.75

    traj = s.get_trajectory("run_2", "say_hello", "mock:m", 0)
    assert traj == [{"kind": "model", "content": "hi"}]


def test_missing_run_returns_none(tmp_path: Path):
    s = Store(tmp_path / "t.db")
    assert s.get_run("nope") is None
