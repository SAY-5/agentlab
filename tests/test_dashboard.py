"""Dashboard HTTP integration tests via FastAPI TestClient."""

from __future__ import annotations

import time
from pathlib import Path

import os
import pytest

httpx = pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from agentlab.core.types import AgentDef, Result, Run  # noqa: E402
from agentlab.store import Store  # noqa: E402


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    db = tmp_path / "dash.db"
    store = Store(db)
    run = Run(
        id="run_dash",
        started_at=time.time(),
        finished_at=None,
        suite_name="smoke",
        suite_version="1",
        agents=[AgentDef(id="mock:m", provider="mock", model="m")],
    )
    store.insert_run(run)
    store.insert_result(
        Result(
            run_id="run_dash",
            task_id="say_hello",
            agent_id="mock:m",
            trial_idx=0,
            started_at=time.time(),
            finished_at=time.time(),
            status="ok",
            score=0.9,
            tokens_in=10,
            tokens_out=5,
            trajectory=[{"kind": "model", "content": "hi"}],
        )
    )
    store.close()
    monkeypatch.setenv("AGENTLAB_DB", str(db))
    # Import after env is set so the module-level db path binds correctly.
    from agentlab.dashboard.app import app  # noqa: E402
    return TestClient(app)


def test_healthcheck_root_returns_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "AgentLab" in r.text
    # Sanity-check the XSS escape helper is present.
    assert "function esc(" in r.text


def test_api_runs_lists_inserted_run(client):
    r = client.get("/api/runs")
    assert r.status_code == 200
    runs = r.json()
    assert any(x["id"] == "run_dash" for x in runs)


def test_api_run_detail_includes_results(client):
    r = client.get("/api/runs/run_dash")
    assert r.status_code == 200
    data = r.json()
    assert data["suite_name"] == "smoke"
    assert len(data["results"]) == 1
    assert data["results"][0]["score"] == 0.9


def test_api_run_404_on_unknown_id(client):
    r = client.get("/api/runs/nope")
    assert r.status_code == 404


def test_api_trajectory(client):
    r = client.get("/api/runs/run_dash/results/say_hello/mock:m/0/trajectory")
    assert r.status_code == 200
    assert r.json()["trajectory"][0]["content"] == "hi"


def test_api_trajectory_404(client):
    r = client.get("/api/runs/run_dash/results/nope/mock:m/0/trajectory")
    assert r.status_code == 404
