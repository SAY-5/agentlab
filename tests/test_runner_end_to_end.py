"""End-to-end test: execute the smoke suite against a mock provider."""

from pathlib import Path

import pytest

from agentlab.core.types import AgentDef
from agentlab.providers import register as register_provider
from agentlab.providers.mock import MockProvider
from agentlab.runners import Runner, RunnerConfig
from agentlab.store import Store
from agentlab.suites import load_suite


@pytest.mark.asyncio
async def test_runner_end_to_end(tmp_path: Path):
    mock = MockProvider(["hello there!", "4"])
    register_provider("mock-e2e", mock)

    suite = load_suite(Path(__file__).resolve().parent.parent / "examples" / "suites" / "smoke.yaml")
    store = Store(tmp_path / "runs.db")
    runner = Runner(
        suite=suite,
        agents=[AgentDef(id="mock-e2e:stub", provider="mock-e2e", model="stub")],
        store=store,
        config=RunnerConfig(trials_per_task=1, concurrency=1),
    )
    run = await runner.execute(notes="unit test")
    assert run.suite_name == "smoke"

    results = store.list_results(run.id)
    assert len(results) == 2
    by_task = {r["task_id"]: r for r in results}
    assert by_task["say_hello"]["status"] == "ok"
    assert by_task["say_hello"]["score"] == 1.0
    assert by_task["arithmetic"]["status"] == "ok"
    assert by_task["arithmetic"]["score"] == 1.0
