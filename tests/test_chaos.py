"""Fault-injection / chaos tests.

The real cost of an evaluation harness isn't the happy path — it's how
it behaves when providers rate-limit, networks hiccup, strategies hang,
and tools fail. These tests assert the runner's resilience envelope:
retries fire on transient errors, hangs are bounded by timeouts, and
scorers don't contaminate each other on failure.
"""

from __future__ import annotations

import asyncio
import random
import time
from pathlib import Path

import pytest

from agentlab.core.types import AgentDef, Completion, Message, Usage
from agentlab.providers import register as register_provider
from agentlab.providers.mock import MockProvider
from agentlab.runners import Runner, RunnerConfig, RetryPolicy
from agentlab.store import Store
from agentlab.suites import load_suite


class FlakyProvider:
    """Provider that deterministically injects failures.

    The ``pattern`` is a string where:
      * ``.`` = succeed
      * ``T`` = raise TimeoutError
      * ``E`` = raise RuntimeError("transient")
      * ``S`` = sleep forever (simulates a hang)
    A cursor walks the pattern across calls; the test asserts the runner
    responds appropriately.
    """

    name = "flaky"

    def __init__(self, pattern: str, response: str = "ok") -> None:
        self.pattern = pattern
        self.response = response
        self.cursor = 0
        self.calls = 0

    async def complete(self, messages, *, model, tools=None, temperature=0.0,
                        max_tokens=None, system=None, stop=None):  # noqa: D401
        self.calls += 1
        kind = self.pattern[self.cursor % len(self.pattern)]
        self.cursor += 1
        if kind == "T":
            raise TimeoutError("injected timeout")
        if kind == "E":
            raise RuntimeError("transient")
        if kind == "S":
            await asyncio.sleep(1e6)  # forever
        return Completion(
            message=Message(role="assistant", content=self.response),
            usage=Usage(input_tokens=5, output_tokens=2),
        )


def _suite_path() -> Path:
    return Path(__file__).resolve().parent.parent / "examples" / "suites" / "smoke.yaml"


@pytest.mark.asyncio
async def test_retry_succeeds_after_two_transient_failures(tmp_path: Path):
    """Third attempt should succeed when the first two fail transiently."""
    prov = FlakyProvider("EE.")
    register_provider("flaky-retry", prov)
    store = Store(tmp_path / "runs.db")
    suite = load_suite(_suite_path())
    # Only run one task to keep the test deterministic.
    suite.tasks = suite.tasks[:1]
    runner = Runner(
        suite=suite,
        agents=[AgentDef(id="flaky-retry:m", provider="flaky-retry", model="m")],
        store=store,
        config=RunnerConfig(
            trials_per_task=1,
            concurrency=1,
            retry=RetryPolicy(max_attempts=3, base_delay_s=0.01),
        ),
    )
    run = await runner.execute()
    results = store.list_results(run.id)
    assert len(results) == 1
    assert results[0]["status"] == "ok"
    # 2 failures + 1 success = 3 calls total.
    assert prov.calls == 3


@pytest.mark.asyncio
async def test_max_attempts_exhausted_records_error(tmp_path: Path):
    """When every attempt fails, we record an error rather than crash."""
    prov = FlakyProvider("EEE")
    register_provider("flaky-exhaust", prov)
    store = Store(tmp_path / "runs.db")
    suite = load_suite(_suite_path())
    suite.tasks = suite.tasks[:1]
    runner = Runner(
        suite=suite,
        agents=[AgentDef(id="flaky-exhaust:m", provider="flaky-exhaust", model="m")],
        store=store,
        config=RunnerConfig(
            trials_per_task=1,
            concurrency=1,
            retry=RetryPolicy(max_attempts=3, base_delay_s=0.01),
        ),
    )
    run = await runner.execute()
    results = store.list_results(run.id)
    assert results[0]["status"] == "error"
    assert "transient" in (results[0]["error"] or "").lower()


@pytest.mark.asyncio
async def test_hanging_provider_is_terminated_by_timeout(tmp_path: Path):
    """A strategy hung on a sleeping provider must be cancelled within timeout_s.

    Previously wait_for leaked the hung coroutine; asyncio.timeout cancels it.
    """
    prov = FlakyProvider("S")
    register_provider("flaky-hang", prov)
    store = Store(tmp_path / "runs.db")
    suite = load_suite(_suite_path())
    suite.tasks = suite.tasks[:1]
    # Shrink timeout so the test runs in seconds, not minutes.
    suite.tasks[0].timeout_s = 1
    runner = Runner(
        suite=suite,
        agents=[AgentDef(id="flaky-hang:m", provider="flaky-hang", model="m")],
        store=store,
        config=RunnerConfig(
            trials_per_task=1,
            concurrency=1,
            retry=RetryPolicy(max_attempts=1, base_delay_s=0),
        ),
    )
    started = time.monotonic()
    run = await runner.execute()
    elapsed = time.monotonic() - started
    results = store.list_results(run.id)
    assert results[0]["status"] == "timeout"
    # Runner must not hang beyond the configured timeout + a small buffer.
    assert elapsed < 5.0, f"runner took {elapsed:.1f}s for a 1s-timeout task"


@pytest.mark.asyncio
async def test_concurrent_tasks_dont_contaminate_each_other(tmp_path: Path):
    """Mixed success/failure agents run concurrently without cross-talk."""
    good = MockProvider(["ok"] * 100)
    bad = FlakyProvider("EEE", response="unreachable")
    register_provider("chaos-good", good)
    register_provider("chaos-bad", bad)
    store = Store(tmp_path / "runs.db")
    suite = load_suite(_suite_path())
    runner = Runner(
        suite=suite,
        agents=[
            AgentDef(id="chaos-good:m", provider="chaos-good", model="m"),
            AgentDef(id="chaos-bad:m", provider="chaos-bad", model="m"),
        ],
        store=store,
        config=RunnerConfig(
            trials_per_task=3,
            concurrency=6,
            retry=RetryPolicy(max_attempts=2, base_delay_s=0.01),
        ),
    )
    run = await runner.execute()
    results = store.list_results(run.id)
    good_rows = [r for r in results if r["agent_id"] == "chaos-good:m"]
    bad_rows = [r for r in results if r["agent_id"] == "chaos-bad:m"]
    assert all(r["status"] == "ok" for r in good_rows)
    assert all(r["status"] == "error" for r in bad_rows)


@pytest.mark.asyncio
async def test_random_chaos_respects_invariants(tmp_path: Path):
    """Random-pattern chaos: final row count equals tasks × agents × trials,
    and no status is ever None or missing."""
    rng = random.Random(42)
    pattern = "".join(rng.choice("E.") for _ in range(40))
    prov = FlakyProvider(pattern)
    register_provider("chaos-rand", prov)
    store = Store(tmp_path / "runs.db")
    suite = load_suite(_suite_path())
    trials = 2
    runner = Runner(
        suite=suite,
        agents=[AgentDef(id="chaos-rand:m", provider="chaos-rand", model="m")],
        store=store,
        config=RunnerConfig(
            trials_per_task=trials,
            concurrency=4,
            retry=RetryPolicy(max_attempts=3, base_delay_s=0.01),
        ),
    )
    run = await runner.execute()
    results = store.list_results(run.id)
    expected = len(suite.tasks) * trials
    assert len(results) == expected
    assert all(r["status"] in ("ok", "error", "timeout") for r in results)
