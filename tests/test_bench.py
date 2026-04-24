"""Performance benchmarks via pytest-benchmark.

Run with: ``pytest tests/test_bench.py --benchmark-only``.

Skipped by default in the regular test run to keep ``pytest -q`` fast;
opt in with the flag above.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from agentlab.core.types import AgentDef, Result, Run
from agentlab.scoring.ast_equals import AstEqualsScorer
from agentlab.scoring.regex_match import RegexMatchScorer
from agentlab.scoring.string_equals import StringEqualsScorer
from agentlab.store import Store

pytest_benchmark = pytest.importorskip("pytest_benchmark")  # optional dep

pytestmark = pytest.mark.benchmark


def _trajectory_text(n: int) -> str:
    return "\n".join(f"assistant turn {i}: here's a thought" for i in range(n))


def test_bench_regex_match_1k_chars(benchmark):
    from agentlab.core.types import Task, Trajectory

    s = RegexMatchScorer(["\\bthought\\b"])
    t = Trajectory(agent_id="a", task_id="t", trial_idx=0, final_answer=_trajectory_text(40))
    task = Task(id="t", description="", prompt="")
    benchmark(lambda: __import__("asyncio").run(s.score(task, t, Path("."))))


def test_bench_string_equals(benchmark):
    from agentlab.core.types import Task, Trajectory

    s = StringEqualsScorer("42")
    task = Task(id="t", description="", prompt="")
    t = Trajectory(agent_id="a", task_id="t", trial_idx=0, final_answer="42")
    benchmark(lambda: __import__("asyncio").run(s.score(task, t, Path("."))))


def test_bench_ast_equals_small_program(benchmark):
    from agentlab.core.types import Task, Trajectory

    prog = "def f(x):\n    return x + 1\n"
    s = AstEqualsScorer(prog)
    task = Task(id="t", description="", prompt="")
    t = Trajectory(agent_id="a", task_id="t", trial_idx=0, final_answer=prog)
    benchmark(lambda: __import__("asyncio").run(s.score(task, t, Path("."))))


def test_bench_store_insert_result(benchmark, tmp_path):
    store = Store(tmp_path / "bench.db")
    run = Run(
        id="run_bench",
        started_at=time.time(),
        finished_at=None,
        suite_name="bench",
        suite_version="1",
        agents=[AgentDef(id="mock:m", provider="mock", model="m")],
    )
    store.insert_run(run)
    trial = {"counter": 0}

    def bench_it():
        trial["counter"] += 1
        store.insert_result(
            Result(
                run_id="run_bench",
                task_id="t1",
                agent_id="mock:m",
                trial_idx=trial["counter"],
                started_at=time.time(),
                finished_at=time.time(),
                status="ok",
                score=0.5,
                tokens_in=10,
                tokens_out=5,
                trajectory=[{"kind": "model", "content": "hi"}] * 20,
            )
        )

    benchmark(bench_it)


def test_bench_store_list_results(benchmark, tmp_path):
    store = Store(tmp_path / "bench_list.db")
    run = Run(
        id="run_L",
        started_at=time.time(),
        finished_at=None,
        suite_name="bench",
        suite_version="1",
        agents=[AgentDef(id="mock:m", provider="mock", model="m")],
    )
    store.insert_run(run)
    for i in range(200):
        store.insert_result(
            Result(
                run_id="run_L",
                task_id=f"t{i}",
                agent_id="mock:m",
                trial_idx=0,
                started_at=time.time(),
                finished_at=time.time(),
                status="ok",
                score=0.5,
                tokens_in=1,
                tokens_out=1,
                trajectory=[],
            )
        )
    benchmark(lambda: store.list_results("run_L"))
