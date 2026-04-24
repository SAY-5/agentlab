from pathlib import Path

import pytest

from agentlab.core.types import Task, Trajectory
from agentlab.scoring.ast_equals import AstEqualsScorer
from agentlab.scoring.regex_match import RegexMatchScorer
from agentlab.scoring.string_equals import StringEqualsScorer


def _task():
    return Task(id="t", description="", prompt="")


def _traj(answer: str):
    return Trajectory(agent_id="a", task_id="t", trial_idx=0, final_answer=answer)


@pytest.mark.asyncio
async def test_regex_match_any():
    s = RegexMatchScorer(["hello|hi"])
    res = await s.score(_task(), _traj("hi there"), Path("."))
    assert res.score == 1.0


@pytest.mark.asyncio
async def test_regex_match_missing():
    s = RegexMatchScorer(["^foo$"])
    res = await s.score(_task(), _traj("bar"), Path("."))
    assert res.score == 0.0


@pytest.mark.asyncio
async def test_regex_match_all_requires_every_pattern():
    s = RegexMatchScorer(["one", "two"], mode="all")
    a = await s.score(_task(), _traj("one"), Path("."))
    b = await s.score(_task(), _traj("one two"), Path("."))
    assert a.score == 0.0
    assert b.score == 1.0


@pytest.mark.asyncio
async def test_string_equals_with_strip():
    s = StringEqualsScorer("42")
    res = await s.score(_task(), _traj(" 42\n"), Path("."))
    assert res.score == 1.0


@pytest.mark.asyncio
async def test_ast_equals_ignores_whitespace_and_comments():
    s = AstEqualsScorer("def f(x):\n    return x + 1\n")
    code = "def f(x):\n    # tiny comment\n    return   x  +  1"
    res = await s.score(_task(), _traj(code), Path("."))
    assert res.score == 1.0


@pytest.mark.asyncio
async def test_ast_equals_detects_logical_diff():
    s = AstEqualsScorer("def f(x):\n    return x + 1\n")
    bad = "def f(x):\n    return x - 1\n"
    res = await s.score(_task(), _traj(bad), Path("."))
    assert res.score == 0.0


@pytest.mark.asyncio
async def test_ast_equals_syntax_error_returns_zero():
    s = AstEqualsScorer("def f(): pass")
    res = await s.score(_task(), _traj("def f(:::"), Path("."))
    assert res.score == 0.0
