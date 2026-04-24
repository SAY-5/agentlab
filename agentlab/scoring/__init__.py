"""Scorers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from agentlab.core.types import ScorerResult, Task, Trajectory


class Scorer(Protocol):
    kind: str

    async def score(
        self,
        task: Task,
        trajectory: Trajectory,
        workspace: Path,
    ) -> ScorerResult: ...


_REGISTRY: dict[str, type[Scorer]] = {}


def register(kind: str, cls: type[Scorer]) -> None:
    _REGISTRY[kind] = cls


def build(spec: dict[str, Any]) -> Scorer:
    kind = spec.get("kind")
    if kind not in _REGISTRY:
        raise KeyError(f"unknown scorer kind {kind!r}")
    cls = _REGISTRY[kind]
    return cls(**{k: v for k, v in spec.items() if k != "kind"})


from .ast_equals import AstEqualsScorer  # noqa: E402
from .diff_size import DiffSizeScorer  # noqa: E402
from .pytest_scorer import PyTestScorer  # noqa: E402
from .regex_match import RegexMatchScorer  # noqa: E402
from .rubric import RubricScorer  # noqa: E402
from .string_equals import StringEqualsScorer  # noqa: E402

register("regex_match", RegexMatchScorer)
register("string_equals", StringEqualsScorer)
register("diff_size", DiffSizeScorer)
register("pytest", PyTestScorer)
register("rubric", RubricScorer)
register("ast_equals", AstEqualsScorer)
