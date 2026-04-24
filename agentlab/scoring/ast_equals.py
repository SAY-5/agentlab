from __future__ import annotations

import ast
from pathlib import Path

from agentlab.core.types import ScorerResult, Task, Trajectory


class AstEqualsScorer:
    """Compare two Python snippets structurally (whitespace + comments insensitive)."""

    kind = "ast_equals"

    def __init__(self, expected: str, *, weight: float = 1.0) -> None:
        self.expected = expected
        self.weight = weight

    async def score(
        self, task: Task, trajectory: Trajectory, workspace: Path
    ) -> ScorerResult:
        got = trajectory.final_answer or ""
        try:
            a_tree = ast.parse(got)
            b_tree = ast.parse(self.expected)
        except SyntaxError as e:
            return ScorerResult(
                scorer=self.kind,
                score=0.0,
                weight=self.weight,
                detail={"parse_error": str(e)},
            )
        score = 1.0 if ast.dump(a_tree) == ast.dump(b_tree) else 0.0
        return ScorerResult(scorer=self.kind, score=score, weight=self.weight)
