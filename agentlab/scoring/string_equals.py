from __future__ import annotations

from pathlib import Path

from agentlab.core.types import ScorerResult, Task, Trajectory


class StringEqualsScorer:
    kind = "string_equals"

    def __init__(self, expected: str, *, weight: float = 1.0, strip: bool = True) -> None:
        self.expected = expected
        self.weight = weight
        self.strip = strip

    async def score(
        self, task: Task, trajectory: Trajectory, workspace: Path
    ) -> ScorerResult:
        got = trajectory.final_answer or ""
        if self.strip:
            got = got.strip()
            expected = self.expected.strip()
        else:
            expected = self.expected
        score = 1.0 if got == expected else 0.0
        return ScorerResult(
            scorer=self.kind,
            score=score,
            weight=self.weight,
            detail={"got": got[:200], "expected": expected[:200]},
        )
