from __future__ import annotations

import re
from pathlib import Path

from agentlab.core.types import ScorerResult, Task, Trajectory


class RegexMatchScorer:
    kind = "regex_match"

    def __init__(self, patterns: list[str], *, weight: float = 1.0, mode: str = "any") -> None:
        self.patterns = [re.compile(p, re.MULTILINE) for p in patterns]
        self.weight = weight
        self.mode = mode  # "any" or "all"

    async def score(
        self, task: Task, trajectory: Trajectory, workspace: Path
    ) -> ScorerResult:
        text = trajectory.final_answer or ""
        hits = [bool(p.search(text)) for p in self.patterns]
        aggregate = all if self.mode == "all" else any
        score = 1.0 if aggregate(hits) else 0.0
        return ScorerResult(
            scorer=self.kind,
            score=score,
            weight=self.weight,
            detail={"hits": hits},
        )
