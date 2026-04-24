from __future__ import annotations

import asyncio
from pathlib import Path

from agentlab.core.types import ScorerResult, Task, Trajectory


class DiffSizeScorer:
    """Score linearly by how small the diff against git HEAD is.

    Score = 1.0 when diff is 0 lines, 0.0 when diff >= max_lines.
    """

    kind = "diff_size"

    def __init__(self, max_lines: int = 100, *, weight: float = 1.0) -> None:
        self.max_lines = max_lines
        self.weight = weight

    async def score(
        self, task: Task, trajectory: Trajectory, workspace: Path
    ) -> ScorerResult:
        changed = await _count_diff_lines(workspace)
        score = (
            0.0 if changed >= self.max_lines
            else max(0.0, 1.0 - changed / self.max_lines)
        )
        return ScorerResult(
            scorer=self.kind,
            score=score,
            weight=self.weight,
            detail={"changed_lines": changed, "max_lines": self.max_lines},
        )


async def _count_diff_lines(workspace: Path) -> int:
    proc = await asyncio.create_subprocess_exec(
        "git",
        "diff",
        "--stat=999",
        cwd=workspace,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await proc.communicate()
    total = 0
    for line in out.decode("utf-8", "replace").splitlines():
        # lines look like " filename.py | 12 ++++++------"
        parts = line.split("|")
        if len(parts) >= 2:
            tail = parts[1].strip().split()
            if tail and tail[0].isdigit():
                total += int(tail[0])
    return total
