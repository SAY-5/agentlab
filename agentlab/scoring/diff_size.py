from __future__ import annotations

import asyncio
from pathlib import Path

from agentlab.core.types import ScorerResult, Task, Trajectory


class DiffSizeScorer:
    """Score linearly by how small the diff against git HEAD is.

    Score = 1.0 when diff is 0 lines, 0.0 when diff >= max_lines.

    If the workspace is not a git repository (or `git` isn't available),
    the scorer returns 0.0 with an explanatory ``detail.error`` rather
    than silently returning 1.0 for an un-evaluable workspace.
    """

    kind = "diff_size"

    def __init__(self, max_lines: int = 100, *, weight: float = 1.0) -> None:
        self.max_lines = max_lines
        self.weight = weight

    async def score(
        self, task: Task, trajectory: Trajectory, workspace: Path
    ) -> ScorerResult:
        ok, changed = await _count_diff_lines(workspace)
        if not ok:
            return ScorerResult(
                scorer=self.kind,
                score=0.0,
                weight=self.weight,
                detail={"error": "workspace is not a git repo or git is unavailable"},
            )
        score = (
            0.0
            if changed >= self.max_lines
            else max(0.0, 1.0 - changed / self.max_lines)
        )
        return ScorerResult(
            scorer=self.kind,
            score=score,
            weight=self.weight,
            detail={"changed_lines": changed, "max_lines": self.max_lines},
        )


async def _count_diff_lines(workspace: Path) -> tuple[bool, int]:
    """Return (ok, changed_line_count). ok=False when git or repo missing."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "diff",
            "--stat=999",
            cwd=workspace,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return False, 0
    out, err = await proc.communicate()
    if proc.returncode != 0:
        # Typical: "fatal: not a git repository". Treat as un-evaluable.
        return False, 0
    total = 0
    for line in out.decode("utf-8", "replace").splitlines():
        # lines look like " filename.py | 12 ++++++------"
        parts = line.split("|")
        if len(parts) >= 2:
            tail = parts[1].strip().split()
            if tail and tail[0].isdigit():
                total += int(tail[0])
    return True, total
