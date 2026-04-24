from __future__ import annotations

import asyncio
import re
from pathlib import Path

from agentlab.core.types import ScorerResult, Task, Trajectory


class PyTestScorer:
    """Run pytest in the workspace. Score = passed / collected."""

    kind = "pytest"

    _SUMMARY = re.compile(
        r"(?:(\d+) passed)?"
        r"(?:.*?(\d+) failed)?"
        r"(?:.*?(\d+) error)?"
    )

    def __init__(
        self,
        *,
        cwd: str = ".",
        args: list[str] | None = None,
        timeout_s: int = 120,
        weight: float = 1.0,
    ) -> None:
        self.cwd = cwd
        self.args = args or ["-q"]
        self.timeout_s = timeout_s
        self.weight = weight

    async def score(
        self, task: Task, trajectory: Trajectory, workspace: Path
    ) -> ScorerResult:
        target = (workspace / self.cwd).resolve()
        proc = await asyncio.create_subprocess_exec(
            "pytest",
            *self.args,
            cwd=target,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), self.timeout_s)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return ScorerResult(
                scorer=self.kind, score=0.0, weight=self.weight,
                detail={"error": "timeout"},
            )

        text = out.decode("utf-8", "replace")
        passed, failed, errored = 0, 0, 0
        for m in re.finditer(r"(\d+) (passed|failed|errors?|error)", text):
            n = int(m.group(1))
            kind = m.group(2)
            if kind == "passed":
                passed += n
            elif kind == "failed":
                failed += n
            else:
                errored += n
        total = passed + failed + errored
        score = passed / total if total else 0.0
        return ScorerResult(
            scorer=self.kind,
            score=score,
            weight=self.weight,
            detail={
                "passed": passed,
                "failed": failed,
                "errors": errored,
                "returncode": proc.returncode,
                "tail": text[-1500:],
            },
        )
