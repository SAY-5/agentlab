"""Shell-run tool. Runs locally by default; embedders can swap in a
container-backed runner for stricter isolation."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

from agentlab.core.types import ToolSpec


@dataclass
class ShellResult:
    stdout: str
    stderr: str
    code: int

    def __str__(self) -> str:
        return (
            f"exit={self.code}\n"
            f"stdout:\n{self.stdout}\n"
            + (f"stderr:\n{self.stderr}\n" if self.stderr else "")
        )


class ShellTool:
    spec = ToolSpec(
        name="shell",
        description="Run a shell command from the task workspace. Timeout 30s default.",
        parameters={
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "description": "Shell command"},
                "timeout_s": {"type": "number", "default": 30},
            },
            "required": ["cmd"],
        },
    )

    async def run(self, args: dict, *, workspace: str | None) -> str:
        cmd = str(args.get("cmd", ""))
        timeout_s = float(args.get("timeout_s", 30))
        env = {**os.environ, "AGENTLAB_WORKSPACE": workspace or ""}
        proc = await asyncio.create_subprocess_shell(
            cmd,
            cwd=workspace or None,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout_s)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return f"error: timeout after {timeout_s}s"
        return str(
            ShellResult(
                stdout=(stdout or b"").decode("utf-8", "replace")[-8000:],
                stderr=(stderr or b"").decode("utf-8", "replace")[-2000:],
                code=proc.returncode or 0,
            )
        )
