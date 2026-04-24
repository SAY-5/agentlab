"""Shell-run tool for agent-generated commands.

**Danger**: this tool grants the agent arbitrary code execution on the
host. It is **off by default**; a task must opt in by setting
``AGENTLAB_ALLOW_SHELL=1`` in the environment, or by passing
``allow_shell=True`` to ``ShellTool.__init__``. Even when enabled, the
command runs with the invoker's OS privileges — there is no container or
syscall filter. Production users should swap in a container-backed
runner that sets ``cwd`` to a mount, drops capabilities, and disables
network access.
"""

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


class ShellDisabledError(RuntimeError):
    """Raised when the shell tool is invoked without an explicit opt-in."""


class ShellTool:
    spec = ToolSpec(
        name="shell",
        description=(
            "Run a shell command from the task workspace. Timeout 30s default. "
            "This tool is OFF unless AGENTLAB_ALLOW_SHELL=1 or ShellTool was "
            "constructed with allow_shell=True."
        ),
        parameters={
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "description": "Shell command"},
                "timeout_s": {"type": "number", "default": 30},
            },
            "required": ["cmd"],
        },
    )

    def __init__(self, *, allow_shell: bool | None = None) -> None:
        if allow_shell is None:
            allow_shell = os.environ.get("AGENTLAB_ALLOW_SHELL", "") in {
                "1",
                "true",
                "TRUE",
            }
        self._allowed = allow_shell

    async def run(self, args: dict, *, workspace: str | None) -> str:
        if not self._allowed:
            raise ShellDisabledError(
                "shell tool is disabled. Set AGENTLAB_ALLOW_SHELL=1 or construct "
                "ShellTool(allow_shell=True) only if you understand that this "
                "grants the agent arbitrary code execution on the host."
            )
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
