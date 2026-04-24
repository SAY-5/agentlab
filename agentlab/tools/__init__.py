"""Sandboxed tools exposed to agents."""

from __future__ import annotations

from typing import Protocol

from agentlab.core.types import ToolSpec


class Tool(Protocol):
    spec: ToolSpec

    async def run(self, args: dict, *, workspace: str | None) -> str: ...


_REGISTRY: dict[str, Tool] = {}


def register(tool: Tool) -> None:
    _REGISTRY[tool.spec.name] = tool


def get_tool(name: str) -> Tool | None:
    return _REGISTRY.get(name)


def tool_specs(names: list[str]) -> list[ToolSpec]:
    out: list[ToolSpec] = []
    for n in names:
        t = _REGISTRY.get(n)
        if t is not None:
            out.append(t.spec)
    return out


from .file_tools import FileReadTool, FileWriteTool  # noqa: E402
from .shell import ShellTool  # noqa: E402

register(FileReadTool())
register(FileWriteTool())
register(ShellTool())
