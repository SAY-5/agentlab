"""File IO tools. Path-traversal-safe: all operations stay within workspace."""

from __future__ import annotations

import os
from pathlib import Path

from agentlab.core.types import ToolSpec


def _resolve_within(workspace: str | None, rel: str) -> Path:
    if not workspace:
        raise ValueError("no workspace configured for this task")
    root = Path(workspace).resolve()
    target = (root / rel).resolve()
    # Ensure target is inside root.
    try:
        target.relative_to(root)
    except ValueError as e:
        raise ValueError(f"path escapes workspace: {rel}") from e
    return target


class FileReadTool:
    spec = ToolSpec(
        name="file_read",
        description="Read a UTF-8 text file from the task workspace.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to workspace"}
            },
            "required": ["path"],
        },
    )

    async def run(self, args: dict, *, workspace: str | None) -> str:
        path = _resolve_within(workspace, str(args.get("path", "")))
        if not path.exists():
            return f"error: {path} does not exist"
        if path.is_dir():
            return "\n".join(sorted(os.listdir(path)))
        try:
            return path.read_text("utf-8")
        except UnicodeDecodeError:
            return f"error: {path} is not UTF-8 text"


class FileWriteTool:
    spec = ToolSpec(
        name="file_write",
        description="Write a UTF-8 text file in the task workspace (creates directories as needed).",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    )

    async def run(self, args: dict, *, workspace: str | None) -> str:
        path = _resolve_within(workspace, str(args.get("path", "")))
        path.parent.mkdir(parents=True, exist_ok=True)
        content = str(args.get("content", ""))
        path.write_text(content, "utf-8")
        return f"wrote {len(content)} bytes to {path.relative_to(Path(workspace or '.').resolve())}"
