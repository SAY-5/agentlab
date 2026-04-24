"""File IO tools. Path-traversal-safe: all operations stay within workspace."""

from __future__ import annotations

import os
from pathlib import Path

from agentlab.core.types import ToolSpec


def _resolve_within(workspace: str | None, rel: str) -> Path:
    """Return ``workspace / rel`` resolved, guaranteed to stay inside workspace.

    Rejects:
      * absolute ``rel`` paths (they'd bypass the workspace root entirely),
      * paths that escape via ``..`` (``resolve()`` normalizes them; we then
        verify with ``is_relative_to`` against the resolved workspace root),
      * symlinks anywhere on the path whose target lies outside the
        workspace — ``resolve()`` follows symlinks, and if the resolved
        target is outside the workspace it fails the relative_to check.

    Note we intentionally resolve the workspace itself to its canonical
    form first; if the workspace *is* a symlink the resolved root is its
    target, not the symlink source, so subsequent comparisons stay
    internally consistent.
    """
    if not workspace:
        raise ValueError("no workspace configured for this task")
    if Path(rel).is_absolute():
        raise ValueError(f"absolute paths not permitted: {rel!r}")
    root = Path(workspace).resolve(strict=False)
    target = (root / rel).resolve(strict=False)
    if not _is_relative_to(target, root):
        raise ValueError(f"path escapes workspace: {rel!r}")
    # Walk each intermediate component explicitly. A directory symlink whose
    # target is outside the workspace would already have been caught above
    # via ``target.resolve()``, but this gives a clearer error message.
    probe = root / rel
    for parent in [probe, *probe.parents]:
        if parent == root or not parent.exists() and not parent.is_symlink():
            continue
        if parent.is_symlink():
            linked = parent.resolve(strict=False)
            if not _is_relative_to(linked, root):
                raise ValueError(
                    f"path traverses symlink leaving workspace: {rel!r}"
                )
    return target


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


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
