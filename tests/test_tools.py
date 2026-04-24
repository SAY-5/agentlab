from pathlib import Path

import pytest

from agentlab.tools.file_tools import FileReadTool, FileWriteTool, _resolve_within
from agentlab.tools.shell import ShellDisabledError, ShellTool


@pytest.mark.asyncio
async def test_file_write_then_read(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    w = FileWriteTool()
    r = FileReadTool()
    await w.run({"path": "a/b.txt", "content": "hello"}, workspace=str(ws))
    out = await r.run({"path": "a/b.txt"}, workspace=str(ws))
    assert out == "hello"


def test_path_traversal_rejected(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    with pytest.raises(ValueError):
        _resolve_within(str(ws), "../etc/passwd")


def test_absolute_path_rejected(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    with pytest.raises(ValueError):
        _resolve_within(str(ws), "/etc/passwd")


def test_symlink_escape_rejected(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("sensitive")
    # A symlink inside the workspace pointing to an outside file.
    (ws / "link").symlink_to(outside / "secret.txt")
    with pytest.raises(ValueError):
        _resolve_within(str(ws), "link")


@pytest.mark.asyncio
async def test_file_read_missing(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    r = FileReadTool()
    out = await r.run({"path": "nope.txt"}, workspace=str(ws))
    assert "does not exist" in out


@pytest.mark.asyncio
async def test_shell_disabled_by_default(tmp_path: Path):
    s = ShellTool()  # no opt-in
    with pytest.raises(ShellDisabledError):
        await s.run({"cmd": "ls"}, workspace=str(tmp_path))


@pytest.mark.asyncio
async def test_shell_runs_in_workspace_when_enabled(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "marker.txt").write_text("x")
    s = ShellTool(allow_shell=True)
    out = await s.run({"cmd": "ls"}, workspace=str(ws))
    assert "marker.txt" in out
    assert "exit=0" in out


@pytest.mark.asyncio
async def test_shell_timeout(tmp_path: Path):
    s = ShellTool(allow_shell=True)
    out = await s.run({"cmd": "sleep 5", "timeout_s": 0.3}, workspace=str(tmp_path))
    assert "timeout" in out
