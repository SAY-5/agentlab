from pathlib import Path

import pytest

from agentlab.tools.file_tools import FileReadTool, FileWriteTool, _resolve_within
from agentlab.tools.shell import ShellTool


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


@pytest.mark.asyncio
async def test_file_read_missing(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    r = FileReadTool()
    out = await r.run({"path": "nope.txt"}, workspace=str(ws))
    assert "does not exist" in out


@pytest.mark.asyncio
async def test_shell_runs_in_workspace(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "marker.txt").write_text("x")
    s = ShellTool()
    out = await s.run({"cmd": "ls"}, workspace=str(ws))
    assert "marker.txt" in out
    assert "exit=0" in out


@pytest.mark.asyncio
async def test_shell_timeout(tmp_path: Path):
    s = ShellTool()
    out = await s.run({"cmd": "sleep 5", "timeout_s": 0.3}, workspace=str(tmp_path))
    assert "timeout" in out
