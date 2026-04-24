import pytest

from agentlab.core.types import Message
from agentlab.providers.mock import MockProvider


@pytest.mark.asyncio
async def test_mock_returns_canned_string():
    p = MockProvider(["hello world"])
    c = await p.complete([Message(role="user", content="hi")], model="mock-m")
    assert c.message.content == "hello world"
    assert c.usage.output_tokens == 2


@pytest.mark.asyncio
async def test_mock_falls_back_when_queue_empty():
    p = MockProvider()
    c = await p.complete([Message(role="user", content="hi")], model="mock-m")
    assert "mock: no response" in c.message.content


@pytest.mark.asyncio
async def test_mock_records_calls():
    p = MockProvider(["ok"])
    await p.complete(
        [Message(role="user", content="ping")],
        model="mock-m",
        temperature=0.3,
        system="be helpful",
    )
    assert p.calls[0]["model"] == "mock-m"
    assert p.calls[0]["temperature"] == 0.3
    assert p.calls[0]["system"] == "be helpful"


@pytest.mark.asyncio
async def test_mock_tool_call_enqueue():
    p = MockProvider()
    p.enqueue_tool_call("file_read", {"path": "foo.py"})
    c = await p.complete([Message(role="user", content="read foo")], model="mock-m")
    assert c.stop_reason == "tool_use"
    assert c.message.tool_calls[0].name == "file_read"
    assert c.message.tool_calls[0].arguments == {"path": "foo.py"}
