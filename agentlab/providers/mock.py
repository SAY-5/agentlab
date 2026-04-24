"""Deterministic mock provider used by tests and for offline work."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agentlab.core.types import Completion, Message, ToolCall, ToolSpec, Usage


class MockProvider:
    """A mock provider that returns canned completions.

    Configure by setting ``responses`` to a list of strings or callables; each
    call to ``complete`` consumes one entry. A callable is invoked with the
    messages and must return a Completion.
    """

    name = "mock"

    def __init__(
        self,
        responses: list[str | Completion | Callable[[list[Message]], Completion]] | None = None,
    ) -> None:
        self.responses: list[str | Completion | Callable[[list[Message]], Completion]] = (
            responses or []
        )
        self._idx = 0
        self.calls: list[dict[str, Any]] = []

    def enqueue(
        self, response: str | Completion | Callable[[list[Message]], Completion]
    ) -> None:
        self.responses.append(response)

    def enqueue_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        call_id: str = "call_mock_1",
    ) -> None:
        tc = ToolCall(id=call_id, name=tool_name, arguments=arguments)
        self.responses.append(
            Completion(
                message=Message(role="assistant", tool_calls=[tc]),
                stop_reason="tool_use",
                usage=Usage(input_tokens=10, output_tokens=5),
            )
        )

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        system: str | None = None,
        stop: list[str] | None = None,
    ) -> Completion:
        self.calls.append(
            {
                "messages": [m.model_dump() for m in messages],
                "model": model,
                "temperature": temperature,
                "tools": [t.model_dump() for t in tools] if tools else None,
                "system": system,
            }
        )
        if self._idx >= len(self.responses):
            return Completion(
                message=Message(role="assistant", content="(mock: no response queued)"),
                usage=Usage(input_tokens=1, output_tokens=1),
            )
        r = self.responses[self._idx]
        self._idx += 1
        if callable(r):
            return r(messages)
        if isinstance(r, Completion):
            return r
        return Completion(
            message=Message(role="assistant", content=r),
            usage=Usage(input_tokens=10, output_tokens=len(r.split())),
        )
