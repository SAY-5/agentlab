"""Anthropic provider adapter."""

from __future__ import annotations

from typing import Any

from agentlab.core.types import Completion, Message, ToolCall, ToolSpec, Usage


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str | None = None) -> None:
        try:
            from anthropic import AsyncAnthropic  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "Anthropic provider requires the anthropic package; "
                "`pip install agentlab[anthropic]`"
            ) from e
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        self._client = AsyncAnthropic(**kwargs)

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
        anthropic_messages: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                # Anthropic takes system as a separate top-level arg.
                continue
            if m.role == "tool":
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.tool_call_id or "",
                                "content": m.content,
                            }
                        ],
                    }
                )
            elif m.role == "assistant" and m.tool_calls:
                blocks: list[dict[str, Any]] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                anthropic_messages.append({"role": "assistant", "content": blocks})
            else:
                anthropic_messages.append({"role": m.role, "content": m.content})

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or 1024,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system
        if stop:
            kwargs["stop_sequences"] = stop
        if tools:
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in tools
            ]

        resp = await self._client.messages.create(**kwargs)

        content_text = ""
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                content_text += block.text
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        stop_reason = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "tool_use": "tool_use",
        }.get(resp.stop_reason or "end_turn", "other")

        return Completion(
            message=Message(role="assistant", content=content_text, tool_calls=tool_calls),
            stop_reason=stop_reason,  # type: ignore[arg-type]
            usage=Usage(
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
            ),
            raw=resp.model_dump() if hasattr(resp, "model_dump") else None,
        )
