"""OpenAI provider adapter.

Uses the official ``openai`` Python SDK if present; falls back to a clear
ImportError with install instructions.
"""

from __future__ import annotations

import json
from typing import Any

from agentlab.core.types import Completion, Message, ToolCall, ToolSpec, Usage


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        try:
            from openai import AsyncOpenAI  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "OpenAI provider requires the openai package; `pip install agentlab[openai]`"
            ) from e
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**kwargs)

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
        oai_messages: list[dict[str, Any]] = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        for m in messages:
            if m.role == "tool":
                oai_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": m.tool_call_id or "",
                        "content": m.content,
                    }
                )
            elif m.role == "assistant" and m.tool_calls:
                oai_messages.append(
                    {
                        "role": "assistant",
                        "content": m.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments),
                                },
                            }
                            for tc in m.tool_calls
                        ],
                    }
                )
            else:
                oai_messages.append({"role": m.role, "content": m.content})

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": oai_messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if stop:
            kwargs["stop"] = stop
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]

        resp = await self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        tc_out: list[ToolCall] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}
                tc_out.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        stop_reason_raw = choice.finish_reason or "stop"
        stop_reason = {
            "stop": "stop",
            "length": "length",
            "tool_calls": "tool_use",
            "content_filter": "content_filter",
        }.get(stop_reason_raw, "other")

        return Completion(
            message=Message(
                role="assistant",
                content=choice.message.content or "",
                tool_calls=tc_out,
            ),
            stop_reason=stop_reason,  # type: ignore[arg-type]
            usage=Usage(
                input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                output_tokens=resp.usage.completion_tokens if resp.usage else 0,
            ),
            raw=resp.model_dump() if hasattr(resp, "model_dump") else None,
        )
