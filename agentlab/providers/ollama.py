"""Ollama local provider via HTTP."""

from __future__ import annotations

from typing import Any

import aiohttp

from agentlab.core.types import Completion, Message, ToolSpec, Usage


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self.base_url = base_url.rstrip("/")

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
        ollama_messages: list[dict[str, Any]] = []
        if system:
            ollama_messages.append({"role": "system", "content": system})
        for m in messages:
            if m.role == "tool":
                # Ollama: fold tool result into an assistant observation.
                ollama_messages.append({"role": "user", "content": f"[tool:{m.tool_call_id}] {m.content}"})
            else:
                ollama_messages.append({"role": m.role, "content": m.content})

        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens:
            options["num_predict"] = max_tokens
        if stop:
            options["stop"] = stop

        payload: dict[str, Any] = {
            "model": model,
            "messages": ollama_messages,
            "stream": False,
            "options": options,
        }
        async with aiohttp.ClientSession() as sess, sess.post(
            f"{self.base_url}/api/chat", json=payload, timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        msg_content = data.get("message", {}).get("content", "")
        prompt_eval = data.get("prompt_eval_count", 0)
        eval_count = data.get("eval_count", 0)
        return Completion(
            message=Message(role="assistant", content=msg_content),
            stop_reason="stop" if data.get("done") else "length",
            usage=Usage(input_tokens=prompt_eval, output_tokens=eval_count),
            raw=data,
        )
