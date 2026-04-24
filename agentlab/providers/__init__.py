"""Provider adapters for calling LLMs.

Add a provider by implementing the ``Provider`` protocol and registering
in ``REGISTRY``. The runner resolves ``AgentDef.provider`` through this.
"""

from __future__ import annotations

from typing import Protocol

from agentlab.core.types import Completion, Message, ToolSpec


class Provider(Protocol):
    name: str

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
    ) -> Completion: ...


_REGISTRY: dict[str, Provider] = {}


def register(name: str, provider: Provider) -> None:
    _REGISTRY[name] = provider


def get(name: str) -> Provider:
    if name not in _REGISTRY:
        raise KeyError(f"unknown provider: {name!r}. Registered: {list(_REGISTRY)}")
    return _REGISTRY[name]


def registered() -> list[str]:
    return sorted(_REGISTRY)


from .mock import MockProvider  # noqa: E402

register("mock", MockProvider())
