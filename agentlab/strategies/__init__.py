"""Prompt strategies.

A strategy drives an interaction between an agent (provider+model) and a task:
it decides what system prompt to use, whether to give the agent tools,
whether to loop, and when to stop.
"""

from __future__ import annotations

from typing import Protocol

from agentlab.core.types import AgentDef, Task, Trajectory
from agentlab.providers import Provider


class Strategy(Protocol):
    name: str

    async def run(
        self,
        provider: Provider,
        agent_def: AgentDef,
        task: Task,
        *,
        trial_idx: int,
        tools_available: list[str],
    ) -> Trajectory: ...


_REGISTRY: dict[str, Strategy] = {}


def register(name: str, strategy: Strategy) -> None:
    _REGISTRY[name] = strategy


def get(name: str) -> Strategy:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown strategy: {name!r}. Registered: {list(_REGISTRY)}"
        )
    return _REGISTRY[name]


def registered() -> list[str]:
    return sorted(_REGISTRY)


from .direct import DirectStrategy  # noqa: E402
from .react import ReActStrategy  # noqa: E402

register("direct", DirectStrategy())
register("react", ReActStrategy())
