"""AgentLab: multi-model AI coding agent evaluation harness."""

from __future__ import annotations

__version__ = "0.1.0"

from .core.types import (
    Agent,
    AgentDef,
    Completion,
    Message,
    Result,
    Run,
    Task,
    Trajectory,
    Turn,
)

__all__ = [
    "Agent",
    "AgentDef",
    "Completion",
    "Message",
    "Run",
    "Result",
    "Task",
    "Trajectory",
    "Turn",
    "__version__",
]
