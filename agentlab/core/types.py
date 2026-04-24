"""Shared data types for AgentLab.

These are the vocabulary every layer speaks: providers emit Completions,
strategies produce Trajectories, the runner writes Results into Runs, and
scorers consume Trajectories to return ScorerResults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel):
    """One turn in a conversation, including tool messages."""

    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolSpec(BaseModel):
    """Declarative tool schema passed to providers."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


class Usage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0


class Completion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: Message
    stop_reason: Literal["stop", "tool_use", "length", "content_filter", "other"] = "stop"
    usage: Usage = Field(default_factory=Usage)
    raw: dict[str, Any] | None = None


class ScorerResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scorer: str
    score: float  # 0..1
    detail: dict[str, Any] = Field(default_factory=dict)
    weight: float = 1.0


class AgentDef(BaseModel):
    """A configurable agent: provider + model + strategy + params."""

    model_config = ConfigDict(extra="forbid")

    id: str  # unique per-run id, e.g. "openai:gpt-5:react"
    provider: str
    model: str
    strategy: str = "direct"
    temperature: float = 0.0
    max_tokens: int | None = None
    max_turns: int = 10
    system_prompt: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    prompt: str
    workspace: str | None = None
    tools: list[str] = Field(default_factory=list)
    timeout_s: int = 120
    max_turns: int = 10
    scoring: list[dict[str, Any]] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


@dataclass
class Turn:
    """One record in a trajectory: either a model turn, a tool call, or an observation."""

    kind: Literal["model", "tool_call", "tool_result", "user", "system"]
    content: str = ""
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: Any | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    t_start: float = 0.0
    t_end: float = 0.0


@dataclass
class Trajectory:
    agent_id: str
    task_id: str
    trial_idx: int
    turns: list[Turn] = field(default_factory=list)
    status: Literal["ok", "error", "timeout"] = "ok"
    error: str | None = None
    final_answer: str | None = None

    @property
    def total_tokens_in(self) -> int:
        return sum(t.tokens_in for t in self.turns)

    @property
    def total_tokens_out(self) -> int:
        return sum(t.tokens_out for t in self.turns)


class Result(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    task_id: str
    agent_id: str
    trial_idx: int
    started_at: float
    finished_at: float
    status: Literal["ok", "error", "timeout"]
    score: float | None = None
    scorer_results: list[ScorerResult] = Field(default_factory=list)
    trajectory: list[dict[str, Any]] = Field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    error: str | None = None


class Run(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    started_at: float
    finished_at: float | None
    suite_name: str
    suite_version: str
    agents: list[AgentDef]
    notes: str | None = None


class Agent(Protocol):
    """An agent runs a task and produces a trajectory."""

    id: str

    async def run(self, task: Task, trial_idx: int) -> Trajectory: ...


Message.model_rebuild()
