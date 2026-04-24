"""Suite loading + parametric expansion."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment
from pydantic import BaseModel, ConfigDict, Field

from agentlab.core.types import Task


class Suite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str = "1"
    description: str | None = None
    defaults: dict[str, Any] = Field(default_factory=dict)
    tasks: list[Task]


def load_suite(path: str | Path) -> Suite:
    raw = yaml.safe_load(Path(path).read_text("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"suite root must be a mapping, got {type(raw).__name__}")
    defaults = raw.get("defaults") or {}
    raw_tasks = raw.get("tasks") or []
    expanded: list[Task] = []
    env = Environment(autoescape=False)
    for raw_task in raw_tasks:
        expanded.extend(_expand_params(raw_task, defaults, env))
    return Suite(
        name=str(raw.get("name", "unnamed")),
        version=str(raw.get("version", "1")),
        description=raw.get("description"),
        defaults=defaults,
        tasks=expanded,
    )


def _expand_params(
    task_dict: dict[str, Any], defaults: dict[str, Any], env: Environment
) -> list[Task]:
    params = task_dict.pop("params", None)
    base = {**defaults, **task_dict}
    if not params:
        return [_as_task(base, env, {})]
    keys = list(params.keys())
    values = [params[k] if isinstance(params[k], list) else [params[k]] for k in keys]
    out: list[Task] = []
    for combo in itertools.product(*values):
        bindings = dict(zip(keys, combo, strict=False))
        out.append(_as_task(base, env, bindings))
    return out


def _as_task(raw: dict[str, Any], env: Environment, bindings: dict[str, Any]) -> Task:
    def render(v: Any) -> Any:
        if isinstance(v, str):
            return env.from_string(v).render(**bindings)
        if isinstance(v, list):
            return [render(x) for x in v]
        if isinstance(v, dict):
            return {k: render(x) for k, x in v.items()}
        return v

    rendered = render(raw)
    # id is special: support {placeholders}.
    rid = rendered.get("id", "")
    if isinstance(rid, str):
        rendered["id"] = rid.format(**bindings) if bindings else rid
    return Task(**rendered)
