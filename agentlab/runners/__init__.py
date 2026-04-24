"""Async runner that executes a suite × agents grid."""

from __future__ import annotations

import asyncio
import json
import shutil
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentlab.core.types import (
    AgentDef,
    Result,
    Run,
    ScorerResult,
    Task,
    Trajectory,
    Turn,
)
from agentlab.providers import get as get_provider
from agentlab.scoring import build as build_scorer
from agentlab.store import Store
from agentlab.strategies import get as get_strategy
from agentlab.suites import Suite
from agentlab.tools import _REGISTRY as TOOL_REGISTRY


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0

    def delay_for_attempt(self, attempt: int) -> float:
        return min(self.max_delay_s, self.base_delay_s * (2 ** (attempt - 1)))


@dataclass
class RunnerConfig:
    trials_per_task: int = 1
    concurrency: int = 4
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    workspaces_root: Path | None = None
    progress: Callable[[str], None] | None = None


class Runner:
    def __init__(
        self,
        *,
        suite: Suite,
        agents: list[AgentDef],
        store: Store,
        config: RunnerConfig | None = None,
    ) -> None:
        self.suite = suite
        self.agents = agents
        self.store = store
        self.config = config or RunnerConfig()

    async def execute(self, *, notes: str | None = None) -> Run:
        run = Run(
            id="run_" + uuid.uuid4().hex[:12],
            started_at=time.time(),
            finished_at=None,
            suite_name=self.suite.name,
            suite_version=self.suite.version,
            agents=self.agents,
            notes=notes,
        )
        self.store.insert_run(run)

        sem = asyncio.Semaphore(self.config.concurrency)
        tasks_fut: list[asyncio.Task[Result]] = []
        for task in self.suite.tasks:
            for agent_def in self.agents:
                for trial in range(self.config.trials_per_task):
                    tasks_fut.append(
                        asyncio.create_task(
                            self._run_one(sem, run.id, task, agent_def, trial)
                        )
                    )

        results: list[Result] = []
        for coro in asyncio.as_completed(tasks_fut):
            res = await coro
            self.store.insert_result(res)
            results.append(res)
            if self.config.progress:
                self.config.progress(
                    f"{res.task_id} × {res.agent_id} trial {res.trial_idx}: "
                    f"{res.status} score={res.score}"
                )

        run.finished_at = time.time()
        self.store.finish_run(run.id, run.finished_at)
        return run

    async def _run_one(
        self,
        sem: asyncio.Semaphore,
        run_id: str,
        task: Task,
        agent_def: AgentDef,
        trial_idx: int,
    ) -> Result:
        async with sem:
            started = time.time()
            workspace = self._prepare_workspace(run_id, task, agent_def, trial_idx)
            traj = await self._run_with_retry(agent_def, task, trial_idx)
            finished = time.time()

            scorer_results: list[ScorerResult] = []
            if traj.status == "ok":
                for spec in task.scoring:
                    try:
                        scorer = build_scorer(spec)
                        sr = await scorer.score(task, traj, workspace)
                        scorer_results.append(sr)
                    except Exception as e:  # noqa: BLE001
                        scorer_results.append(
                            ScorerResult(
                                scorer=str(spec.get("kind", "?")),
                                score=0.0,
                                weight=float(spec.get("weight", 1.0)),
                                detail={"error": f"{type(e).__name__}: {e}"},
                            )
                        )

            score = _combined_score(scorer_results)
            return Result(
                run_id=run_id,
                task_id=task.id,
                agent_id=agent_def.id,
                trial_idx=trial_idx,
                started_at=started,
                finished_at=finished,
                status=traj.status,
                score=score,
                scorer_results=scorer_results,
                trajectory=[_turn_to_dict(t) for t in traj.turns],
                tokens_in=traj.total_tokens_in,
                tokens_out=traj.total_tokens_out,
                cost_usd=0.0,  # populated by a post-hoc price table
                error=traj.error,
            )

    async def _run_with_retry(
        self, agent_def: AgentDef, task: Task, trial_idx: int
    ) -> Trajectory:
        last_exc: BaseException | None = None
        for attempt in range(1, self.config.retry.max_attempts + 1):
            try:
                provider = get_provider(agent_def.provider)
                strategy = get_strategy(agent_def.strategy)
                return await asyncio.wait_for(
                    strategy.run(
                        provider,
                        agent_def,
                        task,
                        trial_idx=trial_idx,
                        tools_available=list(TOOL_REGISTRY.keys()),
                    ),
                    timeout=task.timeout_s,
                )
            except TimeoutError:
                t = Trajectory(
                    agent_id=agent_def.id,
                    task_id=task.id,
                    trial_idx=trial_idx,
                    status="timeout",
                    error=f"timeout after {task.timeout_s}s",
                )
                return t
            except Exception as e:  # noqa: BLE001
                last_exc = e
                if attempt >= self.config.retry.max_attempts:
                    break
                await asyncio.sleep(self.config.retry.delay_for_attempt(attempt))
        return Trajectory(
            agent_id=agent_def.id,
            task_id=task.id,
            trial_idx=trial_idx,
            status="error",
            error=f"{type(last_exc).__name__}: {last_exc}" if last_exc else "unknown",
        )

    def _prepare_workspace(
        self, run_id: str, task: Task, agent_def: AgentDef, trial_idx: int
    ) -> Path:
        if not task.workspace:
            return Path(".").resolve()
        src = Path(task.workspace).resolve()
        if not src.exists():
            return Path(".").resolve()
        root = self.config.workspaces_root or Path(".") / "runs" / run_id / "workspaces"
        dst = root / f"{task.id}__{agent_def.id}__trial{trial_idx}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        return dst.resolve()


def _combined_score(results: list[ScorerResult]) -> float | None:
    if not results:
        return None
    total_weight = sum(r.weight for r in results) or 1.0
    return sum(r.score * r.weight for r in results) / total_weight


def _turn_to_dict(t: Turn) -> dict[str, Any]:
    return {
        "kind": t.kind,
        "content": t.content,
        "tool_name": t.tool_name,
        "tool_args": t.tool_args,
        "tool_result": json.dumps(t.tool_result) if t.tool_result is not None else None,
        "tokens_in": t.tokens_in,
        "tokens_out": t.tokens_out,
        "t_start": t.t_start,
        "t_end": t.t_end,
    }
