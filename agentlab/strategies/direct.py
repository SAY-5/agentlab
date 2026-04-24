"""Direct strategy: single prompt → single completion, no tools, no loop."""

from __future__ import annotations

import time

from agentlab.core.types import AgentDef, Message, Task, Trajectory, Turn
from agentlab.providers import Provider


class DirectStrategy:
    name = "direct"

    async def run(
        self,
        provider: Provider,
        agent_def: AgentDef,
        task: Task,
        *,
        trial_idx: int,
        tools_available: list[str],
    ) -> Trajectory:
        # We intentionally do NOT catch provider errors here: the runner's
        # retry loop needs the exception to bubble so it can decide whether
        # to retry (transient) or give up. Catching here would pre-commit
        # a single attempt into a permanent error Trajectory.
        traj = Trajectory(agent_id=agent_def.id, task_id=task.id, trial_idx=trial_idx)
        t0 = time.time()
        traj.turns.append(Turn(kind="user", content=task.prompt, t_start=t0, t_end=t0))
        completion = await provider.complete(
            [Message(role="user", content=task.prompt)],
            model=agent_def.model,
            temperature=agent_def.temperature,
            max_tokens=agent_def.max_tokens,
            system=agent_def.system_prompt,
        )
        t1 = time.time()
        traj.turns.append(
            Turn(
                kind="model",
                content=completion.message.content,
                tokens_in=completion.usage.input_tokens,
                tokens_out=completion.usage.output_tokens,
                t_start=t0,
                t_end=t1,
            )
        )
        traj.final_answer = completion.message.content
        return traj
