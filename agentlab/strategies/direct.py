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
        traj = Trajectory(agent_id=agent_def.id, task_id=task.id, trial_idx=trial_idx)
        t0 = time.time()
        traj.turns.append(Turn(kind="user", content=task.prompt, t_start=t0, t_end=t0))
        try:
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
        except Exception as e:  # noqa: BLE001
            traj.status = "error"
            traj.error = f"{type(e).__name__}: {e}"
        return traj
