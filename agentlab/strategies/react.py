"""ReAct strategy: tool-using loop up to max_turns."""

from __future__ import annotations

import time

from agentlab.core.types import AgentDef, Message, Task, Trajectory, Turn
from agentlab.providers import Provider
from agentlab.tools import get_tool, tool_specs


class ReActStrategy:
    name = "react"

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
        messages: list[Message] = [Message(role="user", content=task.prompt)]
        t0 = time.time()
        traj.turns.append(Turn(kind="user", content=task.prompt, t_start=t0, t_end=t0))

        tools = tool_specs([t for t in tools_available if t in task.tools])

        for _turn_idx in range(agent_def.max_turns):
            try:
                ts = time.time()
                completion = await provider.complete(
                    messages,
                    model=agent_def.model,
                    temperature=agent_def.temperature,
                    max_tokens=agent_def.max_tokens,
                    system=agent_def.system_prompt,
                    tools=tools or None,
                )
                te = time.time()
            except Exception as e:  # noqa: BLE001
                traj.status = "error"
                traj.error = f"{type(e).__name__}: {e}"
                return traj

            traj.turns.append(
                Turn(
                    kind="model",
                    content=completion.message.content,
                    tokens_in=completion.usage.input_tokens,
                    tokens_out=completion.usage.output_tokens,
                    t_start=ts,
                    t_end=te,
                )
            )
            messages.append(completion.message)

            if completion.stop_reason != "tool_use" or not completion.message.tool_calls:
                traj.final_answer = completion.message.content
                return traj

            for tc in completion.message.tool_calls:
                tool = get_tool(tc.name)
                if tool is None:
                    observation = f"error: unknown tool {tc.name}"
                    ok = False
                else:
                    try:
                        result = await tool.run(tc.arguments, workspace=task.workspace)
                        observation = str(result)
                        ok = True
                    except Exception as e:  # noqa: BLE001
                        observation = f"error: {type(e).__name__}: {e}"
                        ok = False
                traj.turns.append(
                    Turn(
                        kind="tool_call",
                        tool_name=tc.name,
                        tool_args=tc.arguments,
                        content=f"{tc.name}({tc.arguments})",
                    )
                )
                traj.turns.append(
                    Turn(
                        kind="tool_result",
                        tool_name=tc.name,
                        tool_result=observation,
                        content=observation[:500] + ("…" if len(observation) > 500 else ""),
                    )
                )
                messages.append(
                    Message(
                        role="tool",
                        tool_call_id=tc.id,
                        content=observation if ok else observation,
                    )
                )

        traj.status = "error"
        traj.error = f"max_turns ({agent_def.max_turns}) exceeded"
        return traj
