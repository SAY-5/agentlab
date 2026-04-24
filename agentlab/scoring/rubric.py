from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agentlab.core.types import Message, ScorerResult, Task, Trajectory
from agentlab.providers import get as get_provider

JUDGE_SYSTEM = (
    "You are an impartial judge scoring AI assistant outputs. "
    "Read the task, the assistant's trajectory, and the rubric. "
    "Return a JSON object with keys: {\"score\": number 0..5, \"rationale\": string}. "
    "Score only based on the rubric. Output ONLY valid JSON."
)


class RubricScorer:
    """LLM-based rubric scorer. Delegates to a separate judge provider+model."""

    kind = "rubric"

    def __init__(
        self,
        prompt: str,
        *,
        judge_provider: str = "openai",
        judge_model: str = "gpt-5-mini",
        weight: float = 1.0,
    ) -> None:
        self.prompt = prompt
        self.judge_provider = judge_provider
        self.judge_model = judge_model
        self.weight = weight

    async def score(
        self, task: Task, trajectory: Trajectory, workspace: Path
    ) -> ScorerResult:
        traj_text = _trajectory_summary(trajectory)
        user = (
            f"TASK: {task.description}\n\n"
            f"PROMPT GIVEN TO AGENT:\n{task.prompt}\n\n"
            f"AGENT TRAJECTORY (truncated):\n{traj_text}\n\n"
            f"AGENT FINAL ANSWER:\n{trajectory.final_answer or '(none)'}\n\n"
            f"RUBRIC:\n{self.prompt}\n\n"
            f"Return ONLY a JSON object: {{\"score\": 0..5, \"rationale\": \"...\"}}."
        )
        try:
            provider = get_provider(self.judge_provider)
        except KeyError as e:
            return ScorerResult(
                scorer=self.kind,
                score=0.0,
                weight=self.weight,
                detail={"error": f"judge provider unavailable: {e}"},
            )
        resp = await provider.complete(
            [Message(role="user", content=user)],
            model=self.judge_model,
            system=JUDGE_SYSTEM,
            temperature=0.0,
        )
        parsed = _parse_judge(resp.message.content)
        if parsed is None:
            return ScorerResult(
                scorer=self.kind,
                score=0.0,
                weight=self.weight,
                detail={"error": "judge returned invalid json", "raw": resp.message.content},
            )
        score_0_5 = max(0.0, min(5.0, float(parsed.get("score", 0))))
        return ScorerResult(
            scorer=self.kind,
            score=score_0_5 / 5.0,
            weight=self.weight,
            detail={"rationale": parsed.get("rationale", ""), "raw_score": score_0_5},
        )


def _trajectory_summary(t: Trajectory) -> str:
    pieces: list[str] = []
    for turn in t.turns[-20:]:
        if turn.kind == "model":
            pieces.append(f"[assistant] {turn.content}")
        elif turn.kind == "tool_call":
            pieces.append(f"[tool:{turn.tool_name}] args={turn.tool_args}")
        elif turn.kind == "tool_result":
            val = str(turn.tool_result)[:300]
            pieces.append(f"[tool_result] {val}")
        elif turn.kind == "user":
            pieces.append(f"[user] {turn.content}")
    return "\n".join(pieces)


def _parse_judge(content: str) -> dict[str, Any] | None:
    # First try direct parse.
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    # Try to extract the first JSON object.
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
