from agentlab.core.types import (
    AgentDef,
    Completion,
    Message,
    Task,
    ToolCall,
    Trajectory,
    Turn,
    Usage,
)


def test_message_rejects_extra_fields():
    m = Message(role="user", content="hi")
    assert m.role == "user"
    assert m.content == "hi"


def test_toolcall_roundtrip():
    tc = ToolCall(id="c1", name="file_read", arguments={"path": "x"})
    dumped = tc.model_dump()
    assert dumped == {"id": "c1", "name": "file_read", "arguments": {"path": "x"}}


def test_trajectory_token_accumulation():
    t = Trajectory(agent_id="a", task_id="t", trial_idx=0)
    t.turns.append(Turn(kind="model", content="x", tokens_in=10, tokens_out=5))
    t.turns.append(Turn(kind="model", content="y", tokens_in=7, tokens_out=3))
    assert t.total_tokens_in == 17
    assert t.total_tokens_out == 8


def test_agent_def_defaults():
    a = AgentDef(id="openai:gpt-5", provider="openai", model="gpt-5")
    assert a.strategy == "direct"
    assert a.temperature == 0.0
    assert a.max_turns == 10


def test_task_accepts_scoring_list():
    t = Task(id="t1", description="d", prompt="p", scoring=[{"kind": "regex_match", "patterns": ["x"]}])
    assert t.scoring[0]["kind"] == "regex_match"


def test_completion_defaults():
    c = Completion(message=Message(role="assistant", content="hi"))
    assert c.stop_reason == "stop"
    assert isinstance(c.usage, Usage)
