# AgentLab Architecture

## Overview

AgentLab is a harness for evaluating AI coding agents across models, prompts,
and tasks. It was built for calibrating prompt engineering: run the same suite
of tasks against N models with M prompt variants, score deterministically, and
diff the results.

Stack: Python 3.11+, async-first. SQLite for results storage. FastAPI + a
small React dashboard for browsing runs. Plugin points for providers, tools,
and scorers.

## Core concepts

```
Suite ── a YAML file containing Tasks
Task  ── a named job: prompt, inputs, expected outputs, scorer spec
Agent ── a (Provider, Model, PromptStrategy) tuple; takes a Task, produces
         a Trajectory
Trajectory ── the sequence of LLM turns and tool calls produced by the agent
Run   ── one execution of a Suite × [Agent...] grid
Result── one (Task, Agent, Trial) row in the Run; has Trajectory + Scores
```

## Package layout

```
agentlab/
  providers/          # openai.py, anthropic.py, google.py, ollama.py, mock.py
  runners/            # async runner, retry policy, worker pool
  scoring/            # rubric.py, pytest_score.py, diff_similarity.py, ...
  suites/             # loader, validator, parametric expansion
  tools/              # sandboxed code-exec tool, file-edit tool, shell tool
  store/              # SQLite schema + queries
  dashboard/          # FastAPI app + built React bundle served from /ui
  cli/                # Typer CLI: `agentlab run|diff|show|serve`
tests/                # pytest
examples/suites/      # shipped example suites
```

## Provider abstraction

```python
class Provider(Protocol):
    name: str
    async def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        system: str | None = None,
        stop: list[str] | None = None,
    ) -> Completion: ...
```

`Completion` contains the assistant message, any tool calls, usage (input/output
tokens + cached tokens), and the raw provider response for debugging.

Implementations: `openai`, `anthropic`, `google` (Gemini), `ollama` (local),
and `mock` (for tests and offline sandbox work).

### Tool-calling normalization

Each provider returns tool calls in its own format. The provider layer
normalizes to:

```python
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
```

Tool results are round-tripped in each provider's native format (OpenAI: tool
role; Anthropic: `tool_result` content block; Gemini: `function_response`).

## Suite YAML

```yaml
name: python-refactor-basics
version: 1
description: Small Python refactoring tasks with unit tests.

defaults:
  timeout_s: 120
  max_turns: 10

tasks:
  - id: extract_method_01
    description: |
      Extract the duplicated total-calculation into a helper method.
    workspace: ./workspaces/extract_method_01
    prompt: |
      In `cart.py`, the total is calculated in three places. Extract it into
      a single `compute_total(items: list[Item]) -> Decimal` helper and
      update callers. Keep behavior identical.
    tools: [file_read, file_write, shell]
    scoring:
      - kind: pytest
        cwd: .
        args: ["-q"]
        weight: 0.7
      - kind: diff_size
        max_lines: 60
        weight: 0.1
      - kind: rubric
        prompt: |
          Rate 0-5: is the extraction clean? Are all callers updated? Is the
          signature reasonable?
        weight: 0.2
```

### Parametric expansion

A task can declare `params:` with lists. The suite loader expands the
Cartesian product into one concrete Task per combination:

```yaml
- id: solve_leetcode_{difficulty}_{language}
  params:
    difficulty: [easy, medium]
    language: [python, typescript]
  prompt: |
    Solve the following {{ difficulty }} problem in {{ language }}: ...
```

## Runner

Top-level API:

```python
run = await AgentLabRunner(
    suite=load_suite("suites/python-refactor.yaml"),
    agents=[
        Agent("openai", "gpt-5-2025-11", strategy=ReActStrategy()),
        Agent("anthropic", "claude-opus-4-7", strategy=ReActStrategy()),
        Agent("anthropic", "claude-sonnet-4-6", strategy=ReActStrategy()),
    ],
    trials_per_task=3,
    concurrency=8,
    store=Store("runs.db"),
).execute()
```

- Global concurrency budget is a semaphore (per-provider sub-limits optional).
- Retries on transient errors (429, 5xx, timeouts) with exponential backoff.
- Every run gets a UUID; every trial row is idempotently identifiable by
  `(run_id, task_id, agent_id, trial_idx)`.

### Prompt strategies

- **DirectStrategy** — single turn, prompt → completion, no tools.
- **ReActStrategy** — tool-using loop, up to `max_turns`, stops when the
  model emits a final answer or a sentinel `stop` tool call.
- **PlanActStrategy** — first call asks for a plan JSON; second pass
  executes the plan with tool access.

Adding a strategy is a subclass of `PromptStrategy` with an `async run(agent,
task) -> Trajectory`.

## Tools (sandboxed)

- `file_read(path)` — read within the task workspace.
- `file_write(path, content)` — write within workspace (path-traversal checked).
- `shell(cmd, timeout_s=30)` — run in an ephemeral Docker container
  (`python:3.12-slim` default, configurable per task). Network off by default.
- `web_fetch(url)` — off by default; opt-in per task with allow-list.

Each task has a private working directory copied from `workspace:`. No state
leaks between tasks.

## Scoring

Scorer interface:

```python
class Scorer(Protocol):
    kind: str
    async def score(
        self,
        task: Task,
        trajectory: Trajectory,
        workspace: Path,
    ) -> ScorerResult: ...
```

Built-in scorers:

- **`pytest`** — run pytest in the workspace, parse JUnit XML. Pass ratio = score.
- **`diff_size`** — measure lines changed vs. `max_lines`; score drops linearly past it.
- **`rubric`** — call a rubric judge (an LLM) with the task + trajectory + a
  scoring prompt; parse numeric verdict.
- **`regex_match`** — assistant's final answer matches one or more regexes.
- **`string_equals`** — exact match (post-trim).
- **`ast_equals`** — parse both expected and actual as Python AST, compare
  structurally (whitespace/comments insensitive).

Scorers combine via weighted sum; weights normalized per task.

### Rubric scorer integrity

Rubric judges are themselves subject to prompt injection by the agent's
trajectory. Mitigations:
- Judge prompt is a separate system turn with a fixed schema output.
- Output is parsed as strict JSON; malformed → score = 0.
- The rubric judge model is configurable (often a different provider than
  the one being evaluated, to reduce "home-team" bias).

## Storage

SQLite. Schema (abbreviated):

```sql
CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    started_at INTEGER NOT NULL,
    finished_at INTEGER,
    suite_name TEXT NOT NULL,
    suite_version TEXT NOT NULL,
    agents_json TEXT NOT NULL,
    git_sha TEXT,
    notes TEXT
);

CREATE TABLE results (
    run_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    trial_idx INTEGER NOT NULL,
    started_at INTEGER NOT NULL,
    finished_at INTEGER,
    status TEXT NOT NULL,  -- ok | error | timeout
    score REAL,
    scorer_results_json TEXT,
    trajectory_json TEXT,
    tokens_in INTEGER,
    tokens_out INTEGER,
    cost_usd REAL,
    PRIMARY KEY (run_id, task_id, agent_id, trial_idx)
);

CREATE INDEX idx_results_run ON results(run_id);
CREATE INDEX idx_results_score ON results(score);
```

Trajectories can get large. We gzip `trajectory_json` before storing.

## CLI

```
agentlab run SUITE [--model M ...] [--trials N] [--concurrency N] [--tag T]
agentlab ls [--limit N]
agentlab show RUN_ID [--task T] [--agent A]
agentlab diff RUN_A RUN_B
agentlab serve [--port 8787]
agentlab costs [--since DATE]
agentlab export RUN_ID --format {json,csv,parquet}
```

## Dashboard

FastAPI at `/api`, React SPA served at `/`. Views:
- Runs list (sortable, filterable by suite/agent/date).
- Run detail: per-task heatmap (rows = tasks, cols = agents), click-through to
  trajectory viewer.
- Trajectory viewer: side-by-side LLM messages and tool calls, with syntax
  highlighting for code.
- Diff view: two runs side-by-side, per-task score delta, trajectory diffs.
- Cost dashboard: tokens/cost by provider/model over time.

## Prompt engineering calibration loop

The intended workflow:

1. Author a suite reflecting the real tasks you care about.
2. Run with a baseline agent. Record scores.
3. Tweak prompts, strategies, or model. Re-run.
4. `agentlab diff` against baseline. Drill into regressions in the dashboard.

This pattern is exactly the human side of evals-driven prompt work; AgentLab
is the tooling that makes it fast enough to iterate on.

## Testing

pytest. Mock provider for deterministic tests. Real-provider tests are
gated behind env vars (`AGENTLAB_LIVE_TESTS=1`) and hit a small always-on
suite with tight token caps.

## CI

GitHub Actions: lint (ruff), typecheck (mypy), pytest. No live-provider
tests in CI by default.

## Non-goals

- Not a training harness — no gradient updates, no RLHF.
- Not a chat UI — this is a batch eval tool.
- Not a substitute for end-to-end product tests.
