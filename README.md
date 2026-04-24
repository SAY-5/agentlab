# AgentLab

[![ci](https://github.com/SAY-5/agentlab/actions/workflows/ci.yml/badge.svg)](https://github.com/SAY-5/agentlab/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![tests](https://img.shields.io/badge/tests-44%20passing-brightgreen)](#)
[![python](https://img.shields.io/badge/python-3.11+-3776ab)](#)

Multi-model AI coding agent evaluation harness for prompt-engineering
calibration. Define task suites in YAML, run them against any mix of
OpenAI / Anthropic / Ollama (and anything else you register), score with
rubric LLMs or real test runners, and browse results in a dashboard.

- **Async runner** with global + per-provider concurrency, exponential-backoff
  retries, and per-task workspace isolation.
- **Six scorers** out of the box: `regex_match`, `string_equals`, `ast_equals`,
  `diff_size`, `pytest`, `rubric` (LLM-judge).
- **Three strategies**: `direct`, `react` (tool loop), and an easy Strategy
  interface for your own.
- **SQLite store** with gzipped trajectories; queryable and diffable between
  runs.
- **Dashboard**: FastAPI + a single-file HTML UI at `/`.
- **CLI**: `agentlab run|ls|show|diff|serve|export`.

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the design writeup.

## Install

```bash
pip install -e ".[dev,openai,anthropic]"
```

Python ≥ 3.11.

## Quick start with the mock provider

```bash
cd /path/to/agentlab
pytest                                           # 25+ tests pass
agentlab run examples/suites/smoke.yaml \
    --model mock:stub --trials 1 --db runs.db
agentlab ls --db runs.db
agentlab show <run_id> --db runs.db
agentlab serve --db runs.db                      # dashboard at :8787
```

## Real model run

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...

agentlab run examples/suites/python_refactor.yaml \
    --model openai:gpt-5 \
    --model openai:gpt-5-mini \
    --model anthropic:claude-opus-4-7 \
    --model anthropic:claude-sonnet-4-6 \
    --trials 3 \
    --concurrency 8 \
    --notes "baseline sweep"
```

Add `--model <provider>:<model>:<strategy>` to pick a strategy (default:
`direct`). Strategies: `direct`, `react`.

## Diff two runs

```bash
agentlab diff run_abc123 run_def456
```

Shows per-task-per-agent score deltas.

## Suite format

```yaml
name: python-refactor-basics
version: "1"

defaults:
  timeout_s: 120
  max_turns: 10

tasks:
  - id: extract_method_01
    description: Extract the duplicated total calculation.
    workspace: ./workspaces/extract_method_01
    prompt: |
      In `cart.py` the total is calculated three times. Extract it into a
      single helper and update callers. Keep behavior identical.
    tools: [file_read, file_write, shell]
    scoring:
      - kind: pytest
        args: ["-q"]
        weight: 0.7
      - kind: diff_size
        max_lines: 60
        weight: 0.1
      - kind: rubric
        prompt: |
          Rate 0-5: is the extraction clean? Are all callers updated?
        weight: 0.2
```

Parametric expansion via `params:`:

```yaml
- id: solve_{difficulty}_{lang}
  params:
    difficulty: [easy, medium]
    lang: [python, typescript]
  prompt: Solve a {{ difficulty }} problem in {{ lang }}: ...
```

## Dashboard

`agentlab serve` starts a FastAPI app at `:8787`. The UI is a single HTML file
showing a runs list and a task-by-agent score heatmap per run.

## Extending

- **Provider** — implement `Provider.complete`, register via
  `agentlab.providers.register("mine", MyProvider())`.
- **Strategy** — implement `Strategy.run`, register via
  `agentlab.strategies.register("mine", MyStrategy())`.
- **Scorer** — subclass the `Scorer` protocol, register via
  `agentlab.scoring.register("mine", MyScorer)`.
- **Tool** — add to `agentlab/tools/` and register in `tools/__init__.py`.

## Companion projects

Part of a five-repo set:

- **[canvaslive](https://github.com/SAY-5/canvaslive)** — real-time multiplayer whiteboard with operational-transform convergence.
- **[pluginforge](https://github.com/SAY-5/pluginforge)** — Web Worker plugin sandbox with capability-based permissions.
- **[agentlab](https://github.com/SAY-5/agentlab)** — you're here. AI coding agent evaluation harness.
- **[payflow](https://github.com/SAY-5/payflow)** — Spring Boot payments API with idempotent transactions and Stripe webhooks.
- **[queryflow](https://github.com/SAY-5/queryflow)** — natural-language SQL engine with pgvector RAG.

## License

MIT — see [LICENSE](./LICENSE).
