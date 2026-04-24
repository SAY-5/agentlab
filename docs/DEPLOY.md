# Deploying AgentLab

AgentLab is two things in one package:

1. **A batch eval runner** — `agentlab run suite.yaml` against some models,
   write results to SQLite, exit. No long-running service needed.
2. **A dashboard** — `agentlab serve`, an always-on FastAPI + single-file
   HTML view over the same SQLite database.

## Running evals from the Docker image

```bash
docker build -t agentlab:local .
# Point at a mounted suite; the image ships /app/examples/suites/smoke.yaml
docker run --rm \
  -e OPENAI_API_KEY -e ANTHROPIC_API_KEY \
  -v $PWD/runs:/data \
  agentlab:local \
    run /app/examples/suites/python_refactor.yaml \
    --model openai:gpt-5 \
    --model anthropic:claude-opus-4-7 \
    --trials 3 \
    --db /data/runs.db
```

The image's `ENTRYPOINT` is `agentlab`, so any CLI subcommand works:

```bash
docker run --rm -v $PWD/runs:/data agentlab:local ls --db /data/runs.db
docker run --rm -v $PWD/runs:/data agentlab:local show <run_id> --db /data/runs.db
docker run --rm -v $PWD/runs:/data agentlab:local \
    diff run_a run_b --db /data/runs.db
```

## Running the dashboard

```bash
docker compose up -d dashboard
# → http://localhost:8787
```

The dashboard is **unauthenticated** and intended for localhost or a
trusted internal network. For external exposure, put it behind an
authenticating reverse proxy (oauth2-proxy, Tailscale ACL, etc.) — do
not open port 8787 to the public internet as-is.

## Security posture

- The `shell` tool is **off by default**. Set
  `AGENTLAB_ALLOW_SHELL=1` *only* after you've thought about it: it
  grants the agent arbitrary code execution on the host. Production
  users should wrap the runner in a container-per-task sandbox (Docker
  Python SDK, Firecracker, or gVisor) before flipping this on.
- File tools reject absolute paths and symlinks-to-outside-workspace.
  Task workspaces are copied, not symlinked, so a malicious agent can't
  escape through a symlink planted during the run.
- Dashboard HTML escapes every interpolated string via a helper, so a
  suite/run containing `<script>` tags can't XSS the dashboard.
- API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY) are read from env and
  never persisted. They are not written to the results DB.

## Operational notes

- `runs.db` uses WAL mode; backing up means copying `runs.db`,
  `runs.db-wal`, and `runs.db-shm` as a unit, or using `sqlite3
  runs.db ".backup ..."`.
- Per-run workspaces are created under `runs/<run_id>/workspaces/...`
  and are *not* cleaned up automatically. A nightly cron that deletes
  runs older than N days is typical.
- Costs: the dashboard doesn't aggregate costs by default because
  per-provider pricing changes frequently. The `cost_usd` column
  exists; populate it with a post-hoc script that joins your
  pricing table to the tokens_in/tokens_out columns.

## Horizontal scaling

The runner is trivially parallel: N workers each running different
suites (or subsets of one suite) and appending to the same SQLite DB
will contend at the WAL boundary but not corrupt. For >~50 concurrent
writers, swap the SQLite `Store` for a Postgres implementation (the
interface is a single class in `agentlab/store/__init__.py` and has
five methods).

Out of scope:
- A job queue. Plug into whatever you already have (Celery, Temporal,
  a k8s Job per run).
- Cost-aware scheduling. Rank your runs by model and pick workers
  accordingly — AgentLab itself is provider-agnostic.
