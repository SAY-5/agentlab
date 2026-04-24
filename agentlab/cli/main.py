"""`agentlab` CLI entry point."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from agentlab.core.types import AgentDef
from agentlab.providers import register as register_provider
from agentlab.runners import RetryPolicy, Runner, RunnerConfig
from agentlab.store import Store
from agentlab.suites import load_suite

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command()
def run(
    suite_path: str = typer.Argument(..., help="Path to a suite YAML file"),
    model: list[str] = typer.Option(  # noqa: B008
        [],
        "--model",
        "-m",
        help="Agent spec as provider:model[:strategy], repeatable",
    ),
    trials: int = typer.Option(1, "--trials", help="Trials per task"),
    concurrency: int = typer.Option(4, "--concurrency", "-c"),
    db: str = typer.Option("runs.db", "--db"),
    notes: str | None = typer.Option(None, "--notes", "-n"),
    max_attempts: int = typer.Option(3, "--max-attempts"),
) -> None:
    """Execute a suite against one or more agents."""
    if not model:
        console.print("[red]at least one --model is required[/]")
        raise typer.Exit(2)
    _eager_register_providers()
    agents = [_parse_agent(m) for m in model]
    suite = load_suite(suite_path)
    store = Store(db)
    runner = Runner(
        suite=suite,
        agents=agents,
        store=store,
        config=RunnerConfig(
            trials_per_task=trials,
            concurrency=concurrency,
            retry=RetryPolicy(max_attempts=max_attempts),
            progress=lambda s: console.print(f"[dim]{s}[/]"),
        ),
    )
    run_obj = asyncio.run(runner.execute(notes=notes))
    console.print(f"\n[bold green]run {run_obj.id} complete[/]")
    _print_summary(store, run_obj.id)


@app.command("ls")
def list_runs(limit: int = typer.Option(20, "--limit", "-l"), db: str = typer.Option("runs.db", "--db")) -> None:
    """List recent runs."""
    store = Store(db)
    table = Table(title="Runs")
    table.add_column("id")
    table.add_column("suite")
    table.add_column("started")
    table.add_column("notes", overflow="fold")
    for row in store.list_runs(limit=limit):
        table.add_row(
            row["id"],
            f"{row['suite_name']}@{row['suite_version']}",
            _fmt_ts(row["started_at"]),
            row["notes"] or "",
        )
    console.print(table)


@app.command()
def show(run_id: str, db: str = typer.Option("runs.db", "--db")) -> None:
    """Show details of a single run."""
    store = Store(db)
    _print_summary(store, run_id)


@app.command()
def diff(
    run_a: str,
    run_b: str,
    db: str = typer.Option("runs.db", "--db"),
) -> None:
    """Compare two runs task-by-task."""
    store = Store(db)
    ra = {(r["task_id"], r["agent_id"]): r for r in store.list_results(run_a)}
    rb = {(r["task_id"], r["agent_id"]): r for r in store.list_results(run_b)}
    keys = sorted(set(ra) | set(rb))
    table = Table(title=f"diff {run_a} → {run_b}")
    table.add_column("task × agent")
    table.add_column(f"{run_a[:10]}", justify="right")
    table.add_column(f"{run_b[:10]}", justify="right")
    table.add_column("Δ", justify="right")
    for k in keys:
        a = ra.get(k, {}).get("score")
        b = rb.get(k, {}).get("score")
        delta = (b - a) if (a is not None and b is not None) else None
        table.add_row(
            f"{k[0]} × {k[1]}",
            _fmt_score(a),
            _fmt_score(b),
            _fmt_delta(delta),
        )
    console.print(table)


@app.command()
def serve(port: int = typer.Option(8787, "--port", "-p"), db: str = typer.Option("runs.db", "--db")) -> None:
    """Start the dashboard server."""
    os.environ["AGENTLAB_DB"] = db
    import uvicorn

    uvicorn.run("agentlab.dashboard.app:app", host="127.0.0.1", port=port, reload=False)


@app.command()
def export(
    run_id: str,
    fmt: str = typer.Option("json", "--format", "-f"),
    out: str | None = typer.Option(None, "--out"),
    db: str = typer.Option("runs.db", "--db"),
) -> None:
    """Export a run's results as JSON or CSV."""
    store = Store(db)
    rows = store.list_results(run_id)
    if fmt == "json":
        payload = json.dumps(rows, indent=2, default=str)
    elif fmt == "csv":
        import csv
        import io

        buf = io.StringIO()
        if rows:
            w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow({k: json.dumps(v) if not isinstance(v, str | int | float | type(None)) else v for k, v in r.items()})
        payload = buf.getvalue()
    else:
        console.print(f"[red]unknown format {fmt!r}[/]")
        raise typer.Exit(2)
    if out:
        Path(out).write_text(payload, "utf-8")
        console.print(f"wrote {out}")
    else:
        console.print(payload)


# --- helpers --------------------------------------------------------------


def _parse_agent(spec: str) -> AgentDef:
    parts = spec.split(":")
    if len(parts) < 2:
        raise typer.BadParameter(f"expected provider:model[:strategy], got {spec!r}")
    provider = parts[0]
    model = parts[1]
    strategy = parts[2] if len(parts) > 2 else "direct"
    return AgentDef(
        id=spec,
        provider=provider,
        model=model,
        strategy=strategy,
    )


def _eager_register_providers() -> None:
    try:
        from agentlab.providers.openai import OpenAIProvider

        register_provider("openai", OpenAIProvider())
    except ImportError:
        pass
    try:
        from agentlab.providers.anthropic import AnthropicProvider

        register_provider("anthropic", AnthropicProvider())
    except ImportError:
        pass
    try:
        from agentlab.providers.ollama import OllamaProvider

        register_provider("ollama", OllamaProvider())
    except ImportError:
        pass


def _print_summary(store: Store, run_id: str) -> None:
    run = store.get_run(run_id)
    if not run:
        console.print(f"[red]run {run_id} not found[/]")
        return
    results = store.list_results(run_id)
    by_agent: dict[str, list[dict]] = {}
    for r in results:
        by_agent.setdefault(r["agent_id"], []).append(r)
    table = Table(title=f"run {run_id} — {run['suite_name']}@{run['suite_version']}")
    table.add_column("agent")
    table.add_column("n", justify="right")
    table.add_column("ok", justify="right")
    table.add_column("errors", justify="right")
    table.add_column("avg score", justify="right")
    table.add_column("tokens in/out", justify="right")
    for agent_id, rows in sorted(by_agent.items()):
        scores = [r["score"] for r in rows if r["score"] is not None]
        avg = sum(scores) / len(scores) if scores else None
        ok = sum(1 for r in rows if r["status"] == "ok")
        errs = sum(1 for r in rows if r["status"] != "ok")
        tin = sum(r["tokens_in"] or 0 for r in rows)
        tout = sum(r["tokens_out"] or 0 for r in rows)
        table.add_row(
            agent_id,
            str(len(rows)),
            str(ok),
            str(errs),
            _fmt_score(avg),
            f"{tin}/{tout}",
        )
    console.print(table)


def _fmt_score(s: float | None) -> str:
    return "-" if s is None else f"{s:.3f}"


def _fmt_delta(d: float | None) -> str:
    if d is None:
        return "-"
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.3f}"


def _fmt_ts(t: float) -> str:
    import datetime as dt

    return dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    app()
