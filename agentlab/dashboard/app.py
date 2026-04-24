"""FastAPI app — API endpoints and a single-page HTML view."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from agentlab.store import Store

app = FastAPI(title="AgentLab Dashboard")


def _store() -> Store:
    return Store(os.environ.get("AGENTLAB_DB", "runs.db"))


@app.get("/api/runs")
def api_runs(limit: int = 50) -> list[dict[str, Any]]:
    return _store().list_runs(limit=limit)


@app.get("/api/runs/{run_id}")
def api_run(run_id: str) -> dict[str, Any]:
    run = _store().get_run(run_id)
    if not run:
        raise HTTPException(404, f"run {run_id} not found")
    run["results"] = _store().list_results(run_id)
    return run


@app.get("/api/runs/{run_id}/results/{task_id}/{agent_id}/{trial_idx}/trajectory")
def api_trajectory(
    run_id: str, task_id: str, agent_id: str, trial_idx: int
) -> dict[str, Any]:
    t = _store().get_trajectory(run_id, task_id, agent_id, trial_idx)
    if t is None:
        raise HTTPException(404, "trajectory not found")
    return {"trajectory": t}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


INDEX_HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>AgentLab Dashboard</title>
<style>
  :root { --bg:#0d1117; --panel:#161b22; --border:#30363d; --fg:#c9d1d9; --dim:#8b949e; --ok:#3fb950; --bad:#f85149; --accent:#58a6ff; }
  body { margin:0; background:var(--bg); color:var(--fg); font:13px/1.5 ui-monospace, Menlo, monospace; }
  header { padding:10px 16px; border-bottom:1px solid var(--border); }
  main { display:grid; grid-template-columns: 280px 1fr; height:calc(100vh - 42px); }
  .runs-list { border-right:1px solid var(--border); overflow:auto; }
  .run-item { padding:10px 12px; border-bottom:1px solid var(--border); cursor:pointer; }
  .run-item:hover { background:#1f2937; }
  .run-item.active { background:#243247; }
  .run-id { color:var(--accent); }
  .detail { padding:16px; overflow:auto; }
  table { border-collapse: collapse; width: 100%; margin-bottom: 16px; }
  th, td { text-align:left; padding: 4px 8px; border-bottom:1px solid var(--border); }
  th { color:var(--dim); font-weight: 600; }
  .score-cell { font-variant-numeric: tabular-nums; }
  .score-good { color: var(--ok); } .score-bad { color: var(--bad); }
  .pill { display:inline-block; padding:1px 6px; border-radius:10px; border:1px solid var(--border); color:var(--dim); font-size:11px; }
  pre { background:#0b0f14; border:1px solid var(--border); border-radius:6px; padding:10px; white-space: pre-wrap; }
</style>
</head>
<body>
<header><b>AgentLab</b> <span style="color:var(--dim)"> — evaluation dashboard</span></header>
<main>
  <aside class="runs-list" id="runs"></aside>
  <section class="detail" id="detail"><p style="color:var(--dim)">← pick a run</p></section>
</main>
<script>
async function loadRuns() {
  const r = await fetch("/api/runs");
  const runs = await r.json();
  const el = document.getElementById("runs");
  el.innerHTML = runs.map(x => `
    <div class="run-item" data-id="${x.id}">
      <div class="run-id">${x.id}</div>
      <div style="color:var(--dim)">${x.suite_name}@${x.suite_version}</div>
      <div style="color:var(--dim); font-size:11px">${new Date(x.started_at * 1000).toLocaleString()}</div>
    </div>`).join("");
  el.querySelectorAll(".run-item").forEach(n => n.onclick = () => showRun(n.dataset.id));
  if (runs[0]) showRun(runs[0].id);
}

async function showRun(id) {
  document.querySelectorAll(".run-item").forEach(n =>
    n.classList.toggle("active", n.dataset.id === id));
  const r = await fetch("/api/runs/" + encodeURIComponent(id));
  const run = await r.json();
  const byTask = {};
  const agents = Array.from(new Set(run.results.map(r => r.agent_id)));
  for (const x of run.results) {
    const k = x.task_id;
    byTask[k] = byTask[k] || {};
    byTask[k][x.agent_id] = (byTask[k][x.agent_id] || []).concat([x]);
  }
  const rows = Object.keys(byTask).sort().map(task => {
    const cells = agents.map(a => {
      const xs = byTask[task][a] || [];
      const avg = xs.length ? xs.map(x => x.score ?? 0).reduce((p,c)=>p+c,0) / xs.length : null;
      const cls = avg === null ? "" : avg >= 0.5 ? "score-good" : "score-bad";
      return `<td class="score-cell ${cls}">${avg === null ? "—" : avg.toFixed(3)}</td>`;
    }).join("");
    return `<tr><td>${task}</td>${cells}</tr>`;
  }).join("");
  document.getElementById("detail").innerHTML = `
    <h2 style="margin-top:0">${run.id} — ${run.suite_name}@${run.suite_version}</h2>
    <p style="color:var(--dim)">${new Date(run.started_at * 1000).toLocaleString()} · ${run.results.length} result rows</p>
    <table>
      <thead><tr><th>task</th>${agents.map(a => `<th>${a}</th>`).join("")}</tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <p><span class="pill">tip</span> scores = 1.0 best; blank = not run</p>`;
}
loadRuns();
</script>
</body>
</html>
"""
