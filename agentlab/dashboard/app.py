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
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>AgentLab — Evaluation Terminal</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=Newsreader:ital,wght@0,400;0,500;1,400&display=swap">
<style>
  :root {
    /* Dark moss / terminal ground */
    --bg: #0a0d0a;
    --bg-2: #0f1310;
    --surface: #141915;
    --surface-2: #1a201c;
    --line: #242a25;
    --line-2: #333b34;

    --fg: #d8dfd5;
    --fg-dim: #8a9388;
    --fg-muted: #586058;

    /* Matcha/moss accent + amber warning */
    --accent: #9dc585;
    --accent-dim: #5c8547;
    --accent-glow: rgba(157, 197, 133, 0.2);
    --warn: #e6a84a;
    --err: #e67a5a;

    --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
    --sans: "IBM Plex Sans", -apple-system, system-ui, sans-serif;
    --serif: "Newsreader", ui-serif, Georgia, serif;

    color-scheme: dark;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0;
    background: var(--bg);
    color: var(--fg);
    font-family: var(--mono);
    font-size: 12.5px;
    line-height: 1.55;
    -webkit-font-smoothing: antialiased;
    font-feature-settings: "ss01", "ss03", "zero";
  }
  body::before {
    content: "";
    position: fixed; inset: 0;
    background: radial-gradient(ellipse at 50% -20%, rgba(157,197,133,.06) 0, transparent 45%);
    pointer-events: none;
    z-index: 100;
  }
  header.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 20px;
    background: var(--bg-2);
    border-bottom: 1px solid var(--line);
    position: relative;
    z-index: 10;
  }
  .brand {
    display: flex; align-items: baseline; gap: 10px;
    font-family: var(--serif);
    font-weight: 500;
    font-size: 22px;
    letter-spacing: -0.015em;
    color: var(--fg);
  }
  .brand em {
    font-style: italic;
    color: var(--accent);
    font-weight: 400;
  }
  .brand .tag {
    font-family: var(--mono);
    font-size: 10.5px;
    color: var(--fg-muted);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    padding-left: 10px;
    border-left: 1px solid var(--line-2);
    margin-left: 2px;
    position: relative;
    top: -2px;
  }
  .topbar-stats {
    display: flex; gap: 26px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--fg-dim);
  }
  .topbar-stats .stat b {
    display: block;
    font-size: 18px;
    color: var(--fg);
    font-weight: 500;
    font-variant-numeric: tabular-nums;
    margin-top: 2px;
  }
  .topbar-stats .stat span.label {
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--fg-muted);
    font-size: 9.5px;
  }
  main {
    display: grid;
    grid-template-columns: 320px 1fr;
    height: calc(100vh - 54px);
  }
  aside.runs-list {
    border-right: 1px solid var(--line);
    overflow: auto;
    background: var(--bg);
  }
  .section-header {
    padding: 12px 20px 10px;
    font-family: var(--sans);
    font-weight: 500;
    font-size: 10px;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--fg-muted);
    border-bottom: 1px solid var(--line);
    background: var(--bg-2);
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .section-header::before {
    content: "▸";
    color: var(--accent);
    margin-right: 8px;
    display: inline-block;
  }
  .section-header > span:last-child {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--fg-muted);
    letter-spacing: 0;
  }
  .run-item {
    padding: 14px 20px;
    border-bottom: 1px solid var(--line);
    cursor: pointer;
    transition: background 120ms ease;
    position: relative;
  }
  .run-item:hover { background: var(--surface); }
  .run-item.active {
    background: var(--surface);
  }
  .run-item.active::before {
    content: "";
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 2px;
    background: var(--accent);
    box-shadow: 0 0 10px var(--accent-glow);
  }
  .run-item .run-id {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--accent);
    font-weight: 500;
    letter-spacing: 0.01em;
  }
  .run-item .suite {
    font-family: var(--sans);
    color: var(--fg);
    font-size: 13px;
    margin-top: 4px;
  }
  .run-item .ts {
    font-family: var(--mono);
    color: var(--fg-muted);
    font-size: 10.5px;
    margin-top: 4px;
    letter-spacing: 0.02em;
  }
  section.detail { padding: 28px 32px; overflow: auto; }
  .detail-header { margin-bottom: 28px; }
  .detail-eyebrow {
    font-family: var(--mono);
    font-size: 10.5px;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--fg-muted);
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
  }
  .detail-eyebrow::before {
    content: "";
    width: 8px; height: 8px;
    background: var(--accent);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--accent-glow);
  }
  .detail-title {
    font-family: var(--serif);
    font-weight: 500;
    font-size: 32px;
    letter-spacing: -0.02em;
    margin: 0 0 8px;
    line-height: 1.1;
    color: var(--fg);
  }
  .detail-title .run-ref {
    font-family: var(--mono);
    color: var(--accent);
    font-size: 14px;
    font-weight: 400;
    letter-spacing: 0.02em;
    margin-left: 12px;
    vertical-align: middle;
  }
  .detail-sub {
    font-family: var(--mono);
    color: var(--fg-dim);
    font-size: 11.5px;
    letter-spacing: 0.02em;
    display: flex;
    gap: 20px;
    margin-top: 6px;
  }
  .detail-sub b {
    color: var(--fg);
    font-weight: 500;
  }
  table.heatmap {
    border-collapse: separate;
    border-spacing: 0;
    width: 100%;
    margin: 8px 0 24px;
    font-family: var(--mono);
    font-size: 12px;
    border: 1px solid var(--line);
    border-radius: 6px;
    overflow: hidden;
  }
  table.heatmap th, table.heatmap td {
    text-align: left;
    padding: 10px 14px;
    border-bottom: 1px solid var(--line);
    border-right: 1px solid var(--line);
  }
  table.heatmap th:last-child, table.heatmap td:last-child { border-right: 0; }
  table.heatmap tr:last-child td { border-bottom: 0; }
  table.heatmap thead th {
    font-family: var(--sans);
    font-weight: 500;
    font-size: 10px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--fg-muted);
    background: var(--bg-2);
    position: sticky;
    top: 0;
  }
  table.heatmap tbody td:first-child {
    font-family: var(--sans);
    color: var(--fg);
    font-weight: 500;
  }
  table.heatmap tbody tr:hover { background: var(--surface); }
  td.score-cell {
    text-align: right;
    font-variant-numeric: tabular-nums;
    font-weight: 500;
    letter-spacing: 0.02em;
    position: relative;
  }
  td.score-cell::after {
    content: "";
    position: absolute;
    left: 0; bottom: 0;
    width: var(--fill, 0%);
    height: 2px;
    background: currentColor;
    opacity: 0.6;
  }
  .score-top { color: var(--accent); }
  .score-mid { color: #d4c07a; }
  .score-low { color: var(--err); }
  .score-none { color: var(--fg-muted); }
  .pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 8px;
    border-radius: 3px;
    border: 1px solid var(--line-2);
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.12em;
    color: var(--fg-dim);
    text-transform: uppercase;
  }
  .pill::before {
    content: "";
    width: 5px; height: 5px;
    background: var(--accent);
    border-radius: 50%;
    box-shadow: 0 0 6px var(--accent-glow);
  }
  .legend {
    display: flex;
    align-items: center;
    gap: 18px;
    font-family: var(--mono);
    font-size: 10.5px;
    color: var(--fg-muted);
    letter-spacing: 0.02em;
  }
  .legend-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  .legend-swatch {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 2px;
  }
  .empty-state {
    padding: 40px;
    color: var(--fg-muted);
    font-family: var(--sans);
    font-style: italic;
    text-align: center;
  }
  ::-webkit-scrollbar { width: 8px; height: 8px; }
  ::-webkit-scrollbar-thumb { background: var(--line-2); border-radius: 4px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--fg-muted); }
</style>
</head>
<body>
<header class="topbar">
  <div class="brand">
    <span>agent<em>·</em>lab</span>
    <span class="tag">evaluation terminal</span>
  </div>
  <div class="topbar-stats" id="topbar-stats">
    <div class="stat"><span class="label">runs</span><b id="stat-runs">—</b></div>
    <div class="stat"><span class="label">latest</span><b id="stat-latest">—</b></div>
  </div>
</header>
<main>
  <aside class="runs-list">
    <div class="section-header"><span>Runs</span><span id="runs-count">0</span></div>
    <div id="runs"></div>
  </aside>
  <section class="detail" id="detail">
    <div class="empty-state">Select a run from the rail to the left.</div>
  </section>
</main>
<script>
// Escape untrusted strings before interpolating into innerHTML.
function esc(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function fmtScore(avg) {
  if (avg === null || avg === undefined) return { cls: "score-none", text: "—", fill: 0 };
  if (avg >= 0.7) return { cls: "score-top", text: avg.toFixed(3), fill: Math.round(avg * 100) };
  if (avg >= 0.35) return { cls: "score-mid", text: avg.toFixed(3), fill: Math.round(avg * 100) };
  return { cls: "score-low", text: avg.toFixed(3), fill: Math.round(avg * 100) };
}

async function loadRuns() {
  const r = await fetch("/api/runs");
  const runs = await r.json();
  const el = document.getElementById("runs");
  document.getElementById("runs-count").textContent = String(runs.length).padStart(3, "0");
  document.getElementById("stat-runs").textContent = runs.length;
  document.getElementById("stat-latest").textContent = runs[0]
    ? new Date(runs[0].started_at * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : "—";

  el.innerHTML = runs.map(x => `
    <div class="run-item" data-id="${esc(x.id)}">
      <div class="run-id">${esc(x.id)}</div>
      <div class="suite">${esc(x.suite_name)}<span style="color:var(--fg-muted)"> @ ${esc(x.suite_version)}</span></div>
      <div class="ts">${esc(new Date(x.started_at * 1000).toLocaleString([], { dateStyle: "short", timeStyle: "short" }))}</div>
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
  const agentAvgs = {};
  for (const x of run.results) {
    const k = x.task_id;
    byTask[k] = byTask[k] || {};
    byTask[k][x.agent_id] = (byTask[k][x.agent_id] || []).concat([x]);
    agentAvgs[x.agent_id] = (agentAvgs[x.agent_id] || []).concat([x.score ?? 0]);
  }
  const rows = Object.keys(byTask).sort().map(task => {
    const cells = agents.map(a => {
      const xs = byTask[task][a] || [];
      const avg = xs.length ? xs.map(x => x.score ?? 0).reduce((p,c)=>p+c,0) / xs.length : null;
      const f = fmtScore(avg);
      return `<td class="score-cell ${f.cls}" style="--fill:${f.fill}%">${f.text}</td>`;
    }).join("");
    return `<tr><td>${esc(task)}</td>${cells}</tr>`;
  }).join("");

  // Row of per-agent averages (footer).
  const footerCells = agents.map(a => {
    const scores = agentAvgs[a] || [];
    const avg = scores.length ? scores.reduce((p,c)=>p+c,0) / scores.length : null;
    const f = fmtScore(avg);
    return `<td class="score-cell ${f.cls}" style="--fill:${f.fill}%"><b>${f.text}</b></td>`;
  }).join("");

  const started = new Date(run.started_at * 1000);
  const elapsed = run.finished_at
    ? Math.round(run.finished_at - run.started_at)
    : null;

  document.getElementById("detail").innerHTML = `
    <div class="detail-header">
      <div class="detail-eyebrow">run · ${esc(run.suite_name)}</div>
      <h1 class="detail-title">
        ${esc(run.suite_name)}@${esc(run.suite_version)}
        <span class="run-ref">${esc(run.id)}</span>
      </h1>
      <div class="detail-sub">
        <span>started <b>${esc(started.toLocaleString())}</b></span>
        ${elapsed !== null ? `<span>duration <b>${elapsed}s</b></span>` : ``}
        <span>rows <b>${run.results.length}</b></span>
        <span>agents <b>${agents.length}</b></span>
      </div>
    </div>
    <table class="heatmap">
      <thead>
        <tr>
          <th style="width:40%">task</th>
          ${agents.map(a => `<th>${esc(a)}</th>`).join("")}
        </tr>
      </thead>
      <tbody>${rows}</tbody>
      <tfoot>
        <tr><td style="font-family:var(--mono); color:var(--fg-muted); letter-spacing:.18em; text-transform:uppercase; font-size:10px">mean</td>${footerCells}</tr>
      </tfoot>
    </table>
    <div class="legend">
      <span class="legend-item"><span class="legend-swatch" style="background:var(--accent)"></span>≥ 0.70</span>
      <span class="legend-item"><span class="legend-swatch" style="background:#d4c07a"></span>0.35 – 0.70</span>
      <span class="legend-item"><span class="legend-swatch" style="background:var(--err)"></span>&lt; 0.35</span>
      <span class="legend-item"><span class="legend-swatch" style="background:var(--fg-muted); opacity:.4"></span>not run</span>
    </div>`;
}
loadRuns();
</script>
</body>
</html>
"""
