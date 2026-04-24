"""SQLite-backed result store.

Thread/asyncio safety: one ``Store`` wraps one sqlite3 connection. The
connection is opened with ``check_same_thread=False`` and all writes are
serialized by an internal ``threading.Lock`` so multiple concurrent
asyncio tasks (running on the same loop thread) plus an opportunistic
dashboard reader thread can safely share it. SQLite's own busy-timeout
is set to 5s so brief contention yields retries rather than errors.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from agentlab.core.types import AgentDef, Result, Run

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    finished_at REAL,
    suite_name TEXT NOT NULL,
    suite_version TEXT NOT NULL,
    agents_json TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS results (
    run_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    trial_idx INTEGER NOT NULL,
    started_at REAL NOT NULL,
    finished_at REAL,
    status TEXT NOT NULL,
    score REAL,
    scorer_results_json TEXT,
    trajectory_blob BLOB,
    tokens_in INTEGER,
    tokens_out INTEGER,
    cost_usd REAL,
    error TEXT,
    PRIMARY KEY (run_id, task_id, agent_id, trial_idx),
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id);
CREATE INDEX IF NOT EXISTS idx_results_score ON results(score);
"""


class Store:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(
            self.db_path, isolation_level=None, check_same_thread=False
        )
        self._lock = threading.Lock()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def insert_run(self, run: Run) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO runs(id, started_at, finished_at, "
                "suite_name, suite_version, agents_json, notes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    run.id,
                    run.started_at,
                    run.finished_at,
                    run.suite_name,
                    run.suite_version,
                    json.dumps([a.model_dump() for a in run.agents]),
                    run.notes,
                ),
            )

    def finish_run(self, run_id: str, finished_at: float) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE runs SET finished_at=? WHERE id=?", (finished_at, run_id)
            )

    def insert_result(self, res: Result) -> None:
        traj_blob = gzip.compress(json.dumps(res.trajectory).encode("utf-8"))
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO results("
                "run_id, task_id, agent_id, trial_idx, started_at, finished_at, "
                "status, score, scorer_results_json, trajectory_blob, "
                "tokens_in, tokens_out, cost_usd, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    res.run_id,
                    res.task_id,
                    res.agent_id,
                    res.trial_idx,
                    res.started_at,
                    res.finished_at,
                    res.status,
                    res.score,
                    json.dumps([s.model_dump() for s in res.scorer_results]),
                    traj_blob,
                    res.tokens_in,
                    res.tokens_out,
                    res.cost_usd,
                    res.error,
                ),
            )

    def list_runs(self, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, started_at, finished_at, suite_name, suite_version, notes "
                "FROM runs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "started_at": r[1],
                "finished_at": r[2],
                "suite_name": r[3],
                "suite_version": r[4],
                "notes": r[5],
            }
            for r in rows
        ]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, started_at, finished_at, suite_name, suite_version, agents_json, notes "
                "FROM runs WHERE id=?",
                (run_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "started_at": row[1],
            "finished_at": row[2],
            "suite_name": row[3],
            "suite_version": row[4],
            "agents": [AgentDef(**a).model_dump() for a in json.loads(row[5])],
            "notes": row[6],
        }

    def list_results(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT task_id, agent_id, trial_idx, started_at, finished_at, "
                "status, score, scorer_results_json, tokens_in, tokens_out, cost_usd, error "
                "FROM results WHERE run_id=? ORDER BY task_id, agent_id, trial_idx",
                (run_id,),
            )
            rows = cur.fetchall()
        return [
            {
                "task_id": r[0],
                "agent_id": r[1],
                "trial_idx": r[2],
                "started_at": r[3],
                "finished_at": r[4],
                "status": r[5],
                "score": r[6],
                "scorer_results": json.loads(r[7] or "[]"),
                "tokens_in": r[8],
                "tokens_out": r[9],
                "cost_usd": r[10],
                "error": r[11],
            }
            for r in rows
        ]

    def get_trajectory(
        self, run_id: str, task_id: str, agent_id: str, trial_idx: int
    ) -> list[dict[str, Any]] | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT trajectory_blob FROM results "
                "WHERE run_id=? AND task_id=? AND agent_id=? AND trial_idx=?",
                (run_id, task_id, agent_id, trial_idx),
            )
            row = cur.fetchone()
        if not row:
            return None
        blob = row[0]
        if blob is None:
            return []
        try:
            return json.loads(gzip.decompress(blob).decode("utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            # Corrupted blob — surface as an empty trajectory rather than
            # blowing up the whole request.
            return []
