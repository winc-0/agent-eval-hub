"""SQLite 持久化：运行、trial 结果（含完整轨迹和评分明细）、Badcase 池、人工复核队列。

每次 trial 的完整轨迹和裁判的输入输出都落库——分数出人意料时，能直接回放定位是
模型的问题还是评测自身的 bug，不用重跑。
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Optional

from .models import TrialResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, suite TEXT, suite_fingerprint TEXT, agent TEXT, model TEXT, judge TEXT,
  tag TEXT, config_json TEXT, cases_json TEXT, gating_json TEXT, status TEXT,
  started_at REAL, finished_at REAL);
CREATE TABLE IF NOT EXISTS trials (
  run_id TEXT, case_id TEXT, trial INTEGER, status TEXT, passed INTEGER, error_class TEXT,
  attempts INTEGER, data_json TEXT, PRIMARY KEY (run_id, case_id, trial));
CREATE TABLE IF NOT EXISTS badcases (
  id INTEGER PRIMARY KEY AUTOINCREMENT, suite TEXT, case_id TEXT, signature TEXT, failure_tags TEXT,
  first_run TEXT, last_run TEXT, occurrences INTEGER, status TEXT, source TEXT, note TEXT,
  created_at REAL, UNIQUE (suite, case_id, signature));
CREATE TABLE IF NOT EXISTS human_reviews (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, case_id TEXT, trial INTEGER, dimension TEXT,
  detail TEXT, status TEXT, label TEXT, created_at REAL);
"""


def new_run_id() -> str:
    return f"r{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"


class Store:
    def __init__(self, path: str = "evalhub.db"):
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self.conn.executescript(SCHEMA)

    def _exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    def _all(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    # ---- 运行 ----

    def create_run(self, run_id: str, suite: str, fingerprint: str, agent: str, model: str, judge: str,
                   tag: str, config: dict, cases: dict, gating: list[str]) -> None:
        self._exec("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (run_id, suite, fingerprint, agent, model, judge, tag,
                    json.dumps(config, ensure_ascii=False), json.dumps(cases, ensure_ascii=False),
                    json.dumps(gating), "running", time.time(), None))

    def finish_run(self, run_id: str, status: str = "finished") -> None:
        self._exec("UPDATE runs SET status=?, finished_at=? WHERE run_id=?", (status, time.time(), run_id))

    @staticmethod
    def _run_row(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["config"] = json.loads(d.pop("config_json"))
        d["cases"] = json.loads(d.pop("cases_json"))
        d["gating"] = json.loads(d.pop("gating_json"))
        return d

    def get_run(self, run_id: str) -> dict[str, Any]:
        rows = self._all("SELECT * FROM runs WHERE run_id=?", (run_id,))
        if not rows:
            raise KeyError(f"没有这次运行：{run_id}")
        return self._run_row(rows[0])

    def list_runs(self, suite: Optional[str] = None, limit: int = 50) -> list[dict[str, Any]]:
        sql, params = "SELECT * FROM runs", ()
        if suite:
            sql, params = sql + " WHERE suite=?", (suite,)
        return [self._run_row(r) for r in self._all(sql + " ORDER BY started_at DESC LIMIT ?", params + (limit,))]

    def resolve_run(self, ref: str, suite: Optional[str] = None) -> str:
        """ref 可以是 run_id、`latest`，或 `tag:<标签>`（取该标签最近一次已完成的运行）。"""
        if ref == "latest" or ref.startswith("tag:"):
            tag = ref[4:] if ref.startswith("tag:") else None
            for run in self.list_runs(suite, limit=500):
                if run["status"] == "finished" and (tag is None or run["tag"] == tag):
                    return run["run_id"]
            raise KeyError(f"找不到符合 {ref} 的已完成运行")
        return self.get_run(ref)["run_id"]

    # ---- trial ----

    def save_trial(self, result: TrialResult) -> None:
        self._exec("INSERT OR REPLACE INTO trials VALUES (?,?,?,?,?,?,?,?)",
                   (result.run_id, result.case_id, result.trial, result.status, int(result.passed),
                    result.error_class, result.attempts, result.model_dump_json()))

    def trials(self, run_id: str) -> list[TrialResult]:
        rows = self._all("SELECT data_json FROM trials WHERE run_id=? ORDER BY case_id, trial", (run_id,))
        return [TrialResult.model_validate_json(r["data_json"]) for r in rows]

    def completed(self, run_id: str) -> set[tuple[str, int]]:
        """断点续跑用：已经有可评分结果的 (用例, 第几次)。故障的那些会重跑。"""
        rows = self._all("SELECT case_id, trial FROM trials WHERE run_id=? AND status!='error'", (run_id,))
        return {(r["case_id"], r["trial"]) for r in rows}

    # ---- Badcase 池 ----

    def upsert_badcase(self, suite: str, case_id: str, tags: list[str], run_id: str,
                       source: str = "eval", note: str = "") -> None:
        signature = "|".join(tags) or "unknown"
        cur = self._exec("UPDATE badcases SET occurrences=occurrences+1, last_run=? "
                         "WHERE suite=? AND case_id=? AND signature=?", (run_id, suite, case_id, signature))
        if cur.rowcount == 0:
            self._exec("INSERT INTO badcases (suite, case_id, signature, failure_tags, first_run, last_run, "
                       "occurrences, status, source, note, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                       (suite, case_id, signature, json.dumps(tags), run_id, run_id, 1, "open", source,
                        note, time.time()))

    def badcases(self, suite: Optional[str] = None, status: Optional[str] = None) -> list[dict[str, Any]]:
        sql, cond, params = "SELECT * FROM badcases", [], []
        if suite:
            cond.append("suite=?"); params.append(suite)
        if status:
            cond.append("status=?"); params.append(status)
        if cond:
            sql += " WHERE " + " AND ".join(cond)
        out = []
        for r in self._all(sql + " ORDER BY occurrences DESC, id", tuple(params)):
            d = dict(r)
            d["failure_tags"] = json.loads(d["failure_tags"])
            out.append(d)
        return out

    def set_badcase_status(self, badcase_id: int, status: str, note: str = "") -> None:
        self._exec("UPDATE badcases SET status=?, note=CASE WHEN ?='' THEN note ELSE ? END WHERE id=?",
                   (status, note, note, badcase_id))

    # ---- 人工复核队列 ----

    def add_human_review(self, run_id: str, case_id: str, trial: int, dimension: str, detail: str) -> None:
        self._exec("INSERT INTO human_reviews (run_id, case_id, trial, dimension, detail, status, label, "
                   "created_at) VALUES (?,?,?,?,?,?,?,?)",
                   (run_id, case_id, trial, dimension, detail, "pending", None, time.time()))

    def human_reviews(self, run_id: Optional[str] = None, status: str = "pending") -> list[dict[str, Any]]:
        sql, params = "SELECT * FROM human_reviews WHERE status=?", [status]
        if run_id:
            sql += " AND run_id=?"; params.append(run_id)
        return [dict(r) for r in self._all(sql + " ORDER BY id", tuple(params))]

    def close(self) -> None:
        self.conn.close()
