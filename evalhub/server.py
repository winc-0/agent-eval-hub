"""HTTP 服务：让评测从“某个人本地跑的脚本”变成团队共用的中台。

  GET  /                      运行列表页
  GET  /api/runs              运行列表
  POST /api/runs              发起一次评测（后台执行，立即返回 run_id）
  GET  /api/runs/{id}         汇总指标（不含轨迹）
  GET  /api/runs/{id}/report  HTML 报告，可带 ?baseline=<run_id>
  GET  /api/badcases          Badcase 池
  GET  /api/reviews           待人工复核队列

默认只监听 127.0.0.1。POST /api/runs 会触发模型调用（真实模型要花钱），
对外暴露前必须加鉴权和配额。
"""
from __future__ import annotations

import asyncio
import html
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import badcase as bc
from .agent import make_agent
from .judge import make_judge
from .report import build_report, render_html
from .scheduler import RunConfig, run_suite
from .store import Store, new_run_id
from .suite import Suite


class RunRequest(BaseModel):
    suite: str
    agent: str = "mock:cs-v2"
    judge: str = "mock"
    trials: int = 3
    concurrency: int = 8
    tag: str = ""


def create_app(db_path: str = "evalhub.db", suites_dir: str = "suites") -> FastAPI:
    app = FastAPI(title="Agent 评测中台")
    store = Store(db_path)
    tasks: set[asyncio.Task] = set()

    def get_run(run_id: str) -> dict:
        try:
            return store.get_run(run_id)
        except KeyError:
            raise HTTPException(404, f"没有这次运行：{run_id}")

    @app.get("/api/runs")
    def list_runs(suite: Optional[str] = None) -> list[dict]:
        return [{k: r[k] for k in ("run_id", "suite", "agent", "judge", "tag", "status", "started_at", "finished_at")}
                for r in store.list_runs(suite)]

    @app.post("/api/runs")
    async def start_run(req: RunRequest) -> dict:
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", req.suite):   # 只接受目录名，防止路径穿越
            raise HTTPException(400, "评测集名称不合法")
        root = Path(suites_dir) / req.suite
        if not (root / "suite.yaml").exists():
            raise HTTPException(404, f"没有评测集 {req.suite}")
        if not 1 <= req.trials <= 20 or not 1 <= req.concurrency <= 64:
            raise HTTPException(400, "trials 需在 1~20，concurrency 需在 1~64")
        try:
            suite = Suite.load(root)
            agent, judge = make_agent(req.agent), make_judge(req.judge)
        except Exception as e:
            raise HTTPException(400, f"{type(e).__name__}: {e}")
        run_id = new_run_id()

        async def job() -> None:
            await run_suite(suite, agent, store, RunConfig(trials=req.trials, concurrency=req.concurrency,
                                                           tag=req.tag), judge, run_id=run_id)
            bc.collect_badcases(store, run_id)

        task = asyncio.create_task(job())
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return {"run_id": run_id, "status": "running"}

    @app.get("/api/runs/{run_id}")
    def run_summary(run_id: str, baseline: Optional[str] = None) -> dict:
        get_run(run_id)
        report = build_report(store, run_id, baseline)
        for c in report["cases"]:
            c["trials"] = [{k: t[k] for k in ("trial", "status", "passed", "failure_tags", "attempts")}
                           for t in c["trials"]]
        return report

    @app.get("/api/runs/{run_id}/report", response_class=HTMLResponse)
    def run_report(run_id: str, baseline: Optional[str] = None) -> str:
        get_run(run_id)
        return render_html(build_report(store, run_id, baseline))

    @app.get("/api/badcases")
    def badcases(suite: Optional[str] = None, status: Optional[str] = None) -> list[dict]:
        return store.badcases(suite, status)

    @app.get("/api/reviews")
    def reviews(run_id: Optional[str] = None) -> list[dict]:
        return store.human_reviews(run_id)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        rows = "".join(
            f"<tr><td><a href='/api/runs/{html.escape(r['run_id'])}/report'>{html.escape(r['run_id'])}</a></td>"
            f"<td>{html.escape(r['suite'])}</td><td>{html.escape(r['agent'])}</td><td>{html.escape(r['tag'] or '')}</td>"
            f"<td>{html.escape(r['status'])}</td><td>{datetime.fromtimestamp(r['started_at']):%Y-%m-%d %H:%M}</td></tr>"
            for r in store.list_runs())
        return ("<!doctype html><meta charset='utf-8'><title>Agent 评测中台</title>"
                "<style>body{font:14px/1.6 system-ui,'Microsoft YaHei',sans-serif;max-width:960px;margin:32px auto;padding:0 16px}"
                "table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:8px;text-align:left}</style>"
                "<h1>Agent 评测中台</h1><p>最近的运行（点击 run_id 查看报告）。接口文档见 <a href='/docs'>/docs</a>。</p>"
                f"<table><tr><th>运行</th><th>评测集</th><th>被测</th><th>标签</th><th>状态</th><th>开始时间</th></tr>{rows}</table>")

    return app
