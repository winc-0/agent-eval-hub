"""Badcase 闭环：失败不是看一眼就过，而是沉淀成以后每次都会跑的用例。

  离线评测失败 ─┐
                ├→ Badcase 池（按 用例 + 失败签名 去重、累计出现次数）
  线上轨迹回流 ─┘         │
                          ├→ pin：修复后钉为回归用例，此后零容忍
                          └→ 线上问题自动生成用例草稿（drafts/），人工补全期望后移入 cases/

线上轨迹没有沙箱终态，所以只跑不依赖终态的评估器（忠实度、工具参数出处、安全策略），
再加上用户点踩信号。草稿里的期望必须由人来写——让模型自己给自己出标准答案没有意义。
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Optional

import yaml

from .evaluators import EvalContext, verdict
from .evaluators import faithfulness, safety, tool_use
from .models import Case, ToolCallRecord, Trace, UserSpec
from .store import Store
from .suite import Suite


def collect_badcases(store: Store, run_id: str) -> int:
    """把一次运行里失败的 trial 归入 Badcase 池，返回本次归入的条数。"""
    run = store.get_run(run_id)
    n = 0
    for t in store.trials(run_id):
        if t.status == "scored" and not t.passed:
            store.upsert_badcase(run["suite"], t.case_id, t.failure_tags, run_id)
            n += 1
    return n


def pin_regression(suite: Suite, case_id: str) -> Path:
    """把用例钉为回归用例（写入 regression.txt）。之后门禁对它零容忍。"""
    suite.case(case_id)   # 不存在会抛 KeyError
    path = suite.path / "regression.txt"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else [
        "# 被钉为回归用例的 id，一行一个。门禁对这些用例零容忍。"]
    if case_id not in lines:
        lines.append(case_id)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def trace_from_record(rec: dict[str, Any]) -> Trace:
    """线上轨迹（JSONL 一行）→ Trace。字段：trace_id, user_messages, tool_calls, answers, model?"""
    calls = [ToolCallRecord(id=f"call_{i + 1}", step=i + 1, name=c["name"], args=c.get("args", {}),
                            result=c.get("result"), error=c.get("error"), turn=c.get("turn", 0))
             for i, c in enumerate(rec.get("tool_calls", []))]
    return Trace(trace_id=str(rec.get("trace_id", "")), case_id=f"prod_{rec.get('trace_id', 'x')}",
                 agent="production", model=rec.get("model", "unknown"),
                 user_messages=list(rec.get("user_messages", [])), answers=list(rec.get("answers", [])),
                 tool_calls=calls)


async def _screen(suite: Suite, trace: Trace) -> list[str]:
    case = Case(id=trace.case_id, title="线上轨迹", user=UserSpec(turns=trace.user_messages or [""]))
    ctx = EvalContext(case, trace, suite, None)
    scores = [await fn(ctx) for fn in (faithfulness.evaluate, tool_use.evaluate, safety.evaluate)]
    return verdict(scores, [])[1]


def import_traces(suite: Suite, store: Store, jsonl_path: str | Path) -> list[dict[str, Any]]:
    """筛查线上轨迹，把有问题的生成用例草稿。返回每条轨迹的筛查结果。"""
    drafts = suite.path / "drafts"
    out = []
    for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        trace = trace_from_record(rec)
        tags = asyncio.run(_screen(suite, trace))
        if rec.get("feedback") == "down":
            tags = sorted(set(tags) | {"user.thumbs_down"})
        row = {"trace_id": trace.trace_id, "tags": tags, "draft": None}
        if tags:
            drafts.mkdir(exist_ok=True)
            case_id = "prod_" + re.sub(r"[^A-Za-z0-9_]", "_", trace.trace_id)
            draft = {
                "id": case_id, "title": "【待补全】线上问题：" + "、".join(tags),
                "tags": ["production"], "category": "regression", "source": "production",
                "user": {"turns": trace.user_messages},
                "env": {"overrides": {}},
                "expect": {"final_state": [], "tools": {"required": [], "forbidden": []},
                           "answer": {"must_include": [], "must_not_include": []}},
                "notes": "自动筛查命中：" + "、".join(tags) + "。线上实际回复：" + " / ".join(trace.answers)
                         + "。请补全 env（复现当时的数据）与 expect（正确行为），再移入 cases/。",
            }
            path = drafts / f"{case_id}.yaml"
            path.write_text(yaml.safe_dump(draft, allow_unicode=True, sort_keys=False), encoding="utf-8")
            store.upsert_badcase(suite.config.name, case_id, tags, "production", source="production",
                                 note=str(path))
            row["draft"] = str(path)
        out.append(row)
    return out


def promote_draft(suite: Suite, case_id: str) -> Path:
    """草稿补全后移入 cases/production/。期望为空的草稿拒绝入库。"""
    src = suite.path / "drafts" / f"{case_id}.yaml"
    case = Case(**yaml.safe_load(src.read_text(encoding="utf-8")))
    e = case.expect
    if not (e.final_state or e.tools.required or e.tools.forbidden or e.answer.must_include
            or e.answer.must_not_include):
        raise ValueError(f"{case_id} 的期望还是空的：先由人补全正确行为，再入库")
    dst_dir = suite.path / "cases" / "production"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    src.replace(dst)
    return dst
