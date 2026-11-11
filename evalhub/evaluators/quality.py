"""维度 5 · 回答质量（quality）—— LLM 裁判，默认只做诊断。

语气、是否答到点上、有没有给出下一步，这些代码判不了，交给分级裁判。
rubric 逐条独立判定（一条一次调用），不让裁判给一个笼统的总分。
没配置裁判、或用例没有 rubric 时，本维度跳过。
"""
from __future__ import annotations

import asyncio

from ..judge import JudgeRequest
from ..models import Check, JudgeMeta
from .base import EvalContext, dumps, make_score


async def evaluate(ctx: EvalContext):
    rubric = ctx.suite.rubric_for(ctx.case)
    if ctx.judge is None or not rubric:
        return make_score("quality", [], grader="llm")
    trace = ctx.trace
    dialogue = "\n".join(f"用户：{u}" for u in trace.user_messages)
    tools = dumps([{"tool": c.name, "args": c.args, "result": c.result} for c in trace.tool_calls])
    answer = "\n".join(trace.answers)
    results = await asyncio.gather(*(ctx.judge.judge(JudgeRequest(item, answer, dialogue, tools))
                                     for item in rubric))
    checks = []
    for item, res in zip(rubric, results):
        ok = res.verdict == "pass"
        where = "规则" if res.tier == 0 else f"第 {res.tier} 级裁判"
        checks.append(Check(id=f"rubric:{item.id}", passed=ok,
                            detail=f"{item.text} —— {res.reason}（{where}{'，待人工复核' if res.needs_human else ''}）",
                            tag="" if ok else f"quality.{item.id}"))
    meta = JudgeMeta(tier=max(r.tier for r in results), model=",".join(sorted({r.model for r in results})),
                     escalated=any(r.escalated for r in results),
                     needs_human=any(r.needs_human for r in results),
                     calls=sum(r.calls for r in results), cost_usd=sum(r.cost_usd for r in results))
    return make_score("quality", checks, grader="llm", judge=meta)
