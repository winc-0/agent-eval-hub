"""维度 1 · 任务结果（outcome）—— 代码判定，门禁维度。

判的是环境的【终态】，不是对话记录：退款单在不在、金额对不对、不该动的有没有被动。
Agent 说“已为您退款”不算数，数据库里有才算。回复文本只核对少量关键事实。
"""
from __future__ import annotations

from ..models import Check
from .base import EvalContext, includes, lookup, make_score, normalize_text, same


async def evaluate(ctx: EvalContext):
    expect, trace = ctx.case.expect, ctx.trace
    checks: list[Check] = []

    for a in expect.final_state:
        found, value = lookup(trace.final_state, a.path)
        if a.unchanged:
            was_found, was = lookup(trace.initial_state, a.path)
            ok = found == was_found and same(value, was)
            detail = f"{a.path} 应保持不变" + ("" if ok else f"，实际变为 {value!r}")
        elif a.exists is not None:
            ok = found == a.exists
            detail = f"{a.path} 应{'存在' if a.exists else '不存在'}" + ("" if ok else "，实际相反")
        else:
            ok = found and same(value, a.equals)
            detail = f"{a.path} 应为 {a.equals!r}" + ("" if ok else f"，实际为 {value!r}" if found else "，实际不存在")
        checks.append(Check(id=f"state:{a.path}", passed=ok, detail=detail, tag="" if ok else "outcome.wrong_state"))

    answer = normalize_text("\n".join(trace.answers))
    for item in expect.answer.must_include:
        ok = includes(answer, item)
        checks.append(Check(id=f"answer_has:{item}", passed=ok,
                            detail=f"回复应包含“{item}”" + ("" if ok else "，未找到"),
                            tag="" if ok else "outcome.missing_fact"))
    for item in expect.answer.must_not_include:
        ok = not includes(answer, item)
        checks.append(Check(id=f"answer_lacks:{item}", passed=ok,
                            detail=f"回复不应包含“{item}”" + ("" if ok else "，但出现了"),
                            tag="" if ok else "outcome.forbidden_content"))
    return make_score("outcome", checks)
