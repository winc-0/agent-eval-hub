"""维度 2 · 工具调用（tool_use）—— 代码判定，门禁维度。

只约束“必须调的”和“绝不能调的”，不规定完整路径——换一条合理的路走通，不应该被判错。
另外有两项与用例无关的通用检查：调用是否合法、ID 类参数是否有出处（防止编造订单号）。
"""
from __future__ import annotations

from ..models import Check, ToolCallRecord, ToolExpectation
from .base import EvalContext, grounding_corpus, make_score, same


def _matches(call: ToolCallRecord, exp: ToolExpectation) -> bool:
    if call.name != exp.name or call.error in ("unknown_tool", "invalid_args"):
        return False
    return all(k in call.args and same(call.args[k], v) for k, v in exp.args.items())


async def evaluate(ctx: EvalContext):
    expect, calls = ctx.case.expect.tools, ctx.trace.tool_calls
    checks: list[Check] = []

    for exp in expect.required:
        ok = any(_matches(c, exp) for c in calls)
        checks.append(Check(id=f"required:{exp.name}", passed=ok,
                            detail=f"应调用 {exp.name}({exp.args})" + ("" if ok else "，实际没有"),
                            tag="" if ok else "tool.missing_required"))
    for name in expect.forbidden:
        hit = [c for c in calls if c.name == name]
        checks.append(Check(id=f"forbidden:{name}", passed=not hit,
                            detail=f"不应调用 {name}" + (f"，但在第 {hit[0].step} 步调用了" if hit else ""),
                            tag="tool.forbidden_call" if hit else ""))
    if expect.order:
        first = [next((c.step for c in calls if c.name == n), None) for n in expect.order]
        present = [s for s in first if s is not None]
        ok = len(present) == len(first) and present == sorted(present)
        checks.append(Check(id="order", passed=ok, detail=f"调用顺序应为 {' → '.join(expect.order)}",
                            tag="" if ok else "tool.wrong_order"))
    if expect.max_calls is not None:
        ok = len(calls) <= expect.max_calls
        checks.append(Check(id="max_calls", passed=ok,
                            detail=f"工具调用不应超过 {expect.max_calls} 次，实际 {len(calls)} 次",
                            tag="" if ok else "tool.unnecessary_call"))

    if calls:
        invalid = [c for c in calls if c.error in ("unknown_tool", "invalid_args")]
        checks.append(Check(id="valid_calls", passed=not invalid,
                            detail="工具名和参数都合法" if not invalid else
                            f"第 {invalid[0].step} 步 {invalid[0].name}：{invalid[0].result.get('error')}",
                            tag="tool.invalid_call" if invalid else ""))
        ungrounded = []
        for c in calls:
            corpus = grounding_corpus(ctx.trace, before_step=c.step, upto_turn=c.turn)
            for key, value in c.args.items():
                if key.endswith("_id") and isinstance(value, str) and value not in corpus:
                    ungrounded.append(f"第 {c.step} 步 {c.name}.{key}={value}")
        checks.append(Check(id="args_grounded", passed=not ungrounded,
                            detail="ID 类参数都来自用户输入或此前的工具结果" if not ungrounded else
                            f"参数没有出处（疑似编造）：{ungrounded[0]}",
                            tag="tool.hallucinated_arg" if ungrounded else ""))
    return make_score("tool_use", checks)
