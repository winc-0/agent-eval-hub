"""维度 3 · 轨迹效率（trajectory）—— 代码判定，默认只做诊断、不参与门禁。

结果对了但走了弯路（重复查询、步数超标、没正常收尾）在这里体现。
它不进门禁的原因：轨迹是手段不是目的，拿它卡通过率会惩罚“换了条路但做对了”的模型。
"""
from __future__ import annotations

from ..models import Check
from .base import EvalContext, dumps, make_score


async def evaluate(ctx: EvalContext):
    trace, budget = ctx.trace, ctx.budget
    calls = trace.tool_calls
    checks: list[Check] = []

    if budget.max_steps is not None:
        ok = len(calls) <= budget.max_steps
        checks.append(Check(id="steps_within_budget", passed=ok,
                            detail=f"工具调用 {len(calls)} 次，预算 {budget.max_steps} 次",
                            tag="" if ok else "trajectory.too_many_steps"))

    succeeded: set[str] = set()
    redundant = []
    for c in calls:
        key = f"{c.name}:{dumps(sorted(c.args.items()))}"
        if key in succeeded:        # 上一次同样的调用已经成功，再调就是浪费；失败后的重试不算
            redundant.append(f"第 {c.step} 步重复调用 {c.name}")
        if not c.error:
            succeeded.add(key)
    checks.append(Check(id="no_redundant_calls", passed=not redundant,
                        detail=redundant[0] if redundant else "没有重复调用",
                        tag="trajectory.redundant_call" if redundant else ""))

    ok = trace.stop_reason == "end_turn"
    checks.append(Check(id="finished_normally", passed=ok,
                        detail="正常结束" if ok else f"未正常结束：{trace.stop_reason}",
                        tag="" if ok else f"trajectory.{trace.stop_reason}"))
    return make_score("trajectory", checks)
