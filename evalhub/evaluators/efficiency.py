"""维度 7 · 成本与延迟（efficiency）—— 代码判定，默认只做诊断。

token 和费用取自 API 返回的 usage 与实际服务模型；延迟只含模型调用和工具调用，
不含重试等待和排队——否则撞了限流的那个版本会被冤枉成“更慢”。
"""
from __future__ import annotations

from ..models import Check
from .base import EvalContext, make_score


async def evaluate(ctx: EvalContext):
    trace, budget = ctx.trace, ctx.budget
    if trace.usage.llm_calls == 0:   # oracle / null 这类不调模型的被测对象，没有成本可言
        return make_score("efficiency", [])
    checks: list[Check] = []
    for name, actual, limit, unit in (
        ("tokens", trace.usage.total_tokens, budget.max_tokens, ""),
        ("latency", trace.latency_ms, budget.max_latency_ms, " ms"),
        ("cost", trace.usage.cost_usd, budget.max_cost_usd, ""),
    ):
        if limit is None:
            continue
        ok = actual <= limit
        checks.append(Check(id=f"{name}_within_budget", passed=ok,
                            detail=f"{name} {actual:.4g}{unit}，预算 {limit:g}{unit}",
                            tag="" if ok else f"efficiency.{name}_over_budget"))
    return make_score("efficiency", checks)
