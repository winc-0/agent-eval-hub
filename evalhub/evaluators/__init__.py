"""7 维评估器的注册与汇总。"""
from __future__ import annotations

import asyncio

from ..models import Score
from . import efficiency, faithfulness, outcome, quality, safety, tool_use, trajectory
from .base import EvalContext

EVALUATORS = {
    "outcome": outcome.evaluate,
    "tool_use": tool_use.evaluate,
    "trajectory": trajectory.evaluate,
    "faithfulness": faithfulness.evaluate,
    "quality": quality.evaluate,
    "safety": safety.evaluate,
    "efficiency": efficiency.evaluate,
}


async def evaluate_all(ctx: EvalContext) -> list[Score]:
    return list(await asyncio.gather(*(fn(ctx) for fn in EVALUATORS.values())))


def verdict(scores: list[Score], gating: list[str]) -> tuple[bool, list[str]]:
    """一次 trial 是否通过 = 所有门禁维度都通过。失败标签收集全部维度的，诊断时更完整。"""
    passed = all(s.passed for s in scores if s.dimension in gating and not s.skipped)
    tags = sorted({c.tag for s in scores for c in s.checks if not c.passed and c.tag})
    return passed, tags


__all__ = ["EVALUATORS", "EvalContext", "evaluate_all", "verdict"]
