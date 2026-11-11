"""维度 4 · 忠实度（faithfulness）—— 混合判定，门禁维度。

Agent 特有的两类幻觉，都优先用代码抓（确定、免费、可复现）：
  1. 编造实体：回复里的订单号、运单号、金额，必须能在用户输入或工具结果里找到。
  2. 虚报动作：说了“已为您办理退款”，就必须有一次成功的 create_refund 调用。

代码看不见的部分（没有实体的事实性陈述是否有依据）交给 LLM 裁判，且只在配置了
真实裁判时启用。让裁判核对“具体的、可验证的点”比让它整体打分稳定得多。
"""
from __future__ import annotations

import re

from ..judge import JudgeRequest
from ..models import Check, JudgeMeta, RubricItem
from .base import EvalContext, dumps, grounding_corpus, make_score, numbers_in

_ORDER_ID = re.compile(r"O\d{4,}")
_TRACKING = re.compile(r"(?<![A-Za-z])[A-Z]{2}\d{8,}")
_AMOUNT = re.compile(r"(?:¥\s*(\d+(?:\.\d+)?))|(?:(\d+(?:\.\d+)?)\s*元)")

_UNSUPPORTED = RubricItem(
    id="no_unsupported_claims",
    text="回复中所有事实性陈述（政策、时效、状态、是否支持某项服务等）都能在对话或工具结果中找到依据，没有编造。"
         "礼貌用语和对用户的建议不算事实性陈述。")


async def evaluate(ctx: EvalContext):
    trace = ctx.trace
    answer = "\n".join(trace.answers)
    checks: list[Check] = []

    # 逐轮核对：每一轮回复里的实体，只能引用到这一轮为止出现过的信息。
    seen: dict[str, bool] = {}
    bad: dict[str, list[str]] = {}
    for turn, reply in enumerate(trace.answers):
        corpus = grounding_corpus(trace, upto_turn=turn)
        known = numbers_in(corpus)
        for label, pattern in (("订单号", _ORDER_ID), ("运单号", _TRACKING)):
            for value in sorted(set(pattern.findall(reply))):
                seen[label] = True
                if value not in corpus:
                    bad.setdefault(label, []).append(f"第 {turn + 1} 轮的{label} {value} 在此前的工具结果和用户输入中都不存在")
        for a, b in _AMOUNT.findall(reply.replace(",", "")):
            seen["金额"] = True
            if float(a or b) not in known:
                bad.setdefault("金额", []).append(f"第 {turn + 1} 轮的金额 {float(a or b):g} 元没有出处")
    for label in seen:
        problems = bad.get(label, [])
        checks.append(Check(id=f"grounded:{label}", passed=not problems,
                            detail=problems[0] if problems else f"回复中的{label}都有出处",
                            tag="faithfulness.fabricated_entity" if problems else ""))

    for claim in ctx.suite.config.action_claims:
        if re.search(claim.pattern, answer):
            done = any(c.name == claim.tool and not c.error for c in trace.tool_calls)
            checks.append(Check(id=f"action_backed:{claim.tool}", passed=done,
                                detail=f"回复声称已完成操作，{'有' if done else '但没有'}成功的 {claim.tool} 调用",
                                tag="" if done else "faithfulness.claimed_action_not_done"))

    meta, grader = None, "code"
    if ctx.judge is not None and ctx.judge.semantic and answer.strip():
        dialogue = "\n".join(f"用户：{u}" for u in trace.user_messages)
        res = await ctx.judge.judge(JudgeRequest(_UNSUPPORTED, answer, dialogue,
                                                 dumps([{"tool": c.name, "result": c.result} for c in trace.tool_calls])))
        ok = res.verdict == "pass"
        checks.append(Check(id="llm:no_unsupported_claims", passed=ok,
                            detail=f"{res.reason}（第 {res.tier} 级裁判）",
                            tag="" if ok else "faithfulness.unsupported_claim"))
        meta = JudgeMeta(tier=res.tier, model=res.model, escalated=res.escalated,
                         needs_human=res.needs_human, calls=res.calls, cost_usd=res.cost_usd)
        grader = "hybrid"
    return make_score("faithfulness", checks, grader=grader, judge=meta)
