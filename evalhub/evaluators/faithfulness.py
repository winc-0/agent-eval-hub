"""维度 4 · 忠实度（faithfulness）—— 代码判定，门禁维度。

Agent 特有的两类幻觉，都优先用代码抓（确定、免费、可复现）：
  1. 编造实体：回复里的订单号、运单号、金额，必须能在用户输入或工具结果里找到。
  2. 虚报动作：说了“已为您办理退款”，就必须有一次成功的 create_refund 调用。

代码看不见的部分（没有实体的事实性陈述是否有依据）由 quality 维度里的 LLM 裁判判，
只做诊断：那条标准没有经过人工标注校准，不能让它决定一次试验算不算通过。
"""
from __future__ import annotations

import re

from ..models import Check
from .base import EvalContext, grounding_corpus, make_score, numbers_in

_ORDER_ID = re.compile(r"O\d{4,}")
_TRACKING = re.compile(r"(?<![A-Za-z])[A-Z]{2}\d{8,}")

# “请提供订单号（例如 O1001）”是在说明格式，不是在陈述事实，不算编造。
# 判据只看实体所在的那一小句：举例线索必须和实体同处一个小句，
# 否则前半句的“例如”会把后半句真正的断言一起豁免掉。
_CLAUSE_STOP = "。！？；，,、\n（）()[]【】"
_EXAMPLE_WORD = re.compile(r"例如|比如|譬如|举例|示例|样例|格式如|形如|e\.g\.", re.I)
# 单字“如”有歧义——如果、不如、比如都会命中——所以只认紧贴实体的那种：
# “请提供订单号，如 O1001”“如：O1001”。句子中间的“如果订单 O1001”不算。
_EXAMPLE_RU = re.compile(r"(?:^|[，,、\s（(])如\s*[:：]?\s*$")


def _leading_clause(reply: str, start: int) -> str:
    """实体前面的同一小句（到最近一个句读标点为止）。"""
    cut = max((reply.rfind(ch, 0, start) for ch in _CLAUSE_STOP), default=-1)
    return reply[cut + 1:start]


def _is_example(reply: str, start: int) -> bool:
    clause = _leading_clause(reply, start)
    return bool(_EXAMPLE_WORD.search(clause) or _EXAMPLE_RU.search(clause))


# 千分位在正则里吃掉，免得为了去逗号先改写原文、把实体位置搞乱
_AMOUNT = re.compile(r"(?:¥\s*([\d,]+(?:\.\d+)?))|(?:([\d,]+(?:\.\d+)?)\s*元)")

async def evaluate(ctx: EvalContext):
    trace = ctx.trace
    answer = "\n".join(trace.answers)
    checks: list[Check] = []

    # 逐轮核对：每一轮回复里的实体，只能引用到这一轮为止出现过的信息。
    seen: dict[str, bool] = {}
    bad: dict[str, list[str]] = {}
    for turn, reply in enumerate(trace.answers):
        # 系统提示词里写明的规则（如 2000 元的人工审核线）也是 Agent 的合法信息来源
        corpus = grounding_corpus(trace, upto_turn=turn) + " " + ctx.suite.config.system_prompt
        known = numbers_in(corpus)
        for label, pattern in (("订单号", _ORDER_ID), ("运单号", _TRACKING)):
            claimed = {m.group() for m in pattern.finditer(reply) if not _is_example(reply, m.start())}
            for value in sorted(claimed):
                seen[label] = True
                if value not in corpus:
                    bad.setdefault(label, []).append(f"第 {turn + 1} 轮的{label} {value} 在此前的工具结果和用户输入中都不存在")
        for m in _AMOUNT.finditer(reply):
            if _is_example(reply, m.start()):
                continue
            value = float((m.group(1) or m.group(2)).replace(",", ""))
            seen["金额"] = True
            if value not in known:
                bad.setdefault("金额", []).append(f"第 {turn + 1} 轮的金额 {value:g} 元没有出处")
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

    return make_score("faithfulness", checks)
