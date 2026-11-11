"""评估器的公共部分：上下文、打分汇总、文本归一化。

每个维度都拆成若干【原子检查项】（Check），一项只判一件事。
维度得分 = 通过项占比；维度通过 = 全部通过。这样失败能直接定位到具体哪一项。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from ..judge import CascadeJudge
from ..models import Budget, Case, Check, JudgeMeta, Score, Trace
from ..suite import Suite


@dataclass
class EvalContext:
    case: Case
    trace: Trace
    suite: Suite
    judge: Optional[CascadeJudge] = None

    @property
    def budget(self) -> Budget:
        return self.suite.budget_for(self.case)


def make_score(dimension: str, checks: list[Check], grader: str = "code",
               judge: Optional[JudgeMeta] = None) -> Score:
    if not checks:   # 这条用例不涉及该维度：标记跳过，不计入该维度的统计
        return Score(dimension=dimension, score=1.0, passed=True, grader=grader, skipped=True)
    ok = sum(c.passed for c in checks)
    failed = [c for c in checks if not c.passed]
    return Score(dimension=dimension, score=ok / len(checks), passed=not failed, grader=grader,
                 checks=checks, reason=failed[0].detail if failed else "", judge=judge)


def normalize_text(s: str) -> str:
    """去掉空白和千分位逗号、转小写。判断“回复里有没有 299”不该因为排版不同而失败。"""
    return re.sub(r"[\s,]", "", s).lower()


def includes(haystack_norm: str, needle: str) -> bool:
    """needle 支持 a|b 多选一；纯数字按数值匹配（299 能匹配 299.00，但不匹配 2990 或 299.5）。"""
    for alt in needle.split("|"):
        a = normalize_text(alt)
        if not a:
            continue
        if re.fullmatch(r"\d+(\.\d+)?", a):
            num = a.rstrip("0").rstrip(".") if "." in a else a
            if re.search(rf"(?<![\d.]){re.escape(num)}(?:\.0+)?(?!\.?\d)", haystack_norm):
                return True
        elif a in haystack_norm:
            return True
    return False


def numbers_in(text: str) -> set[float]:
    return {float(x) for x in re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))}


def lookup(state: Any, path: str) -> tuple[bool, Any]:
    """按点分路径取值，返回 (是否存在, 值)。列表用数字下标。"""
    cur = state
    for seg in path.split("."):
        if isinstance(cur, dict) and seg in cur:
            cur = cur[seg]
        elif isinstance(cur, list) and seg.isdigit() and int(seg) < len(cur):
            cur = cur[int(seg)]
        else:
            return False, None
    return True, cur


def same(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(float(a) - float(b)) < 1e-6
    return a == b


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def grounding_corpus(trace: Trace, before_step: Optional[int] = None, upto_turn: Optional[int] = None) -> str:
    """Agent 在某个时刻能合法引用的全部信息：到那时为止用户说过的话 + 工具返回过的内容。

    必须按时间截断：第 1 轮回复里冒出的订单号，不能因为用户在第 2 轮才提到它就算“有出处”。
    """
    users = trace.user_messages if upto_turn is None else trace.user_messages[:upto_turn + 1]
    results = [c.result for c in trace.tool_calls
               if (before_step is None or c.step < before_step) and (upto_turn is None or c.turn <= upto_turn)]
    return " ".join(users) + " " + dumps(results)
