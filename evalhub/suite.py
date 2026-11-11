"""评测集（suite）：配置 + 用例的加载，以及对用例集本身的体检（lint）。

目录约定：
  suites/<name>/suite.yaml        领域、系统提示词、门禁维度、默认策略与 rubric
  suites/<name>/cases/**/*.yaml   用例，一个文件可以放一条或一个列表
  suites/<name>/regression.txt    被钉为回归用例的 id（零容忍，不允许从通过变失败）
  suites/<name>/drafts/           线上 Badcase 自动生成的用例草稿，人工补全期望后再移入 cases/
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Callable

import yaml
from pydantic import BaseModel, Field

from .models import Budget, Case, Check, RubricItem, Trace
from .policies import get_policy
from .sandbox import DOMAINS, Sandbox, make_sandbox


class ActionClaim(BaseModel):
    """回复里出现 pattern（声称做了某事）时，必须有一次成功的 tool 调用与之对应。"""
    pattern: str
    tool: str


class SuiteConfig(BaseModel):
    name: str
    description: str = ""
    domain: str
    system_prompt: str
    # 哪些维度决定一次 trial 算不算通过；其余维度只做诊断，不影响通过率。
    gating_dimensions: list[str] = Field(default_factory=lambda: ["outcome", "tool_use", "faithfulness", "safety"])
    default_policies: list[str] = Field(default_factory=list)
    default_rubric: list[RubricItem] = Field(default_factory=list)
    action_claims: list[ActionClaim] = Field(default_factory=list)
    budget: Budget = Field(default_factory=lambda: Budget(
        max_steps=8, max_tokens=8000, max_latency_ms=30000, max_cost_usd=0.05))


class Suite:
    def __init__(self, config: SuiteConfig, cases: list[Case], path: Path):
        self.config = config
        self.cases = cases
        self.path = path
        self._by_id = {c.id: c for c in cases}

    @classmethod
    def load(cls, path: str | Path) -> "Suite":
        root = Path(path)
        config = SuiteConfig(**yaml.safe_load((root / "suite.yaml").read_text(encoding="utf-8")))
        if config.domain not in DOMAINS:
            raise ValueError(f"未知领域 {config.domain}")
        pinned = set()
        pin_file = root / "regression.txt"
        if pin_file.exists():
            pinned = {ln.strip() for ln in pin_file.read_text(encoding="utf-8").splitlines()
                      if ln.strip() and not ln.startswith("#")}
        cases: list[Case] = []
        for file in sorted((root / "cases").rglob("*.yaml")):
            data = yaml.safe_load(file.read_text(encoding="utf-8"))
            for raw in data if isinstance(data, list) else [data]:
                case = Case(**raw)
                if case.id in pinned and case.category == "capability":
                    case.category = "regression"
                cases.append(case)
                for v in case.variants:   # 展开换种问法的变体，期望不变
                    clone = case.model_copy(deep=True)
                    clone.id, clone.variants = f"{case.id}__{v.suffix}", []
                    clone.user.turns = list(v.turns)
                    clone.tags = case.tags + ["variant"]
                    clone.category = "capability" if case.category == "regression" else case.category
                    cases.append(clone)
        ids = [c.id for c in cases]
        dupes = [i for i, n in Counter(ids).items() if n > 1]
        if dupes:
            raise ValueError(f"用例 id 重复：{dupes}")
        return cls(config, cases, root)

    def case(self, case_id: str) -> Case:
        return self._by_id[case_id]

    def make_sandbox(self, case: Case) -> Sandbox:
        return make_sandbox(self.config.domain, case.env)

    def budget_for(self, case: Case) -> Budget:
        return self.config.budget.merged(case.expect.budget)

    def rubric_for(self, case: Case) -> list[RubricItem]:
        return self.config.default_rubric + case.expect.rubric

    def policies_for(self, case: Case) -> list[tuple[str, Callable[[Case, Trace], list[Check]]]]:
        names = list(dict.fromkeys(self.config.default_policies + case.expect.policies))
        return [(n, get_policy(n, self.config.domain)) for n in names]

    def fingerprint(self) -> str:
        """用例集 + 配置的内容指纹。指纹不同的两次运行，分数不可直接比较。"""
        payload = json.dumps({"config": self.config.model_dump(),
                              "cases": [c.model_dump() for c in self.cases]},
                             sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    def case_index(self) -> dict[str, dict]:
        return {c.id: {"title": c.title, "tags": c.tags, "category": c.category, "source": c.source}
                for c in self.cases}


def lint(suite: Suite) -> list[tuple[str, str]]:
    """用例集体检：全量、程序化、便宜。返回 [(级别, 说明)]，级别为 error / warn / info。"""
    out: list[tuple[str, str]] = []
    cases = suite.cases
    tools = set(suite.make_sandbox(cases[0]).tools) if cases else set()

    seen: dict[str, str] = {}
    for c in cases:
        # 输入相同但环境不同（比如注入了故障）是有意为之，不算重复
        key = re.sub(r"\s+", "", "|".join(c.user.turns)) + c.env.model_dump_json()
        if key in seen:
            out.append(("warn", f"{c.id} 与 {seen[key]} 的用户输入和环境完全相同"))
        seen.setdefault(key, c.id)

        e = c.expect
        has_gate = bool(e.final_state or e.tools.required or e.tools.forbidden
                        or e.tools.max_calls is not None or e.answer.must_include or e.answer.must_not_include)
        if not has_gate:
            out.append(("error", f"{c.id} 没有任何可判定的期望（终态/工具/回复），无法评分"))
        if not c.reference:
            out.append(("warn", f"{c.id} 没有参考解，harness 自检（oracle）无法覆盖它"))
        elif len(c.reference) != len(c.user.turns):
            out.append(("error", f"{c.id} 参考解的轮数与用户轮数不一致"))
        for name in [t.name for t in e.tools.required] + e.tools.forbidden + e.tools.order:
            if name not in tools:
                out.append(("error", f"{c.id} 引用了不存在的工具 {name}"))
        user_text = " ".join(c.user.turns)
        for item in e.answer.must_include:
            leaked = [a for a in item.split("|") if len(a) >= 4 and a in user_text]
            if leaked and not re.fullmatch(r"[A-Z]+\d+", leaked[0]):
                out.append(("warn", f"{c.id} 期望回复包含的“{leaked[0]}”已出现在用户输入里，照抄即可得分"))

    if cases:
        act = sum(1 for c in cases if any(a.exists or (a.equals is not None and not a.unchanged)
                                          for a in c.expect.final_state))
        hold = sum(1 for c in cases if c.expect.tools.forbidden or c.expect.tools.max_calls == 0
                   or any(a.unchanged for a in c.expect.final_state))
        out.append(("info", f"共 {len(cases)} 条用例；应当动手的 {act} 条，应当克制的 {hold} 条"))
        if hold == 0:
            out.append(("warn", "没有“不该做”的用例：一个见单就退款的 Agent 也能拿高分"))
        cat = Counter(c.category for c in cases)
        out.append(("info", "类别分布：" + "，".join(f"{k} {v}" for k, v in sorted(cat.items()))))
        tags = Counter(t for c in cases for t in c.tags)
        out.append(("info", "标签分布：" + "，".join(f"{k} {v}" for k, v in tags.most_common())))
        src = Counter(c.source for c in cases)
        out.append(("info", "来源：" + "，".join(f"{k} {v}" for k, v in sorted(src.items()))))
    return out
