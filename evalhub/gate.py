"""质量门禁：把评测结果变成 CI 里的“能不能合并”。

两类规则：
  red_lines   绝对红线。任意一条不满足就失败（如安全维度必须 100%、pass^3 不低于 75%）。
  regression  相对基线的回归。只在“下降超过容忍值，且统计上显著”时才失败——
              否则 26 条用例 × 3 次的噪声就足以让 CI 随机变红，没人会再信它。
              但安全类和已钉住的回归用例零容忍：从全过变成不全过，直接失败。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

from .report import flat_metrics


def load_gate(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _fmt(v: Optional[float], metric: str = "rate") -> str:
    if v is None:
        return "—"
    is_ratio = any(k in metric for k in ("rate", "pass_at_k", "pass_hat_k", "mean_score"))
    if is_ratio:
        return f"{v * 100:.1f}%"
    return f"{v:.0f}" if abs(v) >= 1000 else f"{v:.4g}"


def evaluate_gate(report: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    metrics = flat_metrics(report)
    checks: list[dict[str, Any]] = []

    for rule in gate.get("red_lines", []):
        name, actual = rule["metric"], metrics.get(rule["metric"])
        if actual is None:   # 指标不存在（比如没开裁判却要求 quality）也算不通过，避免门禁被悄悄绕过
            checks.append({"name": name, "passed": False, "actual": "缺失", "threshold": "指标必须存在",
                           "why": rule.get("why", "")})
            continue
        ok, parts = True, []
        if "min" in rule:
            ok &= actual >= rule["min"] - 1e-9
            parts.append(f"≥ {_fmt(rule['min'], name)}")
        if "max" in rule:
            ok &= actual <= rule["max"] + 1e-9
            parts.append(f"≤ {_fmt(rule['max'], name)}")
        checks.append({"name": name, "passed": bool(ok), "actual": _fmt(actual, name),
                       "threshold": " 且 ".join(parts), "why": rule.get("why", "")})

    reg, base = gate.get("regression"), report.get("baseline")
    if reg and base:
        max_drop = reg.get("max_drop", 0.0)
        dropped = base["delta"] < -max_drop
        significant = base["ci_high"] < 0
        failed = dropped and (significant or not reg.get("require_significant", True))
        checks.append({"name": "regression.pass_rate", "passed": not failed,
                       "actual": f"{base['delta'] * 100:+.1f}pp（{base['ci_low'] * 100:+.1f} ~ {base['ci_high'] * 100:+.1f}）",
                       "threshold": f"下降不超过 {max_drop * 100:.1f}pp" +
                                    ("（且需显著才判失败）" if reg.get("require_significant", True) else ""),
                       "why": "相对基线不得显著变差"})
        strict = set(reg.get("zero_tolerance_categories", []))
        broken = [r["case_id"] for r in base["regressions"]
                  if r.get("category") in strict and r["base"] >= 1.0 - 1e-9]
        if strict:
            checks.append({"name": "regression.zero_tolerance", "passed": not broken,
                           "actual": "、".join(broken) if broken else "无",
                           "threshold": f"{'/'.join(sorted(strict))} 类用例不得由全过变为不全过",
                           "why": "安全与已修复问题不允许回退"})
        if not base.get("comparable", True):
            checks.append({"name": "regression.comparable", "passed": False, "actual": "评测集指纹不同",
                           "threshold": "基线与本次必须是同一版评测集", "why": "用例变了，分数不可比"})
    elif reg and reg.get("require_baseline"):
        checks.append({"name": "regression.baseline", "passed": False, "actual": "未提供",
                       "threshold": "必须指定基线运行", "why": ""})

    return {"passed": all(c["passed"] for c in checks), "checks": checks}
