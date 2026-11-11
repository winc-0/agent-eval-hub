"""报告输出：HTML（给人看）、JSON（给程序）、JUnit XML（给 CI 界面）、Markdown（贴到 PR）。"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional

from ..store import Store
from .aggregate import DIM_LABEL, build_report, case_pass_rates, flat_metrics
from .html import currency_symbol, render_html


def render_junit(report: dict[str, Any]) -> str:
    cases = report["cases"]
    suite = ET.Element("testsuite", name=f"{report['run']['suite']}:{report['run']['agent']}",
                       tests=str(len(cases)),
                       failures=str(sum(c["state"] in ("fail", "flaky") for c in cases)),
                       errors=str(sum(c["state"] == "error" for c in cases)))
    for c in cases:
        tc = ET.SubElement(suite, "testcase", classname=report["run"]["suite"], name=c["case_id"])
        if c["state"] == "error":
            first = c["trials"][0]
            ET.SubElement(tc, "error", message=first.get("error_class") or "error").text = first.get("error_message", "")
        elif c["state"] != "pass":
            tags = sorted({t for tr in c["trials"] for t in tr["failure_tags"]})
            ET.SubElement(tc, "failure", message=f"{c['passes']}/{c['scored']} 通过：{', '.join(tags)}").text = c.get("title", "")
    return ET.tostring(suite, encoding="unicode")


def render_markdown(report: dict[str, Any], gate: Optional[dict[str, Any]] = None) -> str:
    h, e, run = report["headline"], report["efficiency"], report["run"]
    lines = [f"## Agent 评测 · {run['suite']} · `{run['agent']}`", ""]
    if gate is not None:
        lines += [f"**质量门禁：{'✅ 通过' if gate['passed'] else '❌ 未通过'}**", ""]
    lines += [f"- 通过率 **{h['rate'] * 100:.1f}%**（95% 区间 {h['ci_low'] * 100:.1f}% ~ {h['ci_high'] * 100:.1f}%）",
              f"- 已评分 {h['scored']}/{h['n_trials']}，基础设施故障 {h['errors']}"]
    if report["reliability"]:
        r = report["reliability"][-1]
        lines.append(f"- pass^{r['k']} **{r['pass_hat_k'] * 100:.1f}%**，pass@{r['k']} {r['pass_at_k'] * 100:.1f}%")
    lines.append(f"- P95 延迟 {e['latency_p95_ms'] / 1000:.2f}s，单次成本 {currency_symbol(report)}{e['cost_per_trial_usd']:.4f}")
    b = report.get("baseline")
    if b:
        sig = "显著" if b["significant"] else "在噪声范围内"
        lines.append(f"- 对比基线 `{b['run_id']}`：{b['delta'] * 100:+.1f} 个百分点"
                     f"（{b['ci_low'] * 100:+.1f} ~ {b['ci_high'] * 100:+.1f}，{sig}）")
    lines += ["", "| 维度 | 角色 | 通过率 |", "|---|---|---|"]
    for d in report["dimensions"]:
        if d["n_trials"]:
            lines.append(f"| {d['label']} | {'门禁' if d['gating'] else '诊断'} | {d['rate'] * 100:.1f}% |")
    if gate is not None:
        lines += ["", "| 门禁规则 | 实际 | 要求 | 结果 |", "|---|---|---|---|"]
        for c in gate["checks"]:
            lines.append(f"| {c['name']} | {c['actual']} | {c['threshold']} | {'✅' if c['passed'] else '❌'} |")
    if report["failures"]:
        lines += ["", "失败归因 Top 5：" + "，".join(f"`{f['tag']}`×{f['trials']}" for f in report["failures"][:5])]
    return "\n".join(lines) + "\n"


def write_reports(store: Store, run_id: str, out_dir: str | Path, baseline_id: Optional[str] = None,
                  gate: Optional[dict[str, Any]] = None) -> dict[str, Path]:
    report = build_report(store, run_id, baseline_id)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"html": out / f"{run_id}.html", "json": out / f"{run_id}.json",
             "junit": out / f"{run_id}.junit.xml", "md": out / f"{run_id}.md"}
    paths["html"].write_text(render_html(report), encoding="utf-8")
    paths["json"].write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    paths["junit"].write_text(render_junit(report), encoding="utf-8")
    paths["md"].write_text(render_markdown(report, gate), encoding="utf-8")
    return paths


__all__ = ["DIM_LABEL", "build_report", "case_pass_rates", "flat_metrics", "render_html",
           "render_junit", "render_markdown", "write_reports"]
