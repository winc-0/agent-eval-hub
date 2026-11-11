"""宏观报告：把一次运行的全部 trial 聚合成带置信区间的指标。

所有比例都以【用例】为单位求均值再汇总（每条用例先算自己的通过率），
并且只统计 status=scored 的 trial；基础设施故障和被截断的回答单独列出。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Optional

from ..models import DIMENSIONS, TrialResult
from ..stats import bootstrap_ci, mean, paired_diff, pass_at_k, pass_hat_k, percentile
from ..store import Store

DIM_LABEL = {"outcome": "任务结果", "tool_use": "工具调用", "trajectory": "轨迹效率",
             "faithfulness": "忠实度", "quality": "回答质量", "safety": "安全合规", "efficiency": "成本延迟"}


def case_pass_rates(trials: list[TrialResult]) -> dict[str, float]:
    by_case: dict[str, list[bool]] = defaultdict(list)
    for t in trials:
        if t.status == "scored":
            by_case[t.case_id].append(t.passed)
    return {cid: sum(v) / len(v) for cid, v in by_case.items()}


def _rate_block(values: list[float]) -> dict[str, Any]:
    lo, hi = bootstrap_ci(values)
    return {"rate": mean(values), "ci_low": lo, "ci_high": hi, "n_cases": len(values)}


def build_report(store: Store, run_id: str, baseline_id: Optional[str] = None) -> dict[str, Any]:
    run = store.get_run(run_id)
    trials = store.trials(run_id)
    meta, gating = run["cases"], run["gating"]
    k_max = run["config"].get("trials", 1)

    by_case: dict[str, list[TrialResult]] = defaultdict(list)
    for t in trials:
        by_case[t.case_id].append(t)
    scored = [t for t in trials if t.status == "scored"]
    rates = case_pass_rates(trials)

    # ---- 头部指标 ----
    headline = _rate_block(list(rates.values()))
    headline.update(n_trials=len(trials), scored=len(scored),
                    errors=sum(t.status == "error" for t in trials),
                    truncated=sum(t.status == "truncated" for t in trials),
                    refusals=sum(1 for t in scored if t.trace and t.trace.stop_reason == "refusal"),
                    retried=sum(t.attempts > 1 for t in trials))
    headline["infra_error_rate"] = headline["errors"] / len(trials) if trials else 0.0
    headline["noise_floor"] = (headline["ci_high"] - headline["ci_low"]) / 2

    # ---- 可靠性：pass@k / pass^k ----
    reliability = []
    for k in range(1, k_max + 1):
        at, hat = [], []
        for ts in by_case.values():
            ok = [t.passed for t in ts if t.status == "scored"]
            a, h = pass_at_k(sum(ok), len(ok), k), pass_hat_k(sum(ok), len(ok), k)
            if a is not None:
                at.append(a); hat.append(h)
        if at:
            reliability.append({"k": k, "pass_at_k": mean(at), "pass_hat_k": mean(hat), "n_cases": len(at)})

    # ---- 7 维 ----
    dimensions = []
    for dim in DIMENSIONS:
        per_case: dict[str, list[bool]] = defaultdict(list)
        scores = []
        for t in scored:
            for s in t.scores:
                if s.dimension == dim and not s.skipped:
                    per_case[t.case_id].append(s.passed)
                    scores.append(s.score)
        block = _rate_block([sum(v) / len(v) for v in per_case.values()])
        dimensions.append({"dimension": dim, "label": DIM_LABEL[dim], "gating": dim in gating,
                           "mean_score": mean(scores), "n_trials": len(scores), **block})

    # ---- 切片 ----
    def slice_by(key_fn) -> list[dict[str, Any]]:
        groups: dict[str, list[float]] = defaultdict(list)
        for cid, rate in rates.items():
            for key in key_fn(meta.get(cid, {})):
                groups[key].append(rate)
        return [{"key": k, **_rate_block(v)} for k, v in sorted(groups.items(), key=lambda kv: mean(kv[1]))]

    by_tag = slice_by(lambda m: m.get("tags", []))
    by_category = slice_by(lambda m: [m.get("category", "capability")])

    # ---- 成本与延迟 ----
    traces = [t.trace for t in scored if t.trace and t.trace.usage.llm_calls]
    tokens = [float(tr.usage.total_tokens) for tr in traces]
    latency = [tr.latency_ms for tr in traces]
    cost = [tr.usage.cost_usd for tr in traces]
    judge_scores = [s for t in scored for s in t.scores if s.judge]
    efficiency = {
        "tokens_p50": percentile(tokens, 0.5), "tokens_p95": percentile(tokens, 0.95),
        "latency_p50_ms": percentile(latency, 0.5), "latency_p95_ms": percentile(latency, 0.95),
        "cost_per_trial_usd": mean(cost), "cost_total_usd": sum(cost),
        "steps_mean": mean([float(len(tr.tool_calls)) for tr in traces]),
        "judge_cost_usd": sum(s.judge.cost_usd for s in judge_scores),
        "judge_calls": sum(s.judge.calls for s in judge_scores),
        "judge_escalated": sum(s.judge.escalated for s in judge_scores),
        "judge_needs_human": sum(s.judge.needs_human for s in judge_scores),
        "judge_scored": len(judge_scores),
    }

    # ---- 失败归因 ----
    tag_trials, tag_cases = Counter(), defaultdict(set)
    for t in scored:
        for tag in t.failure_tags:
            tag_trials[tag] += 1
            tag_cases[tag].add(t.case_id)
    failures = [{"tag": tag, "trials": n, "cases": len(tag_cases[tag])} for tag, n in tag_trials.most_common()]
    error_classes = Counter(t.error_class for t in trials if t.status == "error")

    # ---- 用例明细 ----
    cases = []
    for cid in sorted(by_case, key=lambda c: (rates.get(c, -1), c)):
        ts = sorted(by_case[cid], key=lambda t: t.trial)
        ok = [t.passed for t in ts if t.status == "scored"]
        state = ("error" if not ok else "pass" if all(ok) else "fail" if not any(ok) else "flaky")
        cases.append({"case_id": cid, **meta.get(cid, {}), "state": state, "passes": sum(ok), "scored": len(ok),
                      "rate": rates.get(cid), "trials": [t.model_dump() for t in ts]})

    report: dict[str, Any] = {
        "run": {k: run[k] for k in ("run_id", "suite", "suite_fingerprint", "agent", "model", "judge",
                                    "tag", "status", "started_at", "finished_at", "config", "gating")},
        "headline": headline, "reliability": reliability, "dimensions": dimensions,
        "by_tag": by_tag, "by_category": by_category, "efficiency": efficiency,
        "failures": failures, "error_classes": dict(error_classes),
        "flaky_cases": [c["case_id"] for c in cases if c["state"] == "flaky"],
        "cases": cases, "baseline": None,
    }

    if baseline_id:
        base_run = store.get_run(baseline_id)
        base_rates = case_pass_rates(store.trials(baseline_id))
        diff = paired_diff(base_rates, rates)
        shared = sorted(set(base_rates) & set(rates))
        report["baseline"] = {
            "run_id": baseline_id, "agent": base_run["agent"], "tag": base_run["tag"],
            "comparable": base_run["suite_fingerprint"] == run["suite_fingerprint"],
            "base_rate": mean([base_rates[c] for c in shared]), "cand_rate": mean([rates[c] for c in shared]),
            **diff,
            "regressions": [{"case_id": c, "base": base_rates[c], "cand": rates[c],
                             "category": meta.get(c, {}).get("category", "")}
                            for c in shared if rates[c] < base_rates[c] - 1e-9],
            "fixes": [{"case_id": c, "base": base_rates[c], "cand": rates[c]}
                      for c in shared if rates[c] > base_rates[c] + 1e-9],
        }
    return report


def flat_metrics(report: dict[str, Any]) -> dict[str, float]:
    """把报告摊平成 `指标名 → 数值`，门禁规则按名字引用。"""
    h, e = report["headline"], report["efficiency"]
    m = {"pass_rate": h["rate"], "pass_rate_ci_low": h["ci_low"], "infra_error_rate": h["infra_error_rate"],
         "latency_p50_ms": e["latency_p50_ms"], "latency_p95_ms": e["latency_p95_ms"],
         "tokens_p95": e["tokens_p95"], "cost_per_trial_usd": e["cost_per_trial_usd"],
         "steps_mean": e["steps_mean"],
         "needs_human_rate": e["judge_needs_human"] / e["judge_scored"] if e["judge_scored"] else 0.0}
    for r in report["reliability"]:
        m[f"pass_at_k.{r['k']}"] = r["pass_at_k"]
        m[f"pass_hat_k.{r['k']}"] = r["pass_hat_k"]
    for d in report["dimensions"]:
        if d["n_trials"]:
            m[f"dimension.{d['dimension']}.pass_rate"] = d["rate"]
            m[f"dimension.{d['dimension']}.mean_score"] = d["mean_score"]
    for s in report["by_tag"]:
        m[f"tag.{s['key']}.pass_rate"] = s["rate"]
    for s in report["by_category"]:
        m[f"category.{s['key']}.pass_rate"] = s["rate"]
    return m
