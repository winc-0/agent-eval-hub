"""Harness 自检：在花钱跑真实模型之前，先确认评测本身没坏。

两次免费的运行就能抓出大部分接线错误：
  oracle（回放参考解）应当全过   —— 过不了，说明用例写错了，或评估器太严
  null（什么都不做）  应当全挂   —— 挂不了，说明评估器太松，什么都不做也能得分
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from .agent import NullAgent, OracleAgent
from .scheduler import RunConfig, run_suite
from .store import Store
from .suite import Suite


async def selftest(suite: Suite, store: Store) -> dict[str, Any]:
    cfg = RunConfig(trials=1, concurrency=8, backoff_base_s=0.0)
    with_ref = {c.id for c in suite.cases if c.reference}
    oracle_run = await run_suite(suite, OracleAgent(), store, replace(cfg, tag="selftest-oracle"), only=with_ref)
    null_run = await run_suite(suite, NullAgent(), store, replace(cfg, tag="selftest-null"))

    oracle_failures, oracle_warnings = [], []
    for t in store.trials(oracle_run):
        if t.status != "scored":
            oracle_failures.append({"case_id": t.case_id, "why": f"{t.error_class}: {t.error_message}"})
            continue
        failed = [f"[{s.dimension}] {c.detail}" for s in t.scores for c in s.checks if not c.passed]
        if not t.passed:
            oracle_failures.append({"case_id": t.case_id, "why": "；".join(failed)})
        elif failed:   # 门禁维度过了，但诊断维度有瑕疵（比如参考解里有重复调用）
            oracle_warnings.append({"case_id": t.case_id, "why": "；".join(failed)})
    null_passes = [t.case_id for t in store.trials(null_run) if t.passed]

    return {"ok": not oracle_failures and not null_passes,
            "oracle_run": oracle_run, "null_run": null_run,
            "oracle_cases": len(with_ref), "no_reference": len(suite.cases) - len(with_ref),
            "oracle_failures": oracle_failures, "oracle_warnings": oracle_warnings,
            "null_passes": null_passes}
