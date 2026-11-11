"""端到端测试：调度引擎的容错、整条评测流水线、门禁、Badcase 闭环、HTTP 服务。全部离线。"""
from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import time
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from evalhub import badcase as bc
from evalhub.agent import NullAgent, OracleAgent, make_agent
from evalhub.export import trial_to_otel
from evalhub.gate import evaluate_gate, load_gate
from evalhub.judge import calibrate, make_judge
from evalhub.llm import InfraError
from evalhub.report import build_report, flat_metrics, render_html, render_junit, write_reports
from evalhub.scheduler import RunConfig, run_suite
from evalhub.selftest import selftest
from evalhub.store import Store
from evalhub.suite import Suite

SUITE_DIR = Path(__file__).resolve().parents[1] / "suites" / "ecommerce_cs"
FAST = dict(concurrency=8, backoff_base_s=0.0)


def run(coro):
    return asyncio.run(coro)


class TempDirCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.tmp = Path(self._tmp.name)
        self.store = Store(str(self.tmp / "test.db"))
        self.suite = Suite.load(SUITE_DIR)

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()


class _FailingAgent(OracleAgent):
    """前 fail_times 次调用抛基础设施故障，之后表现为 oracle。"""

    def __init__(self, fail_times: int, error: Exception):
        self.fail_times, self.error, self.calls = fail_times, error, {}

    async def run(self, case, sandbox, system_prompt, seed):
        n = self.calls[case.id] = self.calls.get(case.id, 0) + 1
        if n <= self.fail_times:
            raise self.error
        return await super().run(case, sandbox, system_prompt, seed)


class _SlowAgent(OracleAgent):
    async def run(self, case, sandbox, system_prompt, seed):
        await asyncio.sleep(5)
        return await super().run(case, sandbox, system_prompt, seed)


class SchedulerTests(TempDirCase):
    ONLY = {"refund_eligible_basic", "logistics_in_transit"}

    def go(self, agent, **cfg):
        run_id = run(run_suite(self.suite, agent, self.store, RunConfig(trials=1, **{**FAST, **cfg}),
                               only=self.ONLY))
        return run_id, self.store.trials(run_id)

    def test_transient_infra_error_is_retried_and_then_scored(self):
        _, trials = self.go(_FailingAgent(1, InfraError("rate_limited", "429")))
        self.assertEqual([(t.status, t.attempts, t.passed) for t in trials], [("scored", 2, True)] * 2)

    def test_exhausted_retries_become_errors_excluded_from_pass_rate(self):
        run_id, trials = self.go(_FailingAgent(99, InfraError("provider_5xx", "503")), max_attempts=3)
        self.assertEqual([(t.status, t.attempts, t.error_class) for t in trials], [("error", 3, "provider_5xx")] * 2)
        report = build_report(self.store, run_id)
        self.assertEqual(report["headline"]["scored"], 0)
        self.assertEqual(report["headline"]["infra_error_rate"], 1.0)
        self.assertEqual(report["error_classes"], {"provider_5xx": 2})
        self.assertEqual(report["headline"]["n_cases"], 0, "故障的 trial 不进入通过率的分母")

    def test_non_retryable_error_fails_fast(self):
        _, trials = self.go(_FailingAgent(99, InfraError("bad_request", "400", retryable=False)))
        self.assertEqual({t.attempts for t in trials}, {1})

    def test_bug_in_harness_is_surfaced_not_retried(self):
        _, trials = self.go(_FailingAgent(99, ValueError("评估器写错了")))
        self.assertEqual({(t.status, t.error_class, t.attempts) for t in trials}, {("error", "harness_error", 1)})
        self.assertIn("评估器写错了", trials[0].error_message)

    def test_hard_timeout_reclaims_the_slot(self):
        started = time.monotonic()
        _, trials = self.go(_SlowAgent(), trial_timeout_s=0.2, max_attempts=1)
        self.assertLess(time.monotonic() - started, 3)
        self.assertEqual({t.error_class for t in trials}, {"timeout"})

    def test_resume_reruns_only_the_failed_trials(self):
        run_id, _ = self.go(_FailingAgent(99, InfraError("network", "断网")), max_attempts=1)
        self.assertEqual(self.store.completed(run_id), set())
        counting = _FailingAgent(0, InfraError("x", "x"))
        run(run_suite(self.suite, counting, self.store, RunConfig(trials=1, **FAST), only=self.ONLY,
                      resume_run_id=run_id))
        self.assertEqual(len(self.store.completed(run_id)), 2)
        run(run_suite(self.suite, counting, self.store, RunConfig(trials=1, **FAST), only=self.ONLY,
                      resume_run_id=run_id))
        self.assertEqual(sum(counting.calls.values()), 2, "已完成的不应重跑")

    def test_same_seed_gives_same_results(self):
        outcomes = []
        for _ in range(2):   # v1 有随机性，但由种子决定
            run_id = run(run_suite(self.suite, make_agent("mock:cs-v1"), self.store, RunConfig(trials=3, **FAST)))
            outcomes.append([(t.case_id, t.trial, t.passed) for t in self.store.trials(run_id)])
        self.assertEqual(outcomes[0], outcomes[1])


class PipelineTests(TempDirCase):
    def evaluate(self, agent_spec: str, tag: str = "") -> str:
        run_id = run(run_suite(self.suite, make_agent(agent_spec), self.store,
                               RunConfig(trials=3, tag=tag, **FAST), make_judge("mock")))
        bc.collect_badcases(self.store, run_id)
        return run_id

    def test_harness_selftest(self):
        res = run(selftest(self.suite, self.store))
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["oracle_failures"], [])
        self.assertEqual(res["null_passes"], [])

    def test_null_and_oracle_bracket_the_scale(self):
        cfg = RunConfig(trials=1, **FAST)
        null = build_report(self.store, run(run_suite(self.suite, NullAgent(), self.store, cfg)))
        oracle = build_report(self.store, run(run_suite(self.suite, OracleAgent(), self.store, cfg)))
        self.assertEqual(null["headline"]["rate"], 0.0)
        self.assertEqual(oracle["headline"]["rate"], 1.0)

    def test_v1_defects_are_all_detected_and_v2_is_better(self):
        v1, v2 = self.evaluate("mock:cs-v1", "baseline"), self.evaluate("mock:cs-v2", "candidate")
        r1, r2 = build_report(self.store, v1), build_report(self.store, v2, baseline_id=v1)
        tags = {f["tag"] for f in r1["failures"]}
        for expected in ("safety.refund_without_check", "safety.prompt_injection", "safety.pii_leak",
                         "safety.high_value_refund", "faithfulness.fabricated_entity",
                         "faithfulness.claimed_action_not_done", "trajectory.redundant_call",
                         "tool.unnecessary_call"):
            self.assertIn(expected, tags)
        self.assertLess(r1["headline"]["rate"], 0.5)
        self.assertGreater(r2["headline"]["rate"], 0.85)
        self.assertEqual(flat_metrics(r2)["dimension.safety.pass_rate"], 1.0)
        self.assertTrue(r2["baseline"]["significant"])
        self.assertGreater(r2["baseline"]["delta"], 0.5)
        self.assertEqual(r2["baseline"]["regressions"], [])
        # v1 有时不查物流直接猜：至少成一次的比例高于次次都成的比例
        last = r1["reliability"][-1]
        self.assertGreater(last["pass_at_k"], last["pass_hat_k"])
        self.assertTrue(r1["flaky_cases"])
        # v2 的短板也被如实测出来，没有被“修好了”的印象掩盖
        failed = {c["case_id"] for c in r2["cases"] if c["state"] != "pass"}
        self.assertEqual(failed, {"multi_intent_refund_and_invoice", "refund_eligible_basic__colloquial",
                                  "logistics_in_transit__colloquial"})

    def test_gate_blocks_v1_and_passes_v2(self):
        gate = load_gate(SUITE_DIR / "gate.yaml")
        v1, v2 = self.evaluate("mock:cs-v1"), self.evaluate("mock:cs-v2")
        blocked = evaluate_gate(build_report(self.store, v1), gate)
        self.assertFalse(blocked["passed"])
        self.assertIn("dimension.safety.pass_rate", [c["name"] for c in blocked["checks"] if not c["passed"]])
        self.assertTrue(evaluate_gate(build_report(self.store, v2, baseline_id=v1), gate)["passed"])

    def test_gate_catches_regression_against_baseline(self):
        gate = load_gate(SUITE_DIR / "gate.yaml")
        good, bad = self.evaluate("mock:cs-v2"), self.evaluate("mock:cs-v1")
        result = evaluate_gate(build_report(self.store, bad, baseline_id=good), gate)
        failed = {c["name"] for c in result["checks"] if not c["passed"]}
        self.assertIn("regression.pass_rate", failed)
        self.assertIn("regression.zero_tolerance", failed)

    def test_missing_metric_fails_the_gate_instead_of_being_skipped(self):
        run_id = run(run_suite(self.suite, make_agent("mock:cs-v2"), self.store, RunConfig(trials=1, **FAST)))
        gate = {"red_lines": [{"metric": "dimension.quality.pass_rate", "min": 0.9}]}   # 没开裁判
        self.assertFalse(evaluate_gate(build_report(self.store, run_id), gate)["passed"])

    def test_reports_are_written_in_all_formats(self):
        run_id = self.evaluate("mock:cs-v1")
        paths = write_reports(self.store, run_id, self.tmp / "reports")
        html = paths["html"].read_text(encoding="utf-8")
        self.assertIn("injection_in_logistics_note", html)
        self.assertIn("PWNED-7731", html, "微观轨迹里应能看到被注入后的实际回复")
        self.assertNotIn("<script", html)
        data = json.loads(paths["json"].read_text(encoding="utf-8"))
        self.assertEqual(data["run"]["run_id"], run_id)
        junit = ET.fromstring(paths["junit"].read_text(encoding="utf-8"))
        self.assertEqual(int(junit.get("tests")), len(self.suite.cases))
        self.assertGreater(int(junit.get("failures")), 0)
        self.assertIn("通过率", paths["md"].read_text(encoding="utf-8"))

    def test_cli_run_prints_summary_in_the_chosen_currency(self):
        import contextlib
        import io

        from evalhub.cli import main

        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = main(["--db", str(self.tmp / "cli.db"), "run", str(SUITE_DIR), "--agent", "mock:cs-v2",
                         "--trials", "1", "--only", "refund_eligible_basic", "--currency", "CNY",
                         "--out", str(self.tmp / "reports")])
        self.assertEqual(code, 0)
        self.assertIn("单次成本 ¥", out.getvalue())
        self.assertNotIn("$", out.getvalue())
        html = next((self.tmp / "reports").glob("*.html")).read_text(encoding="utf-8")
        self.assertIn("¥", html)
        self.assertNotIn("$0.", html)

    def test_html_escapes_untrusted_model_output(self):
        run_id = self.evaluate("mock:cs-v2")
        report = build_report(self.store, run_id)
        report["cases"][0]["trials"][0]["trace"]["timeline"][-1]["content"] = "<img src=x onerror=alert(1)>"
        self.assertNotIn("<img src=x", render_html(report))
        self.assertIn("&lt;img", render_html(report))
        ET.fromstring(render_junit(report))

    def test_failures_flow_into_the_badcase_pool_and_dedupe(self):
        self.evaluate("mock:cs-v1")
        once = {b["case_id"]: b["occurrences"] for b in self.store.badcases("ecommerce_cs")}
        self.evaluate("mock:cs-v1")
        twice = {b["case_id"]: b["occurrences"] for b in self.store.badcases("ecommerce_cs")}
        self.assertIn("injection_in_kb", once)
        self.assertEqual(twice["injection_in_kb"], once["injection_in_kb"] * 2, "同一问题累计次数，不重复建条目")

    def test_production_traces_become_drafts_that_need_human_expectations(self):
        root = self.tmp / "suite"
        shutil.copytree(SUITE_DIR, root, ignore=shutil.ignore_patterns("drafts"))
        suite = Suite.load(root)
        rows = bc.import_traces(suite, self.store, root / "production_traces.sample.jsonl")
        by_id = {r["trace_id"]: r for r in rows}
        self.assertEqual(by_id["p-20261108-001"]["tags"], [])
        self.assertIn("faithfulness.fabricated_entity", by_id["p-20261108-002"]["tags"])
        self.assertIn("safety.refund_without_check", by_id["p-20261109-003"]["tags"])
        self.assertEqual(by_id["p-20261109-004"]["tags"], ["user.thumbs_down"])
        self.assertEqual(len(list((root / "drafts").glob("*.yaml"))), 3)
        with self.assertRaises(ValueError):   # 期望还没人写，不能入库
            bc.promote_draft(suite, "prod_p_20261108_002")
        before = len(suite.cases)
        self.assertEqual(len(Suite.load(root).cases), before, "草稿不参与评测")

    def test_pinning_makes_a_case_zero_tolerance(self):
        root = self.tmp / "suite"
        shutil.copytree(SUITE_DIR, root, ignore=shutil.ignore_patterns("drafts"))
        suite = Suite.load(root)
        self.assertEqual(suite.case("kb_invoice").category, "capability")
        bc.pin_regression(suite, "kb_invoice")
        self.assertEqual(Suite.load(root).case("kb_invoice").category, "regression")
        with self.assertRaises(KeyError):
            bc.pin_regression(suite, "no_such_case")

    def test_calibration_reports_agreement_with_human_labels(self):
        rows = [json.loads(ln) for ln in (SUITE_DIR / "human_labels.jsonl").read_text(encoding="utf-8").splitlines()]
        rep = run(calibrate(make_judge("mock"), rows))
        self.assertEqual(rep["n"], len(rows))
        self.assertLess(rep["cascade"]["agreement"], 0.9, "关键词裁判本来就不该通过校准")
        self.assertTrue(any("negation" in str(d["id"]) for d in rep["disagreements"]))
        self.assertLess(rep["cascade"]["cost_usd"], rep["cascade"]["cost_if_all_strong"])

    def test_otel_export_shape(self):
        run_id = self.evaluate("mock:cs-v2")
        trial = next(t for t in self.store.trials(run_id) if t.case_id == "refund_eligible_basic")
        spans = list(trial_to_otel(trial))
        names = [s["name"] for s in spans]
        self.assertIn("execute_tool create_refund", names)
        root = spans[-1]
        self.assertTrue(root["name"].startswith("invoke_agent "))
        self.assertEqual({e["name"] for e in root["events"]}, {"gen_ai.evaluation.result"})
        self.assertTrue(all(s["parent_span_id"] == root["span_id"] for s in spans[:-1]))


try:
    from fastapi.testclient import TestClient

    from evalhub.server import create_app
except ImportError:   # 没装 server 依赖时跳过
    TestClient = None


@unittest.skipIf(TestClient is None, "未安装 fastapi / httpx")
class ServerTests(TempDirCase):
    def test_run_lifecycle_over_http(self):
        self.store.close()
        app = create_app(str(self.tmp / "test.db"), str(SUITE_DIR.parent))
        with TestClient(app) as client:
            self.assertEqual(client.post("/api/runs", json={"suite": "../etc"}).status_code, 400)
            self.assertEqual(client.post("/api/runs", json={"suite": "nope"}).status_code, 404)
            self.assertEqual(client.post("/api/runs", json={"suite": "ecommerce_cs", "trials": 99}).status_code, 400)
            resp = client.post("/api/runs", json={"suite": "ecommerce_cs", "agent": "mock:cs-v2", "trials": 1})
            self.assertEqual(resp.status_code, 200)
            run_id = resp.json()["run_id"]
            for _ in range(100):
                runs = {r["run_id"]: r for r in client.get("/api/runs").json()}
                if runs[run_id]["status"] == "finished":
                    break
                time.sleep(0.1)
            self.assertEqual(runs[run_id]["status"], "finished")
            summary = client.get(f"/api/runs/{run_id}").json()
            self.assertGreater(summary["headline"]["rate"], 0.85)
            self.assertNotIn("trace", summary["cases"][0]["trials"][0])
            self.assertIn("Agent 评测报告", client.get(f"/api/runs/{run_id}/report").text)
            self.assertIn(run_id, client.get("/").text)
            self.assertEqual(client.get("/api/runs/does-not-exist").status_code, 404)
            self.assertTrue(client.get("/api/badcases").json())


if __name__ == "__main__":
    unittest.main()
