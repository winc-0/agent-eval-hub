"""命令行入口：python -m evalhub <命令>

  lint       用例集体检
  selftest   harness 自检（oracle 应全过、null 应全挂）
  run        跑一次评测
  report     生成 HTML / JSON / JUnit / Markdown 报告
  gate       质量门禁，不通过时退出码为 1
  badcase    Badcase 池：list / pin / import / promote
  calibrate  用人工标注集校准裁判
  reviews    查看待人工复核的条目
  export     导出 OpenTelemetry GenAI 形状的 JSONL
  serve      启动 HTTP 服务
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import badcase as bc
from .agent import make_agent
from .envvars import MissingCredential
from .export import export_jsonl
from .gate import evaluate_gate, load_gate
from .judge import calibrate, make_judge
from .llm import register_pricing
from .report import build_report, render_markdown, write_reports
from .report.html import currency_symbol
from .scheduler import RunConfig, regrade_run, run_suite
from .selftest import selftest
from .store import Store
from .suite import Suite, lint


def _pct(v: float) -> str:
    return f"{v * 100:5.1f}%"


def print_summary(report: dict[str, Any]) -> None:
    h, e, run = report["headline"], report["efficiency"], report["run"]
    sym = currency_symbol(report)
    print(f"\n运行 {run['run_id']}  评测集 {run['suite']}  被测 {run['agent']}  裁判 {run['judge']}")
    print(f"通过率 {_pct(h['rate'])}  (95% 区间 {_pct(h['ci_low'])} ~ {_pct(h['ci_high'])})   "
          f"已评分 {h['scored']}/{h['n_trials']}  故障 {h['errors']}  截断 {h['truncated']}")
    for r in report["reliability"]:
        print(f"  k={r['k']}  pass@k {_pct(r['pass_at_k'])}   pass^k {_pct(r['pass_hat_k'])}")
    print("维度：")
    for d in report["dimensions"]:
        if d["n_trials"]:
            role = "门禁" if d["gating"] else "诊断"
            print(f"  {d['label']:<6}{role}  {_pct(d['rate'])}  ({_pct(d['ci_low'])} ~ {_pct(d['ci_high'])})")
    print(f"延迟 P50 {e['latency_p50_ms'] / 1000:.2f}s / P95 {e['latency_p95_ms'] / 1000:.2f}s   "
          f"单次成本 {sym}{e['cost_per_trial_usd']:.4f}   裁判成本 {sym}{e['judge_cost_usd']:.4f}"
          f"（升级 {e['judge_escalated']}，待人工 {e['judge_needs_human']}）")
    if report["failures"]:
        print("失败归因：" + "，".join(f"{f['tag']}×{f['trials']}" for f in report["failures"][:6]))
    b = report.get("baseline")
    if b:
        sig = "显著" if b["significant"] else "在噪声范围内"
        print(f"对比基线 {b['run_id']}：{b['delta'] * 100:+.1f}pp "
              f"({b['ci_low'] * 100:+.1f} ~ {b['ci_high'] * 100:+.1f})，{sig}；"
              f"变差 {len(b['regressions'])} 条，变好 {len(b['fixes'])} 条")


def cmd_lint(args: argparse.Namespace) -> int:
    findings = lint(Suite.load(args.suite))
    for level, msg in findings:
        print(f"[{level}] {msg}")
    return 1 if any(level == "error" for level, _ in findings) else 0


def cmd_selftest(args: argparse.Namespace) -> int:
    res = asyncio.run(selftest(Suite.load(args.suite), Store(args.db)))
    print(f"oracle：{res['oracle_cases'] - len(res['oracle_failures'])}/{res['oracle_cases']} 通过"
          f"（另有 {res['no_reference']} 条没有参考解）")
    for f in res["oracle_failures"]:
        print(f"  ✗ {f['case_id']}：{f['why']}")
    for w in res["oracle_warnings"]:
        print(f"  ! {w['case_id']}（门禁通过，诊断项有瑕疵）：{w['why']}")
    print(f"null：{len(res['null_passes'])} 条用例在“什么都不做”时也通过" + ("" if res["null_passes"] else "，符合预期"))
    for cid in res["null_passes"]:
        print(f"  ✗ {cid}：评估器太松")
    print("自检" + ("通过" if res["ok"] else "未通过：先修用例或评估器，再跑真实模型"))
    return 0 if res["ok"] else 1


def cmd_run(args: argparse.Namespace) -> int:
    suite, store = Suite.load(args.suite), Store(args.db)
    if args.price:
        inp, out = (float(x) for x in args.price.split(","))
        register_pricing(args.agent.partition(":")[2], inp, out)
    agent = make_agent(args.agent, args.user_sim)
    judge = make_judge(args.judge, args.judge_threshold, args.judge_votes)
    cfg = RunConfig(trials=args.trials, concurrency=args.concurrency, max_attempts=args.max_attempts,
                    trial_timeout_s=args.timeout, rps=args.rps, seed=args.seed, tag=args.tag, currency=args.currency)

    def progress(result, done: int, total: int) -> None:
        if args.verbose:
            mark = "故障" if result.status == "error" else "✓" if result.passed else "✗"
            print(f"[{done}/{total}] {result.case_id}#{result.trial + 1} {mark}", file=sys.stderr)
        elif done == total or done % max(1, total // 10) == 0:
            print(f"  进度 {done}/{total}", file=sys.stderr)

    only = set(args.only.split(",")) if args.only else None
    run_id = asyncio.run(run_suite(suite, agent, store, cfg, judge, resume_run_id=args.resume,
                                   progress=progress, only=only))
    added = bc.collect_badcases(store, run_id)
    baseline = store.resolve_run(args.baseline, suite.config.name) if args.baseline else None
    print_summary(build_report(store, run_id, baseline))
    paths = write_reports(store, run_id, args.out, baseline)
    print(f"\n报告：{paths['html']}\n本次归入 Badcase 池 {added} 条。run_id={run_id}")
    return 0


def cmd_regrade(args: argparse.Namespace) -> int:
    suite, store = Suite.load(args.suite), Store(args.db)
    source = store.resolve_run(args.run, suite.config.name)
    judge = make_judge(args.judge, args.judge_threshold, args.judge_votes)
    run_id = asyncio.run(regrade_run(suite, store, source, judge, args.tag))
    print(f"已用当前评估器重新评分 {source} 的轨迹（未调用被测模型）")
    print_summary(build_report(store, run_id, source))
    paths = write_reports(store, run_id, args.out, source)
    print(f"\n报告：{paths['html']}\nrun_id={run_id}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    store = Store(args.db)
    run_id = store.resolve_run(args.run)
    baseline = store.resolve_run(args.baseline) if args.baseline else None
    print_summary(build_report(store, run_id, baseline))
    paths = write_reports(store, run_id, args.out, baseline)
    print("\n" + "\n".join(f"{k}: {v}" for k, v in paths.items()))
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    store = Store(args.db)
    run_id = store.resolve_run(args.run)
    baseline = store.resolve_run(args.baseline) if args.baseline else None
    report = build_report(store, run_id, baseline)
    result = evaluate_gate(report, load_gate(args.gate))
    for c in result["checks"]:
        print(f"{'✓' if c['passed'] else '✗'} {c['name']:<34} 实际 {c['actual']:<28} 要求 {c['threshold']}")
    print("门禁" + ("通过" if result["passed"] else "未通过"))
    write_reports(store, run_id, args.out, baseline, gate=result)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:   # 在 GitHub Actions 里把摘要贴到运行页面
        with open(summary, "a", encoding="utf-8") as f:
            f.write(render_markdown(report, result))
    return 0 if result["passed"] else 1


def cmd_badcase(args: argparse.Namespace) -> int:
    store = Store(args.db)
    if args.action == "list":
        rows = store.badcases(args.suite, args.status)
        for r in rows:
            print(f"#{r['id']:<4} {r['status']:<9} ×{r['occurrences']:<3} {r['case_id']:<38} "
                  f"{', '.join(r['failure_tags'])}")
        print(f"共 {len(rows)} 条")
        return 0
    suite = Suite.load(args.suite)
    if args.action == "pin":
        print(f"已钉为回归用例：{bc.pin_regression(suite, args.target)}")
    elif args.action == "import":
        rows = bc.import_traces(suite, store, args.target)
        for r in rows:
            print(f"{r['trace_id']}: " + (f"命中 {', '.join(r['tags'])} → 草稿 {r['draft']}" if r["tags"] else "未发现问题"))
        print(f"共筛查 {len(rows)} 条线上轨迹，生成草稿 {sum(bool(r['draft']) for r in rows)} 条")
    elif args.action == "promote":
        print(f"已入库：{bc.promote_draft(suite, args.target)}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    suite = Suite.load(args.suite)
    labels = Path(args.labels) if args.labels else suite.path / "human_labels.jsonl"
    rows = [json.loads(ln) for ln in labels.read_text(encoding="utf-8").splitlines() if ln.strip()]
    judge = make_judge(args.judge, args.judge_threshold, args.judge_votes)
    if judge is None:
        print("需要指定裁判（--judge）")
        return 2
    rep = asyncio.run(calibrate(judge, rows, args.concurrency))
    print(f"人工标注 {rep['n']} 条")
    for t in rep["tiers"]:
        bad = f"  故障 {t['errors']}" if t["errors"] else ""
        print(f"  第 {t['tier']} 级 {t['model']:<20} 一致率 {_pct(t['agreement'])}  kappa {t['kappa']:.2f}  "
              f"误放 {t['false_pass']}  误杀 {t['false_fail']}  拿不准 {t['unsure']}  成本 ${t['cost_usd']:.4f}{bad}")
    c = rep["cascade"]
    print(f"  分级整体                     一致率 {_pct(c['agreement'])}  kappa {c['kappa']:.2f}  "
          f"升级率 {_pct(c['escalation_rate'])}  转人工 {_pct(c['needs_human_rate'])}  "
          f"成本 ${c['cost_usd']:.4f}（全用强模型 ${c['cost_if_all_strong']:.4f}）")
    if c["unsure"]:   # 拿不准的条目会被保守算成不一致，所以另给一个只看已判定的口径
        print(f"  其中 {c['unsure']} 条裁判没给出判定，只看已判定的 {c['n'] - c['unsure']} 条："
              f"一致率 {_pct(c['agreement_decided'])}")
    for d in rep["disagreements"]:
        print(f"  ≠ {d['id']}: 人工 {d['human']} / 裁判 {d['judge']} —— {d['reason']} ｜ {d['answer']}")
    for name, ids in rep["tier_errors"].items():   # 某一级整体调不通时，它的统计不能当真
        print(f"注意：{name} 有 {len(ids)} 条调用失败，该级的一致率不可信：{', '.join(map(str, ids))}")
    if rep["judge_errors"]:
        print(f"裁判故障 {len(rep['judge_errors'])} 条：{', '.join(map(str, rep['judge_errors']))}")
        print("结论：本次校准不完整，先修接口再下判断")
        return 2
    if c["cost_usd"] == 0.0:
        print("（成本记 0：这两级模型没有登记价格，只影响金额展示，不影响一致率）")
    ok = c["agreement"] >= args.min_agreement
    print(f"结论：一致率{'达到' if ok else '低于'} {args.min_agreement:.0%} 的要求" +
          ("" if ok else "，裁判提示词需要继续迭代，暂不能用它的分数做门禁"))
    return 0 if ok else 1


def cmd_reviews(args: argparse.Namespace) -> int:
    rows = Store(args.db).human_reviews(args.run)
    for r in rows:
        print(f"#{r['id']} {r['run_id']} {r['case_id']}#{r['trial'] + 1} [{r['dimension']}] {r['detail']}")
    print(f"待人工复核 {len(rows)} 条")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    store = Store(args.db)
    n = export_jsonl(store.trials(store.resolve_run(args.run)), args.out)
    print(f"已导出 {n} 个 span → {args.out}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server import create_app
    uvicorn.run(create_app(args.db, args.suites), host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evalhub", description="Agent 与 LLM 评测中台")
    p.add_argument("--db", default="evalhub.db", help="SQLite 文件路径")
    sub = p.add_subparsers(dest="cmd", required=True)

    def judge_opts(sp: argparse.ArgumentParser, default: str) -> None:
        sp.add_argument("--judge", default=default, help="none / mock / anthropic[:小,强] / openai:小,强")
        sp.add_argument("--judge-threshold", type=float, default=0.75, help="低于该把握度就升级到下一级裁判")
        sp.add_argument("--judge-votes", type=int, default=1, help="第一级裁判的投票数，票不一致即升级")

    sp = sub.add_parser("lint", help="用例集体检"); sp.add_argument("suite"); sp.set_defaults(func=cmd_lint)
    sp = sub.add_parser("selftest", help="harness 自检"); sp.add_argument("suite"); sp.set_defaults(func=cmd_selftest)

    sp = sub.add_parser("run", help="跑一次评测")
    sp.add_argument("suite")
    sp.add_argument("--agent", default="mock:cs-v2", help="mock:cs-v1 / mock:cs-v2 / anthropic[:模型] / openai:模型")
    judge_opts(sp, "mock")
    sp.add_argument("--trials", type=int, default=3, help="每条用例跑几次")
    sp.add_argument("--concurrency", type=int, default=8)
    sp.add_argument("--max-attempts", type=int, default=3, help="基础设施故障的最大尝试次数")
    sp.add_argument("--timeout", type=float, default=120.0, help="单个 trial 的硬超时（秒）")
    sp.add_argument("--rps", type=float, default=None, help="每秒最多启动多少个 trial")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--tag", default="", help="给这次运行起个标签，如 baseline")
    sp.add_argument("--baseline", default=None, help="对比的基线：run_id / latest / tag:<标签>")
    sp.add_argument("--only", default=None, help="只跑这些用例 id，逗号分隔")
    sp.add_argument("--resume", default=None, help="断点续跑的 run_id")
    sp.add_argument("--user-sim", default=None, help="用模型扮演用户做多轮，如 anthropic:claude-haiku-5-5")
    sp.add_argument("--price", default=None, help="被测模型价格：输入,输出（每百万 token，币种见 --currency）")
    sp.add_argument("--currency", default="USD", choices=["USD", "CNY"], help="报告里的货币符号；不做汇率换算")
    sp.add_argument("--out", default="reports")
    sp.add_argument("-v", "--verbose", action="store_true")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("regrade", help="用当前评估器给已有轨迹重新评分，不调用被测模型")
    sp.add_argument("suite"); sp.add_argument("run", help="run_id / latest / tag:<标签>")
    judge_opts(sp, "none")
    sp.add_argument("--tag", default=None, help="新运行的标签，默认 <原标签>-regrade")
    sp.add_argument("--out", default="reports"); sp.set_defaults(func=cmd_regrade)

    sp = sub.add_parser("report", help="生成报告")
    sp.add_argument("run"); sp.add_argument("--baseline", default=None); sp.add_argument("--out", default="reports")
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("gate", help="质量门禁")
    sp.add_argument("run"); sp.add_argument("--gate", required=True); sp.add_argument("--baseline", default=None)
    sp.add_argument("--out", default="reports"); sp.set_defaults(func=cmd_gate)

    sp = sub.add_parser("badcase", help="Badcase 池")
    sp.add_argument("action", choices=["list", "pin", "import", "promote"])
    sp.add_argument("suite", nargs="?", default=None, help="list 时是评测集名（可省略），其余是评测集目录")
    sp.add_argument("target", nargs="?", default=None, help="pin/promote：用例 id；import：线上轨迹 JSONL")
    sp.add_argument("--status", default=None)
    sp.set_defaults(func=cmd_badcase)

    sp = sub.add_parser("calibrate", help="校准裁判")
    sp.add_argument("suite"); sp.add_argument("--labels", default=None)
    sp.add_argument("--min-agreement", type=float, default=0.9)
    sp.add_argument("--concurrency", type=int, default=4, help="并发判几条标注（真实模型下单条要十几秒）")
    judge_opts(sp, "mock"); sp.set_defaults(func=cmd_calibrate)

    sp = sub.add_parser("reviews", help="待人工复核队列"); sp.add_argument("--run", default=None)
    sp.set_defaults(func=cmd_reviews)

    sp = sub.add_parser("export", help="导出 OTel GenAI JSONL")
    sp.add_argument("run"); sp.add_argument("--out", default="traces.otel.jsonl"); sp.set_defaults(func=cmd_export)

    sp = sub.add_parser("serve", help="启动 HTTP 服务")
    sp.add_argument("--suites", default="suites"); sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8600); sp.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):   # Windows 控制台默认编码不是 UTF-8
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except MissingCredential as e:   # 配置问题：给一句说明就够了，不打印堆栈
        print(f"配置缺失：{e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
