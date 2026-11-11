"""调度引擎：把 (用例 × 试验次数) 铺成任务，受控并发地跑完。

  - 并发上限 + 可选限速（rps），几百条用例不会把模型接口打挂；
  - 每个 trial 一个全新沙箱、一个由 (运行种子, 用例, 第几次) 推导出的确定性种子；
  - 每个 trial 有独立于流式心跳的硬超时；
  - 基础设施故障按指数退避 + 抖动重试，记录尝试次数；重试耗尽记为 error，
    单独统计，绝不混进通过率（否则是把管道的问题算到模型头上）；
  - 结果逐条落库，支持断点续跑（只重跑没有可评分结果的那些）。
"""
from __future__ import annotations

import asyncio
import hashlib
import random
import time
from dataclasses import asdict, dataclass
from typing import Callable, Optional

from .agent import AgentAdapter
from .evaluators import EvalContext, evaluate_all, verdict
from .judge import CascadeJudge
from .llm import InfraError
from .models import Case, TrialResult
from .store import Store, new_run_id
from .suite import Suite


@dataclass
class RunConfig:
    trials: int = 3
    concurrency: int = 8
    max_attempts: int = 3
    trial_timeout_s: float = 120.0
    backoff_base_s: float = 0.5
    backoff_cap_s: float = 20.0
    rps: Optional[float] = None
    seed: int = 42
    tag: str = ""
    currency: str = "USD"   # 只影响报告里的货币符号；金额按登记的价格原样计算，不做汇率换算


def stable_seed(seed: int, case_id: str, trial: int) -> int:
    return int(hashlib.sha1(f"{seed}:{case_id}:{trial}".encode()).hexdigest()[:8], 16)


class _RateLimiter:
    def __init__(self, rps: Optional[float]):
        self.interval = 1.0 / rps if rps else 0.0
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        if not self.interval:
            return
        async with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.interval
        if wait:
            await asyncio.sleep(wait)


async def run_trial(suite: Suite, agent: AgentAdapter, judge: Optional[CascadeJudge], case: Case,
                    run_id: str, trial: int, seed: int) -> TrialResult:
    sandbox = suite.make_sandbox(case)
    trace = await agent.run(case, sandbox, suite.config.system_prompt, seed)
    scores = await evaluate_all(EvalContext(case, trace, suite, judge))
    passed, tags = verdict(scores, suite.config.gating_dimensions)
    # 被 max_tokens 截断的回答：展示并单独计数，但不当成“答错”计入均值。
    status = "truncated" if trace.stop_reason == "max_tokens" else "scored"
    return TrialResult(run_id=run_id, case_id=case.id, trial=trial, status=status, seed=seed,
                       passed=passed and status == "scored", scores=scores, failure_tags=tags, trace=trace)


async def run_suite(suite: Suite, agent: AgentAdapter, store: Store, cfg: RunConfig,
                    judge: Optional[CascadeJudge] = None, resume_run_id: Optional[str] = None,
                    progress: Optional[Callable[[TrialResult, int, int], None]] = None,
                    only: Optional[set[str]] = None, run_id: Optional[str] = None) -> str:
    cases = [c for c in suite.cases if only is None or c.id in only]
    if resume_run_id:
        run_id = resume_run_id
        done = store.completed(run_id)
    else:
        run_id, done = run_id or new_run_id(), set()
        index = suite.case_index()
        store.create_run(run_id, suite.config.name, suite.fingerprint(), agent.name, agent.model,
                         ",".join(judge.models) if judge else "none", cfg.tag, asdict(cfg),
                         {c.id: index[c.id] for c in cases}, suite.config.gating_dimensions)
    todo = [(c, t) for c in cases for t in range(cfg.trials) if (c.id, t) not in done]
    sem = asyncio.Semaphore(cfg.concurrency)
    limiter = _RateLimiter(cfg.rps)
    jitter = random.Random(cfg.seed)
    finished = 0

    async def one(case: Case, trial: int) -> None:
        nonlocal finished
        seed = stable_seed(cfg.seed, case.id, trial)
        attempts, result = 0, None
        while result is None:
            attempts += 1
            error_class, message, retryable = "", "", True
            try:
                async with sem:   # 退避等待时不占并发槽
                    await limiter.acquire()
                    result = await asyncio.wait_for(
                        run_trial(suite, agent, judge, case, run_id, trial, seed), cfg.trial_timeout_s)
            except asyncio.TimeoutError:
                error_class, message = "timeout", f"超过 {cfg.trial_timeout_s:g}s 硬超时"
            except InfraError as e:
                error_class, message, retryable = e.error_class, str(e), e.retryable
            except Exception as e:   # 评测代码自身的 bug：不重试、不吞掉，原样暴露
                error_class, message, retryable = "harness_error", f"{type(e).__name__}: {e}", False
            if result is None:
                if not retryable or attempts >= cfg.max_attempts:
                    result = TrialResult(run_id=run_id, case_id=case.id, trial=trial, status="error",
                                         seed=seed, error_class=error_class, error_message=message)
                else:
                    delay = min(cfg.backoff_cap_s, cfg.backoff_base_s * 2 ** (attempts - 1))
                    await asyncio.sleep(delay * (0.5 + jitter.random()))
        result.attempts = attempts
        store.save_trial(result)
        for s in result.scores:
            if s.judge and s.judge.needs_human:
                detail = "；".join(c.detail for c in s.checks if "待人工复核" in c.detail) or s.reason
                store.add_human_review(run_id, case.id, trial, s.dimension, detail)
        finished += 1
        if progress:
            progress(result, finished, len(todo))

    await asyncio.gather(*(one(c, t) for c, t in todo))
    store.finish_run(run_id)
    return run_id


async def regrade_run(suite: Suite, store: Store, source_run_id: str, judge: Optional[CascadeJudge] = None,
                      tag: Optional[str] = None) -> str:
    """用当前的评估器和用例期望，给一次已有运行里保存的轨迹重新打分，不再调用被测模型。

    修了评估器的误判之后用它验证效果，不花被测模型的额度。新结果另存为一次运行，
    原运行保持不变；故障的 trial 原样带过来（它们没有轨迹可评）。
    """
    src = store.get_run(source_run_id)
    trials = [t for t in store.trials(source_run_id) if t.case_id in {c.id for c in suite.cases}]
    run_id = new_run_id()
    index = suite.case_index()
    config = {**src["config"], "tag": tag if tag is not None else f"{src['tag']}-regrade",
              "regraded_from": source_run_id}
    store.create_run(run_id, suite.config.name, suite.fingerprint(), src["agent"], src["model"],
                     ",".join(judge.models) if judge else "none", config["tag"], config,
                     {t.case_id: index[t.case_id] for t in trials}, suite.config.gating_dimensions)
    for old in trials:
        if old.status == "error" or old.trace is None:
            store.save_trial(old.model_copy(update={"run_id": run_id}))
            continue
        case = suite.case(old.case_id)
        scores = await evaluate_all(EvalContext(case, old.trace, suite, judge))
        passed, tags = verdict(scores, suite.config.gating_dimensions)
        store.save_trial(old.model_copy(update={
            "run_id": run_id, "scores": scores, "failure_tags": tags,
            "passed": passed and old.status == "scored"}))
    store.finish_run(run_id)
    return run_id
