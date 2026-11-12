"""把轨迹和评分导出成 OpenTelemetry GenAI 语义约定的形状，方便接入已有的可观测平台
（Langfuse、Phoenix、SigNoz 等都认这套 span）。

span：`invoke_agent {agent}` 为根，`chat {model}` 与 `execute_tool {tool}` 为子；
评分：挂在根 span 上的 `gen_ai.evaluation.result` 事件，一个维度一条。

注意（2026-11 核对过）：GenAI 语义约定整体仍是 Development 状态，且已从
open-telemetry/semantic-conventions 迁到独立仓库 semantic-conventions-genai。
`gen_ai.operation.name`、`gen_ai.response.finish_reasons` 这类命名与这里一致；
但 `gen_ai.tool.call.arguments` / `gen_ai.tool.call.result` 属于「内容捕获」类属性，
后端默认可关闭，格式也偏向结构化 AnyValue 而非 JSON 字符串；`gen_ai.agent.name`
在 A2A 提案里被移出过，说明 agent 相关命名尚未定稿。所以这里输出 JSONL 供转换，
接入具体后端前请对照它支持的约定版本。
"""
from __future__ import annotations

import json
from typing import Any, Iterator

from .models import TrialResult


def trial_to_otel(trial: TrialResult) -> Iterator[dict[str, Any]]:
    tr = trial.trace
    if tr is None:
        return
    root_id = f"{tr.trace_id}-root"
    clock = 0.0
    for i, ev in enumerate(tr.timeline):
        if ev["kind"] == "chat":
            yield {"trace_id": tr.trace_id, "span_id": f"{tr.trace_id}-{i}", "parent_span_id": root_id,
                   "name": f"chat {ev['model']}", "start_ms": clock, "end_ms": clock + ev["latency_ms"],
                   "attributes": {"gen_ai.operation.name": "chat", "gen_ai.request.model": tr.model,
                                  "gen_ai.response.model": ev["model"],
                                  "gen_ai.usage.input_tokens": ev["input_tokens"],
                                  "gen_ai.usage.output_tokens": ev["output_tokens"],
                                  "gen_ai.response.finish_reasons": [ev["stop_reason"]]}}
            clock += ev["latency_ms"]
        elif ev["kind"] == "tool":
            attrs = {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": ev["name"],
                     "gen_ai.tool.call.id": ev["id"],
                     "gen_ai.tool.call.arguments": json.dumps(ev["args"], ensure_ascii=False),
                     "gen_ai.tool.call.result": json.dumps(ev["result"], ensure_ascii=False)}
            if ev.get("error"):
                attrs["error.type"] = ev["error"]
            yield {"trace_id": tr.trace_id, "span_id": f"{tr.trace_id}-{i}", "parent_span_id": root_id,
                   "name": f"execute_tool {ev['name']}", "start_ms": clock,
                   "end_ms": clock + ev["latency_ms"], "attributes": attrs}
            clock += ev["latency_ms"]
    events = [{"name": "gen_ai.evaluation.result",
               "attributes": {"gen_ai.evaluation.name": s.dimension,
                              "gen_ai.evaluation.score.value": s.score,
                              "gen_ai.evaluation.score.label": "pass" if s.passed else "fail",
                              "gen_ai.evaluation.explanation": s.reason}}
              for s in trial.scores if not s.skipped]
    yield {"trace_id": tr.trace_id, "span_id": root_id, "parent_span_id": None,
           "name": f"invoke_agent {tr.agent}", "start_ms": 0.0, "end_ms": clock,
           "attributes": {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": tr.agent,
                          "gen_ai.request.model": tr.model,
                          "gen_ai.usage.input_tokens": tr.usage.input_tokens,
                          "gen_ai.usage.output_tokens": tr.usage.output_tokens,
                          "evalhub.case_id": trial.case_id, "evalhub.trial": trial.trial,
                          "evalhub.passed": trial.passed},
           "events": events}


def export_jsonl(trials: list[TrialResult], path: str) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for t in trials:
            for span in trial_to_otel(t):
                f.write(json.dumps(span, ensure_ascii=False) + "\n")
                n += 1
    return n
