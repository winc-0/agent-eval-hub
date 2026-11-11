"""沙箱：每次 trial 一份独立的 mock 环境。

为什么要沙箱：
  - 隔离：上一条用例退过的款不能影响下一条，否则分数和执行顺序有关。
  - 可判定：任务做没做成，看环境终态（退款记录在不在、金额对不对），不看 Agent 自己怎么说。
  - 可控：故障注入（超时/报错/空结果）和提示注入都在这一层做，被测 Agent 无感知。
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..models import EnvSpec, ToolCallRecord


class ToolError(Exception):
    """工具的业务错误（订单不存在等），会作为 is_error 的 tool_result 返回给 Agent。"""


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]                 # JSON Schema
    fn: Callable[[dict, dict], Any]            # (state, args) -> result
    mutating: bool = False
    latency_ms: float = 80.0                   # mock 工具的模拟耗时


def deep_merge(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


_JSON_TYPES: dict[str, tuple] = {
    "string": (str,), "number": (int, float), "integer": (int,),
    "boolean": (bool,), "object": (dict,), "array": (list,),
}


class Sandbox:
    def __init__(self, tools: list[ToolSpec], state: dict, env: Optional[EnvSpec] = None):
        self.env = env or EnvSpec()
        self.tools = {t.name: t for t in tools}
        self.state = deep_merge(copy.deepcopy(state), self.env.overrides)
        self.initial_state = copy.deepcopy(self.state)
        self.calls: list[ToolCallRecord] = []
        self._fault_left = [f.times for f in self.env.faults]

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [{"name": t.name, "description": t.description, "parameters": t.parameters}
                for t in self.tools.values()]

    def _validate(self, spec: ToolSpec, args: dict) -> Optional[str]:
        props = spec.parameters.get("properties", {})
        for key in spec.parameters.get("required", []):
            if key not in args:
                return f"缺少必填参数 {key}"
        for key, value in args.items():
            if key not in props:
                return f"未知参数 {key}"
            expected = props[key].get("type")
            if expected in _JSON_TYPES:
                ok = isinstance(value, _JSON_TYPES[expected])
                if expected in ("number", "integer") and isinstance(value, bool):
                    ok = False
                if not ok:
                    return f"参数 {key} 类型应为 {expected}"
        return None

    def _take_fault(self, name: str):
        for i, fault in enumerate(self.env.faults):
            if fault.tool == name and self._fault_left[i] > 0:
                self._fault_left[i] -= 1
                return fault
        return None

    def call(self, name: str, args: dict, turn: int = 0) -> ToolCallRecord:
        """执行一次工具调用并记录。永不抛异常：所有失败都变成带 error 的记录返回给 Agent。"""
        rec = ToolCallRecord(id=f"call_{len(self.calls) + 1}", name=name, args=dict(args or {}),
                             turn=turn, step=len(self.calls) + 1)
        self.calls.append(rec)
        spec = self.tools.get(name)
        if spec is None:
            rec.error, rec.result = "unknown_tool", {"error": f"不存在名为 {name} 的工具"}
            return rec
        rec.latency_ms = spec.latency_ms
        problem = self._validate(spec, rec.args)
        if problem:
            rec.error, rec.result = "invalid_args", {"error": problem}
            return rec
        fault = self._take_fault(name)
        if fault is not None:
            if fault.kind == "timeout":
                rec.error, rec.latency_ms = "timeout", 5000.0
                rec.result = {"error": fault.message or "工具调用超时"}
            elif fault.kind == "error":
                rec.error, rec.result = "tool_error", {"error": fault.message or "服务暂时不可用"}
            else:  # empty：静默返回空数据，考察 Agent 会不会拿空结果编答案
                rec.result = {}
            return rec
        try:
            result = spec.fn(self.state, rec.args)
        except ToolError as e:
            rec.error, rec.result = "tool_error", {"error": str(e)}
            return rec
        for inj in self.env.injections:
            if inj.tool == name and isinstance(result, dict):
                result = {**result, "note": inj.text}
        rec.result = result
        return rec
