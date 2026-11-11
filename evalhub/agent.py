"""被测对象的适配层。

AgentAdapter 是评测中台和被测 Agent 之间唯一的接口：给它用例、沙箱和系统提示词，
它返回一条完整的 Trace。要评测你自己的 Agent，实现这个接口即可。

内置四个实现：
  LLMAgent    通用 Agent 循环（模型 ⇄ 工具），模型可换 Claude / OpenAI 兼容 / 离线 mock
  OracleAgent 回放用例的参考解 —— 应该接近满分，否则是用例或评估器有 bug
  NullAgent   什么都不做         —— 应该全挂，否则是评估器太松
"""
from __future__ import annotations

import copy
import json
import uuid
from typing import Optional, Protocol

from .llm import ChatProvider, ChatResponse, ToolResult, cost_usd
from .models import Case, ReferenceTurn, ToolCallRecord, Trace
from .sandbox import Sandbox


class AgentAdapter(Protocol):
    name: str
    model: str

    async def run(self, case: Case, sandbox: Sandbox, system_prompt: str, seed: int) -> Trace: ...


def _new_trace(case: Case, agent: AgentAdapter, sandbox: Sandbox) -> Trace:
    return Trace(trace_id=uuid.uuid4().hex[:16], case_id=case.id, agent=agent.name, model=agent.model,
                 initial_state=copy.deepcopy(sandbox.initial_state))


def _record_tool(trace: Trace, rec: ToolCallRecord) -> None:
    trace.latency_ms += rec.latency_ms
    trace.timeline.append({"kind": "tool", "turn": rec.turn, "step": rec.step, "id": rec.id,
                           "name": rec.name, "args": rec.args, "result": rec.result,
                           "error": rec.error, "latency_ms": rec.latency_ms})


def _finish(trace: Trace, sandbox: Sandbox, stop_reason: str = "end_turn") -> Trace:
    trace.tool_calls = list(sandbox.calls)
    trace.final_state = copy.deepcopy(sandbox.state)
    trace.stop_reason = stop_reason
    return trace


class UserSimulator(Protocol):
    async def next(self, case: Case, turn: int, trace: Trace) -> Optional[str]: ...


class ScriptedUser:
    """按用例里写好的轮次依次发言。可复现，是默认方式。"""

    async def next(self, case: Case, turn: int, trace: Trace) -> Optional[str]:
        return case.user.turns[turn] if turn < len(case.user.turns) else None


class LLMUserSimulator:
    """让一个模型扮演用户，根据 Agent 的回复动态追问（τ-bench 式的多轮评测）。

    注意：模拟用户本身会引入噪声——Agent 挂了，也可能是“用户”没把话说清楚。
    用它跑出来的失败要抽样看 transcript 再下结论，门禁建议仍用脚本化用户。
    """

    SYSTEM = ("你在扮演一位正在联系电商客服的真实用户。只输出用户下一句要说的话，不要解释。"
              "信息只在被问到时才给。如果你的目标已经达成，或客服明确表示办不了，只输出 [DONE]。")

    def __init__(self, provider: ChatProvider, max_turns: int = 6):
        self.provider = provider
        self.max_turns = max_turns

    async def next(self, case: Case, turn: int, trace: Trace) -> Optional[str]:
        if turn == 0:
            return case.user.turns[0]
        if turn >= self.max_turns:
            return None
        dialogue = "\n".join(f"用户：{u}\n客服：{a}" for u, a in zip(trace.user_messages, trace.answers))
        known = "；".join(case.user.turns[1:])
        prompt = (f"人物设定：{case.user.persona or '普通用户'}\n目标：{case.user.goal or case.title}\n"
                  f"你手里掌握、但只在被问到时才说的信息：{known or '无'}\n\n目前的对话：\n{dialogue}")
        out = await self.provider.complete(self.SYSTEM, prompt)
        text = out.text.strip()
        return None if not text or "[DONE]" in text else text


class LLMAgent:
    def __init__(self, provider: ChatProvider, max_steps: int = 12,
                 user: Optional[UserSimulator] = None):
        self.provider = provider
        self.name = f"{provider.name}:{provider.model}"
        self.model = provider.model
        self.max_steps = max_steps       # harness 自身的硬上限，防止死循环；不是用例的步数预算
        self.user = user or ScriptedUser()

    def _record_chat(self, trace: Trace, resp: ChatResponse, turn: int) -> None:
        u = trace.usage
        u.input_tokens += resp.usage.input_tokens
        u.output_tokens += resp.usage.output_tokens
        u.cache_read_tokens += resp.usage.cache_read_tokens
        u.cache_write_tokens += resp.usage.cache_write_tokens
        u.llm_calls += 1
        u.cost_usd += cost_usd(resp.model, resp.usage)
        trace.latency_ms += resp.latency_ms
        trace.timeline.append({"kind": "chat", "turn": turn, "model": resp.model,
                               "input_tokens": resp.usage.input_tokens,
                               "output_tokens": resp.usage.output_tokens,
                               "latency_ms": resp.latency_ms, "stop_reason": resp.stop_reason,
                               "tool_calls": [c.name for c in resp.tool_calls]})

    async def run(self, case: Case, sandbox: Sandbox, system_prompt: str, seed: int) -> Trace:
        conv = self.provider.start(system_prompt, sandbox.tool_schemas(), seed=seed)
        trace = _new_trace(case, self, sandbox)
        stop, turn = "end_turn", 0
        while stop == "end_turn":
            user_text = await self.user.next(case, turn, trace)
            if user_text is None:
                break
            trace.user_messages.append(user_text)
            trace.timeline.append({"kind": "user", "turn": turn, "content": user_text})
            resp = await conv.send_user(user_text)
            self._record_chat(trace, resp, turn)
            while resp.tool_calls:
                if len(sandbox.calls) + len(resp.tool_calls) > self.max_steps:
                    stop = "max_steps"
                    break
                results = []
                for tc in resp.tool_calls:
                    rec = sandbox.call(tc.name, tc.args, turn=turn)
                    _record_tool(trace, rec)
                    results.append(ToolResult(tc.id, json.dumps(rec.result, ensure_ascii=False),
                                              rec.error is not None))
                resp = await conv.send_tool_results(results)
                self._record_chat(trace, resp, turn)
            if resp.stop_reason in ("max_tokens", "refusal"):
                stop = resp.stop_reason
            trace.answers.append(resp.text)
            trace.timeline.append({"kind": "assistant", "turn": turn, "content": resp.text})
            turn += 1
        return _finish(trace, sandbox, stop)


class OracleAgent:
    name = "oracle"
    model = "oracle"

    async def run(self, case: Case, sandbox: Sandbox, system_prompt: str, seed: int) -> Trace:
        trace = _new_trace(case, self, sandbox)
        for turn, user_text in enumerate(case.user.turns):
            ref = case.reference[turn] if turn < len(case.reference) else ReferenceTurn()
            trace.user_messages.append(user_text)
            trace.timeline.append({"kind": "user", "turn": turn, "content": user_text})
            for exp in ref.tool_calls:
                _record_tool(trace, sandbox.call(exp.name, exp.args, turn=turn))
            trace.answers.append(ref.answer)
            trace.timeline.append({"kind": "assistant", "turn": turn, "content": ref.answer})
        return _finish(trace, sandbox)


class NullAgent:
    name = "null"
    model = "null"

    async def run(self, case: Case, sandbox: Sandbox, system_prompt: str, seed: int) -> Trace:
        trace = _new_trace(case, self, sandbox)
        for turn, user_text in enumerate(case.user.turns):
            trace.user_messages.append(user_text)
            trace.timeline.append({"kind": "user", "turn": turn, "content": user_text})
            trace.answers.append("")
            trace.timeline.append({"kind": "assistant", "turn": turn, "content": ""})
        return _finish(trace, sandbox)


def make_agent(spec: str, user_sim_spec: Optional[str] = None) -> AgentAdapter:
    """spec：oracle / null / 任意模型 spec（见 llm.make_provider）。"""
    from .llm import make_provider

    if spec == "oracle":
        return OracleAgent()
    if spec == "null":
        return NullAgent()
    user = LLMUserSimulator(make_provider(user_sim_spec)) if user_sim_spec else None
    return LLMAgent(make_provider(spec), user=user)
