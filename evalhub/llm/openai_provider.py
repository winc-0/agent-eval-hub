"""OpenAI 兼容接口接入：DeepSeek、豆包、通义千问、Kimi 等国内模型都走这一个实现。

只用于非 Claude 模型。环境变量（终端里没有时，会去读 Windows 系统里保存的环境变量）：
  OPENAI_BASE_URL   例如 https://dashscope.aliyuncs.com/compatible-mode/v1（没有时读 DASHSCOPE_BASE_URL）
  OPENAI_API_KEY    对应厂商的 key（没有时读 DASHSCOPE_API_KEY）
价格各家不同，用 `--price 输入,输出`（每百万 token）登记，否则成本记 0。
"""
from __future__ import annotations

import json
import time
from typing import Any, Optional

from .base import ChatResponse, Completion, InfraError, LLMUsage, ToolCall, ToolResult

_STOP = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens", "content_filter": "refusal"}


class OpenAICompatProvider:
    name = "openai"

    def __init__(self, model: str, base_url: Optional[str] = None, max_tokens: int = 4096):
        import openai

        from ..envvars import require_env

        # 两个都必填：不给接口地址时 SDK 会默认发往 api.openai.com，
        # 把别家的 Key 误发给第三方，所以这里不留默认值。
        # OPENAI_* 优先；没有时用阿里云百炼的习惯命名 DASHSCOPE_*。
        base_url = base_url or require_env(
            "OPENAI_BASE_URL", "DASHSCOPE_BASE_URL",
            hint="需要填模型服务的 OpenAI 兼容接口地址，例如 https://dashscope.aliyuncs.com/compatible-mode/v1")
        api_key = require_env("OPENAI_API_KEY", "DASHSCOPE_API_KEY", hint="需要填模型服务的 API Key。")
        self._sdk = openai
        self.client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key,
                                         max_retries=0)   # 重试交给调度器
        self.model = model
        self.max_tokens = max_tokens

    async def _create(self, **kwargs: Any):
        sdk = self._sdk
        started = time.perf_counter()
        try:
            resp = await self.client.chat.completions.create(
                model=self.model, max_tokens=self.max_tokens, **kwargs)
        except sdk.RateLimitError as e:
            raise InfraError("rate_limited", str(e)) from e
        except sdk.APIStatusError as e:
            if e.status_code >= 500:
                raise InfraError("provider_5xx", str(e)) from e
            raise InfraError("bad_request", str(e), retryable=False) from e
        except sdk.APIConnectionError as e:
            raise InfraError("network", str(e)) from e
        return resp, (time.perf_counter() - started) * 1000

    @staticmethod
    def _usage(u: Any) -> LLMUsage:
        if u is None:
            return LLMUsage()
        details = getattr(u, "prompt_tokens_details", None)
        cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
        return LLMUsage(input_tokens=(u.prompt_tokens or 0) - cached,
                        output_tokens=u.completion_tokens or 0, cache_read_tokens=cached)

    def start(self, system: str, tools: list[dict[str, Any]], seed: int = 0) -> "_OpenAIConversation":
        return _OpenAIConversation(self, system, tools)

    async def complete(self, system: str, user: str, json_schema: Optional[dict] = None) -> Completion:
        kwargs: dict[str, Any] = {}
        if json_schema:
            # 各家对 json_schema 的支持不一，json_object 是最大公约数；schema 写进提示词里。
            system += "\n\n只输出一个 JSON 对象，符合这个 JSON Schema：\n" + json.dumps(json_schema, ensure_ascii=False)
            kwargs["response_format"] = {"type": "json_object"}
        resp, latency_ms = await self._create(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **kwargs)
        choice = resp.choices[0]
        return Completion(choice.message.content or "", self._usage(resp.usage), latency_ms,
                          resp.model or self.model, _STOP.get(choice.finish_reason, "end_turn"))


class _OpenAIConversation:
    def __init__(self, provider: OpenAICompatProvider, system: str, tools: list[dict[str, Any]]):
        self.p = provider
        self.tools = [{"type": "function", "function": t} for t in tools]
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system}]

    async def _turn(self) -> ChatResponse:
        kwargs: dict[str, Any] = {"messages": self.messages}
        if self.tools:
            kwargs["tools"] = self.tools
        resp, latency_ms = await self.p._create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message
        calls, raw_calls = [], []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_unparseable": tc.function.arguments}   # 让沙箱按 invalid_args 记一次失败调用
            calls.append(ToolCall(tc.id, tc.function.name, args if isinstance(args, dict) else {"_value": args}))
            raw_calls.append({"id": tc.id, "type": "function",
                              "function": {"name": tc.function.name, "arguments": tc.function.arguments}})
        assistant: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if raw_calls:
            assistant["tool_calls"] = raw_calls
        self.messages.append(assistant)
        return ChatResponse(msg.content or "", calls, _STOP.get(choice.finish_reason, "end_turn"),
                            self.p._usage(resp.usage), latency_ms, resp.model or self.p.model)

    async def send_user(self, text: str) -> ChatResponse:
        self.messages.append({"role": "user", "content": text})
        return await self._turn()

    async def send_tool_results(self, results: list[ToolResult]) -> ChatResponse:
        for r in results:
            self.messages.append({"role": "tool", "tool_call_id": r.id, "content": r.content})
        return await self._turn()
