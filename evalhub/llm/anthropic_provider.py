"""Claude 接入（官方 anthropic SDK，Messages API + 手写工具循环）。

几个和“评测”强相关的取舍：
  - max_retries=0：重试交给调度器统一做，这样延迟列只包含最后一次成功请求，
    不会因为某个模型多撞了几次限流就显得更慢。
  - 不开服务端 fallback：评测要的是“这个模型”的分数。被别的模型兜底的回答
    会污染结果，所以每次响应都校验实际服务模型，不一致就直接报错。
  - 拒答（stop_reason=refusal）是一种被评分的结果，不是基础设施错误。
"""
from __future__ import annotations

import time
from typing import Any, Optional

from .base import ChatResponse, Completion, InfraError, LLMUsage, ToolCall, ToolResult

DEFAULT_MODEL = "claude-opus-5-5"

_STOP = {"end_turn": "end_turn", "tool_use": "tool_use", "max_tokens": "max_tokens", "refusal": "refusal"}


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "medium", max_tokens: int = 16000):
        import inspect

        import anthropic   # 延迟导入：只跑离线 mock 时不需要装 SDK

        from anthropic.resources.messages import AsyncMessages

        if "output_config" not in inspect.signature(AsyncMessages.create).parameters:
            raise RuntimeError(f"anthropic SDK 版本过旧（{anthropic.__version__}），不支持 output_config。"
                               "请先执行：pip install -U anthropic")
        self._sdk = anthropic
        from ..envvars import read_env

        # 终端里没有、但系统环境变量里存了的，补给 SDK；都没有时交给 SDK 自己解析
        # （ANTHROPIC_AUTH_TOKEN、`ant auth login` 的 profile 等），所以这里不强制要求。
        options: dict[str, Any] = {}
        for env_name, option in (("ANTHROPIC_API_KEY", "api_key"), ("ANTHROPIC_BASE_URL", "base_url")):
            value = read_env(env_name)
            if value:
                options[option] = value
        self.client = anthropic.AsyncAnthropic(max_retries=0, **options)
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    async def _create(self, output_format: Optional[dict] = None, **kwargs: Any):
        sdk = self._sdk
        output_config: dict[str, Any] = {"effort": self.effort}
        if output_format is not None:
            output_config["format"] = output_format
        started = time.perf_counter()
        try:
            resp = await self.client.messages.create(
                model=self.model, max_tokens=self.max_tokens, output_config=output_config, **kwargs)
        except sdk.RateLimitError as e:
            raise InfraError("rate_limited", str(e)) from e
        except sdk.APIStatusError as e:
            if e.status_code >= 500:
                raise InfraError("provider_5xx", str(e)) from e
            raise InfraError("bad_request", str(e), retryable=False) from e   # 4xx 重试也没用，直接暴露
        except sdk.APIConnectionError as e:   # 含超时
            raise InfraError("network", str(e)) from e
        latency_ms = (time.perf_counter() - started) * 1000
        if not resp.model.startswith(self.model):
            raise InfraError("served_model_mismatch",
                             f"请求的是 {self.model}，实际由 {resp.model} 提供服务", retryable=False)
        return resp, latency_ms

    @staticmethod
    def _usage(u: Any) -> LLMUsage:
        return LLMUsage(
            input_tokens=u.input_tokens or 0,
            output_tokens=u.output_tokens or 0,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )

    def start(self, system: str, tools: list[dict[str, Any]], seed: int = 0) -> "_AnthropicConversation":
        return _AnthropicConversation(self, system, tools)

    async def complete(self, system: str, user: str, json_schema: Optional[dict] = None) -> Completion:
        fmt = {"type": "json_schema", "schema": json_schema} if json_schema else None
        resp, latency_ms = await self._create(
            output_format=fmt, system=system, messages=[{"role": "user", "content": user}])
        text = "".join(b.text for b in resp.content if b.type == "text")
        return Completion(text, self._usage(resp.usage), latency_ms, resp.model,
                          _STOP.get(resp.stop_reason, "end_turn"))


class _AnthropicConversation:
    def __init__(self, provider: AnthropicProvider, system: str, tools: list[dict[str, Any]]):
        self.p = provider
        # 系统提示词和工具定义每轮都不变，打上缓存断点
        self.system = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        self.tools = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                      for t in tools]
        self.messages: list[dict[str, Any]] = []

    async def _turn(self) -> ChatResponse:
        kwargs: dict[str, Any] = {"system": self.system, "messages": self.messages}
        if self.tools:
            kwargs["tools"] = self.tools
        resp, latency_ms = await self.p._create(**kwargs)
        # 整个 content 原样追加：thinking 块必须不改动地回传，只留文本会破坏后续请求
        self.messages.append({"role": "assistant", "content": resp.content})
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"]
        return ChatResponse(text, calls, _STOP.get(resp.stop_reason, "end_turn"),
                            self.p._usage(resp.usage), latency_ms, resp.model)

    async def send_user(self, text: str) -> ChatResponse:
        self.messages.append({"role": "user", "content": text})
        return await self._turn()

    async def send_tool_results(self, results: list[ToolResult]) -> ChatResponse:
        # 同一轮的所有工具结果必须放在一条 user 消息里
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": r.id, "content": r.content, "is_error": r.is_error}
            for r in results]})
        return await self._turn()
