"""按 spec 字符串创建模型接入：

  mock:cs-v1 / mock:cs-v2        离线规则模型（默认，不需要 key）
  anthropic[:model]              Claude，默认 claude-opus-5-5
  openai:<model>                 任意 OpenAI 兼容接口（DeepSeek、豆包、通义千问……）
"""
from __future__ import annotations

from .base import (ChatProvider, ChatResponse, Completion, Conversation, InfraError, LLMUsage,
                   ToolCall, ToolResult, cost_usd, price_of, register_pricing)


def make_provider(spec: str) -> ChatProvider:
    kind, _, model = spec.partition(":")
    if kind == "mock":
        from .mock import MockCSProvider
        return MockCSProvider(model.removeprefix("cs-") or "v2")
    if kind == "anthropic":
        from .anthropic_provider import DEFAULT_MODEL, AnthropicProvider
        return AnthropicProvider(model or DEFAULT_MODEL)
    if kind == "openai":
        if not model:
            raise ValueError("openai 接入需要写明模型名，例如 openai:deepseek-chat")
        from .openai_provider import OpenAICompatProvider
        return OpenAICompatProvider(model)
    raise ValueError(f"无法识别的模型 spec：{spec}")


__all__ = ["ChatProvider", "ChatResponse", "Completion", "Conversation", "InfraError", "LLMUsage",
           "ToolCall", "ToolResult", "cost_usd", "make_provider", "price_of", "register_pricing"]
