"""模型接入层的公共类型。

评测中台天然要跨厂商（模型选型就是在比不同模型），所以这里只定义最小公共接口：
  - Conversation：带工具的多轮对话，各家把自己的原生消息历史藏在实现里
    （Claude 的 thinking 块必须原样回传，OpenAI 系的 tool_calls 格式又不一样）。
  - complete()：一次性补全，给 LLM 裁判和用户模拟器用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class ToolResult:
    id: str
    content: str
    is_error: bool = False


@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class ChatResponse:
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str            # end_turn | tool_use | max_tokens | refusal
    usage: LLMUsage
    latency_ms: float
    model: str                  # 实际提供服务的模型，取自响应而不是请求


@dataclass
class Completion:
    text: str
    usage: LLMUsage
    latency_ms: float
    model: str
    stop_reason: str = "end_turn"


class InfraError(Exception):
    """基础设施故障（限流、5xx、网络、超时）。

    调度器据此重试；重试耗尽后这次 trial 记为 error，不进入通过率的分母。
    把这类失败记成 0 分，等于把管道的问题算到模型头上。
    """

    def __init__(self, error_class: str, message: str, retryable: bool = True):
        super().__init__(message)
        self.error_class = error_class
        self.retryable = retryable


class Conversation(Protocol):
    async def send_user(self, text: str) -> ChatResponse: ...
    async def send_tool_results(self, results: list[ToolResult]) -> ChatResponse: ...


class ChatProvider(Protocol):
    name: str
    model: str

    def start(self, system: str, tools: list[dict[str, Any]], seed: int = 0) -> Conversation: ...
    async def complete(self, system: str, user: str,
                       json_schema: Optional[dict[str, Any]] = None) -> Completion: ...


# 每百万 token 的美元价格：(输入, 输出, 缓存读)。缓存写按输入价的 1.25 倍估算。
# Claude 价格取自官方定价表（2026-11）；mock 模型给一个虚拟价，让成本维度离线也能跑。
PRICING: dict[str, tuple[float, float, float]] = {
    "claude-fable-5-2": (8.00, 40.00, 0.20),
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "mock-cs-v1": (1.00, 5.00, 0.10),
    "mock-cs-v2": (1.00, 5.00, 0.10),
    "mock-judge-small": (0.10, 0.50, 0.01),
    "mock-judge-large": (4.00, 20.00, 0.20),
}


def register_pricing(model: str, input_per_mtok: float, output_per_mtok: float,
                     cache_read_per_mtok: Optional[float] = None) -> None:
    PRICING[model] = (input_per_mtok, output_per_mtok,
                      input_per_mtok * 0.1 if cache_read_per_mtok is None else cache_read_per_mtok)


def price_of(model: str) -> Optional[tuple[float, float, float]]:
    if model in PRICING:
        return PRICING[model]
    for key, value in PRICING.items():   # 别名解析成带日期的快照名时按前缀匹配
        if model.startswith(key):
            return value
    return None


def cost_usd(model: str, usage: LLMUsage) -> float:
    """成本一律从 API 返回的 usage 和实际服务模型算，不按字符数估。价格未知时返回 0。"""
    price = price_of(model)
    if price is None:
        return 0.0
    inp, out, cache_read = price
    return (usage.input_tokens * inp + usage.output_tokens * out
            + usage.cache_read_tokens * cache_read + usage.cache_write_tokens * inp * 1.25) / 1_000_000
