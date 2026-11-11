"""评测中台的核心数据模型。

三组对象：
  1. 用例（Case）   —— 评什么：用户输入、环境、期望。
  2. 轨迹（Trace）  —— 被测 Agent 实际做了什么：对话、工具调用、终态、消耗。
  3. 评分（Score）  —— 7 个维度各自的原子检查项与结论。

术语沿用业界通行叫法：task/case、trial（一次尝试）、grader/evaluator、transcript/trace、outcome。
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

DIMENSIONS = (
    "outcome",       # 任务结果：环境终态 + 回复关键事实
    "tool_use",      # 工具调用：选对工具、参数正确、参数有出处
    "trajectory",    # 轨迹效率：步数、重复调用、正常收尾
    "faithfulness",  # 忠实度：回复里的实体/动作都有工具结果支撑
    "quality",       # 回答质量：LLM 裁判按原子 rubric 逐条判
    "safety",        # 安全合规：业务红线、PII、提示注入
    "efficiency",    # 成本与延迟：token、费用、耗时预算
)


# ----------------------------- 用例 -----------------------------

class RubricItem(BaseModel):
    """一条原子评审标准。一条只判一件事，方便校准，也方便定位。"""
    id: str
    text: str
    # 下面两个字段只给离线 mock 裁判用（没有 API Key 时让流水线能跑通），真实 LLM 裁判不看。
    hints: list[str] = Field(default_factory=list)
    anti_hints: list[str] = Field(default_factory=list)


class ToolExpectation(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)  # 子集匹配


class StateAssertion(BaseModel):
    """对沙箱终态的断言。优先级：unchanged > exists > equals。"""
    path: str                      # 点分路径，如 refunds.O1001.amount
    equals: Any = None
    exists: Optional[bool] = None
    unchanged: bool = False        # 与初始状态相同（“不该动的没动”）


class ToolsExpect(BaseModel):
    required: list[ToolExpectation] = Field(default_factory=list)
    forbidden: list[str] = Field(default_factory=list)
    order: list[str] = Field(default_factory=list)   # 这些工具首次出现的相对顺序
    max_calls: Optional[int] = None                  # 0 = 这个场景不该调任何工具


class AnswerExpect(BaseModel):
    must_include: list[str] = Field(default_factory=list)      # "a|b" 表示任一即可
    must_not_include: list[str] = Field(default_factory=list)


class Budget(BaseModel):
    max_steps: Optional[int] = None
    max_tokens: Optional[int] = None
    max_latency_ms: Optional[float] = None
    max_cost_usd: Optional[float] = None

    def merged(self, override: "Budget") -> "Budget":
        data = self.model_dump()
        data.update({k: v for k, v in override.model_dump().items() if v is not None})
        return Budget(**data)


class Expect(BaseModel):
    final_state: list[StateAssertion] = Field(default_factory=list)
    tools: ToolsExpect = Field(default_factory=ToolsExpect)
    answer: AnswerExpect = Field(default_factory=AnswerExpect)
    budget: Budget = Field(default_factory=Budget)
    policies: list[str] = Field(default_factory=list)
    rubric: list[RubricItem] = Field(default_factory=list)


class Fault(BaseModel):
    """故障注入：让某个工具的前 N 次调用失败。"""
    tool: str
    kind: Literal["timeout", "error", "empty"] = "error"
    times: int = 1
    message: str = ""


class Injection(BaseModel):
    """间接提示注入：把一段指令塞进工具返回值，看 Agent 会不会照做。"""
    tool: str
    text: str
    canary: list[str] = Field(default_factory=list)  # 一旦出现在回复或工具参数里，说明被注入成功


class EnvSpec(BaseModel):
    overrides: dict[str, Any] = Field(default_factory=dict)
    faults: list[Fault] = Field(default_factory=list)
    injections: list[Injection] = Field(default_factory=list)


class UserSpec(BaseModel):
    turns: list[str]          # 脚本化用户轮次
    persona: str = ""         # 给 LLM 用户模拟器用
    goal: str = ""


class ReferenceTurn(BaseModel):
    tool_calls: list[ToolExpectation] = Field(default_factory=list)
    answer: str = ""


class Variant(BaseModel):
    """同一期望下的另一种问法，用来测措辞鲁棒性。"""
    suffix: str
    turns: list[str]


class Case(BaseModel):
    id: str
    title: str
    tags: list[str] = Field(default_factory=list)
    category: Literal["capability", "regression", "safety"] = "capability"
    source: Literal["human", "production", "synthetic"] = "human"
    user: UserSpec
    env: EnvSpec = Field(default_factory=EnvSpec)
    expect: Expect = Field(default_factory=Expect)
    reference: list[ReferenceTurn] = Field(default_factory=list)  # 参考解，用于 harness 自检
    variants: list[Variant] = Field(default_factory=list)
    notes: str = ""

    @field_validator("reference", mode="before")
    @classmethod
    def _single_turn_shorthand(cls, v: Any) -> Any:
        return [v] if isinstance(v, dict) else v


# ----------------------------- 轨迹 -----------------------------

class ToolCallRecord(BaseModel):
    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    error: Optional[str] = None   # timeout | tool_error | unknown_tool | invalid_args
    latency_ms: float = 0.0
    turn: int = 0
    step: int = 0


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    llm_calls: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class Trace(BaseModel):
    trace_id: str
    case_id: str
    agent: str
    model: str
    user_messages: list[str] = Field(default_factory=list)
    answers: list[str] = Field(default_factory=list)       # 每个用户轮次对应的最终回复
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    timeline: list[dict[str, Any]] = Field(default_factory=list)  # 按时间排好的完整过程，报告直接渲染
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0
    stop_reason: str = "end_turn"   # end_turn | max_steps | max_tokens | refusal
    initial_state: dict[str, Any] = Field(default_factory=dict)
    final_state: dict[str, Any] = Field(default_factory=dict)

    @property
    def final_answer(self) -> str:
        return self.answers[-1] if self.answers else ""


# ----------------------------- 评分 -----------------------------

class Check(BaseModel):
    id: str
    passed: bool
    detail: str = ""
    tag: str = ""      # 失败归因标签，Badcase 聚类用


class JudgeMeta(BaseModel):
    tier: int = 0              # 最终采信的是第几级裁判（0 = 规则直接判）
    model: str = ""
    escalated: bool = False
    needs_human: bool = False
    calls: int = 0
    cost_usd: float = 0.0


class Score(BaseModel):
    dimension: str
    score: float = 0.0         # 0~1，通过的检查项占比
    passed: bool = False
    grader: Literal["code", "llm", "hybrid"] = "code"
    checks: list[Check] = Field(default_factory=list)
    reason: str = ""
    skipped: bool = False      # 该用例不适用此维度（不计入统计）
    judge: Optional[JudgeMeta] = None


class TrialResult(BaseModel):
    run_id: str
    case_id: str
    trial: int
    # scored：正常产出并已打分；truncated：被 max_tokens 截断，展示但不计入均值；
    # error：基础设施故障，没有可评分的输出——绝不能当成模型失败记 0 分。
    status: Literal["scored", "truncated", "error"] = "scored"
    error_class: Optional[str] = None
    error_message: str = ""
    attempts: int = 1
    seed: int = 0
    passed: bool = False
    scores: list[Score] = Field(default_factory=list)
    failure_tags: list[str] = Field(default_factory=list)
    trace: Optional[Trace] = None
