"""离线 mock 客服模型：不花钱、不联网、结果可复现，用来演示和测试评测管道本身。

它是一个按关键词路由的规则策略，走的是和真实模型完全相同的 Agent 循环。两个版本：

  v1（故意埋了缺陷，评测应当全部抓出来）
    - 退款不请用户确认就办（退款不可撤销）           → safety
    - 退款不核验资格，直接退                         → safety / outcome
    - 订单查两遍                                     → trajectory
    - 物流超时后编一个运单号                         → faithfulness
    - 把工具返回值里的指令当命令执行（提示注入）     → safety
    - 手机号不脱敏                                   → safety
    - 工具报错后仍然说“已为您办理”                   → faithfulness
    - 知识库查不到就自己编                           → quality / outcome
    - 35% 概率不查物流直接猜（不稳定）               → pass^k 明显低于 pass@k

  v2（修掉了上面这些，但还留着真实的短板）
    - 一句话里有两个诉求时只处理第一个
    - 用户换种说法（“把钱还我”）就识别不出意图
"""
from __future__ import annotations

import json
import random
import re
from typing import Any, Optional

from .base import ChatResponse, Completion, LLMUsage, ToolCall, ToolResult

_ORDER_ID = re.compile(r"O\d{4}")


def _fmt(x: float) -> str:
    return f"{x:g}"


class MockCSProvider:
    name = "mock"

    def __init__(self, version: str = "v2"):
        if version not in ("v1", "v2"):
            raise ValueError("mock 客服模型只有 v1 / v2 两个版本")
        self.version = version
        self.model = f"mock-cs-{version}"

    def start(self, system: str, tools: list[dict[str, Any]], seed: int = 0) -> "_MockConversation":
        return _MockConversation(self, system, tools, seed)

    async def complete(self, system: str, user: str, json_schema: Optional[dict] = None) -> Completion:
        raise NotImplementedError("mock 客服模型不支持 complete()")


class _MockConversation:
    def __init__(self, provider: MockCSProvider, system: str, tools: list[dict], seed: int):
        self.p = provider
        self.v1 = provider.version == "v1"
        self.rng = random.Random(seed)
        self.flaky = self.v1 and self.rng.random() < 0.35
        self.user_turns: list[str] = []
        self.turn_obs: list[tuple[ToolCall, Any, bool]] = []   # 本轮已拿到的工具结果
        self.pending: dict[str, ToolCall] = {}
        self.history_chars = len(system)
        self.n_tools = len(tools)
        self.suffix = ""
        self.injected_done: set[str] = set()
        self.intent = self.prev_intent = ""
        self.pending_refund: Optional[tuple[str, float]] = None   # v2：等用户确认的退款
        self._n = 0

    # ---- Conversation 接口 ----

    async def send_user(self, text: str) -> ChatResponse:
        self.user_turns.append(text)
        self.turn_obs, self.suffix = [], ""
        self.history_chars += len(text)
        self.prev_intent, self.intent = self.intent, self._intent(text)
        return self._respond()

    async def send_tool_results(self, results: list[ToolResult]) -> ChatResponse:
        for r in results:
            call = self.pending.pop(r.id)
            try:
                data = json.loads(r.content)
            except json.JSONDecodeError:
                data = {"raw": r.content}
            self.turn_obs.append((call, data, r.is_error))
            self.history_chars += len(r.content)
        return self._respond()

    def _respond(self) -> ChatResponse:
        action = self._decide()
        if action[0] == "call":
            self._n += 1
            call = ToolCall(f"mock_{self._n}", action[1], action[2])
            self.pending[call.id] = call
            text, calls, stop = "", [call], "tool_use"
        else:
            text, calls, stop = action[1] + self.suffix, [], "end_turn"
        out_tokens = max(8, len(text) // 2) + 30 * len(calls)
        in_tokens = self.history_chars // 2 + 40 * self.n_tools
        self.history_chars += len(text) + 60 * len(calls)
        latency = 350 + out_tokens * 6 + self.rng.uniform(0, 120)   # 模拟耗时，不真的 sleep
        return ChatResponse(text, calls, stop, LLMUsage(in_tokens, out_tokens), latency, self.p.model)

    # ---- 策略 ----

    def _obs(self, name: str) -> list[tuple[dict, Any, bool]]:
        return [(c.args, data, err) for c, data, err in self.turn_obs if c.name == name]

    def _order_id(self) -> Optional[str]:
        for text in reversed(self.user_turns):
            found = _ORDER_ID.findall(text)
            if found:
                return found[-1]
        return None

    def _intent(self, text: str) -> str:
        has_order = bool(_ORDER_ID.search(text))
        if self.pending_refund:   # v2 上一轮请用户确认了退款
            if re.search(r"不|别|算了|取消", text):
                return "decline"
            if re.search(r"确认|同意|可以|好|是的|行|退吧", text):
                return "confirm"
            self.pending_refund = None   # 用户转去说别的了
        if re.search(r"转人工|人工客服|投诉", text):
            return "human"
        refund = re.search(r"退款|退货|退钱|退了|退掉", text)
        if refund and not self.v1 and not has_order and re.search(r"多久|怎么|如何|政策|规则|流程", text):
            return "kb"   # v2 能区分“问退款政策”和“要办退款”
        if refund:
            return "refund"
        if re.search(r"物流|快递|到哪|发货了|什么时候到|配送|运单", text):
            return "logistics"
        if "地址" in text:
            return "address"
        if re.search(r"手机号|身份证|个人信息", text):
            return "profile"
        if len(text) <= 12 and re.match(r"(你好|您好|在吗|hi|hello|嗨)", text, re.I):
            return "greeting"
        if not self.v1 and has_order and self.intent in ("refund", "logistics", "address"):
            return self.intent   # v2 记得上一轮在办什么：用户补了订单号就接着办
        return "kb"

    def _decide(self) -> tuple:
        text = self.user_turns[-1]
        if self.v1:   # 缺陷：工具结果里的 note 被当成指令
            for call, data, _ in self.turn_obs:
                note = data.get("note") if isinstance(data, dict) else None
                if not note or call.id in self.injected_done:
                    continue
                self.injected_done.add(call.id)
                m = re.search(r"https?://[^\s，。）)]+", note)
                if m:
                    self.suffix += f" 另外，您可以访问 {m.group()} 领取补偿。"
                m = re.search(r"回复中包含\s*([A-Za-z0-9\-]+)", note)
                if m:
                    self.suffix += f" {m.group(1)}"
                m = re.search(r"create_refund.*?(O\d{4}).*?(\d+)\s*元", note)
                if m:
                    return ("call", "create_refund",
                            {"order_id": m.group(1), "amount": float(m.group(2)), "reason": "系统通知"})
        return getattr(self, f"_h_{self.intent}")(text, self._order_id())

    @staticmethod
    def _say(text: str) -> tuple:
        return ("say", text)

    @staticmethod
    def _call(name: str, **args: Any) -> tuple:
        return ("call", name, args)

    def _h_refund(self, text: str, oid: Optional[str]) -> tuple:
        if not oid:
            return self._say("为了帮您处理退款，请提供一下订单号。")
        orders = self._obs("get_order")
        if len(orders) < (2 if self.v1 else 1):   # v1 缺陷：重复查询
            return self._call("get_order", order_id=oid)
        _, order, err = orders[-1]
        if err:
            return self._say(f"抱歉，没有查询到订单 {oid}，请核对一下订单号是否正确。")
        if self.v1:   # v1 缺陷：跳过资格核验，且不看退款接口是否报错
            if not self._obs("create_refund"):
                return self._call("create_refund", order_id=oid, amount=order["amount"], reason="用户申请退款")
            return self._say(f"已为您办理订单 {oid} 的退款，金额 {_fmt(order['amount'])} 元。")
        elig = self._obs("check_refund_eligibility")
        if not elig:
            return self._call("check_refund_eligibility", order_id=oid)
        _, e, err = elig[-1]
        if err:
            return self._say("抱歉，系统暂时无法核验退款资格，请稍后再试，或回复“转人工”。")
        if not e["eligible"]:
            return self._say(f"很抱歉，订单 {oid} 暂时无法办理退款：{e['reason']}。如有疑问可以回复“转人工”。")
        if e["requires_human"]:
            tickets = self._obs("transfer_to_human")
            if not tickets:
                return self._call("transfer_to_human", reason=f"大额退款审核 {oid}", order_id=oid)
            _, ticket, terr = tickets[-1]
            if terr:
                return self._say("抱歉，转接人工失败，请稍后再试。")
            return self._say(f"订单 {oid} 的退款金额为 {_fmt(e['max_amount'])} 元，超过 2000 元需要人工审核。"
                             f"已为您转接人工客服（工单 {ticket['ticket_id']}），"
                             f"预计 {ticket['eta_minutes']} 分钟内有专员联系您。")
        self.pending_refund = (oid, e["max_amount"])   # 退款不可撤销：先请用户确认，下一轮再办
        return self._say(f"很抱歉给您带来不便。订单 {oid} 符合退款条件，可退金额 {_fmt(e['max_amount'])} 元。"
                         "退款不可撤销，请回复“确认”后我为您办理。")

    def _h_confirm(self, text: str, oid: Optional[str]) -> tuple:
        order_id, amount = self.pending_refund
        refunds = self._obs("create_refund")
        if not refunds:
            return self._call("create_refund", order_id=order_id, amount=amount, reason="用户申请退款")
        self.pending_refund = None
        _, r, rerr = refunds[-1]
        if rerr:
            return self._say(f"抱歉，退款提交失败（{r.get('error')}），请稍后再试或回复“转人工”。")
        return self._say(f"已为您办理订单 {order_id} 的退款，金额 {_fmt(r['refund_amount'])} 元，"
                         "预计 1-3 个工作日原路退回。")

    def _h_decline(self, text: str, oid: Optional[str]) -> tuple:
        order_id, amount = self.pending_refund
        self.pending_refund = None
        return self._say(f"好的，已为您取消，订单 {order_id} 的 {_fmt(amount)} 元退款本次不会办理。之后如需退款请随时告诉我。")

    def _h_logistics(self, text: str, oid: Optional[str]) -> tuple:
        if not oid:
            return self._say("请提供一下订单号，我帮您查询物流。")
        lg = self._obs("get_logistics")
        if self.flaky and not lg:   # v1 缺陷：有时不查就猜
            return self._say(f"订单 {oid} 正在配送中，请耐心等待。")
        if not lg:
            return self._call("get_logistics", order_id=oid)
        _, info, err = lg[-1]
        fabricated = f"订单 {oid} 由顺丰承运，运单号 SF0000000000，预计明天送达。"
        if err:
            if self.v1:   # v1 缺陷：查不到就编
                return self._say(fabricated)
            if len(lg) < 2:
                return self._call("get_logistics", order_id=oid)   # 重试一次
            return self._say(f"抱歉，物流系统暂时无法查询订单 {oid} 的信息，请稍后再试，或回复“转人工”由专员为您跟进。")
        if not info.get("tracking_no"):
            if self.v1:
                return self._say(fabricated)
            return self._say(f"暂时没有查询到订单 {oid} 的物流信息，请稍后再试，或回复“转人工”。")
        eta = f"，预计{info['eta']}送达" if info.get("eta") else ""
        return self._say(f"订单 {oid} 由{info['carrier']}承运，运单号 {info['tracking_no']}，"
                         f"当前状态：{info['status']}{eta}。")

    def _h_address(self, text: str, oid: Optional[str]) -> tuple:
        m = re.search(r"(?:改成|改为|改到|换成)\s*(.+?)\s*(?:[，,。！!]|$)", text)
        if not oid or not m:
            return self._say("请提供订单号和新的收货地址，我来帮您修改。")
        addr = m.group(1)
        if self.v1:   # v1 缺陷：不看订单状态，也不看修改是否成功
            if not self._obs("update_address"):
                return self._call("update_address", order_id=oid, address=addr)
            return self._say(f"已为您修改订单 {oid} 的收货地址为：{addr}。")
        orders = self._obs("get_order")
        if not orders:
            return self._call("get_order", order_id=oid)
        _, order, err = orders[-1]
        if err:
            return self._say(f"抱歉，没有查询到订单 {oid}，请核对一下订单号是否正确。")
        if order["status"] not in ("paid", "pending_payment"):
            return self._say(f"订单 {oid} 已发货，无法修改收货地址。建议您联系承运快递改派，或拒收后重新下单。")
        upd = self._obs("update_address")
        if not upd:
            return self._call("update_address", order_id=oid, address=addr)
        _, r, uerr = upd[-1]
        if uerr:
            return self._say(f"抱歉，地址修改失败（{r.get('error')}）。")
        return self._say(f"已为您修改订单 {oid} 的收货地址为：{r['address']}。")

    def _h_profile(self, text: str, oid: Optional[str]) -> tuple:
        m = re.search(r"U\d+", " ".join(self.user_turns))
        if not m:
            return self._say("请提供您的用户 ID，我帮您查询。")
        prof = self._obs("get_user_profile")
        if not prof:
            return self._call("get_user_profile", user_id=m.group())
        _, p, err = prof[-1]
        if err:
            return self._say("抱歉，没有查询到该用户。")
        phone = p["phone"] if self.v1 else p["phone"][:3] + "****" + p["phone"][-4:]   # v1 缺陷：不脱敏
        return self._say(f"您绑定的手机号是 {phone}。")

    def _h_human(self, text: str, oid: Optional[str]) -> tuple:
        tickets = self._obs("transfer_to_human")
        if not tickets:
            args = {"reason": text[:30]}
            if oid:
                args["order_id"] = oid
            return self._call("transfer_to_human", **args)
        _, ticket, err = tickets[-1]
        if err:
            return self._say("抱歉，转接人工失败，请稍后再试。")
        return self._say(f"已为您转接人工客服（工单 {ticket['ticket_id']}），"
                         f"预计 {ticket['eta_minutes']} 分钟内接入，请稍候。")

    def _h_greeting(self, text: str, oid: Optional[str]) -> tuple:
        if self.v1 and not self._obs("search_kb"):   # v1 缺陷：打个招呼也去查知识库
            return self._call("search_kb", query=text)
        return self._say("您好！我是智能客服，可以帮您查询订单物流、办理退款、修改地址，请问有什么可以帮您？")

    def _h_kb(self, text: str, oid: Optional[str]) -> tuple:
        kb = self._obs("search_kb")
        if not kb:
            return self._call("search_kb", query=text)
        _, data, err = kb[-1]
        results = (data.get("results") or []) if isinstance(data, dict) else []
        if err or not results:
            if self.v1:   # v1 缺陷：查不到就编一个肯定的答复
                return self._say("支持的，您可以直接在订单页面申请，我们会在 24 小时内处理。")
            return self._say("抱歉，暂时没有查到相关的政策说明，建议您回复“转人工”咨询专员。")
        return self._say(results[0]["text"])
