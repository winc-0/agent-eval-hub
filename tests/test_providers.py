"""真实模型接入层的测试：用官方 SDK 向一个本地假服务器发请求。

不花钱、不联网，但走的是真实 SDK 的序列化与解析，所以能验证：
  - 我们发出去的请求体是否符合预期（参数名、工具定义、缓存断点、thinking 块原样回传）；
  - SDK 返回的对象我们是否解析对了（文本、工具调用、用量、停止原因）；
  - HTTP 错误是否映射成了正确的故障类别；
  - SDK 自身不重试（重试归调度器管）。
验证不了的：真实模型的行为、真实接口对请求体的校验。那需要拿真 key 跑一次。
"""
from __future__ import annotations

import asyncio
import inspect
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from evalhub.agent import LLMAgent
from evalhub.judge import JudgeRequest, LLMJudgeBackend
from evalhub.llm import InfraError, register_pricing
from evalhub.models import RubricItem
from evalhub.scheduler import run_trial
from evalhub.suite import Suite

SUITE_DIR = Path(__file__).resolve().parents[1] / "suites" / "ecommerce_cs"
FINAL = "很抱歉给您带来不便。已为您办理订单 O1001 的退款，金额 299 元，预计 1-3 个工作日原路退回。"
JUDGE_JSON = json.dumps({"reason": "回复里有“很抱歉”", "verdict": "pass", "confidence": 0.9}, ensure_ascii=False)
STEPS = [("check_refund_eligibility", {"order_id": "O1001"}),
         ("create_refund", {"order_id": "O1001", "amount": 299, "reason": "耳机有杂音"})]


class FakeLLMServer:
    """按“已经收到几个工具结果”决定下一步的假模型服务，同时提供 Claude 和 OpenAI 两种接口。"""

    def __init__(self):
        self.requests: list[dict] = []
        self.status = 200
        self.served_model = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "body": body})
                if outer.status != 200:
                    payload = {"type": "error", "error": {"type": "api_error", "message": "boom"}}
                elif self.path.endswith("/messages"):
                    payload = outer.anthropic(body)
                else:
                    payload = outer.openai(body)
                data = json.dumps(payload).encode()
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def anthropic(self, body: dict) -> dict:
        usage = {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 50,
                 "cache_creation_input_tokens": 0}
        base = {"id": "msg_1", "type": "message", "role": "assistant", "stop_sequence": None,
                "model": self.served_model or body["model"], "usage": usage}
        if "format" in body.get("output_config", {}):
            return {**base, "content": [{"type": "text", "text": JUDGE_JSON}], "stop_reason": "end_turn"}
        done = sum(1 for m in body["messages"] if isinstance(m["content"], list)
                   for b in m["content"] if b.get("type") == "tool_result")
        if done < len(STEPS):
            name, args = STEPS[done]
            return {**base, "stop_reason": "tool_use", "content": [
                {"type": "thinking", "thinking": "", "signature": f"sig_{done}"},
                {"type": "text", "text": "我来处理。"},
                {"type": "tool_use", "id": f"toolu_{done}", "name": name, "input": args}]}
        return {**base, "stop_reason": "end_turn", "content": [{"type": "text", "text": FINAL}]}

    def openai(self, body: dict) -> dict:
        def reply(message: dict, finish: str) -> dict:
            return {"id": "c1", "object": "chat.completion", "created": 0, "model": body["model"],
                    "choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", **message}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}
        if "response_format" in body:
            return reply({"content": JUDGE_JSON}, "stop")
        done = sum(1 for m in body["messages"] if m["role"] == "tool")
        if done < len(STEPS):
            name, args = STEPS[done]
            return reply({"content": None, "tool_calls": [{"id": f"call_{done}", "type": "function", "function": {
                "name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]}, "tool_calls")
        return reply({"content": FINAL}, "stop")


def run(coro):
    return asyncio.run(coro)


class ProviderCase(unittest.TestCase):
    ENV: dict[str, str] = {}

    def setUp(self):
        self.server = FakeLLMServer()
        env = {k: v.format(url=self.server.url) for k, v in self.ENV.items()}
        patcher = mock.patch.dict("os.environ", env)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.server.close)
        self.suite = Suite.load(SUITE_DIR)
        self.case = self.suite.case("refund_eligible_basic")

    def bodies(self) -> list[dict]:
        return [r["body"] for r in self.server.requests]

    def assert_trial_passes(self, provider):
        result = run(run_trial(self.suite, LLMAgent(provider), None, self.case, "r", 0, 0))
        self.assertTrue(result.passed, result.failure_tags)
        trace = result.trace
        self.assertEqual([c.name for c in trace.tool_calls], [name for name, _ in STEPS])
        self.assertEqual(trace.final_state["refunds"]["O1001"]["amount"], 299.0)
        self.assertEqual(trace.usage.llm_calls, 3)
        self.assertEqual(trace.usage.output_tokens, 60)
        return trace

    def assert_error(self, provider, status: int, error_class: str, retryable: bool):
        self.server.status, self.server.requests = status, []
        with self.assertRaises(InfraError) as ctx:
            run(provider.complete("s", "u"))
        self.assertEqual((ctx.exception.error_class, ctx.exception.retryable), (error_class, retryable))
        self.assertEqual(len(self.server.requests), 1, "SDK 不应自行重试，重试归调度器管")
        self.server.status = 200


def _anthropic_supported() -> bool:
    try:
        from anthropic.resources.messages import AsyncMessages
    except ImportError:
        return False
    return "output_config" in inspect.signature(AsyncMessages.create).parameters


@unittest.skipUnless(_anthropic_supported(), "anthropic SDK 未安装或版本过旧（需支持 output_config）")
class AnthropicProviderTests(ProviderCase):
    ENV = {"ANTHROPIC_API_KEY": "test-key", "ANTHROPIC_BASE_URL": "{url}"}

    def provider(self, **kwargs):
        from evalhub.llm.anthropic_provider import AnthropicProvider
        return AnthropicProvider(**kwargs)

    def test_agent_loop_end_to_end(self):
        trace = self.assert_trial_passes(self.provider())
        self.assertEqual(trace.model, "claude-opus-5-5")
        self.assertEqual((trace.usage.input_tokens, trace.usage.cache_read_tokens), (300, 150))
        self.assertGreater(trace.usage.cost_usd, 0, "按实际服务模型的价格计费")

        first, second, third = self.bodies()
        self.assertEqual(first["model"], "claude-opus-5-5")
        self.assertEqual(first["output_config"], {"effort": "medium"})
        self.assertEqual(first["system"][0]["cache_control"], {"type": "ephemeral"})
        self.assertIn("input_schema", first["tools"][0])
        for banned in ("thinking", "temperature", "top_p", "fallbacks", "tool_choice"):
            self.assertNotIn(banned, first, f"不应发送 {banned}")
        # thinking 块原样回传；同一轮的工具结果在同一条 user 消息里
        assistant = second["messages"][1]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual([b["type"] for b in assistant["content"]], ["thinking", "text", "tool_use"])
        self.assertEqual(assistant["content"][0]["signature"], "sig_0")
        results = second["messages"][2]["content"]
        self.assertEqual([(b["type"], b["tool_use_id"], b["is_error"]) for b in results],
                         [("tool_result", "toolu_0", False)])
        self.assertTrue(json.loads(results[0]["content"])["eligible"])
        self.assertEqual(len(third["messages"]), 5)

    def test_judge_uses_structured_output(self):
        backend = LLMJudgeBackend(self.provider(model="claude-haiku-5-5", effort="low"))
        v = run(backend.judge(JudgeRequest(RubricItem(id="apology", text="表达了歉意"), FINAL, "用户：坏了", "[]")))
        self.assertEqual((v.verdict, v.confidence, v.model), ("pass", 0.9, "claude-haiku-5-5"))
        self.assertGreater(v.cost_usd, 0)
        body = self.bodies()[0]
        self.assertEqual(body["output_config"]["format"]["type"], "json_schema")
        self.assertEqual(body["output_config"]["effort"], "low")
        self.assertNotIn("tools", body)

    def test_http_errors_map_to_infra_error_classes(self):
        p = self.provider()
        self.assert_error(p, 429, "rate_limited", True)
        self.assert_error(p, 500, "provider_5xx", True)
        self.assert_error(p, 529, "provider_5xx", True)
        self.assert_error(p, 400, "bad_request", False)

    def test_response_from_a_different_model_is_rejected(self):
        self.server.served_model = "claude-sonnet-5-5"
        with self.assertRaises(InfraError) as ctx:
            run(self.provider().complete("s", "u"))
        self.assertEqual(ctx.exception.error_class, "served_model_mismatch")
        self.assertFalse(ctx.exception.retryable)


try:
    import openai  # noqa: F401
except ImportError:
    openai = None


@unittest.skipIf(openai is None, "未安装 openai SDK")
class OpenAICompatProviderTests(ProviderCase):
    ENV = {"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": "{url}/v1"}

    def provider(self, model: str = "deepseek-chat"):
        from evalhub.llm.openai_provider import OpenAICompatProvider
        return OpenAICompatProvider(model)

    def test_agent_loop_end_to_end(self):
        register_pricing("deepseek-chat", 0.5, 2.0)
        trace = self.assert_trial_passes(self.provider())
        self.assertEqual(trace.model, "deepseek-chat")
        self.assertAlmostEqual(trace.usage.cost_usd, (300 * 0.5 + 60 * 2.0) / 1e6)

        first, second, _ = self.bodies()
        self.assertEqual(self.server.requests[0]["path"], "/v1/chat/completions")
        self.assertEqual(first["messages"][0]["role"], "system")
        self.assertEqual(first["tools"][0]["type"], "function")
        self.assertIn("parameters", first["tools"][0]["function"])
        self.assertEqual(second["messages"][2]["tool_calls"][0]["id"], "call_0")
        self.assertEqual((second["messages"][3]["role"], second["messages"][3]["tool_call_id"]), ("tool", "call_0"))

    def test_judge_asks_for_json(self):
        v = run(LLMJudgeBackend(self.provider()).judge(
            JudgeRequest(RubricItem(id="apology", text="表达了歉意"), FINAL)))
        self.assertEqual((v.verdict, v.confidence), ("pass", 0.9))
        body = self.bodies()[0]
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertIn("JSON Schema", body["messages"][0]["content"])

    def test_http_errors_map_to_infra_error_classes(self):
        p = self.provider()
        self.assert_error(p, 429, "rate_limited", True)
        self.assert_error(p, 503, "provider_5xx", True)
        self.assert_error(p, 400, "bad_request", False)

    def test_unknown_model_price_costs_zero_rather_than_guessing(self):
        trace = self.assert_trial_passes(self.provider("some-unpriced-model"))
        self.assertEqual(trace.usage.cost_usd, 0.0)


if __name__ == "__main__":
    unittest.main()
