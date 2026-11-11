"""领域注册表：一个领域 = 初始状态 + 工具集 + 业务红线策略。接新业务就在这里加一个模块。"""
from __future__ import annotations

from typing import Optional

from ..models import EnvSpec
from . import ecommerce
from .core import Sandbox, ToolError, ToolSpec

DOMAINS = {"ecommerce": ecommerce}


def make_sandbox(domain: str, env: Optional[EnvSpec] = None) -> Sandbox:
    mod = DOMAINS[domain]
    return Sandbox(mod.TOOLS, mod.build_state(), env)


__all__ = ["DOMAINS", "Sandbox", "ToolError", "ToolSpec", "make_sandbox"]
