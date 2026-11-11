"""读取环境变量，兼顾 Windows 上“改了系统环境变量但终端没重开”的情况。

Windows 的进程只在启动时继承一次环境变量。在“系统属性 → 环境变量”里新加的值，
已经打开的终端读不到，必须重开窗口。为了少踩这个坑，这里按以下顺序查找：

  1. 当前进程的环境变量（终端里 $env:XXX 临时设置的、或启动前就已存在的）
  2. Windows 注册表里保存的用户环境变量    HKCU\\Environment
  3. Windows 注册表里保存的系统环境变量    HKLM\\...\\Session Manager\\Environment

非 Windows 系统只查第 1 步。值只在内存里使用，不打印、不落盘。
"""
from __future__ import annotations

import os
import sys
from typing import Optional

_USER_KEY = r"Environment"
_MACHINE_KEY = r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"


def _registry_value(name: str) -> Optional[str]:
    if sys.platform != "win32":
        return None
    import winreg

    for root, path in ((winreg.HKEY_CURRENT_USER, _USER_KEY), (winreg.HKEY_LOCAL_MACHINE, _MACHINE_KEY)):
        try:
            with winreg.OpenKey(root, path) as key:
                value, kind = winreg.QueryValueEx(key, name)
        except OSError:
            continue
        if kind == winreg.REG_EXPAND_SZ:
            value = os.path.expandvars(value)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def read_env(name: str) -> Optional[str]:
    value = os.environ.get(name)
    if value and value.strip():
        return value.strip()
    return _registry_value(name)


class MissingCredential(RuntimeError):
    """缺少必需的环境变量。CLI 捕获后只打印说明，不打印堆栈。"""


def require_env(*names: str, hint: str) -> str:
    """按顺序取第一个有值的变量；都没有时报错。多个名字用于兼容不同厂商的习惯命名。"""
    for name in names:
        value = read_env(name)
        if value:
            return value
    listed = " 或 ".join(names)
    raise MissingCredential(
        f"没有找到环境变量 {listed}。{hint}\n"
        f"设置方法：Win 键搜索“编辑账户的环境变量”，在“用户变量”里新建 {names[0]}；"
        f"或在当前终端执行 $env:{names[0]} = \"...\"")
