"""统计：让报告里的每个数字都带着“有多可信”。

要点：
  - 同一条用例的多次 trial 高度相关，所以置信区间按【用例】重采样（聚类 bootstrap），
    按 trial 重采样会把区间算得过窄。
  - pass@k 是“k 次里至少成一次”（最好情况），pass^k 是“k 次全成”（可靠性）。
    单次成功率 75% 时 pass@3 ≈ 98%，pass^3 ≈ 42%——上线后用户遇到的是后者。
  - 两个版本的差异用配对 bootstrap：同一条用例上做差再重采样，区间不含 0 才算显著。
"""
from __future__ import annotations

import random
from math import comb
from typing import Optional, Sequence


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def percentile(xs: Sequence[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """对“每条用例一个数”的序列做 bootstrap，返回均值的 (1-alpha) 置信区间。"""
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    n = len(values)
    means = sorted(mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    return (means[int(n_boot * alpha / 2)], means[min(n_boot - 1, int(n_boot * (1 - alpha / 2)))])


def paired_diff(base: dict[str, float], cand: dict[str, float], n_boot: int = 2000,
                seed: int = 0) -> dict:
    """配对比较：只在两边都有的用例上，按用例做差后 bootstrap。"""
    shared = sorted(set(base) & set(cand))
    diffs = [cand[c] - base[c] for c in shared]
    lo, hi = bootstrap_ci(diffs, n_boot=n_boot, seed=seed)
    return {"n_cases": len(shared), "delta": mean(diffs), "ci_low": lo, "ci_high": hi,
            "significant": bool(diffs) and (lo > 0 or hi < 0)}


def pass_at_k(passes: int, n: int, k: int) -> Optional[float]:
    """n 次里成了 passes 次，随机抽 k 次“至少成一次”的概率（无放回的无偏估计）。"""
    if k > n or n == 0:
        return None
    return 1.0 - comb(n - passes, k) / comb(n, k)


def pass_hat_k(passes: int, n: int, k: int) -> Optional[float]:
    """随机抽 k 次“全部成功”的概率，即 τ-bench 的 pass^k 估计量 C(c,k)/C(n,k)。"""
    if k > n or n == 0:
        return None
    return comb(passes, k) / comb(n, k)


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """两个标注者的一致性，扣除了“碰巧一致”的部分。裁判 vs 人工就用它。"""
    n = len(a)
    if n == 0 or n != len(b):
        return 0.0
    labels = set(a) | set(b)
    observed = sum(x == y for x, y in zip(a, b)) / n
    expected = sum((list(a).count(lab) / n) * (list(b).count(lab) / n) for lab in labels)
    return 1.0 if expected == 1.0 else (observed - expected) / (1.0 - expected)
