"""双层报告的 HTML 渲染：上半部分是宏观指标，下半部分是每个 trial 的微观轨迹。

单文件、无外部依赖、无 JS（展开用 <details>），可以直接作为 CI 产物归档。
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from jinja2 import Environment, pass_context

from .aggregate import DIM_LABEL

_env = Environment(autoescape=True, trim_blocks=True, lstrip_blocks=True)
_env.filters["pct"] = lambda v: "—" if v is None else f"{v * 100:.1f}%"
_env.filters["js"] = lambda v: json.dumps(v, ensure_ascii=False, default=str)
_env.filters["ts"] = lambda v: datetime.fromtimestamp(v).strftime("%Y-%m-%d %H:%M:%S") if v else "—"
@pass_context
def _money(ctx, v: float) -> str:
    return f"{ctx.get('sym', '$')}{v:.4f}"


_env.filters["usd"] = _money
_env.filters["ms"] = lambda v: f"{v / 1000:.2f}s"
_env.globals["DIM_LABEL"] = DIM_LABEL
_env.globals["STATE"] = {"pass": "全部通过", "fail": "全部失败", "flaky": "不稳定", "error": "未能评分"}

TEMPLATE = _env.from_string(r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>评测报告 · {{ r.run.suite }} · {{ r.run.agent }}</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1c2330;--mute:#667085;--line:#e3e6ec;--ok:#1a7f4b;--okbg:#e4f5ea;
--bad:#b42318;--badbg:#fde8e6;--warn:#9a6700;--warnbg:#fff3d1;--accent:#2f5bd8;--code:#f1f3f7}
@media (prefers-color-scheme:dark){:root{--bg:#12151b;--card:#1b2029;--ink:#e6e9ef;--mute:#98a2b3;--line:#2c3340;
--ok:#4cc38a;--okbg:#153224;--bad:#ff8a80;--badbg:#3a1a18;--warn:#f0c14b;--warnbg:#3a2f10;--accent:#8aa8ff;--code:#232a36}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.6 -apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
main{max-width:1120px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:32px 0 12px}
.sub{color:var(--mute);font-size:13px}.sub code{background:var(--code);padding:1px 5px;border-radius:4px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-top:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card .k{color:var(--mute);font-size:12px}.card .v{font-size:24px;font-weight:650;font-variant-numeric:tabular-nums}
.card .d{color:var(--mute);font-size:12px}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:12px;color:var(--mute);font-weight:600}tr:last-child td{border-bottom:0}
td.n{font-variant-numeric:tabular-nums;white-space:nowrap}
.bar{height:8px;background:var(--line);border-radius:4px;min-width:120px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--accent)}
.tag{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;background:var(--code);color:var(--mute);margin-right:4px}
.b{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;font-weight:600}
.b.pass{background:var(--okbg);color:var(--ok)}.b.fail,.b.error{background:var(--badbg);color:var(--bad)}
.b.flaky,.b.skip{background:var(--warnbg);color:var(--warn)}.b.gate{background:var(--code);color:var(--accent)}.b.skip{opacity:.7}
.note{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:8px;padding:12px 16px}
.note.bad{border-left-color:var(--bad)}.note.ok{border-left-color:var(--ok)}
details{background:var(--card);border:1px solid var(--line);border-radius:10px;margin-bottom:8px}
details details{margin:8px 12px;background:transparent}
summary{cursor:pointer;padding:10px 14px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
summary .id{font-family:ui-monospace,Consolas,monospace;font-size:13px}
summary .grow{flex:1;min-width:140px}.inner{padding:0 14px 14px}
.tl{border-left:2px solid var(--line);margin:10px 0 0 6px;padding-left:14px}
.ev{margin:8px 0}.ev .who{font-size:12px;color:var(--mute)}
.bubble{background:var(--code);border-radius:8px;padding:8px 10px;white-space:pre-wrap;word-break:break-word}
.ev.user .bubble{background:transparent;border:1px solid var(--line)}
pre{margin:4px 0 0;background:var(--code);border-radius:6px;padding:8px 10px;font:12px/1.5 ui-monospace,Consolas,monospace;
white-space:pre-wrap;word-break:break-all;max-height:240px;overflow:auto}
.chk{list-style:none;margin:6px 0;padding:0}.chk li{padding:2px 0}.chk .x{color:var(--bad)}.chk .o{color:var(--ok)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
.scroll{overflow-x:auto}
</style></head><body><main>

<h1>Agent 评测报告</h1>
<div class="sub">评测集 <code>{{ r.run.suite }}</code>（指纹 <code>{{ r.run.suite_fingerprint }}</code>）
· 被测 <code>{{ r.run.agent }}</code> · 裁判 <code>{{ r.run.judge }}</code>
· 运行 <code>{{ r.run.run_id }}</code>{% if r.run.tag %} · 标签 <code>{{ r.run.tag }}</code>{% endif %}
· {{ r.run.started_at | ts }} · 每条用例 {{ r.run.config.trials }} 次</div>

{% set h = r.headline %}{% set e = r.efficiency %}{% set last = r.reliability[-1] if r.reliability else none %}
<div class="cards">
  <div class="card"><div class="k">通过率（按用例平均）</div><div class="v">{{ h.rate | pct }}</div>
    <div class="d">95% 区间 {{ h.ci_low | pct }} ~ {{ h.ci_high | pct }}</div></div>
  {% if last %}
  <div class="card"><div class="k">可靠性 pass^{{ last.k }}（{{ last.k }} 次全过）</div><div class="v">{{ last.pass_hat_k | pct }}</div>
    <div class="d">pass@{{ last.k }}（至少过一次）{{ last.pass_at_k | pct }}</div></div>
  {% endif %}
  <div class="card"><div class="k">已评分 / 全部 trial</div><div class="v">{{ h.scored }} / {{ h.n_trials }}</div>
    <div class="d">故障 {{ h.errors }} · 截断 {{ h.truncated }} · 拒答 {{ h.refusals }} · 重试过 {{ h.retried }}</div></div>
  <div class="card"><div class="k">P95 延迟</div><div class="v">{{ e.latency_p95_ms | ms }}</div>
    <div class="d">P50 {{ e.latency_p50_ms | ms }} · 平均 {{ "%.1f" | format(e.steps_mean) }} 步</div></div>
  <div class="card"><div class="k">单次成本</div><div class="v">{{ e.cost_per_trial_usd | usd }}</div>
    <div class="d">被测合计 {{ e.cost_total_usd | usd }} · 裁判 {{ e.judge_cost_usd | usd }}</div></div>
</div>

{% if r.baseline %}{% set b = r.baseline %}
<h2>与基线对比</h2>
<div class="note {{ 'bad' if b.significant and b.delta < 0 else 'ok' if b.significant and b.delta > 0 else '' }}">
  基线 <code>{{ b.run_id }}</code>（{{ b.agent }}）{{ b.base_rate | pct }} → 本次 {{ b.cand_rate | pct }}，
  差值 <b>{{ "%+.1f" | format(b.delta * 100) }} 个百分点</b>（95% 区间 {{ "%+.1f" | format(b.ci_low * 100) }} ~ {{ "%+.1f" | format(b.ci_high * 100) }}，
  {{ b.n_cases }} 条共同用例配对比较）。
  {% if b.significant %}区间不含 0，差异<b>显著</b>。{% else %}区间包含 0，差异<b>在噪声范围内</b>，不能据此下结论。{% endif %}
  {% if not b.comparable %}<br>⚠ 两次运行的评测集指纹不同，分数不可直接比较。{% endif %}
  {% if b.regressions %}<br>变差的用例（{{ b.regressions | length }}）：{% for x in b.regressions %}<span class="tag">{{ x.case_id }} {{ x.base | pct }}→{{ x.cand | pct }}</span>{% endfor %}{% endif %}
  {% if b.fixes %}<br>变好的用例（{{ b.fixes | length }}）：{% for x in b.fixes %}<span class="tag">{{ x.case_id }} {{ x.base | pct }}→{{ x.cand | pct }}</span>{% endfor %}{% endif %}
</div>
{% endif %}

<h2>七个维度</h2>
<div class="scroll"><table>
<tr><th>维度</th><th>角色</th><th>通过率</th><th></th><th>95% 区间</th><th>平均得分</th><th>适用 trial</th></tr>
{% for d in r.dimensions %}
<tr><td>{{ d.label }} <span class="sub">{{ d.dimension }}</span></td>
<td>{% if d.gating %}<span class="b gate">门禁</span>{% else %}<span class="tag">诊断</span>{% endif %}</td>
{% if d.n_trials %}<td class="n">{{ d.rate | pct }}</td><td><div class="bar"><i style="width:{{ (d.rate * 100) | round(1) }}%"></i></div></td>
<td class="n">{{ d.ci_low | pct }} ~ {{ d.ci_high | pct }}</td><td class="n">{{ "%.2f" | format(d.mean_score) }}</td><td class="n">{{ d.n_trials }}</td>
{% else %}<td colspan="5" class="sub">未启用或不适用</td>{% endif %}</tr>
{% endfor %}</table></div>

<div class="grid2">
<div><h2>可靠性</h2><table>
<tr><th>k</th><th>pass@k 至少一次</th><th>pass^k 次次都过</th></tr>
{% for x in r.reliability %}<tr><td class="n">{{ x.k }}</td><td class="n">{{ x.pass_at_k | pct }}</td><td class="n">{{ x.pass_hat_k | pct }}</td></tr>{% endfor %}
</table>{% if r.flaky_cases %}<p class="sub">不稳定用例：{% for c in r.flaky_cases %}<span class="tag">{{ c }}</span>{% endfor %}</p>{% endif %}</div>
<div><h2>失败归因</h2>{% if r.failures %}<table>
<tr><th>标签</th><th>trial 数</th><th>涉及用例</th></tr>
{% for f in r.failures[:12] %}<tr><td><code>{{ f.tag }}</code></td><td class="n">{{ f.trials }}</td><td class="n">{{ f.cases }}</td></tr>{% endfor %}
</table>{% else %}<p class="sub">没有失败。</p>{% endif %}
{% if r.error_classes %}<p class="sub">基础设施故障：{% for k, v in r.error_classes.items() %}<span class="tag">{{ k }} × {{ v }}</span>{% endfor %}（不计入通过率）</p>{% endif %}</div>
</div>

<div class="grid2">
<div><h2>按类别</h2><table><tr><th>类别</th><th>通过率</th><th>用例数</th></tr>
{% for s in r.by_category %}<tr><td>{{ s.key }}</td><td class="n">{{ s.rate | pct }}</td><td class="n">{{ s.n_cases }}</td></tr>{% endfor %}</table></div>
<div><h2>按标签（从低到高）</h2><table><tr><th>标签</th><th>通过率</th><th>用例数</th></tr>
{% for s in r.by_tag %}<tr><td>{{ s.key }}</td><td class="n">{{ s.rate | pct }}</td><td class="n">{{ s.n_cases }}</td></tr>{% endfor %}</table></div>
</div>

{% if e.judge_scored %}
<h2>裁判</h2>
<div class="note">LLM 裁判共参与 {{ e.judge_scored }} 次维度评分，实际调用 {{ e.judge_calls }} 次（相同输出命中缓存不重复计费），
其中 {{ e.judge_escalated }} 次升级到更强的模型，{{ e.judge_needs_human }} 次仍拿不准、已进入人工复核队列。裁判成本 {{ e.judge_cost_usd | usd }}，与被测成本分开统计。</div>
{% endif %}

<h2>用例明细（{{ r.cases | length }} 条，失败的排在前面）</h2>
{% for c in r.cases %}
<details><summary><span class="b {{ c.state }}">{{ STATE[c.state] }}</span><span class="id">{{ c.case_id }}</span>
<span class="grow">{{ c.title }}</span>{% for t in c.tags %}<span class="tag">{{ t }}</span>{% endfor %}
<span class="n">{{ c.passes }}/{{ c.scored }}</span></summary><div class="inner">
{% for t in c.trials %}
<details><summary><span class="b {{ 'error' if t.status == 'error' else 'pass' if t.passed else 'fail' }}">第 {{ t.trial + 1 }} 次 · {{ '故障' if t.status == 'error' else '截断' if t.status == 'truncated' else '通过' if t.passed else '失败' }}</span>
{% for s in t.scores %}<span class="b {{ 'skip' if s.skipped else 'pass' if s.passed else 'fail' }}" title="{{ s.dimension }}">{{ DIM_LABEL[s.dimension] }}</span>{% endfor %}
{% if t.attempts > 1 %}<span class="tag">尝试 {{ t.attempts }} 次</span>{% endif %}
{% if t.trace %}<span class="sub">{{ t.trace.tool_calls | length }} 步 · {{ t.trace.usage.input_tokens + t.trace.usage.output_tokens }} tok · {{ t.trace.latency_ms | ms }} · {{ t.trace.usage.cost_usd | usd }}</span>{% endif %}
</summary><div class="inner">
{% if t.status == 'error' %}<p><b>{{ t.error_class }}</b>：{{ t.error_message }}</p>{% endif %}
{% for s in t.scores if not s.skipped %}
<div><b>{{ DIM_LABEL[s.dimension] }}</b> <span class="sub">得分 {{ "%.2f" | format(s.score) }} · {{ {'code': '代码判定', 'llm': 'LLM 裁判', 'hybrid': '代码 + LLM'}[s.grader] }}</span>
<ul class="chk">{% for k in s.checks %}<li><span class="{{ 'o' if k.passed else 'x' }}">{{ '✓' if k.passed else '✗' }}</span> {{ k.detail }}</li>{% endfor %}</ul></div>
{% endfor %}
{% if t.trace %}<div class="tl">
{% for ev in t.trace.timeline %}
{% if ev.kind == 'user' %}<div class="ev user"><div class="who">用户</div><div class="bubble">{{ ev.content }}</div></div>
{% elif ev.kind == 'assistant' %}<div class="ev"><div class="who">Agent</div><div class="bubble">{{ ev.content or '（空回复）' }}</div></div>
{% elif ev.kind == 'tool' %}<div class="ev"><div class="who">工具 · 第 {{ ev.step }} 步 · {{ ev.latency_ms | round | int }} ms{% if ev.error %} · <span class="b fail">{{ ev.error }}</span>{% endif %}</div>
<pre>{{ ev.name }}({{ ev.args | js }})
→ {{ ev.result | js }}</pre></div>
{% elif ev.kind == 'chat' %}<div class="ev"><div class="who">模型调用 · {{ ev.model }} · 入 {{ ev.input_tokens }} / 出 {{ ev.output_tokens }} tok · {{ ev.latency_ms | round | int }} ms · {{ ev.stop_reason }}</div></div>
{% endif %}{% endfor %}</div>{% endif %}
</div></details>
{% endfor %}
</div></details>
{% endfor %}
</main></body></html>
""")


def render_html(report: dict[str, Any]) -> str:
    return TEMPLATE.render(r=report, sym=currency_symbol(report))


def currency_symbol(report: dict[str, Any]) -> str:
    return "¥" if report["run"]["config"].get("currency", "USD").upper() in ("CNY", "RMB") else "$"
