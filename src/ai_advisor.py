# -*- coding: utf-8 -*-
"""
AI 综合结论模块。

- 有 API Key：调用 OpenAI 兼容接口（OpenAI / DeepSeek / 豆包方舟 / 通义 / Kimi 等），
  把各估值结果打包成结构化提示词，生成综合研判。
- 无 API Key：使用本地规则引擎生成结构化综合结论（不依赖外部服务）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PROVIDERS = {
    "OpenAI": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "DeepSeek": {"base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
    "豆包·火山方舟": {"base_url": "https://ark.cn-beijing.volces.com/api/v3", "model": "doubao-1-5-pro-32k-250115"},
    "通义千问": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-plus"},
    "Kimi(Moonshot)": {"base_url": "https://api.moonshot.cn/v1", "model": "moonshot-v1-8k"},
}


def _fmt_currency(v: float) -> str:
    if not isinstance(v, float) and not isinstance(v, int):
        return "N/A"
    if abs(v) >= 1e8:
        return f"{v/1e8:.1f}亿"
    if abs(v) >= 1e4:
        return f"{v/1e4:.1f}万"
    return f"{v:,.0f}"


def _pct(v: float | None) -> str:
    if v is None or not isinstance(v, (int, float)) or not __import__("math").isfinite(v):
        return "N/A"
    return f"{v:.1%}"


def build_prompt(cd, results: dict[str, Any], extra: dict | None = None) -> str:
    """把估值结果打包成提示词。results: {dcf, ddm, reverse, comps}"""
    lines = []
    lines.append("你是一名专业的股票估值分析师。请基于以下自动计算的估值结果，")
    lines.append("用中文写一段300-500字的综合分析，结构如下：")
    lines.append("1. 公司基本面一句话概况；")
    lines.append("2. 各估值方法的结论汇总与差异解释；")
    lines.append("3. 当前股价所处的位置与安全边际判断；")
    lines.append("4. 最关键的假设与风险点（哪些假设变了结论会反转）；")
    lines.append("5. 给出明确的操作参考倾向（注意：仅作研究参考，不构成投资建议）。")
    lines.append("")
    lines.append(f"【公司】{cd.name} ({cd.symbol})  市场: {'A股' if cd.market=='A' else '美股'}  币种: {cd.currency}")
    lines.append(f"当前股价: {cd.latest_price():,.2f}   总市值: {_fmt_currency(cd.market_cap)}")
    lines.append(f"PE(TTM): {cd.pe_ttm:.1f}   PB: {cd.pb:.2f}   Beta: {cd.beta:.2f}")
    lines.append(f"无风险利率: {cd.rf:.2%}   股权风险溢价: {cd.erp:.2%}")
    lines.append("")

    dcf = results.get("dcf")
    if dcf and getattr(dcf, "per_share_value", None) and not getattr(dcf, "error", ""):
        lines.append("【DCF 现金流折现】")
        lines.append(f"每股内在价值: {dcf.per_share_value:,.2f}  相对现价: {_pct(dcf.upside)}  结论: {dcf.conclusion}")
        lines.append(f"WACC: {dcf.wacc:.2%}  永续增长率: {dcf.assumptions.get('terminal_growth', 0):.2%}")
        lines.append(f"显式期首年收入增速: {dcf.assumptions.get('revenue_growth', 0):.2%}  营业利润率: {dcf.assumptions.get('operating_margin', 0):.2%}")
        lines.append(f"终值占企业价值比例: {_pct(dcf.terminal_ratio)}")
        lines.append("")

    ddm = results.get("ddm")
    if ddm and getattr(ddm, "per_share_value", None) and not getattr(ddm, "error", ""):
        lines.append("【DDM 股利贴现】")
        lines.append(f"每股内在价值: {ddm.per_share_value:,.2f}  相对现价: {_pct(ddm.upside)}  结论: {ddm.conclusion}")
        lines.append(f"当前每股股利: {ddm.dps_now:,.3f}  股利增速: {_pct(ddm.dividend_growth)}  股权成本: {ddm.re:.2%}")
        lines.append("")

    rev = results.get("reverse_dcf")
    if rev and not getattr(rev, "error", ""):
        lines.append("【反向 DCF】")
        if rev.mode == "growth":
            lines.append(f"市场隐含显式期首年收入增长率: {_pct(rev.implied_growth)}  (公司历史增长: {_pct(rev.current_growth)})")
        else:
            lines.append(f"市场隐含永续增长率: {_pct(rev.implied_terminal_growth)}")
        lines.append(rev.detail)
        lines.append("")

    comps = results.get("comps")
    if comps and not getattr(comps, "error", ""):
        lines.append("【可比公司法】")
        lines.append(f"可比公司数: {len(comps.peer_table) if comps.peer_table is not None else 0}")
        lines.append(f"PE法: {_fmt_currency(comps.per_share_pe)}({_pct(comps.upside_pe)})  PB法: {_fmt_currency(comps.per_share_pb)}({_pct(comps.upside_pb)})")
        if comps.per_share_ev_ebitda and not __import__("math").isnan(comps.per_share_ev_ebitda):
            lines.append(f"EV/EBITDA法: {_fmt_currency(comps.per_share_ev_ebitda)}({_pct(comps.upside_ev_ebitda)})")
        lines.append("")

    if extra:
        lines.append("【补充说明】")
        lines.append(extra.get("notes", ""))
    lines.append("")
    lines.append("注意：这是研究参考工具输出，务必在最后附上'以上不构成投资建议'。")
    return "\n".join(lines)


def call_ai(prompt: str, provider: str, api_key: str, model: str | None = None,
            base_url: str | None = None) -> str:
    """调用 OpenAI 兼容接口生成分析。"""
    from openai import OpenAI
    p = PROVIDERS.get(provider, {})
    client = OpenAI(
        api_key=api_key,
        base_url=base_url or p.get("base_url"),
    )
    resp = client.chat.completions.create(
        model=model or p.get("model", ""),
        messages=[
            {"role": "system", "content": "你是资深股票估值分析师，输出专业、客观、结构化的中文分析。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=1600,
    )
    return resp.choices[0].message.content


def rule_based_summary(cd, results: dict[str, Any]) -> str:
    """无 API Key 时的本地规则引擎综合结论。"""
    lines = []
    lines.append(f"### {cd.name}（{cd.symbol}）综合研判（规则引擎版）")
    lines.append("")
    lines.append(f"- 当前股价 **{cd.latest_price():,.2f} {cd.currency}**，总市值 {_fmt_currency(cd.market_cap)}，PE(TTM) {cd.pe_ttm:.1f}，PB {cd.pb:.2f}。")
    lines.append("")

    rows = []
    dcf = results.get("dcf")
    if dcf and not getattr(dcf, "error", "") and getattr(dcf, "per_share_value", None):
        rows.append(("DCF", dcf.per_share_value, dcf.upside, dcf.conclusion))
    ddm = results.get("ddm")
    if ddm and not getattr(ddm, "error", "") and getattr(ddm, "per_share_value", None):
        rows.append(("DDM", ddm.per_share_value, ddm.upside, ddm.conclusion))
    rev = results.get("reverse_dcf")
    comps = results.get("comps")
    if comps and not getattr(comps, "error", ""):
        vals = [v for v in (comps.per_share_pe, comps.per_share_pb, comps.per_share_ev_ebitda)
                if v is not None and not (isinstance(v, float) and __import__("math").isnan(v))]
        if vals:
            avg = float(sum(vals) / len(vals))
            price = cd.latest_price()
            up = avg / price - 1 if price else None
            rows.append(("可比公司(均值)", avg, up, comps.conclusion))

    if rows:
        lines.append("| 方法 | 每股价值 | 相对现价 | 结论 |")
        lines.append("|---|---|---|---|")
        for name, v, up, conc in rows:
            lines.append(f"| {name} | {v:,.2f} | {_pct(up)} | {conc} |")
        lines.append("")

    # 汇总判断
    valid = [r for r in rows if r[2] is not None]
    if valid:
        upsides = [r[2] for r in valid]
        avg_up = sum(upsides) / len(upsides)
        if avg_up >= 0.15:
            verdict = f"多数方法显示**低估**（平均潜在上行 {_pct(avg_up)}），具备一定的安全边际。"
        elif avg_up <= -0.15:
            verdict = f"多数方法显示**高估**（平均潜在下行 {-avg_up:.1%}），当前价格可能透支了预期。"
        else:
            verdict = f"各方法估值中枢与当前价接近（平均偏差 {_pct(abs(avg_up))}），估值**基本合理**。"
        lines.append(f"**综合结论**：{verdict}")
        lines.append("")

    # 一致性检查
    if len(valid) >= 2:
        maxd = max(r[2] for r in valid)
        mind = min(r[2] for r in valid)
        if maxd - mind > 0.5:
            lines.append("⚠️ **方法间分歧较大**：不同估值方法给出的潜在空间差异超过 50%，"
                         "说明结论对假设高度敏感，建议重点复核增长率与折现率假设。")
        lines.append("")

    dcf = results.get("dcf")
    if dcf and not getattr(dcf, "error", "") and getattr(dcf, "terminal_ratio", None):
        lines.append(f"**关键敏感点**：DCF 中终值占企业价值 {_pct(dcf.terminal_ratio)}，"
                     "意味着远期假设（永续增长率、WACC）对结论影响很大，请审慎对待。")
        lines.append("")
    rev = results.get("reverse_dcf")
    if rev and not getattr(rev, "error", ""):
        lines.append(f"**预期差提示**：{rev.detail}")
        lines.append("")

    lines.append("---")
    lines.append("*以上为规则引擎自动汇总，仅作研究参考，不构成投资建议。接入 AI 接口可获得更深入的综合研判。*")
    return "\n".join(lines)
