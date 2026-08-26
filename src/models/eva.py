# -*- coding: utf-8 -*-
"""
EVA 经济增加值法：价值 = 投入资本 + 未来经济增加值现值。

核心逻辑：公司只有赚取超过资本成本的回报（NOPAT > 投入资本×WACC）才算创造价值。
- EVA = NOPAT − WACC × 投入资本(IC)
- 企业价值 = IC₀ + Σ EVAₜ/(1+WACC)ᵗ + 终值EVA
- 股权价值 = 企业价值 − 净债务 → 每股价值
- 对高 ROIC 的优质公司（茅台/宁德等）比传统 DCF 更能体现"超额回报"；
  金融股 IC 结构失真不适用（综合研判权重 0）
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data_fetcher import CompanyData
from ..wacc import calc_wacc


@dataclass
class EVAResult:
    per_share_value: float = np.nan
    upside: float = np.nan
    wacc: float = np.nan
    ic: float = np.nan          # 投入资本
    nopat: float = np.nan       # 税后营业利润
    eva_now: float = np.nan     # 当年 EVA
    roic: float = np.nan        # ROIC = NOPAT/IC
    terminal_ratio: float = np.nan  # 终值占企业价值比
    conclusion: str = ""
    detail: str = ""
    error: str = ""


def run_eva(
    cd: CompanyData,
    style: str = "auto",
    wacc: float | None = None,
    forecast_years: int = 5,
) -> EVAResult:
    """EVA 估值。WACC 默认按 CAPM 自动估算。"""
    res = EVAResult()
    ann = cd.annual
    if ann is None or len(ann) < 2:
        res.error = "缺少年度财务数据"
        return res

    # 税后营业利润（近3年均值抗噪；营业利润缺失则用 EBITDA 折）
    op_inc = pd_to_numeric(ann["operating_income"]).tail(3).mean()
    if not np.isfinite(op_inc) or op_inc <= 0:
        if "ebitda" in ann.columns:
            op_inc = pd_to_numeric(ann["ebitda"]).tail(3).mean()
    if not np.isfinite(op_inc) or op_inc <= 0:
        res.error = "营业利润数据缺失或非正，EVA 估值不适用。"
        return res
    # 有效税率：近3年所得税/利润总额
    tax_rate = 0.25
    tax = pd_to_numeric(ann["tax"]).tail(3).mean()
    ni = pd_to_numeric(ann["net_income"]).tail(3).mean()
    if np.isfinite(tax) and np.isfinite(ni) and ni > 0:
        tr = tax / ni
        if 0.0 < tr < 0.5:
            tax_rate = tr
    res.nopat = op_inc * (1 - tax_rate)

    # 投入资本 = 股东权益 + 有息负债 − 现金（近3年均值）
    eq = pd_to_numeric(ann["total_equity"]).tail(3).mean()
    debt = pd_to_numeric(ann["total_debt"]).tail(3).mean()
    cash = pd_to_numeric(ann["cash"]).tail(3).mean()
    if not np.isfinite(eq) or eq <= 0:
        res.error = "净资产数据缺失，EVA 估值不适用。"
        return res
    if not np.isfinite(debt):
        debt = 0.0
    if not np.isfinite(cash):
        cash = 0.0
    ic = eq + max(debt, 0) - max(cash, 0)
    if ic <= 0:
        res.error = "投入资本非正，EVA 估值不适用。"
        return res
    res.ic = ic
    res.roic = res.nopat / ic

    # WACC
    if wacc is None:
        w = calc_wacc(cd)
        wacc = w["wacc"]
    if not np.isfinite(wacc) or wacc <= 0:
        res.error = "WACC 计算失败，EVA 估值不适用。"
        return res
    res.wacc = wacc

    # EVA 与预测
    from ..style_presets import resolve_style, style_terminal_g, STYLE_PRESETS
    actual, _ = resolve_style(style, cd)
    presets = STYLE_PRESETS.get(actual, {})
    g_terminal = style_terminal_g(style, cd.market)
    if g_terminal >= wacc:
        res.error = f"永续增长率({g_terminal:.2%}) ≥ WACC({wacc:.2%})，EVA 终值不收敛。"
        return res

    # 显式期增速：风格首年收入增速 × 历史收入 CAGR 约束，逐年向永续过渡
    from .dcf import _trend_growth
    g0 = _trend_growth(ann, 5)
    if not np.isfinite(g0) or g0 <= 0:
        g0 = 0.03
    g0 = min(g0, 0.25)
    decline = presets.get("decline_divisor")
    if decline:
        step = (g0 - g_terminal) / decline
    else:
        step = (g0 - g_terminal) / max(forecast_years, 1)

    eva_now = res.nopat - wacc * ic
    res.eva_now = eva_now

    # 显式期：NOPAT 按增速增长，IC 保持当前（保守下限，避免终值膨胀虚高）
    nopat = res.nopat
    eva_pv_sum = 0.0
    eva_n = eva_now
    for i in range(1, forecast_years + 1):
        g = g0 - step * (i - 1)
        g = max(g, g_terminal)
        nopat *= (1 + g)
        eva_t = nopat - wacc * ic
        eva_pv_sum += eva_t / (1 + wacc) ** i
        eva_n = eva_t

    # 终值
    tv_eva = eva_n * (1 + g_terminal) / (wacc - g_terminal)
    tv_pv = tv_eva / (1 + wacc) ** forecast_years
    enterprise_value = ic + eva_pv_sum + tv_pv
    res.terminal_ratio = tv_pv / enterprise_value if enterprise_value > 0 else np.nan

    # 股权价值
    net_debt = max(debt, 0) - max(cash, 0)
    equity_value = enterprise_value - net_debt
    shares = cd.shares
    if np.isnan(shares) or shares <= 0:
        shares = cd.last_value("shares")
    if not np.isfinite(shares) or shares <= 0 or equity_value <= 0:
        res.error = "股本数据不足或股权价值非正。"
        return res
    res.per_share_value = equity_value / shares

    price = cd.latest_price()
    res.upside = (res.per_share_value / price - 1) if np.isfinite(price) and price > 0 else np.nan
    if res.upside >= 0.2:
        res.conclusion = "低估"
    elif res.upside >= 0:
        res.conclusion = "合理偏低"
    elif res.upside >= -0.2:
        res.conclusion = "合理偏高"
    else:
        res.conclusion = "高估"
    res.detail = (f"ROIC {res.roic:.1%} vs WACC {wacc:.2%}，当年 EVA {eva_now:,.0f}（"
                  + ("正，持续创造价值" if eva_now > 0 else "负，价值被侵蚀")
                  + f"）；企业价值=投入资本 {ic:,.0f} + 未来 EVA 现值，每股 {res.per_share_value:,.2f}。")
    return res


def pd_to_numeric(s) -> "np.ndarray-like float series":
    import pandas as pd
    return pd.to_numeric(s, errors="coerce")
