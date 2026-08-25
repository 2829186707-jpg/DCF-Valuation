# -*- coding: utf-8 -*-
"""
反向 DCF（Reverse DCF）：给定当前股价，反推市场隐含的关键假设。

默认反推「显式期首年收入增长率 g」：固定 WACC、利润率、永续增长等假设，
求解使 DCF 每股价值 = 当前股价的 g。可切换为反推「永续增长率 g_terminal」。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data_fetcher import CompanyData
from ..wacc import calc_wacc


@dataclass
class ReverseDCFResult:
    implied_growth: float = np.nan
    implied_terminal_growth: float = np.nan
    mode: str = "growth"          # growth | terminal
    price: float = np.nan
    wacc: float = np.nan
    current_growth: float = np.nan
    conclusion: str = ""
    detail: str = ""
    error: str = ""


def _dcf_per_share(
    cd: CompanyData, wacc: float, tax_rate: float, margin: float,
    capex_pct: float, da_pct: float, nwc_pct: float,
    g0: float, decline: float, g_terminal: float, years: int,
) -> float:
    """给定假设，返回 DCF 每股价值。"""
    last_rev = cd.last_value("revenue")
    shares = cd.shares
    if not np.isfinite(last_rev) or last_rev <= 0 or not np.isfinite(shares) or shares <= 0:
        return np.nan

    growths = [g0 - decline * i for i in range(years)]
    growths = [max(g, -0.2) for g in growths]

    rev = last_rev
    pv_sum = 0.0
    fcff_last = 0.0
    for i in range(years):
        rev *= (1 + growths[i])
        ebit = rev * margin
        nopat = ebit * (1 - tax_rate)
        fcff = nopat + rev * da_pct - rev * capex_pct - rev * nwc_pct
        fcff_last = fcff
        pv_sum += fcff / (1 + wacc) ** (i + 1)
    if wacc <= g_terminal:
        return np.nan
    tv = fcff_last * (1 + g_terminal) / (wacc - g_terminal)
    tv_pv = tv / (1 + wacc) ** years
    ev = pv_sum + tv_pv
    debt = cd.last_value("total_debt")
    cash = cd.last_value("cash")
    if not np.isfinite(debt) or debt < 0:
        debt = 0.0
    if not np.isfinite(cash) or cash < 0:
        cash = 0.0
    return (ev - (debt - cash)) / shares


def run_reverse_dcf(
    cd: CompanyData,
    mode: str = "growth",
    wacc: float | None = None,
    tax_rate: float | None = None,
    margin: float | None = None,
    capex_pct: float | None = None,
    da_pct: float | None = None,
    nwc_pct: float | None = None,
    g_terminal_fixed: float = 0.025,
    g0_fixed: float = 0.08,
    decline: float = 0.0,
    years: int = 5,
) -> ReverseDCFResult:
    res = ReverseDCFResult()
    res.mode = mode
    res.price = cd.latest_price()
    if not np.isfinite(res.price) or res.price <= 0:
        res.error = "当前价格缺失"
        return res
    if cd.annual is None or len(cd.annual) == 0:
        res.error = "缺少年度财务数据"
        return res

    w = calc_wacc(cd)
    wacc = w["wacc"] if wacc is None else wacc
    tax_rate = w["tax_rate"] if tax_rate is None else tax_rate
    res.wacc = wacc

    if margin is None:
        m = (cd.annual["operating_income"] / cd.annual["revenue"]).dropna()
        margin = float(np.clip(m.tail(3).median(), 0.005, 0.85)) if len(m) else 0.15
    if capex_pct is None:
        cp = (cd.annual["capex"] / cd.annual["revenue"]).dropna()
        capex_pct = float(np.clip(cp.tail(3).median(), 0.0, 0.6)) if len(cp) else 0.05
    if da_pct is None:
        dp = (cd.annual["da"] / cd.annual["revenue"]).dropna()
        da_pct = float(np.clip(dp.tail(3).median(), 0.0, 0.6)) if len(dp) else 0.05
    if nwc_pct is None:
        nw = (cd.annual["change_wc"] / cd.annual["revenue"]).dropna()
        nwc_pct = float(np.clip(nw.tail(3).median(), -0.10, 0.20)) if len(nw) else 0.0

    def f_growth(g0):
        return _dcf_per_share(
            cd, wacc, tax_rate, margin, capex_pct, da_pct, nwc_pct,
            g0, decline, g_terminal_fixed, years,
        ) - res.price

    def f_terminal(gterm):
        return _dcf_per_share(
            cd, wacc, tax_rate, margin, capex_pct, da_pct, nwc_pct,
            g0_fixed, decline, gterm, years,
        ) - res.price

    if mode == "growth":
        lo, hi = -0.15, 0.60
        # 找区间
        flo = f_growth(lo)
        fhi = f_growth(hi)
        if np.isnan(flo) or np.isnan(fhi):
            res.error = "无法求解（检查 WACC 与永续增长率）"
            return res
        if flo * fhi > 0:
            res.error = "当前价格超出模型可求解范围"
            return res
        for _ in range(60):
            mid = (lo + hi) / 2
            fm = f_growth(mid)
            if np.isnan(fm):
                res.error = "求解过程数值异常"
                return res
            if flo * fm <= 0:
                hi = mid
            else:
                lo, flo = mid, fm
        res.implied_growth = (lo + hi) / 2
        res.current_growth = g0_fixed
        # 对比公司当前隐含（用近5年历史增长）
        hist_g = np.nan
        rev = cd.annual["revenue"].dropna()
        if len(rev) >= 2:
            r5 = rev.tail(5)
            if len(r5) >= 2:
                hist_g = (r5.iloc[-1] / r5.iloc[0]) ** (1 / (len(r5) - 1)) - 1
        res.current_growth = float(hist_g) if np.isfinite(hist_g) else np.nan
        res.detail = _judge(res.implied_growth, res.current_growth, "显式期首年收入增长率")
    else:
        lo, hi = 0.0, wacc - 0.005
        flo = f_terminal(lo)
        fhi = f_terminal(hi)
        if np.isnan(flo) or np.isnan(fhi):
            res.error = "无法求解（检查 WACC）"
            return res
        if flo * fhi > 0:
            res.error = "当前价格超出模型可求解范围"
            return res
        for _ in range(60):
            mid = (lo + hi) / 2
            fm = f_terminal(mid)
            if np.isnan(fm):
                res.error = "求解过程数值异常"
                return res
            if flo * fm <= 0:
                hi = mid
            else:
                lo, flo = mid, fm
        res.implied_terminal_growth = (lo + hi) / 2
        res.detail = _judge(res.implied_terminal_growth, g_terminal_fixed, "永续增长率")

    return res


def _judge(implied: float, ref: float, label: str) -> str:
    if not np.isfinite(ref):
        return f"市场隐含{label}为 {implied:.2%}（无历史参考值可对比）。"
    diff = implied - ref
    if diff > 0.02:
        return (f"市场隐含{label} {implied:.2%}，高于公司历史增长 {ref:.2%}。"
                f"市场已定价较高的成长预期，需警惕预期过高风险。")
    if diff < -0.02:
        return (f"市场隐含{label} {implied:.2%}，低于公司历史增长 {ref:.2%}。"
                f"当前定价相对保守，存在预期差机会。")
    return (f"市场隐含{label} {implied:.2%}，与公司历史增长 {ref:.2%} 基本相当，"
            f"当前定价较为中性。")
