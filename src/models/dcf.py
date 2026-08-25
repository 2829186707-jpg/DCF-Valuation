# -*- coding: utf-8 -*-
"""
DCF 估值模型（FCFF 两阶段）。

流程：
  历史数据 → 自动生成基准假设 → 显式预测期 FCFF → 终值 → EV → 股权价值 → 每股价值
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data_fetcher import CompanyData
from ..wacc import calc_wacc


@dataclass
class DCFAssumptions:
    forecast_years: int = 5
    revenue_growth: float = 0.05          # 显式期首年收入增长率
    growth_decline: float = 0.0           # 每年增长率衰减(百分点)，0=恒定
    terminal_growth: float = 0.025        # 永续增长率
    operating_margin: float = 0.20        # 目标营业利润率
    tax_rate: float = 0.25
    capex_pct: float = 0.05               # 资本开支/收入
    da_pct: float = 0.05                  # 折旧摊销/收入
    nwc_pct: float = 0.00                 # 营运资本变动/收入(正=占用增加流出)
    beta: float = 1.0
    debt_rate: float = 0.040
    erp: float | None = None
    rf: float | None = None


@dataclass
class DCFResult:
    per_share_value: float = np.nan
    upside: float = np.nan
    ev: float = np.nan
    equity_value: float = np.nan
    wacc: float = np.nan
    terminal_value: float = np.nan
    terminal_pv: float = np.nan
    terminal_ratio: float = np.nan        # 终值现值占EV比例
    forecast: pd.DataFrame | None = None
    assumptions: dict = field(default_factory=dict)
    conclusion: str = ""
    detail: str = ""
    error: str = ""


# ---------------- 自动生成基准假设 ----------------

def auto_assumptions(cd: CompanyData, wacc_info: dict) -> DCFAssumptions:
    a = DCFAssumptions()
    ann = cd.annual

    if ann is not None and len(ann) >= 2:
        rev = ann["revenue"].dropna()
        # 历史增长率（近5年 CAGR，更贴近当前增长阶段）
        if len(rev) >= 2:
            r5 = rev.tail(5)
            if len(r5) >= 2:
                hist_g = (r5.iloc[-1] / r5.iloc[0]) ** (1 / (len(r5) - 1)) - 1
            else:
                hist_g = np.nan
            if not np.isfinite(hist_g):
                hist_g = rev.pct_change().tail(3).mean()
            if np.isnan(hist_g) or not np.isfinite(hist_g):
                hist_g = 0.05
            # 增长率打 6-8 折作为预测起点
            a.revenue_growth = float(max(hist_g * 0.7, 0.005))
            a.growth_decline = float(max((hist_g - 0.03) / 5, 0.0))

        # 营业利润率：最近3年均值，并做合理区间约束
        margin = (ann["operating_income"] / ann["revenue"]).dropna()
        if len(margin):
            m = float(margin.tail(3).median())
            a.operating_margin = float(np.clip(m, 0.005, 0.85))

        # capex/da/nwc 比例：历史均值
        rev_mean = ann["revenue"].replace(0, np.nan).mean()
        if np.isfinite(rev_mean) and rev_mean > 0:
            cp = (ann["capex"] / ann["revenue"]).dropna()
            if len(cp):
                a.capex_pct = float(np.clip(cp.tail(3).median(), 0.0, 0.6))
            dp = (ann["da"] / ann["revenue"]).dropna()
            if len(dp):
                a.da_pct = float(np.clip(dp.tail(3).median(), 0.0, 0.6))
            nw = (ann["change_wc"] / ann["revenue"]).dropna()
            if len(nw):
                # 营运资本变动占收入（带符号：正=占用增加流出，负=释放现金）
                a.nwc_pct = float(np.clip(nw.tail(3).median(), -0.10, 0.20))

    a.tax_rate = float(wacc_info["tax_rate"])
    a.beta = float(wacc_info["beta"])
    a.erp = wacc_info["erp"]
    a.rf = wacc_info["rf"]
    return a


# ---------------- 核心计算 ----------------

def run_dcf(cd: CompanyData, assump: DCFAssumptions | None = None) -> DCFResult:
    res = DCFResult()
    if cd.annual is None or len(cd.annual) == 0:
        res.error = "缺少年度财务数据"
        return res

    a = assump or DCFAssumptions()
    if a.erp is None:
        a.erp = cd.erp
    if a.rf is None:
        a.rf = cd.rf

    wacc_info = calc_wacc(cd, beta=a.beta, debt_rate=a.debt_rate, erp=a.erp, rf=a.rf)
    wacc = wacc_info["wacc"]
    res.wacc = wacc

    # 基准年（用最新年）
    last_rev = cd.last_value("revenue")
    if not np.isfinite(last_rev) or last_rev <= 0:
        res.error = "营业收入数据缺失"
        return res

    # 增长率序列
    n = a.forecast_years
    growths = [a.revenue_growth - a.growth_decline * i for i in range(n)]
    growths = [max(g, a.terminal_growth - 0.01) for g in growths]

    rows = []
    rev = last_rev
    fcff_sum = 0.0
    for i in range(n):
        g = growths[i]
        rev *= (1 + g)
        ebit = rev * a.operating_margin
        nopat = ebit * (1 - a.tax_rate)
        da = rev * a.da_pct
        capex = rev * a.capex_pct
        nwc = rev * a.nwc_pct
        fcff = nopat + da - capex - nwc
        year = cd.latest_year() + i + 1
        rows.append({
            "年份": year,
            "收入": rev,
            "EBIT": ebit,
            "NOPAT": nopat,
            "折旧摊销": da,
            "资本开支": capex,
            "营运资本变动": nwc,
            "FCFF": fcff,
            "增长率": g,
        })
    fcff_last = rows[-1]["FCFF"]

    # 终值（Gordon）
    if wacc <= a.terminal_growth:
        res.error = f"WACC({wacc:.2%}) 必须大于永续增长率({a.terminal_growth:.2%})"
        return res
    tv = fcff_last * (1 + a.terminal_growth) / (wacc - a.terminal_growth)
    tv_pv = tv / (1 + wacc) ** n

    # EV
    ev = 0.0
    for i, row in enumerate(rows):
        pv = row["FCFF"] / (1 + wacc) ** (i + 1)
        ev += pv
        rows[i]["FCFF现值"] = pv
    ev += tv_pv

    # 股权价值 = EV - 净债务
    debt = cd.last_value("total_debt")
    cash = cd.last_value("cash")
    if not np.isfinite(debt) or debt < 0:
        debt = 0.0
    if not np.isfinite(cash) or cash < 0:
        cash = 0.0
    net_debt = debt - cash
    equity_value = ev - net_debt

    # 每股
    shares = cd.shares
    if np.isnan(shares) or shares <= 0:
        shares = cd.last_value("shares")
    if np.isnan(shares) or shares <= 0:
        res.error = "总股本缺失，无法计算每股价值"
        return res
    per_share = equity_value / shares

    price = cd.latest_price()
    upside = (per_share / price - 1) if np.isfinite(price) and price > 0 else np.nan
    if upside >= 0.2:
        conclusion = "低估"
        detail = f"内在价值高于当前价 {upside:.1%}，安全边际较充足，具备吸引力。"
    elif upside >= 0:
        conclusion = "合理偏低"
        detail = f"内在价值略高于当前价 {upside:.1%}，估值处于合理区间下沿。"
    elif upside >= -0.2:
        conclusion = "合理偏高"
        detail = f"内在价值低于当前价 {-upside:.1%}，估值处于合理区间上沿。"
    else:
        conclusion = "高估"
        detail = f"内在价值低于当前价 {-upside:.1%}，当前价格透支了较多预期。"

    res.per_share_value = per_share
    res.upside = upside
    res.ev = ev
    res.equity_value = equity_value
    res.terminal_value = tv
    res.terminal_pv = tv_pv
    res.terminal_ratio = tv_pv / ev if ev else np.nan
    res.forecast = pd.DataFrame(rows)
    res.assumptions = {
        "forecast_years": n,
        "revenue_growth": a.revenue_growth,
        "growth_decline": a.growth_decline,
        "terminal_growth": a.terminal_growth,
        "operating_margin": a.operating_margin,
        "tax_rate": a.tax_rate,
        "capex_pct": a.capex_pct,
        "da_pct": a.da_pct,
        "nwc_pct": a.nwc_pct,
        "beta": a.beta,
        "debt_rate": a.debt_rate,
        "wacc": wacc,
    }
    res.conclusion = conclusion
    res.detail = detail
    return res
