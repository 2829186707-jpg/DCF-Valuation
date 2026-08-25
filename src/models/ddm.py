# -*- coding: utf-8 -*-
"""
DDM 股利贴现模型（两阶段）：适合稳定分红、可预测的公司（银行/公用事业/消费龙头）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data_fetcher import CompanyData
from ..wacc import calc_wacc


@dataclass
class DDMResult:
    per_share_value: float = np.nan
    upside: float = np.nan
    dps_now: float = np.nan
    payout_ratio: float = np.nan
    dividend_growth: float = np.nan
    re: float = np.nan
    forecast: pd.DataFrame | None = None
    conclusion: str = ""
    detail: str = ""
    error: str = ""


def run_ddm(
    cd: CompanyData,
    growth_phase1: float | None = None,
    growth_phase2: float = 0.02,
    years_phase1: int = 5,
    cost_of_equity: float | None = None,
) -> DDMResult:
    res = DDMResult()
    ann = cd.annual
    if ann is None or len(ann) == 0:
        res.error = "缺少年度财务数据"
        return res

    # 当前每股股利 = 最近一年分红总额 / 股本
    div_total = cd.last_value("dividends")
    shares = cd.shares
    if np.isnan(shares) or shares <= 0:
        shares = cd.last_value("shares")
    if np.isnan(div_total) or np.isnan(shares) or shares <= 0 or div_total < 0:
        res.error = "分红数据缺失（DDM 不适用，可改用 DCF/可比公司）"
        return res
    dps = div_total / shares
    res.dps_now = dps
    res.payout_ratio = div_total / cd.last_value("net_income") if cd.last_value("net_income") > 0 else np.nan

    # 历史分红增速（近5年 CAGR）
    if growth_phase1 is None:
        divs = ann["dividends"].dropna()
        divs = divs[divs > 0]
        if len(divs) >= 2:
            cagr = (divs.iloc[-1] / divs.iloc[0]) ** (1 / (len(divs) - 1)) - 1
            growth_phase1 = float(np.clip(cagr if np.isfinite(cagr) else 0.04, 0.0, 0.30))
        else:
            growth_phase1 = 0.04

    # 股权成本
    if cost_of_equity is None:
        w = calc_wacc(cd)
        re = w["re"]
    else:
        re = cost_of_equity
    res.re = re
    res.dividend_growth = growth_phase1

    if re <= growth_phase2:
        res.error = f"股权成本({re:.2%}) 必须大于永续增长率({growth_phase2:.2%})"
        return res

    # 两阶段
    rows = []
    d = dps
    pv_sum = 0.0
    for i in range(1, years_phase1 + 1):
        g = growth_phase1 - (growth_phase1 - growth_phase2) * (i - 1) / max(years_phase1, 1)
        d *= (1 + g)
        pv = d / (1 + re) ** i
        pv_sum += pv
        rows.append({"年份": cd.latest_year() + i, "每股股利": d, "增长率": g, "现值": pv})
    tv = d * (1 + growth_phase2) / (re - growth_phase2)
    tv_pv = tv / (1 + re) ** years_phase1
    intrinsic = pv_sum + tv_pv
    res.per_share_value = intrinsic
    res.forecast = pd.DataFrame(rows)

    price = cd.latest_price()
    res.upside = (intrinsic / price - 1) if np.isfinite(price) and price > 0 else np.nan
    if res.upside >= 0.2:
        res.conclusion = "低估"
        res.detail = f"DDM 内在价值高于当前价 {res.upside:.1%}。"
    elif res.upside >= 0:
        res.conclusion = "合理偏低"
        res.detail = f"DDM 内在价值略高于当前价 {res.upside:.1%}。"
    elif res.upside >= -0.2:
        res.conclusion = "合理偏高"
        res.detail = f"DDM 内在价值略低于当前价 {-res.upside:.1%}。"
    else:
        res.conclusion = "高估"
        res.detail = f"DDM 内在价值低于当前价 {-res.upside:.1%}。"
    return res
