# -*- coding: utf-8 -*-
"""
WACC 计算：CAPM 股权成本 + 税后债务成本 + 加权。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_fetcher import CompanyData


def calc_wacc(
    cd: CompanyData,
    beta: float | None = None,
    debt_rate: float | None = None,
    erp: float | None = None,
    rf: float | None = None,
) -> dict:
    """
    计算 WACC。返回 {re, rd, tax_rate, e_weight, d_weight, wacc, note}
    """
    rf_ = cd.rf if rf is None else rf
    erp_ = cd.erp if erp is None else erp
    beta_ = cd.beta if beta is None else beta

    # 税率：用最新年 tax/(税前利润)，缺失时按市场默认
    tax_rate = np.nan
    if cd.annual is not None and "tax" in cd.annual and "operating_income" in cd.annual:
        t = cd.annual["tax"]
        ebit = cd.annual["operating_income"]
        m = (ebit > 0) & (~t.isna()) & (~ebit.isna())
        if m.sum() > 0:
            tax_rate = float((t[m] / ebit[m]).median())
    if np.isnan(tax_rate) or tax_rate <= 0 or tax_rate >= 0.6:
        tax_rate = 0.25 if cd.market == "A" else 0.21

    # 股权成本
    re = rf_ + beta_ * erp_

    # 债务成本：默认 A股4.0%，美股4.0%（可在面板调整）
    if debt_rate is None:
        debt_rate = 0.040

    # 权重：市值 + 有息负债
    e = cd.market_cap if not np.isnan(cd.market_cap) else np.nan
    d = cd.last_value("total_debt")
    if np.isnan(d) or d < 0:
        d = 0.0

    if np.isnan(e) or e <= 0:
        # 市值缺失时用账面权益
        eq = cd.last_value("total_equity")
        if np.isnan(eq) or eq <= 0:
            e_weight, d_weight = 1.0, 0.0
        else:
            e, d = max(eq - d, 0.0), d
            e_weight = e / (e + d)
            d_weight = d / (e + d)
    else:
        e_weight = e / (e + d)
        d_weight = d / (e + d)

    # 金融股警示（高杠杆）：仍计算但标记
    note = ""
    if d_weight > 0.5:
        note = "⚠️ 有息负债占比较高，WACC 对债务假设敏感，请结合行业复核。"

    rd_after = debt_rate * (1 - tax_rate)
    wacc = e_weight * re + d_weight * rd_after

    return {
        "re": float(re),
        "rd": float(debt_rate),
        "tax_rate": float(tax_rate),
        "e_weight": float(e_weight),
        "d_weight": float(d_weight),
        "wacc": float(wacc),
        "rf": float(rf_),
        "erp": float(erp_),
        "beta": float(beta_),
        "note": note,
    }
