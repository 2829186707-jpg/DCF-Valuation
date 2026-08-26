# -*- coding: utf-8 -*-
"""
PEG 估值法：目标 PE = PEG系数 × 增长率(%)，目标价 = EPS × 目标PE。

核心逻辑：市盈率应与盈利增速匹配——成长股可享更高 PE，价值/低增长股 PE 应更低。
- PEG=1 表示"价格增速比"平衡（经典彼得·林奇标准）
- 按风格调整目标 PEG：成长股 1.2（高景气可溢价）、稳健股 1.0、价值股 0.8（低增长折价）
- 周期股盈利失真，不适用（返回错误，综合研判中权重为 0）
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data_fetcher import CompanyData

# 风格 → 目标 PEG 系数（0=不适用）
STYLE_TARGET_PEG: dict[str, float] = {
    "growth": 1.2,
    "steady": 1.0,
    "value": 0.8,
    "cyclical": 0.0,   # 周期股盈利峰谷失真，PE 无意义
}


@dataclass
class PEGResult:
    per_share_value: float = np.nan
    upside: float = np.nan
    eps: float = np.nan          # 每股收益
    growth: float = np.nan       # 盈利增速（净利润 CAGR）
    target_pe: float = np.nan    # 目标 PE
    target_peg: float = np.nan   # 目标 PEG 系数
    peg_now: float = np.nan      # 当前 PEG = 当前PE / 增速
    conclusion: str = ""
    detail: str = ""
    error: str = ""


def _ni_cagr(cd: CompanyData, years: int = 5) -> float:
    """近 N 年净利润 CAGR（稳健版：首尾须为正）。"""
    ann = cd.annual
    if ann is None or len(ann) == 0 or "net_income" not in ann.columns:
        return np.nan
    ni = ann["net_income"].dropna()
    ni = ni[ni > 0]
    if len(ni) < 2:
        return np.nan
    r = ni.tail(years)
    if len(r) < 2:
        r = ni
    first, last = r.iloc[0], r.iloc[-1]
    if np.isfinite(first) and np.isfinite(last) and first > 0 and last > 0:
        return float((last / first) ** (1 / (len(r) - 1)) - 1)
    return np.nan


def run_peg(
    cd: CompanyData,
    style: str = "auto",
    growth: float | None = None,
    target_peg: float | None = None,
) -> PEGResult:
    """PEG 估值。growth 可显式指定，否则取历史净利润 CAGR。"""
    res = PEGResult()
    ann = cd.annual
    if ann is None or len(ann) == 0:
        res.error = "缺少年度财务数据"
        return res

    # 解析实际风格
    from ..style_presets import resolve_style
    actual, _ = resolve_style(style, cd)

    # 目标 PEG 系数（cyclical 不适用）
    if target_peg is None:
        target_peg = STYLE_TARGET_PEG.get(actual, 1.0)
    res.target_peg = target_peg
    if target_peg <= 0:
        res.error = "周期股盈利峰谷失真，PEG 估值不适用（综合研判中已自动给低权重）。"
        return res

    # 每股收益
    shares = cd.shares
    if np.isnan(shares) or shares <= 0:
        shares = cd.last_value("shares")
    net_income = cd.last_value("net_income")
    if not np.isfinite(shares) or shares <= 0 or not np.isfinite(net_income) or net_income <= 0:
        res.error = "净利润或股本数据不足，PEG 估值不适用。"
        return res
    eps = net_income / shares
    res.eps = eps

    # 增速
    if growth is None:
        g = _ni_cagr(cd, 5)
        if not np.isfinite(g):
            g = np.nan
    else:
        g = float(growth)
    if not np.isfinite(g) or g <= 0:
        res.error = "历史净利润未正增长，PEG 估值不适用（增速≤0 时 PE 无参照意义）。"
        return res
    g = min(g, 0.40)  # 增速封顶 40%（>40% 不可持续）
    res.growth = g

    # 目标 PE 与目标价
    target_pe = target_peg * (g * 100)
    target_pe = float(np.clip(target_pe, 3.0, 60.0))  # 合理 PE 区间 [3, 60]
    res.target_pe = target_pe
    res.per_share_value = eps * target_pe

    # 当前 PEG
    price = cd.latest_price()
    cur_pe = cd.pe_ttm if np.isfinite(cd.pe_ttm) and cd.pe_ttm > 0 else price / eps if price and eps > 0 else np.nan
    if np.isfinite(cur_pe):
        res.peg_now = cur_pe / (g * 100)

    res.upside = (res.per_share_value / price - 1) if np.isfinite(price) and price > 0 else np.nan
    if res.upside >= 0.2:
        res.conclusion = "低估"
    elif res.upside >= 0:
        res.conclusion = "合理偏低"
    elif res.upside >= -0.2:
        res.conclusion = "合理偏高"
    else:
        res.conclusion = "高估"
    res.detail = (f"按【{actual} 口径】目标 PEG {target_peg:.1f} × 盈利增速 {g:.1%} → 目标 PE {target_pe:.1f}，"
                  f"当前 PE {cur_pe:.1f}" + (f"（PEG {res.peg_now:.2f}）" if np.isfinite(res.peg_now) else "")
                  + f"，目标每股价值 {res.per_share_value:,.2f}。")
    return res
