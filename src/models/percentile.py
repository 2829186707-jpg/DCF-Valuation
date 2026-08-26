# -*- coding: utf-8 -*-
"""
历史估值分位法：把当前 PE/PB 放到近 8 年历史分布中，判断"贵还是便宜"。

核心逻辑：估值有均值回归倾向——当前 PE/PB 处于历史高分位说明偏贵、低分位说明偏便宜。
- 目标价 = 历史 PE/PB 中位数(P50) × 当前 EPS/BVPS
- 同时给出当前分位，作为安全边际提示
- 仅支持 A 股（乐咕乐股历史估值接口）；美股暂不适用
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data_fetcher import CompanyData


@dataclass
class PercentileResult:
    per_share_value: float = np.nan
    upside: float = np.nan
    current_pe: float = np.nan
    current_pb: float = np.nan
    pe_p50: float = np.nan
    pb_p50: float = np.nan
    pe_percentile: float = np.nan   # 当前 PE 历史分位(0~1)
    pb_percentile: float = np.nan   # 当前 PB 历史分位(0~1)
    history_days: int = 0
    conclusion: str = ""
    detail: str = ""
    error: str = ""


def run_percentile(cd: CompanyData, years: int = 8, target_percentile: float = 0.5) -> PercentileResult:
    """历史估值分位法。目标价 = 历史中位数(默认 P50) × 当前 EPS/BVPS。"""
    res = PercentileResult()
    if cd.market != "A":
        res.error = "美股历史估值分位暂不支持（乐咕接口仅覆盖 A 股），综合研判中权重已置 0。"
        return res

    import warnings
    import pandas as pd
    import akshare as ak
    warnings.filterwarnings("ignore")
    symbol = str(cd.symbol).strip().zfill(6)
    try:
        pe_df = ak.stock_zh_valuation_baidu(symbol=symbol, indicator="市盈率(TTM)", period="全部")
        pb_df = ak.stock_zh_valuation_baidu(symbol=symbol, indicator="市净率", period="全部")
    except Exception as e:
        res.error = f"历史估值数据获取失败：{type(e).__name__}"
        return res

    def _clean(df, name):
        if df is None or len(df) == 0:
            return pd.Series(dtype=float)
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"])
        cutoff = d["date"].max() - pd.DateOffset(years=years)
        d = d[d["date"] >= cutoff]
        s = pd.to_numeric(d["value"], errors="coerce").dropna()
        s = s[(s > 0) & (s < (300 if name == "pe" else 100))]
        return s

    pe = _clean(pe_df, "pe")
    pb = _clean(pb_df, "pb")
    if len(pe) < 60 or len(pb) < 60:
        res.error = f"历史 PE/PB 有效样本不足（PE {len(pe)}/PB {len(pb)}，需≥60）"
        return res

    res.history_days = len(pe)
    res.pe_p50 = float(np.nanpercentile(pe, target_percentile * 100))
    res.pb_p50 = float(np.nanpercentile(pb, target_percentile * 100))

    # 当前 PE/PB（优先腾讯实时；缺失时用最新价/EPS 估算）
    cur_pe = cd.pe_ttm
    cur_pb = cd.pb
    shares = cd.shares if np.isfinite(cd.shares) and cd.shares > 0 else cd.last_value("shares")
    eps = np.nan
    if np.isfinite(shares) and shares > 0:
        ni = cd.last_value("net_income")
        if np.isfinite(ni) and ni > 0:
            eps = ni / shares
    bvps = np.nan
    if np.isfinite(shares) and shares > 0:
        eq = cd.last_value("total_equity")
        if np.isfinite(eq) and eq > 0:
            bvps = eq / shares
    price = cd.latest_price()
    if (not np.isfinite(cur_pe) or cur_pe <= 0) and np.isfinite(eps) and eps > 0 and np.isfinite(price):
        cur_pe = price / eps
    if (not np.isfinite(cur_pb) or cur_pb <= 0) and np.isfinite(bvps) and bvps > 0 and np.isfinite(price):
        cur_pb = price / bvps
    res.current_pe = cur_pe if np.isfinite(cur_pe) else np.nan
    res.current_pb = cur_pb if np.isfinite(cur_pb) else np.nan

    # 当前分位
    if np.isfinite(cur_pe) and cur_pe > 0:
        res.pe_percentile = float((pe < cur_pe).mean())
    if np.isfinite(cur_pb) and cur_pb > 0:
        res.pb_percentile = float((pb < cur_pb).mean())

    # 目标价 = 历史中位数 × 当前 EPS/BVPS
    vals = []
    if np.isfinite(res.pe_p50) and np.isfinite(eps) and eps > 0:
        vals.append(res.pe_p50 * eps)
    if np.isfinite(res.pb_p50) and np.isfinite(bvps) and bvps > 0:
        vals.append(res.pb_p50 * bvps)
    if not vals:
        res.error = "EPS/净资产数据不足，无法计算目标价。"
        return res
    res.per_share_value = float(np.mean(vals))

    res.upside = (res.per_share_value / price - 1) if np.isfinite(price) and price > 0 else np.nan
    if res.upside >= 0.2:
        res.conclusion = "低估"
    elif res.upside >= 0:
        res.conclusion = "合理偏低"
    elif res.upside >= -0.2:
        res.conclusion = "合理偏高"
    else:
        res.conclusion = "高估"

    pe_pos = f"{res.pe_percentile:.0%}" if np.isfinite(res.pe_percentile) else "N/A"
    pb_pos = f"{res.pb_percentile:.0%}" if np.isfinite(res.pb_percentile) else "N/A"
    res.detail = (f"近 {years} 年 PE/PB 分位：当前 PE {cur_pe:.1f}（历史 {pe_pos} 分位）、"
                  f"PB {cur_pb:.2f}（历史 {pb_pos} 分位）；按历史中位数 PE {res.pe_p50:.1f}/PB {res.pb_p50:.2f} "
                  f"对应目标价 {res.per_share_value:,.2f}。")
    return res
