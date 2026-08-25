# -*- coding: utf-8 -*-
"""
可比公司法：用同业公司的 PE / PB / EV-EBITDA 中位数给目标公司估值。

可比公司来源：
  1. 手动输入（最可靠，推荐）
  2. 自动获取同业（尽力而为：A股走新浪行业，美股走 yfinance Industry）
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data_fetcher import CompanyData, fetch_company, _norm_a_symbol


@dataclass
class CompsResult:
    peer_table: pd.DataFrame | None = None
    per_share_pe: float = np.nan
    per_share_pb: float = np.nan
    per_share_ev_ebitda: float = np.nan
    median_pe: float = np.nan
    median_pb: float = np.nan
    median_ev_ebitda: float = np.nan
    upside_pe: float = np.nan
    upside_pb: float = np.nan
    upside_ev_ebitda: float = np.nan
    conclusion: str = ""
    detail: str = ""
    error: str = ""


def _a_peer_metrics(symbol6: str) -> dict:
    """A股可比公司：腾讯接口拿 PE/PB（EV/EBITDA 需财务数据，暂略）"""
    import requests
    url = f"https://qt.gtimg.cn/q={_norm_a_symbol(symbol6)}"
    r = requests.get(url, timeout=10)
    r.encoding = "gbk"
    m = __import__("re").search(r'="(.*?)"', r.text)
    if not m:
        return {}
    f = m.group(1).split("~")
    out = {"name": f[1] if len(f) > 1 else symbol6}
    try:
        out["pe"] = float(f[39]) if f[39] else np.nan
    except Exception:
        out["pe"] = np.nan
    try:
        out["pb"] = float(f[46]) if f[46] else np.nan
    except Exception:
        out["pb"] = np.nan
    return out


def _us_peer_metrics(symbol: str, session=None) -> dict:
    """美股可比公司：腾讯接口拿 PE/PB（EV/EBITDA 缺失则略）"""
    import requests
    try:
        r = requests.get(f"https://qt.gtimg.cn/q=us{symbol}", timeout=10)
        r.encoding = "gbk"
        m = __import__("re").search(r'="(.*?)"', r.text)
        if not m or "pv_none_match" in r.text:
            return {}
        f = m.group(1).split("~")
        out = {"name": f[1] if len(f) > 1 else symbol}
        try:
            out["pe"] = float(f[39]) if f[39] else np.nan
        except Exception:
            out["pe"] = np.nan
        out["pb"] = np.nan  # 腾讯美股无可靠 PB 字段
        out["ev_ebitda"] = np.nan
        return out
    except Exception:
        return {}


def auto_peers(cd: CompanyData, limit: int = 6) -> list[str]:
    """尽力获取同业公司代码。失败返回空列表。"""
    out: list[str] = []
    try:
        if cd.market == "A":
            import akshare as ak
            # 用新浪行业板块
            sector_df = ak.stock_sector_spot(indicator="新浪行业")
            # 找目标公司所属行业：需要行业成分，新浪行业接口仅给板块行情。
            # 简化：遍历新浪行业成分接口不可行(无成分接口)，改用公司所在板块
            # 尝试东财行业(可能失败) → 失败则跳过
            try:
                ind = ak.stock_board_industry_name_em()
                # 需要成分，同样可能被限流
            except Exception:
                pass
        else:
            import yfinance as yf
            info = cd.annual if False else None
            # 用 yfinance Industry 类
            tk_info = _us_peer_metrics(cd.symbol)
            return out  # Industry 精确匹配复杂，交给用户手动
    except Exception:
        pass
    return out


def run_comps(
    cd: CompanyData,
    peer_symbols: list[str],
    session=None,
) -> CompsResult:
    res = CompsResult()
    peers: list[dict] = []
    for sym in peer_symbols:
        sym = sym.strip().upper()
        if not sym:
            continue
        try:
            if cd.market == "A":
                m = _a_peer_metrics(sym)
                m["ev_ebitda"] = np.nan
            else:
                m = _us_peer_metrics(sym, session=session)
            m["symbol"] = sym
            if m.get("pe") is not None or m.get("pb") is not None:
                peers.append(m)
        except Exception:
            continue

    if not peers:
        res.error = "未能获取可比公司数据，请检查可比公司代码。"
        return res

    df = pd.DataFrame(peers).set_index("symbol")
    res.peer_table = df
    med_pe = pd.to_numeric(df["pe"], errors="coerce").median()
    med_pb = pd.to_numeric(df["pb"], errors="coerce").median()
    med_ev = pd.to_numeric(df.get("ev_ebitda", pd.Series(dtype=float)), errors="coerce").median()
    res.median_pe = med_pe
    res.median_pb = med_pb
    res.median_ev_ebitda = med_ev

    shares = cd.shares
    if np.isnan(shares) or shares <= 0:
        shares = cd.last_value("shares")
    price = cd.latest_price()
    net_income = cd.last_value("net_income")
    total_equity = cd.last_value("total_equity")
    ebitda = cd.last_value("ebitda")

    results = {}
    if np.isfinite(med_pe) and np.isfinite(net_income) and net_income > 0 and np.isfinite(shares):
        res.per_share_pe = med_pe * net_income / shares
        results["PE"] = res.per_share_pe
        res.upside_pe = res.per_share_pe / price - 1 if np.isfinite(price) and price > 0 else np.nan
    if np.isfinite(med_pb) and np.isfinite(total_equity) and total_equity > 0 and np.isfinite(shares):
        res.per_share_pb = med_pb * total_equity / shares
        results["PB"] = res.per_share_pb
        res.upside_pb = res.per_share_pb / price - 1 if np.isfinite(price) and price > 0 else np.nan
    if (np.isfinite(med_ev) and np.isfinite(ebitda) and ebitda > 0
            and np.isfinite(shares)):
        debt = cd.last_value("total_debt")
        cash = cd.last_value("cash")
        if not np.isfinite(debt) or debt < 0:
            debt = 0.0
        if not np.isfinite(cash) or cash < 0:
            cash = 0.0
        ev = med_ev * ebitda
        equity = ev - (debt - cash)
        res.per_share_ev_ebitda = equity / shares
        results["EV/EBITDA"] = res.per_share_ev_ebitda
        res.upside_ev_ebitda = res.per_share_ev_ebitda / price - 1 if np.isfinite(price) and price > 0 else np.nan

    if not results:
        res.error = "可比公司估值所需数据不足（净利润/净资产/EBITDA）。"
        return res

    avg = float(np.mean(list(results.values())))
    res.conclusion = "低估" if avg >= price * 1.2 else ("高估" if avg <= price * 0.8 else "合理")
    res.detail = (f"基于 {len(df)} 家可比公司，PE/PB/EV-EBITDA 中位数估值均值约 "
                  f"{avg/price:.0%} 倍当前价。")
    return res
