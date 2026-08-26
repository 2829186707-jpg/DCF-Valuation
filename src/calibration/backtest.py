# -*- coding: utf-8 -*-
"""
历史回测引擎：在历史时点用「当时可得数据」跑当前估值模型，对比之后 1/2 年实际表现。

关键防前视设计：
  1. 财务只用「回测年份 - 1」的年报（模拟当年财报尚未披露的滞后）
  2. Beta 只用截至回测时点的前 1 年日线回归
  3. 无风险利率取历史时点的 10 年国债收益率
  4. 每只股票只拉一次全历史数据，多个回测时点复用

价格口径（重要）：
  - 收益率 / 实际涨幅：用「复权价」（A股前复权 / 美股后复权），连续无除权跳空
  - 时点价 / 市值对比：用「原始价」（与当年股本口径一致，才能与每股内在价值可比）
  - 美股无原始价接口（新浪后复权 / yfinance 限流 / 东财被封），
    用「股本变化因子」还原：factor = 当前股本 / 当年股本，真实价 = 后复权价 / factor
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data_fetcher import CompanyData


# ---------------- 历史数据加载（一次拉全，多时点复用） ----------------

@dataclass
class HistoryData:
    """一只股票的完整历史数据（供多个回测时点复用）。"""
    symbol: str
    market: str            # "A" / "US"
    name: str
    annual: pd.DataFrame   # 全历史年度财务（索引=年份）
    hist: pd.DataFrame     # 复权价日线（索引=Date，含 Close；算收益率用）
    hist_raw: pd.DataFrame # 原始价日线（对比/市值用；美股为后复权占位）
    index_hist: pd.DataFrame | None = None
    rf_series: pd.Series | None = None
    erp: float = 0.05
    current_shares: float = np.nan  # 当前股本（美股还原真实价用）


def _load_a_history(symbol: str) -> HistoryData:
    import akshare as ak
    from ..data_fetcher import _fetch_a_annual_sina, _norm_a_symbol, _tx_quote

    sym6 = _norm_a_symbol(symbol).lstrip("shszbj")
    annual = _fetch_a_annual_sina(sym6)
    if annual is None or len(annual) == 0:
        raise RuntimeError(f"A股财务数据缺失: {symbol}")

    today = dt.date.today().strftime("%Y%m%d")
    # 复权价（前复权，算收益率）
    hist = ak.stock_zh_a_daily(symbol=_norm_a_symbol(symbol), start_date="20130101",
                               end_date=today, adjust="qfq")
    hist = hist.rename(columns={"date": "Date", "close": "Close"})
    hist["Date"] = pd.to_datetime(hist["Date"])
    hist = hist.set_index("Date").sort_index()
    # 原始价（不复权，对比/市值用）
    raw = ak.stock_zh_a_daily(symbol=_norm_a_symbol(symbol), start_date="20130101",
                              end_date=today, adjust="")
    raw = raw.rename(columns={"date": "Date", "close": "Close"})
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.set_index("Date").sort_index()

    index_hist = None
    try:
        idx = ak.stock_zh_index_daily(symbol="sh000300")
        idx = idx.rename(columns={"date": "Date", "close": "Close"})
        idx["Date"] = pd.to_datetime(idx["Date"])
        index_hist = idx.set_index("Date").sort_index()
    except Exception:
        pass

    name = symbol
    try:
        q = _tx_quote(sym6)
        name = q["name"]
    except Exception:
        pass

    return HistoryData(symbol=sym6, market="A", name=name, annual=annual,
                       hist=hist, hist_raw=raw, index_hist=index_hist, erp=0.05)


def _load_us_history(symbol: str) -> HistoryData:
    import akshare as ak
    from ..sec_fetcher import fetch_us_annual_sec

    s = symbol.upper()
    annual = fetch_us_annual_sec(s)
    if annual is None or len(annual) == 0:
        raise RuntimeError(f"美股财务数据缺失: {s}")

    # 新浪美股日线（后复权）
    hist = ak.stock_us_daily(symbol=s)
    hist = hist.rename(columns={"date": "Date", "close": "Close"})
    hist["Date"] = pd.to_datetime(hist["Date"])
    hist = hist.set_index("Date").sort_index()
    # 美股无原始价接口 → hist_raw 用后复权占位，回测时按股本因子还原
    raw = hist.copy()

    index_hist = None
    try:
        idx = ak.index_us_stock_sina(symbol=".INX")
        idx = idx.rename(columns={"date": "Date", "close": "Close"})
        idx["Date"] = pd.to_datetime(idx["Date"])
        index_hist = idx.set_index("Date").sort_index()
    except Exception:
        pass

    # 当前股本（最新年报）→ 还原真实价用
    sh = annual["shares"].dropna()
    current_shares = float(sh.iloc[-1]) if len(sh) else np.nan

    return HistoryData(symbol=s, market="US", name=s, annual=annual,
                       hist=hist, hist_raw=raw, index_hist=index_hist,
                       erp=0.042, current_shares=current_shares)


def _load_rf_series() -> dict[str, pd.Series]:
    """历史 10 年国债收益率序列 {市场: Series(索引=年份)}。"""
    out: dict[str, pd.Series] = {}
    try:
        import akshare as ak
        df = ak.bond_zh_us_rate(start_date="20130101")
        df["日期"] = pd.to_datetime(df["日期"])
        for market, key in [("A", "中国国债收益率10年"), ("US", "美国国债收益率10年")]:
            cols = [c for c in df.columns if key in c]
            if not cols:
                continue
            s = pd.to_numeric(df[cols[0]], errors="coerce").dropna()
            s.index = df.loc[s.index, "日期"]
            yearly = s.groupby(s.index.year).last() / 100.0
            out[market] = yearly
    except Exception:
        pass
    return out


def load_history_data(symbol: str, market: str | None = None) -> HistoryData:
    """加载单只股票全历史数据。market 缺省时自动判断。"""
    from ..data_fetcher import detect_market
    m = market or detect_market(symbol)
    if m == "A":
        return _load_a_history(symbol)
    if m == "US":
        return _load_us_history(symbol)
    raise ValueError(f"无法识别市场: {symbol}")


# ---------------- 单时点回测 ----------------

@dataclass
class BacktestSample:
    symbol: str
    name: str
    market: str
    year: int
    style: str
    price_at: float                # 回测时点真实价（原始价/还原价）
    mcap_at: float                 # 回测时点真实市值
    intrinsic_dcf: float           # DCF 内在价值（每股，真实口径）
    intrinsic_ddm: float
    implied_dcf: float             # DCF 隐含预期收益 (intrinsic/price - 1)
    implied_ddm: float
    actual_1y: float               # 实际 1 年收益（复权价）
    actual_2y: float
    mkt_1y: float
    mkt_2y: float
    excess_1y: float
    excess_2y: float
    error_dcf_1y: float
    error_ddm_1y: float
    note: str = ""


def _year_end_price(hist: pd.DataFrame, year: int) -> float:
    sub = hist[hist.index.year == year]
    if len(sub) == 0:
        return np.nan
    return float(sub["Close"].iloc[-1])


def _hist_beta(stock: pd.DataFrame, index: pd.DataFrame | None, end_date) -> float:
    if index is None:
        return np.nan
    s = stock[stock.index <= end_date].tail(260)["Close"].pct_change().dropna()
    m = index[index.index <= end_date].tail(260)["Close"].pct_change().dropna()
    df = pd.concat([s.rename("s"), m.rename("m")], axis=1, join="inner").dropna()
    if len(df) < 60:
        return np.nan
    cov = np.cov(df["s"], df["m"])
    return float(cov[0, 1] / cov[1, 1])


def _build_cd(annual: pd.DataFrame, price: float, beta: float, rf: float,
              erp: float, market: str, symbol: str, name: str) -> CompanyData:
    cd = CompanyData()
    cd.market = market
    cd.symbol = symbol
    cd.name = name
    cd.currency = "CNY" if market == "A" else "USD"
    cd.price = price
    cd.beta = beta if np.isfinite(beta) else np.nan
    cd.rf = rf
    cd.erp = erp
    cd.annual = annual
    sh = annual["shares"].dropna()
    shares = float(sh.iloc[-1]) if len(sh) else np.nan
    cd.shares = shares
    cd.market_cap = price * shares if np.isfinite(price) and np.isfinite(shares) else np.nan
    return cd


def _year_shares(annual: pd.DataFrame, year: int) -> float:
    sh = annual["shares"].dropna()
    if year in sh.index:
        return float(sh.loc[year])
    # 回退到最近一年
    sub = sh[sh.index <= year]
    return float(sub.iloc[-1]) if len(sub) else np.nan


def backtest_one(
    hd: HistoryData,
    year: int,
    rf_series: pd.Series | None,
    years_ahead: tuple[int, int] = (1, 2),
) -> BacktestSample:
    """在历史 year 年末回测（财务用 year-1 年报，模拟披露滞后）。"""
    ann = hd.annual
    if year - 1 not in ann.index:
        raise ValueError(f"{hd.symbol} 缺 {year-1} 年财务，无法回测 {year}")

    a = ann.loc[:year - 1].copy()
    if len(a) < 3:
        raise ValueError(f"{hd.symbol} 在 {year} 财务历史不足")

    # ---- 价格口径 ----
    price_adj = _year_end_price(hd.hist, year)      # 复权价（收益率基准）
    price_raw = _year_end_price(hd.hist_raw, year)  # 原始价
    if not np.isfinite(price_adj) or price_adj <= 0:
        raise ValueError(f"{hd.symbol} 在 {year} 无行情")

    # 美股：hist_raw 即后复权价（无原始价接口），按股本因子还原为真实价
    s_then = _year_shares(hd.annual, year)
    if hd.market == "US" and np.isfinite(hd.current_shares) and hd.current_shares > 0 \
            and np.isfinite(s_then) and s_then > 0:
        factor = hd.current_shares / s_then
        if 0.5 < factor < 30:  # 防异常股本变化
            price_raw = price_adj / factor
    if not np.isfinite(price_raw) or price_raw <= 0:
        price_raw = price_adj  # 兜底：无原始价时用复权价（口径可能略偏）

    price = price_raw

    # ---- Beta（前 1 年）、无风险利率 ----
    end = dt.datetime(year, 12, 31)
    beta = _hist_beta(hd.hist, hd.index_hist, end)
    rf = np.nan
    if rf_series is not None and len(rf_series):
        rf = float(rf_series.get(year, rf_series.iloc[-1]))

    # ---- 构造 cd 并跑模型 ----
    cd = _build_cd(a, price, beta, rf, hd.erp, hd.market, hd.symbol, hd.name)
    from ..wacc import calc_wacc
    from ..models.dcf import auto_assumptions, run_dcf
    from ..models.ddm import run_ddm
    from ..style_presets import auto_detect_style

    style = auto_detect_style(cd)
    wi = calc_wacc(cd)
    assump = auto_assumptions(cd, wi, style=style)
    dcf = run_dcf(cd, assump)
    ddm = run_ddm(cd, style=style)

    intr_dcf = dcf.per_share_value if not dcf.error and np.isfinite(dcf.per_share_value) else np.nan
    intr_ddm = ddm.per_share_value if not ddm.error and np.isfinite(ddm.per_share_value) else np.nan

    implied_dcf = intr_dcf / price - 1 if np.isfinite(intr_dcf) else np.nan
    implied_ddm = intr_ddm / price - 1 if np.isfinite(intr_ddm) else np.nan

    # ---- 实际表现（复权价收益率）----
    def _ret(end_year: int) -> float:
        p_end = _year_end_price(hd.hist, end_year)
        return p_end / price_adj - 1 if np.isfinite(p_end) else np.nan

    actual_1y = _ret(year + years_ahead[0])
    actual_2y = _ret(year + years_ahead[1])

    def _mkt(end_year: int) -> float:
        if hd.index_hist is None:
            return np.nan
        p_end = _year_end_price(hd.index_hist, end_year)
        p0 = _year_end_price(hd.index_hist, year)
        return p_end / p0 - 1 if np.isfinite(p_end) and np.isfinite(p0) else np.nan

    mkt_1y, mkt_2y = _mkt(year + years_ahead[0]), _mkt(year + years_ahead[1])
    excess_1y = actual_1y - mkt_1y if np.isfinite(actual_1y) and np.isfinite(mkt_1y) else np.nan
    excess_2y = actual_2y - mkt_2y if np.isfinite(actual_2y) and np.isfinite(mkt_2y) else np.nan

    return BacktestSample(
        symbol=hd.symbol, name=hd.name, market=hd.market, year=year, style=style,
        price_at=price, mcap_at=cd.market_cap,
        intrinsic_dcf=intr_dcf, intrinsic_ddm=intr_ddm,
        implied_dcf=implied_dcf, implied_ddm=implied_ddm,
        actual_1y=actual_1y, actual_2y=actual_2y,
        mkt_1y=mkt_1y, mkt_2y=mkt_2y,
        excess_1y=excess_1y, excess_2y=excess_2y,
        error_dcf_1y=(implied_dcf - excess_1y) if np.isfinite(implied_dcf) and np.isfinite(excess_1y) else np.nan,
        error_ddm_1y=(implied_ddm - excess_1y) if np.isfinite(implied_ddm) and np.isfinite(excess_1y) else np.nan,
    )


# ---------------- 批量回测 ----------------

def run_backtest_batch(
    hd: HistoryData,
    years: list[int] | None = None,
    rf_series: pd.Series | None = None,
) -> list[BacktestSample]:
    """对一只股票跑多个历史时点回测。
    美股默认只用 2022-2024（近期拆股已完成，行情口径可靠；
    更早时点新浪美股日线复权口径混乱，待 yfinance 恢复后扩展）。
    """
    if years is None:
        years = [y for y in range(2018, 2025)]
        if hd.market == "US":
            years = [y for y in range(2022, 2025)]
    out = []
    for y in years:
        try:
            out.append(backtest_one(hd, y, rf_series))
        except Exception:
            continue
    return out
