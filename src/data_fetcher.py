# -*- coding: utf-8 -*-
"""
数据抓取层：统一 A股(新浪/腾讯接口) 与 美股(yfinance) 数据为标准化结构。

对外暴露:
    fetch_company(symbol, market=None) -> CompanyData
    detect_market(symbol) -> 'A' | 'US' | None
"""
from __future__ import annotations

import datetime as dt
import re

import numpy as np
import pandas as pd
import requests

# ---------------- 市场识别 ----------------


def detect_market(symbol: str) -> str | None:
    """根据代码规则识别市场。A股纯数字6位；美股含字母。"""
    s = symbol.strip().upper()
    if re.fullmatch(r"\d{6}", s):
        return "A"
    if re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", s):
        return "US"
    return None


def _norm_a_symbol(symbol: str) -> str:
    """标准化 A股代码为 sh/sz/bj + 6位数字。"""
    s = symbol.strip().upper()
    s = re.sub(r"\.(SH|SZ|BJ)$", "", s)
    s = re.sub(r"^(SH|SZ|BJ)", "", s)
    s = re.sub(r"[^\d]", "", s)
    if len(s) != 6:
        raise ValueError(f"无法解析 A股代码: {symbol}")
    if s.startswith(("6", "9")):
        return f"sh{s}"
    if s.startswith(("0", "1", "2", "3")):
        return f"sz{s}"
    if s.startswith(("4", "8")):
        return f"bj{s}"
    return f"sz{s}"


# ---------------- 无风险利率 ----------------


def get_rf(market: str) -> float:
    """获取10年期国债收益率作为无风险利率。失败时用默认值。"""
    try:
        import akshare as ak
        df = ak.bond_zh_us_rate(start_date="20150101")
        if market == "A":
            col = [c for c in df.columns if "中国国债收益率10年" in c]
        else:
            col = [c for c in df.columns if "美国国债收益率10年" in c]
        if col:
            series = pd.to_numeric(df[col[0]], errors="coerce").dropna()
            if len(series):
                return float(series.iloc[-1]) / 100.0
    except Exception:
        pass
    return 0.0200 if market == "US" else 0.0210


def get_erp(market: str) -> float:
    """股权风险溢价 ERP（默认：A股6.0%，美股4.5%）"""
    return 0.060 if market == "A" else 0.045


# ---------------- 统一数据结构 ----------------


class CompanyData:
    """标准化公司数据。annual: DataFrame(索引=年份, 含标准列)"""

    REQUIRED_COLS = [
        "revenue", "operating_income", "ebitda", "net_income", "tax",
        "da", "capex", "change_wc", "total_debt", "cash",
        "dividends", "shares", "total_equity",
    ]

    def __init__(self):
        self.market: str = "A"
        self.symbol: str = ""
        self.name: str = ""
        self.currency: str = "CNY"
        self.price: float = np.nan
        self.shares: float = np.nan          # 总股本(股)
        self.market_cap: float = np.nan      # 总市值(本币)
        self.beta: float = np.nan
        self.rf: float = np.nan
        self.erp: float = np.nan
        self.industry: str = ""
        self.pe_ttm: float = np.nan
        self.pb: float = np.nan
        self.annual: pd.DataFrame | None = None   # 年度财务数据
        self.history: pd.DataFrame | None = None  # 日线(用于beta/图)
        self.notes: list[str] = []                 # 数据口径说明

    # ---- 便捷取数 ----
    def years(self) -> list[int]:
        return list(self.annual.index) if self.annual is not None else []

    def latest_year(self) -> int | None:
        if self.annual is None or len(self.annual) == 0:
            return None
        return int(self.annual.index[-1])

    def col(self, name: str) -> pd.Series:
        return self.annual[name] if self.annual is not None else pd.Series(dtype=float)

    def last_value(self, name: str) -> float:
        if self.annual is None or name not in self.annual.columns:
            return np.nan
        s = self.annual[name].dropna()
        return float(s.iloc[-1]) if len(s) else np.nan

    def latest_price(self) -> float:
        if not np.isnan(self.price):
            return self.price
        if self.history is not None and len(self.history):
            return float(self.history["Close"].iloc[-1])
        return np.nan


# ---------------- 腾讯实时行情（A股基础信息） ----------------

def _tx_quote(symbol6: str) -> dict:
    """腾讯实时行情 → {name, price, shares, market_cap, pe, pb, currency}"""
    url = f"https://qt.gtimg.cn/q={_norm_a_symbol(symbol6)}"
    r = requests.get(url, timeout=10)
    r.encoding = "gbk"
    m = re.search(r'="(.*?)"', r.text)
    if not m:
        raise RuntimeError("腾讯行情返回异常")
    f = m.group(1).split("~")
    out = {"name": f[1] if len(f) > 1 else symbol6}
    try:
        out["price"] = float(f[3])
    except Exception:
        out["price"] = np.nan
    try:
        out["pe"] = float(f[39]) if f[39] else np.nan
    except Exception:
        out["pe"] = np.nan
    try:
        out["pb"] = float(f[46]) if f[46] else np.nan
    except Exception:
        out["pb"] = np.nan
    try:
        out["market_cap"] = float(f[45]) * 1e8 if f[45] else np.nan  # 亿元→元
    except Exception:
        out["market_cap"] = np.nan
    try:
        out["shares"] = float(f[72]) if f[72] else np.nan  # 总股本(股)
    except Exception:
        out["shares"] = np.nan
    out["currency"] = f[82] if len(f) > 82 else "CNY"
    return out


# ---------------- Beta 计算 ----------------


def _calc_beta_a(stock_ret: pd.Series, idx: pd.DataFrame) -> float:
    """A股 Beta：个股日收益率 对 沪深300 回归。"""
    try:
        mkt = idx.set_index(pd.to_datetime(idx["date"]))["close"].pct_change().dropna()
        s = stock_ret.copy()
        s.index = pd.to_datetime(s.index)
        s = s.dropna()
        df = pd.concat([s.rename("s"), mkt.rename("m")], axis=1, join="inner").dropna()
        if len(df) < 60:
            return np.nan
        cov = np.cov(df["s"], df["m"])
        return float(cov[0, 1] / cov[1, 1])
    except Exception:
        return np.nan


def _calc_beta_us(stock_ret: pd.Series) -> float:
    """美股 Beta：个股日收益率 对 标普500(.INX) 回归。"""
    try:
        import akshare as ak
        mkt = ak.index_us_stock_sina(symbol=".INX")
        mkt_ret = mkt.set_index(pd.to_datetime(mkt["date"]))["close"].pct_change().dropna()
        s = stock_ret.copy()
        s.index = pd.to_datetime(s.index)
        s = s.dropna()
        df = pd.concat([s.rename("s"), mkt_ret.rename("m")], axis=1, join="inner").dropna()
        if len(df) < 60:
            return np.nan
        cov = np.cov(df["s"], df["m"])
        return float(cov[0, 1] / cov[1, 1])
    except Exception:
        return np.nan


# ---------------- A股数据抓取 ----------------

def _fetch_a_share(symbol: str) -> CompanyData:
    import akshare as ak

    cd = CompanyData()
    cd.market = "A"
    cd.symbol = symbol
    cd.currency = "CNY"
    cd.rf = get_rf("A")
    cd.erp = get_erp("A")

    # 1. 腾讯实时行情（名称/价格/股本/市值/PE/PB）
    try:
        q = _tx_quote(symbol)
        cd.name = q["name"]
        cd.price = q["price"]
        cd.shares = q["shares"]
        cd.market_cap = q["market_cap"]
        cd.pe_ttm = q["pe"]
        cd.pb = q["pb"]
    except Exception as e:
        cd.notes.append(f"实时行情获取失败: {e}")

    # 2. 新浪日线历史行情
    try:
        sym = _norm_a_symbol(symbol)
        hist = ak.stock_zh_a_daily(
            symbol=sym, start_date="20200101",
            end_date=dt.date.today().strftime("%Y%m%d"), adjust="qfq",
        )
        hist = hist.rename(columns={"date": "Date", "close": "Close",
                                    "open": "Open", "high": "High",
                                    "low": "Low", "volume": "Volume"})
        hist["Date"] = pd.to_datetime(hist["Date"])
        hist = hist.set_index("Date")
        cd.history = hist
        if np.isnan(cd.price) and len(hist):
            cd.price = float(hist["Close"].iloc[-1])
    except Exception as e:
        cd.notes.append(f"历史行情获取失败: {e}")

    # 3. 年度财务报表（新浪）
    cd.annual = _fetch_a_annual_sina(symbol)

    # 4. Beta（沪深300回归）
    try:
        if cd.history is not None and len(cd.history) > 60:
            idx = ak.stock_zh_index_daily(symbol="sh000300")
            ret = cd.history["Close"].pct_change()
            cd.beta = _calc_beta_a(ret, idx)
    except Exception:
        pass

    # 5. 市值兜底
    if np.isnan(cd.market_cap) and not np.isnan(cd.price) and not np.isnan(cd.shares):
        cd.market_cap = cd.price * cd.shares

    return cd


def _fetch_a_annual_sina(symbol: str) -> pd.DataFrame | None:
    """新浪财务三表 → 年度汇总 DataFrame（行=年份，单位元）"""
    import akshare as ak

    try:
        inc = ak.stock_financial_report_sina(stock=symbol, symbol="利润表")
        bal = ak.stock_financial_report_sina(stock=symbol, symbol="资产负债表")
        cas = ak.stock_financial_report_sina(stock=symbol, symbol="现金流量表")
    except Exception:
        return None

    def clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.set_index("报告日")
        df = df.apply(pd.to_numeric, errors="coerce")
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
        df = df.sort_index()
        return df

    inc, bal, cas = clean(inc), clean(bal), clean(cas)
    if len(inc) == 0:
        return None

    def pick(df: pd.DataFrame, *keys):
        for k in keys:
            if k in df.columns:
                return df[k]
        return None

    annual = pd.DataFrame(index=inc.index)
    annual["revenue"] = pick(inc, "营业总收入", "营业收入")
    annual["operating_income"] = pick(inc, "营业利润")
    annual["net_income"] = pick(inc, "归属于母公司所有者的净利润", "净利润")
    annual["tax"] = pick(inc, "所得税费用")
    annual["total_equity"] = pick(bal, "归属于母公司股东权益合计")

    # 累计折旧原始值（年报间 diff 得到全年折旧）
    dep_col = pick(bal, "累计折旧")
    annual["_accum_dep"] = dep_col if dep_col is not None else np.nan

    annual["capex"] = pick(
        cas, "购建固定资产、无形资产和其他长期资产所支付的现金"
    )
    annual["cfo"] = pick(cas, "经营活动产生的现金流量净额")

    # 分红：分配股利、利润或偿付利息所支付的现金
    div_col = pick(cas, "分配股利、利润或偿付利息所支付的现金")
    annual["dividends"] = div_col

    def series_or_zero(df: pd.DataFrame, *keys) -> pd.Series:
        s = pick(df, *keys)
        if s is None:
            return pd.Series(0.0, index=annual.index)
        return s.reindex(annual.index).fillna(0.0)

    # 有息负债
    st = series_or_zero(bal, "短期借款")
    lt = series_or_zero(bal, "长期借款")
    bond = series_or_zero(bal, "应付债券")
    annual["total_debt"] = st + lt + bond

    annual["cash"] = pick(bal, "货币资金")

    # 总股本：实收资本(或股本) / 面值1元
    paid = pick(bal, "实收资本(或股本)")
    annual["shares"] = paid

    # 营运资本原始值（年报间 diff 得全年变动）
    ca = series_or_zero(bal, "流动资产合计")
    cl = series_or_zero(bal, "流动负债合计")
    annual["_wc"] = ca - cl

    # 只取年报，年份索引
    annual = annual[annual.index.month == 12]
    annual["year"] = annual.index.year
    annual = annual.groupby("year").last()

    # 全年折旧 = 年报累计折旧差
    annual["da"] = annual["_accum_dep"].diff()
    # 全年营运资本变动 = 年报 WC 差（正=占用增加流出）
    annual["change_wc"] = annual["_wc"].diff()
    annual = annual.drop(columns=["_accum_dep", "_wc"])

    # EBITDA 估算
    annual["ebitda"] = annual["operating_income"] + annual["da"].fillna(0)

    # 统一列
    for c in CompanyData.REQUIRED_COLS:
        if c not in annual.columns:
            annual[c] = np.nan
    annual = annual.reindex(sorted(annual.index))
    return annual


# ---------------- 美股数据抓取 ----------------

def _us_quote(symbol: str) -> dict:
    """腾讯美股实时行情 usAAPL → {name, price, pe, market_cap, shares, currency}"""
    url = f"https://qt.gtimg.cn/q=us{symbol}"
    r = requests.get(url, timeout=10)
    r.encoding = "gbk"
    m = re.search(r'="(.*?)"', r.text)
    if not m or "pv_none_match" in r.text:
        raise RuntimeError(f"腾讯美股行情无此代码: {symbol}")
    f = m.group(1).split("~")
    out = {"name": f[1] if len(f) > 1 else symbol}
    try:
        out["price"] = float(f[3])
    except Exception:
        out["price"] = np.nan
    try:
        out["pe"] = float(f[39]) if f[39] else np.nan
    except Exception:
        out["pe"] = np.nan
    try:
        # 总市值单位：亿美元 → 美元
        out["market_cap"] = float(f[45]) * 1e8 if f[45] else np.nan
    except Exception:
        out["market_cap"] = np.nan
    try:
        out["shares"] = float(f[62]) if f[62] else np.nan
    except Exception:
        out["shares"] = np.nan
    out["currency"] = f[35] if len(f) > 35 else "USD"
    return out


def _fetch_us_stock(symbol: str) -> CompanyData:
    cd = CompanyData()
    cd.market = "US"
    cd.symbol = symbol.upper()
    cd.currency = "USD"
    cd.rf = get_rf("US")
    cd.erp = get_erp("US")

    # 1. 腾讯行情（名称/价格/PE/市值/股本）
    try:
        q = _us_quote(cd.symbol)
        cd.name = q["name"]
        cd.price = q["price"]
        cd.pe_ttm = q["pe"]
        cd.market_cap = q["market_cap"]
        cd.shares = q["shares"]
    except Exception as e:
        cd.notes.append(f"腾讯美股行情失败: {e}")

    # 2. 新浪美股日线（历史行情）
    try:
        import akshare as ak
        hist = ak.stock_us_daily(symbol=cd.symbol)
        hist = hist.rename(columns={"date": "Date", "close": "Close",
                                    "open": "Open", "high": "High",
                                    "low": "Low", "volume": "Volume"})
        hist["Date"] = pd.to_datetime(hist["Date"])
        hist = hist.set_index("Date")
        cd.history = hist
        if np.isnan(cd.price) and len(hist):
            cd.price = float(hist["Close"].iloc[-1])
    except Exception as e:
        cd.notes.append(f"新浪美股日线失败: {e}")

    # 3. 财务三表（优先 SEC EDGAR 官方数据；失败则 yfinance 重试）
    try:
        from .sec_fetcher import fetch_us_annual_sec
        cd.annual = fetch_us_annual_sec(cd.symbol)
        if cd.annual is not None and len(cd.annual):
            cd.notes.append("财务数据来源：SEC EDGAR")
    except Exception:
        pass
    if cd.annual is None or len(cd.annual) == 0:
        try:
            cd.annual = _fetch_us_annual_retry(cd.symbol)
            if cd.annual is not None and len(cd.annual):
                cd.notes.append("财务数据来源：Yahoo Finance")
        except Exception as e:
            cd.notes.append(f"财务数据获取失败: {e}")

    # 4. Beta（自算，用新浪数据）
    if cd.history is not None and len(cd.history) > 60:
        cd.beta = _calc_beta_us(cd.history["Close"].pct_change())

    # 5. PB：从 SEC 净资产计算（腾讯美股字段无可靠 PB）
    if (np.isnan(cd.pb) and cd.annual is not None and len(cd.annual)):
        eq = cd.last_value("total_equity")
        if np.isfinite(eq) and eq > 0 and np.isfinite(cd.market_cap) and cd.market_cap > 0:
            cd.pb = cd.market_cap / eq

    # 6. 市值/股本兜底
    if np.isnan(cd.market_cap) and not np.isnan(cd.price) and not np.isnan(cd.shares):
        cd.market_cap = cd.price * cd.shares

    return cd


def _yf_session():
    import requests as _r
    s = _r.Session()
    s.headers["User-Agent"] = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                               "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36")
    return s


def _fetch_us_annual_retry(symbol: str, retries: int = 3) -> pd.DataFrame | None:
    """yfinance 取美股财务，带限流重试。"""
    import time
    last_err = None
    for attempt in range(retries):
        try:
            import yfinance as yf
            tk = yf.Ticker(symbol, session=_yf_session())
            return _fetch_us_annual(tk)
        except Exception as e:
            last_err = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"美股财务数据获取失败(可能被限流): {last_err}")


def _fetch_us_annual(tk) -> pd.DataFrame | None:
    """yfinance 三表 → 年度 DataFrame（行=年份）"""
    try:
        income = tk.income_stmt
        balance = tk.balance_sheet
        cashflow = tk.cashflow
    except Exception:
        return None

    def pick(df, *keys):
        for k in keys:
            for c in df.index:
                if str(c).strip().lower() == k.lower() or k.lower() in str(c).lower():
                    return df.loc[c]
        return None

    dates = sorted({d for d in list(income.columns) + list(balance.columns) + list(cashflow.columns)
                    if d is not None}, reverse=True)
    rows = []
    for d in dates:
        def v(df, *keys, default=np.nan):
            s = pick(df, *keys)
            if s is None or d not in df.columns:
                return default
            try:
                x = s.get(d)
                return float(x) if x is not None and not pd.isna(x) else default
            except Exception:
                return default

        row = {"year": d.year}
        row["revenue"] = v(income, "Total Revenue")
        row["operating_income"] = v(income, "Operating Income")
        row["net_income"] = v(income, "Net Income")
        row["tax"] = v(income, "Tax Provision")
        row["da"] = v(cashflow, "Depreciation & Amortization", "Depreciation And Amortization",
                     "Depreciation", default=np.nan)
        row["ebitda"] = v(income, "EBITDA")
        row["capex"] = abs(v(cashflow, "Capital Expenditure",
                             "Purchase Of Property, Plant And Equipment", default=np.nan))
        row["cfo"] = v(cashflow, "Operating Cash Flow")
        row["total_debt"] = v(balance, "Total Debt",
                              default=v(balance, "Long Term Debt", default=0)
                              + v(balance, "Current Debt", default=0))
        row["cash"] = v(balance, "Cash And Cash Equivalents")
        row["total_equity"] = v(balance, "Stockholders Equity")
        row["dividends"] = v(cashflow, "Cash Dividends Paid", "Common Stock Dividend Paid")
        row["change_wc"] = np.nan
        row["shares"] = np.nan
        rows.append(row)

    if not rows:
        return None
    annual = pd.DataFrame(rows).set_index("year").sort_index()
    try:
        sh = float(tk.info.get("sharesOutstanding") or np.nan)
        annual["shares"] = sh
    except Exception:
        pass
    annual["ebitda"] = annual["ebitda"].fillna(annual["operating_income"] + annual["da"].fillna(0))
    for c in CompanyData.REQUIRED_COLS:
        if c not in annual.columns:
            annual[c] = np.nan
    return annual.reindex(sorted(annual.index))


# ---------------- 对外入口 ----------------


def fetch_company(symbol: str, market: str | None = None) -> CompanyData:
    """主入口：输入代码(可含市场后缀)，返回标准化 CompanyData。"""
    symbol = symbol.strip().upper()
    if market is None:
        market = detect_market(symbol)
    if market == "US":
        return _fetch_us_stock(re.sub(r"\.US$", "", symbol))
    if market == "A":
        return _fetch_a_share(_norm_a_symbol(symbol).lstrip("shszbj"))
    raise ValueError(f"无法识别市场: {symbol}")
