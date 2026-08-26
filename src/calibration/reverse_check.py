# -*- coding: utf-8 -*-
"""
反向修正检查：传统教科书 DCF vs 增强模型 DCF 双跑对比。

原理：传统 DCF（classic_assumptions，无风格/无校准/中性参数）作为「无偏锚」，
在历史回测样本上重建时点，分别用传统与增强模型估值，对比隐含收益与实际
1 年超额收益的误差。若某层增强模型「更优%」显著低于 50%，说明该层风格预设
引入了无根据的偏差，应反向修正该层参数。

用法：.venv\\Scripts\\python.exe -m src.calibration.reverse_check A
产物：data/cache/diagnostic_classic_vs_enhanced.csv（全样本明细）
"""
from __future__ import annotations

import datetime as dt
import os
import pickle

import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "data", "cache")


def _year_end_price(hist: pd.DataFrame, year: int) -> float:
    sub = hist[hist.index.year == year]
    return float(sub["Close"].iloc[-1]) if len(sub) else np.nan


def _hist_beta(stock: pd.DataFrame, index: pd.DataFrame | None, end_date) -> float:
    if index is None:
        return np.nan
    s = stock[stock.index <= end_date].tail(260)["Close"].pct_change().dropna()
    m = index[index.index <= end_date].tail(260)["Close"].pct_change().dropna()
    dd = pd.concat([s.rename("s"), m.rename("m")], axis=1, join="inner").dropna()
    if len(dd) < 60:
        return np.nan
    cov = np.cov(dd["s"], dd["m"])
    return float(cov[0, 1] / cov[1, 1])


def _build_cd(annual, price, beta, rf, erp, market, symbol, name):
    from ..data_fetcher import CompanyData
    cd = CompanyData()
    cd.market = market; cd.symbol = symbol; cd.name = name
    cd.currency = "CNY" if market == "A" else "USD"
    cd.price = price; cd.beta = beta if np.isfinite(beta) else np.nan
    cd.rf = rf; cd.erp = erp; cd.annual = annual
    sh = annual["shares"].dropna()
    cd.shares = float(sh.iloc[-1]) if len(sh) else np.nan
    cd.market_cap = price * cd.shares if np.isfinite(price) and np.isfinite(cd.shares) else np.nan
    return cd


def run_reverse_check(market: str = "A") -> pd.DataFrame:
    """对历史回测样本做传统 vs 增强双跑对比，返回明细 DataFrame。"""
    from ..models.dcf import run_dcf, classic_assumptions, auto_assumptions
    from ..style_presets import auto_detect_style
    from ..wacc import calc_wacc

    samples_csv = os.path.join(CACHE_DIR, f"backtest_samples_{market}.csv")
    if not os.path.exists(samples_csv):
        raise FileNotFoundError(f"缺历史样本 {samples_csv}，先跑 runner 生成")
    df = pd.read_csv(samples_csv)
    rows = []
    for _, r in df.iterrows():
        sym, yr = r["symbol"], int(r["year"])
        p = os.path.join(CACHE_DIR, f"hist_{market}_{sym}.pkl")
        if not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            hd = pickle.load(f)
        ann = hd.annual
        if yr - 1 not in ann.index or len(ann.loc[:yr - 1]) < 3:
            continue
        a = ann.loc[:yr - 1].copy()
        price_adj = _year_end_price(hd.hist, yr)
        if not np.isfinite(price_adj) or price_adj <= 0:
            continue
        price_raw = _year_end_price(hd.hist_raw, yr)
        if not np.isfinite(price_raw) or price_raw <= 0:
            price_raw = price_adj
        end = dt.datetime(yr, 12, 31)
        beta = _hist_beta(hd.hist, hd.index_hist, end)
        rf = np.nan
        if hd.rf_series is not None and len(hd.rf_series):
            rf = float(hd.rf_series.get(yr, hd.rf_series.iloc[-1]))
        cd = _build_cd(a, price_raw, beta, rf, hd.erp, market, hd.symbol, hd.name)

        style = auto_detect_style(cd)
        wi = calc_wacc(cd)
        c_v = None
        e_v = None
        try:
            c_d = run_dcf(cd, classic_assumptions(cd, wi))
            c_v = c_d.per_share_value if not c_d.error and np.isfinite(c_d.per_share_value) else np.nan
        except Exception:
            pass
        try:
            e_d = run_dcf(cd, auto_assumptions(cd, wi, style=style))
            e_v = e_d.per_share_value if not e_d.error and np.isfinite(e_d.per_share_value) else np.nan
        except Exception:
            pass

        def _ret(ey):
            p_end = _year_end_price(hd.hist, ey)
            return p_end / price_adj - 1 if np.isfinite(p_end) else np.nan

        def _mkt(ey):
            if hd.index_hist is None:
                return np.nan
            p_end = _year_end_price(hd.index_hist, ey)
            p0 = _year_end_price(hd.index_hist, yr)
            return p_end / p0 - 1 if np.isfinite(p_end) and np.isfinite(p0) else np.nan

        actual = _ret(yr + 1)
        exc = actual - _mkt(yr + 1) if np.isfinite(actual) and np.isfinite(_mkt(yr + 1)) else np.nan
        if not np.isfinite(exc):
            continue
        c_imp = c_v / price_raw - 1 if np.isfinite(c_v) else np.nan
        e_imp = e_v / price_raw - 1 if np.isfinite(e_v) else np.nan
        rows.append({
            "symbol": sym, "name": r["name"], "year": yr, "style": style,
            "price": price_raw, "excess": exc,
            "classic_imp": c_imp, "enhanced_imp": e_imp,
            "classic_err": c_imp - exc if np.isfinite(c_imp) else np.nan,
            "enhanced_err": e_imp - exc if np.isfinite(e_imp) else np.nan,
        })
    res = pd.DataFrame(rows)
    out = os.path.join(CACHE_DIR, "diagnostic_classic_vs_enhanced.csv")
    res.to_csv(out, index=False)
    return res


def _wmed(x):
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return np.nan
    lo, hi = np.percentile(x, 5), np.percentile(x, 95)
    return float(np.median(np.clip(x, lo, hi)))


def _hitrate(g: pd.DataFrame, imp_col: str) -> float:
    v = g[np.isfinite(g[imp_col]) & np.isfinite(g["excess"]) & (g["excess"].abs() > 0.02)]
    if len(v) == 0:
        return np.nan
    return float(((v[imp_col] > 0) == (v["excess"] > 0)).mean())


def print_report(res: pd.DataFrame) -> None:
    print(f"双跑样本: {len(res)}")
    print(f"\n{'层':<10}{'样本':>5} | {'传统中位误差':>10} {'增强中位误差':>10} "
          f"{'传统命中':>8} {'增强命中':>8} | {'增强更优%':>8}")
    for st in ["growth", "steady", "value", "cyclical"]:
        g = res[res["style"] == st]
        if len(g) == 0:
            continue
        cm = _wmed(g["classic_err"]); em = _wmed(g["enhanced_err"])
        ch, eh = _hitrate(g, "classic_imp"), _hitrate(g, "enhanced_imp")
        both = g[np.isfinite(g["classic_err"]) & np.isfinite(g["enhanced_err"])]
        better = float((both["enhanced_err"].abs() < both["classic_err"].abs()).mean()) if len(both) else np.nan
        verdict = "增强有效" if (np.isfinite(better) and better >= 0.5) else "⚠增强有害→需反向修正"
        print(f"{st:<10}{len(g):>5} | {cm:>10.3f} {em:>10.3f} {ch:>8.0%} {eh:>8.0%} | {better:>8.0%}  {verdict}")


if __name__ == "__main__":
    import sys
    mkt = sys.argv[1] if len(sys.argv) > 1 else "A"
    res = run_reverse_check(mkt)
    print_report(res)
