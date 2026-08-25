# -*- coding: utf-8 -*-
"""
SEC EDGAR 美股财务数据抓取（美国证监会官方公开数据，免费、无需 key）。

从 companyfacts API 解析 XBRL 事实，输出标准年度 DataFrame（行=财年）。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import requests

SEC_UA = "DCF-Valuation-Tool (contact: research@example.com)"
_TICKERS_CACHE: dict | None = None
_cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache")
os.makedirs(_cache_dir, exist_ok=True)


def _get_json(url: str, cache_file: str | None = None):
    cache_path = os.path.join(_cache_dir, cache_file) if cache_file else None
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    r = requests.get(url, headers={"User-Agent": SEC_UA}, timeout=20)
    r.raise_for_status()
    data = r.json()
    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    return data


def cik_for_ticker(ticker: str) -> str | None:
    """ticker → 10位 CIK。"""
    global _TICKERS_CACHE
    if _TICKERS_CACHE is None:
        _TICKERS_CACHE = _get_json("https://www.sec.gov/files/company_tickers.json",
                                   "company_tickers.json")
    t = ticker.strip().upper()
    for v in _TICKERS_CACHE.values():
        if v.get("ticker", "").upper() == t:
            return str(v["cik_str"]).zfill(10)
    return None


def _annual_flows(facts: dict, concept: str) -> dict[int, float]:
    """流量科目（收入/利润/现金流等）：取年报(10-K)期间值。返回 {财年: 值}

    注意：10-K 里往往同时披露分季度明细（end 为季度末）与财年累计值，
    必须按「期间最长」取完整年度值，否则会用 Q1 等季度值覆盖年度值。
    """
    if concept not in facts:
        return {}
    best: dict[int, tuple[float, int]] = {}
    for unit, arr in facts[concept].get("units", {}).items():
        if unit not in ("USD",):
            continue
        for item in arr:
            end = item.get("end")
            start = item.get("start")
            form = item.get("form", "")
            if not end:
                continue
            dur = None
            if start:
                try:
                    dur = (pd.to_datetime(end) - pd.to_datetime(start)).days
                except Exception:
                    dur = None
            # 年度判定：有 duration 时须接近完整年度；无 duration 时仅接受 10-K
            is_annual = False
            if dur is not None:
                is_annual = (345 <= dur <= 380)
            elif form == "10-K":
                is_annual = True
            if not is_annual:
                continue
            try:
                year = int(pd.to_datetime(end).year)
            except Exception:
                continue
            # 同一年取期间最长（最接近完整年度）的一条
            prev = best.get(year)
            if prev is None or (dur or 0) > prev[1]:
                best[year] = (float(item["val"]), dur or 0)
    return {y: v[0] for y, v in best.items()}


def _annual_stocks(facts: dict, concept: str) -> dict[int, float]:
    """存量科目（现金/债务/权益等）：取 10-K 财年末值。返回 {财年: 值}"""
    if concept not in facts:
        return {}
    best: dict[int, tuple[float, str]] = {}
    for unit, arr in facts[concept].get("units", {}).items():
        if unit not in ("USD", "shares"):
            continue
        for item in arr:
            end = item.get("end")
            form = item.get("form", "")
            if not end or form != "10-K":
                continue
            try:
                year = int(pd.to_datetime(end).year)
            except Exception:
                continue
            # 同一年取 end 日期最晚的一条（财年末值）
            prev = best.get(year)
            if prev is None or end > prev[1]:
                best[year] = (float(item["val"]), end)
    return {y: v[0] for y, v in best.items()}


def fetch_us_annual_sec(ticker: str) -> pd.DataFrame | None:
    """从 SEC EDGAR 拉取并解析美股年度财务数据。"""
    cik = cik_for_ticker(ticker)
    if not cik:
        return None
    try:
        cf = _get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
                       f"companyfacts_{cik}.json")
    except Exception:
        return None
    facts = cf.get("facts", {}).get("us-gaap", {})
    if not facts:
        return None

    def flow(*keys):
        for k in keys:
            d = _annual_flows(facts, k)
            if d:
                return d
        return {}

    def stock(*keys):
        for k in keys:
            d = _annual_stocks(facts, k)
            if d:
                return d
        return {}

    revenue = flow("RevenueFromContractWithCustomerExcludingAssessedTax",
                   "Revenues", "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax")
    op_income = flow("OperatingIncomeLoss", "OperatingIncome")
    net_income = flow("NetIncomeLoss", "ProfitLoss")
    tax = flow("IncomeTaxExpenseBenefit", "IncomeTaxes")
    da = flow("DepreciationDepletionAndAmortization",
              "DepreciationAmortizationAndAccretionNet",
              "DepreciationAndAmortization")
    capex = flow("PaymentsToAcquirePropertyPlantAndEquipment",
                 "PaymentsToAcquireProductiveAssets")
    cfo = flow("NetCashProvidedByUsedInOperatingActivities",
               "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")
    dividends = flow("PaymentsOfDividends",
                     "PaymentsOfDividendsCommonStock",
                     "PaymentsOfDividendsToCommonStockholders")
    # 负债/现金/权益/股本（存量）
    debt_cur = stock("DebtCurrent", "ShortTermBorrowings", "LongTermDebtCurrent")
    debt_lt = stock("LongTermDebtNoncurrent", "LongTermDebt")
    cash = stock("CashAndCashEquivalentsAtCarryingValue",
                 "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents")
    equity = stock("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")
    assets_cur = stock("AssetsCurrent")
    liab_cur = stock("LiabilitiesCurrent")
    shares_dil = stock("WeightedAverageNumberOfDilutedSharesOutstanding")
    shares_out = stock("CommonStockSharesOutstanding")

    years = sorted(set(list(revenue.keys()) + list(net_income.keys()) + list(cash.keys())))
    if not years:
        return None

    rows = []
    for y in years:
        rows.append({
            "year": y,
            "revenue": revenue.get(y, np.nan),
            "operating_income": op_income.get(y, np.nan),
            "net_income": net_income.get(y, np.nan),
            "tax": tax.get(y, np.nan),
            "da": da.get(y, np.nan),
            "capex": capex.get(y, np.nan),
            "cfo": cfo.get(y, np.nan),
            "dividends": dividends.get(y, np.nan),
            "total_debt": (debt_cur.get(y, 0) or 0) + (debt_lt.get(y, 0) or 0),
            "cash": cash.get(y, np.nan),
            "total_equity": equity.get(y, np.nan),
            "shares": shares_dil.get(y, shares_out.get(y, np.nan)),
            "_assets_cur": assets_cur.get(y, np.nan),
            "_liab_cur": liab_cur.get(y, np.nan),
        })

    annual = pd.DataFrame(rows).set_index("year").sort_index()
    # 营运资本变动
    wc = annual["_assets_cur"].fillna(0) - annual["_liab_cur"].fillna(0)
    annual["change_wc"] = wc.diff()
    annual = annual.drop(columns=["_assets_cur", "_liab_cur"])
    annual["capex"] = annual["capex"].abs()
    annual["ebitda"] = annual["operating_income"] + annual["da"].fillna(0)

    # 统一列
    from .data_fetcher import CompanyData
    for c in CompanyData.REQUIRED_COLS:
        if c not in annual.columns:
            annual[c] = np.nan
    return annual.reindex(sorted(annual.index))
