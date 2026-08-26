# -*- coding: utf-8 -*-
"""
股票池构建：A股总市值 ≥ 500 亿人民币；美股总市值 ≥ 1000 亿美元。
用于回测校准的样本范围，避免小市值/流动性差的股票噪声。
"""
from __future__ import annotations


def _build_a_pool(min_mcap_cny: float = 500e8, limit: int = 200) -> list[str]:
    """A股：东财实时行情（含总市值），筛选市值≥min_mcap 元。"""
    import akshare as ak
    df = ak.stock_zh_a_spot_em()
    df["总市值"] = df["总市值"].astype(float)
    hit = df[df["总市值"] >= min_mcap_cny]
    hit = hit.sort_values("总市值", ascending=False)
    # 排除非A股（如北交所代码规则），取 6 位代码
    codes = [str(c).zfill(6) for c in hit["代码"].tolist()]
    # 排除北交所（8开头/4开头）以保持财务数据源一致（新浪接口）
    codes = [c for c in codes if not c.startswith(("4", "8"))]
    return codes[:limit]


def _build_us_pool(min_mcap_usd: float = 1000e8, limit: int = 200) -> list[str]:
    """美股：东财美股实时行情（含总市值美元），筛选市值≥min_mcap 美元。"""
    import akshare as ak
    df = ak.stock_us_spot_em()
    df["总市值"] = df["总市值"].astype(float)
    hit = df[df["总市值"] >= min_mcap_usd]
    hit = hit.sort_values("总市值", ascending=False)
    return hit["代码"].tolist()[:limit]


def build_stock_pool(market: str, limit: int = 200) -> list[str]:
    """构建校准股票池。market: 'A' 或 'US'。"""
    if market == "A":
        return _build_a_pool(limit=limit)
    if market == "US":
        return _build_us_pool(limit=limit)
    raise ValueError(f"未知市场: {market}")
