# -*- coding: utf-8 -*-
"""
股票池构建：A股总市值 ≥ 500 亿人民币；美股总市值 ≥ 1000 亿美元。
用于回测校准的样本范围，避免小市值/流动性差的股票噪声。
"""
from __future__ import annotations


def _build_a_pool(min_mcap_cny: float = 500e8, limit: int = 200) -> list[str]:
    """A股：新浪实时行情（含总市值，mktcap 单位万元），筛选市值≥min_mcap 元。

    说明：akshare 的东财 stock_zh_a_spot_em 默认 UA 被风控拒绝，改走新浪
    Market_Center.getHQNodeData（带浏览器 UA 实测可用），按总市值降序分页拉取。
    """
    import requests

    url = "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": "https://finance.sina.com.cn/",
    }
    codes: list[str] = []
    page = 1
    while len(codes) < limit and page <= 200:
        params = {
            "page": str(page), "num": "100", "sort": "mktcap", "asc": "0",
            "node": "hs_a", "symbol": "", "_s_r_a": "page",
        }
        r = requests.get(url, params=params, headers=headers, timeout=25)
        r.raise_for_status()
        arr = r.json()
        if not isinstance(arr, list) or not arr:
            break
        for it in arr:
            try:
                mktcap_yuan = float(it.get("mktcap", 0)) * 1e4  # 万元 -> 元
            except (TypeError, ValueError):
                continue
            # 按市值降序分页：一旦低于门槛，后续页都更小，提前结束
            if mktcap_yuan < min_mcap_cny:
                return codes[:limit]
            code = str(it.get("code", "")).zfill(6)
            # 排除北交所（4/8/920 开头），保持财务数据源一致（新浪接口）
            if code.startswith(("4", "8", "920")):
                continue
            codes.append(code)
            if len(codes) >= limit:
                return codes[:limit]
        page += 1
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
