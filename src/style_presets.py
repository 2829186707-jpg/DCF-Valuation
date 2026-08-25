# -*- coding: utf-8 -*-
"""
估值风格预设系统：按股票属性（成长/稳健/价值/周期）套用不同的 DCF 参数组合。

自动识别规则（auto）：
  1. 历史收入 CAGR > 15%           → 成长股
  2. 近8年营收变异系数 > 0.35       → 周期股（高波动）
  3. 历史收入 CAGR < 5% 且股息率>2% → 价值股
  4. 其余                          → 稳健股
"""
from __future__ import annotations

import numpy as np

from .data_fetcher import CompanyData

STYLES = ["auto", "growth", "steady", "value", "cyclical"]

STYLE_LABELS = {
    "auto": "自动识别",
    "growth": "成长股",
    "steady": "稳健股",
    "value": "价值股",
    "cyclical": "周期股",
}

# 每风格参数：
#   growth_discount  : 历史增速折扣（1.0=不打折，<1=保守折扣）
#   decline_divisor  : 增长率衰减除数（None=显式期不衰减持平；数值越小衰减越快）
#   accel            : 显式期逐年加速幅度（0=持平；>0 支持成长股加速情景）
#   terminal_g       : 永续增长率 {市场: 值}
#   erp_adj          : 股权风险溢价调整（相对基准：A股5.0%/美股4.2%）
#   margin_window    : 营业利润率统计窗口（年）
#   normalize        : 是否用周期正常化盈利（周期股用）
STYLE_PRESETS: dict[str, dict] = {
    "growth": {
        "label": "成长股",
        "growth_discount": 1.0,
        "decline_divisor": None,
        "accel": 0.01,
        "terminal_g": {"A": 0.040, "US": 0.030},
        "erp_adj": 0.002,
        "margin_window": 3,
        "normalize": False,
    },
    "steady": {
        "label": "稳健股",
        "growth_discount": 0.90,
        "decline_divisor": None,
        "accel": 0.0,
        "terminal_g": {"A": 0.032, "US": 0.025},
        "erp_adj": 0.0,
        "margin_window": 3,
        "normalize": False,
    },
    "value": {
        "label": "价值股",
        "growth_discount": 0.75,
        "decline_divisor": 5,
        "accel": 0.0,
        "terminal_g": {"A": 0.025, "US": 0.020},
        "erp_adj": -0.003,
        "margin_window": 3,
        "normalize": False,
    },
    "cyclical": {
        "label": "周期股",
        "growth_discount": 0.80,
        "decline_divisor": 8,
        "accel": 0.0,
        "terminal_g": {"A": 0.028, "US": 0.022},
        "erp_adj": 0.0,
        "margin_window": 8,
        "normalize": True,
    },
}

# ERP 基准（与 wacc 模块一致）
BASE_ERP = {"A": 0.050, "US": 0.042}


def style_erp(style: str, market: str) -> float:
    """按风格调整后的 ERP。"""
    if style == "auto":
        return BASE_ERP[market]
    ps = STYLE_PRESETS.get(style, STYLE_PRESETS["steady"])
    return BASE_ERP[market] + ps.get("erp_adj", 0.0)


def style_terminal_g(style: str, market: str) -> float:
    if style == "auto":
        return STYLE_PRESETS["steady"]["terminal_g"][market]
    ps = STYLE_PRESETS.get(style, STYLE_PRESETS["steady"])
    return ps["terminal_g"][market]


def _hist_cagr(cd: CompanyData, years: int = 5) -> float:
    """近 N 年收入 CAGR。"""
    ann = cd.annual
    if ann is None or len(ann) == 0:
        return np.nan
    rev = ann["revenue"].dropna()
    if len(rev) < 2:
        return np.nan
    r = rev.tail(years)
    if len(r) < 2:
        r = rev
    return float((r.iloc[-1] / r.iloc[0]) ** (1 / (len(r) - 1)) - 1)


def auto_detect_style(cd: CompanyData) -> str:
    """自动识别公司属性。"""
    g = _hist_cagr(cd, 5)
    if np.isfinite(g) and g > 0.15:
        return "growth"

    # 周期特征：近8年营业利润率变异系数（利润率大幅波动=典型周期股）
    # 辅助信号：营收变异系数 > 0.5（排除数据异常后仍显著波动）
    ann = cd.annual
    if ann is not None and len(ann) >= 4:
        margin = (ann["operating_income"] / ann["revenue"]).dropna().tail(8)
        if len(margin) >= 4 and margin.mean() > 0:
            cv_m = float(margin.std() / abs(margin.mean()))
            if cv_m > 0.5:
                return "cyclical"
        rev = ann["revenue"].dropna().tail(8)
        if len(rev) >= 4 and rev.mean() > 0:
            cv_r = float(rev.std() / rev.mean())
            if cv_r > 0.5:
                return "cyclical"

    # 低增长 + 高股息 → 价值
    if np.isfinite(g) and g < 0.05:
        div = cd.last_value("dividends")
        mcap = cd.market_cap
        if np.isfinite(div) and div > 0 and np.isfinite(mcap) and mcap > 0:
            if div / mcap > 0.02:
                return "value"
    return "steady"


def normalize_revenue(cd: CompanyData, window: int = 8) -> float | None:
    """周期股正常化收入：近 N 年收入中位数（避免周期峰谷失真）。

    仅当最新营收与历史中位数处于同一量级（0.5~2.0 倍）时启用，
    避免把「已成长上台阶」的公司（如最新营收数倍于历史均值）错误拉回低基数。
    """
    ann = cd.annual
    if ann is None or len(ann) == 0:
        return None
    rev = ann["revenue"].dropna().tail(window)
    if len(rev) >= 4:
        med = float(rev.median())
        latest = float(rev.iloc[-1])
        if med > 0 and latest > 0:
            ratio = latest / med
            if 0.5 <= ratio <= 2.0:
                return med
            return None  # 已上台阶，不适用均值回归
    return None


def resolve_style(style: str, cd: CompanyData) -> tuple[str, dict]:
    """把 'auto' 解析为具体风格，返回 (实际风格, 参数表)。"""
    s = style if style in STYLE_PRESETS else "steady"
    if s == "auto":
        s = auto_detect_style(cd)
    return s, STYLE_PRESETS.get(s, STYLE_PRESETS["steady"])
