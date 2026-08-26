# -*- coding: utf-8 -*-
"""
DCF 估值模型（FCFF 两阶段）。

流程：
  历史数据 → 自动生成基准假设 → 显式预测期 FCFF → 终值 → EV → 股权价值 → 每股价值
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data_fetcher import CompanyData
from ..wacc import calc_wacc


@dataclass
class DCFAssumptions:
    forecast_years: int = 5
    revenue_growth: float = 0.05          # 显式期首年收入增长率
    growth_decline: float = 0.0           # 每年增长率衰减(百分点)，0=恒定
    accel: float = 0.0                    # 每年增长率加速(百分点)，>0 支持成长股加速
    terminal_growth: float = 0.025        # 永续增长率
    operating_margin: float = 0.20        # 目标营业利润率
    tax_rate: float = 0.25
    capex_pct: float = 0.05               # 资本开支/收入
    da_pct: float = 0.05                  # 折旧摊销/收入
    nwc_pct: float = 0.00                 # 营运资本变动/收入(正=占用增加流出)
    beta: float = 1.0
    debt_rate: float = 0.040
    erp: float | None = None
    rf: float | None = None
    base_rev: float | None = None         # 周期股正常化收入基准（None=用最新年）


@dataclass
class DCFResult:
    per_share_value: float = np.nan
    upside: float = np.nan
    ev: float = np.nan
    equity_value: float = np.nan
    wacc: float = np.nan
    terminal_value: float = np.nan
    terminal_pv: float = np.nan
    terminal_ratio: float = np.nan        # 终值现值占EV比例
    forecast: pd.DataFrame | None = None
    assumptions: dict = field(default_factory=dict)
    conclusion: str = ""
    detail: str = ""
    note: str = ""
    error: str = ""


# ---------------- 自动生成基准假设 ----------------

def _safe_cagr(series: pd.Series) -> float:
    """稳健 CAGR：仅当首尾均为正有限数时计算；否则返回 NaN。

    避免 0/负值/异常数据导致 inf 或复数（float(complex) 会抛 TypeError）。
    """
    first, last = series.iloc[0], series.iloc[-1]
    n = len(series)
    if (np.isfinite(first) and np.isfinite(last) and first > 0 and last > 0 and n >= 2):
        return float((last / first) ** (1 / (n - 1)) - 1)
    return float("nan")


def _trend_growth(ann: pd.DataFrame, years: int = 5) -> float:
    """趋势修正的收入增长率基础（供 DCF / DDM 共用）。

    近5年 CAGR 可能掩盖近期增速拐点：最近一年增速显著放缓 → 向最近一年下修；
    显著加速 → 适度上修（不追高）。返回修正后的基准增速。
    """
    rev = ann["revenue"].dropna()
    rev = rev[rev > 0]  # 剔除 0/负值异常后计算
    if len(rev) < 2:
        return 0.05
    r5 = rev.tail(years)
    if len(r5) < 2:
        r5 = rev
    hist_g = _safe_cagr(r5)
    if not np.isfinite(hist_g):
        # 首尾异常时用平均同比增速兜底
        pc = rev.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        hist_g = float(pc.tail(3).mean()) if len(pc) else 0.05
    if not np.isfinite(hist_g):
        hist_g = 0.05
    last_g = np.nan
    if len(rev) >= 2 and rev.iloc[-2] > 0:
        last_g = rev.iloc[-1] / rev.iloc[-2] - 1
    if np.isfinite(last_g) and last_g > -0.30:
        if last_g < hist_g * 0.5:
            return float(hist_g * 0.5 + last_g * 0.5)   # 下滑趋势，向最近一年靠拢
        if last_g > hist_g * 1.5:
            return float(hist_g * 0.6 + last_g * 0.4)   # 加速但不追高
    return float(hist_g)


def classic_assumptions(cd: CompanyData, wacc_info: dict) -> DCFAssumptions:
    """传统教科书 DCF 基准假设（无风格预设、无校准、中性参数）。

    作为「无偏锚」与增强模型对比：所有风格增强（成长加速/周期正常化/
    价值低折现等）都应相对传统 DCF 带来可验证的改进；若某层增强模型误差
    反而大于传统 DCF，说明该层风格预设引入了偏差，需反向修正。

    与 auto_assumptions 的差异：
      - 5 年显式期、增长率不打折/不加速/不衰减（历史趋势 CAGR 原值）
      - 永续增长率用市场中性值（A 3.0% / US 2.5%），不随风格浮动
      - 利润率/资本开支比例用近 5 年历史中位数（固定窗口）
      - ERP 用市场基准（A 5.0% / US 4.2%），不按风格调整
      - 不做周期正常化（base_rev=None）
    """
    from ..style_presets import BASE_ERP
    a = DCFAssumptions()
    ann = cd.annual
    if ann is not None and len(ann) >= 2:
        rev = ann["revenue"].dropna()
        if len(rev) >= 2:
            base_g = _trend_growth(ann, 5)
            a.revenue_growth = float(max(base_g if np.isfinite(base_g) else 0.05, 0.005))
        margin = (ann["operating_income"] / ann["revenue"]).dropna()
        if len(margin):
            a.operating_margin = float(np.clip(margin.tail(5).median(), 0.005, 0.85))
        rev_mean = ann["revenue"].replace(0, np.nan).mean()
        if np.isfinite(rev_mean) and rev_mean > 0:
            cp = (ann["capex"] / ann["revenue"]).dropna()
            if len(cp):
                a.capex_pct = float(np.clip(cp.tail(5).median(), 0.0, 0.6))
            dp = (ann["da"] / ann["revenue"]).dropna()
            if len(dp):
                a.da_pct = float(np.clip(dp.tail(5).median(), 0.0, 0.6))
            nw = (ann["change_wc"] / ann["revenue"]).dropna()
            if len(nw):
                a.nwc_pct = float(np.clip(nw.tail(5).median(), -0.10, 0.20))
    a.terminal_growth = 0.030 if cd.market == "A" else 0.025
    a.revenue_growth = max(a.revenue_growth, a.terminal_growth)
    a.tax_rate = float(wacc_info["tax_rate"])
    a.beta = float(wacc_info["beta"])
    a.erp = BASE_ERP.get(cd.market, 0.05)
    a.rf = wacc_info["rf"]
    return a


def auto_assumptions(cd: CompanyData, wacc_info: dict, style: str = "auto") -> DCFAssumptions:
    """按估值风格生成基准假设。style: auto/growth/steady/value/cyclical。"""
    from ..style_presets import resolve_style, style_erp, style_terminal_g, normalize_revenue

    a = DCFAssumptions()
    style, ps = resolve_style(style, cd)
    ann = cd.annual
    mw = ps["margin_window"]  # 利润率统计窗口
    a.forecast_years = int(ps.get("forecast_years", 5))  # 成长股显式期更长

    if ann is not None and len(ann) >= 2:
        rev = ann["revenue"].dropna()
        if len(rev) >= 2:
            base_g = _trend_growth(ann, 5)
            # 按风格折扣后作为预测起点
            a.revenue_growth = float(max(base_g * ps["growth_discount"], 0.005))
            # 显式期路径：加速 / 持平 / 衰减
            a.accel = float(ps["accel"])
            if ps["decline_divisor"]:
                a.growth_decline = float(max((a.revenue_growth - 0.03) / ps["decline_divisor"], 0.0))
            else:
                a.growth_decline = 0.0

        # 营业利润率：按风格窗口取历史中位数，并做合理区间约束
        margin = (ann["operating_income"] / ann["revenue"]).dropna()
        if len(margin):
            m = float(margin.tail(mw).median())
            a.operating_margin = float(np.clip(m, 0.005, 0.85))

        # capex/da/nwc 比例：历史均值（周期股用较长窗口）
        rev_mean = ann["revenue"].replace(0, np.nan).mean()
        if np.isfinite(rev_mean) and rev_mean > 0:
            cp = (ann["capex"] / ann["revenue"]).dropna()
            if len(cp):
                a.capex_pct = float(np.clip(cp.tail(mw).median(), 0.0, 0.6))
            dp = (ann["da"] / ann["revenue"]).dropna()
            if len(dp):
                a.da_pct = float(np.clip(dp.tail(mw).median(), 0.0, 0.6))
            nw = (ann["change_wc"] / ann["revenue"]).dropna()
            if len(nw):
                a.nwc_pct = float(np.clip(nw.tail(mw).median(), -0.10, 0.20))

        # 周期股：正常化收入基准（周期均值）
        if ps["normalize"]:
            base = normalize_revenue(cd, window=8)
            if base is not None and base > 0:
                a.base_rev = float(base)

    a.terminal_growth = style_terminal_g(style, cd.market)
    # 逻辑一致性：显式期增长率不应低于永续增长率
    # （否则出现"显式期低增长 → 永续高增长"的矛盾设定）
    a.revenue_growth = max(a.revenue_growth, a.terminal_growth)
    a.tax_rate = float(wacc_info["tax_rate"])
    a.beta = float(wacc_info["beta"])
    a.erp = style_erp(style, cd.market)
    a.rf = wacc_info["rf"]
    return a


# ---------------- 核心计算 ----------------

def run_dcf(cd: CompanyData, assump: DCFAssumptions | None = None) -> DCFResult:
    res = DCFResult()
    if cd.annual is None or len(cd.annual) == 0:
        res.error = "缺少年度财务数据"
        return res

    # 金融股（银行/高杠杆机构）FCFF 结构失真：利息收入计入 EBIT 且营运资本变动巨大，
    # DCF 会算出荒谬高值（回测中工行/中行隐含收益高达 30-67 倍）。直接禁用 DCF，改用 DDM。
    from ..style_presets import is_financial
    if is_financial(cd):
        res.error = ("金融股（银行/保险等高杠杆机构）FCFF 结构失真，DCF 模型不适用；"
                     "建议以 DDM（股利贴现）结果为准。")
        return res

    a = assump or DCFAssumptions()
    if a.erp is None:
        a.erp = cd.erp
    if a.rf is None:
        a.rf = cd.rf

    wacc_info = calc_wacc(cd, beta=a.beta, debt_rate=a.debt_rate, erp=a.erp, rf=a.rf)
    wacc = wacc_info["wacc"]
    res.wacc = wacc

    # 基准年（周期股用正常化收入，否则最新年）
    last_rev = a.base_rev if (a.base_rev is not None and np.isfinite(a.base_rev) and a.base_rev > 0) else cd.last_value("revenue")
    if not np.isfinite(last_rev) or last_rev <= 0:
        res.error = "营业收入数据缺失"
        return res

    # 增长率序列（支持加速/持平/衰减）
    n = a.forecast_years
    growths = [a.revenue_growth - a.growth_decline * i + a.accel * i for i in range(n)]
    growths = [max(g, a.terminal_growth - 0.01) for g in growths]

    rows = []
    rev = last_rev
    fcff_sum = 0.0
    for i in range(n):
        g = growths[i]
        rev *= (1 + g)
        ebit = rev * a.operating_margin
        nopat = ebit * (1 - a.tax_rate)
        da = rev * a.da_pct
        capex = rev * a.capex_pct
        nwc = rev * a.nwc_pct
        fcff = nopat + da - capex - nwc
        year = cd.latest_year() + i + 1
        rows.append({
            "年份": year,
            "收入": rev,
            "EBIT": ebit,
            "NOPAT": nopat,
            "折旧摊销": da,
            "资本开支": capex,
            "营运资本变动": nwc,
            "FCFF": fcff,
            "增长率": g,
        })
    fcff_last = rows[-1]["FCFF"]

    # 终值（Gordon）
    if wacc <= a.terminal_growth:
        # 低 WACC 公司（公用事业等低 beta）：永续增长率高于 WACC 会令 Gordon 分母为负。
        # 自动下调永续增长率到安全边界，保持模型成立并给出提示（而非直接报错）。
        new_g = max(wacc - 0.01, 0.005)
        res.note = (f"该股 WACC({wacc:.2%}) 较低，原永续增长率 {a.terminal_growth:.2%} 无法成立，"
                    f"已自动下调至 {new_g:.2%}。低 WACC 下终值占比高，估值偏保守，建议结合 DDM 交叉验证。")
        a.terminal_growth = new_g
    tv = fcff_last * (1 + a.terminal_growth) / (wacc - a.terminal_growth)
    tv_pv = tv / (1 + wacc) ** n

    # 低折现率差距提示：永续增长率接近 WACC 时终值对估值影响极大
    if wacc - a.terminal_growth < 0.02:
        res.note = ("当前折现率与永续增长率差距较小，终值占比很高，估值对这两个假设极其敏感。"
                    "低利率环境下 DCF 可能系统性高估，建议结合可比公司/DDM 交叉验证。")

    # EV
    ev = 0.0
    for i, row in enumerate(rows):
        pv = row["FCFF"] / (1 + wacc) ** (i + 1)
        ev += pv
        rows[i]["FCFF现值"] = pv
    ev += tv_pv

    # 股权价值 = EV - 净债务
    debt = cd.last_value("total_debt")
    cash = cd.last_value("cash")
    if not np.isfinite(debt) or debt < 0:
        debt = 0.0
    if not np.isfinite(cash) or cash < 0:
        cash = 0.0
    net_debt = debt - cash
    equity_value = ev - net_debt

    # 每股
    shares = cd.shares
    if np.isnan(shares) or shares <= 0:
        shares = cd.last_value("shares")
    if np.isnan(shares) or shares <= 0:
        res.error = "总股本缺失，无法计算每股价值"
        return res
    per_share = equity_value / shares

    # 负值防护：FCFF/股权价值为负时模型不适用（常见于重资产低利润率公司按周期口径）
    if not np.isfinite(per_share) or per_share <= 0:
        res.error = ("该估值风格/假设下 FCFF 为负，DCF 模型不适用。"
                     "建议改用「价值股」或「稳健股」风格，或检查利润率/资本开支假设。")
        return res

    price = cd.latest_price()
    upside = (per_share / price - 1) if np.isfinite(price) and price > 0 else np.nan
    if upside >= 0.2:
        conclusion = "低估"
        detail = f"内在价值高于当前价 {upside:.1%}，安全边际较充足，具备吸引力。"
    elif upside >= 0:
        conclusion = "合理偏低"
        detail = f"内在价值略高于当前价 {upside:.1%}，估值处于合理区间下沿。"
    elif upside >= -0.2:
        conclusion = "合理偏高"
        detail = f"内在价值低于当前价 {-upside:.1%}，估值处于合理区间上沿。"
    else:
        conclusion = "高估"
        detail = f"内在价值低于当前价 {-upside:.1%}，当前价格透支了较多预期。"

    res.per_share_value = per_share
    res.upside = upside
    res.ev = ev
    res.equity_value = equity_value
    res.terminal_value = tv
    res.terminal_pv = tv_pv
    res.terminal_ratio = tv_pv / ev if ev else np.nan
    res.forecast = pd.DataFrame(rows)
    res.assumptions = {
        "forecast_years": n,
        "revenue_growth": a.revenue_growth,
        "growth_decline": a.growth_decline,
        "accel": a.accel,
        "terminal_growth": a.terminal_growth,
        "operating_margin": a.operating_margin,
        "tax_rate": a.tax_rate,
        "capex_pct": a.capex_pct,
        "da_pct": a.da_pct,
        "nwc_pct": a.nwc_pct,
        "beta": a.beta,
        "debt_rate": a.debt_rate,
        "wacc": wacc,
        "base_rev": a.base_rev,
    }
    res.conclusion = conclusion
    res.detail = detail
    return res
