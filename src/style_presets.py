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
        # 反向修正(2026-08-26)：成长豁免后高成长稳定盈利公司（光模块/电池龙头）
        # 大量迁入 growth 层，其估值已较充分定价，增强模型 8年期/4%永续/加速易
        # 高估（reverse_check 更优 54%→46%）。历史增速打 5% 折扣 + 永续率降档
        # (A 3.5%/US 2.5%) 温和抑制乐观，仍显著高于 steady(2.8%/2.3%) 体现成长性。
        "growth_discount": 0.95,
        "decline_divisor": None,
        "accel": 0.01,
        "terminal_g": {"A": 0.035, "US": 0.025},
        "erp_adj": 0.002,
        "margin_window": 3,
        "normalize": False,
        "forecast_years": 8,   # 成长股显式预测期更长：5 年高增长后仍有增长惯性，
                               # 提前进入永续(4%)会低估成长性（回测系统性低估 -41%）
    },
    "steady": {
        "label": "稳健股",
        "growth_discount": 0.90,
        "decline_divisor": None,
        "accel": 0.0,
        # 反向修正(2026-08-26)：与教科书 DCF(永续 A 3.0%/US 2.5%)双跑对比，
        # 稳态层增强仅 43% 样本更优——低 WACC 高分红公司（银行/公用/资源）在
        # 更高永续增长率下终值膨胀、系统性高估。故永续增长率调到比传统更保守
        # (A 2.8%/US 2.3%)，压低低 WACC 终值占比。
        "terminal_g": {"A": 0.028, "US": 0.023},
        "erp_adj": 0.0,
        "margin_window": 3,
        "normalize": False,
    },
    "value": {
        "label": "价值股",
        "growth_discount": 0.75,
        "decline_divisor": 5,
        "accel": 0.0,
        # 反向修正(2026-08-26)：value 层校准系数长期撞 0.5 下限（均值误差 +70%，
        # 系统性高估压不住），根因是低增长稳定公司 FCFF 中自由现金流占比低、
        # 估值高度依赖永续期（终值占比常 >85%）。永续增长率再降档
        # (A 2.0%/US 1.5%，比传统 DCF 的 A3.0%/US2.5% 更保守)以压缩终值贡献，
        # 让 DCF 更接近"现金流折现"而非"永续假设折现"。
        "terminal_g": {"A": 0.020, "US": 0.015},
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
    """近 N 年收入 CAGR（稳健版：首尾须为正，否则返回 NaN）。"""
    ann = cd.annual
    if ann is None or len(ann) == 0:
        return np.nan
    rev = ann["revenue"].dropna()
    rev = rev[rev > 0]
    if len(rev) < 2:
        return np.nan
    r = rev.tail(years)
    if len(r) < 2:
        r = rev
    first, last = r.iloc[0], r.iloc[-1]
    if np.isfinite(first) and np.isfinite(last) and first > 0 and last > 0:
        return float((last / first) ** (1 / (len(r) - 1)) - 1)
    return np.nan


def _detect_cyclical(cd: CompanyData, ann) -> bool:
    """周期股检测：盈利/营收大幅波动。

    先于成长判断执行——周期股即使某几年高增长（如周期底部反弹），
    也应归入 cyclical 用正常化盈利估值，避免混入 growth 层造成 DCF 隐含收益荒谬放大。
    信号（近 8 年窗口，任一命中即判周期）：
      ① 营业利润率变异系数 > 0.5
      ② 营收变异系数 > 0.6
      ③ 净利率变异系数 > 0.7
      ④ 净利峰谷比（近8年全为正时最大/最小 > 6 倍）

    成长豁免（2026-08-26）：高速成长且盈利稳定的公司（如光模块/动力电池龙头）
    营收因基数扩张变异系数天然高（0.6~0.9），但这是"规模扩张"而非"景气周期"。
    若近5年营收 CAGR>15% 且净利率变异系数 <0.45（盈利能力稳定），豁免周期判定，
    让其进入 growth 层——否则正常化收入会把高成长公司压回低基数、系统性低估。
    验证：天孚通信(CAGR50%/niCV0.13)→growth、宁德时代(34%/0.21)→growth、
    胜宏科技(27%/0.40)→growth；中国船舶(26%/0.83，真周期)→cyclical 保留；
    中芯国际(17%/0.68，半导体周期)→cyclical 保留。
    """
    if ann is None or len(ann) < 4:
        return False
    # 成长豁免：高成长 + 盈利稳定 → 营收 CV 高是扩张而非周期
    g = _hist_cagr(cd, 5)
    if np.isfinite(g) and g > 0.15 and "net_income" in ann.columns:
        ni_m = (ann["net_income"] / ann["revenue"].replace(0, np.nan)).dropna().tail(8)
        if len(ni_m) >= 4 and ni_m.mean() > 0:
            if float(ni_m.std() / abs(ni_m.mean())) < 0.45:
                return False
    # ① 营业利润率变异系数（主信号）
    margin = (ann["operating_income"] / ann["revenue"]).dropna().tail(8)
    if len(margin) >= 4 and margin.mean() > 0:
        if float(margin.std() / abs(margin.mean())) > 0.5:
            return True
    # ② 营收变异系数
    rev = ann["revenue"].dropna().tail(8)
    if len(rev) >= 4 and rev.mean() > 0:
        if float(rev.std() / rev.mean()) > 0.6:
            return True
    # ③ 净利率变异系数
    if "net_income" in ann.columns:
        ni_margin = (ann["net_income"] / ann["revenue"]).dropna().tail(8)
        if len(ni_margin) >= 4 and ni_margin.mean() > 0:
            if float(ni_margin.std() / abs(ni_margin.mean())) > 0.7:
                return True
    # ④ 净利峰谷比（近8年全为正时跨度>6倍）
    if "net_income" in ann.columns:
        ni = ann["net_income"].dropna().tail(8)
        if len(ni) >= 4 and (ni > 0).sum() >= 4:
            mx, mn = float(ni.max()), float(ni.min())
            if mn > 0 and mx / mn > 6.0:
                return True
    return False


def is_financial(cd: CompanyData) -> bool:
    """金融股识别：银行/高杠杆金融机构的 FCFF 结构失真，DCF 不适用。

    财务特征（近 3 年均值）：(有息负债+现金) / 收入 > 5 且 收入 / 净资产 < 0.5。
    —— 银行负债表规模远大于收入（高杠杆）、净资产相对收入占比高。
    验证：工行(6.7/0.21)、中行(7.5/0.22)、浦发(9.5/0.23) 命中；
          茅台(0.36/0.72)、宁德(1.03/1.52) 不命中；长江电力(3.0/0.39) 不命中。
    全部为有限值才判定，避免数据缺失误判。

    保险/投资控股补充信号（2026-08-26）：保险（平安/人寿等）负债结构特殊，
    银行型指标(负债+现金)/收入常仅 ~0.9（浮存金不在有息负债口径），且权益字段
    常缺失，银行判定漏网；但其"资金池"特征明显——营运现金流远超净利润（含
    准备金/投资资金沉淀）且账面现金高企。若 营运现金流/净利润 > 2.5 且 现金/收入 > 0.2，
    判为金融机构，DCF 同样失真，改用 DDM。
    验证：平安(3.38/0.67)、中国人寿(3.84/0.25)、招行(2.80/1.81)、工行(3.52/4.41)
          命中；移动(2.09/0.19)、长电(1.96/0.08)、神华(1.52/0.40)、石化(3.32/0.05)
          不命中（cfo/ni 高但现金/收入低，非资金池型）。
    """
    ann = cd.annual
    if ann is None or len(ann) < 2:
        return False
    tail = ann.tail(3)
    rev = float(tail["revenue"].mean())
    eq = float(tail["total_equity"].mean())
    debt = float(tail["total_debt"].mean())
    cash = float(tail["cash"].mean())
    if not all(np.isfinite([rev, eq, debt, cash])):
        # 银行型判定因权益缺失无法进行时，回退到保险信号再判
        eq = np.nan
    if np.isfinite(rev) and rev > 0 and np.isfinite(eq) and eq > 0 \
            and np.isfinite(debt) and np.isfinite(cash):
        if (debt + cash) / rev > 5.0 and rev / eq < 0.5:
            return True
    # 保险/投资控股"资金池"信号
    # 必要条件：折旧摊销缺失（保险无固定资产折旧；重资产制造如中船/中芯 cfo/ni
    # 同样高——来自折旧加回/订单周期，但 da 存在，需排除）
    da = ann["da"].dropna() if "da" in ann.columns else pd.Series(dtype=float)
    da_missing = len(da) == 0
    if da_missing and "cfo" in ann.columns and "net_income" in ann.columns:
        cfo = ann["cfo"].dropna().tail(3)
        ni = ann["net_income"].dropna().tail(3)
        cash_s = ann["cash"].dropna().tail(3)
        rev_s = ann["revenue"].dropna().tail(3)
        if len(cfo) >= 2 and len(ni) >= 2 and len(cash_s) >= 2 and len(rev_s) >= 2:
            if ni.mean() > 0 and rev_s.mean() > 0:
                if cfo.mean() / ni.mean() > 2.5 and cash_s.mean() / rev_s.mean() > 0.2:
                    return True
    return False


def auto_detect_style(cd: CompanyData) -> str:
    """自动识别公司属性（周期性优先，避免周期股/重资产误入成长层）。"""
    ann = cd.annual
    # ① 周期特征优先：盈利/营收大幅波动（周期股即使高增长也按周期处理）
    if _detect_cyclical(cd, ann):
        return "cyclical"

    # ② 成长：历史收入 CAGR > 15%（且盈利稳定，周期信号已在上方排除）
    g = _hist_cagr(cd, 5)
    if np.isfinite(g) and g > 0.15:
        return "growth"

    # ③ 价值：低增长 + 高股息
    if np.isfinite(g) and g < 0.05:
        div = cd.last_value("dividends")
        mcap = cd.market_cap
        if np.isfinite(div) and div > 0 and np.isfinite(mcap) and mcap > 0:
            if div / mcap > 0.02:
                return "value"
    return "steady"


def normalize_revenue(cd: CompanyData, window: int = 8) -> float | None:
    """周期股正常化收入：周期均值收入（避免周期峰谷失真）。

    基准选择（2026-08-26 增长感知）：
      - 周期成长股（近5年营收 CAGR>8%）：用近 3 年均值——其收入处上行通道，
        8 年中位数会把成长中段拖回低基数、系统性低估（周期成长股命中率 38%
        偏弱的主因之一）。
      - 纯周期股（增速平稳/下行）：用近 N 年收入中位数（抗峰谷）。
    仅当最新营收与基准处于同一量级（0.5~2.0 倍）时启用，
    避免把「已成长上台阶」的公司（如最新营收数倍于历史均值）错误拉回低基数。
    """
    ann = cd.annual
    if ann is None or len(ann) == 0:
        return None
    rev = ann["revenue"].dropna().tail(window)
    if len(rev) >= 4:
        latest = float(rev.iloc[-1])
        g5 = _hist_cagr(cd, 5)
        if np.isfinite(g5) and g5 > 0.08:
            base = float(rev.tail(3).mean())   # 周期成长：近3年均值
        else:
            base = float(rev.median())         # 纯周期：中位数抗峰谷
        if base > 0 and latest > 0:
            ratio = latest / base
            if 0.5 <= ratio <= 2.0:
                return base
            return None  # 已上台阶，不适用均值回归
    return None


def resolve_style(style: str, cd: CompanyData) -> tuple[str, dict]:
    """把 'auto' 解析为具体风格，返回 (实际风格, 参数表)。

    修复(2026-08-26)：此前 `s = style if style in STYLE_PRESETS else "steady"`
    因 STYLE_PRESETS 无 "auto" key，导致 auto 永远落回 steady，面板"自动识别"
    从未真正触发。校准/回测不受影响（它们直接调 auto_detect_style）。
    """
    if style == "auto":
        s = auto_detect_style(cd)
        return s, STYLE_PRESETS[s]
    s = style if style in STYLE_PRESETS else "steady"
    return s, STYLE_PRESETS.get(s, STYLE_PRESETS["steady"])


# ---------------- 行业 WACC 参考（借鉴 Dexter/sector-wacc.md） ----------------
# 典型区间作为起始参考 + 合理性校验锚，非硬约束。数据源不提供真实行业分类，
# 故用财务特征推断近似行业类别。
WACC_SECTOR_RANGES: dict[str, tuple[float, float]] = {
    "Communication Services": (0.08, 0.10),
    "Consumer Discretionary": (0.08, 0.10),
    "Consumer Staples": (0.07, 0.08),
    "Energy": (0.09, 0.11),
    "Financials": (0.08, 0.10),
    "Health Care": (0.08, 0.10),
    "Industrials": (0.08, 0.09),
    "Information Technology": (0.08, 0.12),
    "Materials": (0.08, 0.10),
    "Real Estate": (0.07, 0.09),
    "Utilities": (0.06, 0.07),
}

SECTOR_STYLE_MAP = {
    "growth": "Information Technology",
    "steady": "Consumer Staples",
    "value": "Consumer Staples",
    "cyclical": "Materials",
}


def infer_sector(cd: CompanyData) -> str:
    """财务特征推断近似行业（用于 WACC 合理性校验提示）。"""
    style = auto_detect_style(cd)
    if is_financial(cd):
        return "Financials"
    ann = cd.annual
    beta = cd.beta if np.isfinite(cd.beta) else np.nan
    capex_ratio = np.nan
    if ann is not None and len(ann) and "capex" in ann.columns:
        r = (ann["capex"] / ann["revenue"].replace(0, np.nan)).dropna()
        if len(r):
            capex_ratio = float(r.tail(5).median())
    # 低 beta + 高资本开支 + 稳定 → 公用事业/基础设施
    if np.isfinite(beta) and beta < 0.5 and np.isfinite(capex_ratio) and capex_ratio > 0.10:
        return "Utilities"
    return SECTOR_STYLE_MAP.get(style, "Industrials")


def sector_wacc_check(cd: CompanyData, wacc: float) -> str:
    """行业 WACC 合理性校验：计算值与推断行业的典型区间对比，返回提示文本（空=合理）。

    sector-wacc 表为美股口径（无风险利率约 4.3%）；A 股无风险利率约 2.1%，
    整体折现率中枢低约 1.5%，故 A 股统一下调区间下沿，避免提示泛滥。
    """
    sector = infer_sector(cd)
    lo, hi = WACC_SECTOR_RANGES.get(sector, (0.08, 0.10))
    if cd.market == "A":
        lo, hi = lo - 0.015, hi - 0.015
    if wacc < lo - 0.005:
        return f"WACC({wacc:.2%}) 低于 {sector} 行业典型区间({lo:.0%}~{hi:.0%})下沿，估值对折现率敏感，建议复核。"
    if wacc > hi + 0.02:
        return f"WACC({wacc:.2%}) 高于 {sector} 行业典型区间({lo:.0%}~{hi:.0%})上沿，估值偏保守。"
    return ""


# ---------------- 综合研判：风格 × 估值方法 权重矩阵 ----------------
# 核心思想：不同公司类型下，各估值方法的可信度不同，综合目标价应是
# "方法擅长度"加权的均值，而非简单平均/中位数。
#   growth(成长): DCF/PEG 主导（高增长价值在现金流与增速），DDM 低（成长股分红少）
#   steady(稳健): DCF/DDM/可比 均衡
#   value(价值):  DDM/可比 主导（低增长高股息，现金流法与市场倍数更可靠），PEG 不适用
#   cyclical(周期): PB 可比/历史分位/EVA 主导（PE 对周期盈利失真），PEG 不适用
# 金融股（is_financial）：DCF/EVA 因 FCFF/IC 结构失真禁用，DDM/历史分位/PB 主导
METHOD_WEIGHTS: dict[str, dict[str, float]] = {
    "growth":   {"dcf": 0.30, "ddm": 0.05, "comps": 0.20, "peg": 0.25, "percentile": 0.10, "eva": 0.10},
    "steady":   {"dcf": 0.30, "ddm": 0.25, "comps": 0.20, "peg": 0.05, "percentile": 0.10, "eva": 0.10},
    "value":    {"dcf": 0.20, "ddm": 0.30, "comps": 0.25, "peg": 0.00, "percentile": 0.15, "eva": 0.10},
    "cyclical": {"dcf": 0.25, "ddm": 0.10, "comps": 0.20, "peg": 0.00, "percentile": 0.25, "eva": 0.20},
}

# 金融股单独权重（DCF/EVA 结构失真禁用）
FINANCIAL_WEIGHTS: dict[str, float] = {
    "dcf": 0.0, "ddm": 0.45, "comps": 0.25, "peg": 0.0, "percentile": 0.30, "eva": 0.0,
}

# 方法中文名（用于权重展示）
METHOD_WEIGHT_LABELS: dict[str, str] = {
    "dcf": "DCF现金流折现",
    "ddm": "DDM股利贴现",
    "comps": "可比公司",
    "peg": "PEG增速匹配",
    "percentile": "历史估值分位",
    "eva": "EVA经济增加值",
}


def method_weights(style: str, cd: CompanyData) -> dict[str, float]:
    """返回综合研判用的方法权重（已按金融股/实际风格解析）。"""
    if is_financial(cd):
        return dict(FINANCIAL_WEIGHTS)
    if style == "auto":
        s = auto_detect_style(cd)
    else:
        s = style if style in METHOD_WEIGHTS else "steady"
    return dict(METHOD_WEIGHTS.get(s, METHOD_WEIGHTS["steady"]))
