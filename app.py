# -*- coding: utf-8 -*-
"""
DCF 智能估值面板 —— Streamlit 主应用

运行: streamlit run app.py
功能: 输入 A股/美股 代码 → 自动抓取财务数据 → DCF/DDM/反向DCF/可比公司 多方法估值
      → 方法论讲解 → AI 综合研判（可选接入大模型）
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_fetcher import fetch_company, detect_market
from src.models.dcf import DCFAssumptions, auto_assumptions, run_dcf
from src.models.ddm import run_ddm
from src.models.reverse_dcf import run_reverse_dcf
from src.models.comps import run_comps, auto_peers
from src.methodology import METHODOLOGY
from src.ai_advisor import PROVIDERS, build_prompt, call_ai, rule_based_summary

st.set_page_config(
    page_title="DCF 智能估值面板",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============ 工具函数（放在顶部，供各 Tab 使用） ============

class DCFAssumptionsSafe:
    """用于敏感性分析的轻量假设容器（复用 DCFAssumptions 字段）。"""
    def __init__(self, a):
        for f in ["forecast_years", "revenue_growth", "growth_decline", "accel", "terminal_growth",
                  "operating_margin", "tax_rate", "capex_pct", "da_pct", "nwc_pct",
                  "beta", "debt_rate", "erp", "rf"]:
            setattr(self, f, getattr(a, f))


def _dcf_with_wacc(cd, assump, wacc: float) -> float:
    """用指定 WACC 重算 DCF 每股价值（敏感性分析用）。

    与 run_dcf 保持同一折现约定（年中：现金流折现期 i+0.5，终值折现期 n-0.5）。
    """
    a = DCFAssumptions()
    for f in ["forecast_years", "revenue_growth", "growth_decline", "accel", "terminal_growth",
              "operating_margin", "tax_rate", "capex_pct", "da_pct", "nwc_pct",
              "beta", "debt_rate", "erp", "rf"]:
        setattr(a, f, getattr(assump, f))
    r = run_dcf(cd, a)
    if r.error or r.forecast is None or len(r.forecast) == 0:
        return np.nan
    rows = r.forecast.copy()
    n = len(rows)
    ev = sum(rows["FCFF"].iloc[i] / (1 + wacc) ** (i + 0.5) for i in range(n))
    if wacc <= a.terminal_growth:
        return np.nan
    tv = rows["FCFF"].iloc[-1] * (1 + a.terminal_growth) / (wacc - a.terminal_growth)
    ev += tv / (1 + wacc) ** (n - 0.5)
    debt = cd.last_value("total_debt")
    cash = cd.last_value("cash")
    if not math.isfinite(debt) or debt < 0:
        debt = 0
    if not math.isfinite(cash) or cash < 0:
        cash = 0
    shares = cd.shares
    if not math.isfinite(shares) or shares <= 0:
        shares = cd.last_value("shares")
    if not math.isfinite(shares) or shares <= 0:
        return np.nan
    return (ev - (debt - cash)) / shares


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_cached(symbol: str, market: str):
    return fetch_company(symbol, market)


def fmt(v: float, digits: int = 2) -> str:
    if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
        return "N/A"
    return f"{v:,.{digits}f}"


def fmt_pct(v: float) -> str:
    if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
        return "N/A"
    return f"{v:+.1%}"


def fmt_big(v: float) -> str:
    if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
        return "N/A"
    a = abs(v)
    if a >= 1e12:
        return f"{v/1e12:.2f}万亿"
    if a >= 1e8:
        return f"{v/1e8:.2f}亿"
    if a >= 1e4:
        return f"{v/1e4:.2f}万"
    return f"{v:,.0f}"


# ============ 侧边栏 ============
with st.sidebar:
    st.title("📊 DCF 智能估值")
    st.caption("A股 / 美股 · 自动抓取 · 多方法交叉估值")
    st.divider()

    symbol_input = st.text_input("股票代码 / 名称", value="600519", key="symbol_input")

    auto_market = detect_market(symbol_input)
    market = st.radio("市场（一般自动识别，可手动修正）", ["自动", "A股", "美股"],
                      index=0, horizontal=True)
    if market == "A股":
        mkt = "A"
    elif market == "美股":
        mkt = "US"
    else:
        mkt = auto_market

    run_btn = st.button("🚀 开始估值", type="primary", use_container_width=True)

    st.divider()
    st.caption("💡 输入：A股 6位数字或中文名称（如 600519 / 贵州茅台），美股代码或英文名称（如 AAPL / Apple）。")

    # 估值风格：按公司属性套用不同参数口径
    style_map = {"auto": "自动识别", "growth": "成长股", "steady": "稳健股",
                 "value": "价值股", "cyclical": "周期股"}
    st.markdown("**估值风格**")
    style = st.radio(
        "估值风格", list(style_map.keys()),
        format_func=lambda k: style_map[k], index=0, label_visibility="collapsed",
        help="按公司属性套用对应估值参数。自动：高增长→成长股、高营收波动→周期股、"
             "低增长高股息→价值股、其余→稳健股；也可手动指定。",
    )
    # 估值风格切换 → 清掉缓存的 DCF 假设，使其按新风格重建
    if st.session_state.get("_style_used") != style:
        st.session_state["_style_used"] = style
        if "dcf_assump" in st.session_state:
            st.session_state.pop("dcf_assump", None)

# ============ 主区域 ============
if not run_btn and "cd" not in st.session_state:
    st.markdown("## 👋 欢迎使用 DCF 智能估值面板")
    st.markdown(
        "在左侧输入 **A股**（6位代码或中文名称，如 `600519` / `贵州茅台`）或 **美股**（代码或英文名称，如 `AAPL` / `Apple`）"
        "，点击 **开始估值**。\n\n"
        "系统会自动：\n"
        "- 抓取近 5-10 年财务报表、当前行情、Beta、无风险利率\n"
        "- 运行 **DCF / DDM / 反向DCF / 可比公司** 四种估值方法\n"
        "- 给出每方法的内在价值与安全边际，并做 **综合研判**\n\n"
        "> 📖 建议先看「方法论」页，理解每种方法的公式与适用场景，再解读结果。"
    )
    st.stop()

if run_btn:
    for k in ["cd", "results", "dcf_assump", "wacc_info"]:
        st.session_state.pop(k, None)
    symbol = symbol_input.strip().upper()
    if not symbol:
        st.warning("请输入股票代码")
        st.stop()
    st.session_state["style"] = style
    with st.spinner(f"正在抓取 {symbol} 数据并计算估值（A股约需15-30秒，美股视网络情况）..."):
        try:
            cd = _fetch_cached(symbol, mkt)
            if cd.annual is None or len(cd.annual) == 0:
                st.error(f"未能获取 {symbol} 的财务数据，请检查代码/名称是否正确（A股6位数字或名称/美股字母代码或英文名称）。")
                st.stop()
            st.session_state["cd"] = cd
            st.session_state["symbol"] = symbol
        except Exception as e:
            st.error(f"数据抓取失败：{e}\n请检查代码或网络后重试。")
            st.stop()

if "cd" in st.session_state:
    cd = st.session_state["cd"]

    # ---------- 头部信息 ----------
    price = cd.latest_price()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(f"{cd.name}", f"{fmt(price)} {cd.currency}", f"{cd.symbol}")
    c2.metric("总市值", fmt_big(cd.market_cap), "本币")
    c3.metric("PE (TTM)", fmt(cd.pe_ttm))
    c4.metric("PB", fmt(cd.pb))
    c5.metric("Beta", fmt(cd.beta))
    if cd.industry:
        st.caption(f"行业：{cd.industry}    无风险利率：{cd.rf:.2%}    股权风险溢价：{cd.erp:.2%}")
    # 估值风格口径显示
    from src.style_presets import resolve_style, STYLE_LABELS
    _actual, _ = resolve_style(style, cd)
    st.caption(f"估值口径：**{STYLE_LABELS[style]}**"
               + (f"（自动识别为 **{STYLE_LABELS[_actual]}**）" if style == "auto" else ""))
    if cd.notes:
        with st.expander("数据抓取备注"):
            for n in cd.notes:
                st.write(f"- {n}")

    st.divider()

    # ============ Tab 结构 ============
    tab_method, tab_dcf, tab_ddm, tab_rev, tab_comps, tab_summary = st.tabs(
        ["📖 方法论", "💰 DCF 现金流折现", "🏦 DDM 股利贴现",
         "🔄 反向 DCF", "🏷️ 可比公司", "📈 综合研判"]
    )

    # ---------- Tab1 方法论 ----------
    with tab_method:
        st.markdown("## 估值方法论速览")
        st.caption("每种方法都给出公式、原理、用法、适用场景与局限，帮助你正确解读结果。")
        st.divider()
        for key, m in METHODOLOGY.items():
            with st.expander(f"### {m['name']}", expanded=(key == "dcf")):
                st.markdown("**公式**")
                st.code(m["公式"], language="text")
                st.markdown(f"**原理**：{m['原理']}")
                st.markdown(f"**用法**\n\n{m['用法']}")
                st.markdown(f"**✅ 适用**：{m['适用']}")
                st.markdown(f"**⚠️ 局限**：{m['局限']}")
                st.divider()

    # ============ 各估值 Tab ============
    if "wacc_info" not in st.session_state:
        from src.wacc import calc_wacc
        st.session_state["wacc_info"] = calc_wacc(cd)
    wacc_info = st.session_state["wacc_info"]

    results = st.session_state.get("results", {})

    # ---------- Tab2 DCF ----------
    with tab_dcf:
        st.markdown("## DCF 现金流折现（FCFF 两阶段）")
        st.caption("自动生成基准假设，可在下方调整。所有假设改变都会实时重算。")

        if "dcf_assump" not in st.session_state:
            try:
                st.session_state["dcf_assump"] = auto_assumptions(cd, wacc_info, style=style)
            except Exception as e:
                # 兜底：自动风格重试；仍失败则用最小保守假设，避免整个应用崩溃
                try:
                    st.session_state["dcf_assump"] = auto_assumptions(cd, wacc_info, style="auto")
                    st.warning(f"按「{style}」口径生成假设失败（{type(e).__name__}），已自动回退到「自动识别」口径。"
                               f"如需排查可反馈以下信息：{e}")
                except Exception as e2:
                    st.error(f"估值假设生成失败：{type(e2).__name__}: {e2}。请更换股票或风格后重试。")
                    st.stop()
        a = st.session_state["dcf_assump"]

        with st.expander("⚙️ 预测假设（可调整）", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                a.forecast_years = st.slider("显式预测期(年)", 3, 10, int(a.forecast_years))
                a.revenue_growth = st.number_input(
                    "显式期首年收入增长率", 0.0, 1.0,
                    float(a.revenue_growth), step=0.005, format="%.3f")
                a.growth_decline = st.number_input(
                    "增长率逐年衰减(百分点)", 0.0, 0.10,
                    float(a.growth_decline), step=0.005, format="%.3f")
                a.accel = st.number_input(
                    "增长率逐年加速(百分点)", 0.0, 0.05,
                    float(a.accel), step=0.005, format="%.3f",
                    help=">0 时增长率逐年递增，适用于成长期公司（成长股默认启用）。")
            with col2:
                a.terminal_growth = st.number_input(
                    "永续增长率 g", 0.0, 0.06,
                    float(a.terminal_growth), step=0.005, format="%.3f")
                a.operating_margin = st.number_input(
                    "目标营业利润率", 0.0, 0.90,
                    float(a.operating_margin), step=0.01, format="%.3f")
                a.tax_rate = st.number_input(
                    "税率", 0.0, 0.50,
                    float(a.tax_rate), step=0.01, format="%.3f")
            with col3:
                a.capex_pct = st.number_input(
                    "资本开支/收入", 0.0, 0.60,
                    float(a.capex_pct), step=0.005, format="%.3f")
                a.da_pct = st.number_input(
                    "折旧摊销/收入", 0.0, 0.60,
                    float(a.da_pct), step=0.005, format="%.3f")
                a.nwc_pct = st.number_input(
                    "营运资本变动/收入(可负)", -0.10, 0.30,
                    float(a.nwc_pct), step=0.005, format="%.3f")
            st.divider()
            col4, col5, col6 = st.columns(3)
            with col4:
                a.beta = st.number_input("Beta", 0.2, 3.0, float(a.beta), step=0.05, format="%.2f")
            with col5:
                a.debt_rate = st.number_input("债务成本 Rd", 0.0, 0.15, float(a.debt_rate), step=0.005, format="%.3f")
            with col6:
                # 用风格调整后的 WACC 显示
                from src.wacc import calc_wacc as _cw
                style_wacc = _cw(cd, erp=a.erp, beta=a.beta, debt_rate=a.debt_rate)
                st.markdown(f"**WACC ≈ {style_wacc['wacc']:.2%}**（Re={style_wacc['re']:.2%}, Rd={style_wacc['rd']:.2%}, "
                            f"E={style_wacc['e_weight']:.0%}, D={style_wacc['d_weight']:.0%})")
                if wacc_info.get("note"):
                    st.caption(wacc_info["note"])
                # 行业 WACC 合理性校验（借鉴 Dexter 行业 WACC 参考表）
                from src.style_presets import sector_wacc_check as _swc
                _swc_txt = _swc(cd, style_wacc["wacc"])
                if _swc_txt:
                    st.caption(_swc_txt)

        dcf_res = run_dcf(cd, a)
        st.session_state["results"] = {**results, "dcf": dcf_res}

        if dcf_res.error:
            st.error(dcf_res.error)
        else:
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("每股内在价值", f"{fmt(dcf_res.per_share_value)} {cd.currency}",
                      f"{fmt_pct(dcf_res.upside)} vs 现价")
            r2.metric("企业价值 EV", fmt_big(dcf_res.ev))
            r3.metric("股权价值", fmt_big(dcf_res.equity_value))
            r4.metric("结论", dcf_res.conclusion)

            st.markdown(f"**解读**：{dcf_res.detail}  终值占企业价值 **{dcf_res.terminal_ratio:.0%}**"
                        f"，永续假设对结果影响大。")
            if dcf_res.note:
                st.warning(dcf_res.note)

            colA, colB = st.columns(2)
            with colA:
                st.markdown("#### 预测期 FCFF")
                fdf = dcf_res.forecast.copy()
                disp = fdf.copy()
                for col in disp.columns:
                    if col != "年份":
                        disp[col] = disp[col].map(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)
                st.dataframe(disp, width="stretch", hide_index=True)
                # 维护 vs 增长资本开支说明（重资产公司 capex 一刀切会低估其真实现金流）
                if "维护资本开支" in fdf.columns and "增长资本开支" in fdf.columns:
                    _g_capex = float(fdf["增长资本开支"].iloc[-1])
                    _m_capex = float(fdf["维护资本开支"].iloc[-1])
                    _capex_tot = _g_capex + _m_capex
                    if _capex_tot > 0:
                        st.caption(
                            f"资本开支拆分（末年为参考）：维护性（≈折旧）**{_m_capex:,.0f}** / "
                            f"增长性 **{_g_capex:,.0f}**（占资本开支 {_g_capex/_capex_tot:.0%}）。"
                            "增长性资本开支代表扩张投资，若公司处于资本开支高峰期（如 5G/新产能建设），"
                            "FCFF 会被暂时压低，DCF 偏保守；资本开支回落后现金流与估值会上修。"
                            "可下调「资本开支/收入」假设观察影响。")

                fig = go.Figure()
                fig.add_bar(x=fdf["年份"], y=fdf["FCFF"], name="FCFF",
                            marker_color="steelblue")
                fig.add_scatter(x=fdf["年份"], y=fdf["收入"] / 10, name="收入(右轴/10)",
                                mode="lines+markers", yaxis="y2", line=dict(color="orange"))
                fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                                  yaxis2=dict(overlaying="y", side="right"))
                st.plotly_chart(fig, width="stretch")
            with colB:
                st.markdown("#### 价值瀑布")
                debt = cd.last_value("total_debt")
                cash = cd.last_value("cash")
                if not math.isfinite(debt) or debt < 0:
                    debt = 0
                if not math.isfinite(cash) or cash < 0:
                    cash = 0
                wf = go.Figure(go.Waterfall(
                    orientation="v",
                    measure=["absolute", "relative", "relative", "total"],
                    x=["预测期 FCFF 现值", "终值现值", "净债务(减)", "股权价值"],
                    y=[dcf_res.ev - dcf_res.terminal_pv, dcf_res.terminal_pv, -(debt - cash), dcf_res.equity_value],
                    connector=dict(line=dict(color="lightgray")),
                ))
                wf.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(wf, width="stretch")

            # ===== 三情景估值区间（借鉴 dcf-model 工作流 Scenario Blocks） =====
            from src.models.dcf import scenario_range
            st.markdown("#### 三情景估值区间（悲观 / 基准 / 乐观）")
            scen = scenario_range(cd, a, wacc=dcf_res.wacc)
            _s1, _s2, _s3 = st.columns(3)
            scen_vals = []
            for k, (_sb, _lb) in {"bear": (_s1, "悲观 (Bear)"),
                                  "base": (_s2, "基准 (Base)"),
                                  "bull": (_s3, "乐观 (Bull)")}.items():
                v = scen[k]["value"]
                if np.isfinite(v):
                    _sb.metric(_lb, f"{fmt(v)} {cd.currency}", f"{fmt_pct(scen[k]['upside'])} vs 现价")
                    scen_vals.append(float(v))
                else:
                    _sb.metric(_lb, "N/A", f"({scen[k]['error'] or '不适用'})")
            if len(scen_vals) == 3:
                st.caption(
                    f"悲观~乐观区间：**{min(scen_vals):,.2f} ~ {max(scen_vals):,.2f} {cd.currency}**"
                    f"（跨度 {(max(scen_vals) / min(scen_vals) - 1):.0%}，反映对增长/利润率/折现率假设的敏感度）")
                # 概率加权期望合理估值（默认 25/50/25，可自定义权重；归一化）
                st.caption("**期望合理估值权重**（默认 25/50/25，可调）：")
                wc1, wc2, wc3 = st.columns(3)
                w_b0 = wc1.number_input("悲观权重", 0.0, 1.0, 0.25, 0.05)
                w_m0 = wc2.number_input("基准权重", 0.0, 1.0, 0.50, 0.05)
                w_g0 = wc3.number_input("乐观权重", 0.0, 1.0, 0.25, 0.05)
                _wsum = w_b0 + w_m0 + w_g0
                if _wsum <= 0:
                    _wsum, w_b0, w_m0, w_g0 = 1.0, 0.25, 0.50, 0.25
                w_b, w_m, w_g = w_b0 / _wsum, w_m0 / _wsum, w_g0 / _wsum
                _exp = w_b * scen["bear"]["value"] + w_m * scen["base"]["value"] + w_g * scen["bull"]["value"]
                _exp_up = _exp / price - 1 if price and price > 0 else np.nan
                st.metric(
                    f"期望合理估值（概率加权 {w_b:.0%}/{w_m:.0%}/{w_g:.0%}）",
                    f"{fmt(_exp)} {cd.currency}",
                    f"{fmt_pct(_exp_up)} vs 现价",
                    help=f"= 悲观×{w_b:.0%} + 基准×{w_m:.0%} + 乐观×{w_g:.0%}（已归一化）。"
                         "相比单点基准值，综合了三情景的不确定性，更适合作为估值中枢参考。"
                         "默认 25/50/25 为基于历史校准误差分布的简化近似，可按个人判断调整。"
                         "若明显偏离基准，说明三情景不对称（多数源自增长/利润率假设分布）。")
            st.caption("悲观=增速×0.6/利润率-3pct/WACC+0.5pct/永续率-0.5pct；乐观=增速×1.4/利润率+3pct/"
                       "WACC-0.5pct/永续率+0.5pct；资本开支与增长加速项同步微调。")

            # ===== Excel 工作簿导出（带公式可编辑，借鉴 dcf-model） =====
            try:
                from src.models.excel_export import export_dcf_excel
                xlsx_bytes = export_dcf_excel(cd, a, dcf_res, wacc=dcf_res.wacc)
                st.download_button(
                    "⬇️ 下载可编辑 DCF Excel 工作簿（含敏感性表）",
                    data=xlsx_bytes,
                    file_name=f"{cd.symbol}_DCF_Model_{pd.Timestamp.now():%Y%m%d}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
                st.caption("工作簿内所有估值均为活公式：修改黄色输入单元格（增长率/利润率/WACC 等）整表自动重算；"
                           "敏感性表中心格 = 基准每股价值，可直接改 WACC/永续率对比。")
            except Exception as e:
                st.caption(f"Excel 导出暂不可用：{e}")

            # 敏感性热力图
            st.markdown("#### 敏感性分析：每股价值 vs (WACC × 永续增长率)")
            with st.spinner("计算敏感性..."):
                base_wacc = dcf_res.wacc
                g_base = a.terminal_growth
                waccs = [base_wacc + (i - 2) * 0.005 for i in range(5)]
                gs = [max(g_base + (i - 2) * 0.005, 0.0) for i in range(5)]
                mat = np.full((5, 5), np.nan)
                for i, w_ in enumerate(waccs):
                    for j, g_ in enumerate(gs):
                        aa = DCFAssumptionsSafe(a)
                        aa.terminal_growth = g_
                        ps = _dcf_with_wacc(cd, aa, w_)
                        mat[i, j] = ps if ps and math.isfinite(ps) else np.nan
                fig_heat = go.Figure(go.Heatmap(
                    z=mat, x=[f"{g_:.1%}" for g_ in gs], y=[f"{w_:.1%}" for w_ in waccs],
                    colorscale="RdYlGn", text=np.round(mat, 1), texttemplate="%{text}",
                    hovertemplate="WACC %{y}<br>g %{x}<br>价值 %{z:.1f}<extra></extra>",
                ))
                fig_heat.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                                      xaxis_title="永续增长率", yaxis_title="WACC")
                st.plotly_chart(fig_heat, width="stretch")

            # ===== 第二张敏感性：收入增长 × 营业利润率（业务驱动因素，借鉴 dcf-model） =====
            st.markdown("#### 敏感性分析：每股价值 vs (首年收入增长率 × 营业利润率)")
            with st.spinner("计算业务驱动敏感性..."):
                g0_ = a.revenue_growth
                m0_ = a.operating_margin
                g_grid = [max(g0_ + (i - 2) * 0.03, 0.005) for i in range(5)]
                m_grid = [min(max(m0_ + (i - 2) * 0.03, 0.01), 0.85) for i in range(5)]
                mat2 = np.full((5, 5), np.nan)
                for i, m_ in enumerate(m_grid):
                    for j, g_ in enumerate(g_grid):
                        aa = DCFAssumptionsSafe(a)
                        aa.revenue_growth = g_
                        aa.operating_margin = m_
                        r2 = run_dcf(cd, aa)
                        mat2[i, j] = r2.per_share_value if (not r2.error and math.isfinite(r2.per_share_value)) else np.nan
                fig_heat2 = go.Figure(go.Heatmap(
                    z=mat2, x=[f"{g_:.0%}" for g_ in g_grid], y=[f"{m_:.0%}" for m_ in m_grid],
                    colorscale="RdYlGn", text=np.round(mat2, 1), texttemplate="%{text}",
                    hovertemplate="收入增长 %{x}<br>利润率 %{y}<br>价值 %{z:.1f}<extra></extra>",
                ))
                fig_heat2.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                                        xaxis_title="首年收入增长率", yaxis_title="营业利润率")
                st.plotly_chart(fig_heat2, width="stretch")
                st.caption("业务驱动敏感性：增长与利润率是 DCF 价值的两大经营驱动。"
                           "对成长股通常“增长率”维度更敏感，对稳定/价值股通常“利润率”更敏感。"
                           "Excel 导出工作簿的「Sensitivity-Growth」表与此一致（活公式）。")

    # ---------- Tab3 DDM ----------
    with tab_ddm:
        st.markdown("## DDM 股利贴现模型（两阶段）")
        st.caption("适合稳定分红公司；若公司不分红会给出提示。")
        with st.expander("⚙️ 假设", expanded=True):
            from src.style_presets import style_terminal_g as _stg
            _ddm_g2_def = _stg(style, cd.market)
            from src.wacc import calc_wacc as _ddm_cw
            _ddm_re = _ddm_cw(cd)["re"]
            col1, col2, col3 = st.columns(3)
            g1 = st.number_input("阶段一股利增速", 0.0, 0.50,
                                 float(results.get("ddm_g1", 0.04)), step=0.005, format="%.3f", key="ddm_g1_in")
            g2 = st.number_input("永续股利增速", 0.0, 0.06, _ddm_g2_def, step=0.005, format="%.3f", key="ddm_g2_in")
            yrs = st.slider("阶段一年数", 3, 10, 5, key="ddm_yrs")
            ddm_re = st.number_input("股权成本 Re（默认按 CAPM）", 0.0, 0.25, _ddm_re,
                                     step=0.005, format="%.3f", key="ddm_re_in",
                                     help="贴现率，越高内在价值越低。默认取 CAPM 计算结果，可手动调整。")
        ddm_res = run_ddm(cd, growth_phase1=g1, growth_phase2=g2, years_phase1=yrs,
                          cost_of_equity=ddm_re, style=style)
        st.session_state["results"] = {**results, "ddm": ddm_res}
        if ddm_res.error:
            st.warning(ddm_res.error)
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("每股内在价值", f"{fmt(ddm_res.per_share_value)} {cd.currency}",
                        f"{fmt_pct(ddm_res.upside)} vs 现价")
            col2.metric("当前每股股利", fmt(ddm_res.dps_now, 3))
            col3.metric("股利增速", f"{ddm_res.dividend_growth:.2%}")
            col4.metric("股权成本 Re", f"{ddm_res.re:.2%}")
            st.markdown(f"**解读**：{ddm_res.detail}")
            if ddm_res.forecast is not None:
                ddf = ddm_res.forecast.copy()
                for col in ddf.columns:
                    ddf[col] = ddf[col].map(lambda x: f"{x:,.3f}" if isinstance(x, (int, float)) else x)
                st.dataframe(ddf, width="stretch", hide_index=True)

    # ---------- Tab4 反向DCF ----------
    with tab_rev:
        st.markdown("## 反向 DCF：市场隐含增长率")
        st.caption("固定 WACC/利润率等假设，反推当前股价隐含的增长预期，并与公司历史增长对比。")
        col1, col2 = st.columns(2)
        mode = col1.radio("反推对象", ["显式期首年收入增长率", "永续增长率"], index=0)
        rev_mode = "growth" if mode == "显式期首年收入增长率" else "terminal"
        with st.expander("⚙️ 固定假设（默认取 DCF 自动值）"):
            from src.style_presets import style_terminal_g as _stg2
            _rev_gterm_def = _stg2(style, cd.market)
            c1, c2 = st.columns(2)
            rev_margin = c1.number_input("营业利润率", 0.0, 0.90,
                                         float(results.get("rev_margin", a.operating_margin)), step=0.01, format="%.3f",
                                         key="rev_margin_in")
            rev_gterm = c2.number_input("永续增长率(反推增长率时用)", 0.0, 0.06,
                                        float(results.get("rev_gterm", _rev_gterm_def)), step=0.005, format="%.3f",
                                        key="rev_gterm_in")
            c3, c4 = st.columns(2)
            _rev_wacc_def = dcf_res.wacc if math.isfinite(dcf_res.wacc) else 0.10
            rev_wacc = c3.number_input("WACC（固定折现率）", 0.0, 0.25, float(_rev_wacc_def),
                                       step=0.005, format="%.3f", key="rev_wacc_in",
                                       help="默认取 DCF 页计算值；WACC 越高，反推的隐含增长率越高。")
            rev_capex = c4.number_input("资本开支/收入", 0.0, 0.60, float(a.capex_pct), step=0.005, format="%.3f",
                                        key="rev_capex_in")
            c5, c6 = st.columns(2)
            rev_da = c5.number_input("折旧摊销/收入", 0.0, 0.60, float(a.da_pct), step=0.005, format="%.3f",
                                     key="rev_da_in")
            rev_nwc = c6.number_input("营运资本变动/收入(可负)", -0.10, 0.30, float(a.nwc_pct), step=0.005, format="%.3f",
                                      key="rev_nwc_in")
        rev_res = run_reverse_dcf(cd, mode=rev_mode, margin=rev_margin, g_terminal_fixed=rev_gterm,
                                  wacc=rev_wacc, capex_pct=rev_capex, da_pct=rev_da, nwc_pct=rev_nwc,
                                  style=style)
        st.session_state["results"] = {**results, "reverse_dcf": rev_res}
        if rev_res.error:
            st.warning(rev_res.error)
        else:
            col1, col2, col3 = st.columns(3)
            if rev_res.mode == "growth":
                col1.metric("市场隐含首年收入增速", f"{rev_res.implied_growth:.2%}")
                col2.metric("公司历史增速(参考)", f"{rev_res.current_growth:.2%}" if math.isfinite(rev_res.current_growth) else "N/A")
                col3.metric("WACC", f"{rev_res.wacc:.2%}")
            else:
                col1.metric("市场隐含永续增长率", f"{rev_res.implied_terminal_growth:.2%}")
                col2.metric("WACC", f"{rev_res.wacc:.2%}")
            st.markdown(f"**解读**：{rev_res.detail}")
            st.info("当隐含增速显著高于历史增长时，说明市场已定价较高预期（风险点）；"
                    "显著低于历史时，可能存在预期差机会。")

    # ---------- Tab5 可比公司 ----------
    with tab_comps:
        st.markdown("## 可比公司法（相对估值）")
        st.caption("用同业公司中位 PE/PB/EV-EBITDA 给目标公司估值。A 股可自动识别行业并抓取同业。")
        # 自动获取同业（A股申万三级行业链路；美股暂手动）
        _peer_key = f"_auto_peers_{cd.symbol}"
        if _peer_key not in st.session_state:
            try:
                st.session_state[_peer_key] = auto_peers(cd, limit=6)
            except Exception:
                st.session_state[_peer_key] = ([], "")
        auto_peers_res, auto_sector = st.session_state[_peer_key]
        auto_str = ",".join(p["symbol"] for p in auto_peers_res)
        if auto_sector:
            names = "、".join(f'{p["name"]}({p["symbol"]})' for p in auto_peers_res)
            st.success(f"已自动识别行业 **{auto_sector}**，自动抓取同业（市值相近优先）：{names}")
        elif cd.market == "A":
            st.info("未能自动识别行业（申万分类缺失），可手动输入同业代码。")
        else:
            st.caption("美股暂不支持自动同业，请手动输入可比公司代码。")
        st.markdown(
            "**示例**：若目标是 `600519`（贵州茅台），可比公司可填：`000858`（五粮液）`000568`（泸州老窖）"
            "`600809`（山西汾酒）`002304`（洋河股份）。美股示例：`AAPL` 可比 `MSFT` `GOOGL` `META` `NVDA`。")
        hardcoded = {"600519": "000858,000568,600809,002304",
                     "AAPL": "MSFT,GOOGL,META,NVDA"}.get(cd.symbol, "")
        default_peers = auto_str or hardcoded
        peer_input = st.text_input("可比公司代码（用英文逗号分隔，留空自动使用上方同业）",
                                   value=default_peers)
        peer_list = [s.strip().upper() for s in peer_input.replace("，", ",").split(",") if s.strip()]
        # 用户手动输入时也可与自动同业合并去重（以手动为准，自动补全）
        if peer_list and auto_peers_res:
            auto_syms = [p["symbol"] for p in auto_peers_res]
            merged = list(dict.fromkeys(peer_list + auto_syms))
            if len(merged) > len(peer_list):
                peer_list = merged
        if not peer_list:
            st.info("请输入至少 1 家可比公司代码。")
        else:
            with st.spinner("抓取可比公司数据..."):
                comps_res = run_comps(cd, peer_list)
            st.session_state["results"] = {**results, "comps": comps_res}
            if comps_res.error:
                st.warning(comps_res.error)
            else:
                st.markdown("#### 可比公司倍数")
                if comps_res.peer_table is not None:
                    tbl = comps_res.peer_table.reset_index()
                    st.dataframe(tbl, width="stretch", hide_index=True)
                col1, col2, col3 = st.columns(3)
                if comps_res.median_pe and math.isfinite(comps_res.median_pe):
                    col1.metric("中位 PE", f"{comps_res.median_pe:.1f}",
                                f"目标价 {fmt(comps_res.per_share_pe)}")
                if comps_res.median_pb and math.isfinite(comps_res.median_pb):
                    col2.metric("中位 PB", f"{comps_res.median_pb:.2f}",
                                f"目标价 {fmt(comps_res.per_share_pb)}")
                if comps_res.median_ev_ebitda and math.isfinite(comps_res.median_ev_ebitda):
                    col3.metric("中位 EV/EBITDA", f"{comps_res.median_ev_ebitda:.1f}",
                                f"目标价 {fmt(comps_res.per_share_ev_ebitda)}")
                st.markdown(f"**结论**：{comps_res.detail}")

    # ---------- Tab6 综合研判 ----------
    with tab_summary:
        st.markdown("## 综合研判")
        results = st.session_state.get("results", {})

        # ========== 模型校准状态区（历史回测自主学习） ==========
        try:
            from src.calibration.app_utils import calib_context, apply_calib
            _calib = calib_context(_actual, cd.market)
            calib_enabled = st.session_state.get("calib_enabled", True)
        except Exception:
            _calib = {"has_data": False}
            calib_enabled = False

        with st.expander("🧠 模型校准状态（基于历史回测的自主学习）", expanded=_calib["has_data"]):
            if _calib["has_data"]:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("校准系数（DCF）", f"{_calib['factor']:.3f}",
                          "内在价值 × 系数")
                c2.metric("校准系数（DDM）", f"{_calib['ddm_factor']:.3f}")
                c3.metric("回测样本数", f"{_calib['samples']}")
                vs = _calib.get("valid_samples", 0)
                hr = _calib.get("hit_rate")
                c4.metric("方向命中率", f"{hr:.0%}" if isinstance(hr, (int, float)) and np.isfinite(hr) else "N/A")
                me = _calib.get("mean_error")
                st.caption(
                    f"历史均值误差 **{me:+.1%}**（>0 表示模型系统性高估 → 系数<1 下调；<0 表示系统性低估 → 系数>1 上调）。"
                    f"校准表更新于 **{_calib.get('updated', '未知')[:10]}**。"
                    f"当前股票判定风格 **{_actual}**，匹配层 {_calib['key']}"
                    f"（原始样本 {_calib['samples']}，失真过滤后有效 **{vs}**）。")
                st.toggle("启用历史校准（对 DCF/DDM 内在价值应用系数）",
                          value=calib_enabled, key="calib_enabled")
                if not calib_enabled:
                    st.info("校准已关闭，下方估值将使用未校准的原始值。")
            else:
                st.info("本风格/市场暂无可用的校准数据。校准系统会随历史回测样本积累自动生效"
                        "（每个 风格×市场 分层至少 3 个有效样本）。")
                st.caption("说明：校准系统在历史时点用当时可得数据重跑估值模型，对比之后 1 年实际表现，"
                           "按 风格×市场 分层统计偏差生成系数，用于修正系统性高估/低估。")
            _calib_enabled = st.session_state.get("calib_enabled", True)

        # 方法对比表（应用校准系数）
        rows = []
        dcf_r = results.get("dcf")
        if dcf_r and not getattr(dcf_r, "error", "") and getattr(dcf_r, "per_share_value", None) and math.isfinite(dcf_r.per_share_value):
            _v = dcf_r.per_share_value
            _up = dcf_r.upside
            if _calib_enabled and _calib.get("has_data"):
                _v = apply_calib(_v, "dcf", _calib)
                _up = _v / cd.latest_price() - 1 if cd.latest_price() and cd.latest_price() > 0 else np.nan
            rows.append(["DCF 现金流折现" + ("（已校准）" if _calib_enabled and _calib.get("has_data") else ""),
                         _v, _up, dcf_r.conclusion])
        ddm_r = results.get("ddm")
        if ddm_r and not getattr(ddm_r, "error", "") and getattr(ddm_r, "per_share_value", None) and math.isfinite(ddm_r.per_share_value):
            _v = ddm_r.per_share_value
            _up = ddm_r.upside
            if _calib_enabled and _calib.get("has_data"):
                _v = apply_calib(_v, "ddm", _calib)
                _up = _v / cd.latest_price() - 1 if cd.latest_price() and cd.latest_price() > 0 else np.nan
            rows.append(["DDM 股利贴现" + ("（已校准）" if _calib_enabled and _calib.get("has_data") else ""),
                         _v, _up, ddm_r.conclusion])
        rev_r = results.get("reverse_dcf")
        comps_r = results.get("comps")
        if comps_r and not getattr(comps_r, "error", ""):
            vals = []
            for v in (comps_r.per_share_pe, comps_r.per_share_pb, comps_r.per_share_ev_ebitda):
                if isinstance(v, (int, float)) and math.isfinite(v) and v > 0:
                    vals.append(v)
            if vals:
                avg = float(sum(vals) / len(vals))
                price = cd.latest_price()
                up = avg / price - 1 if price and price > 0 else np.nan
                rows.append(["可比公司(均值)", avg, up, comps_r.conclusion])

        if rows:
            st.markdown("### 方法对比")
            comp = pd.DataFrame(rows, columns=["方法", "每股价值", "相对现价", "结论"])
            disp = comp.copy()
            disp["每股价值"] = disp["每股价值"].map(lambda x: f"{x:,.2f}")
            disp["相对现价"] = disp["相对现价"].map(lambda x: f"{x:+.1%}")
            st.dataframe(disp, width="stretch", hide_index=True)

            fig_cmp = go.Figure()
            fig_cmp.add_bar(x=comp["方法"], y=comp["每股价值"], name="各方法内在价值",
                            marker_color=["steelblue", "seagreen", "orange", "purple"][:len(comp)])
            price_now = cd.latest_price()

            # 推荐合理估值 = 各方法中位数（稳健，不受极端值干扰）
            _vals = [float(v) for v in comp["每股价值"].tolist() if isinstance(v, (int, float)) and math.isfinite(v)]
            if _vals:
                median_v = float(np.median(_vals))
                lo_v, hi_v = min(_vals), max(_vals)
            else:
                median_v = lo_v = hi_v = np.nan

            fig_cmp.add_hline(y=price_now, line_dash="dash", line_color="red",
                              annotation_text=f"当前价 {price_now:.2f}")
            if math.isfinite(median_v):
                fig_cmp.add_hline(y=median_v, line_dash="dot", line_color="seagreen",
                                  annotation_text=f"推荐 {median_v:.2f}")
            fig_cmp.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_cmp, width="stretch")

            # ========== 推荐合理估值区块 ==========
            if math.isfinite(median_v):
                st.markdown("### 🎯 推荐合理估值")
                up = median_v / price_now - 1 if price_now and price_now > 0 else np.nan
                if up >= 0.2:
                    rec_concl, rec_txt = "低估", "推荐合理估值高于当前价，安全边际较充足，具备配置价值。"
                elif up >= 0:
                    rec_concl, rec_txt = "合理偏低", "推荐合理估值略高于当前价，处于合理区间下沿。"
                elif up >= -0.2:
                    rec_concl, rec_txt = "合理偏高", "推荐合理估值略低于当前价，处于合理区间上沿。"
                else:
                    rec_concl, rec_txt = "高估", "推荐合理估值低于当前价，当前价格透支较多预期，需谨慎。"

                rc1, rc2, rc3 = st.columns(3)
                rc1.metric("推荐合理估值（中位数）", f"{median_v:.2f} {cd.currency}",
                           f"{up:+.1%} vs 现价")
                rc2.metric("合理区间", f"{lo_v:.2f} ~ {hi_v:.2f} {cd.currency}")
                rc3.metric("综合结论", rec_concl)

                shares = cd.shares
                if not math.isfinite(shares) or shares <= 0:
                    shares = cd.last_value("shares")
                if math.isfinite(shares) and shares > 0:
                    rec_mcap, lo_mcap, hi_mcap = median_v * shares, lo_v * shares, hi_v * shares
                    cur_mcap = cd.market_cap
                    _calib_note = "（已按历史校准调整）" if _calib_enabled and _calib.get("has_data") else ""
                    if math.isfinite(cur_mcap) and cur_mcap > 0:
                        st.markdown(f"**对应合理市值**：约 **{fmt_big(rec_mcap)}**{_calib_note}（区间 {fmt_big(lo_mcap)} ~ "
                                    f"{fmt_big(hi_mcap)}），当前市值 {fmt_big(cur_mcap)}。{rec_txt}")
                    else:
                        st.markdown(f"**对应合理市值**：约 **{fmt_big(rec_mcap)}**{_calib_note}。{rec_txt}")
                else:
                    st.markdown(f"**解读**：{rec_txt}")

                # 方法一致性提示：DCF 与 DDM 分歧过大时提醒（银行/保险等资本消耗型行业 FCFF 法常偏高）
                _dcf_v = next((r_[1] for r_ in rows if str(r_[0]).startswith("DCF")
                               and isinstance(r_[1], (int, float)) and math.isfinite(r_[1])), np.nan)
                _ddm_v = next((r_[1] for r_ in rows if str(r_[0]).startswith("DDM")
                               and isinstance(r_[1], (int, float)) and math.isfinite(r_[1])), np.nan)
                if math.isfinite(_dcf_v) and math.isfinite(_ddm_v) and _ddm_v > 0:
                    gap = _dcf_v / _ddm_v
                    if gap > 1.6:
                        st.warning(f"⚠️ **方法分歧较大**：DCF 估值约 {_dcf_v:.0f}，约为 DDM（{_ddm_v:.0f}）的 "
                                   f"{gap:.1f} 倍。若为银行/保险等资本消耗型行业，FCFF 类 DCF 通常系统性偏高，"
                                   "建议以 **DDM 或可比法**为主要参考。")

        st.markdown("### AI 综合研判")
        st.markdown("默认使用本地规则引擎汇总。接入大模型 API 后可获得更深入的定性分析。")

        with st.expander("⚙️ 配置 AI 接口（可选）"):
            provider = st.selectbox("服务商", list(PROVIDERS.keys()))
            api_key = st.text_input("API Key", type="password",
                                    help="Key 只保存在当前会话，不会上传服务器")
            model = st.text_input("模型名（留空用默认）",
                                  value=PROVIDERS[provider]["model"])
            extra_note = st.text_input("补充要求/背景（可选）", placeholder="如：重点关注XX风险")
        ai_btn = st.button("✨ 生成 AI 综合研判")

        if ai_btn and api_key:
            try:
                with st.spinner("AI 分析中..."):
                    prompt = build_prompt(cd, results, {"notes": extra_note} if extra_note else None)
                    ai_text = call_ai(prompt, provider, api_key, model)
                st.success("AI 综合研判生成完毕")
                st.markdown(ai_text)
            except Exception as e:
                st.error(f"AI 调用失败：{e}\n请检查 API Key、服务商与模型名是否正确。")
        else:
            st.markdown(rule_based_summary(cd, results))
            if not api_key:
                st.caption("提示：在上方配置 API Key 后可生成更深入的 AI 综合研判。")

    st.divider()
    st.caption("⚠️ 免责声明：本工具基于公开财务数据自动计算，仅供研究参考，不构成任何投资建议。"
               "数据可能滞后或有误，重要决策请以公司公告和独立尽调为准。")
