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
from src.models.comps import run_comps
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
    """用指定 WACC 重算 DCF 每股价值（敏感性分析用）。"""
    a = DCFAssumptions()
    for f in ["forecast_years", "revenue_growth", "growth_decline", "accel", "terminal_growth",
              "operating_margin", "tax_rate", "capex_pct", "da_pct", "nwc_pct",
              "beta", "debt_rate", "erp", "rf"]:
        setattr(a, f, getattr(assump, f))
    r = run_dcf(cd, a)
    if r.error or r.forecast is None or len(r.forecast) == 0:
        return np.nan
    rows = r.forecast.copy()
    ev = sum(rows["FCFF"].iloc[i] / (1 + wacc) ** (i + 1) for i in range(len(rows)))
    if wacc <= a.terminal_growth:
        return np.nan
    tv = rows["FCFF"].iloc[-1] * (1 + a.terminal_growth) / (wacc - a.terminal_growth)
    ev += tv / (1 + wacc) ** a.forecast_years
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
            st.session_state["dcf_assump"] = auto_assumptions(cd, wacc_info, style=style)
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

            # 敏感性热力图
            st.markdown("#### 敏感性分析：每股价值 vs (WACC × 永续增长率)")
            with st.spinner("计算敏感性..."):
                base_wacc = dcf_res.wacc
                g_base = a.terminal_growth
                waccs = [base_wacc + (i - 2) * 0.005 for i in range(5)]
                gs = [max(g_base + (i - 2) * 0.005, 0.0) for i in range(5)]
                mat = np.zeros((5, 5))
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

    # ---------- Tab3 DDM ----------
    with tab_ddm:
        st.markdown("## DDM 股利贴现模型（两阶段）")
        st.caption("适合稳定分红公司；若公司不分红会给出提示。")
        with st.expander("⚙️ 假设", expanded=True):
            from src.style_presets import style_terminal_g as _stg
            _ddm_g2_def = _stg(style, cd.market)
            col1, col2, col3 = st.columns(3)
            g1 = st.number_input("阶段一股利增速", 0.0, 0.50,
                                 float(results.get("ddm_g1", 0.04)), step=0.005, format="%.3f", key="ddm_g1_in")
            g2 = st.number_input("永续股利增速", 0.0, 0.06, _ddm_g2_def, step=0.005, format="%.3f", key="ddm_g2_in")
            yrs = st.slider("阶段一年数", 3, 10, 5, key="ddm_yrs")
        ddm_res = run_ddm(cd, growth_phase1=g1, growth_phase2=g2, years_phase1=yrs, style=style)
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
                                         float(results.get("rev_margin", 0.20)), step=0.01, format="%.3f")
            rev_gterm = c2.number_input("永续增长率(反推增长率时用)", 0.0, 0.06,
                                        float(results.get("rev_gterm", _rev_gterm_def)), step=0.005, format="%.3f")
        rev_res = run_reverse_dcf(cd, mode=rev_mode, margin=rev_margin,
                                  g_terminal_fixed=rev_gterm, style=style)
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
        st.caption("输入同业公司代码，用同业中位 PE/PB/EV-EBITDA 给目标公司估值。")
        st.markdown(
            "**示例**：若目标是 `600519`（贵州茅台），可比公司可填：`000858`（五粮液）`000568`（泸州老窖）"
            "`600809`（山西汾酒）`002304`（洋河股份）。美股示例：`AAPL` 可比 `MSFT` `GOOGL` `META` `NVDA`。")
        default_peers = {"600519": "000858,000568,600809,002304",
                         "AAPL": "MSFT,GOOGL,META,NVDA"}.get(cd.symbol, "")
        peer_input = st.text_input("可比公司代码（用英文逗号分隔）", value=default_peers)
        peer_list = [s.strip().upper() for s in peer_input.replace("，", ",").split(",") if s.strip()]
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
        # 方法对比表
        rows = []
        dcf_r = results.get("dcf")
        if dcf_r and not getattr(dcf_r, "error", "") and getattr(dcf_r, "per_share_value", None) and math.isfinite(dcf_r.per_share_value):
            rows.append(["DCF 现金流折现", dcf_r.per_share_value, dcf_r.upside, dcf_r.conclusion])
        ddm_r = results.get("ddm")
        if ddm_r and not getattr(ddm_r, "error", "") and getattr(ddm_r, "per_share_value", None) and math.isfinite(ddm_r.per_share_value):
            rows.append(["DDM 股利贴现", ddm_r.per_share_value, ddm_r.upside, ddm_r.conclusion])
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
            fig_cmp.add_hline(y=price_now, line_dash="dash", line_color="red",
                              annotation_text=f"当前价 {price_now:.2f}")
            fig_cmp.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_cmp, width="stretch")

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
