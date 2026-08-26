# -*- coding: utf-8 -*-
"""
可比公司法：用同业公司的 PE / PB / EV-EBITDA 中位数给目标公司估值。

可比公司来源：
  1. 自动获取同业（A股，优先，申万三级行业链路）：
       cninfo 行业变更接口(个股→申万三级行业名) → 申万三级行业列表(名→代码)
       → 申万三级成分接口(代码→同业成分+PE/PB/ROE/股息/市值/增速)
     入选规则：剔除自身、市值相近优先（|log(市值比)| 最小）、排除极端值。
  2. 手动输入（最可靠，推荐）：逐只用腾讯行情接口拉 PE/PB。
  3. 美股：暂缓（暂不支持自动同业，手动输入）。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data_fetcher import CompanyData

# 缓存放 data/ 根目录（提交到 GitHub）：云上 Streamlit 直接读缓存，
# 无需实时拉取申万列表/成分，避免网络依赖与接口限流。本地预取后提交即可。
_SW_INFO_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sw_level3_info.json")
_SW_INFO_TTL = 7 * 24 * 3600  # 申万行业列表 7 天缓存（分类相对静态）
_SW_CONS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sw_cons")
_SW_CONS_TTL = 24 * 3600      # 申万行业成分 24h 缓存（含 PE/PB 快照，避免重复拉取触发限流）

_sw_info_cache: dict | None = None
_sw_cons_cache: dict = {}


def _norm_a_symbol(sym: str) -> str:
    """A股 6 位代码 → 腾讯格式 (sh/sz 前缀)。"""
    sym = str(sym).strip().lower()
    if not sym:
        return sym
    if sym.startswith(("sh", "sz", "bj")):
        return sym
    sym = sym.zfill(6)
    if sym.startswith(("6", "9")):
        return "sh" + sym
    if sym.startswith(("4", "8")):
        return "bj" + sym
    return "sz" + sym


@dataclass
class CompsResult:
    peer_table: pd.DataFrame | None = None
    per_share_pe: float = np.nan
    per_share_pb: float = np.nan
    per_share_ev_ebitda: float = np.nan
    median_pe: float = np.nan
    median_pb: float = np.nan
    median_ev_ebitda: float = np.nan
    upside_pe: float = np.nan
    upside_pb: float = np.nan
    upside_ev_ebitda: float = np.nan
    conclusion: str = ""
    detail: str = ""
    sector_name: str = ""  # 自动识别到的行业名
    error: str = ""


def _norm_sym(sym: str) -> str:
    """成分接口返回 603589.SH → 603589"""
    return str(sym).split(".")[0].strip().zfill(6)


def _load_sw_level3_info(force: bool = False) -> dict:
    """申万三级行业信息（行业名称→行业代码）。内存 + 本地文件 24h 缓存。"""
    global _sw_info_cache
    now = time.time()
    if not force and _sw_info_cache is not None:
        return _sw_info_cache
    if not force and os.path.exists(_SW_INFO_FILE):
        try:
            with open(_SW_INFO_FILE, encoding="utf-8") as f:
                d = json.load(f)
            if now - d.get("ts", 0) < _SW_INFO_TTL and d.get("rows"):
                _sw_info_cache = d["rows"]
                return _sw_info_cache
        except Exception:
            pass
    try:
        import warnings
        import akshare as ak
        warnings.filterwarnings("ignore")
        df = ak.sw_index_third_info()
        rows: dict = {}
        for _, r in df.iterrows():
            code = str(r["行业代码"]).replace(".SI", "").strip()
            rows[str(r["行业名称"]).strip()] = code
        _sw_info_cache = rows
        try:
            os.makedirs(os.path.dirname(_SW_INFO_FILE), exist_ok=True)
            with open(_SW_INFO_FILE, "w", encoding="utf-8") as f:
                json.dump({"ts": now, "rows": rows}, f, ensure_ascii=False)
        except Exception:
            pass
        return rows
    except Exception:
        return _sw_info_cache or {}


def _cninfo_sw_name(symbol6: str) -> str | None:
    """个股→申万三级行业名。巨潮行业变更接口。

    优先取「申银万国行业分类标准」（新版，不含"旧"）最新一条——其三级行业名与
    当前申万 2021 版一致（如移动→电信运营商）。旧版名称（通信运营等）与现行
    sw_index 体系不一致，仅作兜底。
    """
    try:
        import warnings
        import akshare as ak
        warnings.filterwarnings("ignore")
        df = ak.stock_industry_change_cninfo(symbol=symbol6)
        if df is None or not len(df):
            return None
        # 新版申万标准（精确匹配，不含"(旧)"）
        sw_new = df[df["分类标准"].astype(str) == "申银万国行业分类标准"]
        if len(sw_new):
            if "变更日期" in sw_new.columns:
                sw_new = sw_new.sort_values("变更日期", na_position="last")
            row = sw_new.iloc[-1]
            name = str(row["行业中类"]).strip()
            if name and name.lower() != "nan":
                return name
            name = str(row["行业大类"]).strip()
            if name and name.lower() != "nan":
                return name
        # 兜底：含"申银万国"（含旧版）最新一条
        sw_any = df[df["分类标准"].astype(str).str.contains("申银万国")]
        if len(sw_any):
            if "变更日期" in sw_any.columns:
                sw_any = sw_any.sort_values("变更日期", na_position="last")
            row = sw_any.iloc[-1]
            name = str(row["行业中类"]).strip()
            if name and name.lower() != "nan":
                return name
            name = str(row["行业大类"]).strip()
            if name and name.lower() != "nan":
                return name
    except Exception:
        pass
    return None


def _cninfo_giant_name(symbol6: str) -> str | None:
    """无申万记录时，用巨潮分类「行业中类」作兜底（如 多元化保险→保险Ⅲ）。

    只用行业中类（更具体，避免"专用设备"等宽泛大类错配到无关行业）。
    """
    try:
        import warnings
        import akshare as ak
        warnings.filterwarnings("ignore")
        df = ak.stock_industry_change_cninfo(symbol=symbol6)
        if df is None or not len(df):
            return None
        gj = df[df["分类标准"].astype(str) == "巨潮行业分类标准"]
        if not len(gj):
            gj = df[df["分类标准"].astype(str).str.contains("巨潮")]
        if not len(gj):
            return None
        if "变更日期" in gj.columns:
            gj = gj.sort_values("变更日期", na_position="last")
        name = str(gj.iloc[-1]["行业中类"]).strip()
        if name and name.lower() != "nan":
            return name
    except Exception:
        pass
    return None


def _match_sw_code(sw_name: str) -> str | None:
    """申万三级行业名 → 行业代码。精确匹配优先，其次去罗马数字、包含匹配。"""
    info = _load_sw_level3_info()
    if not info:
        return None
    if sw_name in info:
        return info[sw_name]
    import re
    norm = re.sub(r"[ⅠⅡⅢⅣⅤ]", "", sw_name).strip()
    for k, v in info.items():
        if re.sub(r"[ⅠⅡⅢⅣⅤ]", "", k).strip() == norm:
            return v
    for k, v in info.items():
        kn = re.sub(r"[ⅠⅡⅢⅣⅤ]", "", k).strip()
        if kn and (norm in kn or kn in norm):
            return v
    return None


def _sw_cons(code: str) -> pd.DataFrame | None:
    """申万三级成分（含 PE/PB/ROE/股息/市值/增速）。

    三层缓存：内存 → 本地文件(24h TTL) → 实时拉取并回写缓存。
    申万成分接口有频率限制，缓存可避免重复请求触发限流。
    """
    if code in _sw_cons_cache:
        return _sw_cons_cache[code]
    # 本地文件缓存
    fpath = os.path.join(_SW_CONS_DIR, f"{code}.json")
    now = time.time()
    if os.path.exists(fpath):
        try:
            with open(fpath, encoding="utf-8") as f:
                d = json.load(f)
            if now - d.get("ts", 0) < _SW_CONS_TTL and d.get("rows"):
                df = pd.DataFrame(d["rows"])
                _sw_cons_cache[code] = df
                return df
        except Exception:
            pass
    try:
        import warnings
        import akshare as ak
        warnings.filterwarnings("ignore")
        df = ak.sw_index_third_cons(symbol=code + ".SI")
        if df is not None and len(df):
            _sw_cons_cache[code] = df
            try:
                os.makedirs(_SW_CONS_DIR, exist_ok=True)
                rows = df.to_dict("records")
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump({"ts": now, "code": code, "rows": rows}, f,
                              ensure_ascii=False, default=str)
            except Exception:
                pass
            return df
    except Exception:
        pass
    return None


def _to_peer_dict(row: pd.Series) -> dict:
    """成分行 → 可比公司 dict（归一化列名 + 数值化）"""
    def num(v):
        try:
            f = float(v)
            return f if np.isfinite(f) else np.nan
        except Exception:
            return np.nan
    sym = _norm_sym(row.get("股票代码"))
    return {
        "symbol": sym,
        "name": str(row.get("股票简称", "")).strip(),
        "pe": num(row.get("市盈率ttm") if pd.notna(row.get("市盈率ttm")) else row.get("市盈率")),
        "pb": num(row.get("市净率")),
        "roe": num(row.get("ROE(%)")),
        "div_yield": num(row.get("股息率")),
        "mkt_cap": num(row.get("市值")),          # 亿元
        "net_growth": num(row.get("净利润增速(%)")),
        "rev_growth": num(row.get("营收增速(%)")),
        "ev_ebitda": np.nan,                       # 申万成分接口无 EBITDA
    }


def auto_peers(cd: CompanyData, limit: int = 6) -> tuple[list[dict], str]:
    """自动获取同业可比公司。返回 (peer_dicts, sector_name)。失败返回 ([], "")。"""
    if cd.market != "A":
        return [], ""   # 美股暂缓
    symbol6 = str(cd.symbol).strip().zfill(6)
    sw_name = _cninfo_sw_name(symbol6)
    code = _match_sw_code(sw_name) if sw_name else None
    if not code:
        # 无申万记录 → 巨潮行业大类兜底（保险/银行等通用名）
        alt = _cninfo_giant_name(symbol6)
        if alt:
            code = _match_sw_code(alt)
            if code:
                sw_name = alt
    if not sw_name:
        return [], ""
    if not code:
        return [], sw_name  # 知道行业但无申万代码 → 让上层提示
    cons = _sw_cons(code)
    if cons is None or not len(cons):
        return [], sw_name
    peers = [_to_peer_dict(r) for _, r in cons.iterrows()]
    peers = [p for p in peers if p["symbol"] != symbol6]
    if not peers:
        return [], sw_name
    # 市值相近优先（|log 比值| 最小）+ 排除市值缺失
    peers = [p for p in peers if np.isfinite(p["mkt_cap"]) and p["mkt_cap"] > 0]
    target_mv = cd.last_value("mkt_cap")
    if np.isfinite(target_mv) and target_mv > 0:
        peers.sort(key=lambda p: abs(np.log(p["mkt_cap"] / target_mv)))
    else:
        peers.sort(key=lambda p: -p["mkt_cap"])
    return peers[:limit], sw_name


def _a_peer_metrics(symbol6: str) -> dict:
    """A股可比公司：腾讯接口拿 PE/PB（EV/EBITDA 需财务数据，暂略）"""
    import requests
    import re
    url = f"https://qt.gtimg.cn/q={_norm_a_symbol(symbol6)}"
    r = requests.get(url, timeout=10)
    r.encoding = "gbk"
    m = re.search(r'="(.*?)"', r.text)
    if not m:
        return {}
    f = m.group(1).split("~")
    out = {"name": f[1] if len(f) > 1 else symbol6, "symbol": symbol6}
    try:
        out["pe"] = float(f[39]) if f[39] else np.nan
    except Exception:
        out["pe"] = np.nan
    try:
        out["pb"] = float(f[46]) if f[46] else np.nan
    except Exception:
        out["pb"] = np.nan
    return out


def _us_peer_metrics(symbol: str, session=None) -> dict:
    """美股可比公司：腾讯接口拿 PE/PB（EV/EBITDA 缺失则略）"""
    import requests
    import re
    try:
        r = requests.get(f"https://qt.gtimg.cn/q=us{symbol}", timeout=10)
        r.encoding = "gbk"
        m = re.search(r'="(.*?)"', r.text)
        if not m or "pv_none_match" in r.text:
            return {}
        f = m.group(1).split("~")
        out = {"name": f[1] if len(f) > 1 else symbol, "symbol": symbol}
        try:
            out["pe"] = float(f[39]) if f[39] else np.nan
        except Exception:
            out["pe"] = np.nan
        out["pb"] = np.nan  # 腾讯美股无可靠 PB 字段
        out["ev_ebitda"] = np.nan
        return out
    except Exception:
        return {}


def _fetch_peer(sym: str, cd: CompanyData, session=None) -> dict:
    """按市场拉取单只可比公司数据（手动输入场景）。"""
    if cd.market == "A":
        m = _a_peer_metrics(sym)
        m["ev_ebitda"] = np.nan
    else:
        m = _us_peer_metrics(sym, session=session)
    m["symbol"] = sym
    return m


def run_comps(
    cd: CompanyData,
    peers,
    session=None,
) -> CompsResult:
    """可比公司估值。

    peers 支持两种形态：
      - list[str]：手动输入的代码，逐只用腾讯行情拉取 PE/PB；
      - list[dict]：自动获取的同业（含 pe/pb/roe/div/mkt_cap/growth）。
    """
    res = CompsResult()
    rows: list[dict] = []
    if peers and isinstance(peers[0], str):
        for sym in peers:
            sym = str(sym).strip().upper()
            if not sym:
                continue
            try:
                m = _fetch_peer(sym, cd, session=session)
                if m.get("pe") is not None or m.get("pb") is not None:
                    rows.append(m)
            except Exception:
                continue
    else:
        for m in peers:
            if not isinstance(m, dict):
                continue
            if (m.get("pe") is not None and not np.isnan(m.get("pe", np.nan))) or \
               (m.get("pb") is not None and not np.isnan(m.get("pb", np.nan))):
                rows.append(dict(m))

    if not rows:
        res.error = "未能获取可比公司数据，请检查可比公司代码。"
        return res

    df = pd.DataFrame(rows)
    if "symbol" in df.columns and len(df):
        df = df.set_index("symbol", drop=False)
    df = df[~df.index.duplicated(keep="first")]
    res.peer_table = df
    res.sector_name = getattr(cd, "sector_name", "") or ""

    med_pe = pd.to_numeric(df["pe"], errors="coerce").median()
    med_pb = pd.to_numeric(df.get("pb", pd.Series(dtype=float)), errors="coerce").median()
    med_ev = pd.to_numeric(df.get("ev_ebitda", pd.Series(dtype=float)), errors="coerce").median()
    res.median_pe = med_pe
    res.median_pb = med_pb
    res.median_ev_ebitda = med_ev

    shares = cd.shares
    if np.isnan(shares) or shares <= 0:
        shares = cd.last_value("shares")
    price = cd.latest_price()
    net_income = cd.last_value("net_income")
    total_equity = cd.last_value("total_equity")
    ebitda = cd.last_value("ebitda")

    results = {}
    if np.isfinite(med_pe) and np.isfinite(net_income) and net_income > 0 and np.isfinite(shares):
        res.per_share_pe = med_pe * net_income / shares
        results["PE"] = res.per_share_pe
        res.upside_pe = res.per_share_pe / price - 1 if np.isfinite(price) and price > 0 else np.nan
    if np.isfinite(med_pb) and np.isfinite(total_equity) and total_equity > 0 and np.isfinite(shares):
        res.per_share_pb = med_pb * total_equity / shares
        results["PB"] = res.per_share_pb
        res.upside_pb = res.per_share_pb / price - 1 if np.isfinite(price) and price > 0 else np.nan
    if (np.isfinite(med_ev) and np.isfinite(ebitda) and ebitda > 0
            and np.isfinite(shares)):
        debt = cd.last_value("total_debt")
        cash = cd.last_value("cash")
        if not np.isfinite(debt) or debt < 0:
            debt = 0.0
        if not np.isfinite(cash) or cash < 0:
            cash = 0.0
        ev = med_ev * ebitda
        equity = ev - (debt - cash)
        res.per_share_ev_ebitda = equity / shares
        results["EV/EBITDA"] = res.per_share_ev_ebitda
        res.upside_ev_ebitda = res.per_share_ev_ebitda / price - 1 if np.isfinite(price) and price > 0 else np.nan

    if not results:
        res.error = "可比公司估值所需数据不足（净利润/净资产/EBITDA）。"
        return res

    avg = float(np.mean(list(results.values())))
    res.conclusion = "低估" if avg >= price * 1.2 else ("高估" if avg <= price * 0.8 else "合理")
    res.detail = (f"基于 {len(df)} 家可比公司（{'行业：' + res.sector_name + '，' if res.sector_name else ''}"
                  f"中位 PE {med_pe:.1f}/PB {med_pb:.2f}），PE/PB 中位数估值均值约 "
                  f"{avg/price:.0%} 倍当前价。")
    return res
