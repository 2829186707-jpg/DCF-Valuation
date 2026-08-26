# -*- coding: utf-8 -*-
"""
分层校准系数：基于历史回测误差，按「估值风格 × 市场」产出校准系数。

口径说明：
  - 误差 = 模型隐含预期收益 − 实际超额收益（相对同期指数，剥离市场 Beta）
  - 误差 > 0 → 模型系统性高估（预期未兑现）→ 校准系数 < 1 下调
  - 校准系数 = 1 − mean(误差)；用中位数抗极端值
  - 防过拟合：样本 < 8 时置信度低，系数向 1.0 收缩（shrinkage）
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CALIB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          "data", "calibration.json")


@dataclass
class StyleCalib:
    style: str
    market: str
    factor: float = 1.0        # 校准系数（DCF 内在价值 × factor）
    ddm_factor: float = 1.0
    samples: int = 0
    hit_rate: float = np.nan   # 方向命中率（隐含预期与超额的符号一致率）
    mean_error: float = np.nan
    note: str = ""


def _shrink_factor(mean_err: float, n: int, min_n: int = 8) -> float:
    """收缩：样本不足时向 1.0 靠拢，避免小样本过拟合。"""
    f = 1.0 - mean_err
    if n < min_n:
        w = n / min_n
        f = 1.0 + (f - 1.0) * w
    # 限制校准系数范围（防止过度修正）
    return float(np.clip(f, 0.6, 1.4))


def _valid_sample(s, method: str) -> tuple[bool, float, float, float]:
    """质量门槛：判断回测样本对该估值方法是否有效。
    返回 (是否有效, 误差, 隐含收益, 该方法的每股价值)。
    排除：
      - 该方法报错 / 值为 nan
      - 隐含预期收益 |implied| > 3.0（>300%，结构性失真，多为金融/地产 FCFF 失真）
      - DCF 与 DDM 分歧 > 2 倍（两种方法打架，各自失真）
    """
    implied = getattr(s, f"implied_{method}", np.nan)
    err = getattr(s, f"error_{method}_1y", np.nan)
    val = getattr(s, f"intrinsic_{method}", np.nan)
    if not np.isfinite(implied) or not np.isfinite(err) or not np.isfinite(val):
        return False, np.nan, np.nan, np.nan
    if abs(implied) > 3.0:
        return False, np.nan, np.nan, np.nan
    # 方法分歧（两种方法都有效时才比较）
    other = getattr(s, f"intrinsic_{'ddm' if method == 'dcf' else 'dcf'}", np.nan)
    if np.isfinite(other) and np.isfinite(val) and other > 0 and val > 0:
        ratio = val / other
        if ratio > 2.0 or ratio < 0.5:
            return False, np.nan, np.nan, np.nan
    return True, err, implied, val


def compute_calibration(samples: list, method: str = "dcf",
                        ref_year: int | None = None,
                        time_decay: float = 0.20) -> dict[str, StyleCalib]:
    """从回测样本计算分层校准系数（按 风格×市场，含失真样本过滤 + 时间衰减）。

    时间衰减：近期样本权重更高（w = exp(-time_decay * (ref_year - year))），
    使校准系数更快反映市场结构的最新变化，旧样本缓慢退出。
    ref_year 缺省取样本最大年份；time_decay=0 关闭衰减（等权）。
    """
    from datetime import datetime
    ref_year = ref_year or (max(int(s.year) for s in samples if s.year) if samples else datetime.now().year)
    calib: dict[str, StyleCalib] = {}
    keys = {}
    for s in samples:
        if s.year is None:
            continue
        keys.setdefault((s.style, s.market), []).append(s)

    def _weighted_median_approx(errs, ws):
        """加权中位数近似：按误差排序后累计权重到 50% 处的误差。"""
        order = np.argsort(errs)
        errs_s = np.asarray(errs)[order]
        ws_s = np.asarray(ws)[order]
        cw = np.cumsum(ws_s)
        if cw[-1] <= 0:
            return float(np.median(errs_s))
        half = cw[-1] / 2
        idx = int(np.searchsorted(cw, half))
        idx = min(idx, len(errs_s) - 1)
        return float(errs_s[idx])

    for (style, market), group in keys.items():
        valid = [_valid_sample(s, method) for s in group]
        valid = [v for v in valid if v[0]]
        if len(valid) < 3:
            continue
        errs = [v[1] for v in valid]
        ws = [float(np.exp(-time_decay * (ref_year - int(s.year))))
              for s, v in zip(group, valid) if v[0]]
        # valid 顺序与 group 对应，重建 ws
        ws = []
        for s in group:
            vv = _valid_sample(s, method)
            if vv[0]:
                ws.append(float(np.exp(-time_decay * (ref_year - int(s.year)))))
        mean_err = _weighted_median_approx(errs, ws)
        # 方向命中率：隐含预期与实际超额同号的比例
        hit = 0
        total = 0
        for s in group:
            implied = getattr(s, f"implied_{method}", np.nan)
            exc = s.excess_1y
            if np.isfinite(implied) and np.isfinite(exc) and abs(exc) > 0.02 and abs(implied) <= 3.0:
                total += 1
                if (implied > 0) == (exc > 0):
                    hit += 1
        hit_rate = hit / total if total else np.nan
        factor = _shrink_factor(mean_err, len(errs))
        calib[f"{market}:{style}"] = StyleCalib(
            style=style, market=market, factor=factor,
            ddm_factor=_shrink_factor(float(np.median(
                [v[1] for v in valid if np.isfinite(v[1])])), len(errs)),
            samples=len(group), hit_rate=hit_rate, mean_error=mean_err,
        )
    return calib


def save_calibration(calib: dict[str, StyleCalib], path: str | None = None,
                     time_decay: float = 0.20, ref_year: int | None = None) -> str:
    p = path or CALIB_PATH
    os.makedirs(os.path.dirname(p), exist_ok=True)
    payload = {
        "version": 2,
        "updated": pd.Timestamp.now().isoformat(),
        "time_decay": time_decay,
        "ref_year": ref_year,
        "styles": {
            k: {"style": v.style, "market": v.market, "factor": v.factor,
                "ddm_factor": v.ddm_factor, "samples": v.samples,
                "hit_rate": v.hit_rate if np.isfinite(v.hit_rate) else None,
                "mean_error": v.mean_error}
            for k, v in calib.items()
        },
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return p


def load_calibration(path: str | None = None) -> dict:
    p = path or CALIB_PATH
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def apply_calibration(style: str, market: str, value: float, calib: dict | None = None,
                      method: str = "dcf") -> float:
    """应用校准系数到内在价值。calib 为 load_calibration 的 dict 或 None(自动加载)。"""
    if value is None or not np.isfinite(value):
        return value
    c = calib if calib is not None else load_calibration()
    styles = c.get("styles", {}) if isinstance(c, dict) else {}
    key = f"{market}:{style}"
    entry = styles.get(key)
    if not entry:
        return float(value)
    factor = entry.get("ddm_factor" if method == "ddm" else "factor", 1.0)
    if not factor:
        return float(value)
    return float(value * factor)
