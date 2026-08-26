# -*- coding: utf-8 -*-
"""面板集成辅助：读取校准表并在 app 中应用/展示。"""
from __future__ import annotations

import numpy as np

from .calibrate import load_calibration


def calib_context(style: str, market: str) -> dict:
    """读取当前 风格×市场 的校准上下文。无数据时返回空上下文（factor=1）。"""
    calib = load_calibration()
    styles = calib.get("styles", {}) if isinstance(calib, dict) else {}
    key = f"{market}:{style}"
    entry = styles.get(key) or {}
    has_data = bool(entry and entry.get("samples", 0) >= 3)
    return {
        "key": key,
        "factor": float(entry.get("factor", 1.0)) if entry else 1.0,
        "ddm_factor": float(entry.get("ddm_factor", 1.0)) if entry else 1.0,
        "samples": int(entry.get("samples", 0)) if entry else 0,
        "hit_rate": entry.get("hit_rate"),
        "mean_error": entry.get("mean_error"),
        "updated": calib.get("updated", "") if isinstance(calib, dict) else "",
        "has_data": has_data,
    }


def apply_calib(value, method: str, ctx: dict | None = None) -> float:
    """应用校准系数。ctx 来自 calib_context()；method: 'dcf'/'ddm'。"""
    if value is None or not isinstance(value, (int, float)) or not np.isfinite(value):
        return value
    ctx = ctx or {}
    if not ctx.get("has_data"):
        return float(value)
    factor = ctx.get("ddm_factor" if method == "ddm" else "factor", 1.0)
    if not factor:
        return float(value)
    return float(value * factor)
