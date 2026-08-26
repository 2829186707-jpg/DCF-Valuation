# -*- coding: utf-8 -*-
"""
批量回测运行器：带本地缓存、请求限速、断点续跑与数据源健康检测。

目的：外部财经数据源（新浪/东财/yahoo）常间歇反爬，本运行器保证：
  - 每股数据只拉一次并缓存到本地，重复运行不重复请求
  - 请求间隔限速，降低触发反爬概率
  - 单股失败自动跳过并记录，下次运行续跑未完成的股票
  - 连续失败超过阈值时暂停等待（给反爬冷却时间）
"""
from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import HistoryData, load_history_data, run_backtest_batch
from .calibrate import compute_calibration, save_calibration, load_calibration

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "data", "cache")


def _cache_path(market: str, symbol: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"hist_{market}_{symbol}.pkl")


def load_history_cached(symbol: str, market: str | None = None) -> HistoryData:
    """带缓存的行情+财务加载。"""
    m = market
    from ..data_fetcher import detect_market
    m = m or detect_market(symbol)
    path = _cache_path(m, symbol.upper() if m == "US" else symbol)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    hd = load_history_data(symbol, m)
    try:
        with open(path, "wb") as f:
            pickle.dump(hd, f)
    except Exception:
        pass
    return hd


@dataclass
class RunReport:
    pool_size: int
    ok: int
    failed: int
    samples: int
    calib_path: str
    failures: list[tuple[str, str]]  # (symbol, error_type)


def _samples_path(market: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"backtest_samples_{market}.csv")


def _load_samples(market: str) -> list:
    """读取历史积累的回测样本（断点续跑 / 滚动自学习复用）。"""
    from .backtest import BacktestSample
    path = _samples_path(market)
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
        out = []
        for _, r in df.iterrows():
            s = BacktestSample(**{k: (np.nan if pd.isna(v) else v) for k, v in r.to_dict().items()})
            out.append(s)
        return out
    except Exception:
        return []


def _save_samples(market: str, samples: list) -> None:
    path = _samples_path(market)
    df = pd.DataFrame([vars(s) for s in samples])
    df.to_csv(path, index=False)


def run_pool_backtest(
    pool: list[str],
    market: str,
    years: list[int] | None = None,
    delay: float = 1.5,
    max_consec_fail: int = 5,
    pause_sec: int = 90,
    calib_path: str | None = None,
    method: str = "dcf",
    merge_history: bool = True,
) -> RunReport:
    """对股票池批量回测，产出校准系数表。失败自动跳过，连续失败暂停。
    merge_history=True 时并入历史积累样本（滚动自学习），避免重复拉数据。
    """
    import numpy as np
    from .pool import build_stock_pool
    if isinstance(pool, str):
        pool = build_stock_pool(pool, limit=200)

    samples = list(_load_samples(market)) if merge_history else []
    _init_n = len(samples)
    failures: list[tuple[str, str]] = []
    consec = 0

    for i, sym in enumerate(pool):
        try:
            hd = load_history_cached(sym, market)
            time.sleep(delay)  # 限速
            batch = run_backtest_batch(hd, years=years, rf_series=hd.rf_series)
            if batch:
                samples.extend(batch)
                consec = 0
                print(f"  [{i+1}/{len(pool)}] {sym}: {len(batch)} 样本 OK（累计 {len(samples)}）")
            else:
                raise RuntimeError("无有效回测样本")
        except Exception as e:
            et = type(e).__name__
            failures.append((sym, et))
            consec += 1
            print(f"  [{i+1}/{len(pool)}] {sym}: 失败 {et}")
            if consec >= max_consec_fail:
                print(f"  连续 {max_consec_fail} 次失败，暂停 {pause_sec}s 冷却…")
                time.sleep(pause_sec)
                consec = 0

    # 保存合并后的样本（含历史，供下次滚动更新）
    _save_samples(market, samples)
    calib = compute_calibration(samples, method=method)
    p = save_calibration(calib, calib_path)
    print(f"\n完成: 本次新增 {len(samples)-_init_n} 样本，累计 {len(samples)}，"
          f"失败 {len(failures)}，校准表={p}")
    return RunReport(pool_size=len(pool), ok=len(pool) - len(failures),
                     failed=len(failures), samples=len(samples),
                     calib_path=p, failures=failures)


if __name__ == "__main__":
    import sys
    market = sys.argv[1] if len(sys.argv) > 1 else "A"
    from .pool import build_stock_pool
    pool = build_stock_pool(market, limit=100)
    print(f"股票池({market}): {len(pool)} 只")
    rep = run_pool_backtest(pool, market)
    print("失败清单:", rep.failures[:20])
