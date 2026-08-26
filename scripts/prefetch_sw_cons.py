# -*- coding: utf-8 -*-
"""
预取常见申万三级行业成分到本地缓存（data/cache/sw_cons/{code}.json）。

目的：申万成分接口有频率限制，全量遍历易触发限流。本脚本以慢速 + 间隔
逐个预取「常见行业」成分（从校准池 93 只 A 股反推 + 高频行业），成功后
写入本地缓存，运行时直接读缓存，避免在 Streamlit 上实时拉取触发限流。

用法：
    python scripts/prefetch_sw_cons.py          # 全量预取（含 93 只样本行业 + 高频行业）
    python scripts/prefetch_sw_cons.py 851251   # 只预取指定行业代码

注意：如遇接口限流（No tables found），脚本会跳过并继续；可稍后重跑。
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 高频行业代码（按申万三级）
COMMON_CODES = [
    "851251",  # 白酒Ⅲ
    "857371",  # 锂电池
    "857352",  # 光伏电池组件
    "852212",  # 电信运营商
    "851941",  # 保险Ⅲ
    "850816",  # 集成电路制造
    "850813",  # 半导体材料
    "850818",  # 半导体设备
    "857821",  # 国有大型银行Ⅲ
    "857831",  # 股份制银行Ⅲ
    "851931",  # 证券Ⅲ
    "859511",  # 动力煤
    "850552",  # 铜
    "851024",  # 通信网络设备及器件
    "850951",  # 电动乘用车
    "859631",  # 炼油化工
    "850935",  # 航海装备Ⅲ
    "857334",  # 火电设备
    "850771",  # 工程机械整机
    "851122",  # 其他黑色家电
    "851232",  # 啤酒
    "851812",  # 商业地产
    "851761",  # 航运
    "850325",  # 煤化工
]


def main() -> None:
    from src.models.comps import (_load_sw_level3_info, _sw_cons,
                                  _cninfo_sw_name, _cninfo_giant_name, _match_sw_code)
    _load_sw_level3_info(force=True)
    codes: list[str] = []
    if len(sys.argv) > 1:
        codes = [c.strip() for c in sys.argv[1:] if c.strip()]
    else:
        codes = list(COMMON_CODES)
        # 补充：校准池 93 只样本反推的行业
        pool = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "cache")
        for fn in sorted(os.listdir(pool)):
            if not fn.startswith("hist_A_") or not fn.endswith(".pkl"):
                continue
            sym = fn.replace("hist_A_", "").replace(".pkl", "")
            try:
                import pickle
                from src.calibration.backtest import _build_cd
                hd = pickle.load(open(os.path.join(pool, fn), "rb"))
                cd = _build_cd(hd.annual, 1.0, 1.0, 0.02, 0.06, "A", sym, hd.name or sym)
                name = _cninfo_sw_name(sym) or _cninfo_giant_name(sym)
                code = _match_sw_code(name) if name else None
                if code and code not in codes:
                    codes.append(code)
                time.sleep(0.3)
            except Exception:
                continue
    print(f"待预取行业: {len(codes)} 个")
    ok = fail = 0
    for code in codes:
        df = _sw_cons(code)
        if df is not None and len(df):
            ok += 1
            print(f"  OK {code}: {len(df)} 只成分（已缓存）")
        else:
            fail += 1
            print(f"  FAIL {code}: 获取失败（可能限流，跳过）")
        time.sleep(1.0)  # 慢速，避免触发限流
    print(f"完成: 成功 {ok}, 失败 {fail}")


if __name__ == "__main__":
    main()
