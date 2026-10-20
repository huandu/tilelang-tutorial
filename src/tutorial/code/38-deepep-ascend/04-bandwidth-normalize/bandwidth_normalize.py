#!/usr/bin/env python3
"""第 38 章 · 示例 04：把 README 性能表换算成「相对物理上限」。

数据只有一处来源：``DeepEP-Ascend/README.md:13-19`` 的性能表。
本脚本做两件事：

1. 原样打印表格（区间上下限），并算出 EP 之间的相对下降；
2. 把 dispatch 的数字除以 0.90 与 0.95，反推 README 那句话
   （``README.md:21``：「Sustained dispatch reaches roughly 90–95% of the
   physical payload bandwidth limit for EP sizes up to 32」）
   隐含的物理上限。

作者验证级别：**CPU 逻辑验证**。纯标准库。反推结果是**区间估计**，
不是硬件规格、也不是实测——理由见脚本末尾与正文 §7.4。
"""

from __future__ import annotations

import argparse

# 原文照抄（DeepEP-Ascend/README.md:15-19）。key = EP size, value = (lo, hi)。
DISPATCH = {8: (373, 375), 16: (348, 352), 32: (335, 340), 64: (323, 327), 128: (313, 320)}
COMBINE = {8: (345, 347), 16: (338, 341), 32: (320, 324), 64: (294, 298), 128: (272, 278)}


def main() -> int:
    ap = argparse.ArgumentParser(description="把 README 带宽表换算成相对上限")
    ap.add_argument("--low", type=float, default=0.90, help="可持续比例下界")
    ap.add_argument("--high", type=float, default=0.95, help="可持续比例上界")
    args = ap.parse_args()

    print("原始表（DeepEP-Ascend/README.md:13-19，单位 GB/s，区间覆盖所有 rank）")
    print(f"{'EP':>4} {'Dispatch':>14} {'Combine':>14} {'combine/dispatch':>18}")
    for ep in (8, 16, 32, 64, 128):
        d_lo, d_hi = DISPATCH[ep]
        c_lo, c_hi = COMBINE[ep]
        ratio = ((c_lo + c_hi) / 2) / ((d_lo + d_hi) / 2)
        print(f"{ep:>4} {d_lo:>6}-{d_hi:<6} {c_lo:>6}-{c_hi:<6} {ratio:>17.1%}")
    print()

    print(f"相对 EP8 的下降（取区间中点）")
    base = sum(DISPATCH[8]) / 2
    for ep in (16, 32, 64, 128):
        mid = sum(DISPATCH[ep]) / 2
        print(f"  EP{ep:<4} dispatch 比 EP8 低 {1 - mid / base:>5.1%}")
    print()

    print(f"反推物理 payload 带宽上限（dispatch 数字 ÷ {args.low:.2f}…{args.high:.2f}）")
    lo_all, hi_all = float("inf"), 0.0
    for ep in (8, 16, 32):
        d_lo, d_hi = DISPATCH[ep]
        lo = d_lo / args.high
        hi = d_hi / args.low
        lo_all, hi_all = min(lo_all, lo), max(hi_all, hi)
        print(f"  EP{ep:<4} {d_lo}-{d_hi} GB/s  →  上限 {lo:.0f}–{hi:.0f} GB/s")
    print(f"  三者并集 ≈ {lo_all:.0f}–{hi_all:.0f} GB/s/rank")
    print()

    print("口径提醒")
    print("  1) 表里的 GB/s = 有效 payload 字节 / 时间；payload 口径见")
    print("     tests/ep/test_ep.py:404-406（不含 padding、不含 barrier）。")
    print("  2) 计时含 issue 与 drain，不含最后的 epilogue（README.md:21）。")
    print("  3) README 原文是 'roughly 90–95%'，所以这只是**区间估计**；")
    print("     corpus 里没有 Ascend 950 UBMEM 链路的官方带宽规格，")
    print("     不能当成硬件手册上的数字。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
