#!/usr/bin/env python3
"""第 36 章示例 01：复现官方技术报告里的「算存比」表。

来源：FlashMLA 昇腾稀疏 attention 技术报告（中文版）
``docs/20260930-ascend-prefill-deep-dive-zh.md:85-111``。

报告用它回答「每个 AI Core 负责多少个 q head 合适」：
把 Cube 的矩阵乘周期数与 L2 的搬运周期数都算出来，谁大谁就是瓶颈。

本脚本**只做算术**，不接触任何硬件，因此可以在任何装了 Python 3 的机器上运行。
运行 ``--assert-report`` 会把输出与报告里的表格逐行对照（忽略行首空白）。

验证级别：**本机运行过**（macOS + Python 3.11）。输出与报告逐行一致。
"""

from __future__ import annotations

import argparse
import sys

# 报告脚本里用到的常量（报告 :85-94）。这些数字是**报告给的**，不是本机实测。
D_QK = 512            # 报告用 d 表示，d_qk = d_v = 512
L2_TOTAL_BW = 5e12    # L2 总带宽 5 TB/s
NUM_CORES = 32        # 每 32 个 AI Core 分一份（对应 config.h 的 GRID_SHAPE = 32）
FREQ = 1650e6         # 主频 1650 MHz
FMA_PER_CYCLE = 4096  # Cube 打满时的 4096 FMA/cycle/AICore

HEAD_CANDIDATES = [16, 32, 64, 128]
TOPK_CANDIDATES = [512, 1024]

# 报告 :104-111 的原始输出（行首空白会被 Markdown 修剪，比较时统一 strip）。
REPORT_TABLE = [
    "16, topk= 512, cube_cycles=  2048, mem_cycles=  5883",
    " 16, topk=1024, cube_cycles=  4096, mem_cycles= 11419",
    " 32, topk= 512, cube_cycles=  4096, mem_cycles=  6229",
    " 32, topk=1024, cube_cycles=  8192, mem_cycles= 11765",
    " 64, topk= 512, cube_cycles=  8192, mem_cycles=  6921",
    " 64, topk=1024, cube_cycles= 16384, mem_cycles= 12457",
    "128, topk= 512, cube_cycles= 16384, mem_cycles=  8305",
    "128, topk=1024, cube_cycles= 32768, mem_cycles= 13841",
]


def l2_bytes_per_core_per_cycle() -> float:
    """每个 AI Core 每个周期能从 L2 拿到的字节数（报告 :94）。"""
    return L2_TOTAL_BW / NUM_CORES / FREQ


def _cube_cycles(h_q: int, topk: int) -> int:
    """QKᵀ 与 PV 两个矩阵乘加起来的 MAC 数，除以 4096 FMA/cycle。

    两个 GEMM 各是 ``h_q * topk * d`` 次 MAC，d = d_qk = d_v = 512，
    所以是 ``h_q * topk * (d + d)``；报告里写成 ``(d+d) // 4096``。
    """
    return h_q * topk * (D_QK + D_QK) // FMA_PER_CYCLE


def _mem_cycles(topk: int, h_q: int) -> float:
    """Q/O 与 gathered KV 的字节数，除以每核每周期 L2 带宽。"""
    qo_mem_vol = h_q * D_QK * 2 * 2     # Q 和 O 各 h_q*d 个 bf16（2 字节）
    kv_mem_vol = topk * D_QK * 2        # 本块读入的 gathered_kv 是 topk*d 个 bf16
    return (qo_mem_vol + kv_mem_vol) / l2_bytes_per_core_per_cycle()


def render_table() -> list[str]:
    """生成与报告 :100 那行 ``print`` 完全一致的输出。"""
    lines = []
    for h_q in HEAD_CANDIDATES:
        for topk in TOPK_CANDIDATES:
            # 与报告 :90-100 一样，把局部变量命名成 cube_cycles / mem_cycles，
            # 这样 f-string 的 `{cube_cycles=:6d}` 才会打印出变量名 + 值。
            cube_cycles = _cube_cycles(h_q, topk)
            mem_cycles = _mem_cycles(topk, h_q)
            lines.append(f"{h_q:3d}, {topk=:4d}, {cube_cycles=:6d}, {mem_cycles=:6.0f}")
    return lines


def classify(h_q: int, topk: int) -> str:
    """报告 :114 的判据：谁大谁就是瓶颈。"""
    cc = _cube_cycles(h_q, topk)
    mc = _mem_cycles(topk, h_q)
    if cc > mc:
        return "compute-bound"
    if cc < mc:
        return "memory-bound"
    return "balanced"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assert-report",
        action="store_true",
        help="把输出与报告 :104-111 的表格逐行对照，不一致就返回非零退出码",
    )
    args = parser.parse_args()

    print(f"# L2 bandwidth per AI Core per cycle: {l2_bytes_per_core_per_cycle():.3f} B/cycle")
    lines = render_table()
    for line in lines:
        print(line)

    print()
    print("# 判据（报告 :114）：cube_cycles vs mem_cycles")
    for h_q in HEAD_CANDIDATES:
        for topk in TOPK_CANDIDATES:
            print(
                f"  h_q={h_q:3d}, topk={topk:4d}: "
                f"cube={_cube_cycles(h_q, topk):6d}, "
                f"mem={_mem_cycles(topk, h_q):8.1f} -> {classify(h_q, topk)}"
            )

    if args.assert_report:
        got = [line.strip() for line in lines]
        want = [line.strip() for line in REPORT_TABLE]
        if got != want:
            print("\nMISMATCH with the report table:", file=sys.stderr)
            for g, w in zip(got, want):
                marker = " " if g == w else "!"
                print(f"  {marker} got {g!r}\n    want {w!r}", file=sys.stderr)
            return 1
        print("\nOK: output matches the report table (14-deepseek-ascend-stack/raw/"
              "FlashMLA/docs/20260930-ascend-prefill-deep-dive-zh.md:104-111)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
