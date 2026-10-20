#!/usr/bin/env python3
"""第 36 章示例 02：复现 1C1V 的三条 bound 里的算术。

来源：FlashMLA 昇腾稀疏 attention 技术报告（中文版）
``docs/20260930-ascend-prefill-deep-dive-zh.md:132-144``（1C1V 的三条 bound）
与 ``:126-128``（HF32 关闭后的 8192 周期）。

三条 bound 里，只有 FixPipe 那条能用一个显式不等式解出阈值
``B_TOPK > 73.14``；本脚本把它解出来、顺便把 MTE2 队列与 HF32 的算术也复算一遍。

验证级别：**本机运行过**（macOS + Python 3.11）。输出与报告一致。
"""

from __future__ import annotations

import argparse
from fractions import Fraction

MMA_M = 64            # 每个 AI Core 处理的 q head 数（报告 :142）
D_QK = 512            # d_qk = d_v = 512
FMA_PER_CYCLE = 4096  # Cube 打满（HF32 / bf16）
FIXPIPE_BYTES_PER_CYCLE = 128   # 1C1V 时 FixPipe 的搬运速度（报告 :142）

MTE2_QUEUE_DEPTH = 16    # 单个 VECTOR Core 未完成的 MTE2 请求上限（报告 :138）
GATHER2_TOKENS = 2       # gather2 一次拷两个 token（报告 :216）
B_TOPK = 64              # config.h:55

HF32_OFF_FMA_PER_CYCLE = 256   # 关闭 HF32 后 Cube 的算力（报告 :128）


def mma_cycles_per_block(b_topk: int) -> int:
    """报告 :142：每个块的 MMA 周期 = MMA_M * B_TOPK * (512+512) / 4096。

    化简后正好是 ``B_TOPK * 16``。
    """
    return MMA_M * b_topk * (D_QK + D_QK) // FMA_PER_CYCLE


def fixpipe_cycles_per_block(b_topk: int) -> int:
    """报告 :142：FixPipe 周期 = (MMA_M*512 + MMA_M*B_TOPK) * sizeof(float) / 128。

    化简后正好是 ``1024 + B_TOPK * 2``。
    """
    floats = MMA_M * D_QK + MMA_M * b_topk  # out 的 64x512 + P 的 64xB_TOPK
    return floats * 4 // FIXPIPE_BYTES_PER_CYCLE


def exact_threshold() -> Fraction:
    """解 ``B_TOPK * 16 > 1024 + B_TOPK * 2``。

    14 * B_TOPK > 1024  =>  B_TOPK > 1024/14 = 512/7。
    """
    return Fraction(1024, 14)


def max_tokens_per_round() -> int:
    """MTE2 队列深度 * gather2 的每请求 token 数（报告 :138）。"""
    return MTE2_QUEUE_DEPTH * GATHER2_TOKENS


def hf32_off_cycles() -> int:
    """报告 :128：64x64x512 的 float 矩阵乘，在 256 FMA/cycle 下需要多少周期。"""
    return 64 * 64 * 512 // HF32_OFF_FMA_PER_CYCLE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assert-report",
        action="store_true",
        help="校验关键数字与报告一致；不一致返回非零退出码",
    )
    args = parser.parse_args()

    print("== bound ③：FixPipe（报告 :142） ==")
    print(f"  MMA 周期/块   = MMA_M*B_TOPK*(512+512)/4096 = B_TOPK*16")
    print(f"  FixPipe 周期/块 = (MMA_M*512+MMA_M*B_TOPK)*4/128 = 1024 + B_TOPK*2")
    thr = exact_threshold()
    print(f"  条件 B_TOPK*16 > 1024 + B_TOPK*2  =>  B_TOPK > 1024/14 = {thr} "
          f"≈ {float(thr):.2f}")
    for b in (64, 96, 128):
        mma = mma_cycles_per_block(b)
        fix = fixpipe_cycles_per_block(b)
        verdict = "FixPipe 更慢（瓶颈）" if fix > mma else "MMA 更慢（FixPipe 不是瓶颈）"
        print(f"    B_TOPK={b:3d}: MMA={mma:6d}, FixPipe={fix:6d} -> {verdict}")

    print()
    print("== bound ①：MTE2 Issue Queue（报告 :138） ==")
    print(f"  单 VECTOR Core 未完成 MTE2 请求上限 = {MTE2_QUEUE_DEPTH}")
    print(f"  用了 gather2（burst_count=2）后，一轮最多拷 "
          f"{MTE2_QUEUE_DEPTH} * {GATHER2_TOKENS} = {max_tokens_per_round()} 个 token")
    print(f"  而 B_TOPK = {B_TOPK}（config.h:55），所以 1C1V 下拷不完一整块")
    print(f"  1C2V 后每个 VECTOR Core 只要拷 B_TOPK_PER_V = {B_TOPK // 2} 个 token")

    print()
    print("== accumulator 回 L0C 的代价（报告 :126-128） ==")
    print(f"  64x64x512 的 float 矩阵乘 = {64 * 64 * 512} MAC")
    print(f"  关 HF32 后 Cube = {HF32_OFF_FMA_PER_CYCLE} FMA/cycle")
    print(f"  => {64 * 64 * 512} / {HF32_OFF_FMA_PER_CYCLE} = {hf32_off_cycles()} 周期")
    print("  （报告 :128 的数字是 8192，一致）")

    if args.assert_report:
        problems = []
        if float(thr) != 73.14285714285714:
            problems.append(f"threshold {float(thr)} != 73.14...")
        if mma_cycles_per_block(B_TOPK) != 1024:
            problems.append("MMA cycles for B_TOPK=64 should be 1024")
        if fixpipe_cycles_per_block(B_TOPK) != 1152:
            problems.append("FixPipe cycles for B_TOPK=64 should be 1152")
        if max_tokens_per_round() != 32:
            problems.append("MTE2 round capacity should be 32 tokens")
        if hf32_off_cycles() != 8192:
            problems.append("HF32-off cycles should be 8192")
        if problems:
            print("\nMISMATCH:", file=__import__("sys").stderr)
            for p in problems:
                print(f"  - {p}", file=__import__("sys").stderr)
            return 1
        print("\nOK: 关键数字与报告一致（:138, :142, :126-128）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
