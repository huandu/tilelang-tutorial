#!/usr/bin/env python3
"""Chapter 39, §3.4/§3.7: classify a shape by computing two ratios.

The classification rule is the one from chapter 28 §2.2:

    ratio_compute = achieved_TFLOPS / peak_TFLOPS
    ratio_memory  = achieved_GBps   / peak_GBps

Both numerators below are *read off the official tables* in chapter 35 §5
(DeepGEMM-Ascend README:90-170). Both denominators come from the same source
tree: the HWDesc constants in heuristics.hpp:46-71, turned into a peak by
flops_per_cycle * 1650 MHz * 32 cores (the chapter's own recomputation).

Doing this is allowed *only* because numerator and denominator come from the
same machine and the same repository. Chapter 39 §4.4 lists the three
arithmetics that are NOT allowed.

Usage:
    python3 inference_latency_model.py
    python3 inference_latency_model.py --m 128 --n 7168 --k 16384 --dtype fp8
"""

from __future__ import annotations

import argparse

FP8_TFLOPS = 865.0
BF16_TFLOPS = 432.5
FP4_TFLOPS = 1730.1
HBM_GBPS = 3500.0

# ---------------------------------------------------------------------------
# Shapes copied from the official tables (chapter 35 §5).
#   name, source table, compute (TFLOPS), memory bandwidth (GB/s), what官方 says
# ---------------------------------------------------------------------------
SHAPES = [
    ("dense FP8 GEMM  M=4096, N=7168, K=16384",
     "README.md:95-104", 861.0, 186.0, "compute-bound"),
    ("M-grouped FP8xFP4  #G=8, M/group=4096, K=4096",
     "README.md:129-140", 861.0, 180.0, "compute-bound"),
    ("FP8 inference GEMM  M=128, N=7168, K=16384",
     "README.md:108-122", 549.0, 2456.0, "balanced"),
    ("MQA logits prefill  Q=4096, K=8192, heads=32, dim=128",
     "README.md:142-149", 681.0, 283.0, "FIX-pipe bound"),
    ("MQA logits decode   Q=256, K=8192, heads=32, dim=128",
     "README.md:142-149", 565.0, 2006.0, "balanced"),
    ("HC prenorm GEMM  M=8192, N=24, K=28672",
     "README.md:166-176", 82.0, 3463.0, "memory-bound"),
]


def ratios(compute_tflops: float, memory_gbps: float) -> tuple[float, float]:
    return compute_tflops / FP8_TFLOPS, memory_gbps / HBM_GBPS


def classify(ratio_compute: float, ratio_memory: float) -> str:
    """Same three readings the chapter uses in §3.7.

    saturated = 0.85, involved = 0.45. These thresholds are the book's own
    convention for *reading* a table, not a property of the hardware.
    """
    saturated, involved = 0.85, 0.45
    if ratio_compute >= saturated and ratio_memory < 0.20:
        return "算力（贴着 compute roof）"
    if ratio_memory >= saturated and ratio_compute < 0.20:
        return "带宽（贴着 memory roof）"
    if ratio_compute >= involved and ratio_memory >= involved:
        return "两列都过半 —— 处在平衡点附近"
    return "两列都不高 —— 去看第三条流水线（例如 FixPipe）"


def print_table() -> None:
    print("=" * 92)
    print("① 用两个比值给六个形状判类（正文 §3.7）")
    print("=" * 92)
    print("分子：官方 README 的 Computation / Memory BW 两列（同一张表、同一台机器）")
    print(f"分母：本书复算的硬件模型常量 —— fp8 {FP8_TFLOPS:.1f} TFLOPS、"
          f"HBM {HBM_GBPS:.0f} GB/s（heuristics.hpp:46-71）")
    print()
    header = f"{'shape':<52} {'comp':>8} {'mem':>8} {'c/peak':>8} {'m/peak':>8}  {'判类'}"
    print(header)
    print("-" * len(header))
    for name, _source, compute, memory, official in SHAPES:
        rc, rm = ratios(compute, memory)
        verdict = classify(rc, rm)
        print(f"{name:<52} {compute:>8.0f} {memory:>8.0f} "
              f"{rc:>7.1%} {rm:>7.1%}  {verdict}")
    print()
    print("官方对这几行的定性（用于对照，不参与上面的计算）：")
    for name, source, _c, _m, official in SHAPES:
        print(f"  · {name:<52} → {official}  ({source})")
    print()


def print_calibration_check() -> None:
    m, n, k = 4096, 7168, 16384
    flops = 2 * m * n * k
    micros = 1117.6
    achieved = flops / (micros * 1e-6) / 1e12
    print("=" * 92)
    print("② 先用一行能自己复算的表来标定口径（正文 §3.7）")
    print("=" * 92)
    print(f"dense FP8 GEMM M={m}, N={n}, K={k}：")
    print(f"  2*M*N*K = {flops:.3e} FLOP；README 的延迟 {micros} us")
    print(f"  → {achieved:.1f} TFLOPS，与 README 的 Computation 列（861）一致 ✅")
    print()
    print("同样的复算套到 MQA logits（Q=4096, K=8192, heads=32, dim=128）：")
    mqa_flops = 2 * 4096 * 8192 * 32 * 128
    mqa = mqa_flops / (302.6 * 1e-6) / 1e12
    print(f"  2*Q*K*heads*dim = {mqa_flops:.3e} FLOP；README 的延迟 302.6 us")
    print(f"  → {mqa:.0f} TFLOPS，比 README 的 681 高 {mqa / 681 - 1:.0%}")
    print("  README 没有给出这一列的公式 → 只报事实、以表格列为准（第 35 章 §5.3）")
    print()


def print_custom(m: int, n: int, k: int, dtype: str) -> None:
    peak = {"fp8": FP8_TFLOPS, "bf16": BF16_TFLOPS, "fp4": FP4_TFLOPS}[dtype]
    elem_bytes = {"fp8": 1, "bf16": 2, "fp4": 0.5}[dtype]
    flops = 2 * m * n * k
    bytes_moved = (m + n) * k * elem_bytes
    intensity = flops / bytes_moved
    balance = peak * 1e12 / (HBM_GBPS * 1e9)
    print("=" * 92)
    print("③ 你自己的形状")
    print("=" * 92)
    print(f"M={m}, N={n}, K={k}, dtype={dtype}")
    print(f"  flops            {flops:.3e}")
    print(f"  bytes (A+B)      {bytes_moved:.3e}")
    print(f"  算术强度         {intensity:.1f} FLOP/byte")
    print(f"  平衡点（{dtype}）    {balance:.0f} FLOP/byte")
    verdict = "compute-bound" if intensity > balance else "memory-bound"
    print(f"  判类             {verdict}")
    if 0.9 <= intensity / balance <= 1.1:
        print("                   ⚠ 落在平衡点附近：读的时间和算的时间同量级，")
        print("                     优化要同时动两边（第 35 章 §5.3）")
    print()
    print("记住两条边界：")
    print("  · 平衡点用的是模型常量，不是实测峰值；换机器要重标定。")
    print("  · 这只判类，不预测时间（第 28 章 §2.2 的正确用法）。")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--m", type=int)
    parser.add_argument("--n", type=int, default=7168)
    parser.add_argument("--k", type=int, default=16384)
    parser.add_argument("--dtype", choices=("fp8", "bf16", "fp4"), default="fp8")
    args = parser.parse_args()

    print("第 39 章 · 推理侧的判类账本（公式复算，无实测）")
    print()
    print_calibration_check()
    print_table()
    if args.m is not None:
        print_custom(args.m, args.n, args.k, args.dtype)
    else:
        print("（加 --m 4096 可以再算一个你自己的形状。）")


if __name__ == "__main__":
    main()
