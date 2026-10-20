#!/usr/bin/env python3
"""第 14 章第 7 节：性能对齐的「口径换算器」（纯标准库，不需要 GPU/NPU）。

它不测量任何真实性能，只做三件事：

1. 按「目标毫秒数 → 迭代次数」的规则，把 warmup / rep 换算成迭代数
   （TileLang `do_bench` 默认 warmup=25ms、rep=100ms；cuBLAS 对比脚本用
   `minimal_repeat_ms=100` 且至少 5 次）；
2. 打印一份「对比前必须填写的口径表」；
3. 提供一个 `tflops(m, n, k, latency_ms)` 的换算，确保大家用同一个 2MNK 口径。

用法::

    python3 bench_methodology.py
    python3 bench_methodology.py --per-iter 0.35 --m 4096 --n 4096 --k 4096
"""

from __future__ import annotations

import argparse
import math


def plan(warmup_ms: float = 25.0, rep_ms: float = 100.0, per_iter_ms: float = 0.1,
         min_repeat: int = 5) -> dict:
    """按目标毫秒数换算预热/重复次数。

    与 cuBLAS 对比脚本（corpus 快照）和 TileLang `do_bench` 的口径一致：
    先估出单次耗时，再按目标毫秒数向上取整得到迭代数；重复次数设下限，
    避免极短的 kernel 只跑一两次。
    """
    if per_iter_ms <= 0:
        raise ValueError("per_iter_ms 必须为正")
    n_warmup = max(1, math.ceil(warmup_ms / per_iter_ms))
    n_repeat = max(min_repeat, math.ceil(rep_ms / per_iter_ms))
    return {
        "n_warmup": n_warmup,
        "n_repeat": n_repeat,
        "estimated_total_ms": (n_warmup + n_repeat) * per_iter_ms,
    }


def tflops(m: int, n: int, k: int, latency_ms: float) -> float:
    """统一按 2MNK 计算 TFLOP/s，避免「是否含 epilogue」之外的算法口径分歧。"""
    if latency_ms <= 0:
        raise ValueError("latency_ms 必须为正")
    return 2.0 * m * n * k / (latency_ms / 1e3) / 1e12


CHECKLIST = """\
对比前必须固定的口径（缺一条，数字就不可比）：
  1. 基线是谁            cuBLAS / cublasLt / CATLASS 模板 / Ascend C 手写 / Torch NPU
  2. shape               M, N, K；batch；是否转置；ROW/COLUMN major 还是 NZ
  3. dtype               输入 dtype、累加 dtype、输出 dtype；fp32 是否允许 TF32
  4. epilogue            是否含 bias / 激活 / 量化 / 类型转换；含与不含分开报
  5. 计时范围            kernel-only（cupti / cudagraph）还是含 host 启动
  6. 预热与重复          warmup 与 rep 的目标毫秒数，或固定迭代数
  7. 取数方式            mean / median / min；是否 flush L2
  8. 算法口径            FLOPs 一律按 2MNK；memory-bound 的再报有效 GB/s
  9. 编译期参数          block_M/N/K, num_stages, threads, policy, split_k
 10. 正确性              与同一 golden 比，写出 rtol / atol
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="性能对比口径换算器")
    parser.add_argument("--warmup-ms", type=float, default=25.0)
    parser.add_argument("--rep-ms", type=float, default=100.0)
    parser.add_argument("--per-iter", type=float, default=None,
                        help="单次 kernel 的估计耗时（ms）；不给则打印一张扫描表")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    args = parser.parse_args()

    print("=" * 72)
    print(f"目标口径：warmup={args.warmup_ms} ms, rep={args.rep_ms} ms")
    print("=" * 72)
    print()

    if args.per_iter is not None:
        rows = [args.per_iter]
    else:
        rows = [0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

    print(f"{'单次耗时(ms)':>12} {'预热次数':>10} {'重复次数':>10} {'总耗时(ms)':>12}")
    print("-" * 48)
    for per_iter in rows:
        p = plan(args.warmup_ms, args.rep_ms, per_iter)
        print(f"{per_iter:>12.4g} {p['n_warmup']:>10d} {p['n_repeat']:>10d} "
              f"{p['estimated_total_ms']:>12.1f}")

    print()
    print(f"换算示例：shape {args.m}x{args.n}x{args.k}，"
          f"若单次耗时 1.0 ms → {tflops(args.m, args.n, args.k, 1.0):.2f} TFLOP/s")
    print()
    print(CHECKLIST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
