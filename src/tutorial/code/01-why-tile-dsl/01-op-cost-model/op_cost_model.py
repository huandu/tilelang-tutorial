#!/usr/bin/env python3
"""算术强度 / 机器平衡点估算器（第 01 章 E9 的参考实现）。

为什么需要这个脚本
------------------
第 01 章说「一个算子慢，通常不是慢在数学运算上」，并用两个量来量化这件事：

    算术强度 AI        = 浮点运算次数 / 访问的字节数        （FLOP/Byte）
    机器平衡点 balance = 峰值算力 / 峰值带宽                （FLOP/Byte）

    AI < balance  ->  memory-bound（先被带宽卡住）
    AI > balance  ->  compute-bound（先被算力卡住）

本脚本用**纯标准库**实现这三个公式，用「理论最小访存量」估算几个常见算子，
并打印一张对照表。它不做任何实测，也不测量你的机器；它只是把公式算清楚。

机器参数从命令行给，默认值取自本章引用的公开资料（见章节 references.md 的 R7）：

    H100 SXM5：HBM3 带宽「over 3 TB/sec」，FP16 张量核密集吞吐 1000 TFLOPS
               （厂商 preliminary specs，未实测）

**警告**：这里的 balance 是一个数量级尺度的判据，不是精确预测。真实 kernel 能拿到
峰值算力的哪一部分，取决于实现；本脚本不对此做任何承诺。

运行
----
    python3 op_cost_model.py
    python3 op_cost_model.py --m 512 --n 512 --k 512
    python3 op_cost_model.py --peak-tflops 100 --bandwidth-tbps 1.5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 机器模型
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Machine:
    """一台机器的两个标量参数。"""

    name: str
    peak_flops: float       # 峰值浮点运算数/秒（FLOP/s）
    bandwidth: float        # 峰值内存带宽（Byte/s）

    @property
    def balance(self) -> float:
        """机器平衡点，单位 FLOP/Byte。"""
        return self.peak_flops / self.bandwidth


# 默认参数：H100 SXM5，来自本章 references.md 的 R7（厂商公开资料，未实测）。
DEFAULT_PEAK_TFLOPS = 1000.0     # 1000 TFLOPS = 1e15 FLOP/s
DEFAULT_BANDWIDTH_TBPS = 3.0     # 「over 3 TB/sec」取下界 3.0 TB/s


# ---------------------------------------------------------------------------
# 算子模型：只算「理论最小访存量」与「核心运算量」
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Cost:
    """一次算子的成本模型。"""

    name: str
    flops: float            # 浮点运算次数
    bytes_moved: float      # 理论最小访存字节数

    @property
    def intensity(self) -> float:
        """算术强度，FLOP/Byte。"""
        return self.flops / self.bytes_moved


def gemm_cost(m: int, n: int, k: int, elem_bytes: int = 2) -> Cost:
    """C[M, N] = A[M, K] @ B[K, N] 的理论最小成本。

    运算量：每个 C 元素做 K 次乘加，即 2 * M * N * K FLOP。
    最小访存：A、B、C 各被完整读写一遍。

    注意：这是**分块之后**才能接近的访存量。不分块的朴素实现要差几个数量级，
    见习题 E11。
    """
    flops = 2.0 * m * n * k
    bytes_moved = float((m * k + k * n + m * n) * elem_bytes)
    return Cost(f"MatMul({m}x{n}x{k})", flops, bytes_moved)


def softmax_cost(m: int, n: int, elem_bytes: int = 2) -> Cost:
    """y = softmax(x, dim=-1) 的理论最小成本。

    运算量：每个元素约 3 次运算（减最大值、指数、除法/累加），沿行归约的常数
    在这里按每元素 3 FLOP 粗估——**它本来也不是这个算子的瓶颈**。
    最小访存：读 x 一遍、写 y 一遍（在线版本可以做到）。
    """
    flops = 3.0 * m * n
    bytes_moved = 2.0 * m * n * elem_bytes
    return Cost(f"Softmax({m}x{n})", flops, bytes_moved)


def rmsnorm_cost(m: int, n: int, elem_bytes: int = 2) -> Cost:
    """y = x * rsqrt(mean(x^2) + eps) 的理论最小成本。

    运算量：每元素平方 + 累加 + 缩放，约 4 FLOP。
    最小访存：理想情况下读 x 一遍、写 y 一遍；但真实实现往往需要两遍
    （先求 mean(x^2)，再归一化），除非把一行留在片上。这里仍然按下界计算，
    所以它是这三个算子里算术强度最低的。
    """
    flops = 4.0 * m * n
    bytes_moved = 2.0 * m * n * elem_bytes
    return Cost(f"RMSNorm({m}x{n})", flops, bytes_moved)


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------

def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.3g} {unit}"
        n /= 1024.0
    return f"{n:.3g} PiB"


def human_flops(n: float) -> str:
    for unit in ("FLOP", "KFLOP", "MFLOP", "GFLOP", "TFLOP", "PFLOP"):
        if abs(n) < 1000.0:
            return f"{n:.3g} {unit}"
        n /= 1000.0
    return f"{n:.3g} EFLOP"


def classify(cost: Cost, machine: Machine) -> str:
    if cost.intensity < machine.balance:
        return "memory-bound"
    return "compute-bound"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--m", type=int, default=4096, help="矩阵 M（默认 4096）")
    parser.add_argument("--n", type=int, default=4096, help="矩阵 N（默认 4096）")
    parser.add_argument("--k", type=int, default=4096, help="矩阵 K（默认 4096）")
    parser.add_argument(
        "--peak-tflops", type=float, default=DEFAULT_PEAK_TFLOPS,
        help=f"峰值算力（TFLOPS，默认 {DEFAULT_PEAK_TFLOPS}）",
    )
    parser.add_argument(
        "--bandwidth-tbps", type=float, default=DEFAULT_BANDWIDTH_TBPS,
        help=f"峰值带宽（TB/s，默认 {DEFAULT_BANDWIDTH_TBPS}）",
    )
    args = parser.parse_args()

    machine = Machine(
        name="user-specified",
        peak_flops=args.peak_tflops * 1e12,
        bandwidth=args.bandwidth_tbps * 1e12,
    )

    costs = [
        gemm_cost(args.m, args.n, args.k),
        softmax_cost(args.m, args.n),
        rmsnorm_cost(args.m, args.n),
    ]

    print("=" * 78)
    print("机器模型")
    print("-" * 78)
    print(f"  峰值算力        : {args.peak_tflops:.6g} TFLOPS")
    print(f"  峰值带宽        : {args.bandwidth_tbps:.6g} TB/s")
    print(f"  机器平衡点      : {machine.balance:.6g} FLOP/Byte")
    print("=" * 78)
    print("算子成本模型（理论最小访存量，fp16 输入输出）")
    print("-" * 78)
    header = f"{'算子':<24}{'运算量':>16}{'最小访存':>14}{'算术强度':>16}{'判定':>18}"
    print(header)
    for cost in costs:
        print(
            f"{cost.name:<24}"
            f"{human_flops(cost.flops):>16}"
            f"{human_bytes(cost.bytes_moved):>14}"
            f"{cost.intensity:>14.6g} F/B"
            f"{classify(cost, machine):>18}"
        )
    print("=" * 78)
    print("注：按本模型，立方 GEMM 的算术强度约为 2M/(3*elem_bytes)，随边长线性增长；")
    print("    矩阵越小越容易落到 memory-bound 一侧。可直接用 --m/--n/--k 验证。")
    print("    本脚本不做实测，结论只用于判断「先撞带宽还是先撞算力」。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
