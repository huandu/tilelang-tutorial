#!/usr/bin/env python3
"""第 12 章 示例 2：低精度对拍的容差该定多少（CPU 逻辑验证）。

「和 PyTorch 参考实现对拍」最难的一步不是写对拍代码，而是**说清楚容差从哪来**。
`atol=1e-3, rtol=1e-3` 这种写法如果只是抄来的，出了偏差你既不敢放宽、也不知道能收紧。

这个脚本做两件事：

1. 从 IEEE-754 的有效位数算出 fp16 / bf16 / fp32 的机器精度 u，给出一个
   长度 K 的归约（dot / GEMM 的 K 维）的误差预算（最坏情形与均方根两种）。
2. 用固定随机种子做一次真实的正交实验：`fp16/bf16 输入 → fp32 累加` 的
   dot 结果与 fp64 精确值比较，实测最大误差与 99.9 分位误差，和上一步的
   预算对照。

它证明的是「容差的数量级从哪来」，**不是**「某个 kernel 一定通过某个容差」——
后者取决于你的数据分布、归约顺序和累加类型，必须在有卡机器上实测。

用法：

    python3 tolerance_model.py
    python3 tolerance_model.py --lengths 128,512,2048
    python3 tolerance_model.py --trials 4000

依赖：Python 3.9+ 与 numpy（TileLang 自身也依赖 numpy）。

作者实际做到的验证级别：**CPU 逻辑验证**（写作机 macOS / Apple Silicon、
Python 3.11 + numpy 1.24.1 上实际运行过，输出见同目录 README.md）。
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RealFormat:
    """一个二进制浮点格式的有效位与指数范围。"""

    name: str
    significand_bits: int  # 含隐含前导 1
    max_exp: int  # 最大规格化指数（用于溢出判断，未在本脚本中展开）
    np_dtype: type

    @property
    def unit_roundoff(self) -> float:
        """单位舍入误差 u = 2^-p（p = 有效位数）。"""
        return math.ldexp(1.0, -self.significand_bits)

    @property
    def atol_hint(self) -> float:
        """工程上常用的绝对容差起点：u 向上取到一个 10 的幂。"""
        return 10.0 ** math.ceil(math.log10(self.unit_roundoff))


FP16 = RealFormat("fp16", 11, 15, np.float16)
BF16 = RealFormat("bf16", 8, 127, np.float32)  # numpy 无原生 bf16，用截断模拟
FP32 = RealFormat("fp32", 24, 127, np.float32)


def _to_bf16(values: np.ndarray) -> np.ndarray:
    """用「截掉 fp32 低 16 位尾数」近似 bf16 舍入（round-to-nearest-even 的近似）。"""
    as_u32 = values.astype(np.float32).view(np.uint32)
    rounded = (as_u32 + np.uint32(0x8000)) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def _quantize(values: np.ndarray, fmt: RealFormat) -> np.ndarray:
    if fmt.name == "bf16":
        return _to_bf16(values)
    return values.astype(fmt.np_dtype).astype(np.float32)


def error_budget(fmt: RealFormat, length: int) -> tuple[float, float]:
    """返回 (最坏情形, 均方根) 误差预算，**以 S = Σ|a_i·b_i| 为基准**。

    推导只用两条假设：
    1. 每个乘积 a_i·b_i 单独舍入到低精度，带来的误差上界是 u_fmt·|a_i·b_i|；
    2. 各次舍入的误差近似独立同分布，K 项相加按 √K 增长。

    于是分子误差 ≈ u_fmt·(S/√K)，除以 S 后相对预算就是 u_fmt/√K；
    最坏情形（误差同号、完全不抵消）是 u_fmt。fp32 累加自身的误差是二阶小量
    （u_fp32 ≈ 6e-8，比 u_fp16 小四个数量级），本模型不计入。

    基准特意选 S 而不是点积结果 |Σa_i·b_i|：零均值数据下两者相差约 √K，
    正文会说明怎么把它换算成更常见的「相对结果」口径。
    """
    return fmt.unit_roundoff, fmt.unit_roundoff / math.sqrt(length)


def measure(fmt: RealFormat, length: int, trials: int, seed: int = 20261001) -> dict[str, float]:
    """fp16/bf16 输入、fp32 累加的 dot，与 fp64 精确值比较。

    返回的相对误差同样以 S = Σ|a_i·b_i| 为基准，才能和 error_budget 直接比。
    """
    rng = np.random.default_rng(seed)
    worst_rel = 0.0
    worst_abs = 0.0
    rel_samples: list[float] = []

    for _ in range(trials):
        x = rng.uniform(-1.0, 1.0, size=length).astype(np.float32)
        y = rng.uniform(-1.0, 1.0, size=length).astype(np.float32)
        exact = float(np.dot(x.astype(np.float64), y.astype(np.float64)))  # 参考：fp64
        xq = _quantize(x, fmt)
        yq = _quantize(y, fmt)
        got = float(np.dot(xq.astype(np.float32), yq.astype(np.float32)))  # 被测：低精度输入 + fp32 累加
        abs_err = abs(got - exact)
        denom = float(np.sum(np.abs(x.astype(np.float64) * y.astype(np.float64)))) or 1.0
        rel = abs_err / denom
        worst_abs = max(worst_abs, abs_err)
        worst_rel = max(worst_rel, rel)
        rel_samples.append(rel)

    rel_arr = np.asarray(rel_samples)
    return {
        "worst_abs": worst_abs,
        "worst_rel": worst_rel,
        "p999_rel": float(np.percentile(rel_arr, 99.9)),
        "median_rel": float(np.median(rel_arr)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lengths", default="64,256,1024,4096", help="归约长度 K 的列表")
    parser.add_argument("--trials", type=int, default=2000, help="每个 K 的随机试验次数")
    args = parser.parse_args()
    lengths = [int(item) for item in args.lengths.split(",") if item.strip()]

    print("== 一、由有效位数得到的机器精度 ==")
    print(f"{'格式':<6}{'有效位 p':<10}{'u = 2^-p':<14}{'10 的幂提示 atol':<16}{'说明'}")
    notes = {
        "fp16": "binary16：1 符号 + 5 指数 + 10 尾数",
        "bf16": "bfloat16：1 符号 + 8 指数 + 7 尾数",
        "fp32": "binary32：1 符号 + 8 指数 + 23 尾数",
    }
    for fmt in (FP16, BF16, FP32):
        print(f"{fmt.name:<8}{fmt.significand_bits:<12}{fmt.unit_roundoff:<16.3e}{fmt.atol_hint:<18.0e}{notes[fmt.name]}")

    print("\n== 二、长度 K 的归约误差预算（基准 S = Σ|a_i·b_i|） ==")
    print("输入舍入是主项。最坏情形与 K 无关，均方根按 1/√K 变小——这是「长归约反而更准」的来源。")
    for fmt in (FP16, BF16):
        print(f"\n-- {fmt.name} 输入 / fp32 累加 --")
        print(f"{'K':<8}{'最坏情形 u':<16}{'均方根 u/√K':<16}{'换算到相对结果 ≈ u·√K':<24}")
        for length in lengths:
            worst, rms = error_budget(fmt, length)
            print(f"{length:<8}{worst:<16.3e}{rms:<16.3e}{worst * math.sqrt(length):<24.3e}")

    print("\n== 三、实测：低精度输入 + fp32 累加 vs fp64 精确值 ==")
    print(f"每个 K 跑 {args.trials} 次随机试验（固定种子 20261001）。")
    print(f"{'格式':<6}{'K':<8}{'实测最大':<16}{'实测 99.9 分位':<18}{'实测中位数':<16}{'RMS 预算':<14}{'中位数/预算':<14}")
    for fmt in (FP16, BF16):
        for length in lengths:
            stats = measure(fmt, length, args.trials)
            _, rms = error_budget(fmt, length)
            ratio = stats["median_rel"] / rms
            print(
                f"{fmt.name:<8}{length:<8}{stats['worst_rel']:<18.3e}"
                f"{stats['p999_rel']:<20.3e}{stats['median_rel']:<18.3e}{rms:<16.3e}{ratio:<14.2f}"
            )

    print("\n== 四、工程建议 ==")
    print("1) 先看实测中位数/RMS 预算这一列：数量级吻合，说明「u_fmt 决定容差」这个模型可用。")
    print("2) 上面用的是 S = Σ|a_i·b_i| 基准。换成常见的「相对结果」口径要再乘约 √K：")
    print("   fp16：K=4096 时 ≈ 3.1e-2（最坏情形），因此长归约不能只用 rtol，必须配 atol 兜住近零输出。")
    print("3) 工程起点：fp16 用 rtol=1e-3、atol=1e-3；bf16 用 rtol=1e-2、atol=1e-2；")
    print("   再把 max_mismatched_ratio 设到 1e-3 量级，允许极少数离群元素。")
    print("4) 这与锁定版本里两个工具的缺省量级一致：")
    print("   torch_assert_close(rtol=1e-2, atol=1e-3, max_mismatched_ratio=0.001)")
    print("   Profiler.assert_allclose(atol=1e-2, rtol=1e-2, max_mismatched_ratio=0.01)")
    print("5) 长归约（K>=1024）不要用逐元素全等下结论，先看误差分布是否只有少量离群点。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
